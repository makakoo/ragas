from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset, concatenate_datasets
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.language_models import BaseLanguageModel as LangchainLLM

from ragas._analytics import EvaluationEvent, track, track_was_completed
from ragas.callbacks import new_group
from ragas.cost import TokenUsage, CostCallbackHandler  # Ensure CostCallbackHandler is properly imported
from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas.embeddings.base import (
    BaseRagasEmbeddings,
    LangchainEmbeddingsWrapper,
    embedding_factory,
)
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.integrations.helicone import helicone_config
from ragas.llms import llm_factory
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics import AspectCritic
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics.base import (
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
    MultiTurnMetric,
    SingleTurnMetric,
    is_reproducable,
)
from ragas.run_config import RunConfig
from ragas.utils import (
    convert_v1_to_v2_dataset,
    convert_v2_to_v1_dataset,
    get_feature_language,
    safe_nanmean,
)
from ragas.validation import (
    remap_column_names,
    validate_required_columns,
    validate_supported_metrics,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from ragas.cost import TokenUsageParser

RAGAS_EVALUATION_CHAIN_NAME = "ragas evaluation"


@track_was_completed
def evaluate(
    dataset: t.Union[Dataset, EvaluationDataset],
    metrics: list[Metric] | None = None,
    llm: t.Optional[BaseRagasLLM | LangchainLLM] = None,
    embeddings: t.Optional[BaseRagasEmbeddings | LangchainEmbeddings] = None,
    callbacks: Callbacks = None,
    in_ci: bool = False,
    run_config: RunConfig = RunConfig(),
    token_usage_parser: t.Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    show_progress: bool = True,
    model_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
) -> Result:
    # Default values for column map and callbacks
    column_map = column_map or {}
    callbacks = callbacks or []

    if helicone_config.is_enabled:
        import uuid
        helicone_config.session_name = "ragas-evaluation"
        helicone_config.session_id = str(uuid.uuid4())

    if dataset is None:
        raise ValueError("Provide dataset!")

    # Default metrics if none provided
    if metrics is None:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        metrics = [answer_relevancy, context_precision, faithfulness, context_recall]

    # Handle Dataset type and conversion
    v1_input = False
    if isinstance(dataset, Dataset):
        v1_input = True
        dataset = remap_column_names(dataset, column_map)
        dataset = convert_v1_to_v2_dataset(dataset)
        dataset = EvaluationDataset.from_list(dataset.to_list())

    if isinstance(dataset, EvaluationDataset):
        validate_required_columns(dataset, metrics)
        validate_supported_metrics(dataset, metrics)

    # Set the LLM and embeddings with model_kwargs passed if applicable
    if isinstance(llm, LangchainLLM):
        llm = LangchainLLMWrapper(llm, run_config=run_config)

    if isinstance(embeddings, LangchainEmbeddings):
        embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Initialize LLMs and embeddings
    binary_metrics = []
    llm_changed: t.List[int] = []
    embeddings_changed: t.List[int] = []
    reproducable_metrics: t.List[int] = []
    answer_correctness_is_set = -1

    for i, metric in enumerate(metrics):
        # Set LLM and embeddings if not set at the metric level
        if isinstance(metric, AspectCritic):
            binary_metrics.append(metric.name)

        if isinstance(metric, MetricWithLLM) and metric.llm is None:
            if llm is None:
                llm = llm_factory(run_config=run_config, model_kwargs=model_kwargs)
            metric.llm = llm
            llm_changed.append(i)

        if isinstance(metric, MetricWithEmbeddings) and metric.embeddings is None:
            if embeddings is None:
                embeddings = embedding_factory(run_config=run_config, model_kwargs=model_kwargs)
            metric.embeddings = embeddings
            embeddings_changed.append(i)

        if isinstance(metric, AnswerCorrectness):
            if metric.answer_similarity is None:
                answer_correctness_is_set = i

        # Set reproducibility for metrics if in CI
        if in_ci and is_reproducable(metric):
            if metric.reproducibility == 1:  # type: ignore
                metric.reproducibility = 3  # type: ignore
                reproducable_metrics.append(i)

        # Initialize the metric with the run config
        metric.init(run_config)

    # Execute the evaluation
    executor = Executor(
        desc="Evaluating",
        keep_progress_bar=True,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
        show_progress=show_progress,
    )

    ragas_callbacks: t.Dict[str, BaseCallbackHandler] = {}

    if token_usage_parser is not None:
        cost_cb = CostCallbackHandler(token_usage_parser=token_usage_parser)
        ragas_callbacks["cost_cb"] = cost_cb

    for cb in ragas_callbacks.values():
        if isinstance(callbacks, BaseCallbackManager):
            callbacks.add_handler(cb)
        else:
            callbacks.append(cb)

    row_run_managers = []
    evaluation_rm, evaluation_group_cm = new_group(
        name=RAGAS_EVALUATION_CHAIN_NAME, inputs={}, callbacks=callbacks
    )

    sample_type = dataset.get_sample_type()
    for i, sample in enumerate(dataset):
        row = t.cast(t.Dict[str, t.Any], sample.dict())
        row_rm, row_group_cm = new_group(
            name=f"row {i}", inputs=row, callbacks=evaluation_group_cm
        )
        row_run_managers.append((row_rm, row_group_cm))

        if sample_type == SingleTurnSample:
            _ = [
                executor.submit(
                    metric.single_turn_ascore,
                    sample,
                    row_group_cm,
                    name=f"{metric.name}-{i}",
                    timeout=run_config.timeout,
                )
                for metric in metrics if isinstance(metric, SingleTurnMetric)
            ]
        elif sample_type == MultiTurnSample:
            _ = [
                executor.submit(
                    metric.multi_turn_ascore,
                    sample,
                    row_group_cm,
                    name=f"{metric.name}-{i}",
                    timeout=run_config.timeout,
                )
                for metric in metrics if isinstance(metric, MultiTurnMetric)
            ]
        else:
            raise ValueError(f"Unsupported sample type {sample_type}")

    scores = []
    try:
        results = executor.results()
        if results == []:
            raise ExceptionInRunner()

        for i, _ in enumerate(dataset):
            s = {m.name: results[len(metrics) * i + j] for j, m in enumerate(metrics)}
            scores.append(s)

            row_rm, row_group_cm = row_run_managers[i]
            if not row_group_cm.ended:
                row_rm.on_chain_end(s)

    except Exception as e:
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_error(e)
        raise e

    else:
        dataset = dataset.to_hf_dataset()
        if v1_input:
            dataset = convert_v2_to_v1_dataset(dataset)

        cost_cb = ragas_callbacks.get("cost_cb", None)
        result = Result(
            scores=Dataset.from_list(scores),
            dataset=dataset,
            binary_columns=binary_metrics,
            cost_cb=t.cast(t.Optional[CostCallbackHandler], cost_cb),
        )
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_end(result)

    finally:
        for i in llm_changed:
            t.cast(MetricWithLLM, metrics[i]).llm = None
        for i in embeddings_changed:
            t.cast(MetricWithEmbeddings, metrics[i]).embeddings = None
        if answer_correctness_is_set != -1:
            t.cast(AnswerCorrectness, metrics[answer_correctness_is_set]).answer_similarity = None
        for i in reproducable_metrics:
            metrics[i].reproducibility = 1  # type: ignore

    metrics_names = [m.name for m in metrics]
    metric_lang = [get_feature_language(m) for m in metrics]
    metric_lang = np.unique([m for m in metric_lang if m is not None])

    track(
        EvaluationEvent(
            event_type="evaluation",
            metrics=metrics_names,
            evaluation_mode="",
            num_rows=len(dataset),
            language=metric_lang[0] if len(metric_lang) > 0 else "",
            in_ci=in_ci,
        )
    )
    return result


@dataclass
class Result(dict):
    scores: Dataset
    dataset: t.Optional[Dataset] = None
    binary_columns: t.List[str] = field(default_factory=list)
    cost_cb: t.Optional[CostCallbackHandler] = None

    def __post_init__(self):
        values = []
        for cn in self.scores[0].keys():
            value = safe_nanmean(self.scores[cn])
            self[cn] = value
            if cn not in self.binary_columns:
                value = t.cast(float, value)
                values.append(value + 1e-10)

    def to_pandas(self, batch_size: int | None = None, batched: bool = False):
        if self.dataset is None:
            raise ValueError("dataset is not provided for the results class")
        assert self.scores.shape[0] == self.dataset.shape[0]
        result_ds = concatenate_datasets([self.dataset, self.scores], axis=1)

        return result_ds.to_pandas(batch_size=batch_size, batched=batched)

    def total_tokens(self) -> t.Union[t.List[TokenUsage], TokenUsage]:
        if self.cost_cb is None:
            raise ValueError(
                "The evaluate() run was not configured for computing cost. Please provide a token_usage_parser function to evaluate() to compute cost."
            )
        return self.cost_cb.total_tokens()

    def total_cost(
        self,
        cost_per_input_token: t.Optional[float] = None,
        cost_per_output_token: t.Optional[float] = None,
        per_model_costs: t.Dict[str, t.Tuple[float, float]] = {},
    ) -> float:
        if self.cost_cb is None:
            raise ValueError(
                "The evaluate() run was not configured for computing cost. Please provide a token_usage_parser function to evaluate() to compute cost."
            )
        return self.cost_cb.total_cost(
            cost_per_input_token, cost_per_output_token, per_model_costs
        )

    def __repr__(self) -> str:
        scores = self.copy()
        score_strs = [f"'{k}': {v:0.4f}" for k, v in scores.items()]
        return "{" + ", ".join(score_strs) + "}"
