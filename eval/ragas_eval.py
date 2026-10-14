"""兼容 RAGAS 的评估，带有确定性的离线回退。当可选的 RAGAS 包和评估模型可用时，
适配器会尝试官方的 ``evaluate`` API。
CI 和离线运行在本地使用相同的两个指标定义，因此 CLI 始终输出数值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.schemas import WorkflowResult


@dataclass(frozen=True)
class EvaluationMetrics:
    """一个工作流程结果的忠实度和上下文回忆分数。"""

    faithfulness: float
    context_recall: float
    sample_count: int
    backend: str
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的度量值。"""
        return {
            "faithfulness": self.faithfulness,
            "context_recall": self.context_recall,
            "sample_count": self.sample_count,
            "backend": self.backend,
            "warnings": list(self.warnings),
        }


def evaluate_workflow(
    result: WorkflowResult, *, evaluator: Any | None = None
) -> EvaluationMetrics:
    """评估来源可信度和上下文回忆。``evaluator`` 可能是一个注入的 RAGAS 兼容可调用对象，
    返回包含 ``faithfulness`` 和 ``context_recall`` 的映射。这可以保持测试和本地运行独立于模型凭证。
    """
    if evaluator is not None:
        try:
            values = evaluator(_ragas_samples(result))
            return EvaluationMetrics(
                faithfulness=_bounded(values["faithfulness"]),
                context_recall=_bounded(values["context_recall"]),
                sample_count=len(result.recommendations),
                backend="ragas",
            )
        except Exception:
            pass

    ragas_result = _try_ragas(result)
    if ragas_result is not None:
        return ragas_result

    return _deterministic_metrics(result)


def _try_ragas(result: WorkflowResult) -> EvaluationMetrics | None:
    """当存在可选依赖项/配置时，尝试官方 RAGAS 评估器。"""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics.collections import context_recall, faithfulness

        samples = _ragas_samples(result)
        if not samples:
            return None
        dataset = Dataset.from_list(samples)
        values = evaluate(
            dataset,
            metrics=[faithfulness, context_recall],
        )
        return EvaluationMetrics(
            faithfulness=_bounded(values["faithfulness"]),
            context_recall=_bounded(values["context_recall"]),
            sample_count=len(samples),
            backend="ragas",
        )
    except Exception:
        # RAGAS 可能缺失，或者其评估器 LLM 可能未配置。
        # 下面的确定性路径保持明确且有界。
        return None


def _ragas_samples(result: WorkflowResult) -> list[dict[str, Any]]:
    """创建 RAGAS 忠诚度/上下文回忆指标使用的字段。"""
    return [
        {
            "user_input": result.query,
            "response": recommendation.reason,
            "retrieved_contexts": [
                result.evidence.get(recommendation.reason_paper_id, "")
            ],
            "reference": recommendation.evidence_quote,
        }
        for recommendation in result.recommendations
    ]


def _deterministic_metrics(result: WorkflowResult) -> EvaluationMetrics:
    """在不假装是大型语言模型裁判的情况下，评估明确来源。"""
    total = len(result.recommendations)
    if total == 0:
        return EvaluationMetrics(
            faithfulness=0.0,
            context_recall=0.0,
            sample_count=0,
            backend="deterministic_fallback",
            warnings=("no verified recommendations",),
        )

    faithful = 0
    recalled = 0
    known_ids = set(result.evidence)
    for recommendation in result.recommendations:
        context = result.evidence.get(recommendation.reason_paper_id, "")
        if (
            recommendation.reason_paper_id in known_ids
            and recommendation.evidence_quote in context
        ):
            faithful += 1
            recalled += 1
    penalty = 0.0 if not result.unresolved_citations else 0.25
    return EvaluationMetrics(
        faithfulness=_bounded(faithful / total - penalty),
        context_recall=_bounded(recalled / total),
        sample_count=total,
        backend="deterministic_fallback",
        warnings=("RAGAS unavailable; used provenance metrics",),
    )


def _bounded(value: Any) -> float:
    """将一个度量值限制在常规的 [0, 1] 范围内。"""
    number = float(value)
    return max(0.0, min(1.0, number))
