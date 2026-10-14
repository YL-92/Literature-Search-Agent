"""Offline regression metrics for the question set.

This module evaluates supplied prediction records. It does not fabricate
RAGAS scores: real RAGAS values require model-generated responses and an
evaluator model. The deterministic proxy is useful for CI and provenance-only
runs, and is labeled separately in its output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class QuestionPrediction:
    """Evaluation facts for one question produced by a system run."""

    question_id: str
    claim_count: int
    supported_claim_count: int
    citation_count: int
    valid_citation_count: int


@dataclass(frozen=True)
class QuestionBenchmarkMetrics:
    """Aggregated deterministic faithfulness and citation-validity metrics."""

    faithfulness: float
    citation_validity: float
    question_count: int
    claim_count: int
    citation_count: int
    backend: str = "deterministic_proxy"

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible benchmark metrics."""
        return {
            "faithfulness": self.faithfulness,
            "citation_validity": self.citation_validity,
            "question_count": self.question_count,
            "claim_count": self.claim_count,
            "citation_count": self.citation_count,
            "backend": self.backend,
        }


def evaluate_predictions(
    predictions: Iterable[QuestionPrediction],
) -> QuestionBenchmarkMetrics:
    """Aggregate supported-claim and valid-citation ratios."""
    rows = list(predictions)
    if not rows:
        return QuestionBenchmarkMetrics(0.0, 0.0, 0, 0, 0)
    claims = sum(max(row.claim_count, 0) for row in rows)
    supported = sum(
        min(max(row.supported_claim_count, 0), max(row.claim_count, 0))
        for row in rows
    )
    citations = sum(max(row.citation_count, 0) for row in rows)
    valid = sum(
        min(max(row.valid_citation_count, 0), max(row.citation_count, 0))
        for row in rows
    )
    return QuestionBenchmarkMetrics(
        faithfulness=_ratio(supported, claims),
        citation_validity=_ratio(valid, citations),
        question_count=len(rows),
        claim_count=claims,
        citation_count=citations,
    )


def compare_metrics(
    baseline: QuestionBenchmarkMetrics,
    improved: QuestionBenchmarkMetrics,
) -> dict[str, float]:
    """Return absolute improvements in percentage points."""
    return {
        "faithfulness_delta": round(
            improved.faithfulness - baseline.faithfulness, 6
        ),
        "citation_validity_delta": round(
            improved.citation_validity - baseline.citation_validity, 6
        ),
    }


def format_improvement_sentence(
    baseline: QuestionBenchmarkMetrics,
    improved: QuestionBenchmarkMetrics,
) -> str:
    """Format the requested Chinese benchmark summary sentence."""
    return (
        f"RAGAS 忠实度从 {baseline.faithfulness:.2%} 提升至 "
        f"{improved.faithfulness:.2%}，引用有效率达 "
        f"{improved.citation_validity:.2%}。"
    )


def _ratio(numerator: int, denominator: int) -> float:
    """Return a bounded ratio, avoiding division-by-zero claims."""
    return numerator / denominator if denominator else 0.0
