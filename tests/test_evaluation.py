"""M5 evaluation tests."""

from eval.ragas_eval import EvaluationMetrics, evaluate_workflow
from tests.test_review_service import _result


def test_evaluation_outputs_numeric_faithfulness_and_context_recall() -> None:
    """The evaluator always returns bounded numeric metrics."""
    metrics = evaluate_workflow(_result())

    assert isinstance(metrics, EvaluationMetrics)
    assert 0.0 <= metrics.faithfulness <= 1.0
    assert 0.0 <= metrics.context_recall <= 1.0
    assert metrics.sample_count == 2
    assert metrics.backend in {"ragas", "deterministic_fallback"}


def test_evaluation_penalizes_unresolved_citations() -> None:
    """Unresolved provenance cannot receive a perfect faithfulness score."""
    result = _result().model_copy(
        update={
            "unresolved_citations": [
                {"paper_id": "unknown", "problem": "unresolved citation"}
            ]
        }
    )

    metrics = evaluate_workflow(result)

    assert metrics.faithfulness < 1.0
