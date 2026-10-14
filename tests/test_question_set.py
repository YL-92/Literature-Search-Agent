"""Question-set and offline benchmark tests."""

from eval.question_benchmark import (
    QuestionPrediction,
    compare_metrics,
    evaluate_predictions,
    format_improvement_sentence,
)
from eval.question_set import load_question_set, summarize_question_set


def test_question_set_has_100_plus_unique_cases_and_broad_coverage() -> None:
    """The checked-in dataset is large enough for regression testing."""
    cases = load_question_set()
    summary = summarize_question_set(cases)

    assert summary["count"] >= 100
    assert len(summary["categories"]) >= 8
    assert len({case.question_id for case in cases}) == summary["count"]
    assert len({case.query for case in cases}) == summary["count"]


def test_question_set_has_required_fields() -> None:
    """Every query declares its expected answer and citation behavior."""
    for case in load_question_set():
        assert case.category
        assert case.expected_answer_type
        assert case.citation_requirement
        assert case.difficulty in {"easy", "medium", "hard"}


def test_offline_benchmark_calculates_faithfulness_and_citation_validity() -> None:
    """Ratios are bounded and improvements are reported in percentage points."""
    baseline = evaluate_predictions(
        [QuestionPrediction("q1", 10, 7, 10, 8)]
    )
    improved = evaluate_predictions(
        [QuestionPrediction("q1", 10, 9, 10, 10)]
    )

    assert baseline.faithfulness == 0.7
    assert improved.faithfulness == 0.9
    assert improved.citation_validity == 1.0
    assert compare_metrics(baseline, improved)["faithfulness_delta"] == 0.2
    assert format_improvement_sentence(baseline, improved) == (
        "RAGAS 忠实度从 70.00% 提升至 90.00%，引用有效率达 100.00%。"
    )
