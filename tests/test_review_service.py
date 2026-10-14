"""M5 review generation tests."""

from agents.schemas import (
    Finding,
    PaperAnalysis,
    Recommendation,
    WorkflowResult,
)
from services.review_service import ReviewGenerator


def _result() -> WorkflowResult:
    """Create a verified workflow result with two source papers."""
    first = Finding(
        text="The method retrieves evidence.",
        paper_id="p1",
        evidence_quote="The method retrieves evidence.",
    )
    second = Finding(
        text="The benchmark measures faithfulness.",
        paper_id="p2",
        evidence_quote="The benchmark measures faithfulness.",
    )
    return WorkflowResult(
        query="retrieval evaluation",
        recommendations=[
            Recommendation(
                paper_id="p1",
                title="Retrieval Method",
                reason=first.text,
                reason_paper_id="p1",
                evidence_quote=first.evidence_quote,
                mode="extractive",
                evidence_scope="abstract",
            ),
            Recommendation(
                paper_id="p2",
                title="Evaluation Benchmark",
                reason=second.text,
                reason_paper_id="p2",
                evidence_quote=second.evidence_quote,
                mode="extractive",
                evidence_scope="abstract",
            ),
        ],
        analyses=[
            PaperAnalysis(
                paper_id="p1",
                title="Retrieval Method",
                methods=[first],
                experiments=[],
                conclusions=[first],
                reason=first,
                mode="extractive",
                evidence_scope="abstract",
            ),
            PaperAnalysis(
                paper_id="p2",
                title="Evaluation Benchmark",
                methods=[],
                experiments=[second],
                conclusions=[second],
                reason=second,
                mode="extractive",
                evidence_scope="abstract",
            ),
        ],
        evidence={
            "p1": "The method retrieves evidence.",
            "p2": "The benchmark measures faithfulness.",
        },
        unresolved_citations=[],
        errors=[],
        warnings=[],
        steps=["retrieve", "filter", "analyze", "verify"],
    )


def test_review_has_comparison_table_and_source_tagged_summary() -> None:
    """Every generated factual sentence must carry a paper-title source tag."""
    report = ReviewGenerator().generate(_result())

    assert [row.paper_id for row in report.comparison_table] == ["p1", "p2"]
    assert "Retrieval Method" in report.summary
    assert "Evaluation Benchmark" in report.summary
    assert "[来源: Retrieval Method]" in report.summary
    assert "[来源: Evaluation Benchmark]" in report.summary
    assert all(row.source_title for row in report.comparison_table)
    assert report.summary.count("[来源: Retrieval Method]") >= 2


def test_review_rejects_unverified_result() -> None:
    """Reports must not silently summarize unresolved citations."""
    result = _result().model_copy(
        update={
            "unresolved_citations": [
                {"paper_id": "unknown", "problem": "unresolved citation"}
            ]
        }
    )

    report = ReviewGenerator().generate(result)

    assert report.summary == ""
    assert report.warnings


def test_review_is_deterministic_without_llm() -> None:
    """Offline generation produces stable, non-empty source-backed text."""
    generator = ReviewGenerator()
    assert generator.generate(_result()) == generator.generate(_result())
