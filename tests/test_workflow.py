"""LangGraph workflow tests with deterministic injected services."""

from services.semanticscholar_client import SemanticPaper
from agents.schemas import PaperAnalysis, Finding
from agents.workflow import ResearchWorkflow


def _papers() -> list[SemanticPaper]:
    """Create one relevant and one unrelated candidate."""
    return [
        SemanticPaper(
            paper_id="p1",
            title="Relevant Retrieval",
            abstract="Retrieval method evaluated on datasets.",
            year=2024,
        ),
        SemanticPaper(
            paper_id="p2",
            title="Unrelated Biology",
            abstract="Protein folding experiment.",
            year=2020,
        ),
    ]


class FakeAnalysis:
    """Deterministic analysis provider for workflow tests."""

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, paper: SemanticPaper) -> PaperAnalysis:
        """Return a traceable structured finding."""
        self.calls += 1
        finding = Finding(
            text=f"{paper.paper_id} is relevant",
            paper_id=paper.paper_id,
            evidence_quote=paper.abstract or "",
        )
        return PaperAnalysis(
            paper_id=paper.paper_id,
            title=paper.title or "",
            methods=[],
            experiments=[],
            conclusions=[finding],
            reason=finding,
            mode="extractive",
            evidence_scope="abstract",
        )


def test_workflow_runs_in_order_and_has_no_unresolved_citations() -> None:
    """The compiled graph produces recommendations with valid paper IDs."""
    events: list[str] = []
    analysis = FakeAnalysis()

    def search(query: str, limit: int) -> list[SemanticPaper]:
        assert query == "retrieval method"
        return _papers()[:limit]

    result = ResearchWorkflow(
        search=search,
        analyzer=analysis,
        event_sink=events.append,
    ).run("retrieval method", limit=1)

    assert events == ["retrieve", "filter", "analyze", "verify"]
    assert result.recommendations[0].paper_id == "p1"
    assert result.recommendations[0].reason_paper_id == "p1"
    assert not result.unresolved_citations
    assert analysis.calls == 1


def test_filter_keeps_query_related_papers_before_analysis() -> None:
    """The relevance node should avoid spending analysis calls on unrelated papers."""
    analysis = FakeAnalysis()
    result = ResearchWorkflow(
        search=lambda _query, _limit: _papers(),
        analyzer=analysis,
    ).run("retrieval method", limit=2)

    assert [item.paper_id for item in result.recommendations] == ["p1"]
    assert analysis.calls == 1
    assert len(result.recommendations[0].reason.split()) <= 30


def test_workflow_drops_hallucinated_citations() -> None:
    """Claims citing an unavailable paper are rejected by the verifier."""
    class HallucinatingAnalysis:
        """Provider returning an invalid citation for verifier coverage."""

        def analyze(self, paper: SemanticPaper) -> PaperAnalysis:
            finding = Finding(
                text="unsupported",
                paper_id="not-in-results",
                evidence_quote=paper.abstract or "",
            )
            return PaperAnalysis(
                paper_id=paper.paper_id,
                title=paper.title or "",
                methods=[],
                experiments=[],
                conclusions=[finding],
                reason=finding,
                mode="extractive",
                evidence_scope="abstract",
            )

    result = ResearchWorkflow(
        search=lambda _query, _limit: _papers()[:1],
        analyzer=HallucinatingAnalysis(),
    ).run("retrieval", limit=1)

    assert result.recommendations == []
    assert result.unresolved_citations[0].paper_id == "not-in-results"


def test_workflow_search_failure_is_structured() -> None:
    """External search failures produce a structured empty result."""
    def search(_query: str, _limit: int) -> list[SemanticPaper]:
        raise TimeoutError("provider failure")

    result = ResearchWorkflow(search=search).run("retrieval")

    assert result.recommendations == []
    assert result.errors
    assert "search" in result.errors[0].lower()
    assert result.steps == ["retrieve"]


def test_empty_query_is_rejected_before_tools() -> None:
    """Blank input is a caller error, not an external service request."""
    called = False

    def search(_query: str, _limit: int) -> list[SemanticPaper]:
        nonlocal called
        called = True
        return []

    try:
        ResearchWorkflow(search=search).run("  ")
    except ValueError:
        pass
    else:
        raise AssertionError("blank query must fail")
    assert called is False
