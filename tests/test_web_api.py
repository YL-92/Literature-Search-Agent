"""FastAPI frontend contract tests."""

from dataclasses import replace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from agents.schemas import (
    Finding,
    PaperAnalysis,
    Recommendation,
    WorkflowResult,
)
from services.paper import Paper, RelatedPaper


def _paper(paper_id: str = "https://openalex.org/W1") -> Paper:
    """Create a provider-neutral paper fixture."""
    return Paper(
        paper_id=paper_id,
        title="Retrieval Paper",
        year=2024,
        cited_by_count=12,
        authors=("Ada Lovelace",),
        abstract="A retrieval method improves evidence grounding.",
        doi="https://doi.org/10.1234/example",
        source="openalex",
    )


def _workflow_result() -> WorkflowResult:
    """Create a verified workflow result for web responses."""
    finding = Finding(
        text="A retrieval method improves evidence grounding.",
        paper_id="https://openalex.org/W1",
        evidence_quote="A retrieval method improves evidence grounding.",
    )
    analysis = PaperAnalysis(
        paper_id="https://openalex.org/W1",
        title="Retrieval Paper",
        methods=[finding],
        experiments=[],
        conclusions=[finding],
        reason=finding,
        mode="extractive",
        evidence_scope="abstract",
    )
    recommendation = Recommendation(
        paper_id="https://openalex.org/W1",
        title="Retrieval Paper",
        reason=finding.text,
        reason_paper_id=finding.paper_id,
        evidence_quote=finding.evidence_quote,
        mode="extractive",
        evidence_scope="abstract",
    )
    return WorkflowResult(
        query="retrieval",
        recommendations=[recommendation],
        analyses=[analysis],
        evidence={finding.paper_id: finding.evidence_quote},
        unresolved_citations=[],
        errors=[],
        warnings=[],
        steps=["retrieve", "filter", "analyze", "verify"],
    )


def _client(monkeypatch, *, result: WorkflowResult | None = None):
    """Patch main's production services with deterministic fakes."""
    import main

    openalex = Mock()
    openalex.search_papers.return_value = [_paper()]
    openalex.get_paper.return_value = _paper()
    openalex.backward_citations.return_value = [
        Paper(
            paper_id="https://openalex.org/W2",
            title="Reference Paper",
            year=2019,
            source="openalex",
        )
    ]
    openalex.forward_citations.return_value = [
        Paper(
            paper_id="https://openalex.org/W3",
            title="Citing Paper",
            year=2025,
            source="openalex",
        )
    ]
    workflow = Mock()
    workflow.run.return_value = result or _workflow_result()
    monkeypatch.setattr(main, "get_openalex_client", lambda: openalex)
    monkeypatch.setattr(main, "build_default_workflow", lambda: workflow)
    return openalex, workflow


def test_search_endpoint_returns_paper_cards(monkeypatch) -> None:
    """POST /api/search returns the frontend card contract."""
    openalex, _ = _client(monkeypatch)

    from main import app

    response = TestClient(app).post(
        "/api/search", json={"query": "retrieval", "limit": 10}
    )

    assert response.status_code == 200
    assert response.json()["papers"][0]["title"] == "Retrieval Paper"
    openalex.search_papers.assert_called_once_with("retrieval", limit=10)


def test_paper_endpoint_returns_both_citation_directions(monkeypatch) -> None:
    """GET /api/paper/{pid} returns detail, references and citing papers."""
    openalex, _ = _client(monkeypatch)

    from main import app

    response = TestClient(app).get("/api/paper/https%3A%2F%2Fopenalex.org%2FW1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["detail"]["paper_id"].endswith("W1")
    assert payload["references"][0]["title"] == "Reference Paper"
    assert payload["citing"][0]["title"] == "Citing Paper"
    openalex.forward_citations.assert_called_once()


def test_summarize_endpoint_emits_token_citations_and_done(monkeypatch) -> None:
    """SSE summary stream contains the three required event types."""
    _client(monkeypatch)

    from main import app

    response = TestClient(app).post("/api/summarize", json={"query": "retrieval"})

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: citations" in response.text
    assert "event: done" in response.text
    assert '"n": 1' in response.text


def test_eval_endpoint_returns_numeric_metrics(monkeypatch) -> None:
    """GET /api/eval exposes the same evaluation shape as the CLI."""
    _client(monkeypatch)

    from main import app

    response = TestClient(app).get("/api/eval", params={"query": "retrieval"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["eval_mode"] in {"ragas", "deterministic"}
    assert 0 <= payload["metrics"]["faithfulness"] <= 1
    assert 0 <= payload["metrics"]["context_recall"] <= 1


def test_frontend_is_served(monkeypatch) -> None:
    """The root page is a static single-page frontend."""
    import main

    response = TestClient(main.app).get("/")

    assert response.status_code == 200
    assert "文献检索 Agent" in response.text
    assert "citation-graph" in response.text


def test_favicon_is_served() -> None:
    """Browsers should not receive a 404 for the standard favicon request."""
    from main import app

    response = TestClient(app).get("/favicon.ico")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
