"""Tests for the Semantic Scholar service adapter."""

from __future__ import annotations

from unittest.mock import Mock

import requests
import pytest

from services.semanticscholar_client import SemanticScholarClient


def test_search_and_paper_details_normalize_references_and_citations() -> None:
    """Search and detail endpoints should expose stable normalized records."""
    session = Mock()
    search_response = Mock()
    search_response.raise_for_status.return_value = None
    search_response.json.return_value = {
        "total": 1,
        "data": [
            {
                "paperId": "paper-1",
                "title": "A Paper",
                "abstract": "Abstract",
                "year": 2022,
                "authors": [{"authorId": "a1", "name": "Researcher"}],
                "citationCount": 3,
                "referenceCount": 1,
                "url": "https://www.semanticscholar.org/paper/paper-1",
                "externalIds": {"ArXiv": "2201.00001", "DOI": "10.1/test"},
            }
        ],
    }
    detail_response = Mock()
    detail_response.raise_for_status.return_value = None
    detail_response.json.return_value = {
        "paperId": "paper-1",
        "title": "A Paper",
        "abstract": "Abstract",
        "year": 2022,
        "authors": [{"authorId": "a1", "name": "Researcher"}],
        "references": [{"paperId": "paper-0", "title": "Prior Paper"}],
        "citations": [
            {"citingPaper": {"paperId": "paper-2", "title": "Later Paper"}}
        ],
    }
    session.get.side_effect = [search_response, detail_response]
    client = SemanticScholarClient(
        session=session, base_url="https://s2.test/graph/v1"
    )

    results = client.search("agents")
    paper = client.get_paper("paper-1")

    assert results[0].paper_id == "paper-1"
    assert results[0].arxiv_id == "2201.00001"
    assert paper is not None
    assert paper.references[0].paper_id == "paper-0"
    assert paper.citations[0].paper_id == "paper-2"


def test_paper_detail_retries_then_degrades_to_none() -> None:
    """An unavailable optional record should degrade to None after retry."""
    session = Mock()
    session.get.side_effect = requests.ConnectionError("offline")
    client = SemanticScholarClient(
        session=session, base_url="https://s2.test/graph/v1",
        max_retries=2, backoff_seconds=0
    )

    assert client.get_paper("paper-1") is None
    assert session.get.call_count == 2


def test_graph_detail_requests_related_years() -> None:
    """Year filtering requires explicit year subfields, not just titles."""
    session = Mock()
    session.get.return_value.json.return_value = {
        "paperId": "seed", "title": "Seed",
        "references": [{"paperId": "old", "title": "Old", "year": 2000}],
        "citations": [{"paperId": "new", "title": "New", "year": 2025}],
    }
    paper = SemanticScholarClient(
        session=session, base_url="https://s2.test/graph/v1"
    ).get_paper("seed")
    fields = session.get.call_args.kwargs["params"]["fields"].split(",")
    assert "references.year" in fields
    assert "citations.year" in fields
    assert paper.references[0].year == 2000
    assert paper.citations[0].year == 2025


@pytest.mark.parametrize("bad_record", [None, {}, {"paperId": ""}, {"paperId": "  "}, {"paperId": 4}])
def test_invalid_related_ids_do_not_create_graph_nodes(bad_record: object) -> None:
    """Unresolved papers cannot become empty-ID graph nodes."""
    session = Mock()
    session.get.return_value.json.return_value = {
        "paperId": "seed", "title": "Seed",
        "references": [bad_record, {"paperId": "good", "year": "invalid"}],
    }
    paper = SemanticScholarClient(
        session=session, base_url="https://s2.test/graph/v1"
    ).get_paper("seed")
    assert len(paper.references) == 1
    assert paper.references[0].paper_id == "good"
    assert paper.references[0].year is None
