"""Tests for citation graph construction and provenance traversal."""

from services.semanticscholar_client import RelatedPaper, SemanticPaper
from graph.citation_graph import CitationGraph


def _paper(
    paper_id: str,
    year: int,
    *,
    references: tuple[RelatedPaper, ...] = (),
    citations: tuple[RelatedPaper, ...] = (),
) -> SemanticPaper:
    """Build a compact paper fixture."""
    return SemanticPaper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=year,
        references=references,
        citations=citations,
    )


def test_build_graph_and_forward_backward_direction() -> None:
    """A -> B means A cites B; traversal names match the user-facing meaning."""
    seed = _paper(
        "seed",
        2020,
        references=(
            RelatedPaper("classic", "Classic", 1990),
            RelatedPaper("recent", "Recent", 2024),
        ),
        citations=(RelatedPaper("new", "New", 2025),),
    )
    graph = CitationGraph.from_papers([seed])

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert [paper.paper_id for paper in graph.forward_citations("seed")] == ["new"]
    assert [paper.paper_id for paper in graph.backward_citations("seed")] == [
        "classic", "recent"
    ]


def test_duplicate_papers_merge_metadata_and_edges() -> None:
    """Repeated API pages should not duplicate nodes or lose citation edges."""
    first = _paper("seed", 2020, references=(RelatedPaper("old", "Old", 1999),))
    second = _paper("seed", 2020, citations=(RelatedPaper("new", "New", 2025),))
    graph = CitationGraph.from_papers([first, second])

    assert graph.number_of_nodes() == 3
    assert graph.backward_citations("seed")[0].paper_id == "old"
    assert graph.forward_citations("seed")[0].paper_id == "new"


def test_year_filter_classic_and_latest() -> None:
    """Classic/latest filters select by explicit year and have deterministic order."""
    graph = CitationGraph.from_papers(
        [
            _paper("seed", 2020, references=(
                RelatedPaper("a", "A", 1990),
                RelatedPaper("b", "B", 2000),
                RelatedPaper("c", "C", 2024),
            ))
        ]
    )

    assert [p.paper_id for p in graph.backward_citations("seed", era="classic", max_year=2000)] == ["a", "b"]
    assert [p.paper_id for p in graph.backward_citations("seed", era="latest", min_year=2021)] == ["c"]
    assert [p.paper_id for p in graph.backward_citations("seed", min_year=2000)] == ["b", "c"]
    assert [p.paper_id for p in graph.backward_citations("seed", max_year=2000)] == ["a", "b"]


def test_multi_hop_chain_and_missing_nodes_are_safe() -> None:
    """Traversal can follow citation chains and missing IDs return empty lists."""
    graph = CitationGraph.from_papers(
        [
            _paper("a", 2020, references=(RelatedPaper("b", "B", 2019),)),
            _paper("b", 2019, references=(RelatedPaper("c", "C", 2018),)),
        ]
    )
    assert [p.paper_id for p in graph.walk_backward("a", hops=2)] == ["b", "c"]
    assert graph.forward_citations("unknown") == []
    assert graph.backward_citations("a", hops=0) == []


def test_invalid_input_and_unknown_era() -> None:
    """The graph rejects invalid traversal options rather than guessing."""
    graph = CitationGraph()
    try:
        graph.backward_citations("x", era="middle")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown era must fail")


def test_seed_graph_uses_provider_forward_citations() -> None:
    """A provider with separate forward endpoint still yields both graph chains."""
    from unittest.mock import Mock

    seed = _paper(
        "https://openalex.org/W1",
        2024,
        references=(RelatedPaper("https://openalex.org/W2", "Reference", 2000),),
    )
    provider = Mock()
    provider.get_paper.return_value = seed
    provider.forward_citations.return_value = [
        SemanticPaper(
            paper_id="https://openalex.org/W3",
            title="Citing",
            year=2025,
            source="openalex",
        )
    ]

    graph = CitationGraph.from_seed(seed.paper_id, client=provider)

    assert [p.paper_id for p in graph.backward_citations(seed.paper_id)] == [
        "https://openalex.org/W2"
    ]
    assert [p.paper_id for p in graph.forward_citations(seed.paper_id)] == [
        "https://openalex.org/W3"
    ]
