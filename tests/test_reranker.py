"""RAG-layer reranking tests."""

from unittest.mock import Mock

from config import Settings
from rag.hybrid_retriever import Document, SearchResult
from rag.reranker import rerank


def _results() -> list[SearchResult]:
    return [
        SearchResult(Document("a", "p-a", "old dense candidate"), 0.1, 2, 2),
        SearchResult(Document("b", "p-b", "highly relevant method"), 0.2, 1, 1),
        SearchResult(Document("c", "p-c", "another result"), 0.15, 3, 3),
    ]


def test_rerank_changes_order_and_returns_top_five() -> None:
    """Cross-encoder scores become the final deterministic ranking."""
    client = Mock()
    client.score.return_value = [0.1, 0.95, 0.3]
    output = rerank("method", _results(), client=client, top_k=2)
    assert [item.document.chunk_id for item in output] == ["b", "c"]
    assert [item.rerank_score for item in output] == [0.95, 0.3]
    assert client.score.call_args.args == ("method", ["old dense candidate", "highly relevant method", "another result"])


def test_rerank_preserves_fused_order_on_unavailable_service() -> None:
    """A failed reranker returns candidates unchanged, never fabricated scores."""
    results = _results()
    client = Mock()
    client.score.return_value = None
    assert rerank("q", results, client=client) == results


def test_rerank_handles_empty_and_invalid_top_k() -> None:
    """No reranker request is made for empty results."""
    client = Mock()
    assert rerank("q", [], client=client) == []
    client.score.assert_not_called()
    try:
        rerank("q", _results(), client=client, top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("top_k=0 must fail")
