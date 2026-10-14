"""Real BM25/FAISS tests; only expensive external inference is replaced."""

from unittest.mock import Mock

import numpy as np
import pytest

from config import Settings
from rag.hybrid_retriever import (
    Document, HybridRetriever, reciprocal_rank_fusion, tokenize,
)


def _settings() -> Settings:
    return Settings(_env_file=None, inference_backoff_seconds=0, hyde_enabled=False)


def _documents() -> list[Document]:
    return [
        Document(chunk_id="a", paper_id="paper-a", text="citation graph provenance"),
        Document(chunk_id="b", paper_id="paper-b", text="dense retrieval embeddings"),
        Document(chunk_id="c", paper_id="paper-c", text="protein folding biology"),
    ]


def test_rrf_exact_scores_one_based_ranks_and_deduplication() -> None:
    """RRF sums 1/(60+rank), not raw retrieval scores or zero-based ranks."""
    scores = reciprocal_rank_fusion([["a", "b", "a"], ["b", "c", "a"]])
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert list(scores) == ["b", "a", "c"]
    assert reciprocal_rank_fusion([]) == {}
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


@pytest.mark.parametrize("query,index", [
    ("citation provenance", 0), ("dense embeddings", 1), ("protein biology", 2),
])
def test_three_queries_real_bm25_faiss_rrf(query: str, index: int) -> None:
    """Each query's independently assigned relevant document ranks first."""
    embeddings = Mock()
    embeddings.encode.side_effect = [np.eye(3), np.eye(3)[index:index + 1]]
    retriever = HybridRetriever(_documents(), settings=_settings(), embeddings=embeddings)
    results = retriever.retrieve(query)
    assert results[0].document.chunk_id == "abc"[index]
    assert results[0].bm25_rank == results[0].dense_rank == 1
    assert results[0].rrf_score == pytest.approx(2 / 61)
    assert len(results) == 3
    assert retriever.dense_available is True
    assert all(r.rrf_score == pytest.approx(
        (1 / (60 + r.bm25_rank) if r.bm25_rank else 0)
        + (1 / (60 + r.dense_rank) if r.dense_rank else 0)
    ) for r in results)


def test_hyde_changes_only_dense_query_and_index_is_reused() -> None:
    """BM25 uses the original query; hypothetical text never becomes evidence."""
    embeddings = Mock()
    embeddings.encode.side_effect = [np.eye(3), [[0, 1, 0]], [[0, 1, 0]]]
    llm = Mock()
    llm.hypothetical_abstract.return_value = "dense embeddings"
    retriever = HybridRetriever(_documents(), settings=_settings(), embeddings=embeddings, llm=llm)
    results = retriever.retrieve("citation", use_hyde=True)
    by_id = {r.document.chunk_id: r for r in results}
    assert by_id["a"].bm25_rank == 1
    assert by_id["b"].dense_rank == 1
    assert by_id["b"].bm25_rank is None
    assert embeddings.encode.call_args.args[0] == ["dense embeddings"]
    assert {r.document.paper_id for r in results} <= {"paper-a", "paper-b", "paper-c"}
    retriever.retrieve("citation", use_hyde=True)
    assert embeddings.encode.call_count == 3  # corpus once, query twice


@pytest.mark.parametrize("vectors", [None, [[0, 0]] * 3, [[float("nan")]] * 3, [[1, 0]]])
def test_unavailable_or_bad_embeddings_use_bm25(vectors) -> None:
    """No silent substitution of fake dense vectors when the service fails."""
    embeddings = Mock()
    embeddings.encode.return_value = vectors
    llm = Mock()
    retriever = HybridRetriever(_documents(), settings=_settings(), embeddings=embeddings, llm=llm)
    results = retriever.retrieve("citation", use_hyde=True)
    assert results[0].document.chunk_id == "a"
    assert results[0].dense_rank is None
    assert retriever.dense_available is False
    llm.hypothetical_abstract.assert_not_called()


def test_query_embedding_dimension_mismatch_degrades() -> None:
    """Mismatched query dimensions never crash FAISS or mix indexes."""
    embeddings = Mock()
    embeddings.encode.side_effect = [np.eye(3), [[1, 0]]]
    results = HybridRetriever(_documents(), settings=_settings(), embeddings=embeddings).retrieve("citation")
    assert results[0].dense_rank is None


def test_hyde_exception_uses_original_query() -> None:
    """An injected unavailable HyDE service cannot disable normal retrieval."""
    embeddings = Mock()
    embeddings.encode.side_effect = [np.eye(3), [[1, 0, 0]]]
    llm = Mock()
    llm.hypothetical_abstract.side_effect = OSError("offline")
    result = HybridRetriever(_documents(), settings=_settings(), embeddings=embeddings, llm=llm).retrieve("citation", use_hyde=True)
    assert result[0].dense_rank == 1
    assert embeddings.encode.call_args.args[0] == ["citation"]


def test_blank_empty_duplicate_and_invalid_limits() -> None:
    """Input errors fail clearly and empty requests perform no model calls."""
    embeddings = Mock()
    assert HybridRetriever([], settings=_settings(), embeddings=embeddings).retrieve("q") == []
    retriever = HybridRetriever(_documents(), settings=_settings(), embeddings=embeddings)
    assert retriever.retrieve("  ") == []
    embeddings.encode.assert_not_called()
    with pytest.raises(ValueError, match="chunk_id"):
        HybridRetriever([_documents()[0]] * 2, settings=_settings())
    with pytest.raises(ValueError):
        retriever.retrieve("query", top_k=0)
    with pytest.raises(ValueError):
        Document(chunk_id="x", paper_id="", text="text")


def test_cjk_and_ties_are_deterministic() -> None:
    """Chinese queries are tokenized without requiring spaces."""
    assert "引" in tokenize("引用溯源 BM25")
    assert "bm25" in tokenize("引用溯源 BM25")
    embeddings = Mock()
    embeddings.encode.return_value = None
    docs = [
        Document(chunk_id="b", paper_id="p-b", text="引用溯源"),
        Document(chunk_id="a", paper_id="p-a", text="引用溯源"),
    ]
    results = HybridRetriever(docs, settings=_settings(), embeddings=embeddings).retrieve("引用")
    assert [r.document.chunk_id for r in results] == ["a", "b"]
