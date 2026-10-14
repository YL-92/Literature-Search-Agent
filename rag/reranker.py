"""Final Top-k cross-encoder ranking for hybrid retrieval candidates."""

from collections.abc import Sequence

from rag.hybrid_retriever import SearchResult
from services.reranker_client import RerankerClient, ScoringProvider


def rerank(
    query: str, results: Sequence[SearchResult], *,
    client: ScoringProvider | None = None, top_k: int | None = None,
) -> list[SearchResult]:
    """按 bge-reranker 分数重新排序候选项并截断到前 k 个。
    当评分不可用时，候选项保留其 RRF 顺序，其 ``rerank_score`` 保持 ``None``。
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    candidates = list(results)
    if not candidates:
        return []
    limit = len(candidates) if top_k is None else min(top_k, len(candidates))
    scorer = client or RerankerClient()
    scores = scorer.score(query, [result.document.text for result in candidates])
    if scores is None or len(scores) != len(candidates):
        return candidates[:limit]
    ranked = [
        SearchResult(
            document=result.document,
            rrf_score=result.rrf_score,
            bm25_rank=result.bm25_rank,
            dense_rank=result.dense_rank,
            rerank_score=score,
        )
        for result, score in zip(candidates, scores)
    ]
    ranked.sort(key=lambda result: (-float(result.rerank_score or 0.0), result.document.chunk_id))
    return ranked[:limit]
