"""BM25 余弦 FAISS 检索与仅排名互惠排名融合。"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from config import Settings, get_settings
from services.embedding_client import EmbeddingClient, EmbeddingProvider
from services.llm_client import HyDEProvider, LLMClient

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """一个包含证据的块；多个块可能共享一个论文标识符。"""

    chunk_id: str
    paper_id: str
    text: str
    title: str = ""

    def __post_init__(self) -> None:
        """拒绝无法追踪或空白的证据。"""
        if not self.chunk_id.strip() or not self.paper_id.strip() or not self.text.strip():
            raise ValueError("chunk_id, paper_id and text must be non-empty")


@dataclass(frozen=True)
class SearchResult:
    """可追踪的结果，保留每个检索排名和独立分数。"""

    document: Document
    rrf_score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


def tokenize(text: str) -> list[str]:
    """小写拉丁词并为未分词的汉字分开。"""
    return re.findall(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", text.casefold())


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int | None = None,
) -> dict[str, float]:
    """求和 1/(k 一基排名); 对每个列表去重，并按 ID 打破平局。
    RRF 参考：Cormack 等, SIGIR 2009, https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
    """
    k = get_settings().rrf_k if k is None else k
    if k < 1:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def _unit_vectors(raw: Any, rows: int) -> NDArray[np.float32]:
    vectors = np.array(raw, dtype=np.float32, copy=True)
    if vectors.ndim != 2 or vectors.shape[0] != rows or vectors.shape[1] == 0:
        raise ValueError("Invalid embedding shape")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Invalid embedding values")
    return np.ascontiguousarray(vectors / norms)


class HybridRetriever:
    """构建一个不可变的语料索引并检索可配置的候选项。所有模型/网络工作都委托给可注入的服务。
    不支持的密集检索会回退到 BM25，并且不会记录虚构的密集排名。
    """

    def __init__(
        self, documents: Sequence[Document], *, settings: Settings | None = None,
        embeddings: EmbeddingProvider | None = None, llm: HyDEProvider | None = None,
    ) -> None:
        """立即索引词汇令牌；将昂贵的模型延迟到搜索时使用。"""
        self.settings = settings or get_settings()
        self.documents = tuple(documents)
        self._by_id = {doc.chunk_id: doc for doc in self.documents}
        if len(self._by_id) != len(self.documents):
            raise ValueError("Duplicate chunk_id in corpus")
        self.embeddings = embeddings if embeddings is not None else EmbeddingClient(settings=self.settings)
        self.llm = llm if llm is not None else LLMClient(settings=self.settings)
        self._tokens = [tokenize(doc.text) for doc in self.documents]
        # rank-bm25 需要调用方进行预处理。
        # https://github.com/dorianbrown/rank_bm25#usage
        self._bm25 = BM25Okapi(self._tokens) if any(self._tokens) else None
        self._index: Any = None

    @property
    def dense_available(self) -> bool:
        """报告是否已建立有效的密集语料索引。"""
        return self._index is not None

    def _ensure_dense_index(self) -> bool:
        if self._index is not None:
            return True
        try:
            import faiss
            raw = self.embeddings.encode([doc.text for doc in self.documents])
            if raw is None:
                LOGGER.warning("Dense model unavailable; using BM25 only")
                return False
            vectors = _unit_vectors(raw, len(self.documents))
            # Unit-normalized vectors + IndexFlatIP implement cosine retrieval.
            # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._index = index
            return True
        except Exception as exc:
            LOGGER.warning("Dense index unavailable (%s); using BM25 only", type(exc).__name__)
            return False

    def _dense_ranking(self, query: str, limit: int, use_hyde: bool) -> list[str]:
        if not self._ensure_dense_index():
            return []
        expanded = query
        if use_hyde:
            try:
                hypothesis = self.llm.hypothetical_abstract(query)
                if isinstance(hypothesis, str) and hypothesis.strip():
                    expanded = hypothesis.strip()
            except Exception as exc:
                LOGGER.warning("HyDE unavailable (%s); using original query", type(exc).__name__)
        try:
            raw = self.embeddings.encode([expanded])
            if raw is None:
                LOGGER.warning("Query embeddings unavailable; using BM25 only")
                return []
            vector = _unit_vectors(raw, 1)
            if vector.shape[1] != self._index.d:
                raise ValueError("Query and corpus embedding dimensions differ")
            # 搜索所有精确索引条目，以便关联分数使用稳定的块 ID，
            # 包括候选者截止时的平局。适用于 M2 小型语料库。
            scores, indices = self._index.search(vector, len(self.documents))
            ordered = sorted(
                ((self.documents[int(i)].chunk_id, float(score))
                 for i, score in zip(indices[0], scores[0]) if i >= 0),
                key=lambda item: (-item[1], item[0]),
            )
            return [chunk_id for chunk_id, _ in ordered[:limit]]
        except Exception as exc:
            LOGGER.warning("Dense query unavailable (%s); using BM25 only", type(exc).__name__)
            return []

    def retrieve(
        self, query: str, *, top_k: int | None = None, use_hyde: bool | None = None,
    ) -> list[SearchResult]:
        """融合原始查询的 BM25 和可选的 HyDE 扩展的稠密排名。``top_k`` 是候选预算（默认 20），
        而不是最终重排序的截止（默认 5）。HyDE 文本从不被添加到证据语料库中。
        """
        limit = self.settings.retrieval_candidates if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be positive")
        if not query.strip() or not self.documents:
            return []
        query = query.strip()
        query_tokens = tokenize(query)
        lexical: list[str] = []
        if self._bm25 is not None and query_tokens:
            scores = self._bm25.get_scores(query_tokens)
            matching = [
                i for i, tokens in enumerate(self._tokens)
                if set(tokens).intersection(query_tokens)
            ]
            # 在一个微小/常见词语语料库中，相匹配的 BM25 得分可能是负数。
            # 通过重叠排除不匹配项，而不是通过任意的分数阈值。
            matching.sort(key=lambda i: (-float(scores[i]), self.documents[i].chunk_id))
            lexical = [self.documents[i].chunk_id for i in matching[:limit]]
        dense = self._dense_ranking(
            query, limit, self.settings.hyde_enabled if use_hyde is None else use_hyde,
        )
        fused = reciprocal_rank_fusion([lexical, dense], k=self.settings.rrf_k)
        lexical_ranks = {chunk_id: rank for rank, chunk_id in enumerate(lexical, 1)}
        dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense, 1)}
        return [
            SearchResult(
                document=self._by_id[chunk_id], rrf_score=score,
                bm25_rank=lexical_ranks.get(chunk_id), dense_rank=dense_ranks.get(chunk_id),
            )
            for chunk_id, score in list(fused.items())[:limit]
        ]
