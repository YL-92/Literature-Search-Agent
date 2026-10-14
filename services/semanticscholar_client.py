"""Optional Semantic Scholar Graph API adapter.

The endpoint is intentionally configuration-only so the fallback is disabled
unless the operator supplies an explicit URL and API key.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote

import requests

from config import get_settings
from services.paper import Paper as SemanticPaper, RelatedPaper

LOGGER = logging.getLogger(__name__)


class SemanticScholarClient:
    """Search papers and retrieve references/citations with retry fallback."""

    SEARCH_FIELDS = (
        "paperId,title,abstract,year,authors,citationCount,referenceCount,"
        "url,externalIds"
    )
    DETAIL_FIELDS = (
        "paperId,title,abstract,year,authors,citationCount,referenceCount,"
        # FullPaper.references/citations contain BasePaper records with year.
        # Field names follow the configured provider's official Graph schema.
        "url,externalIds,references.title,references.year,citations.title,citations.year"
    )

    def __init__(
        self,
        session: requests.Session | Any | None = None,
        *,
        settings: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        backoff_seconds: float = 0.5,
    ) -> None:
        """Initialize an injectable client and read credentials from settings."""
        settings = settings or get_settings()
        self.session = session or requests.Session()
        configured_base_url = base_url or settings.semantic_scholar_api_url
        if not configured_base_url:
            raise ValueError(
                "Semantic Scholar fallback requires LITSEARCH_SEMANTIC_SCHOLAR_API_URL"
            )
        self.base_url = configured_base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else settings.semantic_scholar_api_key
        self.timeout = timeout or settings.request_timeout
        self.max_retries = max_retries or settings.request_max_retries
        self.backoff_seconds = backoff_seconds
        self.last_error: str | None = None

    def search(self, query: str, limit: int = 10) -> list[SemanticPaper]:
        """Search the Graph API; return an empty list when unavailable."""
        if not query.strip() or limit < 1:
            return []
        payload = self._get_json(
            f"{self.base_url}/paper/search",
            params={"query": query.strip(), "limit": limit, "fields": self.SEARCH_FIELDS},
        )
        if payload is None:
            return []
        return [
            paper
            for item in payload.get("data", [])
            if (paper := _normalize_paper(item)) is not None
        ]

    def search_papers(self, query: str, limit: int = 10) -> list[SemanticPaper]:
        """Provider-neutral alias for paper search."""
        return self.search(query, limit)

    def get_paper(self, paper_id: str) -> SemanticPaper | None:
        """Fetch a paper and its references/citations by supported identifier."""
        if not paper_id.strip():
            return None
        encoded_id = quote(paper_id.strip(), safe=":")
        payload = self._get_json(
            f"{self.base_url}/paper/{encoded_id}",
            params={"fields": self.DETAIL_FIELDS},
        )
        return _normalize_paper(payload, include_links=True) if payload else None

    def forward_citations(self, paper_id: str) -> list[SemanticPaper]:
        """Return normalized papers that cite the supplied paper."""
        paper = self.get_paper(paper_id)
        return [
            SemanticPaper(
                paper_id=item.paper_id,
                title=item.title,
                year=item.year,
                source="semanticscholar",
            )
            for item in (paper.citations if paper else ())
        ]

    def backward_citations(self, paper_id: str) -> list[SemanticPaper]:
        """Return normalized papers referenced by the supplied paper."""
        paper = self.get_paper(paper_id)
        return [
            SemanticPaper(
                paper_id=item.paper_id,
                title=item.title,
                year=item.year,
                source="semanticscholar",
            )
            for item in (paper.references if paper else ())
        ]

    def _get_json(self, url: str, *, params: dict[str, Any]) -> dict[str, Any] | None:
        """Perform a retried GET and return JSON or ``None`` on failure."""
        self.last_error = None
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                self.last_error = "invalid JSON response"
                return None
            except (requests.RequestException, ValueError) as exc:
                self.last_error = type(exc).__name__
                LOGGER.warning(
                    "Semantic Scholar request attempt %d failed: %s",
                    attempt + 1,
                    exc,
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        return None


def _normalize_paper(
    item: dict[str, Any], *, include_links: bool = False
) -> SemanticPaper | None:
    """Normalize documented Graph API fields into a typed record."""
    paper_id = item.get("paperId")
    if not isinstance(paper_id, str) or not paper_id.strip():
        return None
    external_ids = item.get("externalIds") or {}
    authors = tuple(
        author["name"]
        for author in item.get("authors") or []
        if isinstance(author, dict) and isinstance(author.get("name"), str)
    )
    return SemanticPaper(
        paper_id=paper_id,
        title=item.get("title"),
        abstract=item.get("abstract"),
        year=_normalize_year(item.get("year")),
        authors=authors,
        cited_by_count=item.get("citationCount"),
        citation_count=item.get("citationCount"),
        reference_count=item.get("referenceCount"),
        url=item.get("url"),
        arxiv_id=external_ids.get("ArXiv"),
        doi=external_ids.get("DOI"),
        references=_normalize_related(item.get("references"))
        if include_links
        else (),
        citations=_normalize_related(item.get("citations")) if include_links else (),
        source="semanticscholar",
    )


def _normalize_related(items: Any) -> tuple[RelatedPaper, ...]:
    """Normalize an optional list of citation-edge paper records."""
    related: list[RelatedPaper] = []
    if not isinstance(items, list):
        return ()
    for item in items:
        if not isinstance(item, dict):
            continue
        # The paper detail fields return paper records directly, while the
        # dedicated citations/references endpoints wrap them in citingPaper or
        # citedPaper. Accept both documented response shapes.
        paper = item.get("citingPaper") or item.get("citedPaper") or item
        if (
            not isinstance(paper, dict)
            or not isinstance(paper.get("paperId"), str)
            or not paper["paperId"].strip()
        ):
            continue
        related.append(
            RelatedPaper(
                paper_id=paper["paperId"],
                title=paper.get("title"),
                year=_normalize_year(paper.get("year")),
            )
        )
    return tuple(related)


def _normalize_year(value: Any) -> int | None:
    """Keep positive integer publication years; do not guess malformed values."""
    return value if type(value) is int and value > 0 else None
