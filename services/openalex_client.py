"""OpenAlex Works API adapter used as the primary scholarly source.

The endpoint and field mapping follow the OpenAlex Works API:
https://developers.openalex.org/api-entities/works
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote

import requests

from config import Settings, get_settings
from services.paper import Paper, RelatedPaper

LOGGER = logging.getLogger(__name__)
DEFAULT_SEARCH_SELECT = (
    "id,display_name,publication_year,cited_by_count,doi,authorships,primary_location"
)
DEFAULT_DETAIL_SELECT = (
    "id,display_name,publication_year,cited_by_count,doi,abstract_inverted_index,"
    "authorships,referenced_works"
)
FORWARD_SELECT = "id,display_name,publication_year,cited_by_count,doi,authorships"


def reconstruct_abstract(inv_index: dict[str, list[int]] | None) -> str:
    """Restore an OpenAlex inverted abstract index into reading order."""
    if not isinstance(inv_index, dict) or not inv_index:
        return ""
    positions: dict[int, str] = {}
    for word, locations in inv_index.items():
        if not isinstance(word, str) or not isinstance(locations, list):
            continue
        for position in locations:
            if isinstance(position, int) and position >= 0:
                positions[position] = word
    return " ".join(positions[index] for index in sorted(positions))


class OpenAlexClient:
    """OpenAlex client with local JSON cache, rate limiting and retry fallback."""

    def __init__(
        self,
        session: requests.Session | Any | None = None,
        *,
        settings: Settings | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
    ) -> None:
        """Initialize an injectable, cache-backed OpenAlex transport."""
        self.settings = settings or get_settings()
        self.session = session or requests.Session()
        self.base_url = (base_url or self.settings.openalex_base_url).rstrip("/")
        self.timeout = timeout or self.settings.request_timeout
        self.max_retries = max_retries or self.settings.request_max_retries
        self.backoff_seconds = (
            self.settings.openalex_backoff_seconds
            if backoff_seconds is None
            else backoff_seconds
        )
        self.cache_dir = Path(self.settings.cache_dir) / "openalex"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_error: str | None = None
        self._last_request_at: float | None = None
        self._rate_lock = Lock()

    def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        """Search OpenAlex works by relevance and return normalized papers."""
        if not query.strip() or limit < 1:
            return []
        payload = self._get_json(
            "/works",
            {
                "search": query.strip(),
                "per-page": min(limit, 200),
                "select": DEFAULT_SEARCH_SELECT,
            },
        )
        return _parse_results(payload)

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Compatibility alias for the existing provider interface."""
        return self.search_papers(query, limit)

    def get_paper(self, paper_id: str) -> Paper | None:
        """Fetch one work by OpenAlex ID, URL, DOI, or another provider ID."""
        normalized = _normalize_work_identifier(paper_id)
        if not normalized:
            return None
        payload = self._get_json(
            f"/works/{quote(normalized, safe=':/')}",
            {"select": DEFAULT_DETAIL_SELECT},
        )
        return _parse_paper(payload, include_references=True) if payload else None

    def backward_citations(self, paper_id: str) -> list[Paper]:
        """Return the works referenced by a paper from ``referenced_works``."""
        paper = self.get_paper(paper_id)
        if paper is None:
            return []
        identifiers = [item.paper_id for item in paper.references]
        detailed = self._batch_details(identifiers)
        found = {item.paper_id for item in detailed}
        unresolved = [
            Paper(paper_id=item.paper_id, title=item.title, year=item.year, source="openalex")
            for item in paper.references
            if item.paper_id not in found
        ]
        return [*detailed, *unresolved]

    def forward_citations(self, paper_id: str, limit: int = 50) -> list[Paper]:
        """Return works that cite the supplied work using OpenAlex ``cites``."""
        bare_id = _bare_work_id(paper_id)
        if not bare_id or limit < 1:
            return []
        payload = self._get_json(
            "/works",
            {
                # OpenAlex expresses forward citations as works whose
                # referenced_works contains the seed ID.
                "filter": f"referenced_works:{bare_id}",
                "per-page": min(limit, 200),
                "select": FORWARD_SELECT,
            },
        )
        return _parse_results(payload)

    def _batch_details(self, identifiers: list[str]) -> list[Paper]:
        """Fetch up to 50 related works per request, preserving input order."""
        ids = [_bare_work_id(identifier) for identifier in identifiers]
        ids = [identifier for identifier in ids if identifier]
        result: list[Paper] = []
        for start in range(0, len(ids), 50):
            payload = self._get_json(
                "/works",
                {
                    # Current OpenAlex API uses the nested ids.openalex filter
                    # and canonical work URLs (not bare IDs) for batch lookups.
                    "filter": "ids.openalex:" + "|".join(
                        f"https://openalex.org/{identifier}"
                        for identifier in ids[start:start + 50]
                    ),
                    "select": DEFAULT_SEARCH_SELECT,
                    "per-page": min(50, len(ids) - start),
                },
            )
            result.extend(_parse_results(payload))
        by_id = {paper.paper_id: paper for paper in result}
        return [by_id[identifier] for identifier in identifiers if identifier in by_id]

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """GET a cacheable JSON response with rate limiting and exponential retry."""
        self.last_error = None
        full_params = dict(params)
        if self.settings.openalex_mailto:
            full_params["mailto"] = self.settings.openalex_mailto
        cache_path = self._cache_path(path, full_params)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached
        headers = {"User-Agent": "litsearch-agent/0.1"}
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                response = self.session.get(
                    f"{self.base_url}{path}",
                    params=full_params,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("OpenAlex returned a non-object JSON payload")
                self._write_cache(cache_path, payload)
                return payload
            except (requests.RequestException, ValueError) as exc:
                self.last_error = type(exc).__name__
                LOGGER.warning(
                    "OpenAlex request attempt %d failed (%s)",
                    attempt + 1,
                    type(exc).__name__,
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        return None

    def _wait_for_rate_limit(self) -> None:
        """Keep request spacing at or below the configured requests-per-second."""
        with self._rate_lock:
            now = time.monotonic()
            interval = 1.0 / self.settings.openalex_rps
            if self._last_request_at is not None:
                delay = interval - (now - self._last_request_at)
                if delay > 0:
                    time.sleep(delay)
            self._last_request_at = time.monotonic()

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        """Create a stable cache filename without putting email in a path."""
        serialized = json.dumps([path, sorted(params.items())], sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    @staticmethod
    def _read_cache(path: Path) -> dict[str, Any] | None:
        """Read valid cached JSON and ignore corrupt cache entries."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_cache(path: Path, payload: dict[str, Any]) -> None:
        """Atomically-ish write a successful response to the local cache."""
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            LOGGER.warning("Unable to write OpenAlex cache", exc_info=False)


def _parse_results(payload: dict[str, Any] | None) -> list[Paper]:
    """Normalize a Works collection while skipping malformed records."""
    if not payload or not isinstance(payload.get("results"), list):
        return []
    return [
        paper
        for item in payload["results"]
        if isinstance(item, dict) and (paper := _parse_paper(item)) is not None
    ]


def _parse_paper(
    item: dict[str, Any] | None, *, include_references: bool = False
) -> Paper | None:
    """Map documented OpenAlex work fields to the provider-neutral model."""
    if not isinstance(item, dict):
        return None
    paper_id = item.get("id")
    if not isinstance(paper_id, str) or not paper_id.strip():
        return None
    authors = tuple(
        authorship["author"]["display_name"]
        for authorship in item.get("authorships") or []
        if isinstance(authorship, dict)
        and isinstance(authorship.get("author"), dict)
        and isinstance(authorship["author"].get("display_name"), str)
    )
    references = ()
    if include_references:
        references = tuple(
            RelatedPaper(paper_id=identifier)
            for identifier in item.get("referenced_works") or []
            if isinstance(identifier, str) and identifier.strip()
        )
    doi = item.get("doi")
    return Paper(
        paper_id=paper_id,
        title=item.get("display_name"),
        year=_positive_int(item.get("publication_year")),
        cited_by_count=_nonnegative_int(item.get("cited_by_count")),
        citation_count=_nonnegative_int(item.get("cited_by_count")),
        doi=doi if isinstance(doi, str) else None,
        authors=authors,
        abstract=reconstruct_abstract(item.get("abstract_inverted_index")),
        references=references,
        source="openalex",
    )


def _normalize_work_identifier(value: str) -> str:
    """Normalize a URL or bare OpenAlex/DOI identifier for the works endpoint."""
    value = value.strip()
    if not value:
        return ""
    if value.startswith("https://openalex.org/"):
        return value.rsplit("/", 1)[-1]
    if value.startswith("http://openalex.org/"):
        return value.rsplit("/", 1)[-1]
    if value.lower().startswith("doi:"):
        return "doi:" + value[4:].strip()
    if value.lower().startswith("10."):
        return "doi:" + value
    return value


def _bare_work_id(value: str) -> str:
    """Return the bare W identifier required by OpenAlex filters."""
    normalized = _normalize_work_identifier(value)
    return normalized if normalized.upper().startswith("W") else ""


def _positive_int(value: Any) -> int | None:
    """Normalize optional positive integer fields."""
    return value if type(value) is int and value > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    """Normalize optional non-negative integer counts."""
    return value if type(value) is int and value >= 0 else None
