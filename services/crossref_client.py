"""可选的 Crossref DOI 元数据回退。Crossref 的 REST API 返回一个 JSON 包，
其工作元数据位于 ``message`` 中。
该适配器刻意与 OpenAlex 的主要来源分开，除非应用程序明确将其接入，否则不会调用它。
"""

from __future__ import annotations

import html
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from config import Settings, get_settings
from services.paper import Paper

LOGGER = logging.getLogger(__name__)
_TAG = re.compile(r"<[^>]+>")


class CrossrefClient:
    """使用礼貌池标识和重试来检索 DOI 元数据。"""

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
        """初始化可注入的 Crossref 传输。"""
        self.settings = settings or get_settings()
        self.session = session or requests.Session()
        self.base_url = (base_url or self.settings.crossref_base_url).rstrip("/")
        self.timeout = timeout or self.settings.request_timeout
        self.max_retries = max_retries or self.settings.request_max_retries
        self.backoff_seconds = (
            self.settings.crossref_backoff_seconds
            if backoff_seconds is None
            else backoff_seconds
        )
        self.last_error: str | None = None
        self.cache_dir = Path(self.settings.cache_dir) / "crossref"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_work(self, doi: str) -> Paper | None:
        """通过 DOI 检索并规范化一条 Crossref 作品。"""
        normalized = _normalize_doi(doi)
        if not normalized:
            return None
        payload = self._get_json(f"/works/{quote(normalized, safe='')}")
        return _parse_message(payload)

    def enrich(self, paper: Paper) -> Paper:
        """从 DOI 记录中填写缺失的标题/年份/作者/摘要字段。"""
        if not paper.doi:
            return paper
        fallback = self.get_work(paper.doi)
        if fallback is None:
            return paper
        updates = {
            "title": paper.title or fallback.title,
            "year": paper.year or fallback.year,
            "authors": paper.authors or fallback.authors,
            "abstract": paper.abstract or fallback.abstract,
        }
        return paper.model_copy(update=updates)

    def _get_json(self, path: str) -> dict[str, Any] | None:
        """使用有界指数重试获取 Crossref JSON。"""
        self.last_error = None
        params = {"mailto": self.settings.crossref_mailto} if self.settings.crossref_mailto else {}
        headers = {"User-Agent": "litsearch-agent/0.1"}
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Crossref returned a non-object JSON payload")
                return payload
            except (requests.RequestException, ValueError) as exc:
                self.last_error = type(exc).__name__
                LOGGER.warning(
                    "Crossref request attempt %d failed (%s)",
                    attempt + 1,
                    type(exc).__name__,
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        return None


def _parse_message(payload: dict[str, Any] | None) -> Paper | None:
    """将 Crossref JSON 信封映射为规范化的论文。"""
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, dict):
        return None
    doi = message.get("DOI")
    if not isinstance(doi, str) or not doi.strip():
        return None
    authors = tuple(
        " ".join(part for part in (author.get("given"), author.get("family")) if part)
        for author in message.get("author") or []
        if isinstance(author, dict) and (author.get("given") or author.get("family"))
    )
    return Paper(
        paper_id=f"doi:{doi}",
        title=_first_string(message.get("title")),
        year=_crossref_year(message),
        doi=doi,
        authors=authors,
        abstract=_clean_abstract(message.get("abstract")),
        source="crossref",
    )


def _normalize_doi(value: str) -> str:
    """接受纯 DOI 和 DOI URL，拒绝非 DOI 值。"""
    value = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
            break
    return value if value.lower().startswith("10.") and "/" in value else ""


def _first_string(value: Any) -> str | None:
    """从 Crossref 数组中返回第一个非空字符串。"""
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0].strip() or None
    return None


def _crossref_year(message: dict[str, Any]) -> int | None:
    """从 Crossref date-parts 字段中读取第一个有效年份。"""
    for key in ("published", "published-print", "published-online", "issued", "created"):
        date = message.get(key, {}).get("date-parts") if isinstance(message.get(key), dict) else None
        if isinstance(date, list) and date and isinstance(date[0], list) and date[0]:
            year = date[0][0]
            if type(year) is int and year > 0:
                return year
    return None


def _clean_abstract(value: Any) -> str | None:
    """去除 JATS/XML 标记，同时保留可读的摘要文本。"""
    if not isinstance(value, str) or not value.strip():
        return None
    return html.unescape(_TAG.sub("", value)).strip() or None
