"""arXiv Atom API 适配器。Atom 字段名称和查询参数遵循
arXiv API 用户手册： https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import requests

from config import get_settings

LOGGER = logging.getLogger(__name__)
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


@dataclass(frozen=True)
class Paper:
    """文献服务返回的标准化论文元数据。"""

    paper_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    published: str | None = None
    updated: str | None = None
    categories: tuple[str, ...] = ()
    pdf_url: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    source: str = "arxiv"


class ArxivClient:
    """通过其公共 Atom API 使用重试和回退功能搜索 arXiv。"""

    def __init__(
        self,
        session: requests.Session | Any | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        backoff_seconds: float = 0.5,
    ) -> None:
        """使用可注入传输初始化客户端以进行测试。"""
        settings = get_settings()
        self.session = session or requests.Session()
        self.base_url = base_url or settings.arxiv_api_url
        self.timeout = timeout or settings.request_timeout
        self.max_retries = max_retries or settings.request_max_retries
        self.backoff_seconds = backoff_seconds

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """通过 arXiv 查询语法搜索；在服务失败时返回空列表。"""
        if not query.strip() or max_results < 1:
            return []
        params = {
            "search_query": f"all:{query.strip()}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.base_url, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                return self._parse_feed(response.text)
            except (requests.RequestException, ET.ParseError, ValueError) as exc:
                LOGGER.warning("arXiv search attempt %d failed: %s", attempt + 1, exc)
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        return []

    @staticmethod
    def _parse_feed(xml_text: str) -> list[Paper]:
        """解析文档化的 arXiv Atom 提要，不要猜测字段。"""
        root = ET.fromstring(xml_text)
        papers: list[Paper] = []
        for entry in root.findall(f"{ATOM_NS}entry"):
            raw_id = _text(entry, f"{ATOM_NS}id")
            if not raw_id:
                continue
            paper_id = raw_id.rsplit("/abs/", 1)[-1].split("v", 1)[0]
            pdf_link = next(
                (
                    link.attrib.get("href")
                    for link in entry.findall(f"{ATOM_NS}link")
                    if link.attrib.get("title") == "pdf"
                ),
                None,
            )
            papers.append(
                Paper(
                    paper_id=paper_id,
                    title=" ".join(_text(entry, f"{ATOM_NS}title").split()),
                    abstract=" ".join(_text(entry, f"{ATOM_NS}summary").split()),
                    authors=tuple(
                        name
                        for author in entry.findall(f"{ATOM_NS}author")
                        if (name := _text(author, f"{ATOM_NS}name"))
                    ),
                    published=_optional_text(entry, f"{ATOM_NS}published"),
                    updated=_optional_text(entry, f"{ATOM_NS}updated"),
                    categories=tuple(
                        category.attrib["term"]
                        for category in entry.findall(f"{ATOM_NS}category")
                        if "term" in category.attrib
                    ),
                    pdf_url=pdf_link,
                    doi=_optional_text(entry, f"{ARXIV_NS}doi"),
                    journal_ref=_optional_text(entry, f"{ARXIV_NS}journal_ref"),
                )
            )
        return papers


def _text(element: ET.Element, path: str) -> str:
    """获取子元素的去除空白的文本或一个空字符串。"""
    child = element.find(path)
    return (child.text or "").strip() if child is not None else ""


def _optional_text(element: ET.Element, path: str) -> str | None:
    """获取可选的子文本，将空值规范化为 ``None``。"""
    return _text(element, path) or None
