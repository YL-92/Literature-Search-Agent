"""Provider-neutral paper records shared by search and citation layers."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class RelatedPaper:
    """Compact citation-edge record returned by provider adapters."""

    paper_id: str
    title: str | None = None
    year: int | None = None


class Paper(BaseModel):
    """Normalized academic paper metadata independent of its upstream source."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str | None = None
    year: int | None = None
    cited_by_count: int | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    authors: tuple[str, ...] = ()
    abstract: str | None = None
    references: tuple[RelatedPaper, ...] = ()
    citations: tuple[RelatedPaper, ...] = ()
    source: str = Field(default="unknown")
