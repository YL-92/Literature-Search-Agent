"""Heading-aware paragraph chunking for extracted research text."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """带有其最近检测到的章节标题的文本段落。"""

    text: str
    heading: str | None = None


_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


def chunk_text(
    text: str, max_chars: int = 1200, overlap_chars: int = 120
) -> list[TextChunk]:
    """将段落按标题分组为有边界、可重叠的块。"""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
    if not text.strip():
        return []

    chunks: list[TextChunk] = []
    heading: str | None = None
    paragraphs: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            chunks.extend(_pack_paragraphs(heading, paragraphs, max_chars, overlap_chars))
            paragraphs = []
            heading = match.group(2).strip()
        elif line.strip():
            paragraphs.append(line.strip())
        elif paragraphs and paragraphs[-1] != "":
            paragraphs.append("")
    chunks.extend(_pack_paragraphs(heading, paragraphs, max_chars, overlap_chars))
    return chunks


def _pack_paragraphs(
    heading: str | None,
    paragraphs: list[str],
    max_chars: int,
    overlap_chars: int,
) -> list[TextChunk]:
    """在考虑标题上下文的大小限制时打包段落。"""
    content = "\n".join(paragraphs).strip()
    if not content:
        return []
    prefix = f"{heading}\n" if heading else ""
    capacity = max_chars - len(prefix)
    if capacity <= 0:
        prefix = ""
        capacity = max_chars

    segments: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", content):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > capacity:
            if current:
                segments.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + capacity, len(paragraph))
                segments.append(paragraph[start:end])
                if end == len(paragraph):
                    break
                start = max(start + 1, end - overlap_chars)
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= capacity:
            current = candidate
        else:
            if current:
                segments.append(current)
                available_overlap = max(0, capacity - len(paragraph) - 2)
                overlap = min(overlap_chars, available_overlap)
                tail = current[-overlap:] if overlap else ""
                current = f"{tail}\n\n{paragraph}" if tail else paragraph
            else:
                current = paragraph
    if current:
        segments.append(current)
    return [TextChunk(text=f"{prefix}{segment}", heading=heading) for segment in segments]
