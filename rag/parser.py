"""PDF text extraction using PyMuPDF."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PdfInput = bytes | bytearray | str | os.PathLike[str]


@dataclass(frozen=True)
class PdfPage:
    """从一个以 1 为基础的 PDF 页面提取的文本。"""

    page_number: int
    text: str


def extract_pdf_pages(source: PdfInput) -> list[PdfPage]:
    """从 PDF 字节或文件系统路径逐页提取文本。
    无效文件和提取失败会降级为空页列表，因此单个格式错误的文档不会阻止批处理工作流。
    """
    try:
        import fitz

        if isinstance(source, (bytes, bytearray)):
            document = fitz.open(stream=bytes(source), filetype="pdf")
        else:
            document = fitz.open(Path(source))
        try:
            return [
                PdfPage(page_number=index + 1, text=page.get_text("text").strip())
                for index, page in enumerate(document)
                if page.get_text("text").strip()
            ]
        finally:
            document.close()
    except Exception as exc:
        LOGGER.warning("Unable to parse PDF: %s", exc)
        return []


def extract_pdf_text(source: PdfInput) -> str:
    """提取并合并所有非空的 PDF 页面，并添加页面分隔符。"""
    return "\n\n".join(page.text for page in extract_pdf_pages(source))
