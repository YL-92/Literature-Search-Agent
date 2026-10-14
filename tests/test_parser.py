"""Tests for PDF text extraction."""

from __future__ import annotations

import fitz

from rag.parser import extract_pdf_text, extract_pdf_pages


def _sample_pdf() -> bytes:
    """Create a small in-memory PDF fixture."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A Small Research Paper\n\nMethod and results.")
    data = document.tobytes()
    document.close()
    return data


def test_extract_pdf_text_and_pages() -> None:
    """Parser should preserve page boundaries and recover readable text."""
    data = _sample_pdf()

    assert "Small Research Paper" in extract_pdf_text(data)
    pages = extract_pdf_pages(data)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Method and results" in pages[0].text


def test_invalid_pdf_returns_empty_content() -> None:
    """Malformed PDFs should degrade to empty content instead of crashing."""
    assert extract_pdf_text(b"not a pdf") == ""
    assert extract_pdf_pages(b"not a pdf") == []
