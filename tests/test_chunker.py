"""Tests for heading-aware semantic text chunking."""

from rag.chunker import chunk_text


def test_chunk_text_tracks_headings_and_respects_size() -> None:
    """Chunks should include their heading context and stay within the limit."""
    text = (
        "# Introduction\n"
        "First paragraph explains the problem.\n\n"
        "Second paragraph adds context.\n\n"
        "## Method\n"
        "We use a carefully designed experiment."
    )

    chunks = chunk_text(text, max_chars=75, overlap_chars=10)

    assert len(chunks) >= 2
    assert chunks[0].heading == "Introduction"
    assert chunks[-1].heading == "Method"
    assert all(len(chunk.text) <= 75 for chunk in chunks)


def test_empty_input_has_no_chunks() -> None:
    """Whitespace-only documents should not produce empty chunks."""
    assert chunk_text(" \n\t") == []
