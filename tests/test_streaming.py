"""M6 streaming-output tests."""

from agents.schemas import ReviewReport
from services.streaming import stream_review


def test_stream_review_yields_reconstructable_chunks() -> None:
    """Streaming chunks concatenate to the exact review summary."""
    report = ReviewReport(
        query="q",
        recommendations=[],
        comparison_table=[],
        summary="第一句。第二句。第三句。",
        warnings=[],
    )

    chunks = list(stream_review(report, chunk_size=4))

    assert chunks
    assert all(chunk for chunk in chunks)
    assert "".join(chunks) == report.summary


def test_stream_review_rejects_invalid_chunk_size() -> None:
    """A non-positive chunk size is a caller error."""
    report = ReviewReport(
        query="q",
        recommendations=[],
        comparison_table=[],
        summary="summary",
        warnings=[],
    )
    try:
        list(stream_review(report, chunk_size=0))
    except ValueError:
        pass
    else:
        raise AssertionError("chunk_size=0 must fail")
