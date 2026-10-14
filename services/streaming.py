"""用于评审输出的确定性流式辅助工具。"""

from collections.abc import Iterator

from agents.schemas import ReviewReport


def stream_review(report: ReviewReport, *, chunk_size: int = 120) -> Iterator[str]:
    """在不改变文本的情况下，将评论摘要分块输出。"""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(report.summary), chunk_size):
        yield report.summary[start:start + chunk_size]
