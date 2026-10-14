"""Validate and summarize the checked-in benchmark question set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.question_set import load_question_set, summarize_question_set


def main() -> None:
    """Print a machine-readable dataset summary."""
    cases = load_question_set()
    print(json.dumps(summarize_question_set(cases), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
