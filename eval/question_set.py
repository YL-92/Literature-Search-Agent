"""Load and validate the reproducible literature-search question set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_QUESTION_SET = (
    Path(__file__).resolve().parents[1] / "data" / "test_questions.jsonl"
)


class QuestionCase(BaseModel):
    """One benchmark question and its expected evaluation dimensions."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    query: str = Field(min_length=8)
    expected_answer_type: str = Field(min_length=1)
    citation_requirement: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)


def load_question_set(path: str | Path = DEFAULT_QUESTION_SET) -> list[QuestionCase]:
    """Read JSONL questions and reject malformed or duplicate entries."""
    source = Path(path)
    cases: list[QuestionCase] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            cases.append(QuestionCase.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid question at line {line_number}") from exc
    validate_question_set(cases)
    return cases


def validate_question_set(
    cases: list[QuestionCase], *, minimum: int = 100
) -> None:
    """Validate size, uniqueness, and category coverage."""
    if len(cases) < minimum:
        raise ValueError(f"question set must contain at least {minimum} cases")
    ids = [case.question_id for case in cases]
    queries = [case.query.casefold().strip() for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("question_id values must be unique")
    if len(set(queries)) != len(queries):
        raise ValueError("query values must be unique")
    if len({case.category for case in cases}) < 8:
        raise ValueError("question set must cover at least eight categories")


def summarize_question_set(cases: list[QuestionCase]) -> dict[str, Any]:
    """Return stable counts suitable for reports and CI logs."""
    category_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1
        difficulty_counts[case.difficulty] = difficulty_counts.get(case.difficulty, 0) + 1
    return {
        "count": len(cases),
        "categories": category_counts,
        "difficulties": difficulty_counts,
    }
