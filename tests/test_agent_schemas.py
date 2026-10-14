"""Typed analysis and provenance contracts."""

import pytest
from pydantic import ValidationError

from agents.schemas import Finding, ExtractedFindings


def test_findings_require_ids_and_evidence() -> None:
    """A finding cannot omit its source or provide unknown schema fields."""
    with pytest.raises(ValidationError):
        Finding(text="claim", paper_id="", evidence_quote="quote")
    with pytest.raises(ValidationError):
        Finding(text="claim", paper_id="p", evidence_quote="quote", invented=1)
    with pytest.raises(ValidationError):
        ExtractedFindings(paper_id="p")


def test_findings_are_structured_and_serializable() -> None:
    """Methods, experiments and conclusions remain individually attributable."""
    finding = Finding(text="A method.", paper_id="p", evidence_quote="A method.")
    result = ExtractedFindings(
        paper_id="p", methods=[finding], experiments=[], conclusions=[], reason=finding,
    )
    assert result.model_dump()["methods"][0]["paper_id"] == "p"
