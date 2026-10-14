"""Regression checks for the OpenAlex-first migration contract."""

from pathlib import Path

from config import Settings


def test_default_source_contract() -> None:
    """OpenAlex is primary and Semantic Scholar is opt-in only."""
    settings = Settings(_env_file=None)
    assert settings.scholar_provider == "openalex"
    assert settings.semantic_scholar_enabled is False
    assert settings.openalex_base_url == "https://api.openalex.org"
    assert settings.openalex_rps == 8


def test_no_hardcoded_semantic_scholar_host() -> None:
    """The fallback host must come from configuration, not source code."""
    source_root = Path(__file__).parents[1]
    forbidden_host = "api." + "semanticscholar.org"
    matches = [
        path
        for path in source_root.rglob("*.py")
        if forbidden_host in path.read_text(encoding="utf-8")
    ]
    assert matches == []
