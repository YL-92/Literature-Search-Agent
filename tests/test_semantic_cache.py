"""M6 semantic-cache tests."""

import time
from unittest.mock import Mock, patch

from agents.schemas import WorkflowResult
from cli import run_search
from config import Settings
from services.semantic_cache import SemanticCache


def _result() -> WorkflowResult:
    """Create a minimal cacheable workflow result."""
    return WorkflowResult(
        query="Retrieval  Methods",
        recommendations=[],
        analyses=[],
        evidence={},
        unresolved_citations=[],
        errors=[],
        warnings=[],
        steps=["retrieve"],
    )


def test_cache_normalizes_query_and_round_trips_json(tmp_path) -> None:
    """Equivalent query whitespace/case should address one persisted entry."""
    settings = Settings(
        _env_file=None,
        cache_dir=str(tmp_path),
        semantic_cache_ttl_seconds=3600,
    )
    cache = SemanticCache(settings=settings)
    cache.set(" Retrieval   Methods ", 5, {"value": 42})

    assert cache.get("retrieval methods", 5)["value"] == 42
    assert cache.get("retrieval methods", 5) is not None


def test_cache_expiry_does_not_return_stale_results(tmp_path) -> None:
    """Expired entries are ignored and removed from the logical result set."""
    settings = Settings(
        _env_file=None,
        cache_dir=str(tmp_path),
        semantic_cache_ttl_seconds=1,
    )
    cache = SemanticCache(settings=settings)
    cache.set("query", 5, {"value": 42})

    with patch("services.semantic_cache.time.time", return_value=time.time() + 2):
        assert cache.get("query", 5) is None


def test_run_search_second_call_hits_cache_without_workflow(tmp_path) -> None:
    """Repeated queries should avoid invoking the provider workflow."""
    settings = Settings(
        _env_file=None,
        cache_dir=str(tmp_path),
        semantic_cache_ttl_seconds=3600,
    )
    workflow = Mock()
    workflow.run.return_value = _result()
    cache = SemanticCache(settings=settings)

    with patch("cli.build_default_workflow", return_value=workflow):
        first = run_search("Retrieval Methods", limit=5, settings=settings, cache=cache)
        second = run_search(" retrieval   methods ", limit=5, settings=settings, cache=cache)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.query == "Retrieval  Methods"
    assert workflow.run.call_count == 1
