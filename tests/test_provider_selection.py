"""Academic-source selection and default-disabled S2 tests."""

from unittest.mock import Mock, patch

from config import Settings
from services.openalex_client import OpenAlexClient
from agents.runtime import build_default_workflow


def test_openalex_is_default_and_s2_is_disabled() -> None:
    """No Semantic Scholar client is created under default settings."""
    settings = Settings(_env_file=None)
    assert settings.scholar_provider == "openalex"
    assert settings.semantic_scholar_enabled is False
    with patch("agents.runtime.SemanticScholarClient") as s2:
        build_default_workflow(settings=settings)
        s2.assert_not_called()


def test_s2_is_used_only_when_enabled_and_openalex_fails(tmp_path) -> None:
    """Configured S2 is a fallback, never the primary source."""
    settings = Settings(
        _env_file=None,
        cache_dir=str(tmp_path),
        semantic_scholar_enabled=True,
        semantic_scholar_api_key="configured",
        semantic_scholar_api_url="https://s2.test/graph/v1",
    )
    openalex = Mock()
    openalex.search_papers.return_value = []
    openalex.last_error = "429"
    s2 = Mock()
    s2.search.return_value = []
    with patch("agents.runtime.OpenAlexClient", return_value=openalex), \
         patch("agents.runtime.SemanticScholarClient", return_value=s2):
        workflow = build_default_workflow(settings=settings)
        workflow.search("query", 2)
    s2.search.assert_called_once_with("query", limit=2)


def test_s2_is_not_called_when_openalex_returns_results(tmp_path) -> None:
    """A successful OpenAlex response prevents fallback calls."""
    settings = Settings(
        _env_file=None,
        cache_dir=str(tmp_path),
        semantic_scholar_enabled=True,
        semantic_scholar_api_key="configured",
        semantic_scholar_api_url="https://s2.test/graph/v1",
    )
    openalex = Mock()
    openalex.search_papers.return_value = [Mock(paper_id="W1")]
    openalex.last_error = None
    s2 = Mock()
    with patch("agents.runtime.OpenAlexClient", return_value=openalex), \
         patch("agents.runtime.SemanticScholarClient", return_value=s2):
        workflow = build_default_workflow(settings=settings)
        assert workflow.search("query", 2)
    s2.search.assert_not_called()
