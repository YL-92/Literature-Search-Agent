"""CLI wiring tests for the M4 runnable workflow."""

from unittest.mock import Mock, patch

from agents.schemas import WorkflowResult
from cli import build_parser, run_search
from config import Settings


def test_search_parser_accepts_query_and_json() -> None:
    """The CLI exposes a non-interactive search command."""
    args = build_parser().parse_args(["search", "retrieval", "--limit", "3", "--json"])
    assert args.command == "search"
    assert args.query == "retrieval"
    assert args.limit == 3
    assert args.json is True


def test_run_search_serializes_workflow_result(tmp_path) -> None:
    """CLI runner returns the workflow result without leaking credentials."""
    fake = Mock()
    fake.run.return_value = WorkflowResult(
        query="q",
        recommendations=[],
        analyses=[],
        evidence={},
        unresolved_citations=[],
        errors=[],
        warnings=[],
        steps=["retrieve", "filter", "analyze", "verify"],
    )
    settings = Settings(_env_file=None, cache_dir=str(tmp_path))
    with patch("cli.build_default_workflow", return_value=fake):
        result = run_search("q", limit=2, settings=settings)
    assert result.query == "q"
    fake.run.assert_called_once_with("q", limit=2)
