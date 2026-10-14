"""CLI tests for M5 review and evaluation output."""

from unittest.mock import Mock, patch

from agents.schemas import WorkflowResult
from cli import build_parser, run_search


def test_search_parser_accepts_review_and_eval_flags() -> None:
    """The CLI exposes report and metric output switches."""
    args = build_parser().parse_args([
        "search", "retrieval", "--limit", "3", "--review", "--eval", "--json"
    ])
    assert args.review is True
    assert args.eval is True


def test_search_includes_m5_outputs_by_default() -> None:
    """The standard search command emits the M5 artifacts without extra flags."""
    args = build_parser().parse_args(["search", "retrieval"])
    assert args.review is True
    assert args.eval is True


def test_search_parser_accepts_stream_flag() -> None:
    """The CLI exposes streaming review output."""
    args = build_parser().parse_args(["search", "retrieval", "--stream"])
    assert args.stream is True


def test_run_search_can_generate_review_and_metrics(tmp_path) -> None:
    """Runner returns the workflow, report and evaluation objects together."""
    workflow = Mock()
    result = WorkflowResult(
        query="q",
        recommendations=[],
        analyses=[],
        evidence={},
        unresolved_citations=[],
        errors=[],
        warnings=[],
        steps=["retrieve"],
    )
    workflow.run.return_value = result
    from config import Settings

    settings = Settings(_env_file=None, cache_dir=str(tmp_path))
    with patch("cli.build_default_workflow", return_value=workflow):
        output = run_search(
            "q",
            limit=2,
            include_review=True,
            include_eval=True,
            settings=settings,
        )
    assert output.workflow == result
    assert output.review is not None
    assert output.metrics is not None
    workflow.run.assert_called_once_with("q", limit=2)
