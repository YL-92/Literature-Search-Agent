"""Command-line entry point for the literature search agent."""

import argparse
import json

from agents.runtime import build_default_workflow
from agents.schemas import WorkflowResult
from config import Settings, get_settings
from eval.ragas_eval import EvaluationMetrics, evaluate_workflow
from services.semantic_cache import SemanticCache
from services.review_service import ReviewGenerator
from services.streaming import stream_review


class SearchOutput:
    """Aggregate workflow, review and evaluation outputs for CLI callers."""

    def __init__(
        self,
        workflow: WorkflowResult,
        review: object | None = None,
        metrics: EvaluationMetrics | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Store optional M5 artifacts without changing the workflow contract."""
        self.workflow = workflow
        self.review = review
        self.metrics = metrics
        self.cache_hit = cache_hit

    def __getattr__(self, name: str) -> object:
        """Preserve the pre-M5 runner's direct workflow-result attributes."""
        return getattr(self.workflow, name)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="litsearch-agent",
        description="Discover research papers and trace their citations.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="litsearch-agent 0.1.0",
    )
    commands = parser.add_subparsers(dest="command")
    search = commands.add_parser(
        "search", help="Search papers and verify recommendation provenance."
    )
    search.add_argument("query", help="Research question or topic.")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--json", action="store_true", dest="json")
    search.add_argument("--review", action="store_true")
    search.add_argument("--eval", action="store_true", dest="eval")
    search.add_argument("--stream", action="store_true")
    search.set_defaults(review=True, eval=True)
    return parser


def run_search(
    query: str,
    *,
    limit: int = 10,
    include_review: bool = True,
    include_eval: bool = True,
    stream: bool = False,
    settings: Settings | None = None,
    cache: SemanticCache | None = None,
) -> SearchOutput:
    """Run the workflow and optionally produce M5 review and metrics."""
    runtime_settings = settings or get_settings()
    query = query.strip()
    semantic_cache = cache or SemanticCache(settings=runtime_settings)
    cache_hit = False
    cached = (
        semantic_cache.get(query, limit)
        if runtime_settings.semantic_cache_enabled
        else None
    )
    if cached is not None:
        workflow_result = WorkflowResult.model_validate(cached)
        cache_hit = True
    else:
        workflow_result = build_default_workflow(settings=runtime_settings).run(
            query, limit=limit
        )
        if runtime_settings.semantic_cache_enabled:
            semantic_cache.set(query, limit, workflow_result.model_dump())
    review = ReviewGenerator().generate(workflow_result) if include_review else None
    metrics = evaluate_workflow(workflow_result) if include_eval else None
    return SearchOutput(workflow_result, review, metrics, cache_hit)


def main() -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args()
    if args.command == "search":
        output = run_search(
            args.query,
            limit=args.limit,
            include_review=args.review,
            include_eval=args.eval,
            stream=args.stream,
        )
        result = output.workflow
        if args.json:
            payload = result.model_dump()
            payload["cache_hit"] = output.cache_hit
            if output.review is not None:
                payload["review"] = output.review.model_dump()
            if output.metrics is not None:
                payload["metrics"] = output.metrics.as_dict()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        _print_human_result(result)
        print(f"Cache: {'hit' if output.cache_hit else 'miss'}")
        if output.review is not None:
            print("\nComparison table:")
            for row in output.review.comparison_table:
                print(
                    f"- {row.source_title} [{row.paper_id}] | "
                    f"方法: {row.methods or '未抽取'} | "
                    f"实验: {row.experiments or '未抽取'} | "
                    f"结论: {row.conclusions or '未抽取'}"
                )
            print("\nReview:")
            if args.stream:
                for chunk in stream_review(
                    output.review,
                    chunk_size=get_settings().stream_chunk_size,
                ):
                    print(chunk, end="", flush=True)
                print()
            else:
                print(output.review.summary or "No verified review.")
        if output.metrics is not None:
            print("\nMetrics:")
            print(json.dumps(output.metrics.as_dict(), ensure_ascii=False))


def _print_human_result(result: WorkflowResult) -> None:
    """Print a concise human-readable result without unsupported claims."""
    print(f"Query: {result.query}")
    for recommendation in result.recommendations:
        print(f"- {recommendation.title} [{recommendation.paper_id}]")
        print(f"  Reason: {recommendation.reason}")
        print(f"  Source: {recommendation.reason_paper_id}")
    if not result.recommendations:
        print("No verified recommendations.")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    for error in result.errors:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
