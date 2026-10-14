"""Source-grounded review generation for the M5 CLI output."""

from __future__ import annotations

from agents.schemas import (
    ComparisonRow,
    ReviewReport,
    WorkflowResult,
)


class ReviewGenerator:
    """Build a deterministic review from verified workflow evidence.

    The generator intentionally does not invent cross-paper claims. Every
    sentence is copied from a verified finding or recommendation and receives
    a ``[来源: 论文标题]`` marker.
    """

    def generate(self, result: WorkflowResult) -> ReviewReport:
        """Create a comparison table and a source-tagged research summary."""
        warnings = list(result.warnings)
        if result.unresolved_citations:
            warnings.append("review suppressed because unresolved citations remain")
            return ReviewReport(
                query=result.query,
                recommendations=result.recommendations,
                comparison_table=[],
                summary="",
                warnings=warnings,
            )

        analyses = {analysis.paper_id: analysis for analysis in result.analyses}
        rows: list[ComparisonRow] = []
        sentences: list[str] = []
        for recommendation in result.recommendations:
            analysis = analyses.get(recommendation.paper_id)
            if analysis is None:
                warnings.append(f"missing analysis: {recommendation.paper_id}")
                continue
            methods = _join_findings(analysis.methods)
            experiments = _join_findings(analysis.experiments)
            conclusions = _join_findings(analysis.conclusions)
            rows.append(
                ComparisonRow(
                    paper_id=analysis.paper_id,
                    source_title=analysis.title,
                    methods=methods,
                    experiments=experiments,
                    conclusions=conclusions,
                    evidence_quote=recommendation.evidence_quote,
                )
            )
            sentences.append(
                f"{recommendation.reason} [来源: {analysis.title}]"
            )
            for finding in analysis.methods:
                sentences.append(f"方法：{finding.text} [来源: {analysis.title}]")
            for finding in analysis.experiments:
                sentences.append(f"实验：{finding.text} [来源: {analysis.title}]")
            for finding in analysis.conclusions:
                sentences.append(f"结论：{finding.text} [来源: {analysis.title}]")

        return ReviewReport(
            query=result.query,
            recommendations=result.recommendations,
            comparison_table=rows,
            summary=" ".join(sentences),
            warnings=warnings,
        )


def _join_findings(findings: list) -> str:
    """Join source-validated findings without adding interpretation."""
    return "；".join(finding.text for finding in findings)
