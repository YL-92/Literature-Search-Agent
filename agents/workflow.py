"""用于可追踪论文推荐的单线程 LangGraph 工作流程。"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.schemas import (
    CitationIssue,
    PaperAnalysis,
    Recommendation,
    WorkflowResult,
)
from config import Settings, get_settings
from services.analysis_client import AnalysisClient
from services.paper import Paper

LOGGER = logging.getLogger(__name__)
PaperSearch = Callable[[str, int], list[Paper]]
EventSink = Callable[[str], Any]


class WorkflowState(TypedDict, total=False):
    """在编译图节点之间传递的内部状态。"""

    query: str
    limit: int
    candidates: list[Paper]
    selected: list[Paper]
    analyses: list[PaperAnalysis]
    recommendations: list[Recommendation]
    unresolved_citations: list[CitationIssue]
    evidence: dict[str, str]
    errors: list[str]
    warnings: list[str]
    steps: list[str]
    stop: bool


class ResearchWorkflow:
    """编译并运行检索 → 过滤 → 分析 → 验证 节点。"""

    def __init__(
        self,
        *,
        search: PaperSearch,
        analyzer: AnalysisClient | Any | None = None,
        settings: Settings | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        """Inject service functions so the workflow remains testable offline."""
        self.settings = settings or get_settings()
        self.search = search
        self.analyzer = analyzer or AnalysisClient(settings=self.settings)
        self.event_sink = event_sink or (lambda _step: None)
        builder = StateGraph(WorkflowState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("filter", self._filter)
        builder.add_node("analyze", self._analyze)
        builder.add_node("verify", self._verify)
        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            lambda state: "stop" if state.get("stop") else "continue",
            {"stop": END, "continue": "filter"},
        )
        builder.add_edge("filter", "analyze")
        builder.add_edge("analyze", "verify")
        builder.add_edge("verify", END)
        self.graph = builder.compile()

    def run(self, query: str, *, limit: int = 10) -> WorkflowResult:
        """运行已编译的图并返回可 JSON 序列化的可追踪输出。"""
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")
        if limit < 1:
            raise ValueError("limit must be positive")
        state = self.graph.invoke(
            {
                "query": query,
                "limit": limit,
                "candidates": [],
                "selected": [],
                "analyses": [],
                "recommendations": [],
                "unresolved_citations": [],
                "evidence": {},
                "errors": [],
                "warnings": [],
                "steps": [],
                "stop": False,
            }
        )
        return WorkflowResult(
            query=query,
            recommendations=state.get("recommendations", []),
            analyses=state.get("analyses", []),
            evidence=state.get("evidence", {}),
            unresolved_citations=state.get("unresolved_citations", []),
            errors=state.get("errors", []),
            warnings=state.get("warnings", []),
            steps=state.get("steps", []),
        )

    def _record(self, state: WorkflowState, step: str) -> list[str]:
        """添加一个步骤并通知可选的观察者。"""
        self.event_sink(step)
        return [*state.get("steps", []), step]

    def _retrieve(self, state: WorkflowState) -> dict[str, Any]:
        """通过注入的服务边界检索候选人元数据。"""
        try:
            candidates = self.search(state["query"], state["limit"])
            return {
                "candidates": _deduplicate_papers(candidates),
                "steps": self._record(state, "retrieve"),
            }
        except Exception as exc:
            LOGGER.warning("Retrieval failed (%s)", type(exc).__name__)
            return {
                "candidates": [],
                "errors": [*state.get("errors", []), "search service unavailable"],
                "steps": self._record(state, "retrieve"),
                "stop": True,
            }

    def _filter(self, state: WorkflowState) -> dict[str, Any]:
        """保留有可用ID、标题和摘要的论文。"""
        usable = [
            paper
            for paper in state.get("candidates", [])
            if paper.paper_id.strip() and (paper.title or "").strip()
            and (paper.abstract or "").strip()
        ]
        query_terms = set(_tokens(state["query"]))
        scored = [
            (
                sum(term in set(_tokens(f"{paper.title} {paper.abstract}"))
                    for term in query_terms),
                index,
                paper,
            )
            for index, paper in enumerate(usable)
        ]
        relevant = [item for item in scored if item[0] > 0]
        selected = [
            paper for _score, _index, paper in sorted(
                relevant or scored, key=lambda item: (-item[0], item[1])
            )
        ]
        return {"selected": selected, "steps": self._record(state, "filter")}

    def _analyze(self, state: WorkflowState) -> dict[str, Any]:
        """仅分析选定的论文，保留每篇论文的证据。"""
        analyses: list[PaperAnalysis] = []
        evidence: dict[str, str] = {}
        warnings = list(state.get("warnings", []))
        for paper in state.get("selected", []):
            try:
                analysis = self.analyzer.analyze(paper)
            except Exception as exc:
                LOGGER.warning("Analysis failed (%s)", type(exc).__name__)
                analysis = None
            if analysis is None:
                warnings.append(f"analysis unavailable: {paper.paper_id}")
                continue
            analyses.append(analysis)
            evidence[paper.paper_id] = paper.abstract or ""
        return {
            "analyses": analyses,
            "evidence": evidence,
            "warnings": warnings,
            "steps": self._record(state, "analyze"),
        }
    def _verify(self, state: WorkflowState) -> dict[str, Any]:
        """仅接受其论文编号和引用与检索到的证据相符的发现。"""
        known = set(state.get("evidence", {}))
        analyses: list[PaperAnalysis] = []
        recommendations: list[Recommendation] = []
        issues: list[CitationIssue] = []
        for analysis in state.get("analyses", []):
            if analysis.paper_id not in known:
                issues.append(CitationIssue(
                    paper_id=analysis.paper_id, problem="analysis paper is not a retrieved result"
                ))
                continue
            findings = [
                *analysis.methods, *analysis.experiments,
                *analysis.conclusions, analysis.reason,
            ]
            invalid = [
                finding for finding in findings
                if finding.paper_id not in known
                or finding.paper_id != analysis.paper_id
                or finding.evidence_quote not in state["evidence"].get(finding.paper_id, "")
            ]
            if invalid:
                issues.extend(
                    CitationIssue(paper_id=finding.paper_id, problem="unresolved citation")
                    for finding in invalid
                )
                continue
            analyses.append(analysis)
            recommendations.append(
                Recommendation(
                    paper_id=analysis.paper_id,
                    title=analysis.title,
                    reason=_one_sentence(analysis.reason.text),
                    reason_paper_id=analysis.reason.paper_id,
                    evidence_quote=analysis.reason.evidence_quote,
                    mode=analysis.mode,
                    evidence_scope=analysis.evidence_scope,
                )
            )
        return {
            "analyses": analyses,
            "recommendations": recommendations,
            "unresolved_citations": issues,
            "steps": self._record(state, "verify"),
        }


def _tokens(text: str) -> list[str]:
    """将拉丁术语和单个中日韩字符分词以进行轻量过滤。"""
    return re.findall(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", text.casefold())


def _one_sentence(text: str) -> str:
    """Bound a recommendation reason to its first source-backed sentence."""
    for marker in (". ", "。", "! ", "！", "? ", "？"):
        position = text.find(marker)
        if position >= 0:
            return text[:position + 1].strip()
    return text.strip()

def _deduplicate_papers(papers: Sequence[Paper]) -> list[Paper]:
    """Keep first-seen paper order while preventing duplicate IDs."""
    seen: set[str] = set()
    result: list[Paper] = []
    for paper in papers:
        if paper.paper_id in seen:
            continue
        seen.add(paper.paper_id)
        result.append(paper)
    return result
