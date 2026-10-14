"""M4 工作流程的默认生产接线。"""

from collections.abc import Callable

from agents.workflow import ResearchWorkflow
from config import Settings, get_settings
from services.analysis_client import AnalysisClient
from services.openalex_client import OpenAlexClient
from services.paper import Paper
from services.semanticscholar_client import SemanticScholarClient


def build_default_workflow(
    *, settings: Settings | None = None,
) -> ResearchWorkflow:
    """使用 OpenAlex 构建一个无前端工作流，可选择使用 S2 作为备用。"""
    runtime_settings = settings or get_settings()
    openalex = OpenAlexClient(settings=runtime_settings)
    s2_client: SemanticScholarClient | None = None

    def search(query: str, limit: int) -> list[Paper]:
        """首先搜索 OpenAlex，然后可选择使用配置的 S2 备用。"""
        nonlocal s2_client
        papers = openalex.search_papers(query, limit=limit)
        if openalex.last_error is None:
            enriched: list[Paper] = []
            for paper in papers:
                if paper.abstract:
                    enriched.append(paper)
                    continue
                detail = openalex.get_paper(paper.paper_id)
                enriched.append(detail or paper)
            return enriched
        if (
            not runtime_settings.semantic_scholar_enabled
            or not runtime_settings.semantic_scholar_api_key
            or not runtime_settings.semantic_scholar_api_url
        ):
            raise RuntimeError("OpenAlex search unavailable")
        if s2_client is None:
            s2_client = SemanticScholarClient(settings=runtime_settings)
        fallback = s2_client.search(query, limit=limit)
        if isinstance(s2_client.last_error, str):
            raise RuntimeError("OpenAlex and Semantic Scholar search unavailable")
        return fallback

    return ResearchWorkflow(
        search=search,
        analyzer=AnalysisClient(settings=runtime_settings),
        settings=runtime_settings,
    )
