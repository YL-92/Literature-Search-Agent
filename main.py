"""用于文献搜索代理的 FastAPI 网络前端。

本地运行方式：

    uvicorn main:app --reload
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.runtime import build_default_workflow
from config import get_settings
from eval.ragas_eval import EvaluationMetrics, evaluate_workflow
from services.openalex_client import OpenAlexClient
from services.review_service import ReviewGenerator
from services.paper import Paper

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(
    title="Literature Search Agent",
    version="0.1.0",
    description="Paper discovery, source-grounded review, and citation tracing.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class SearchRequest(BaseModel):
    """浏览器已接受搜索请求。"""

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


class SummarizeRequest(BaseModel):
    """流媒体摘要请求。"""

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


@lru_cache(maxsize=1)
def get_openalex_client() -> OpenAlexClient:
    """返回进程本地的主要学术来源客户端。"""
    return OpenAlexClient(settings=get_settings())


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    """提供单页前端。"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """在标准浏览器路径下提供一个小型本地 SVG 收藏夹图标。"""
    icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64" rx="16" fill="#101d2f"/>'
        '<circle cx="27" cy="27" r="15" fill="none" stroke="#64d5ff" stroke-width="7"/>'
        '<path d="m38 38 14 14" stroke="#8ce6bd" stroke-width="8" stroke-linecap="round"/>'
        '</svg>'
    )
    return Response(content=icon, media_type="image/svg+xml")


@app.get("/api/health")
def health() -> dict[str, str]:
    """返回轻量级存活响应。"""
    return {"status": "ok", "provider": "openalex"}


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    """搜索 OpenAlex 并在左侧面板返回论文卡片。"""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")
    client = get_openalex_client()
    papers = client.search_papers(query, limit=request.limit)
    payload: dict[str, Any] = {
        "papers": [paper.model_dump() for paper in papers],
    }
    if client.last_error:
        payload["error"] = "scholarly source temporarily unavailable"
    return payload


@app.get("/api/paper/{pid:path}")
def paper_detail(pid: str) -> dict[str, Any]:
    """返回一篇论文以及图表面板的两个引用方向。"""
    paper_id = unquote(pid).strip()
    client = get_openalex_client()
    detail = client.get_paper(paper_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="paper not found")
    references = client.backward_citations(paper_id)
    citing = client.forward_citations(paper_id, limit=30)
    return {
        "detail": detail.model_dump(),
        "references": [_as_json(item) for item in references],
        "citing": [_as_json(item) for item in citing],
    }


@app.post("/api/summarize")
def summarize(request: SummarizeRequest) -> StreamingResponse:
    """将带有来源标签的评论作为 SSE token/citation/done 事件流式传输。"""
    query = request.query.strip()

    def events() -> Iterator[str]:
        """生成浏览器可使用的 SSE 事件。"""
        try:
            result = build_default_workflow().run(query, limit=request.limit)
            report = ReviewGenerator().generate(result)
            citations = [
                {
                    "n": index,
                    "paper_id": item.paper_id,
                    "title": item.title,
                }
                for index, item in enumerate(result.recommendations, start=1)
            ]
            yield _event("citations", {"refs": citations})
            text = _with_inline_citations(report.summary, citations)
            if not text:
                text = "暂时没有可核实的综述内容。"
            for character in text:
                yield _event("token", {"text": character})
            metrics = evaluate_workflow(result)
            yield _event("done", _done_payload(metrics))
        except Exception:
            yield _event("token", {"text": "检索服务暂时不可用，请稍后重试。"})
            yield _event(
                "done",
                {
                    "eval_mode": "deterministic",
                    "metrics": {
                        "faithfulness": 0.0,
                        "context_recall": 0.0,
                        "sample_count": 0,
                    },
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/eval")
def evaluate(query: str = Query(..., min_length=1, max_length=500)) -> dict[str, Any]:
    """运行工作流程并返回对前端友好的评估指标。"""
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")
    result = build_default_workflow().run(query)
    metrics = evaluate_workflow(result)
    return _done_payload(metrics)


def _as_json(value: Any) -> Any:
    """将 Pydantic/数据类提供者记录转换为 JSON 兼容的值。"""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    return value


def _event(name: str, payload: dict[str, Any]) -> str:
    """用 JSON 数据编码一个 SSE 事件。"""
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _with_inline_citations(summary: str, citations: list[dict[str, Any]]) -> str:
    """将源标题标记替换为可点击的数字引用标记。"""
    text = summary
    for item in citations:
        title = str(item["title"])
        text = text.replace(f"[来源: {title}]", f"[{item['n']}]")
    return text


def _done_payload(metrics: EvaluationMetrics) -> dict[str, Any]:
    """将评估器后端名称映射到前端合同。"""
    return {
        "eval_mode": "deterministic"
        if metrics.backend == "deterministic_fallback"
        else metrics.backend,
        "metrics": metrics.as_dict(),
    }
