"""结构化、保留来源的论文分析服务。"""

from __future__ import annotations

import json
import logging
from typing import Any

from config import Settings, get_settings
from services.inference import attempt_inference
from services.paper import Paper
from agents.schemas import ExtractedFindings, Finding, PaperAnalysis

LOGGER = logging.getLogger(__name__)

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_paper_findings",
        "description": "Extract methods, experiments and conclusions from supplied evidence.",
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "paper_id": {"type": "string"},
                "methods": {"type": "array", "items": {"$ref": "#/$defs/finding"}},
                "experiments": {"type": "array", "items": {"$ref": "#/$defs/finding"}},
                "conclusions": {"type": "array", "items": {"$ref": "#/$defs/finding"}},
                "reason": {"$ref": "#/$defs/finding"},
            },
            "required": ["paper_id", "methods", "experiments", "conclusions", "reason"],
            "$defs": {
                "finding": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "paper_id": {"type": "string"},
                        "evidence_quote": {"type": "string"},
                    },
                    "required": ["text", "paper_id", "evidence_quote"],
                }
            },
        },
    },
}


class AnalysisClient:
    """从元数据或兼容 OpenAI 的大语言模型中提取结构化发现。"""

    def __init__(self, *, settings: Settings | None = None, client: Any = None) -> None:
        """接受与 SDK 兼容的客户端以进行确定性测试。"""
        self.settings = settings or get_settings()
        self._client = client

    def analyze(self, paper: Paper, full_text: str | None = None) -> PaperAnalysis | None:
        """分析一篇论文，并强制每个返回的发现都使用其论文编号。"""
        evidence_scope = "full_text" if full_text and full_text.strip() else "abstract"
        evidence = (full_text or paper.abstract or "").strip()
        if not evidence:
            return self._fallback(paper, evidence_scope, "No abstract or full text available")
        if not self.settings.openai_model or (
            self._client is None and not self.settings.openai_api_key
        ):
            return self._fallback(paper, evidence_scope, "LLM is not configured")

        def operation() -> PaperAnalysis:
            client = self._ensure_client()
            response = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": (
                        "You are an extractive paper-analysis agent. Use only the supplied "
                        "evidence. Never invent methods, experiments, conclusions, paper IDs, "
                        "or citations. Every finding must include a verbatim evidence_quote. "
                        "Call the provided function exactly once."
                    )},
                    {"role": "user", "content": (
                        f"paper_id: {paper.paper_id}\n"
                        f"title: {paper.title or ''}\n"
                        f"evidence:\n{evidence}"
                    )},
                ],
                tools=[EXTRACT_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_paper_findings"},
                },
            )
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            payload = ExtractedFindings.model_validate(json.loads(arguments))
            return self._host_validate(payload, paper, evidence, evidence_scope)

        result = attempt_inference(operation, self.settings, "paper analysis")
        return result or self._fallback(
            paper, evidence_scope, "Structured analysis failed; using extractive fallback"
        )

    def _ensure_client(self) -> Any:
        """懒惰地创建兼容 OpenAI 的 SDK，并且不进行 SDK 重试。"""
        if self._client is None:
            from openai import OpenAI

            options: dict[str, Any] = {
                "api_key": self.settings.openai_api_key,
                "timeout": self.settings.request_timeout,
                "max_retries": 0,
            }
            if self.settings.openai_base_url:
                options["base_url"] = str(self.settings.openai_base_url)
            self._client = OpenAI(**options)
        return self._client

    @staticmethod
    def _host_validate(
        payload: ExtractedFindings,
        paper: Paper,
        evidence: str,
        evidence_scope: str,
    ) -> PaperAnalysis:
        """覆盖模型提供的ID，并拒绝未在证据中出现的报价。"""
        findings = [*payload.methods, *payload.experiments, *payload.conclusions, payload.reason]
        if any(f.evidence_quote not in evidence for f in findings):
            raise ValueError("Model returned evidence not present in supplied text")
        # 主机拥有来源。模型无法将声明重定向到其他地方
        #论文，即使它发出一个语法上有效的 paper_id。
        normalized = [
            Finding(
                text=finding.text,
                paper_id=paper.paper_id,
                evidence_quote=finding.evidence_quote,
            )
            for finding in findings
        ]
        method_count = len(payload.methods)
        experiment_count = len(payload.experiments)
        methods = normalized[:method_count]
        experiments = normalized[method_count:method_count + experiment_count]
        conclusions = normalized[method_count + experiment_count:-1]
        return PaperAnalysis(
            paper_id=paper.paper_id,
            methods=methods,
            experiments=experiments,
            conclusions=conclusions,
            reason=normalized[-1],
            title=paper.title or paper.paper_id,
            mode="llm",
            evidence_scope=evidence_scope,
        )

    @staticmethod
    def _fallback(
        paper: Paper, evidence_scope: str, warning: str
    ) -> PaperAnalysis:
        """创建一个标明清楚的仅包含元数据的分析，不要编造事实。"""
        text = (paper.abstract or "").strip() or "No evidence was available."
        reason_text = _first_sentence(text)
        conclusion = Finding(
            text=text,
            paper_id=paper.paper_id,
            evidence_quote=text,
        )
        reason = Finding(
            text=reason_text,
            paper_id=paper.paper_id,
            evidence_quote=reason_text,
        )
        return PaperAnalysis(
            paper_id=paper.paper_id,
            title=paper.title or paper.paper_id,
            methods=[],
            experiments=[],
            conclusions=[conclusion],
            reason=reason,
            mode="extractive",
            evidence_scope=evidence_scope,
            warnings=[warning],
        )


def _first_sentence(text: str) -> str:
    """保持推荐理由简明，不要编造其他说法。"""
    for marker in (". ", "。", "! ", "！", "? ", "？"):
        position = text.find(marker)
        if position >= 0:
            return text[: position + 1].strip()
    return text
