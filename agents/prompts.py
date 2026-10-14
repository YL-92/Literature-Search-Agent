"""Prompt fragments used by the M4 structured-analysis service."""

ANALYSIS_SYSTEM_PROMPT = (
    "You are an extractive paper-analysis agent. Use only the supplied evidence. "
    # 你是一个摘录式论文分析助手。只使用提供的证据。
    "Never invent methods, experiments, conclusions, paper IDs, or citations. "
    # 绝不要编造方法、实验、结论、论文编号或引用。
    "Every finding must include a verbatim evidence_quote."
    #每一项发现都必须包含逐字引用的证据。
)


def recommendation_reason_prompt(title: str, evidence: str) -> str:
    """为一句推荐理由构建一个有限的提示。"""
    return (
        f"Paper title: {title}\nEvidence: {evidence}\n"
        "Write one concise recommendation reason using only this evidence."
    #     仅使用这些证据写一个简明的推荐理由。
    )
