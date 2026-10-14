"""经过验证的合同由工作流节点和结构化大语言模型工具共享。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictRecord(BaseModel):
    """拒绝未知字段，而不是默默接受模型发明。"""

    model_config = ConfigDict(extra="forbid")


class Finding(StrictRecord):
    """一条带有明确来源和逐字证据的摘录性陈述。"""

    text: NonEmpty
    paper_id: NonEmpty
    evidence_quote: NonEmpty


class ExtractedFindings(StrictRecord):
    """需要的函数调用负载；空列表表示未找到证据。"""

    paper_id: NonEmpty
    methods: list[Finding]
    experiments: list[Finding]
    conclusions: list[Finding]
    reason: Finding


class PaperAnalysis(ExtractedFindings):
    """分析由主机标注，永远不是由模型标注，并说明其局限性。"""

    title: str
    mode: Literal["llm", "extractive"]
    evidence_scope: Literal["abstract", "full_text"]
    warnings: list[str] = Field(default_factory=list)


class Recommendation(StrictRecord):
    """一个推荐，其原因可以追溯到检索到的证据。"""

    paper_id: NonEmpty
    title: str
    reason: NonEmpty
    reason_paper_id: NonEmpty
    evidence_quote: NonEmpty
    mode: Literal["llm", "extractive"]
    evidence_scope: Literal["abstract", "full_text"]


class CitationIssue(StrictRecord):
    """被拒输出的审计记录；不是被接受的科学发现。"""

    paper_id: str
    problem: str


class WorkflowResult(StrictRecord):
    """可序列化的最终输出，明确保留失败和降级模式。"""

    query: str
    recommendations: list[Recommendation]
    analyses: list[PaperAnalysis]
    evidence: dict[str, str]
    unresolved_citations: list[CitationIssue]
    errors: list[str]
    warnings: list[str]
    steps: list[str]


class ComparisonRow(StrictRecord):
    """最终论文比较表中的一行有来源支持的数据。"""

    paper_id: NonEmpty
    source_title: NonEmpty
    methods: str
    experiments: str
    conclusions: str
    evidence_quote: NonEmpty


class ReviewReport(StrictRecord):
    """生成了包含可审计表格和源标签综合的 M5 报告。"""

    query: NonEmpty
    recommendations: list[Recommendation]
    comparison_table: list[ComparisonRow]
    summary: str
    warnings: list[str] = Field(default_factory=list)
