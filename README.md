#文献检索Agent

一个用于研究论文发现、分析和引用追踪的 Python 3.11 命令行应用程序。

##M0 支架

创建并激活一个虚拟环境，安装项目及其开发依赖项，然后运行：

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python cli.py --help
```

将 `.env.example` 复制为 `.env`，仅在使用需要凭证的服务时提供凭证。配置是通过 Pydantic Settings 读取的。

##M4 工作流

现在，CLI 运行一个可追溯的工作流：

```powershell
python cli.py search "retrieval augmented generation" --limit 5
python cli.py search "citation graph provenance" --limit 5 --json
```

该工作流程使用 OpenAlex 作为主要的学术来源，并辅以 Semantic
Scholar默认情况下被禁用。它执行检索、轻量级相关性
筛选、结构化论文分析和引用验证。每一条被接受的推荐
包含一个 `paper_id`、一个证据引用和一个明确的分析模式。
标准搜索命令还会生成一个比较表，带有来源标签
审查，以及数值 `faithfulness` / `context_recall` 指标；`--review` 和
`--eval` 仍被接受为显式开关。
重复的归一化查询使用本地语义查询缓存，以及 `--stream`
逐步打印评论：

```powershell
python cli.py search "retrieval augmented generation" --limit 5 --stream
python cli.py search "  Retrieval   Augmented Generation " --limit 5 --json
```

当缓存 TTL 尚未过期时，第二个命令会重用第一个查询的缓存工作流结果。JSON 输出包括 `cache_hit`。
当 LLM 不可用时，工作流会使用明显标记的提取型回退，而不是虚构结果。
网络和模型故障会降级为结构化的空结果或部分结果。

OpenAlex 配置可在 `.env` 中找到：

```text
LITSEARCH_SCHOLAR_PROVIDER=openalex
LITSEARCH_OPENALEX_MAILTO=your@email.com
LITSEARCH_OPENALEX_RPS=8
LITSEARCH_SEMANTIC_SCHOLAR_ENABLED=false
```

## 端到端演示

从项目根目录开始：

```powershell
python cli.py search "citation graph provenance" --limit 5 --stream
```

这运行 OpenAlex 检索、过滤、论文分析、引用验证、带来源标签的审查生成，以及两个评估指标。
如果未配置任何 LLM 凭据，输出会明确使用提取式回退；不会添加不支持的结论。
## 回归问题集

该项目包括 120 个中文研究问题，分为 12 类：`data/test_questions.jsonl`。
使用以下方式验证其大小、唯一性和类别覆盖情况：

```powershell
python scripts/validate_question_set.py
```

该问题集基准将真实的 RAGAS 结果与确定性的来源代理区分开。
当没有模型生成的基准输出可用时，它不会报告虚构的 RAGAS 改进。

## Web 前端

轻量级的 FastAPI 前端仅使用 `frontend/index.html` 和浏览器 JavaScript；
没有 npm 或 React 构建步骤：

```powershell
uvicorn main:app --reload
```

打开 `http://127.0.0.1:8000`。
该页面提供论文搜索、SSE 评审流、可点击的来源标记、OpenAlex 引文图加载以及评估指标栏。
