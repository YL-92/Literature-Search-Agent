# 项目章程：自动化论文检索与分析 Agent

项目：自动化论文检索与分析 Agent（文献推荐 + 参考文献来源溯源）

技术栈：Python 3.11，独立可运行的 CLI 应用，不依赖前端。

依赖：openai（兼容接口）、sentence-transformers、faiss-cpu、rank-bm25、
FlagEmbedding（bge-reranker）、arxiv、semanticscholar、requests、
pymupdf（PDF 解析）、networkx（引用图）、langgraph、ragas、pytest、
pydantic。

硬约束：

1. 所有外部 API 调用必须封装在 service 层，可 mock，网络失败要重试并降级；
2. 每个模块写 pytest 单测；public 函数带类型注解和 docstring；
3. 不确定的 API 字段不要臆造，先查官方文档并在代码注释标注来源；
4. 配置全部走 `.env` + pydantic Settings，不硬编码 key；
5. 每一阶段先跑测试再继续，禁止跳过测试直接进入下一阶段。

目录结构:

自动化论文检索与分析 Agent/
├── cli.py                     # CLI 入口：argparse + 缓存 + 工作流 + 综述 + 评估
├── main.py                    # FastAPI 入口：搜索/论文详情/SSE 流式综述/评估 API
├── demo.ps1                   # 一条端到端演示命令
├── pyproject.toml             # 构建/依赖/pytest 配置
├── .env.example               # 环境变量模板（注意：内含一个真实样式 Key，见第六节）
├── README.md / PROJECT_BRIEF.md / PROJECT_GUIDE.md   # 说明 / 章程 / 实施指南
├── config/
│   ├── settings.py            # Pydantic Settings：全部配置项与默认值
│   └── __init__.py            # 导出 Settings / get_settings
├── services/                  # 服务层：所有外部 API/模型调用（可 mock、可降级）
│   ├── paper.py               # 跨数据源统一模型 Paper / RelatedPaper
│   ├── openalex_client.py     # 主数据源：检索/详情/前向后向引用 + JSON缓存 + 限流
│   ├── arxiv_client.py        # arXiv Atom API 适配器（自带一套 Paper dataclass）
│   ├── crossref_client.py     # CrossRef DOI 元数据（可选，按 DOI 补全）
│   ├── semanticscholar_client.py  # Semantic Scholar（默认关闭的备用源）
│   ├── llm_client.py          # LLM：HyDE 假设性摘要生成
│   ├── analysis_client.py     # LLM：Function Calling 结构化抽取 + 宿主校验 + 兜底
│   ├── embedding_client.py    # bge-m3 稠密向量（懒加载）
│   ├── reranker_client.py     # bge-reranker 交叉编码器打分（懒加载）
│   ├── inference.py           # 可选模型推理的统一重试/降级边界
│   ├── review_service.py      # 由已验证证据确定性生成综述与对比表
│   ├── streaming.py           # 综述文本按块切分（伪流式）
│   └── semantic_cache.py      # 查询级持久化缓存（规范化 query+limit，带 TTL）
├── rag/                       # 本地全文 RAG 能力（M1/M2）
│   ├── parser.py              # PyMuPDF 解析 PDF 逐页文本
│   ├── chunker.py             # 标题感知的段落切片（限长 + overlap）
│   ├── hybrid_retriever.py    # BM25 + FAISS 稠密双路召回，RRF 融合，HyDE
│   └── reranker.py            # 用交叉编码器对候选重排并截断 Top-k
├── graph/
│   └── citation_graph.py      # networkx 有向引用图：多跳 BFS、年代/经典最新筛选
├── agents/                    # Agent 编排层（M4）
│   ├── schemas.py             # 全流程 Pydantic 契约（严格模式 extra=forbid）
│   ├── prompts.py             # 分析/推荐理由的 Prompt 片段
│   ├── workflow.py            # LangGraph 四节点工作流（核心）
│   └── runtime.py             # 生产环境接线：OpenAlex + 可选 S2 备用
├── eval/
│   └── ragas_eval.py          # RAGAS 评估 + 确定性离线指标兜底
├── frontend/
│   └── index.html             # 单页前端（ECharts 力导向引用图 + SSE 综述）
└── tests/                     # 28 个测试文件，与各模块一一对应