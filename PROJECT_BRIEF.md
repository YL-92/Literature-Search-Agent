# 项目章程：自动化论文检索与分析 Agent

项目：自动化论文检索与分析 Agent（文献推荐 + 参考文献来源溯源）

技术栈：Python 3.11，独立可运行的 CLI 应用，Web 前端。

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
