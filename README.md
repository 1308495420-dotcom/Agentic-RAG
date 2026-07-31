# Modular RAG — 可插拔知识检索系统

> 一个模块化、可观测的 RAG 检索系统，支持 Hybrid Search + LLM Rerank + MCP 协议集成 + 全链路评估。

---

## 核心能力

| 模块 | 能力 | 说明 |
|------|------|------|
| **离线索引** | PDF/MD 摄入 → 分块 → LLM 精炼 → 混合索引 | 六阶段流水线，MarkItDown PDF 解析，自动去噪与元数据增强 |
| **在线检索** | Dense + Sparse 双路召回 → RRF 融合 → LLM 精排 | 集成问题路由、同义词扩展、低置信度二次检索 |
| **上下文拼装** | 检索后拼装 Chunk → LLM 生成带引用标注的答案 | Small-to-Big 上下文扩展，逐句引用核查 |
| **评估体系** | Ragas + CustomEvaluator（Hit Rate / MRR） | Golden Test Set 回归测试，量化对比策略效果 |
| **MCP 协议** | 标准 MCP Server，暴露 3 个 Tool | Claude Desktop、Cursor 等 AI 客户端即插即用 |
| **可观测性** | 全链路 TraceContext 10 阶段追踪 | Streamlit Dashboard 六页面可视化管理 |

## 架构

```
用户 Query
    │
    ▼
QueryProcessor ─── 分词 / Filter 解析 / 同义词扩展 / 问题路由
    │
    ╱              ╲
Dense 检索           Sparse 检索
DashScope 1024d      jieba + BM25
ChromaDB HNSW        JSON 倒排索引
    ╲              ╱
     RRF(k=60) 融合
          │
     LLM Rerank 精排
          │
  AnswerGenerator ─── 上下文拼装 → LLM 生成答案
          │
  CitationVerifier ─── 逐句核查引用依据
```

## 评估指标（基于 9 题 Golden Test Set 实测）

| 指标 | 纯 Dense | Hybrid | Hybrid+Rerank |
|------|---------|--------|---------------|
| Context Precision | 0.13 | 0.62 | 0.81 |
| Faithfulness | 0.55 | 0.84 | 0.998 |
| Recall@5 | 7.2% | 21.1% | — |
| Top-1 命中率 | 0% | 56% | — |

## 快速开始

```bash
pip install -e .

# 摄入文档
python -c "
from src.ingestion.pipeline import IngestionPipeline
from src.core.settings import load_settings
p = IngestionPipeline(load_settings())
p.run(file_path='data/documents/your-file.md')
"

# 查询
python scripts/query.py --query "你的问题" --collection default

# 评估
python scripts/run_evaluation.py --test-set tests/fixtures/semiconductor_golden.json

# Dashboard
streamlit run src/observability/dashboard/app.py
```

## 配置

`config/settings.yaml`：

```yaml
llm:       {provider: "deepseek", model: "deepseek-v4-pro"}
embedding: {provider: "dashscope", model: "text-embedding-v3"}  # 阿里云，OpenAI 兼容
rerank:    {enabled: true, provider: "llm"}
retrieval: {dense_top_k: 20, sparse_top_k: 20, fusion_top_k: 10, rrf_k: 60}
```

环境变量（`.env`）：

```
DEEPSEEK_API_KEY=sk-xxx
EMBEDDING_API_KEY=sk-xxx
```

## 技术栈

`Python` · `ChromaDB` · `BM25` · `RAGAS` · `RRF` · `LLM Rerank` · `MCP` · `DashScope` · `DeepSeek` · `jieba` · `Streamlit` · `MarkItDown` · `langchain-text-splitters` · `TDD` · `Factory Pattern`

## 目录结构

```
src/
├── core/query_engine/   # Hybrid Search, Reranker, Query Processor, Query Cache
├── core/response/       # Answer Generator, Citation Verifier
├── core/trace/          # TraceContext, TraceCollector
├── ingestion/           # Pipeline, Chunk Refiner, Metadata Enricher
├── libs/                # LLM, Embedding, VectorStore, Reranker, Splitter, Evaluator
├── mcp_server/          # MCP Protocol Server + 3 Tools
└── observability/       # Dashboard, Evaluation
```

## License

MIT
