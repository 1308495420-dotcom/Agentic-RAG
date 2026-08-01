# Agentic RAG — 智能知识检索系统

> 模块化 Agentic RAG 检索框架，支持混合检索 + LLM 推理决策 + 流式对话 + 全链路评估。

## 核心能力

| 模块 | 能力 |
|------|------|
| **离线索引** | 六阶段摄入流水线，MarkItDown PDF/MD 解析，LLM 去噪与元数据增强，DashScope(1024d)+BM25 混合索引 |
| **在线检索** | Dense+Sparse 双路并行召回→RRF(k=60)融合→LLM 精排，查询分词、同义词扩展、问题路由 |
| **Agentic RAG** | QueryPlanner LLM 拆分复杂查询，LLM 自主判断置信度并建议新搜索词，多轮迭代检索 |
| **检索增强** | Small-to-Big 上下文扩展、低置信度二次检索、逐句引用核查 |
| **评估体系** | Ragas + CustomEvaluator（Hit Rate/MRR），Golden Test Set 量化对比 |
| **Dashboard** | Streamlit 七页面管理看板（Chat 流式对话、数据浏览、摄入管理、全链路追踪、评估面板） |

## 评估指标（15 题 Golden Test Set 实测）

| 指标 | 数值 |
|------|------|
| Hit Rate@5 | 0.87 (13/15) |
| MRR | 0.87 |
| Faithfulness | 0.99 (5 题 Ragas) |
| Context Precision | 0.98 (DashScope Hybrid) |
| 摄入成功率 | 102 文档 / 170 chunks / 零失败 |

## 快速开始

```bash
pip install -e .
cp .env.example .env  # 填入 DEEPSEEK_API_KEY + EMBEDDING_API_KEY

# 摄入文档
python -c "from src.ingestion.pipeline import IngestionPipeline; from src.core.settings import load_settings; IngestionPipeline(load_settings()).run(file_path='your-file.pdf')"

# 查询
python scripts/query.py --query "你的问题"

# Agent 模式
python -c "from src.core.agent.rag_agent import RagAgent; from src.core.agent.toolkit import AgentToolkit; ..."

# Dashboard
streamlit run src/observability/dashboard/app.py

# 评估
python scripts/run_evaluation.py --test-set tests/fixtures/full_eval_15.json
```

## 架构

```
用户 Query → QueryProcessor（分词/路由）
              ╱              ╲
         Dense 检索        Sparse 检索
         DashScope 1024d   jieba+BM25
              ╲              ╱
            RRF(k=60) 融合 → LLM 精排
                    ↓
            Agent 推理层（QueryPlanner + LLM Judge）
                    ↓
            AnswerGenerator → CitationVerifier
```

## 配置

`config/settings.yaml`：

```yaml
llm:       {provider: "deepseek", model: "deepseek-v4-pro"}
embedding: {provider: "dashscope", model: "text-embedding-v3", dimensions: 1024}
rerank:    {enabled: true, provider: "llm"}
retrieval: {dense_top_k: 20, sparse_top_k: 20, fusion_top_k: 10, rrf_k: 60}
```

## 技术栈

`Python` · `Agentic RAG` · `Hybrid Search` · `BM25` · `ChromaDB` · `RRF` · `LLM Rerank` · `DashScope` · `DeepSeek` · `Ragas` · `Streamlit` · `TDD` · `Factory Pattern`

## License

MIT
