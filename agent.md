# Agentic RAG 改造基线

## 加 Agent 前评估数据（9 题测试集）

| 指标 | 纯 Dense | Hybrid | Hybrid+Rerank |
|------|---------|--------|---------------|
| Context Precision | 0.13 | 0.62 | 0.81 |
| Faithfulness | 0.55 | 0.84 | 0.998 |
| Recall@5 | 7.2% | 21.1% | — |
| Top-1 命中率 | 0% | 56% | — |
| MRR | 0.83 | 1.0 | 1.0 |

引用核查（中文）: 100%

---

## Agentic RAG 改造方案

### 改造思路

现有 RAG 是"一次检索 → 拼答案"的线性流程。Agentic RAG 改成循环：

```
用户 Query
    │
    ▼
Agent 分析 → 需要查什么？几个子问题？
    │
    ├─→ 子问题1 → 检索 → 拿结果
    ├─→ 子问题2 → 检索 → 拿结果
    └─→ 子问题3 → 检索 → 拿结果
    │
    ▼
信息够了？ → 不够 → 换策略再搜一轮
    │ 够了
    ▼
拼装所有结果 → LLM 生成最终答案 → 引用核查
```

### 新增模块（3 个文件，~240 行）

**1. `src/core/agent/rag_agent.py` (~120 行)**
- Agent 主循环：Think → Act → Observe → 循环
- 判断是否需要继续检索（置信度阈值）
- 多轮检索结果合并去重
- 最大轮次限制防止死循环

**2. `src/core/agent/toolkit.py` (~40 行)**
- 把现有能力封装为 Agent 可调用的工具
- `search(query)` → 调 HybridSearch
- `generate(chunks, query)` → 调 AnswerGenerator
- `verify(answer, chunks)` → 调 CitationVerifier

**3. `src/core/agent/query_planner.py` (~60 行)**
- 复杂 query 拆分成子问题
- 调 LLM：把 "FinFET 和 GAAFET 对比" 拆成 ["搜 FinFET 优缺点", "搜 GAAFET 优缺点"]
- 返回子问题列表

### 改动已有模块（2 个文件，~20 行）

**4. `src/core/response/answer_generator.py` (+15 行)**
- `generate_multi()` 方法：支持接收多轮检索结果拼装

**5. `src/core/query_engine/hybrid_search.py` (+10 行)**
- `search()` 返回结果附上置信度标记，Agent 用来判断是否继续搜

### 不改动的模块

- Ingestion Pipeline（离线索引不变）
- QueryProcessor（查询预处理不变）
- Reranker（精排不变）
- 评估体系（RAGAS + CustomEvaluator 原样可用）
- Dashboard（可观测性不变）

### 评估对比计划

改造完成后，同样 9 道题重跑：
- 新增 5 道复杂多跳题（跨文档对比、多步推理）
- 对比 Agent 多轮检索 vs 单轮检索的 Context Precision 和 Faithfulness

### 面试话术

"在 RAG 系统上封装 Agent 层，实现 Think-Act-Observe 循环——Agent 自动拆分复杂查询为子问题、多轮迭代检索、置信度不足时主动扩召回。相较单轮检索，复杂多跳问题的回答完整性显著提升。"

---

## 当前状态（2026-07-26）

### ✅ 已实现
- QueryPlanner：LLM 拆分复杂查询为子问题
- AgentToolkit：封装 search / generate / verify 为工具
- RagAgent：Think-Act-Observe 主循环（规则驱动）
- Chat 对话页：Streamlit 流式输出，支持检索来源展示与引用核查

### ❌ 暂未实现（后续评估后决定）
- **LLM 推理层**：当前重搜判断用硬规则（分数 < 阈值），未接入 LLM 推理
  - 后续可加：把检索结果 + 验证报告喂给 LLM，让它判断"信息够不够，不够该怎么搜"
  - 优点：多步推理更灵活；代价：额外 LLM 调用 + 延迟

### ✅ 保留的模块
- MCP Server：已恢复，作为独立亮点保留，AGent 改造未依赖 MCP
