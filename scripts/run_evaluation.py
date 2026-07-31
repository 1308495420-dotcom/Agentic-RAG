"""一键评估脚本 —— 接真实检索链路，跑 golden test set 并输出报告。

用法：
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --test-set tests/fixtures/my_golden_test_set.json
    python scripts/run_evaluation.py --provider ragas  # 用 LLM-as-Judge 评估
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── 初始化 ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.settings import load_settings, resolve_path
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner


def build_hybrid_search(settings, collection: str = "default"):
    """搭建完整检索链路（和正常查询一模一样）"""
    embedding = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    dense = create_dense_retriever(settings, embedding, vector_store)

    bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    sparse = create_sparse_retriever(settings, bm25, vector_store)
    sparse.default_collection = collection

    qp = QueryProcessor()
    return create_hybrid_search(settings, qp, dense, sparse)


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--test-set", default="tests/fixtures/my_golden_test_set.json",
        help="Golden test set JSON 文件路径",
    )
    parser.add_argument("--collection", default="default")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--provider", default="custom",
        help="评估器: custom (hit_rate+mrr) | ragas (LLM-as-Judge)",
    )
    parser.add_argument("--output", default=None, help="报告输出 JSON 路径")
    args = parser.parse_args()

    # 1. 加载配置
    settings = load_settings()

    # 2. 创建评估器（强制启用，忽略 settings.yaml 的 enabled: false）
    from dataclasses import replace as dc_replace
    from src.core.settings import EvaluationSettings
    eval_override = EvaluationSettings(
        enabled=True,
        provider=args.provider,
        metrics=["hit_rate", "mrr"] if args.provider == "custom"
        else ["faithfulness", "answer_relevancy", "context_precision"],
    )
    settings = dc_replace(settings, evaluation=eval_override)
    evaluator = EvaluatorFactory.create(settings)
    print(f"评估器: {type(evaluator).__name__}  (provider={args.provider})")

    # 3. 搭建检索链路
    print(f"连接知识库: {args.collection} ...")
    hybrid = build_hybrid_search(settings, args.collection)

    # 4. 跑评估
    runner = EvalRunner(settings=settings, hybrid_search=hybrid, evaluator=evaluator)
    report = runner.run(args.test_set, top_k=args.top_k, collection=args.collection)

    # 5. 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    for i, qr in enumerate(report.query_results):
        print(f"\n[{i+1}] {qr.query}")
        print(f"    命中 chunk: {qr.retrieved_chunk_ids}")
        print(f"    指标: {qr.metrics}")
        print(f"    耗时: {qr.elapsed_ms:.0f}ms")

    print("\n" + "-" * 40)
    print("汇总指标:")
    for name, val in sorted(report.aggregate_metrics.items()):
        bar = "█" * int(val * 20)
        print(f"  {name:20s}  {val:.4f}  {bar}")
    print(f"总耗时: {report.total_elapsed_ms:.0f}ms")

    # 6. 可选：输出 JSON 报告
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n报告已保存: {args.output}")

    return report


if __name__ == "__main__":
    main()
