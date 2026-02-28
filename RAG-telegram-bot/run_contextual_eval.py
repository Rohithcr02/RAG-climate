"""
Evaluate the RAG pipeline using contextual_eval_dataset.jsonl.

For each question:
  1. Run hybrid search to retrieve top-k chunks (with timing)
  2. Compute retrieval metrics: recall@k, MRR@k, NDCG@k for k=1,3,5
  3. Generate an answer via Groq LLM (with timing)
  4. Score generation quality: faithfulness, relevance, completeness (LLM-judged)
  5. Score citation quality: validity, coverage, source grounding
  6. Report all results with latency percentile stats

Usage:
    python run_contextual_eval.py                        # full run
    python run_contextual_eval.py --retrieval-only        # skip LLM, just test retrieval
    python run_contextual_eval.py --top-k 10              # retrieve more chunks
    python run_contextual_eval.py --output results.json   # custom output path
"""

import json
import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime
from typing import List, Dict

from retrieve import HybridRetriever
from llm import build_context, generate_answer_ollama
from eval.metrics import citation_validity, citation_coverage, source_grounding
from eval.retrieval_metrics import compute_retrieval_metrics_at_k
from eval.generation_metrics import judge_generation


K_VALUES = [1, 3, 5]


def load_eval_dataset(path: str) -> List[Dict]:
    """Load the JSONL eval dataset."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def percentile_stats(values: List[float]) -> Dict:
    """Compute mean, p50, p95, min, max for a list of values."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }


def run_eval(
    dataset_path: str,
    retrieval_only: bool = False,
    top_k: int = 5,
    output_path: str = None,
):
    """Run the full evaluation."""
    print("=" * 70)
    print("  RAG Contextual Evaluation — Comprehensive Metrics")
    print("=" * 70)

    # Load dataset
    dataset = load_eval_dataset(dataset_path)
    print(f"\nLoaded {len(dataset)} questions from {dataset_path}")

    # Init retriever
    print("Loading retriever...")
    retriever = HybridRetriever()

    # Init LLM
    if not retrieval_only:
        print("Using local Ollama (llama3.2) for answer generation + scoring")

    # Accumulators
    results_log = []
    all_retrieval_metrics = []  # per-question dicts
    all_generation_metrics = []  # per-question dicts
    latency_embed = []
    latency_search = []
    latency_rerank = []
    latency_generate = []

    citation_scores = {"validity": [], "coverage": [], "grounding": []}
    difficulty_buckets = {}

    for i, entry in enumerate(dataset, 1):
        question = entry["question"]
        gold_sources = entry["gold_sources"]
        difficulty = entry.get("metadata", {}).get("difficulty", "Unknown")

        print(f"\n[{i}/{len(dataset)}] ({difficulty}) {question[:80]}...")

        # ---------- Retrieve with timing ----------
        try:
            search_data = retriever.hybrid_search_timed(
                query=question, top_k=top_k, brand_filter=None
            )
            search_results = search_data["results"]
            timings = search_data["timings"]

            latency_embed.append(timings["embed_ms"])
            latency_search.append(timings["search_ms"])
            latency_rerank.append(timings["rerank_ms"])
        except Exception as e:
            print(f"  ✗ Search error: {e}")
            continue

        # ---------- Retrieval metrics at k=1,3,5 ----------
        retrieved_filenames = [r["metadata"]["filename"] for r in search_results]
        ret_metrics = compute_retrieval_metrics_at_k(
            retrieved_filenames, gold_sources, k_values=K_VALUES
        )
        all_retrieval_metrics.append(ret_metrics)

        hit = any(fn in gold_sources for fn in retrieved_filenames)
        hit_marker = "✓" if hit else "✗"
        print(f"  Retrieval: {hit_marker} | recall@1={ret_metrics['recall@1']:.2f} recall@5={ret_metrics['recall@5']:.2f} | mrr@5={ret_metrics['mrr@5']:.2f} | ndcg@5={ret_metrics['ndcg@5']:.2f}")

        result_entry = {
            "question": question,
            "gold_sources": gold_sources,
            "difficulty": difficulty,
            "retrieval_metrics": ret_metrics,
            "retrieved_sources": retrieved_filenames,
            "timings": timings,
        }

        # ---------- Generate answer + metrics ----------
        gen_metrics = None
        if not retrieval_only and search_results:
            try:
                context = build_context(search_results)

                # Generate answer via local Ollama
                gen_start = time.perf_counter()
                answer = generate_answer_ollama(question, context)
                gen_ms = (time.perf_counter() - gen_start) * 1000
                latency_generate.append(round(gen_ms, 2))

                # Citation metrics
                cv = citation_validity(answer, len(search_results))
                cc = citation_coverage(answer)
                sg = source_grounding(answer, search_results)

                citation_scores["validity"].append(cv["score"])
                citation_scores["coverage"].append(cc["score"])
                citation_scores["grounding"].append(sg["score"])

                # LLM-judged generation quality (via local Ollama)
                gen_metrics = judge_generation(
                    question=question,
                    context=context,
                    answer=answer,
                )
                all_generation_metrics.append(gen_metrics)

                print(f"  Generation: faith={gen_metrics['faithfulness']:.2f} | relev={gen_metrics['relevance']:.2f} | compl={gen_metrics['completeness']:.2f}")
                print(f"  Citations:  valid={cv['score']:.2f} | cover={cc['score']:.2f} | ground={sg['score']:.2f}")

                result_entry["answer"] = answer
                result_entry["citation_validity"] = cv["score"]
                result_entry["citation_coverage"] = cc["score"]
                result_entry["source_grounding"] = sg["score"]
                result_entry["generation_metrics"] = gen_metrics
                result_entry["generate_ms"] = round(gen_ms, 2)

                # Brief pause between questions
                time.sleep(0.5)

            except Exception as e:
                print(f"  ✗ LLM error: {e}")
                time.sleep(0.5)

        # Track by difficulty
        if difficulty not in difficulty_buckets:
            difficulty_buckets[difficulty] = {"retrieval": [], "generation": []}
        difficulty_buckets[difficulty]["retrieval"].append(ret_metrics)
        if gen_metrics:
            difficulty_buckets[difficulty]["generation"].append(gen_metrics)

        results_log.append(result_entry)

    # ================================================================
    # AGGREGATE
    # ================================================================
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    def avg_metric(metric_dicts, key):
        vals = [d[key] for d in metric_dicts if key in d]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    # Retrieval metrics aggregate
    retrieval_agg = {}
    for k in K_VALUES:
        for metric_name in ["recall", "mrr", "ndcg"]:
            key = f"{metric_name}@{k}"
            retrieval_agg[key] = avg_metric(all_retrieval_metrics, key)

    # Generation metrics aggregate
    generation_agg = {}
    if all_generation_metrics:
        for key in ["faithfulness", "relevance", "completeness", "overall"]:
            generation_agg[key] = avg_metric(all_generation_metrics, key)

    # Latency summary
    latency_summary = {
        "embed_ms": percentile_stats(latency_embed),
        "search_ms": percentile_stats(latency_search),
        "rerank_ms": percentile_stats(latency_rerank),
        "generate_ms": percentile_stats(latency_generate),
    }

    # By difficulty
    by_difficulty = {}
    for diff in ["Easy", "Medium", "Hard", "Unknown"]:
        if diff not in difficulty_buckets:
            continue
        bucket = difficulty_buckets[diff]
        diff_ret = {}
        for k in K_VALUES:
            for metric_name in ["recall", "mrr", "ndcg"]:
                key = f"{metric_name}@{k}"
                diff_ret[key] = avg_metric(bucket["retrieval"], key)
        diff_gen = {}
        if bucket["generation"]:
            for key in ["faithfulness", "relevance", "completeness", "overall"]:
                diff_gen[key] = avg_metric(bucket["generation"], key)
        by_difficulty[diff] = {
            "count": len(bucket["retrieval"]),
            "retrieval_metrics": diff_ret,
            "generation_metrics": diff_gen,
        }

    # ================================================================
    # BUILD SUMMARY
    # ================================================================
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(dataset),
        "evaluated": len(results_log),
        "top_k": top_k,
        "retrieval_only": retrieval_only,
        "k_values": K_VALUES,
        "retrieval_metrics": retrieval_agg,
        "generation_metrics": generation_agg,
        "citation_metrics": {
            "citation_validity": avg(citation_scores["validity"]),
            "citation_coverage": avg(citation_scores["coverage"]),
            "source_grounding": avg(citation_scores["grounding"]),
        } if citation_scores["validity"] else {},
        "latency_summary": latency_summary,
        "by_difficulty": by_difficulty,
    }

    # ================================================================
    # PRINT SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n  Questions evaluated: {summary['evaluated']}/{summary['total_questions']}")
    print(f"  Top-K: {top_k}")

    print(f"\n  RETRIEVAL METRICS:")
    for k in K_VALUES:
        r = retrieval_agg[f"recall@{k}"]
        m = retrieval_agg[f"mrr@{k}"]
        n = retrieval_agg[f"ndcg@{k}"]
        print(f"    k={k}:  recall={r:.4f}  mrr={m:.4f}  ndcg={n:.4f}")

    if generation_agg:
        print(f"\n  GENERATION METRICS:")
        for key, val in generation_agg.items():
            print(f"    {key:15s} {val:.4f}")

    if citation_scores["validity"]:
        print(f"\n  CITATION METRICS:")
        print(f"    citation_validity: {avg(citation_scores['validity']):.4f}")
        print(f"    citation_coverage: {avg(citation_scores['coverage']):.4f}")
        print(f"    source_grounding:  {avg(citation_scores['grounding']):.4f}")

    print(f"\n  LATENCY SUMMARY:")
    for stage, stats in latency_summary.items():
        print(f"    {stage:15s}  mean={stats['mean']:7.1f}ms  p50={stats['p50']:7.1f}ms  p95={stats['p95']:7.1f}ms  min={stats['min']:7.1f}ms  max={stats['max']:7.1f}ms")

    print(f"\n  BY DIFFICULTY:")
    for diff in ["Easy", "Medium", "Hard", "Unknown"]:
        if diff in by_difficulty:
            b = by_difficulty[diff]
            r5 = b["retrieval_metrics"].get("recall@5", 0)
            m5 = b["retrieval_metrics"].get("mrr@5", 0)
            n5 = b["retrieval_metrics"].get("ndcg@5", 0)
            print(f"    {diff:<10} n={b['count']:>3}  recall@5={r5:.4f}  mrr@5={m5:.4f}  ndcg@5={n5:.4f}", end="")
            if b["generation_metrics"]:
                f_score = b["generation_metrics"].get("faithfulness", 0)
                print(f"  faith={f_score:.4f}", end="")
            print()

    print("\n" + "=" * 70)

    # ================================================================
    # SAVE
    # ================================================================
    output = {
        "summary": summary,
        "per_question": results_log,
    }

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG eval from contextual dataset")
    parser.add_argument(
        "--dataset",
        default="contextual_eval_dataset.jsonl",
        help="Path to the JSONL eval dataset",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval, skip LLM answer generation",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--output", default=None, help="Output JSON path")

    args = parser.parse_args()
    run_eval(
        dataset_path=args.dataset,
        retrieval_only=args.retrieval_only,
        top_k=args.top_k,
        output_path=args.output,
    )
