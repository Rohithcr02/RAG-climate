"""
Retrieval metrics computed at multiple k cutoffs: Recall@k, MRR@k, NDCG@k.
"""

import math
from typing import List, Dict


def recall_at_k(retrieved_filenames: List[str], gold_sources: List[str], k: int) -> float:
    """Fraction of gold sources found in the top-k retrieved documents."""
    top_k = retrieved_filenames[:k]
    if not gold_sources:
        return 0.0
    found = sum(1 for gs in gold_sources if gs in top_k)
    return found / len(gold_sources)


def mrr_at_k(retrieved_filenames: List[str], gold_sources: List[str], k: int) -> float:
    """Reciprocal rank of first gold-source hit within top-k."""
    for i, fn in enumerate(retrieved_filenames[:k], 1):
        if fn in gold_sources:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_filenames: List[str], gold_sources: List[str], k: int) -> float:
    """
    NDCG@k — binary relevance (1 if from gold source, 0 otherwise).
    Each gold source is only counted as relevant ONCE (first occurrence).
    DCG  = sum( rel_i / log2(i+1) ) for i=1..k
    IDCG = sum( 1 / log2(i+1) )     for i=1..min(k, num_gold)
    """
    top_k = retrieved_filenames[:k]

    # DCG — deduplicate: each gold source only counted once
    dcg = 0.0
    seen_gold = set()
    for i, fn in enumerate(top_k, 1):
        if fn in gold_sources and fn not in seen_gold:
            dcg += 1.0 / math.log2(i + 1)
            seen_gold.add(fn)

    # IDCG — ideal: all unique gold sources ranked first
    num_relevant = min(len(gold_sources), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics_at_k(
    retrieved_filenames: List[str],
    gold_sources: List[str],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """Compute recall, MRR, NDCG at each k. Returns flat dict like {'recall@1': ..., 'mrr@3': ...}."""
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = round(recall_at_k(retrieved_filenames, gold_sources, k), 4)
        metrics[f"mrr@{k}"] = round(mrr_at_k(retrieved_filenames, gold_sources, k), 4)
        metrics[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_filenames, gold_sources, k), 4)
    return metrics
