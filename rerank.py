"""Cross-encoder reranker for scoring query-document pairs."""

import os
from typing import List, Dict
from sentence_transformers import CrossEncoder

# ms-marco-MiniLM-L-6-v2: 80MB, fast on CPU, strong relevance scoring
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker() -> CrossEncoder:
    """Load and return the cross-encoder model."""
    return CrossEncoder(RERANKER_MODEL, token=os.environ.get("HF_TOKEN"))


def rerank(query: str, results: List[Dict], model: CrossEncoder) -> List[Dict]:
    """
    Score each result against the query using the cross-encoder.
    Returns results sorted by relevance score, highest first.
    """
    if not results:
        return results

    pairs = [(query, result["document"]) for result in results]
    scores = model.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)
