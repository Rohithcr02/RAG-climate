"""Custom citation metrics (RAGAS doesn't cover these)."""

import re
from typing import List, Dict


def parse_citations(answer_text: str) -> List[int]:
    """Pull all [N] citation numbers from the answer."""
    return [int(m) for m in re.findall(r'\[(\d+)\]', answer_text)]


def citation_validity(answer_text: str, num_sources: int) -> Dict:
    """Check if every cited number is within the valid source range."""
    citations = parse_citations(answer_text)

    if not citations:
        return {
            "score": 0.0,
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": [],
            "note": "No citations found",
        }

    valid = [c for c in citations if 1 <= c <= num_sources]
    invalid = [c for c in citations if c < 1 or c > num_sources]

    return {
        "score": len(valid) / len(citations) if citations else 0.0,
        "total_citations": len(citations),
        "valid_citations": len(valid),
        "invalid_citations": invalid,
    }


def citation_coverage(answer_text: str) -> Dict:
    """What fraction of factual sentences have at least one [N] citation?"""
    sentences = re.split(r'(?<=[.!?])\s+', answer_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return {"score": 0.0, "total_sentences": 0, "cited_sentences": 0}

    # Skip non-factual sentences
    skip_phrases = ["documents don't cover", "not enough information", "i don't know", "no information"]
    factual_sentences = [s for s in sentences if not any(p in s.lower() for p in skip_phrases)]

    if not factual_sentences:
        return {"score": 1.0, "total_sentences": 0, "cited_sentences": 0, "note": "No factual claims"}

    cited = [s for s in factual_sentences if re.search(r'\[\d+\]', s)]

    return {
        "score": len(cited) / len(factual_sentences),
        "total_sentences": len(factual_sentences),
        "cited_sentences": len(cited),
    }


STOPWORDS = {
    'this', 'that', 'with', 'from', 'have', 'been', 'were',
    'their', 'which', 'these', 'those', 'also', 'into', 'such',
    'more', 'than', 'them', 'only', 'some', 'each', 'other',
    'about', 'would', 'could', 'should', 'does', 'very',
}


def source_grounding(answer_text: str, results: List[Dict]) -> Dict:
    """
    For each [N] citation, check if key terms from that sentence
    actually appear in the corresponding source chunk. Fast lexical check.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer_text.strip())
    checks = []

    for sentence in sentences:
        citation_nums = [int(m) for m in re.findall(r'\[(\d+)\]', sentence)]
        if not citation_nums:
            continue

        # Get meaningful words (4+ chars, skip stopwords)
        clean = re.sub(r'\[\d+\]', '', sentence)
        words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', clean)
            if w.lower() not in STOPWORDS
        )

        for num in citation_nums:
            idx = num - 1
            if 0 <= idx < len(results):
                source_text = results[idx]['document'].lower()
                matched = [w for w in words if w in source_text]
                score = len(matched) / len(words) if words else 0.0
                checks.append({
                    "citation": num,
                    "score": round(score, 3),
                    "matched_terms": len(matched),
                    "total_terms": len(words),
                })

    if not checks:
        return {"score": 0.0, "per_citation": [], "note": "No citations to verify"}

    avg_score = sum(c['score'] for c in checks) / len(checks)
    return {
        "score": round(avg_score, 3),
        "per_citation": checks,
    }


def compute_custom_metrics(answer_text: str, results: List[Dict]) -> Dict:
    """Run all custom citation metrics."""
    return {
        "citation_validity": citation_validity(answer_text, len(results)),
        "citation_coverage": citation_coverage(answer_text),
        "source_grounding": source_grounding(answer_text, results),
    }
