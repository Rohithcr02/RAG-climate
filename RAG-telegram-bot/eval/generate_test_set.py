"""Generate a golden test set for eval. Runs retrieval + LLM to write ground truth answers."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve import HybridRetriever
from llm import get_groq_client, build_context
from config import EVAL_TEST_SET_PATH, EXAMPLE_QUERIES, EVAL_LLM_MODEL

# Mix of app example queries + additional coverage questions
EVAL_QUESTIONS = EXAMPLE_QUERIES + [
    "What are the environmental impacts of HFC refrigerants?",
    "How does the Kigali Amendment relate to the Montreal Protocol?",
    "What is the role of energy efficiency in cooling systems?",
    "What are district cooling systems and their benefits?",
    "How does urbanization impact cooling demand?",
]

GROUND_TRUTH_PROMPT = """Based ONLY on the sources below, write a thorough reference answer to the question.
Include all relevant facts, dates, and specifics. Only include what the sources say.

Sources:
{context}

Question: {question}

Reference Answer:"""


def generate_test_set():
    """Generate test set with LLM-written ground truth answers."""
    print("=" * 60)
    print("Generating Eval Test Set")
    print("=" * 60)

    retriever = HybridRetriever()
    groq_client = get_groq_client()

    test_set = []

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] {question}")

        results = retriever.hybrid_search(query=question, top_k=5)

        if not results:
            print("  ⚠ No results found, skipping")
            continue

        context = build_context(results)

        print("  Generating ground truth answer...")
        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Write detailed, factual answers based only on the provided sources."},
                    {"role": "user", "content": GROUND_TRUTH_PROMPT.format(context=context, question=question)},
                ],
                model=EVAL_LLM_MODEL,
                temperature=0.1,
                max_tokens=1000,
            )
            ground_truth = response.choices[0].message.content
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

        contexts = [result['document'] for result in results]

        test_set.append({
            "question": question,
            "ground_truth": ground_truth,
            "contexts": contexts,
            "source_metadata": [
                {
                    "filename": r['metadata']['filename'],
                    "page_number": r['metadata']['page_number'],
                }
                for r in results
            ],
        })

        print(f"  ✓ Ground truth generated ({len(ground_truth)} chars)")

    os.makedirs(os.path.dirname(EVAL_TEST_SET_PATH), exist_ok=True)
    with open(EVAL_TEST_SET_PATH, "w") as f:
        json.dump(test_set, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(test_set)} test cases to {EVAL_TEST_SET_PATH}")
    print(f"{'=' * 60}")

    return test_set


if __name__ == "__main__":
    generate_test_set()
