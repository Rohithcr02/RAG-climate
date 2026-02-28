"""
Run the full eval pipeline: retrieval -> generation -> RAGAS + custom metrics -> save results.

Usage:
    python -m eval.run_eval              # full run
    python -m eval.run_eval --dry-run    # just check everything loads
"""

import json
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve import HybridRetriever
from llm import get_groq_client, build_context, generate_answer
from config import EVAL_TEST_SET_PATH, EVAL_RESULTS_DIR, EVAL_LLM_MODEL
from eval.metrics import compute_custom_metrics

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq


def get_ragas_llm():
    """Wrap Groq in a RAGAS-compatible LLM."""
    chat_model = ChatGroq(
        model=EVAL_LLM_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )
    return LangchainLLMWrapper(chat_model)


def load_test_set():
    """Load test set from JSON. Exits if not found."""
    if not os.path.exists(EVAL_TEST_SET_PATH):
        print(f"✗ Test set not found at: {EVAL_TEST_SET_PATH}")
        print("  Run 'python -m eval.generate_test_set' first.")
        sys.exit(1)

    with open(EVAL_TEST_SET_PATH) as f:
        return json.load(f)


def run_eval(dry_run: bool = False):
    """Run eval. If dry_run=True, just check that everything loads."""
    print("=" * 60)
    print("RAG Evaluation Pipeline")
    print("=" * 60)

    # Load everything
    print("\n1. Loading components...")
    test_set = load_test_set()
    print(f"   ✓ {len(test_set)} test cases loaded")

    retriever = HybridRetriever()
    print("   ✓ Retriever ready")

    groq_client = get_groq_client()
    print("   ✓ Groq client ready")

    ragas_llm = get_ragas_llm()
    print("   ✓ RAGAS LLM ready")

    if dry_run:
        print("\n✓ Dry run complete. All components loaded.")
        return

    # Eval loop
    print(f"\n2. Running evaluation on {len(test_set)} questions...")

    samples = []
    custom_results = []

    for i, test_case in enumerate(test_set, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        print(f"\n   [{i}/{len(test_set)}] {question}")

        # Retrieve
        results = retriever.hybrid_search(query=question, top_k=5)
        retrieved_contexts = [r['document'] for r in results]

        # Generate
        context = build_context(results)
        answer = generate_answer(question, context, groq_client)
        print(f"   Answer: {answer[:100]}...")

        # RAGAS sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=ground_truth,
        )
        samples.append(sample)

        # Custom metrics
        custom = compute_custom_metrics(answer, results)
        custom["question"] = question
        custom["answer"] = answer
        custom_results.append(custom)

        print(f"   Citation validity: {custom['citation_validity']['score']:.2f}")
        print(f"   Citation coverage: {custom['citation_coverage']['score']:.2f}")
        print(f"   Source grounding:  {custom['source_grounding']['score']:.2f}")

    # RAGAS scoring
    print(f"\n3. Computing RAGAS metrics...")
    eval_dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    ragas_results = evaluate(dataset=eval_dataset, metrics=metrics)

    # Save
    print(f"\n4. Saving results...")
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ragas_scores = ragas_results.to_pandas().to_dict(orient="records")
    ragas_path = os.path.join(EVAL_RESULTS_DIR, f"ragas_{timestamp}.json")
    with open(ragas_path, "w") as f:
        json.dump(ragas_scores, f, indent=2, default=str)
    print(f"   ✓ RAGAS: {ragas_path}")

    custom_path = os.path.join(EVAL_RESULTS_DIR, f"custom_{timestamp}.json")
    with open(custom_path, "w") as f:
        json.dump(custom_results, f, indent=2)
    print(f"   ✓ Custom: {custom_path}")

    summary = {
        "timestamp": timestamp,
        "num_questions": len(test_set),
        "ragas_aggregate": {
            k: float(v) for k, v in ragas_results.items()
            if isinstance(v, (int, float))
        },
        "custom_aggregate": {
            "avg_citation_validity": sum(r["citation_validity"]["score"] for r in custom_results) / len(custom_results),
            "avg_citation_coverage": sum(r["citation_coverage"]["score"] for r in custom_results) / len(custom_results),
            "avg_source_grounding": sum(r["source_grounding"]["score"] for r in custom_results) / len(custom_results),
        },
    }

    summary_path = os.path.join(EVAL_RESULTS_DIR, f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ Summary: {summary_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"\nRAGAS:")
    for k, v in summary["ragas_aggregate"].items():
        print(f"  {k:25s} {v:.3f}")
    print(f"\nCitation:")
    for k, v in summary["custom_aggregate"].items():
        print(f"  {k:25s} {v:.3f}")
    print(f"\n{'=' * 60}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG eval")
    parser.add_argument("--dry-run", action="store_true", help="Check everything loads without running eval")
    args = parser.parse_args()

    run_eval(dry_run=args.dry_run)
