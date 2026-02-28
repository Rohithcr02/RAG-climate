"""
Print or export eval results.

Usage:
    python -m eval.report              # print latest
    python -m eval.report --markdown   # also save markdown file
"""

import json
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EVAL_RESULTS_DIR


def find_latest_results():
    """Find the most recent summary file."""
    if not os.path.exists(EVAL_RESULTS_DIR):
        print("✗ No results found. Run 'python -m eval.run_eval' first.")
        sys.exit(1)

    summaries = sorted(glob.glob(os.path.join(EVAL_RESULTS_DIR, "summary_*.json")))
    if not summaries:
        print("✗ No results found. Run 'python -m eval.run_eval' first.")
        sys.exit(1)

    return summaries[-1]


def load_results(summary_path: str):
    """Load summary + per-question details."""
    with open(summary_path) as f:
        summary = json.load(f)

    timestamp = summary["timestamp"]
    results_dir = os.path.dirname(summary_path)

    ragas_details = []
    ragas_path = os.path.join(results_dir, f"ragas_{timestamp}.json")
    if os.path.exists(ragas_path):
        with open(ragas_path) as f:
            ragas_details = json.load(f)

    custom_details = []
    custom_path = os.path.join(results_dir, f"custom_{timestamp}.json")
    if os.path.exists(custom_path):
        with open(custom_path) as f:
            custom_details = json.load(f)

    return summary, ragas_details, custom_details


def print_report(summary, ragas_details, custom_details):
    """Print results to console."""
    print("\n" + "=" * 70)
    print(f"  RAG EVAL REPORT  |  {summary['timestamp']}")
    print(f"  {summary['num_questions']} questions")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  SCORES                                                 │")
    print("├─────────────────────────────────────────────────────────┤")

    print("│                                                         │")
    print("│  RAGAS:                                                  │")
    for k, v in summary.get("ragas_aggregate", {}).items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"│    {k:28s} {bar} {v:.3f}  │")

    print("│                                                         │")
    print("│  Citation:                                               │")
    for k, v in summary.get("custom_aggregate", {}).items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"│    {k:28s} {bar} {v:.3f}  │")

    print("│                                                         │")
    print("└─────────────────────────────────────────────────────────┘")

    if custom_details:
        print("\n── Per Question ─────────────────────────────────────────")
        for i, detail in enumerate(custom_details, 1):
            q = detail.get("question", "N/A")
            cv = detail["citation_validity"]["score"]
            cc = detail["citation_coverage"]["score"]
            sg = detail["source_grounding"]["score"]
            print(f"\n  Q{i}: {q[:60]}{'...' if len(q) > 60 else ''}")
            print(f"      Validity: {cv:.2f}  Coverage: {cc:.2f}  Grounding: {sg:.2f}")

    print()


def generate_markdown_report(summary, ragas_details, custom_details) -> str:
    """Build a markdown string."""
    lines = [
        f"# RAG Eval Report",
        f"",
        f"**Date**: {summary['timestamp']}",
        f"**Questions**: {summary['num_questions']}",
        f"",
        f"## RAGAS",
        f"",
        f"| Metric | Score |",
        f"|--------|-------|",
    ]

    for k, v in summary.get("ragas_aggregate", {}).items():
        lines.append(f"| {k} | {v:.3f} |")

    lines.extend([
        f"",
        f"## Citation Metrics",
        f"",
        f"| Metric | Score |",
        f"|--------|-------|",
    ])

    for k, v in summary.get("custom_aggregate", {}).items():
        lines.append(f"| {k} | {v:.3f} |")

    if custom_details:
        lines.extend([
            f"",
            f"## Per Question",
            f"",
            f"| # | Question | Validity | Coverage | Grounding |",
            f"|---|----------|----------|----------|-----------|",
        ])

        for i, detail in enumerate(custom_details, 1):
            q = detail.get("question", "N/A")
            cv = detail["citation_validity"]["score"]
            cc = detail["citation_coverage"]["score"]
            sg = detail["source_grounding"]["score"]
            lines.append(f"| {i} | {q[:50]}... | {cv:.2f} | {cc:.2f} | {sg:.2f} |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Show eval results")
    parser.add_argument("--markdown", action="store_true", help="Save a markdown report")
    args = parser.parse_args()

    summary_path = find_latest_results()
    summary, ragas_details, custom_details = load_results(summary_path)

    print_report(summary, ragas_details, custom_details)

    if args.markdown:
        md = generate_markdown_report(summary, ragas_details, custom_details)
        report_path = os.path.join(EVAL_RESULTS_DIR, f"report_{summary['timestamp']}.md")
        with open(report_path, "w") as f:
            f.write(md)
        print(f"✓ Saved: {report_path}")


if __name__ == "__main__":
    main()
