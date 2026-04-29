"""
Exp 01 — Strategy Comparison: Needle in a Haystack

Usage:
    python experiments/exp01_strategy_comparison/run.py [--strategies rag map_reduce hier sliding]
"""

from __future__ import annotations
import sys
import json
import time
import argparse
import csv
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp01_strategy_comparison.config import (
    CORPUS_PATH, NEEDLES_PATH, RESULTS_DIR, ANSWER_MODEL,
)
from experiments.exp01_strategy_comparison.strategies import (
    run_rag, run_map_reduce, run_hierarchical, run_sliding_window,
)
from src.evaluate import RunMetrics, estimate_cost, rouge_l, save_results

STRATEGY_FNS = {
    "rag": run_rag,
    "map_reduce": run_map_reduce,
    "hier": run_hierarchical,
    "sliding": run_sliding_window,
}


def run_experiment(strategies: list[str], needles: list[dict], corpus: str) -> list[RunMetrics]:
    all_runs: list[RunMetrics] = []

    for strategy in strategies:
        fn = STRATEGY_FNS[strategy]
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()}  ({len(needles)} questions)")
        print(f"{'='*60}")

        for needle in needles:
            question = needle["question"]
            reference = needle["answer"]
            print(f"  Q: {question}")

            t0 = time.perf_counter()
            try:
                answer, input_tokens, output_tokens = fn(corpus, question)
            except Exception as e:
                print(f"    ERROR: {e}")
                answer, input_tokens, output_tokens = f"ERROR: {e}", 0, 0
            latency = time.perf_counter() - t0

            score = rouge_l(answer, reference)
            cost = estimate_cost(input_tokens, output_tokens, ANSWER_MODEL)

            print(f"    A: {answer.strip()[:120]}")
            print(f"    ref: {reference}  |  ROUGE-L: {score:.3f}  |  {latency:.1f}s  |  ${cost:.5f}")

            all_runs.append(RunMetrics(
                strategy=strategy,
                model=ANSWER_MODEL,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=round(latency, 3),
                estimated_cost_usd=round(cost, 6),
                answer=answer.strip(),
                notes=f"needle_id={needle['id']} rouge_l={score:.4f} reference={reference}",
            ))

    return all_runs


def summarise(runs: list[RunMetrics]) -> None:
    from collections import defaultdict

    grouped: dict[str, list[RunMetrics]] = defaultdict(list)
    for r in runs:
        grouped[r.strategy].append(r)

    print(f"\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    header = f"{'Strategy':<15} {'ROUGE-L':>8} {'Latency(s)':>12} {'Cost($)':>10} {'Questions':>10}"
    print(header)
    print("-" * 60)

    rows = []
    for strategy, group in grouped.items():
        rouge_scores = [float(r.notes.split("rouge_l=")[1].split()[0]) for r in group]
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        avg_latency = sum(r.latency_s for r in group) / len(group)
        total_cost = sum(r.estimated_cost_usd for r in group)
        print(f"{strategy:<15} {avg_rouge:>8.3f} {avg_latency:>12.1f} {total_cost:>10.5f} {len(group):>10}")
        rows.append({
            "strategy": strategy,
            "avg_rouge_l": round(avg_rouge, 4),
            "avg_latency_s": round(avg_latency, 2),
            "total_cost_usd": round(total_cost, 6),
            "n_questions": len(group),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies", nargs="+",
        choices=list(STRATEGY_FNS.keys()),
        default=list(STRATEGY_FNS.keys()),
        help="Which strategies to run (default: all)",
    )
    parser.add_argument(
        "--needles", type=int, default=None,
        help="Limit to first N needles (default: all)",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        print("Corpus not found. Run: python data/samples/generate_corpus.py")
        sys.exit(1)

    corpus = CORPUS_PATH.read_text()
    needles = json.loads(NEEDLES_PATH.read_text())
    if args.needles:
        needles = needles[:args.needles]

    from src.tokens import count_tokens
    print(f"Corpus: {count_tokens(corpus):,} tokens")
    print(f"Needles: {len(needles)}")
    print(f"Strategies: {args.strategies}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    runs = run_experiment(args.strategies, needles, corpus)

    results_path = RESULTS_DIR / "results.json"
    save_results(runs, str(results_path))
    print(f"\nResults saved to {results_path}")

    summary_rows = summarise(runs)

    summary_path = RESULTS_DIR / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
