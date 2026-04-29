"""Evaluation metrics for comparing context management strategies."""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, asdict
from src.tokens import count_tokens

# Rough cost per million tokens (input/output) as of 2026-04 — update as needed
_COST_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":   {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":  {"input": 3.00,  "output": 15.00},
    "claude-opus-4-7":    {"input": 15.00, "output": 75.00},
    "gpt-4o":             {"input": 2.50,  "output": 10.00},
    "gpt-3.5-turbo":      {"input": 0.50,  "output": 1.50},
}


@dataclass
class RunMetrics:
    strategy: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    estimated_cost_usd: float
    answer: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    rates = _COST_PER_M_TOKENS.get(model, {"input": 3.0, "output": 15.0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


def timed_call(fn, *args, **kwargs) -> tuple[any, float]:
    """Call fn and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def rouge_l(hypothesis: str, reference: str) -> float:
    """Lightweight ROUGE-L F1 without external deps."""
    def lcs(a: list, b: list) -> int:
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[len(a)][len(b)]

    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens or not r_tokens:
        return 0.0
    l = lcs(h_tokens, r_tokens)
    precision = l / len(h_tokens)
    recall = l / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_answer(answer: str, reference: str) -> dict[str, float]:
    """Return a dict of quality scores comparing answer to a reference."""
    return {
        "rouge_l": rouge_l(answer, reference),
        "answer_length_tokens": count_tokens(answer),
        "reference_length_tokens": count_tokens(reference),
    }


def compare_runs(runs: list[RunMetrics], reference_answers: list[str] | None = None) -> list[dict]:
    rows = []
    for i, run in enumerate(runs):
        row = run.to_dict()
        if reference_answers and i < len(reference_answers):
            row.update(score_answer(run.answer, reference_answers[i]))
        rows.append(row)
    return rows


def save_results(runs: list[RunMetrics], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in runs], f, indent=2)


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
