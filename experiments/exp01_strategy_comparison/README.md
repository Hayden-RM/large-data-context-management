# Exp 01 — Strategy Comparison: Needle in a Haystack

## Hypothesis

Retrieval-augmented generation (RAG) will outperform map-reduce and hierarchical
summarisation on factual look-up tasks, but map-reduce will be more robust when
multiple facts must be synthesised across the corpus.

## Method

1. Generate a synthetic corpus of ~50,000 tokens containing 20 "needles" — discrete
   factual claims with known ground truth (e.g. "The boiling point of X is Y").
2. Construct 20 questions, each targeting a single needle.
3. Run four strategies against every question:
   - **RAG** — embed chunks, retrieve top-k, answer
   - **Map-reduce** — answer per chunk, reduce
   - **Hierarchical summarisation** — collapse corpus to target size, answer
   - **Sliding window QA** — answer per window, collect per-window results
4. Score each answer against the reference with ROUGE-L.
5. Record token cost and latency per strategy.

## Running

```bash
# Generate the corpus (one-time)
python data/samples/generate_corpus.py

# Run the experiment
python experiments/exp01_strategy_comparison/run.py

# Results land in results/exp01/
```

## Expected Output

`results/exp01/results.json` — one record per (strategy, question).
`results/exp01/summary.csv` — aggregated mean ROUGE-L, cost, latency per strategy.
