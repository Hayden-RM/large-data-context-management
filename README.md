# Large Data Context Management

Research into strategies for handling extremely large amounts of data within LLM context limits.

## Research Questions

- What are the failure modes when data exceeds context windows?
- Which chunking, summarisation, and retrieval strategies best preserve information fidelity?
- How do different approaches trade off latency, cost, and accuracy?
- At what data scales does each strategy break down?

## Repo Structure

```
large-data-context-management/
├── experiments/        # Standalone experiment scripts, one folder per experiment
├── src/                # Shared utilities (chunking, retrieval, evaluation helpers)
├── results/            # Experiment outputs, metrics, and comparisons
├── notebooks/          # Exploratory analysis and visualisations
├── data/
│   └── samples/        # Small sample datasets for reproducible experiments
└── docs/               # Literature notes, design decisions, and findings
```

## Approaches Under Investigation

| Strategy | Description |
|---|---|
| Chunking + retrieval (RAG) | Split corpus, embed, retrieve relevant chunks at query time |
| Hierarchical summarisation | Recursively summarise sections, query the summary tree |
| Map-reduce | Process chunks independently, reduce results |
| Sliding window | Process with overlapping windows to preserve boundary context |
| Structured extraction | Extract only structured fields before passing to model |

## Getting Started

```bash
pip install -r requirements.txt
```

Results and notes for each experiment live alongside the experiment code in `experiments/<name>/`.
