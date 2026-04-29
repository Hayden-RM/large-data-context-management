# Approaches & Prior Art

Notes on known strategies for large-data-in-context problems.

## 1. Retrieval-Augmented Generation (RAG)

Embed a corpus, retrieve top-k chunks relevant to a query, inject into context.

**Strengths:** scales arbitrarily, well-studied  
**Weaknesses:** retrieval quality is the ceiling; bad embeddings = bad answers; struggles with multi-hop reasoning

Key papers:
- Lewis et al. 2020 — original RAG paper
- Gao et al. 2023 — survey of RAG techniques

## 2. Hierarchical Summarisation

Summarise leaf chunks → summarise summaries → query the tree.

**Strengths:** preserves global structure; good for long-form docs  
**Weaknesses:** lossy at each level; slow; cost scales with corpus size

## 3. Map-Reduce

Map: run prompt independently over each chunk. Reduce: aggregate results.

**Strengths:** parallelisable; simple  
**Weaknesses:** no cross-chunk reasoning during map phase; aggregation is lossy

## 4. Sliding Window

Process the corpus in overlapping windows; maintain a running state/summary.

**Strengths:** handles sequential dependencies; low memory  
**Weaknesses:** stateful — errors compound; hard to parallelise

## 5. Structured Extraction

Parse/extract structured data from raw corpus before the LLM ever sees it. Pass only the schema or a filtered view.

**Strengths:** dramatic context reduction; deterministic  
**Weaknesses:** only works when the target schema is known upfront

## Open Questions

- [ ] How do hybrid approaches (e.g. RAG + map-reduce) compare to single-strategy baselines?
- [ ] What evaluation metrics best capture information fidelity at the output?
- [ ] Does chunk size interact significantly with model context length?
