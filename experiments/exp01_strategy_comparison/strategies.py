"""One function per strategy. Each returns (answer, input_tokens, output_tokens)."""

from __future__ import annotations
import anthropic
from src.chunking import chunk_by_tokens
from src.tokens import count_tokens
from experiments.exp01_strategy_comparison.config import (
    ANSWER_MODEL, CHUNK_TOKENS, CHUNK_OVERLAP, RAG_TOP_K,
    HIER_TARGET_TOKENS, WINDOW_TOKENS, STEP_TOKENS, EMBED_BACKEND,
)

_client = anthropic.Anthropic()


def _ask(prompt: str, model: str = ANSWER_MODEL, max_tokens: int = 256) -> tuple[str, int, int]:
    resp = _client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens


# --- RAG -------------------------------------------------------------------------

def run_rag(corpus: str, question: str) -> tuple[str, int, int]:
    from src.chunking import chunk_by_tokens
    from src.embed import embed_chunks
    from src.retrieval import VectorIndex

    chunks = chunk_by_tokens(corpus, chunk_size=CHUNK_TOKENS, overlap=CHUNK_OVERLAP)
    embeddings = embed_chunks(chunks, backend=EMBED_BACKEND)
    index = VectorIndex(chunks, embeddings)
    results = index.search_text(question, top_k=RAG_TOP_K)
    context = index.get_context(results)

    prompt = (
        f"Answer the question using only the context below. "
        f"Be concise — one sentence if possible.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return _ask(prompt)


# --- Map-reduce ------------------------------------------------------------------

def run_map_reduce(corpus: str, question: str) -> tuple[str, int, int]:
    from src.summarise import map_reduce
    from src.chunking import chunk_by_tokens

    chunks = chunk_by_tokens(corpus, chunk_size=CHUNK_TOKENS, overlap=0)
    total_in, total_out = 0, 0
    partial_answers = []

    for chunk in chunks:
        prompt = (
            f"If the following excerpt contains information relevant to the question, "
            f"extract it. Otherwise reply 'N/A'.\n\n"
            f"Question: {question}\n\nExcerpt:\n{chunk.text}"
        )
        answer, inp, out = _ask(prompt, max_tokens=128)
        total_in += inp
        total_out += out
        if answer.strip().upper() != "N/A":
            partial_answers.append(answer.strip())

    if not partial_answers:
        return "No relevant information found.", total_in, total_out

    combined = "\n".join(f"- {a}" for a in partial_answers)
    reduce_prompt = (
        f"Synthesise the following extracted facts into a single answer.\n\n"
        f"Question: {question}\n\nFacts:\n{combined}"
    )
    final, inp, out = _ask(reduce_prompt, max_tokens=256)
    return final, total_in + inp, total_out + out


# --- Hierarchical summarisation --------------------------------------------------

def run_hierarchical(corpus: str, question: str) -> tuple[str, int, int]:
    from src.summarise import hierarchical_summarise

    total_in, total_out = 0, 0

    def _summarise_chunk(text: str) -> str:
        nonlocal total_in, total_out
        prompt = f"Summarise the following, preserving all specific facts and numbers:\n\n{text}"
        ans, inp, out = _ask(prompt, max_tokens=256)
        total_in += inp
        total_out += out
        return ans

    # Recursively collapse until under target
    text = corpus
    while count_tokens(text) > HIER_TARGET_TOKENS:
        chunks = chunk_by_tokens(text, chunk_size=1024, overlap=0)
        text = "\n\n".join(_summarise_chunk(c.text) for c in chunks)

    prompt = f"Answer the question based on the following summary.\n\nQuestion: {question}\n\nSummary:\n{text}"
    answer, inp, out = _ask(prompt, max_tokens=256)
    return answer, total_in + inp, total_out + out


# --- Sliding window QA -----------------------------------------------------------

def run_sliding_window(corpus: str, question: str) -> tuple[str, int, int]:
    from src.chunking import sliding_window

    windows = sliding_window(corpus, window_tokens=WINDOW_TOKENS, step_tokens=STEP_TOKENS)
    total_in, total_out = 0, 0
    hits = []

    for window in windows:
        prompt = (
            f"If this excerpt answers the question, provide the answer. "
            f"Otherwise reply 'N/A'.\n\n"
            f"Question: {question}\n\nExcerpt:\n{window.text}"
        )
        answer, inp, out = _ask(prompt, max_tokens=128)
        total_in += inp
        total_out += out
        if answer.strip().upper() != "N/A":
            hits.append(answer.strip())

    if not hits:
        return "No relevant window found.", total_in, total_out

    if len(hits) == 1:
        return hits[0], total_in, total_out

    combined = "\n".join(f"- {h}" for h in hits)
    merge_prompt = f"Pick the most specific answer to the question from these candidates.\n\nQuestion: {question}\n\nCandidates:\n{combined}"
    final, inp, out = _ask(merge_prompt, max_tokens=128)
    return final, total_in + inp, total_out + out
