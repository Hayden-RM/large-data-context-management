"""Summarisation strategies: flat, hierarchical, and map-reduce."""

from __future__ import annotations
from src.chunking import Chunk, chunk_by_tokens
from src.tokens import count_tokens, fits_in_context


def _call_llm(prompt: str, model: str, max_tokens: int = 1024) -> str:
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def summarise_chunk(
    chunk: Chunk | str,
    instruction: str = "Summarise the following text concisely, preserving key facts:",
    model: str = "claude-haiku-4-5",
    max_tokens: int = 512,
) -> str:
    text = chunk.text if isinstance(chunk, Chunk) else chunk
    return _call_llm(f"{instruction}\n\n{text}", model=model, max_tokens=max_tokens)


def map_reduce(
    chunks: list[Chunk],
    question: str,
    map_instruction: str = "Answer the following question based only on this excerpt. Be concise.",
    reduce_instruction: str = "Synthesise these partial answers into a single coherent response.",
    model: str = "claude-haiku-4-5",
    max_tokens_map: int = 256,
    max_tokens_reduce: int = 1024,
) -> str:
    # Map
    partial_answers = []
    for chunk in chunks:
        prompt = f"{map_instruction}\n\nQuestion: {question}\n\nExcerpt:\n{chunk.text}"
        partial_answers.append(_call_llm(prompt, model=model, max_tokens=max_tokens_map))

    # Reduce
    combined = "\n\n---\n\n".join(
        f"Partial answer {i+1}:\n{ans}" for i, ans in enumerate(partial_answers)
    )
    reduce_prompt = f"{reduce_instruction}\n\nQuestion: {question}\n\n{combined}"
    return _call_llm(reduce_prompt, model=model, max_tokens=max_tokens_reduce)


def hierarchical_summarise(
    text: str,
    target_tokens: int = 2048,
    chunk_size: int = 1024,
    model: str = "claude-haiku-4-5",
    max_tokens_per_summary: int = 256,
) -> str:
    """Recursively summarise until the text fits within target_tokens."""
    if count_tokens(text) <= target_tokens:
        return text

    chunks = chunk_by_tokens(text, chunk_size=chunk_size, overlap=0, model=model)
    summaries = [summarise_chunk(c, model=model, max_tokens=max_tokens_per_summary) for c in chunks]
    collapsed = "\n\n".join(summaries)

    # Recurse if still too large
    return hierarchical_summarise(
        collapsed,
        target_tokens=target_tokens,
        chunk_size=chunk_size,
        model=model,
        max_tokens_per_summary=max_tokens_per_summary,
    )


def sliding_window_qa(
    text: str,
    question: str,
    window_tokens: int = 4096,
    step_tokens: int = 2048,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 512,
) -> list[str]:
    """Run QA over each sliding window, returning per-window answers."""
    from src.chunking import sliding_window
    windows = sliding_window(text, window_tokens=window_tokens, step_tokens=step_tokens, model=model)
    results = []
    for window in windows:
        prompt = f"Answer the question based on the following excerpt.\n\nQuestion: {question}\n\nExcerpt:\n{window.text}"
        results.append(_call_llm(prompt, model=model, max_tokens=max_tokens))
    return results
