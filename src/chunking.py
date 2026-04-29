"""Text chunking strategies."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from src.tokens import count_tokens


@dataclass
class Chunk:
    text: str
    index: int
    token_count: int = field(init=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.token_count = count_tokens(self.text)


def chunk_by_tokens(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    model: str = "claude-sonnet-4-6",
) -> list[Chunk]:
    """Split text into token-bounded chunks with overlap."""
    import tiktoken
    from src.tokens import _encoder

    enc = _encoder(model)
    token_ids = enc.encode(text)
    chunks = []
    step = chunk_size - overlap
    for i, start in enumerate(range(0, len(token_ids), step)):
        chunk_ids = token_ids[start : start + chunk_size]
        chunk_text = enc.decode(chunk_ids)
        chunks.append(Chunk(text=chunk_text, index=i))
        if start + chunk_size >= len(token_ids):
            break
    return chunks


def chunk_by_sentences(
    text: str,
    max_tokens: int = 512,
    overlap_sentences: int = 1,
    model: str = "claude-sonnet-4-6",
) -> list[Chunk]:
    """Split text at sentence boundaries, keeping chunks under max_tokens."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    chunk_idx = 0

    for sentence in sentences:
        s_tokens = count_tokens(sentence, model)
        if current_tokens + s_tokens > max_tokens and current:
            chunks.append(Chunk(text=" ".join(current), index=chunk_idx))
            chunk_idx += 1
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(count_tokens(s, model) for s in current)
        current.append(sentence)
        current_tokens += s_tokens

    if current:
        chunks.append(Chunk(text=" ".join(current), index=chunk_idx))

    return chunks


def chunk_by_paragraphs(
    text: str,
    max_tokens: int = 1024,
    model: str = "claude-sonnet-4-6",
) -> list[Chunk]:
    """Split on blank lines; merge short paragraphs, split long ones."""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    chunk_idx = 0

    for para in paragraphs:
        p_tokens = count_tokens(para, model)
        if p_tokens > max_tokens:
            if current_parts:
                chunks.append(Chunk(text="\n\n".join(current_parts), index=chunk_idx))
                chunk_idx += 1
                current_parts, current_tokens = [], 0
            # split the oversized paragraph by tokens
            for sub in chunk_by_tokens(para, chunk_size=max_tokens, overlap=0, model=model):
                sub.index = chunk_idx
                chunks.append(sub)
                chunk_idx += 1
        elif current_tokens + p_tokens > max_tokens and current_parts:
            chunks.append(Chunk(text="\n\n".join(current_parts), index=chunk_idx))
            chunk_idx += 1
            current_parts, current_tokens = [para], p_tokens
        else:
            current_parts.append(para)
            current_tokens += p_tokens

    if current_parts:
        chunks.append(Chunk(text="\n\n".join(current_parts), index=chunk_idx))

    return chunks


def sliding_window(
    text: str,
    window_tokens: int = 4096,
    step_tokens: int = 2048,
    model: str = "claude-sonnet-4-6",
) -> list[Chunk]:
    """Overlapping windows — useful for sequential processing strategies."""
    return chunk_by_tokens(text, chunk_size=window_tokens, overlap=window_tokens - step_tokens, model=model)
