"""Embedding utilities — local (sentence-transformers) or OpenAI."""

from __future__ import annotations
import numpy as np
from src.chunking import Chunk


def embed_local(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed using a local sentence-transformers model. Returns (N, D) float32 array."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 50)


def embed_openai(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 512,
) -> np.ndarray:
    """Embed using OpenAI embeddings API. Requires OPENAI_API_KEY env var."""
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])

    return np.array(all_embeddings, dtype=np.float32)


def embed_chunks(
    chunks: list[Chunk],
    backend: str = "local",
    **kwargs,
) -> np.ndarray:
    """Embed a list of Chunk objects. backend: 'local' or 'openai'."""
    texts = [c.text for c in chunks]
    if backend == "openai":
        return embed_openai(texts, **kwargs)
    return embed_local(texts, **kwargs)
