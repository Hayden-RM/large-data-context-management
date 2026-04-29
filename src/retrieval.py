"""Vector retrieval over embedded chunks using FAISS."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from src.chunking import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float  # cosine similarity, higher is better


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        import faiss

        assert len(chunks) == len(embeddings), "chunks and embeddings must be the same length"
        self.chunks = chunks
        dim = embeddings.shape[1]
        # Normalise for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalised = embeddings / np.where(norms == 0, 1, norms)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(normalised.astype(np.float32))
        self._embeddings = normalised

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        vec = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        scores, indices = self._index.search(vec, top_k)
        return [
            RetrievalResult(chunk=self.chunks[idx], score=float(scores[0][i]))
            for i, idx in enumerate(indices[0])
            if idx != -1
        ]

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        embed_fn=None,
    ) -> list[RetrievalResult]:
        """Convenience wrapper — embeds query text then searches."""
        if embed_fn is None:
            from src.embed import embed_local
            embed_fn = lambda t: embed_local([t])[0]
        q_emb = embed_fn(query)
        return self.search(q_emb, top_k=top_k)

    def get_context(self, results: list[RetrievalResult], separator: str = "\n\n---\n\n") -> str:
        return separator.join(r.chunk.text for r in results)


def build_index(chunks: list[Chunk], backend: str = "local", **embed_kwargs) -> VectorIndex:
    from src.embed import embed_chunks
    embeddings = embed_chunks(chunks, backend=backend, **embed_kwargs)
    return VectorIndex(chunks, embeddings)
