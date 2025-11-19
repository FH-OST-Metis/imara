"""
Local embedding utilities.

- Uses Sentence-Transformers locally (no network) to generate embeddings for text chunks.
- Default model is small and fast for on-prem GPUs/CPUs.
- Normalizes embeddings for cosine similarity (Milvus HNSW/IVF + COSINE).

Installation (offline-friendly):
  pip install sentence-transformers

If you must avoid internet, download the model once and point model_name_or_path
to a local directory containing the model files.

This module aligns with the OpenSpec design and Milvus configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _IMPORT_ERROR = e


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # small, strong baseline for RAG
DEFAULT_BATCH_SIZE = 64


@dataclass
class EmbeddingConfig:
    model_name_or_path: str = DEFAULT_MODEL
    device: str | None = None  # e.g., "cuda", "cpu" (auto if None)
    normalize: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE


class EmbeddingModel:
    def __init__(self, cfg: EmbeddingConfig = EmbeddingConfig()) -> None:
        if SentenceTransformer is None:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            ) from _IMPORT_ERROR
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name_or_path, device=cfg.device)
        # Best-effort detection of embedding output dimension for early validation
        try:
            dim = 0
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                dim = int(self.model.get_sentence_embedding_dimension())
            self.output_dim = int(dim) if dim and dim > 0 else 0
        except Exception:
            self.output_dim = 0

    @staticmethod
    def _l2_normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return vecs / norms

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a batch of texts into embeddings.

        Returns:
            np.ndarray of shape (N, D), dtype float32
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vecs = self.model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves for portability
            show_progress_bar=False,
        ).astype(np.float32)

        if self.cfg.normalize:
            vecs = self._l2_normalize(vecs).astype(np.float32)
        return vecs

    def embed(self, text: str) -> List[float]:
        """
        Encode a single string and return a Python list[float] for JSON-ability.
        """
        arr = self.embed_texts([text])
        if arr.size == 0:
            return []
        return arr[0].astype(np.float32).tolist()


def embed_iter(model: EmbeddingModel, texts: Iterable[str]) -> Iterable[List[float]]:
    """
    Convenience generator for streaming embeddings from an iterable of texts.
    """
    batch: list[str] = []
    for t in texts:
        batch.append(t)
        if len(batch) >= model.cfg.batch_size:
            vecs = model.embed_texts(batch)
            for v in vecs:
                yield v.tolist()
            batch.clear()
    if batch:
        vecs = model.embed_texts(batch)
        for v in vecs:
            yield v.tolist()


if __name__ == "__main__":
    m = EmbeddingModel()
    demo = ["hello world", "retrieval augmented generation", "milvus vector database"]
    vecs = m.embed_texts(demo)
    print({"shape": vecs.shape, "norms": np.linalg.norm(vecs, axis=1).round(4).tolist()})