"""
RecruitX — Embedder
====================
Generates dense vector embeddings using Sentence Transformers.
Supports batch encoding with progress tracking and caching.
Optimized for CPU inference on 100K candidates.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from src.config import EMBEDDING_MODEL, EMBEDDING_DIM, BATCH_SIZE, PRECOMPUTED_DIR


class Embedder:
    """
    Sentence-Transformer based text embedder.

    Uses all-MiniLM-L6-v2 — a compact 384-dim model optimized for
    semantic similarity with excellent speed/quality tradeoff.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._cache_path = PRECOMPUTED_DIR / "embeddings.pkl"

    def _load_model(self):
        """Lazy-load the sentence-transformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            # Force CPU
            self.model = self.model.to("cpu")
        return self.model

    def encode(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into dense embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        model = self._load_model()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text], show_progress=False, normalize=normalize)[0]

    def save_embeddings(self, embeddings: np.ndarray, candidate_ids: List[str]):
        """Save pre-computed embeddings to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "embeddings": embeddings,
            "candidate_ids": candidate_ids,
            "model_name": self.model_name,
        }
        with open(self._cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(candidate_ids)} embeddings to {self._cache_path}")

    def load_embeddings(self) -> Optional[dict]:
        """Load pre-computed embeddings from disk."""
        if self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                data = pickle.load(f)
            if data.get("model_name") == self.model_name:
                print(f"Loaded {len(data['candidate_ids'])} embeddings from cache")
                return data
        return None


def cosine_similarity_batch(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and a corpus of vectors.

    Args:
        query: Query vector of shape (dim,).
        corpus: Corpus matrix of shape (n, dim).

    Returns:
        Similarity scores of shape (n,).
    """
    # If vectors are already normalized, dot product = cosine similarity
    return corpus @ query
