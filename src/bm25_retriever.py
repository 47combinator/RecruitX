"""
RecruitX — BM25 Retriever
===========================
Sparse keyword-based retrieval using BM25 (Okapi BM25).
Complements dense retrieval by catching exact keyword matches
that embedding models might miss.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import BM25_TOP_K, PRECOMPUTED_DIR


class BM25Retriever:
    """BM25-based sparse retrieval for candidate search."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.candidate_ids: List[str] = []
        self._cache_path = PRECOMPUTED_DIR / "bm25_index.pkl"

    def build_index(self, texts: List[str], candidate_ids: List[str]):
        """
        Build BM25 index from candidate texts.

        Args:
            texts: List of candidate text representations.
            candidate_ids: Corresponding candidate IDs.
        """
        # Tokenize texts (simple whitespace + lowering)
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.candidate_ids = candidate_ids
        print(f"BM25 index built with {len(candidate_ids)} documents")

    def query(self, query_text: str, top_k: int = BM25_TOP_K) -> List[Tuple[str, float]]:
        """
        Query the BM25 index.

        Args:
            query_text: The search query (JD text or keywords).
            top_k: Number of top results to return.

        Returns:
            List of (candidate_id, bm25_score) tuples, sorted by score descending.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self.candidate_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

        return results

    def get_scores_for_ids(self, query_text: str, target_ids: List[str]) -> dict:
        """
        Get BM25 scores for specific candidate IDs.

        Returns:
            Dict mapping candidate_id -> bm25_score.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built.")

        tokenized_query = self._tokenize(query_text)
        all_scores = self.bm25.get_scores(tokenized_query)

        id_to_idx = {cid: i for i, cid in enumerate(self.candidate_ids)}
        return {
            cid: float(all_scores[id_to_idx[cid]])
            for cid in target_ids
            if cid in id_to_idx
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization with lowering and cleanup."""
        text = text.lower()
        # Remove common punctuation but keep hyphens in words
        import re
        text = re.sub(r'[^\w\s\-]', ' ', text)
        tokens = text.split()
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    def save_index(self):
        """Save BM25 index to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bm25": self.bm25,
            "candidate_ids": self.candidate_ids,
        }
        with open(self._cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"BM25 index saved to {self._cache_path}")

    def load_index(self) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        if self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.candidate_ids = data["candidate_ids"]
            print(f"BM25 index loaded: {len(self.candidate_ids)} documents")
            return True
        return False
