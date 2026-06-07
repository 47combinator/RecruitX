"""
RecruitX — Hybrid Retriever
==============================
Combines dense (semantic) and sparse (BM25) retrieval using
Reciprocal Rank Fusion (RRF).

This is the core retrieval stage that produces a shortlist of
candidates for the downstream re-ranker.
"""

from typing import List, Tuple, Dict, Set
from collections import defaultdict

import numpy as np

from src.config import (
    BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K, RRF_K,
    JD_EMBEDDING_TEXT,
)
from src.embedder import Embedder, cosine_similarity_batch
from src.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse search with RRF fusion.

    The pipeline:
    1. Dense retrieval: Embed JD → cosine similarity → top DENSE_TOP_K
    2. Sparse retrieval: BM25 query → top BM25_TOP_K
    3. RRF fusion: Merge both lists → top HYBRID_TOP_K
    """

    def __init__(
        self,
        embedder: Embedder,
        bm25_retriever: BM25Retriever,
        candidate_embeddings: np.ndarray,
        candidate_ids: List[str],
    ):
        self.embedder = embedder
        self.bm25 = bm25_retriever
        self.candidate_embeddings = candidate_embeddings
        self.candidate_ids = candidate_ids

    def retrieve(
        self,
        query_text: str = JD_EMBEDDING_TEXT,
        dense_top_k: int = DENSE_TOP_K,
        sparse_top_k: int = BM25_TOP_K,
        final_top_k: int = HYBRID_TOP_K,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Perform hybrid retrieval.

        Args:
            query_text: The search query (JD text).
            dense_top_k: Number of dense results.
            sparse_top_k: Number of sparse results.
            final_top_k: Number of final fused results.
            dense_weight: Weight for dense scores in RRF.
            sparse_weight: Weight for sparse scores in RRF.

        Returns:
            List of (candidate_id, rrf_score, dense_score, bm25_score) tuples,
            sorted by rrf_score descending.
        """
        # --- Dense retrieval ---
        query_embedding = self.embedder.encode_single(query_text)
        dense_scores = cosine_similarity_batch(query_embedding, self.candidate_embeddings)

        # Get top-K dense results
        dense_top_indices = np.argsort(dense_scores)[::-1][:dense_top_k]
        dense_results = {
            self.candidate_ids[idx]: (rank + 1, float(dense_scores[idx]))
            for rank, idx in enumerate(dense_top_indices)
        }

        # --- Sparse retrieval ---
        sparse_results_raw = self.bm25.query(query_text, top_k=sparse_top_k)
        sparse_results = {
            cid: (rank + 1, score)
            for rank, (cid, score) in enumerate(sparse_results_raw)
        }

        # --- RRF Fusion ---
        all_candidates = set(dense_results.keys()) | set(sparse_results.keys())

        fused_scores = []
        for cid in all_candidates:
            rrf_score = 0.0

            if cid in dense_results:
                dense_rank, dense_sim = dense_results[cid]
                rrf_score += dense_weight * (1.0 / (RRF_K + dense_rank))
            else:
                dense_sim = 0.0

            if cid in sparse_results:
                sparse_rank, bm25_score = sparse_results[cid]
                rrf_score += sparse_weight * (1.0 / (RRF_K + sparse_rank))
            else:
                bm25_score = 0.0

            fused_scores.append((cid, rrf_score, dense_sim, bm25_score))

        # Sort by RRF score
        fused_scores.sort(key=lambda x: x[1], reverse=True)

        return fused_scores[:final_top_k]

    def get_all_dense_scores(self, query_text: str = JD_EMBEDDING_TEXT) -> Dict[str, float]:
        """Get dense similarity scores for all candidates."""
        query_embedding = self.embedder.encode_single(query_text)
        scores = cosine_similarity_batch(query_embedding, self.candidate_embeddings)
        return {cid: float(scores[i]) for i, cid in enumerate(self.candidate_ids)}
