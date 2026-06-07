"""
RecruitX — Ranking Pipeline
=============================
Main entry point that produces the final submission CSV.
Must complete in ≤5 minutes on CPU with 16GB RAM and no network.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline:
    1. Load pre-computed artifacts (features, embeddings, BM25 index, honeypots)
    2. Run hybrid retrieval → top 300 candidates
    3. Score candidates with weighted ensemble ranker
    4. Filter honeypots
    5. Select top 100
    6. Generate per-candidate reasoning
    7. Write submission CSV
"""

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from tqdm import tqdm

from src.config import (
    CANDIDATES_FILE, PRECOMPUTED_DIR, OUTPUT_DIR,
    JD_EMBEDDING_TEXT, HYBRID_TOP_K, FINAL_TOP_K,
)
from src.candidate_loader import load_candidates
from src.feature_engineer import extract_features
from src.honeypot_detector import detect_honeypot
from src.text_builder import build_candidate_text
from src.embedder import Embedder, cosine_similarity_batch
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.ranker import WeightedEnsembleRanker
from src.explainer import generate_reasoning


def load_precomputed():
    """Load all pre-computed artifacts from disk."""
    artifacts = {}

    # Features
    features_path = PRECOMPUTED_DIR / "features.pkl"
    if features_path.exists():
        with open(features_path, "rb") as f:
            data = pickle.load(f)
        artifacts["features"] = {
            cid: feat for cid, feat in
            zip(data["candidate_ids"], data["features"])
        }
        print(f"  Loaded {len(artifacts['features'])} candidate features")
    else:
        artifacts["features"] = None
        print("  ⚠ No pre-computed features found")

    # Honeypots
    hp_path = PRECOMPUTED_DIR / "honeypots.pkl"
    if hp_path.exists():
        with open(hp_path, "rb") as f:
            artifacts["honeypots"] = pickle.load(f)
        flagged = sum(1 for v in artifacts["honeypots"].values() if v["score"] >= 0.55)
        print(f"  Loaded honeypot data ({flagged} flagged)")
    else:
        artifacts["honeypots"] = {}
        print("  ⚠ No honeypot data found")

    # Embeddings
    embedder = Embedder()
    cached = embedder.load_embeddings()
    if cached:
        artifacts["embeddings"] = cached["embeddings"]
        artifacts["embedding_ids"] = cached["candidate_ids"]
    else:
        artifacts["embeddings"] = None
        artifacts["embedding_ids"] = None
        print("  ⚠ No pre-computed embeddings found")

    # BM25 index
    bm25 = BM25Retriever()
    if bm25.load_index():
        artifacts["bm25"] = bm25
    else:
        artifacts["bm25"] = None
        print("  ⚠ No BM25 index found")

    return artifacts


def run_ranking(
    candidates: List[Dict[str, Any]],
    artifacts: Dict,
) -> List[Tuple[str, int, float, str]]:
    """
    Run the full ranking pipeline.

    Returns:
        List of (candidate_id, rank, score, reasoning) tuples.
    """
    print("\n[Step 1] Setting up retrieval...")
    embedder = Embedder()

    # Build candidate lookup
    cand_lookup = {c["candidate_id"]: c for c in candidates}
    candidate_ids = [c["candidate_id"] for c in candidates]

    # Use pre-computed or compute on-the-fly
    precomputed_features = artifacts.get("features") or {}
    honeypot_data = artifacts.get("honeypots") or {}
    embeddings = artifacts.get("embeddings")
    embedding_ids = artifacts.get("embedding_ids")
    bm25 = artifacts.get("bm25")

    # If no precomputed embeddings, compute on-the-fly (slower but works)
    if embeddings is None:
        print("  Computing embeddings on-the-fly...")
        texts = [build_candidate_text(c) for c in tqdm(candidates, desc="Text")]
        embeddings = embedder.encode(texts, show_progress=True)
        embedding_ids = candidate_ids

    if bm25 is None:
        print("  Building BM25 index on-the-fly...")
        texts = [build_candidate_text(c) for c in candidates]
        bm25 = BM25Retriever()
        bm25.build_index(texts, candidate_ids)

    # --- Hybrid Retrieval ---
    print("\n[Step 2] Running hybrid retrieval...")
    retriever = HybridRetriever(embedder, bm25, embeddings, embedding_ids)
    shortlist = retriever.retrieve(
        query_text=JD_EMBEDDING_TEXT,
        final_top_k=HYBRID_TOP_K,
    )
    print(f"  Shortlisted {len(shortlist)} candidates via hybrid retrieval")

    # Collect scores from retrieval
    shortlist_ids = [cid for cid, _, _, _ in shortlist]
    dense_scores = {cid: dscore for cid, _, dscore, _ in shortlist}
    bm25_scores = {cid: bscore for cid, _, _, bscore in shortlist}

    # --- Feature extraction for shortlist (if not precomputed) ---
    print("\n[Step 3] Preparing features for shortlist...")
    candidate_data = []
    for cid in tqdm(shortlist_ids, desc="Scoring"):
        if cid in precomputed_features:
            features = precomputed_features[cid]
        else:
            cand = cand_lookup.get(cid)
            features = extract_features(cand) if cand else {}

        hp_entry = honeypot_data.get(cid, {})
        hp_score = hp_entry.get("score", 0.0) if isinstance(hp_entry, dict) else 0.0

        candidate_data.append({
            "candidate_id": cid,
            "features": features,
            "dense_similarity": dense_scores.get(cid, 0.0),
            "bm25_score": bm25_scores.get(cid, 0.0),
            "honeypot_score": hp_score,
        })

    # --- Ranking ---
    print("\n[Step 4] Running weighted ensemble ranking...")
    ranker = WeightedEnsembleRanker()
    rankings = ranker.rank_candidates(candidate_data, top_k=FINAL_TOP_K)
    print(f"  Top {len(rankings)} candidates ranked")

    # --- Reasoning generation ---
    print("\n[Step 5] Generating reasoning...")
    results = []
    for rank_idx, (cid, score, components) in enumerate(rankings):
        rank = rank_idx + 1
        candidate = cand_lookup.get(cid, {})
        hp_score = honeypot_data.get(cid, {}).get("score", 0.0) if isinstance(honeypot_data.get(cid), dict) else 0.0

        reasoning = generate_reasoning(candidate, rank, score, components, hp_score)
        results.append((cid, rank, score, reasoning))

    return results


def write_submission_csv(
    results: List[Tuple[str, int, float, str]],
    output_path: Path,
):
    """Write results to submission CSV format."""
    # Normalize scores to be in a nice range
    if results:
        max_score = max(r[2] for r in results)
        min_score = min(r[2] for r in results)
        score_range = max_score - min_score if max_score > min_score else 1.0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for cid, rank, raw_score, reasoning in results:
            # Normalize score to [0.2, 1.0] range, monotonically decreasing
            normalized_score = 0.2 + 0.8 * (raw_score - min_score) / score_range if score_range > 0 else 0.5
            # Round to 4 decimal places
            normalized_score = round(normalized_score, 4)
            writer.writerow([cid, rank, f"{normalized_score:.4f}", reasoning])

    print(f"\nSubmission written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RecruitX Ranking Pipeline")
    parser.add_argument("--candidates", type=str, default=str(CANDIDATES_FILE),
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR / "submission.csv"),
                        help="Output CSV path")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Max candidates (for testing)")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 60)
    print("RecruitX — Intelligent Candidate Ranking Engine")
    print("=" * 60)

    # Load candidates
    print("\nLoading candidates...")
    candidates = load_candidates(Path(args.candidates), max_candidates=args.max_candidates)
    print(f"Loaded {len(candidates)} candidates")

    # Load pre-computed artifacts
    print("\nLoading pre-computed artifacts...")
    artifacts = load_precomputed()

    # Run ranking
    results = run_ranking(candidates, artifacts)

    # Write output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission_csv(results, output_path)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Ranking complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    # Quick validation
    print("\nTop-5 candidates:")
    for cid, rank, score, reasoning in results[:5]:
        print(f"  #{rank} {cid} (score: {score:.4f})")
        print(f"      {reasoning[:100]}...")

    # Check for honeypots in top 100
    honeypot_data = artifacts.get("honeypots", {})
    hp_in_top100 = sum(
        1 for cid, _, _, _ in results
        if isinstance(honeypot_data.get(cid), dict) and
           honeypot_data[cid].get("score", 0) >= 0.55
    )
    print(f"\nHoneypots in top 100: {hp_in_top100} (must be <=10)")

    if elapsed > 300:
        print(f"\n[WARNING] Ranking took {elapsed:.0f}s (> 5 min limit!)")
    else:
        print(f"\n[OK] Runtime: {elapsed:.1f}s < 300s limit")


if __name__ == "__main__":
    main()
