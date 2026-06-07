"""
RecruitX — Pre-computation Pipeline
=====================================
Run this ONCE to pre-compute:
1. All candidate features (40+ per candidate)
2. Sentence-transformer embeddings (100K × 384-dim)
3. BM25 index
4. Honeypot detection scores

Pre-computation can exceed the 5-minute ranking budget.
Only the final ranking step (rank.py) must complete in ≤5 minutes.

Usage:
    python precompute.py [--max-candidates N] [--skip-embeddings]
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import CANDIDATES_FILE, PRECOMPUTED_DIR
from src.candidate_loader import load_candidates
from src.feature_engineer import extract_features
from src.honeypot_detector import detect_honeypot
from src.text_builder import build_candidate_text
from src.embedder import Embedder
from src.bm25_retriever import BM25Retriever


def main():
    parser = argparse.ArgumentParser(description="RecruitX Pre-computation Pipeline")
    parser.add_argument("--candidates", type=str, default=str(CANDIDATES_FILE),
                        help="Path to candidates.jsonl")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Max candidates to process (for testing)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding computation")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for embedding computation")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 60)
    print("RecruitX Pre-computation Pipeline")
    print("=" * 60)

    # --- 1. Load candidates ---
    print("\n[1/5] Loading candidates...")
    candidates = load_candidates(Path(args.candidates), max_candidates=args.max_candidates)
    print(f"Loaded {len(candidates)} candidates")

    candidate_ids = [c["candidate_id"] for c in candidates]

    # --- 2. Extract features ---
    print("\n[2/5] Extracting features...")
    features_list = []
    for c in tqdm(candidates, desc="Features"):
        features_list.append(extract_features(c))

    # Save features
    features_path = PRECOMPUTED_DIR / "features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump({"candidate_ids": candidate_ids, "features": features_list}, f,
                     protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Features saved to {features_path}")

    # --- 3. Honeypot detection ---
    print("\n[3/5] Running honeypot detection...")
    honeypot_data = {}
    honeypot_count = 0
    for c in tqdm(candidates, desc="Honeypots"):
        hp_score, reasons = detect_honeypot(c)
        honeypot_data[c["candidate_id"]] = {
            "score": hp_score,
            "reasons": reasons,
        }
        if hp_score >= 0.55:
            honeypot_count += 1

    hp_path = PRECOMPUTED_DIR / "honeypots.pkl"
    with open(hp_path, "wb") as f:
        pickle.dump(honeypot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Detected {honeypot_count} potential honeypots. Saved to {hp_path}")

    # --- 4. Build text representations & BM25 index ---
    print("\n[4/5] Building text representations & BM25 index...")
    texts = []
    for c in tqdm(candidates, desc="Text building"):
        texts.append(build_candidate_text(c, mode="full"))

    bm25 = BM25Retriever()
    bm25.build_index(texts, candidate_ids)
    bm25.save_index()

    # --- 5. Compute embeddings ---
    if not args.skip_embeddings:
        print("\n[5/5] Computing sentence-transformer embeddings...")
        embedder = Embedder()

        # Check cache first
        cached = embedder.load_embeddings()
        if cached and len(cached["candidate_ids"]) == len(candidate_ids):
            print("Using cached embeddings (already computed)")
        else:
            embeddings = embedder.encode(
                texts,
                batch_size=args.batch_size,
                show_progress=True,
            )
            embedder.save_embeddings(embeddings, candidate_ids)
    else:
        print("\n[5/5] Skipping embeddings (--skip-embeddings)")

    # --- Done ---
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Pre-computation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Artifacts saved to: {PRECOMPUTED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
