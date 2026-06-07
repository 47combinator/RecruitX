"""
RecruitX — Ranker
==================
Weighted ensemble scorer that combines 40+ features into a final
candidate ranking score. Applies penalty multipliers for disqualifiers
like honeypots, keyword stuffers, and consulting-only careers.

The scoring philosophy follows the JD's emphasis:
- Skills and career relevance matter most
- Behavioral signals are a strong multiplier
- Disqualifiers can dramatically reduce scores
"""

from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from src.config import (
    SCORING_WEIGHTS,
    PENALTY_HONEYPOT,
    PENALTY_CONSULTING_ONLY,
    PENALTY_KEYWORD_STUFFER,
    PENALTY_NO_ML_SIGNAL,
    PENALTY_INACTIVE_6_MONTHS,
    PENALTY_VERY_LONG_NOTICE,
    FINAL_TOP_K,
)


class WeightedEnsembleRanker:
    """
    Ranks candidates using a weighted combination of feature categories.

    Each category produces a [0, 1] component score.
    The final score = weighted sum × penalty multipliers.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or SCORING_WEIGHTS

    def score_candidate(
        self,
        features: Dict[str, float],
        dense_similarity: float = 0.0,
        bm25_score: float = 0.0,
        honeypot_score: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute final score for a single candidate.

        Args:
            features: Extracted feature dictionary.
            dense_similarity: Cosine similarity from dense retrieval.
            bm25_score: BM25 score (normalized).
            honeypot_score: Honeypot detection score.

        Returns:
            (final_score, component_scores): Final score and breakdown.
        """
        components = {}

        # --- 1. Skill Match Score (weight: 0.35) ---
        skill_score = (
            features.get("must_have_ratio", 0) * 0.35 +
            features.get("nice_to_have_ratio", 0) * 0.15 +
            features.get("skill_proficiency_score", 0) * 0.15 +
            features.get("skill_endorsement_score", 0) * 0.10 +
            features.get("skill_duration_score", 0) * 0.10 +
            features.get("ai_ml_skill_depth", 0) * 0.10 +
            features.get("skill_assessment_avg", 0) * 0.05
        )
        components["skill_match"] = min(skill_score, 1.0)

        # --- 2. Career Relevance Score (weight: 0.20) ---
        career_score = (
            features.get("current_title_relevance", 0) * 0.25 +
            features.get("max_career_title_relevance", 0) * 0.20 +
            features.get("has_production_ml", 0) * 0.20 +
            features.get("ml_signal_count", 0) * 0.10 +
            features.get("product_company_ratio", 0) * 0.10 +
            features.get("job_stability_score", 0) * 0.05 +
            features.get("tech_industry_ratio", 0) * 0.05 +
            features.get("prod_signal_count", 0) * 0.05
        )
        components["career_relevance"] = min(career_score, 1.0)

        # --- 3. Semantic Similarity Score (weight: 0.15) ---
        # Combine dense and sparse retrieval signals
        sem_score = dense_similarity * 0.7 + min(bm25_score / 50.0, 1.0) * 0.3
        components["semantic_similarity"] = min(max(sem_score, 0), 1.0)

        # --- 4. Behavioral Score (weight: 0.10) ---
        behavioral_score = (
            features.get("recruiter_response_rate", 0) * 0.25 +
            features.get("recency_score", 0) * 0.20 +
            features.get("interview_completion_rate", 0) * 0.10 +
            features.get("profile_completeness", 0) * 0.10 +
            features.get("response_time_score", 0) * 0.10 +
            features.get("github_activity", 0) * 0.10 +
            features.get("open_to_work", 0) * 0.05 +
            features.get("verification_score", 0) * 0.05 +
            features.get("saved_by_recruiters", 0) * 0.05
        )
        components["behavioral_score"] = min(behavioral_score, 1.0)

        # --- 5. Experience Fit Score (weight: 0.08) ---
        exp_score = features.get("experience_in_band", 0)
        if features.get("too_junior", 0) > 0.5:
            exp_score *= 0.3
        components["experience_fit"] = min(exp_score, 1.0)

        # --- 6. Education Quality Score (weight: 0.05) ---
        edu_score = (
            features.get("education_relevance", 0) * 0.50 +
            features.get("education_tier", 0) * 0.25 +
            features.get("degree_level", 0) * 0.25
        )
        components["education_quality"] = min(edu_score, 1.0)

        # --- 7. Location Fit Score (weight: 0.05) ---
        location_score = (
            features.get("location_composite", 0) * 0.60 +
            features.get("work_mode_fit", 0) * 0.20 +
            features.get("notice_period_score", 0) * 0.20
        )
        components["location_fit"] = min(location_score, 1.0)

        # --- 8. Certification Bonus (weight: 0.02) ---
        cert_score = features.get("relevant_cert_count", 0)
        components["certification_bonus"] = min(cert_score, 1.0)

        # --- Compute weighted sum ---
        raw_score = sum(
            self.weights.get(key, 0) * value
            for key, value in components.items()
        )

        # --- Apply penalty multipliers ---
        penalty_multiplier = 1.0

        if honeypot_score >= 0.55:
            penalty_multiplier *= PENALTY_HONEYPOT  # 0.0 = eliminate

        if features.get("consulting_only_flag", 0) > 0.5:
            penalty_multiplier *= PENALTY_CONSULTING_ONLY

        if features.get("keyword_stuffer_flag", 0) > 0.5:
            penalty_multiplier *= PENALTY_KEYWORD_STUFFER

        if (features.get("ml_signal_count", 0) < 0.1 and
            features.get("current_title_relevance", 0) < 0.3 and
            features.get("must_have_ratio", 0) < 0.1):
            penalty_multiplier *= PENALTY_NO_ML_SIGNAL

        if features.get("days_inactive", 0) > 180:
            penalty_multiplier *= PENALTY_INACTIVE_6_MONTHS

        notice = features.get("notice_period_score", 1.0)
        if notice < 0.35:
            penalty_multiplier *= PENALTY_VERY_LONG_NOTICE

        final_score = raw_score * penalty_multiplier
        components["penalty_multiplier"] = penalty_multiplier
        components["raw_score"] = raw_score
        components["final_score"] = final_score

        return final_score, components

    def rank_candidates(
        self,
        candidate_data: List[Dict[str, Any]],
        top_k: int = FINAL_TOP_K,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Rank a list of candidates.

        Args:
            candidate_data: List of dicts with keys:
                - candidate_id: str
                - features: Dict[str, float]
                - dense_similarity: float
                - bm25_score: float
                - honeypot_score: float

        Returns:
            List of (candidate_id, final_score, components) tuples,
            sorted by final_score descending, top_k only.
        """
        scored = []

        for entry in candidate_data:
            cid = entry["candidate_id"]
            features = entry["features"]
            dense_sim = entry.get("dense_similarity", 0.0)
            bm25 = entry.get("bm25_score", 0.0)
            hp_score = entry.get("honeypot_score", 0.0)

            final_score, components = self.score_candidate(
                features, dense_sim, bm25, hp_score
            )

            scored.append((cid, final_score, components))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Handle ties: break by candidate_id ascending (per spec)
        # Group by score and sort ties
        result = []
        i = 0
        while i < len(scored):
            j = i + 1
            while j < len(scored) and abs(scored[j][1] - scored[i][1]) < 1e-10:
                j += 1
            # Sort tie group by candidate_id
            tie_group = sorted(scored[i:j], key=lambda x: x[0])
            result.extend(tie_group)
            i = j

        return result[:top_k]
