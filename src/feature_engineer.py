"""
RecruitX — Feature Engineer
=============================
Extracts 40+ structured features from each candidate profile,
organized into skill-match, career, education, behavioral, and location categories.
These features power the weighted ensemble ranker.
"""

import re
from datetime import date
from typing import Dict, Any, List, Optional, Tuple

from src.config import (
    MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS,
    TITLE_RELEVANCE, CONSULTING_FIRMS, PRODUCT_COMPANIES,
    EDUCATION_FIELD_RELEVANCE, EDUCATION_TIER_SCORES, DEGREE_LEVEL_SCORES,
    PREFERRED_LOCATIONS, INDIA_KEYWORDS,
    JD_EXPERIENCE_RANGE, JD_EXPERIENCE_HARD_MIN,
    REFERENCE_DATE,
)
from src.jd_parser import match_skills


# =============================================================================
# Main Feature Extraction
# =============================================================================

def extract_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract all features from a single candidate.

    Returns a flat dictionary of feature_name -> float_value.
    All features are in [0, 1] range or are raw values to be normalized later.
    """
    features = {}

    # Sub-extractors
    features.update(_extract_skill_features(candidate))
    features.update(_extract_career_features(candidate))
    features.update(_extract_education_features(candidate))
    features.update(_extract_behavioral_features(candidate))
    features.update(_extract_location_features(candidate))
    features.update(_extract_certification_features(candidate))

    return features


def extract_features_batch(candidates: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Extract features for a batch of candidates."""
    return [extract_features(c) for c in candidates]


# =============================================================================
# Skill-Match Features (Weight: 0.35)
# =============================================================================

def _extract_skill_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Extract skill-related features."""
    features = {}
    skills = candidate.get("skills", [])
    skill_names = [s["name"] for s in skills]

    # Must-have skill matching
    must_have_matches = match_skills(skill_names, MUST_HAVE_SKILLS)
    matched_must = sum(1 for v in must_have_matches.values() if v)
    total_must = len(must_have_matches)

    features["must_have_count"] = matched_must
    features["must_have_ratio"] = matched_must / max(total_must, 1)

    # Nice-to-have skill matching
    nice_matches = match_skills(skill_names, NICE_TO_HAVE_SKILLS)
    matched_nice = sum(1 for v in nice_matches.values() if v)
    total_nice = len(nice_matches)

    features["nice_to_have_count"] = matched_nice
    features["nice_to_have_ratio"] = matched_nice / max(total_nice, 1)

    # Proficiency-weighted skill score for matched skills
    prof_weights = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.5, "beginner": 0.25}
    all_jd_aliases = set()
    for aliases in MUST_HAVE_SKILLS.values():
        all_jd_aliases.update(a.lower() for a in aliases)
    for aliases in NICE_TO_HAVE_SKILLS.values():
        all_jd_aliases.update(a.lower() for a in aliases)

    prof_score = 0.0
    endorse_score = 0.0
    duration_score = 0.0
    matched_skill_count = 0

    for s in skills:
        s_lower = s["name"].lower()
        is_relevant = any(
            alias in s_lower or s_lower in alias
            for alias in all_jd_aliases
        )
        if is_relevant:
            matched_skill_count += 1
            prof = s.get("proficiency", "beginner")
            prof_score += prof_weights.get(prof, 0.25)
            endorse_score += min(s.get("endorsements", 0), 50) / 50
            duration_score += min(s.get("duration_months", 0), 60) / 60

    features["skill_proficiency_score"] = prof_score / max(total_must + total_nice, 1)
    features["skill_endorsement_score"] = endorse_score / max(matched_skill_count, 1)
    features["skill_duration_score"] = duration_score / max(matched_skill_count, 1)
    features["matched_relevant_skills"] = matched_skill_count
    features["total_skills"] = len(skills)

    # Skill assessment scores (Redrob platform)
    redrob = candidate.get("redrob_signals", {})
    assessments = redrob.get("skill_assessment_scores", {})

    if assessments:
        relevant_assessments = []
        for skill_name, score_val in assessments.items():
            if any(alias in skill_name.lower() or skill_name.lower() in alias
                   for alias in all_jd_aliases):
                relevant_assessments.append(score_val)

        if relevant_assessments:
            features["skill_assessment_avg"] = sum(relevant_assessments) / len(relevant_assessments) / 100
        else:
            features["skill_assessment_avg"] = sum(assessments.values()) / len(assessments) / 100 if assessments else 0.0
    else:
        features["skill_assessment_avg"] = 0.0

    features["has_assessments"] = 1.0 if assessments else 0.0

    # ---- Keyword stuffer detection ----
    # High skill count + many "advanced"/"expert" but low endorsements & duration
    if len(skills) >= 10:
        high_prof_count = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced"))
        avg_endorsements = sum(s.get("endorsements", 0) for s in skills) / max(len(skills), 1)
        avg_duration = sum(s.get("duration_months", 0) for s in skills) / max(len(skills), 1)

        # Flag if many high-proficiency skills but no backing evidence
        if high_prof_count >= 8 and avg_endorsements < 5 and avg_duration < 12:
            features["keyword_stuffer_flag"] = 1.0
        else:
            features["keyword_stuffer_flag"] = 0.0
    else:
        features["keyword_stuffer_flag"] = 0.0

    # ---- AI/ML skill depth composite ----
    ai_ml_keywords = {
        "machine learning", "deep learning", "nlp", "natural language processing",
        "pytorch", "tensorflow", "bert", "gpt", "transformer", "embeddings",
        "neural network", "classification", "regression", "clustering",
        "recommendation", "ranking", "information retrieval", "search",
        "computer vision", "image classification", "object detection",
        "reinforcement learning", "generative ai", "llm", "fine-tuning",
        "sentence transformers", "faiss", "vector", "qdrant", "elasticsearch",
    }

    ai_depth = 0.0
    for s in skills:
        s_lower = s["name"].lower()
        if any(kw in s_lower for kw in ai_ml_keywords):
            prof = prof_weights.get(s.get("proficiency", "beginner"), 0.25)
            dur = min(s.get("duration_months", 0), 60) / 60
            ai_depth += prof * dur

    features["ai_ml_skill_depth"] = min(ai_depth / 5.0, 1.0)  # Normalize

    return features


# =============================================================================
# Career Features (Weight: 0.20)
# =============================================================================

def _extract_career_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Extract career-related features."""
    features = {}
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # Years of experience
    yoe = profile.get("years_of_experience", 0)
    features["years_of_experience"] = yoe

    # Experience in ideal range (5-9 per JD, but we widen slightly)
    min_exp, max_exp = JD_EXPERIENCE_RANGE
    if min_exp <= yoe <= max_exp:
        features["experience_in_band"] = 1.0
    elif yoe < min_exp:
        features["experience_in_band"] = max(0, yoe / min_exp)
    else:
        # Slight penalty for very high experience but still okay
        features["experience_in_band"] = max(0.5, 1.0 - (yoe - max_exp) * 0.05)

    features["too_junior"] = 1.0 if yoe < JD_EXPERIENCE_HARD_MIN else 0.0

    # Current title relevance
    current_title = profile.get("current_title", "").lower().strip()
    features["current_title_relevance"] = _get_title_relevance(current_title)

    # Max title relevance across all career positions
    max_title_rel = features["current_title_relevance"]
    for job in career:
        title = job.get("title", "").lower().strip()
        rel = _get_title_relevance(title)
        max_title_rel = max(max_title_rel, rel)
    features["max_career_title_relevance"] = max_title_rel

    # Product company experience vs consulting-only
    consulting_months = 0
    product_months = 0
    total_career_months = 0
    company_set = set()

    for job in career:
        company_lower = job.get("company", "").lower().strip()
        dur = job.get("duration_months", 0)
        total_career_months += dur
        company_set.add(company_lower)

        if company_lower in CONSULTING_FIRMS:
            consulting_months += dur
        elif company_lower in PRODUCT_COMPANIES:
            product_months += dur

    features["product_company_ratio"] = product_months / max(total_career_months, 1)
    features["consulting_ratio"] = consulting_months / max(total_career_months, 1)

    # Consulting-only career flag (explicit disqualifier in JD)
    all_consulting = all(
        job.get("company", "").lower().strip() in CONSULTING_FIRMS
        for job in career
    ) if career else False
    features["consulting_only_flag"] = 1.0 if all_consulting else 0.0

    # Job-hop score: average tenure (penalize frequent switching)
    if career:
        tenures = [job.get("duration_months", 0) for job in career]
        avg_tenure = sum(tenures) / len(tenures)
        # Normalize: 36+ months is ideal (3+ years per JD)
        features["avg_tenure_months"] = avg_tenure
        features["job_stability_score"] = min(avg_tenure / 36.0, 1.0)
    else:
        features["avg_tenure_months"] = 0
        features["job_stability_score"] = 0.0

    # Number of companies
    features["num_companies"] = len(company_set)

    # Production ML signals in career descriptions
    prod_keywords = {
        "production", "deployed", "shipped", "real users", "live",
        "scale", "million", "serving", "api", "endpoint",
        "pipeline", "infrastructure", "monitoring", "latency",
        "a/b test", "evaluation", "metric", "ndcg", "mrr",
    }

    ml_keywords = {
        "machine learning", "deep learning", "model", "training",
        "embedding", "vector", "nlp", "ranking", "search",
        "retrieval", "recommendation", "transformer", "bert",
        "neural", "classification", "prediction",
    }

    has_prod_ml = False
    prod_signal_count = 0
    ml_signal_count = 0

    for job in career:
        desc = job.get("description", "").lower()
        has_prod = any(kw in desc for kw in prod_keywords)
        has_ml = any(kw in desc for kw in ml_keywords)

        if has_prod:
            prod_signal_count += 1
        if has_ml:
            ml_signal_count += 1
        if has_prod and has_ml:
            has_prod_ml = True

    features["has_production_ml"] = 1.0 if has_prod_ml else 0.0
    features["prod_signal_count"] = min(prod_signal_count / 3.0, 1.0)
    features["ml_signal_count"] = min(ml_signal_count / 3.0, 1.0)

    # Industry relevance
    industries = set()
    for job in career:
        ind = job.get("industry", "").lower()
        industries.add(ind)

    tech_industries = {"software", "technology", "it services", "internet",
                       "artificial intelligence", "saas", "data analytics"}
    tech_industry_count = sum(1 for ind in industries if any(t in ind for t in tech_industries))
    features["tech_industry_ratio"] = tech_industry_count / max(len(industries), 1)

    return features


def _get_title_relevance(title: str) -> float:
    """Get relevance score for a job title, with fuzzy matching."""
    title = title.lower().strip()

    # Direct match
    if title in TITLE_RELEVANCE:
        return TITLE_RELEVANCE[title]

    # Fuzzy: check if title contains any known title
    best_score = 0.0
    for known_title, score in TITLE_RELEVANCE.items():
        if known_title in title or title in known_title:
            best_score = max(best_score, score)

    # Check for AI/ML keywords in unknown titles
    if best_score == 0.0:
        ai_title_keywords = {"ai", "ml", "machine learning", "deep learning",
                             "nlp", "data science", "artificial intelligence"}
        if any(kw in title for kw in ai_title_keywords):
            best_score = 0.70

        eng_keywords = {"engineer", "developer", "architect", "scientist"}
        if any(kw in title for kw in eng_keywords):
            best_score = max(best_score, 0.40)

    return best_score


# =============================================================================
# Education Features (Weight: 0.05)
# =============================================================================

def _extract_education_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Extract education-related features."""
    features = {}
    education = candidate.get("education", [])

    if not education:
        features["education_relevance"] = 0.0
        features["education_tier"] = 0.0
        features["degree_level"] = 0.0
        return features

    best_field_score = 0.0
    best_tier_score = 0.0
    best_degree_score = 0.0

    for edu in education:
        # Field relevance
        field = edu.get("field_of_study", "").lower().strip()
        field_score = EDUCATION_FIELD_RELEVANCE.get(field, 0.20)  # Default 0.2 for unknown
        best_field_score = max(best_field_score, field_score)

        # Institution tier
        tier = edu.get("tier", "unknown")
        tier_score = EDUCATION_TIER_SCORES.get(tier, 0.35)
        best_tier_score = max(best_tier_score, tier_score)

        # Degree level
        degree = edu.get("degree", "").lower().strip()
        degree_score = 0.30  # Default for unknown degrees
        for deg_pattern, score in DEGREE_LEVEL_SCORES.items():
            if deg_pattern in degree:
                degree_score = max(degree_score, score)
        best_degree_score = max(best_degree_score, degree_score)

    features["education_relevance"] = best_field_score
    features["education_tier"] = best_tier_score
    features["degree_level"] = best_degree_score

    return features


# =============================================================================
# Behavioral / Redrob Signal Features (Weight: 0.10)
# =============================================================================

def _extract_behavioral_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Extract behavioral features from Redrob signals."""
    features = {}
    signals = candidate.get("redrob_signals", {})

    # Profile completeness
    features["profile_completeness"] = signals.get("profile_completeness_score", 0) / 100

    # Recency (days since last active)
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            last_date = date.fromisoformat(last_active)
            days_inactive = (REFERENCE_DATE - last_date).days
            features["days_inactive"] = days_inactive
            # Normalize: 0-30 days = 1.0, 30-90 = 0.7-1.0, 90-180 = 0.4-0.7, 180+ = 0-0.4
            if days_inactive <= 30:
                features["recency_score"] = 1.0
            elif days_inactive <= 90:
                features["recency_score"] = 1.0 - (days_inactive - 30) * 0.005
            elif days_inactive <= 180:
                features["recency_score"] = 0.7 - (days_inactive - 90) * 0.003
            else:
                features["recency_score"] = max(0.0, 0.4 - (days_inactive - 180) * 0.002)
        except (ValueError, TypeError):
            features["days_inactive"] = 999
            features["recency_score"] = 0.0
    else:
        features["days_inactive"] = 999
        features["recency_score"] = 0.0

    # Open to work
    features["open_to_work"] = 1.0 if signals.get("open_to_work_flag", False) else 0.0

    # Recruiter response rate (very important per JD)
    features["recruiter_response_rate"] = signals.get("recruiter_response_rate", 0)

    # Response time (lower is better)
    avg_response = signals.get("avg_response_time_hours", 999)
    features["response_time_score"] = max(0.0, 1.0 - avg_response / 200)

    # Interview completion rate
    features["interview_completion_rate"] = signals.get("interview_completion_rate", 0)

    # Offer acceptance rate
    oar = signals.get("offer_acceptance_rate", -1)
    features["offer_acceptance_rate"] = oar if oar >= 0 else 0.5  # Default 0.5 for no history

    # GitHub activity
    github = signals.get("github_activity_score", -1)
    features["github_activity"] = github / 100 if github >= 0 else 0.0
    features["has_github"] = 1.0 if github >= 0 else 0.0

    # Social proof signals
    features["search_appearances"] = min(signals.get("search_appearance_30d", 0) / 200, 1.0)
    features["saved_by_recruiters"] = min(signals.get("saved_by_recruiters_30d", 0) / 20, 1.0)
    features["profile_views"] = min(signals.get("profile_views_received_30d", 0) / 50, 1.0)

    # Notice period (JD: sub-30 ideal, 30+ higher bar)
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        features["notice_period_score"] = 1.0
    elif notice <= 60:
        features["notice_period_score"] = 0.7
    elif notice <= 90:
        features["notice_period_score"] = 0.5
    else:
        features["notice_period_score"] = 0.3

    # Verification signals
    verified_count = sum([
        1 if signals.get("verified_email", False) else 0,
        1 if signals.get("verified_phone", False) else 0,
        1 if signals.get("linkedin_connected", False) else 0,
    ])
    features["verification_score"] = verified_count / 3.0

    # Connection count (normalized)
    features["connection_score"] = min(signals.get("connection_count", 0) / 500, 1.0)

    return features


# =============================================================================
# Location Features (Weight: 0.05)
# =============================================================================

def _extract_location_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Extract location-related features."""
    features = {}
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    country = profile.get("country", "").lower().strip()
    location = profile.get("location", "").lower().strip()

    # India-based (preferred)
    features["india_based"] = 1.0 if country == "india" else 0.0

    # Preferred city
    in_preferred = any(city in location for city in PREFERRED_LOCATIONS)
    features["preferred_city"] = 1.0 if in_preferred else 0.0

    # Willing to relocate
    features["willing_to_relocate"] = 1.0 if signals.get("willing_to_relocate", False) else 0.0

    # Work mode fit (JD says hybrid/flexible)
    work_mode = signals.get("preferred_work_mode", "")
    mode_scores = {"hybrid": 1.0, "flexible": 1.0, "onsite": 0.7, "remote": 0.5}
    features["work_mode_fit"] = mode_scores.get(work_mode, 0.5)

    # Combined location score
    if features["india_based"]:
        if features["preferred_city"]:
            features["location_composite"] = 1.0
        elif features["willing_to_relocate"]:
            features["location_composite"] = 0.8
        else:
            features["location_composite"] = 0.6
    else:
        if features["willing_to_relocate"]:
            features["location_composite"] = 0.3
        else:
            features["location_composite"] = 0.15

    return features


# =============================================================================
# Certification Features (Weight: 0.02)
# =============================================================================

def _extract_certification_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Extract certification-related features."""
    features = {}
    certs = candidate.get("certifications", [])

    if not certs:
        features["has_certifications"] = 0.0
        features["relevant_cert_count"] = 0.0
        return features

    features["has_certifications"] = 1.0

    # Check for relevant certifications
    relevant_keywords = {
        "aws", "gcp", "azure", "cloud", "machine learning",
        "deep learning", "data", "python", "tensorflow",
        "kubernetes", "docker", "ai", "ml",
    }

    relevant_count = 0
    for cert in certs:
        cert_name = cert.get("name", "").lower()
        if any(kw in cert_name for kw in relevant_keywords):
            relevant_count += 1

    features["relevant_cert_count"] = min(relevant_count / 3.0, 1.0)

    return features
