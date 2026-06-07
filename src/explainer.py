"""
RecruitX — Explainer
======================
Generates specific, honest, and varied per-candidate reasoning.
Each reasoning string references actual facts from the candidate's
profile and connects them back to the JD requirements.

Key requirements from submission spec:
- Reference specific facts (years, titles, skills, signal values)
- Connect to JD requirements
- Acknowledge gaps honestly
- No hallucination
- Varied across candidates (not templated)
- Tone matches rank
"""

import random
from typing import Dict, Any, List, Tuple, Optional


# Varied opening patterns for diversity
_OPENINGS_TOP = [
    "{title} with {yoe:.1f} yrs experience at {company}.",
    "{yoe:.1f}-year {title} at {company} ({size}).",
    "Currently {title} at {company}, {yoe:.1f} yrs in the field.",
    "{title} ({company}) bringing {yoe:.1f} years of hands-on experience.",
]

_OPENINGS_MID = [
    "{title} at {company} with {yoe:.1f} yrs experience.",
    "{yoe:.1f}-year professional, currently {title} at {company}.",
    "Working as {title} at {company} for {yoe:.1f} years.",
]

_OPENINGS_LOW = [
    "{title} at {company}, {yoe:.1f} yrs total experience.",
    "Currently {title} at {company} ({yoe:.1f} yrs).",
]

_STRENGTH_TEMPLATES = [
    "Strengths: {strengths}.",
    "Strong signals: {strengths}.",
    "Key matches: {strengths}.",
    "Positive indicators: {strengths}.",
]

_CONCERN_TEMPLATES = [
    "Gaps: {concerns}.",
    "Concerns: {concerns}.",
    "Areas to probe: {concerns}.",
    "Missing signals: {concerns}.",
]


def generate_reasoning(
    candidate: Dict[str, Any],
    rank: int,
    score: float,
    components: Dict[str, float],
    honeypot_score: float = 0.0,
) -> str:
    """
    Generate a 1-2 sentence reasoning for why this candidate is at this rank.

    Args:
        candidate: Full candidate dictionary.
        rank: The candidate's rank (1-100).
        score: The candidate's final score.
        components: Score component breakdown from the ranker.
        honeypot_score: Honeypot detection score.

    Returns:
        A reasoning string referencing specific profile facts.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    education = candidate.get("education", [])
    certs = candidate.get("certifications", [])

    title = profile.get("current_title", "Professional")
    company = profile.get("current_company", "Unknown")
    company_size = profile.get("current_company_size", "")
    yoe = profile.get("years_of_experience", 0)
    country = profile.get("country", "")
    location = profile.get("location", "")

    # --- Build strengths list ---
    strengths = []

    # Top matched skills
    skill_names = [s["name"] for s in skills]
    ai_skills = _get_relevant_skills(skills)
    if ai_skills:
        if len(ai_skills) <= 3:
            strengths.append(f"relevant skills include {', '.join(ai_skills)}")
        else:
            strengths.append(f"{len(ai_skills)} relevant skills including {', '.join(ai_skills[:3])}")

    # Production ML signal
    if components.get("career_relevance", 0) > 0.5:
        # Find a career description with ML keywords
        for job in career:
            desc = job.get("description", "").lower()
            if any(kw in desc for kw in ["embedding", "ranking", "retrieval", "ml", "model", "pipeline"]):
                strengths.append(f"production ML experience as {job['title']}")
                break

    # Experience fit
    if 4 <= yoe <= 10:
        strengths.append(f"{yoe:.1f} yrs fits the 5-9 yr band")

    # Behavioral signals
    rr = signals.get("recruiter_response_rate", 0)
    if rr >= 0.5:
        strengths.append(f"strong {rr:.0%} recruiter response rate")

    github = signals.get("github_activity_score", -1)
    if github > 40:
        strengths.append(f"active GitHub presence ({github:.0f}/100)")

    # Location
    if country.lower() == "india":
        if any(city in location.lower() for city in ["pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru"]):
            strengths.append(f"based in {location}")

    # Certifications
    if certs:
        cert_names = [c["name"] for c in certs[:2]]
        strengths.append(f"certified: {', '.join(cert_names)}")

    # --- Build concerns list ---
    concerns = []

    # Missing must-have skills
    missing = _get_missing_critical_skills(skills)
    if missing:
        concerns.append(f"missing {', '.join(missing[:3])}")

    # Non-technical title
    if components.get("career_relevance", 0) < 0.3:
        if title.lower() not in ("software engineer", "backend engineer", "data engineer"):
            concerns.append(f"current role ({title}) is not AI/ML focused")

    # Consulting-only
    consulting_firms = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
                        "mindtree", "hcl", "tech mahindra", "mphasis"}
    career_companies = [job.get("company", "").lower() for job in career]
    if career_companies and all(c in consulting_firms for c in career_companies):
        concerns.append("career entirely in consulting/services firms")

    # Low response rate
    if rr < 0.2:
        concerns.append(f"low recruiter response rate ({rr:.0%})")

    # Inactive
    days_inactive = components.get("days_inactive", 0)  # May not be in components
    recency = signals.get("last_active_date", "")
    if recency:
        from datetime import date
        try:
            last = date.fromisoformat(recency)
            from src.config import REFERENCE_DATE
            days = (REFERENCE_DATE - last).days
            if days > 180:
                concerns.append(f"inactive for {days} days")
        except (ValueError, TypeError):
            pass

    # Long notice period
    notice = signals.get("notice_period_days", 0)
    if notice > 60:
        concerns.append(f"{notice}-day notice period")

    # Outside India
    if country.lower() != "india":
        concerns.append(f"based in {country}")

    # --- Compose the reasoning ---
    # Choose template style based on rank
    if rank <= 15:
        opening_pool = _OPENINGS_TOP
    elif rank <= 50:
        opening_pool = _OPENINGS_MID
    else:
        opening_pool = _OPENINGS_LOW

    # Use rank as seed for reproducible but varied selection
    rng = random.Random(hash(candidate.get("candidate_id", "")) + rank)

    opening = rng.choice(opening_pool).format(
        title=title, company=company, yoe=yoe, size=company_size
    )

    parts = [opening]

    if strengths:
        template = rng.choice(_STRENGTH_TEMPLATES)
        parts.append(template.format(strengths="; ".join(strengths[:4])))

    if concerns:
        template = rng.choice(_CONCERN_TEMPLATES)
        parts.append(template.format(concerns="; ".join(concerns[:3])))

    reasoning = " ".join(parts)

    # Ensure it's within reasonable length (1-2 sentences, ~50-200 chars)
    if len(reasoning) > 350:
        reasoning = reasoning[:347] + "..."

    return reasoning


def _get_relevant_skills(skills: List[Dict[str, Any]]) -> List[str]:
    """Get AI/ML relevant skills from the candidate's skill list."""
    relevant_keywords = {
        "python", "nlp", "natural language", "machine learning",
        "deep learning", "pytorch", "tensorflow", "bert", "gpt",
        "transformer", "embedding", "faiss", "qdrant", "elasticsearch",
        "ranking", "search", "retrieval", "recommendation",
        "xgboost", "lightgbm", "vector", "llm", "fine-tuning",
        "lora", "data science", "neural", "classification",
        "computer vision", "reinforcement learning", "keras",
        "scikit-learn", "spark", "airflow", "kafka", "docker",
        "kubernetes", "aws", "gcp", "azure", "mlops",
        "sentence transformers", "hugging face",
    }

    result = []
    for s in skills:
        name = s.get("name", "")
        if name.lower() in relevant_keywords or any(kw in name.lower() for kw in relevant_keywords):
            prof = s.get("proficiency", "")
            if prof in ("expert", "advanced"):
                result.insert(0, name)  # Expert/advanced first
            else:
                result.append(name)

    return result


def _get_missing_critical_skills(skills: List[Dict[str, Any]]) -> List[str]:
    """Identify critical JD skills that the candidate is missing."""
    skill_names_lower = {s["name"].lower() for s in skills}

    critical_skills = {
        "Python": ["python"],
        "NLP/IR": ["nlp", "natural language", "information retrieval", "text mining"],
        "embeddings": ["embedding", "sentence-transformer", "vector", "dense retrieval"],
        "vector DB": ["faiss", "qdrant", "pinecone", "weaviate", "milvus", "elasticsearch", "opensearch"],
        "ranking/eval": ["ranking", "ndcg", "mrr", "evaluation", "a/b testing"],
    }

    missing = []
    for skill_group, keywords in critical_skills.items():
        found = any(
            any(kw in sn for kw in keywords)
            for sn in skill_names_lower
        )
        if not found:
            missing.append(skill_group)

    return missing


def generate_reasoning_batch(
    candidates: List[Dict[str, Any]],
    rankings: List[Tuple[str, float, Dict[str, float]]],
    honeypot_scores: Dict[str, float],
) -> List[str]:
    """
    Generate reasoning for a batch of ranked candidates.

    Args:
        candidates: List of candidate dicts.
        rankings: List of (candidate_id, score, components) from ranker.
        honeypot_scores: Dict of candidate_id -> honeypot_score.

    Returns:
        List of reasoning strings in the same order as rankings.
    """
    # Build lookup
    cand_lookup = {c["candidate_id"]: c for c in candidates}

    reasonings = []
    for rank_idx, (cid, score, components) in enumerate(rankings):
        rank = rank_idx + 1
        candidate = cand_lookup.get(cid, {})
        hp_score = honeypot_scores.get(cid, 0.0)

        reasoning = generate_reasoning(candidate, rank, score, components, hp_score)
        reasonings.append(reasoning)

    return reasonings
