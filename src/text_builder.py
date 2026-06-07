"""
RecruitX — Text Builder
========================
Builds rich searchable text representations from candidate profiles.
Used for both BM25 sparse retrieval and dense embedding generation.
"""

from typing import Dict, Any, List


def build_candidate_text(candidate: Dict[str, Any], mode: str = "full") -> str:
    """
    Build a searchable text representation of a candidate.

    Args:
        candidate: Full candidate dictionary.
        mode: "full" for embedding, "skills_focused" for skill-heavy search,
              "career" for career-description focused text.

    Returns:
        Concatenated text representation.
    """
    if mode == "full":
        return _build_full_text(candidate)
    elif mode == "skills_focused":
        return _build_skills_text(candidate)
    elif mode == "career":
        return _build_career_text(candidate)
    else:
        return _build_full_text(candidate)


def _build_full_text(candidate: Dict[str, Any]) -> str:
    """Build full text combining all candidate information."""
    parts = []

    # Profile headline and summary
    profile = candidate.get("profile", {})
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])
    if profile.get("current_title"):
        parts.append(f"Current role: {profile['current_title']} at {profile.get('current_company', '')}")
    if profile.get("current_industry"):
        parts.append(f"Industry: {profile['current_industry']}")

    # Career descriptions (rich source of domain info)
    for job in candidate.get("career_history", []):
        title = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "")
        parts.append(f"{title} at {company}: {desc}")

    # Skills with proficiency
    skills = candidate.get("skills", [])
    if skills:
        skill_strs = []
        for s in skills:
            name = s.get("name", "")
            prof = s.get("proficiency", "")
            skill_strs.append(f"{name} ({prof})")
        parts.append("Skills: " + ", ".join(skill_strs))

    # Education
    for edu in candidate.get("education", []):
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        inst = edu.get("institution", "")
        parts.append(f"Education: {degree} in {field} from {inst}")

    # Certifications
    for cert in candidate.get("certifications", []):
        parts.append(f"Certification: {cert.get('name', '')} by {cert.get('issuer', '')}")

    return " . ".join(parts)


def _build_skills_text(candidate: Dict[str, Any]) -> str:
    """Build text emphasizing skills for skill-focused retrieval."""
    parts = []

    profile = candidate.get("profile", {})
    if profile.get("headline"):
        parts.append(profile["headline"])

    # Repeat skill names based on proficiency weight
    proficiency_weight = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 1}
    for s in candidate.get("skills", []):
        name = s.get("name", "")
        prof = s.get("proficiency", "beginner")
        weight = proficiency_weight.get(prof, 1)
        parts.extend([name] * weight)

    # Certifications
    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))

    return " ".join(parts)


def _build_career_text(candidate: Dict[str, Any]) -> str:
    """Build text focusing on career descriptions."""
    parts = []

    profile = candidate.get("profile", {})
    if profile.get("summary"):
        parts.append(profile["summary"])

    for job in candidate.get("career_history", []):
        desc = job.get("description", "")
        if desc:
            parts.append(desc)

    return " . ".join(parts)


def build_candidate_texts_batch(
    candidates: List[Dict[str, Any]],
    mode: str = "full",
) -> List[str]:
    """Build text representations for a batch of candidates."""
    return [build_candidate_text(c, mode) for c in candidates]
