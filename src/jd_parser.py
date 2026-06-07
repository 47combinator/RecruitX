"""
RecruitX — JD Parser
====================
Parses the Job Description into structured requirements.
Since we have a single known JD, requirements are pre-extracted for reliability,
but the architecture supports dynamic JD parsing for future extensibility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
from src.config import (
    MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS,
    JD_TITLE, JD_COMPANY, JD_LOCATION,
    JD_EXPERIENCE_RANGE, JD_EMBEDDING_TEXT,
)


@dataclass
class JDRequirements:
    """Structured representation of parsed Job Description requirements."""

    title: str = JD_TITLE
    company: str = JD_COMPANY
    location: str = JD_LOCATION
    experience_min: float = JD_EXPERIENCE_RANGE[0]
    experience_max: float = JD_EXPERIENCE_RANGE[1]
    embedding_text: str = JD_EMBEDDING_TEXT

    # Skill categories
    must_have_skills: Dict[str, List[str]] = field(default_factory=lambda: dict(MUST_HAVE_SKILLS))
    nice_to_have_skills: Dict[str, List[str]] = field(default_factory=lambda: dict(NICE_TO_HAVE_SKILLS))

    # Explicit disqualifiers from the JD
    disqualifiers: List[str] = field(default_factory=lambda: [
        "Entire career at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)",
        "Pure CV/Speech/Robotics without NLP/IR exposure",
        "Pure research with no production deployment",
        "AI experience only from recent LangChain/OpenAI usage (<12 months)",
        "Title-chasers switching every 1.5 years without depth",
        "Senior roles that stopped writing code 18+ months ago",
    ])

    # Key domain areas the JD emphasizes
    domain_keywords: List[str] = field(default_factory=lambda: [
        "ranking", "retrieval", "matching", "search",
        "embeddings", "vector search", "hybrid search",
        "NLP", "information retrieval", "recommendation",
        "production ML", "deployed ML", "real users",
        "evaluation", "NDCG", "MRR", "A/B testing",
    ])

    # Production-oriented keywords (JD heavily emphasizes production experience)
    production_keywords: List[str] = field(default_factory=lambda: [
        "production", "deployed", "shipped", "real users",
        "scale", "large-scale", "million", "serving",
        "pipeline", "infrastructure", "monitoring",
        "latency", "throughput", "SLA", "uptime",
    ])


def get_all_skill_aliases(skills_dict: Dict[str, List[str]]) -> Set[str]:
    """Get a flat set of all skill aliases (lowercased) from a skills dictionary."""
    aliases = set()
    for canonical, alias_list in skills_dict.items():
        aliases.add(canonical.lower())
        for alias in alias_list:
            aliases.add(alias.lower())
    return aliases


def match_skills(candidate_skills: List[str], skills_dict: Dict[str, List[str]]) -> Dict[str, bool]:
    """
    Match candidate skills against a skill taxonomy.

    Returns a dict of canonical_skill_name -> matched (True/False).
    """
    candidate_lower = {s.lower() for s in candidate_skills}
    matches = {}

    for canonical, aliases in skills_dict.items():
        all_forms = [canonical.lower()] + [a.lower() for a in aliases]
        matched = any(
            any(form in cand_skill or cand_skill in form
                for form in all_forms)
            for cand_skill in candidate_lower
        )
        matches[canonical] = matched

    return matches


def parse_jd() -> JDRequirements:
    """
    Parse the Job Description and return structured requirements.

    For this hackathon, requirements are pre-extracted from the known JD.
    In a production system, this would use NLP/LLM to dynamically parse any JD.
    """
    return JDRequirements()
