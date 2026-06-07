"""
RecruitX — Candidate Loader
============================
Efficiently loads and parses 100K candidates from JSONL format.
Handles both raw .jsonl and gzipped .jsonl.gz files.
"""

import json
import gzip
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from tqdm import tqdm

from src.config import CANDIDATES_FILE


def load_candidates(
    filepath: Optional[Path] = None,
    max_candidates: Optional[int] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load all candidates from a JSONL file.

    Args:
        filepath: Path to the candidates file (.jsonl or .jsonl.gz).
                  Defaults to CANDIDATES_FILE from config.
        max_candidates: Optional limit on number of candidates to load.
        show_progress: Whether to show a progress bar.

    Returns:
        List of candidate dictionaries.
    """
    filepath = filepath or CANDIDATES_FILE

    if not filepath.exists():
        raise FileNotFoundError(f"Candidates file not found: {filepath}")

    candidates = []
    open_func = gzip.open if str(filepath).endswith('.gz') else open
    open_kwargs = {"mode": "rt", "encoding": "utf-8"}

    with open_func(filepath, **open_kwargs) as f:
        lines = f if not show_progress else tqdm(f, desc="Loading candidates", unit=" candidates")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            candidates.append(candidate)
            if max_candidates and len(candidates) >= max_candidates:
                break

    return candidates


def stream_candidates(
    filepath: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Stream candidates one at a time (memory-efficient for large datasets).

    Yields:
        Individual candidate dictionaries.
    """
    filepath = filepath or CANDIDATES_FILE

    if not filepath.exists():
        raise FileNotFoundError(f"Candidates file not found: {filepath}")

    open_func = gzip.open if str(filepath).endswith('.gz') else open

    with open_func(filepath, mode="rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_candidate_id(candidate: Dict[str, Any]) -> str:
    """Extract the candidate_id from a candidate dict."""
    return candidate["candidate_id"]


def extract_profile(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the profile section from a candidate dict."""
    return candidate.get("profile", {})


def extract_skills(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the skills list from a candidate dict."""
    return candidate.get("skills", [])


def extract_skill_names(candidate: Dict[str, Any]) -> List[str]:
    """Extract just the skill names from a candidate dict."""
    return [s["name"] for s in candidate.get("skills", [])]


def extract_career_history(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the career history list from a candidate dict."""
    return candidate.get("career_history", [])


def extract_education(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the education list from a candidate dict."""
    return candidate.get("education", [])


def extract_certifications(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the certifications list from a candidate dict."""
    return candidate.get("certifications", [])


def extract_signals(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the redrob_signals from a candidate dict."""
    return candidate.get("redrob_signals", {})


def get_candidate_summary(candidate: Dict[str, Any]) -> str:
    """Get a one-line summary of a candidate for logging/debugging."""
    profile = extract_profile(candidate)
    cid = extract_candidate_id(candidate)
    name = profile.get("anonymized_name", "Unknown")
    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    return f"{cid} | {name} | {title} | {yoe}yrs"
