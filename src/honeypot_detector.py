"""
RecruitX — Honeypot Detector
==============================
Detects ~80 honeypot candidates with subtly impossible profiles.
These are synthetic trap candidates that, if ranked highly, lead to disqualification.

Detection heuristics based on the redrob_signals_doc:
- "8 years of experience at a company founded 3 years ago"
- "Expert proficiency in 10 skills with 0 years used"
- Title-description mismatches (title says X, description describes Y)
"""

import re
from typing import Dict, Any, List, Tuple
from datetime import date

from src.config import (
    HONEYPOT_EXPERT_MIN_DURATION,
    HONEYPOT_MAX_SKILLS_WITH_ZERO_DUR,
    HONEYPOT_SCORE_THRESHOLD,
    REFERENCE_DATE,
    TITLE_DESCRIPTION_MISMATCH_KEYWORDS,
)


def detect_honeypot(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Analyze a candidate for honeypot signals.

    Returns:
        (honeypot_score, reasons): Score 0-1 where higher = more likely honeypot.
        Reasons list explains what triggered each signal.
    """
    signals = []
    score = 0.0

    # ---- Check 1: Expert skills with zero/very low duration ----
    skills = candidate.get("skills", [])
    expert_zero_dur = 0
    expert_low_dur = 0
    total_expert = 0

    for skill in skills:
        prof = skill.get("proficiency", "beginner")
        dur = skill.get("duration_months", 0)

        if prof == "expert":
            total_expert += 1
            if dur == 0:
                expert_zero_dur += 1
            elif dur < HONEYPOT_EXPERT_MIN_DURATION:
                expert_low_dur += 1

    if expert_zero_dur >= 3:
        sig = 0.3
        score += sig
        signals.append(f"Expert proficiency in {expert_zero_dur} skills with 0 months duration")

    if total_expert >= 8 and expert_low_dur >= 5:
        sig = 0.25
        score += sig
        signals.append(f"{total_expert} expert skills, {expert_low_dur} with <{HONEYPOT_EXPERT_MIN_DURATION}mo duration")

    # ---- Check 2: Many skills with 0 endorsements and 0 duration ----
    zero_dur_skills = sum(1 for s in skills if s.get("duration_months", 0) == 0)
    zero_endorse_skills = sum(1 for s in skills if s.get("endorsements", 0) == 0)

    if zero_dur_skills > HONEYPOT_MAX_SKILLS_WITH_ZERO_DUR and len(skills) > 5:
        ratio = zero_dur_skills / max(len(skills), 1)
        if ratio > 0.6:
            sig = 0.2
            score += sig
            signals.append(f"{zero_dur_skills}/{len(skills)} skills have 0 months duration")

    # ---- Check 3: Career duration impossible ----
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    for job in career:
        start = job.get("start_date", "")
        dur_months = job.get("duration_months", 0)
        is_current = job.get("is_current", False)

        if start:
            try:
                start_date = date.fromisoformat(start)
                # Check if claimed duration far exceeds actual elapsed time
                elapsed_months = (REFERENCE_DATE.year - start_date.year) * 12 + \
                                 (REFERENCE_DATE.month - start_date.month)

                if dur_months > elapsed_months + 3:  # 3 month tolerance
                    sig = 0.25
                    score += sig
                    signals.append(
                        f"Job at {job.get('company','?')}: claims {dur_months}mo "
                        f"but only {elapsed_months}mo since start date {start}"
                    )
            except (ValueError, TypeError):
                pass

    # ---- Check 4: Total claimed experience vs career timeline ----
    claimed_yoe = profile.get("years_of_experience", 0)
    if career:
        earliest_start = None
        for job in career:
            start = job.get("start_date", "")
            if start:
                try:
                    sd = date.fromisoformat(start)
                    if earliest_start is None or sd < earliest_start:
                        earliest_start = sd
                except (ValueError, TypeError):
                    pass

        if earliest_start:
            actual_span_years = (REFERENCE_DATE - earliest_start).days / 365.25
            if claimed_yoe > actual_span_years + 2:  # 2 year tolerance
                sig = 0.2
                score += sig
                signals.append(
                    f"Claims {claimed_yoe:.1f} yrs experience but career "
                    f"spans only {actual_span_years:.1f} years from earliest date"
                )

    # ---- Check 5: Title-description mismatch ----
    # The dataset has candidates whose title says one thing but description
    # describes a completely different role
    for job in career:
        title_lower = job.get("title", "").lower()
        desc = job.get("description", "").lower()

        for pattern_title, suspicious_keywords in TITLE_DESCRIPTION_MISMATCH_KEYWORDS.items():
            if pattern_title in title_lower:
                mismatches = [kw for kw in suspicious_keywords if kw.lower() in desc]
                if len(mismatches) >= 2:
                    sig = 0.15
                    score += sig
                    signals.append(
                        f"Title '{job.get('title')}' but description mentions "
                        f"unrelated terms: {', '.join(mismatches[:3])}"
                    )
                    break

    # ---- Check 6: Skill assessment score vs proficiency mismatch ----
    redrob = candidate.get("redrob_signals", {})
    assessments = redrob.get("skill_assessment_scores", {})

    if assessments:
        skill_prof_map = {s["name"]: s["proficiency"] for s in skills}
        bad_assessment_count = 0

        for skill_name, score_val in assessments.items():
            prof = skill_prof_map.get(skill_name, "beginner")
            # Expert claims but scores < 30 on assessment
            if prof in ("expert", "advanced") and score_val < 25:
                bad_assessment_count += 1

        if bad_assessment_count >= 3:
            sig = 0.2
            score += sig
            signals.append(
                f"{bad_assessment_count} skills: claims expert/advanced "
                f"but assessment score < 25"
            )

    # ---- Check 7: Impossibly high skill count with uniform proficiency ----
    if len(skills) >= 15:
        profs = [s.get("proficiency", "beginner") for s in skills]
        if profs.count("expert") >= 10 or profs.count("advanced") >= 12:
            sig = 0.2
            score += sig
            signals.append(
                f"{len(skills)} skills with {profs.count('expert')} expert "
                f"and {profs.count('advanced')} advanced — implausible breadth"
            )

    return min(score, 1.0), signals


def is_honeypot(candidate: Dict[str, Any], threshold: float = HONEYPOT_SCORE_THRESHOLD) -> bool:
    """Quick check if a candidate is a honeypot."""
    hp_score, _ = detect_honeypot(candidate)
    return hp_score >= threshold


def batch_detect_honeypots(
    candidates: List[Dict[str, Any]],
    threshold: float = HONEYPOT_SCORE_THRESHOLD,
) -> Dict[str, Tuple[float, List[str]]]:
    """
    Run honeypot detection on a batch of candidates.

    Returns:
        Dict mapping candidate_id -> (honeypot_score, reasons) for flagged candidates.
    """
    flagged = {}
    for candidate in candidates:
        hp_score, reasons = detect_honeypot(candidate)
        if hp_score >= threshold:
            cid = candidate["candidate_id"]
            flagged[cid] = (hp_score, reasons)
    return flagged
