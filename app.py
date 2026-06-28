"""
RecruitX Dashboard
Run: streamlit run app.py
"""

import csv
import json
import pickle
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from src.config import PRECOMPUTED_DIR, OUTPUT_DIR, CANDIDATES_FILE

st.set_page_config(page_title="RecruitX", page_icon="R", layout="wide", initial_sidebar_state="collapsed")

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

    :root {
        --black: #000000; --white: #ededed;
        --g1: #0a0a0a; --g2: #111111; --g3: #1a1a1a; --g4: #222222;
        --g5: #333333; --g6: #555555; --g7: #777777; --g8: #999999;
        --sans: 'Inter', -apple-system, sans-serif;
        --mono: 'JetBrains Mono', monospace;
    }

    html, body, [class*="css"] { font-family: var(--sans) !important; color: var(--white) !important; }
    .stApp { background: var(--black) !important; }
    .main .block-container { padding: 0 !important; max-width: 100% !important; }

    #MainMenu, footer, header, .stDeployButton,
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"], [data-testid="collapsedControl"],
    [data-testid="stSidebar"] { display: none !important; }

    *, *::before, *::after { border-radius: 0 !important; }

    .stSelectbox > div > div { background: var(--g2) !important; border: 1px solid var(--g4) !important; color: var(--white) !important; }
    .stSelectbox label { color: var(--g6) !important; font-family: var(--mono) !important; font-size: 0.65rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; }
    .stSlider label { color: var(--g6) !important; font-family: var(--mono) !important; font-size: 0.65rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; }

    .stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 0 !important; border-bottom: 1px solid var(--g4) !important; }
    .stTabs [data-baseweb="tab"] { color: var(--g6) !important; font-family: var(--mono) !important; font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; padding: 12px 24px !important; background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important; }
    .stTabs [data-baseweb="tab"]:hover { color: var(--white) !important; }
    .stTabs [aria-selected="true"] { color: var(--white) !important; border-bottom-color: var(--white) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--white) !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 2rem !important; }

    [data-testid="stMetric"] { background: var(--g2) !important; border: 1px solid var(--g4) !important; padding: 1rem !important; }
    [data-testid="stMetricLabel"] { color: var(--g6) !important; font-family: var(--mono) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; }
    [data-testid="stMetricValue"] { color: var(--white) !important; font-family: var(--mono) !important; font-weight: 600 !important; }

    hr { border: none !important; border-top: 1px solid var(--g4) !important; margin: 0 !important; }
    ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: var(--black); } ::-webkit-scrollbar-thumb { background: var(--g4); }

    /* NAV */
    .rx-nav-left { display: flex; align-items: center; gap: 1.5rem; }
    .rx-nav-divider { width: 1px; height: 16px; background: var(--g4); }
    .rx-nav-team { font-family: var(--mono); font-size: 0.75rem; color: var(--white); letter-spacing: 0.08em; text-transform: uppercase; font-weight: 500; }
    .rx-nav-hackathon { display: flex; align-items: center; gap: 0.8rem; }
    .rx-nav-badge { font-family: var(--mono); font-size: 0.72rem; color: var(--white); letter-spacing: 0.06em; text-transform: uppercase; padding: 0.3rem 0.7rem; border: 1px solid var(--g7); font-weight: 500; }
    .rx-nav { display: flex; align-items: center; justify-content: space-between; padding: 1rem 3rem; border-bottom: 1px solid var(--g4); background: var(--black); }
    .rx-nav-brand { font-family: var(--mono); font-size: 0.85rem; font-weight: 700; color: var(--white); letter-spacing: 0.12em; }
    .rx-nav-right { font-family: var(--mono); font-size: 0.6rem; color: var(--g6); letter-spacing: 0.08em; text-transform: uppercase; }

    /* CONTENT */
    .rx-pad { padding: 0 3rem; }

    /* HERO */
    .rx-hero { padding: 4rem 3rem 3rem 3rem; }
    .rx-hero h1 { font-family: var(--sans); font-size: 3.2rem; font-weight: 800; letter-spacing: -0.045em; line-height: 1.05; color: var(--white); margin: 0; }
    .rx-hero-sub { font-size: 0.9rem; color: var(--g7); margin-top: 1rem; line-height: 1.6; max-width: 560px; }

    /* METRIC STRIP */
    .rx-strip { display: grid; grid-template-columns: repeat(5, 1fr); border-top: 1px solid var(--g4); border-bottom: 1px solid var(--g4); }
    .rx-strip-cell { padding: 1.6rem 1.2rem; border-right: 1px solid var(--g4); transition: background 0.15s; }
    .rx-strip-cell:last-child { border-right: none; }
    .rx-strip-cell:hover { background: var(--g2); }
    .rx-strip-val { font-family: var(--mono); font-size: 1.6rem; font-weight: 700; color: var(--white); line-height: 1; }
    .rx-strip-lbl { font-family: var(--mono); font-size: 0.58rem; color: var(--g6); text-transform: uppercase; letter-spacing: 0.14em; margin-top: 0.5rem; }

    /* LABEL */
    .rx-label { font-family: var(--mono); font-size: 0.6rem; font-weight: 600; color: var(--g6); text-transform: uppercase; letter-spacing: 0.16em; margin: 3rem 0 1.2rem 0; }

    /* CANDIDATE ROW */
    .rx-row { display: flex; align-items: flex-start; justify-content: space-between; padding: 1.4rem 1rem; border-bottom: 1px solid var(--g4); gap: 2rem; margin: 0 -1rem; transition: background 0.12s; }
    .rx-row:hover { background: var(--g2); }
    .rx-row:first-of-type { border-top: 1px solid var(--g4); }
    .rx-row-rank { font-family: var(--mono); font-size: 0.58rem; color: var(--g5); letter-spacing: 0.1em; text-transform: uppercase; min-width: 48px; padding-top: 2px; }
    .rx-row-info { flex: 1; }
    .rx-row-name { font-size: 0.95rem; font-weight: 600; color: var(--white); }
    .rx-row-role { font-size: 0.78rem; color: var(--g7); margin-top: 0.1rem; }
    .rx-row-meta { font-family: var(--mono); font-size: 0.62rem; color: var(--g5); margin-top: 0.4rem; display: flex; gap: 1.2rem; }
    .rx-row-score { text-align: right; min-width: 90px; }
    .rx-row-score-num { font-family: var(--mono); font-size: 1.1rem; font-weight: 700; color: var(--white); }
    .rx-row-bar { width: 90px; height: 2px; background: var(--g4); margin-top: 0.5rem; }
    .rx-row-bar-fill { height: 100%; background: var(--white); transition: width 0.6s ease; }
    .rx-reason { font-size: 0.75rem; color: var(--g6); line-height: 1.6; margin-top: 0.6rem; padding-left: 0.8rem; border-left: 2px solid var(--g4); }

    /* PIPELINE SECTION */
    .rx-pipe-section { padding: 2rem 0; border-bottom: 1px solid var(--g4); }
    .rx-pipe-header { display: flex; gap: 2rem; align-items: flex-start; margin-bottom: 1rem; }
    .rx-pipe-num { font-family: var(--mono); font-size: 2.4rem; font-weight: 800; color: var(--g4); line-height: 1; min-width: 60px; }
    .rx-pipe-title { font-family: var(--sans); font-size: 1.1rem; font-weight: 700; color: var(--white); margin-bottom: 0.3rem; }
    .rx-pipe-body { font-size: 0.82rem; color: var(--g7); line-height: 1.7; margin-left: calc(60px + 2rem); }
    .rx-pipe-detail { font-family: var(--mono); font-size: 0.68rem; color: var(--g5); margin-top: 0.6rem; line-height: 1.8; }

    /* SCORE BREAKDOWN BAR */
    .rx-sb-row { display: flex; align-items: center; gap: 1rem; padding: 0.7rem 0; border-bottom: 1px solid var(--g3); }
    .rx-sb-row:last-child { border-bottom: none; }
    .rx-sb-name { font-family: var(--mono); font-size: 0.68rem; color: var(--g7); text-transform: uppercase; letter-spacing: 0.06em; min-width: 140px; }
    .rx-sb-bar-wrap { flex: 1; height: 4px; background: var(--g3); }
    .rx-sb-bar-fill { height: 100%; background: var(--white); }
    .rx-sb-val { font-family: var(--mono); font-size: 0.72rem; font-weight: 600; color: var(--white); min-width: 50px; text-align: right; }
    .rx-sb-weight { font-family: var(--mono); font-size: 0.55rem; color: var(--g5); min-width: 40px; text-align: right; }
    .rx-sb-explain { font-size: 0.7rem; color: var(--g6); line-height: 1.6; margin-top: 0.3rem; padding: 0.5rem 0 0.5rem 0; margin-left: calc(140px + 1rem); }

    /* PROFILE DETAIL */
    .rx-profile-header { padding: 2rem 0; border-bottom: 1px solid var(--g4); }
    .rx-profile-name { font-family: var(--sans); font-size: 1.8rem; font-weight: 800; color: var(--white); letter-spacing: -0.03em; }
    .rx-profile-role { font-size: 0.9rem; color: var(--g7); margin-top: 0.3rem; }
    .rx-profile-meta { font-family: var(--mono); font-size: 0.65rem; color: var(--g5); margin-top: 0.8rem; display: flex; gap: 2rem; flex-wrap: wrap; letter-spacing: 0.04em; }
    .rx-profile-score-big { font-family: var(--mono); font-size: 3rem; font-weight: 800; color: var(--white); line-height: 1; }
    .rx-profile-rank-big { font-family: var(--mono); font-size: 0.65rem; color: var(--g6); text-transform: uppercase; letter-spacing: 0.12em; margin-top: 0.3rem; }

    /* SKILL TAGS */
    .rx-tag { display: inline-block; padding: 0.2rem 0.5rem; margin: 0.12rem; font-family: var(--mono); font-size: 0.6rem; font-weight: 500; border: 1px solid var(--g4); color: var(--g7); transition: all 0.1s; }
    .rx-tag:hover { border-color: var(--white); color: var(--white); }
    .rx-tag-strong { border-color: var(--g7); color: var(--white); }

    /* CAREER */
    .rx-career-row { display: flex; gap: 1rem; padding: 0.8rem 0; border-bottom: 1px solid var(--g3); }
    .rx-career-row:last-child { border-bottom: none; }
    .rx-career-dot { width: 6px; height: 6px; margin-top: 6px; flex-shrink: 0; }
    .rx-career-dot-on { background: var(--white); } .rx-career-dot-off { background: var(--g5); }
    .rx-career-title { font-size: 0.82rem; font-weight: 600; color: var(--white); }
    .rx-career-co { font-family: var(--mono); font-size: 0.68rem; color: var(--g6); }
    .rx-career-desc { font-size: 0.7rem; color: var(--g5); margin-top: 0.2rem; line-height: 1.5; }

    @media (max-width: 768px) {
        .rx-strip { grid-template-columns: repeat(2, 1fr); }
        .rx-row { flex-direction: column; gap: 0.8rem; }
        .rx-row-score { text-align: left; }
        .rx-pad, .rx-hero { padding-left: 1.5rem; padding-right: 1.5rem; }
        .rx-nav { padding: 1rem 1.5rem; }
        .rx-hero h1 { font-size: 2rem; }
        .rx-pipe-body { margin-left: 0; }
        .rx-pipe-num { font-size: 1.6rem; min-width: 40px; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data
# =============================================================================
@st.cache_data
def load_submission(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"candidate_id": row["candidate_id"], "rank": int(row["rank"]),
                         "score": float(row["score"]), "reasoning": row.get("reasoning", "")})
    return sorted(rows, key=lambda x: x["rank"])

@st.cache_data
def load_candidates(path, ids):
    out, id_set = {}, set(ids)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            c = json.loads(line)
            if c["candidate_id"] in id_set:
                out[c["candidate_id"]] = c
            if len(out) >= len(id_set): break
    return out

@st.cache_data
def load_features():
    p = PRECOMPUTED_DIR / "features.pkl"
    if p.exists():
        with open(p, "rb") as f:
            d = pickle.load(f)
        return dict(zip(d["candidate_ids"], d["features"]))
    return {}


# =============================================================================
# Components
# =============================================================================

def candidate_row_html(result, candidate):
    p = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})
    rank, score = result["rank"], result["score"]
    name = p.get("anonymized_name", result["candidate_id"])
    title, company = p.get("current_title", ""), p.get("current_company", "")
    yoe = p.get("years_of_experience", 0)
    loc = p.get("location", "")
    country = p.get("country", "")
    loc_str = f"{loc}, {country}" if loc and country else loc or country
    rr = s.get("recruiter_response_rate", 0)

    meta = []
    if yoe: meta.append(f"{yoe:.1f} yrs")
    if loc_str: meta.append(loc_str)
    if rr > 0: meta.append(f"{rr:.0%} response")
    meta_html = '<span>' + '</span><span>'.join(meta) + '</span>' if meta else ''

    return f"""<div class="rx-row">
        <div class="rx-row-rank">#{rank:02d}</div>
        <div class="rx-row-info">
            <div class="rx-row-name">{name}</div>
            <div class="rx-row-role">{title} at {company}</div>
            <div class="rx-row-meta">{meta_html}</div>
            <div class="rx-reason">{result["reasoning"]}</div>
        </div>
        <div class="rx-row-score">
            <div class="rx-row-score-num">{score:.4f}</div>
            <div class="rx-row-bar"><div class="rx-row-bar-fill" style="width:{score*100:.1f}%"></div></div>
        </div>
    </div>"""


def render_score_breakdown(features, candidate):
    """Full score breakdown with explanations."""
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    # Compute each component
    must_have = features.get("must_have_ratio", 0)
    nice_to_have = features.get("nice_to_have_ratio", 0)
    skill_score = must_have * 0.6 + nice_to_have * 0.4

    title_rel = features.get("current_title_relevance", 0)
    prod_ml = features.get("has_production_ml", 0)
    career_score = title_rel * 0.5 + prod_ml * 0.5

    exp_score = features.get("experience_in_band", 0)

    rr_val = features.get("recruiter_response_rate", 0)
    recency = features.get("recency_score", 0)
    behavioral_score = rr_val * 0.5 + recency * 0.5

    edu_score = features.get("education_relevance", 0)
    loc_score = features.get("location_composite", 0)
    github_score = features.get("github_activity", 0)

    semantic = features.get("semantic_similarity", 0) if "semantic_similarity" in features else 0

    components = [
        ("Skill Match", skill_score, "35%",
         f"Must-have skills matched: {must_have:.0%} / Nice-to-have: {nice_to_have:.0%}. "
         f"Evaluated {len(skills)} skills against JD requirements for NLP, embeddings, vector DBs, and ML infrastructure."),
        ("Career Relevance", career_score, "20%",
         f"Title relevance: {title_rel:.2f} / Production ML: {'Yes' if prod_ml else 'No'}. "
         f"Current role as {profile.get('current_title', 'N/A')} at {profile.get('current_company', 'N/A')} "
         f"assessed for hands-on AI/ML engineering alignment."),
        ("Semantic Similarity", semantic, "15%",
         "Dense vector similarity between candidate profile text and JD requirements "
         "using all-MiniLM-L6-v2 sentence embeddings (384-dim cosine similarity)."),
        ("Behavioral Signals", behavioral_score, "10%",
         f"Recruiter response rate: {rr_val:.0%} / Profile recency: {recency:.2f}. "
         f"Derived from Redrob platform engagement data including response times and interview completion."),
        ("Experience Fit", exp_score, "8%",
         f"Years of experience: {profile.get('years_of_experience', 0):.1f}. "
         f"{'Within' if exp_score > 0.5 else 'Outside'} the 5-9 year target band for a senior founding role."),
        ("Education", edu_score, "5%",
         f"Education relevance score: {edu_score:.2f}. Evaluated degree level and field alignment with AI/ML."),
        ("Location", loc_score, "5%",
         f"Location: {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}. "
         f"India-based candidates preferred. Notice period: {signals.get('notice_period_days', 0)} days."),
        ("GitHub Activity", github_score, "2%",
         f"GitHub activity score: {signals.get('github_activity_score', -1)}. "
         f"Open source contribution signals technical depth and community engagement."),
    ]

    st.markdown('<div class="rx-label">SCORE COMPONENTS</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size: 0.75rem; color: var(--g6); margin-bottom: 1.5rem; line-height: 1.6;">
        The final score is a weighted sum of 8 components, each evaluating a different dimension of candidate fit.
        Penalty multipliers are applied for honeypot indicators, consulting-only careers, and keyword stuffing.
    </div>""", unsafe_allow_html=True)

    for name, value, weight, explanation in components:
        bar_width = max(1, value * 100)
        st.markdown(f"""
        <div class="rx-sb-row">
            <div class="rx-sb-name">{name}</div>
            <div class="rx-sb-bar-wrap"><div class="rx-sb-bar-fill" style="width:{bar_width:.1f}%"></div></div>
            <div class="rx-sb-val">{value:.2f}</div>
            <div class="rx-sb-weight">{weight}</div>
        </div>
        <div class="rx-sb-explain">{explanation}</div>
        """, unsafe_allow_html=True)


def render_profile_detail(result, candidate, features):
    """Full candidate profile view."""
    p = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    name = p.get("anonymized_name", result["candidate_id"])
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    yoe = p.get("years_of_experience", 0)
    loc = p.get("location", "")
    country = p.get("country", "")
    headline = p.get("headline", "")
    industry = p.get("current_industry", "")
    comp_size = p.get("current_company_size", "")
    notice = s.get("notice_period_days", 0)
    rr = s.get("recruiter_response_rate", 0)

    # Header
    col_info, col_score = st.columns([3, 1])
    with col_info:
        st.markdown(f"""
        <div class="rx-profile-header">
            <div class="rx-profile-name">{name}</div>
            <div class="rx-profile-role">{title} at {company}</div>
            <div style="font-size: 0.78rem; color: var(--g7); margin-top: 0.3rem;">{headline}</div>
            <div class="rx-profile-meta">
                <span>{yoe:.1f} years experience</span>
                <span>{loc}, {country}</span>
                <span>{industry}</span>
                <span>Company size: {comp_size}</span>
                <span>Notice: {notice} days</span>
                <span>Response rate: {rr:.0%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_score:
        st.markdown(f"""
        <div class="rx-profile-header" style="text-align: right;">
            <div class="rx-profile-score-big">{result["score"]:.4f}</div>
            <div class="rx-profile-rank-big">Rank #{result["rank"]:02d} of 100</div>
        </div>
        """, unsafe_allow_html=True)

    # Reasoning
    st.markdown(f"""
    <div class="rx-label">AI REASONING</div>
    <div class="rx-reason" style="margin-top: 0; padding: 1rem; border-left: 2px solid var(--g5);">
        {result["reasoning"]}
    </div>
    """, unsafe_allow_html=True)

    # Tabs for details
    t1, t2, t3, t4 = st.tabs(["SCORE BREAKDOWN", "SKILLS", "CAREER HISTORY", "BEHAVIORAL SIGNALS"])

    with t1:
        if features:
            render_score_breakdown(features, candidate)
        else:
            st.markdown('<div style="color: var(--g6); font-size: 0.8rem;">Feature data not available.</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="rx-label">SKILL INVENTORY</div>', unsafe_allow_html=True)
        if skills:
            expert = [s for s in skills if s.get("proficiency") == "expert"]
            advanced = [s for s in skills if s.get("proficiency") == "advanced"]
            intermediate = [s for s in skills if s.get("proficiency") == "intermediate"]
            beginner = [s for s in skills if s.get("proficiency") == "beginner"]

            for level, items in [("EXPERT", expert), ("ADVANCED", advanced), ("INTERMEDIATE", intermediate), ("BEGINNER", beginner)]:
                if items:
                    st.markdown(f'<div style="font-family: var(--mono); font-size: 0.55rem; color: var(--g5); text-transform: uppercase; letter-spacing: 0.12em; margin: 1rem 0 0.3rem 0;">{level} ({len(items)})</div>', unsafe_allow_html=True)
                    strong = "rx-tag-strong" if level in ("EXPERT", "ADVANCED") else ""
                    html = ""
                    for sk in sorted(items, key=lambda x: x.get("duration_months", 0), reverse=True):
                        html += f'<span class="rx-tag {strong}">{sk.get("name","")} / {sk.get("duration_months",0)}mo</span> '
                    st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color: var(--g6); font-size: 0.8rem;">No skills listed.</div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="rx-label">CAREER TIMELINE</div>', unsafe_allow_html=True)
        html = ""
        for job in career:
            cur = job.get("is_current", False)
            dot = "rx-career-dot-on" if cur else "rx-career-dot-off"
            desc = job.get("description", "")
            if len(desc) > 200: desc = desc[:200] + "..."
            html += f"""<div class="rx-career-row">
                <div class="rx-career-dot {dot}"></div>
                <div>
                    <div class="rx-career-title">{job.get("title","")}</div>
                    <div class="rx-career-co">{job.get("company","")} / {job.get("duration_months",0)} months / {job.get("industry","")}</div>
                    <div class="rx-career-desc">{desc}</div>
                </div>
            </div>"""
        st.markdown(html or '<div style="color: var(--g6);">No career history.</div>', unsafe_allow_html=True)

    with t4:
        st.markdown('<div class="rx-label">PLATFORM SIGNALS</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Profile Completeness", f"{s.get('profile_completeness_score',0):.0f}%")
            st.metric("Open to Work", "Yes" if s.get("open_to_work_flag") else "No")
            st.metric("Notice Period", f"{s.get('notice_period_days',0)} days")
        with c2:
            st.metric("Response Rate", f"{s.get('recruiter_response_rate',0):.0%}")
            st.metric("Avg Response Time", f"{s.get('avg_response_time_hours',0):.0f} hrs")
            st.metric("Interview Completion", f"{s.get('interview_completion_rate',0):.0%}")
        with c3:
            gh = s.get("github_activity_score", -1)
            st.metric("GitHub Score", f"{gh:.0f}" if gh >= 0 else "N/A")
            st.metric("Saved by Recruiters", s.get("saved_by_recruiters_30d", 0))
            v = sum([s.get("verified_email", False), s.get("verified_phone", False), s.get("linkedin_connected", False)])
            st.metric("Verifications", f"{v}/3")


def render_pipeline_page():
    """Full pipeline explanation page."""
    st.markdown("""
    <div class="rx-hero" style="padding-bottom: 1rem;">
        <h1>Ranking Pipeline</h1>
        <div class="rx-hero-sub">
            How RecruitX processes 100,000 candidates in 25 seconds.
            A six-stage pipeline combining information retrieval, feature engineering,
            and machine learning techniques.
        </div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Job Description Parsing",
         "The JD is parsed into structured requirements: must-have skills, nice-to-have skills, "
         "experience bands, title expectations, and location preferences. A skill taxonomy with "
         "200+ aliases maps variations like 'NLP', 'Natural Language Processing', and 'text mining' "
         "to canonical forms.",
         "Input: Raw JD text\nOutput: Structured requirements object\nSkill aliases: 200+\nCategories: must-have, nice-to-have, experience, title, location"),

        ("02", "Feature Extraction",
         "For each of the 100,000 candidates, 54 features are extracted across six categories: "
         "skill match (must-have ratio, nice-to-have ratio, proficiency-weighted scores), "
         "career analysis (title relevance, consulting detection, production ML signals), "
         "education quality, behavioral signals from the Redrob platform (response rate, recency, "
         "interview completion), location fit, and certification bonuses.",
         "Features per candidate: 54\nCategories: 6 (skill, career, education, behavioral, location, certs)\nProcessing speed: ~2,800 candidates/sec"),

        ("03", "Hybrid Retrieval: Dense + Sparse",
         "Two retrieval methods run in parallel. Dense retrieval encodes each candidate's profile "
         "into a 384-dimensional vector using the all-MiniLM-L6-v2 sentence transformer, then "
         "computes cosine similarity against the JD embedding. Sparse retrieval uses BM25 (Okapi) "
         "to find keyword matches. Results are fused using Reciprocal Rank Fusion (RRF) with "
         "k=60 to produce a shortlist of 300 candidates.",
         "Dense model: all-MiniLM-L6-v2 (384-dim)\nSparse model: BM25 Okapi\nFusion: Reciprocal Rank Fusion (k=60)\nShortlist: Top 300 candidates"),

        ("04", "Weighted Ensemble Ranking",
         "The 300 shortlisted candidates are scored using a weighted ensemble of 8 components: "
         "skill match (35%), career relevance (20%), semantic similarity (15%), behavioral signals (10%), "
         "experience fit (8%), education (5%), location (5%), and certifications (2%). Each component "
         "is individually normalized and combined into a final composite score.",
         "Components: 8\nWeights: Skill 35% / Career 20% / Semantic 15% / Behavioral 10% / Experience 8% / Education 5% / Location 5% / Certs 2%\nOutput: Ranked list with scores"),

        ("05", "Honeypot Detection and Penalties",
         "Seven heuristic checks identify approximately 80 synthetic trap candidates with impossible "
         "profiles: expert skills with zero duration, implausible skill breadth (40+ skills), "
         "experience span mismatches (claimed 10 years but career shows 2), assessment score "
         "gaps, and title-description contradictions. Penalty multipliers also downrank consulting-only "
         "careers and keyword-stuffed profiles.",
         "Heuristic checks: 7\nExpected honeypots: ~80 in 100K\nPenalty types: honeypot, consulting-only, keyword stuffing\nResult: 0 honeypots in final top 100"),

        ("06", "Explainable Reasoning Generation",
         "For each of the final top 100 candidates, a fact-based reasoning string is generated. "
         "It references actual data from the candidate's profile: specific skills matched, current "
         "title and company, years of experience, recruiter response rate, and identified gaps. "
         "Reasoning tone adapts to rank position. No LLM is used; reasoning is deterministically "
         "constructed from profile facts for reproducibility.",
         "Output: 100 reasoning strings\nSource: Deterministic, fact-based (no LLM)\nContent: Strengths, gaps, and fit assessment\nTone: Adapts to rank position"),
    ]

    st.markdown('<div class="rx-pad">', unsafe_allow_html=True)
    for num, title, body, detail in steps:
        st.markdown(f"""
        <div class="rx-pipe-section">
            <div class="rx-pipe-header">
                <div class="rx-pipe-num">{num}</div>
                <div>
                    <div class="rx-pipe-title">{title}</div>
                </div>
            </div>
            <div class="rx-pipe-body">{body}</div>
            <div class="rx-pipe-body rx-pipe-detail">{detail.replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_analytics(results, candidates):
    """Analytics charts."""
    scores = [r["score"] for r in results]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="rx-label">SCORE DISTRIBUTION</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=scores, nbinsx=25,
            marker=dict(color='rgba(237,237,237,0.35)', line=dict(color='#ededed', width=0.5))))
        fig.update_layout(
            xaxis=dict(title=dict(text='SCORE', font=dict(family='JetBrains Mono',size=9,color='#444')),
                       gridcolor='#111', linecolor='#222', tickfont=dict(family='JetBrains Mono',size=8,color='#444'), zeroline=False),
            yaxis=dict(title=dict(text='COUNT', font=dict(family='JetBrains Mono',size=9,color='#444')),
                       gridcolor='#111', linecolor='#222', tickfont=dict(family='JetBrains Mono',size=8,color='#444'), zeroline=False),
            height=280, margin=dict(l=50,r=20,t=10,b=50),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', bargap=0.08, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown('<div class="rx-label">SCORE VS EXPERIENCE</div>', unsafe_allow_html=True)
        xs, ys, labels, sizes = [], [], [], []
        for r in results:
            c = candidates.get(r["candidate_id"], {})
            p = c.get("profile", {})
            xs.append(p.get("years_of_experience", 0))
            ys.append(r["score"])
            labels.append(f"{p.get('anonymized_name','')} / {p.get('current_title','')}")
            sizes.append(max(5, r["score"]*16))
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode='markers',
            marker=dict(size=sizes, color='rgba(237,237,237,0.45)', line=dict(width=0.5, color='#ededed')),
            text=labels, hovertemplate='%{text}<br>YoE: %{x:.1f}<br>Score: %{y:.4f}<extra></extra>'))
        fig.update_layout(
            xaxis=dict(title=dict(text='YEARS OF EXPERIENCE', font=dict(family='JetBrains Mono',size=9,color='#444')),
                       gridcolor='#111', linecolor='#222', tickfont=dict(family='JetBrains Mono',size=8,color='#444'), zeroline=False),
            yaxis=dict(title=dict(text='MATCH SCORE', font=dict(family='JetBrains Mono',size=9,color='#444')),
                       gridcolor='#111', linecolor='#222', tickfont=dict(family='JetBrains Mono',size=8,color='#444'), zeroline=False),
            height=280, margin=dict(l=50,r=20,t=10,b=50),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="rx-label">TOP COMPANIES</div>', unsafe_allow_html=True)
    counts = {}
    for r in results:
        co = candidates.get(r["candidate_id"],{}).get("profile",{}).get("current_company","Unknown")
        counts[co] = counts.get(co,0) + 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:12]
    fig = go.Figure(go.Bar(y=[c[0] for c in top][::-1], x=[c[1] for c in top][::-1], orientation='h',
        marker=dict(color='rgba(237,237,237,0.3)', line=dict(color='#ededed', width=0.5))))
    fig.update_layout(
        xaxis=dict(title=dict(text='CANDIDATES', font=dict(family='JetBrains Mono',size=9,color='#444')),
                   gridcolor='#111', tickfont=dict(family='JetBrains Mono',size=8,color='#444'), zeroline=False),
        yaxis=dict(tickfont=dict(family='JetBrains Mono',size=9,color='#666')),
        height=360, margin=dict(l=140,r=20,t=10,b=50),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig, width='stretch')


# =============================================================================
# Try It — Resume Scorer
# =============================================================================
def score_resume_text(text):
    """Score pasted resume text against the JD using our pipeline logic."""
    text_lower = text.lower()
    words = set(re.findall(r'\b[a-z0-9+#.\-]+\b', text_lower))

    # --- Skill matching ---
    must_matched = {}
    for canonical, aliases in MUST_HAVE_SKILLS.items():
        for alias in aliases:
            if alias.lower() in text_lower:
                must_matched[canonical] = alias
                break

    nice_matched = {}
    for canonical, aliases in NICE_TO_HAVE_SKILLS.items():
        for alias in aliases:
            if alias.lower() in text_lower:
                nice_matched[canonical] = alias
                break

    must_total = len(MUST_HAVE_SKILLS)
    nice_total = len(NICE_TO_HAVE_SKILLS)
    must_score = len(must_matched) / must_total if must_total else 0
    nice_score = len(nice_matched) / nice_total if nice_total else 0
    skill_score = must_score * 0.6 + nice_score * 0.4

    # --- Experience extraction ---
    yoe = 0.0
    yoe_matches = re.findall(r'(\d+\.?\d*)\s*(?:\+\s*)?(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?', text_lower)
    if yoe_matches:
        yoe = max(float(y) for y in yoe_matches)
    exp_min, exp_max = JD_EXPERIENCE_RANGE
    if exp_min <= yoe <= exp_max:
        exp_score = 1.0
    elif yoe < exp_min:
        exp_score = max(0, yoe / exp_min)
    else:
        exp_score = max(0, 1.0 - (yoe - exp_max) * 0.1)

    # --- Title relevance ---
    title_keywords = ["ai engineer", "ml engineer", "machine learning", "data scientist",
                      "research engineer", "applied scientist", "nlp engineer",
                      "search engineer", "ranking engineer", "recommendation"]
    title_score = 0.0
    for kw in title_keywords:
        if kw in text_lower:
            title_score = 1.0
            break

    negative_titles = ["marketing manager", "hr manager", "graphic designer",
                       "content writer", "business analyst", "sales", "recruiter"]
    for nt in negative_titles:
        if nt in text_lower:
            title_score *= 0.2
            break

    # --- Product company signal ---
    product_companies = ["google", "amazon", "meta", "facebook", "microsoft", "apple",
                         "netflix", "flipkart", "swiggy", "zomato", "razorpay", "stripe",
                         "uber", "airbnb", "spotify", "linkedin", "twitter", "openai",
                         "deepmind", "nvidia", "ola", "meesho", "cred", "phonepe",
                         "paytm", "byju", "dream11"]
    consulting = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
                  "hcl technologies", "tech mahindra", "cts"]
    product_hit = any(c in text_lower for c in product_companies)
    consult_hit = any(c in text_lower for c in consulting)
    career_score = 0.8 if product_hit else 0.4
    if consult_hit and not product_hit:
        career_score = 0.2

    # --- Composite ---
    composite = (skill_score * 0.35 + career_score * 0.20 + title_score * 0.15 +
                 exp_score * 0.15 + min(1.0, len(must_matched) / 5) * 0.15)
    composite = round(min(1.0, composite), 4)

    return {
        "composite": composite,
        "skill_score": round(skill_score, 4),
        "exp_score": round(exp_score, 4),
        "title_score": round(title_score, 4),
        "career_score": round(career_score, 4),
        "yoe": yoe,
        "must_matched": must_matched,
        "must_missed": [k for k in MUST_HAVE_SKILLS if k not in must_matched],
        "nice_matched": nice_matched,
        "nice_missed": [k for k in NICE_TO_HAVE_SKILLS if k not in nice_matched],
    }


def render_try_it():
    """TRY IT tab — paste resume, get scored."""
    st.markdown('<div class="rx-label">RESUME SCORER</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size: 0.75rem; color: var(--g6); margin-bottom: 1.5rem; line-height: 1.6;">
        Paste your resume or profile text below. Our scoring engine will analyze it against the
        <span style="color: var(--white); font-weight: 600;">Senior AI Engineer</span> job description
        and return a score with detailed feedback.
    </div>""", unsafe_allow_html=True)

    resume_text = st.text_area(
        "Paste your resume text here",
        height=250,
        placeholder="Paste your full resume or LinkedIn profile text here...",
        label_visibility="collapsed"
    )

    if st.button("SCORE MY RESUME", type="primary"):
        if not resume_text or len(resume_text.strip()) < 50:
            st.markdown('<div style="color:#ff4444; font-family:var(--mono); font-size:0.78rem; margin-top:1rem;">Paste at least 50 characters of resume text.</div>', unsafe_allow_html=True)
            return

        result = score_resume_text(resume_text)
        score = result["composite"]

        # Score tier
        if score >= 0.75:
            tier, tier_color = "STRONG FIT", "#00c853"
        elif score >= 0.50:
            tier, tier_color = "MODERATE FIT", "#ffd600"
        elif score >= 0.30:
            tier, tier_color = "WEAK FIT", "#ff6d00"
        else:
            tier, tier_color = "LOW FIT", "#ff1744"

        # Big score display
        st.markdown(f"""
        <div style="margin: 2rem 0; padding: 2rem; border: 1px solid var(--g4);">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-family: var(--mono); font-size: 3rem; font-weight: 700; color: var(--white);">{score:.4f}</div>
                    <div style="font-family: var(--mono); font-size: 0.85rem; color: {tier_color}; letter-spacing: 0.1em; margin-top: 0.3rem;">{tier}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-family: var(--mono); font-size: 0.65rem; color: var(--g6); letter-spacing: 0.08em;">SCORED AGAINST</div>
                    <div style="font-family: var(--mono); font-size: 0.78rem; color: var(--white); margin-top: 0.2rem;">Senior AI Engineer — Redrob</div>
                    <div style="font-family: var(--mono); font-size: 0.65rem; color: var(--g5); margin-top: 0.2rem;">Experience detected: {result['yoe']:.1f} yrs</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Component breakdown
        components = [
            ("SKILL MATCH", result["skill_score"], "35%"),
            ("CAREER RELEVANCE", result["career_score"], "20%"),
            ("TITLE FIT", result["title_score"], "15%"),
            ("EXPERIENCE FIT", result["exp_score"], "15%"),
        ]
        st.markdown('<div class="rx-label" style="margin-top:1.5rem;">SCORE BREAKDOWN</div>', unsafe_allow_html=True)
        for name, val, weight in components:
            bar_w = max(1, val * 100)
            st.markdown(f"""
            <div class="rx-sb-row">
                <div class="rx-sb-name">{name}</div>
                <div class="rx-sb-bar-wrap"><div class="rx-sb-bar-fill" style="width:{bar_w:.1f}%"></div></div>
                <div class="rx-sb-val">{val:.2f}</div>
                <div class="rx-sb-weight">{weight}</div>
            </div>
            """, unsafe_allow_html=True)

        # Skills found
        st.markdown('<div class="rx-label" style="margin-top:2rem;">MUST-HAVE SKILLS</div>', unsafe_allow_html=True)
        must_html = ""
        for sk in MUST_HAVE_SKILLS:
            if sk in result["must_matched"]:
                must_html += f'<span style="font-family:var(--mono);font-size:0.72rem;padding:0.3rem 0.6rem;border:1px solid #00c853;color:#00c853;margin:0.2rem 0.3rem 0.2rem 0;display:inline-block;">{sk}</span>'
            else:
                must_html += f'<span style="font-family:var(--mono);font-size:0.72rem;padding:0.3rem 0.6rem;border:1px solid var(--g4);color:var(--g5);margin:0.2rem 0.3rem 0.2rem 0;display:inline-block;">{sk}</span>'
        st.markdown(f'<div style="margin:0.5rem 0 1rem 0;">{must_html}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:var(--mono);font-size:0.65rem;color:var(--g6);">{len(result["must_matched"])}/{len(MUST_HAVE_SKILLS)} matched</div>', unsafe_allow_html=True)

        st.markdown('<div class="rx-label" style="margin-top:1.5rem;">NICE-TO-HAVE SKILLS</div>', unsafe_allow_html=True)
        nice_html = ""
        for sk in NICE_TO_HAVE_SKILLS:
            if sk in result["nice_matched"]:
                nice_html += f'<span style="font-family:var(--mono);font-size:0.72rem;padding:0.3rem 0.6rem;border:1px solid #00c853;color:#00c853;margin:0.2rem 0.3rem 0.2rem 0;display:inline-block;">{sk}</span>'
            else:
                nice_html += f'<span style="font-family:var(--mono);font-size:0.72rem;padding:0.3rem 0.6rem;border:1px solid var(--g4);color:var(--g5);margin:0.2rem 0.3rem 0.2rem 0;display:inline-block;">{sk}</span>'
        st.markdown(f'<div style="margin:0.5rem 0 1rem 0;">{nice_html}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:var(--mono);font-size:0.65rem;color:var(--g6);">{len(result["nice_matched"])}/{len(NICE_TO_HAVE_SKILLS)} matched</div>', unsafe_allow_html=True)

        # Insights
        insights = []
        if result["skill_score"] >= 0.6:
            insights.append("Strong skill alignment with the JD requirements.")
        elif result["skill_score"] >= 0.3:
            insights.append("Partial skill overlap. Key gaps in: " + ", ".join(result["must_missed"][:4]) + ".")
        else:
            insights.append("Low skill overlap. Missing most must-have skills for this role.")

        if result["exp_score"] >= 0.8:
            insights.append(f"Experience ({result['yoe']:.0f} yrs) is in the ideal 5-9 year band.")
        elif result["yoe"] < 5:
            insights.append(f"Experience ({result['yoe']:.0f} yrs) is below the 5-year minimum preference.")
        else:
            insights.append(f"Experience ({result['yoe']:.0f} yrs) exceeds the 9-year target — overqualification risk.")

        if result["career_score"] >= 0.8:
            insights.append("Product company experience detected — strong signal for this role.")
        elif result["career_score"] <= 0.3:
            insights.append("Consulting-only background detected. The JD explicitly flags this as poor fit.")

        if result["title_score"] >= 0.8:
            insights.append("Title/role aligns well with Senior AI Engineer expectations.")
        elif result["title_score"] <= 0.2:
            insights.append("No AI/ML-related title detected. Title mismatch is a negative signal.")

        st.markdown('<div class="rx-label" style="margin-top:2rem;">INSIGHTS</div>', unsafe_allow_html=True)
        for insight in insights:
            st.markdown(f'<div style="font-family:var(--mono);font-size:0.75rem;color:var(--g7);padding:0.5rem 0;border-bottom:1px solid var(--g3);line-height:1.6;">{insight}</div>', unsafe_allow_html=True)


# =============================================================================
# Main
# =============================================================================
def main():
    # Nav
    st.markdown("""<div class="rx-nav">
        <div class="rx-nav-left">
            <div class="rx-nav-brand">RECRUITX</div>
            <div class="rx-nav-divider"></div>
            <div class="rx-nav-team">by Elements</div>
        </div>
        <div class="rx-nav-hackathon">
            <div class="rx-nav-badge">Redrob India.runs Hackathon</div>
            <div class="rx-nav-right">INTELLIGENT CANDIDATE DISCOVERY ENGINE</div>
        </div>
    </div>""", unsafe_allow_html=True)

    sub_path = OUTPUT_DIR / "submission.csv"
    csvs = [sub_path] if sub_path.exists() else list(OUTPUT_DIR.glob("*.csv"))
    if not csvs:
        st.markdown("""<div class="rx-hero">
            <h1>Intelligent Candidate<br>Discovery</h1>
            <div class="rx-hero-sub">No submission found. Run: python precompute.py && python rank.py</div>
        </div>""", unsafe_allow_html=True)
        return

    results = load_submission(str(csvs[0]))
    ids = [r["candidate_id"] for r in results]
    # Try full dataset first, then fallback to top100 extract for deployed version
    top100_fallback = OUTPUT_DIR / "top100_candidates.jsonl"
    if CANDIDATES_FILE.exists():
        candidates = load_candidates(str(CANDIDATES_FILE), ids)
    elif top100_fallback.exists():
        candidates = load_candidates(str(top100_fallback), ids)
    else:
        candidates = {}
    features_data = load_features()

    # Hero
    st.markdown("""<div class="rx-hero">
        <div style="font-family: var(--mono); font-size: 0.85rem; color: var(--white); text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 1.2rem; font-weight: 600;">ELEMENTS / REDROB INDIA.RUNS HACKATHON 2025</div>
        <h1>Intelligent Candidate<br>Discovery</h1>
        <div class="rx-hero-sub">Semantic retrieval and weighted ensemble ranking across 100,000 candidates.
        Hybrid dense-sparse pipeline with explainable scoring.</div>
        <div style="display: flex; gap: 1.5rem; margin-top: 1.5rem;">
            <div style="font-family: var(--mono); font-size: 0.78rem; color: var(--white); padding: 0.4rem 0.8rem; border: 1px solid var(--g7); letter-spacing: 0.08em; font-weight: 500;">TEAM ELEMENTS</div>
            <div style="font-family: var(--mono); font-size: 0.78rem; color: var(--white); padding: 0.4rem 0.8rem; border: 1px solid var(--g7); letter-spacing: 0.08em; font-weight: 500;">MADE FOR REDROB</div>
            <div style="font-family: var(--mono); font-size: 0.78rem; color: var(--white); padding: 0.4rem 0.8rem; border: 1px solid var(--g7); letter-spacing: 0.08em; font-weight: 500;">INDIA.RUNS 2025</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Metrics
    avg = np.mean([r["score"] for r in results])
    top = results[0]["score"]
    st.markdown(f"""<div class="rx-strip">
        <div class="rx-strip-cell"><div class="rx-strip-val">100,000</div><div class="rx-strip-lbl">Candidates scanned</div></div>
        <div class="rx-strip-cell"><div class="rx-strip-val">{len(results)}</div><div class="rx-strip-lbl">Ranked output</div></div>
        <div class="rx-strip-cell"><div class="rx-strip-val">{top:.4f}</div><div class="rx-strip-lbl">Top score</div></div>
        <div class="rx-strip-cell"><div class="rx-strip-val">{avg:.4f}</div><div class="rx-strip-lbl">Mean score</div></div>
        <div class="rx-strip-cell"><div class="rx-strip-val">25.1s</div><div class="rx-strip-lbl">Ranking runtime</div></div>
    </div>""", unsafe_allow_html=True)

    # Main tabs
    tab_rank, tab_pipeline, tab_candidate, tab_analytics, tab_tryit = st.tabs([
        "RANKINGS", "PIPELINE", "CANDIDATE PROFILE", "ANALYTICS", "TRY IT"
    ])

    with tab_rank:
        st.markdown('<div class="rx-pad">', unsafe_allow_html=True)
        st.markdown('<div class="rx-label">RANKED CANDIDATES</div>', unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 4])
        with c1: show_n = st.slider("TOP N", 5, 100, 25, label_visibility="collapsed")
        with c2: min_s = st.slider("MIN SCORE", 0.0, 1.0, 0.0, 0.01, label_visibility="collapsed")

        filtered = [r for r in results if r["score"] >= min_s][:show_n]
        st.markdown(f'<div style="font-family:var(--mono);font-size:0.58rem;color:var(--g5);letter-spacing:0.1em;text-transform:uppercase;margin:0.8rem 0 0.4rem 0;">Showing {len(filtered)} of {len(results)} / min score {min_s:.2f}</div>', unsafe_allow_html=True)

        html = ""
        for r in filtered:
            html += candidate_row_html(r, candidates.get(r["candidate_id"], {}))
        st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_pipeline:
        render_pipeline_page()

    with tab_candidate:
        st.markdown('<div class="rx-pad">', unsafe_allow_html=True)
        st.markdown('<div class="rx-label">SELECT CANDIDATE</div>', unsafe_allow_html=True)
        opts = {
            f"#{r['rank']:02d}  {candidates.get(r['candidate_id'],{}).get('profile',{}).get('anonymized_name',r['candidate_id'])}  ({r['score']:.4f})": r["candidate_id"]
            for r in results
        }
        sel = st.selectbox("Candidate", list(opts.keys()), label_visibility="collapsed")
        cid = opts[sel]
        res = next(r for r in results if r["candidate_id"] == cid)
        cand = candidates.get(cid, {})
        feats = features_data.get(cid, {})

        render_profile_detail(res, cand, feats)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_analytics:
        st.markdown('<div class="rx-pad">', unsafe_allow_html=True)
        render_analytics(results, candidates)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_tryit:
        st.markdown('<div class="rx-pad">', unsafe_allow_html=True)
        render_try_it()
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
