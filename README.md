# RecruitX — Intelligent Candidate Discovery & Ranking Engine

> AI-powered candidate ranking system for the Redrob Hackathon.
> Ranks 100K candidates against a Senior AI Engineer JD using hybrid semantic + keyword retrieval, weighted ensemble scoring, honeypot detection, and explainable AI reasoning.

## Architecture

```
JD Input → Skill Extraction → Embedding Generation → Vector Search
         → BM25 Index Build → Hybrid Retrieval (RRF Fusion)
         → Feature Engineering (40+ features)
         → Weighted Ensemble Ranking (+ penalty multipliers)
         → Honeypot Filtering
         → Explainable Reasoning Generation
         → Ranked Top-100 CSV
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (Python 3.11-3.13 recommended)
py -3.13 -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Pre-compute Artifacts (run once, ~15-30 min)

```bash
python precompute.py
```

This computes:
- Sentence-transformer embeddings for 100K candidates
- BM25 search index
- 40+ structured features per candidate
- Honeypot detection scores

### 3. Generate Submission (≤5 minutes, CPU-only)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### 4. Validate Submission

```bash
python validate_submission.py submission.csv
```

### 5. Launch Dashboard

```bash
streamlit run app.py
```

## Project Structure

```
RecruitX/
├── src/
│   ├── config.py              # Configuration, JD requirements, weights
│   ├── jd_parser.py           # JD parsing and skill extraction
│   ├── candidate_loader.py    # Efficient JSONL candidate loading
│   ├── feature_engineer.py    # 40+ feature extraction pipeline
│   ├── honeypot_detector.py   # Impossible profile detection
│   ├── text_builder.py        # Searchable text representation builder
│   ├── embedder.py            # Sentence-transformer embeddings
│   ├── bm25_retriever.py      # BM25 sparse retrieval
│   ├── hybrid_retriever.py    # RRF fusion (dense + sparse)
│   ├── ranker.py              # Weighted ensemble scorer
│   └── explainer.py           # Per-candidate reasoning generation
├── precompute.py              # Pre-computation pipeline
├── rank.py                    # Main ranking entry point
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Methodology

### Retrieval: Hybrid Dense + Sparse

- **Dense**: Sentence-transformers (`all-MiniLM-L6-v2`) encodes candidates and JD into 384-dim vectors. Cosine similarity retrieves top-500 semantically similar candidates.
- **Sparse**: BM25 (Okapi BM25) retrieves top-500 keyword-matched candidates.
- **Fusion**: Reciprocal Rank Fusion (RRF) merges both lists into top-300.

### Ranking: Weighted Ensemble Scorer

Eight scoring components, each [0,1]:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Skill Match | 35% | Must-have & nice-to-have JD skill alignment |
| Career Relevance | 20% | Title relevance, production ML experience |
| Semantic Similarity | 15% | Dense + sparse retrieval scores |
| Behavioral Signals | 10% | Response rate, activity, completeness |
| Experience Fit | 8% | Years in 5-9 year band |
| Education Quality | 5% | Field relevance, institution tier |
| Location Fit | 5% | India-based, preferred cities |
| Certifications | 2% | Relevant cloud/ML certifications |

### Penalty Multipliers

- **Honeypot detected**: score × 0.0 (eliminated)
- **Consulting-only career**: score × 0.30
- **Keyword stuffer pattern**: score × 0.35
- **No ML signals at all**: score × 0.45
- **Inactive > 6 months**: score × 0.65

### Honeypot Detection

Seven heuristic checks for impossible profiles:
1. Expert proficiency with zero months of usage
2. Many skills with zero endorsements/duration
3. Career duration exceeding elapsed time since start date
4. Claimed experience exceeding career timeline span
5. Title-description mismatches (marketing title, engineering description)
6. Assessment scores contradicting claimed proficiency
7. Implausibly high expert skill count

### Explainability

Each candidate gets a 1-2 sentence reasoning that:
- References specific facts from their profile
- Connects strengths to JD requirements
- Honestly acknowledges gaps
- Varies across candidates (not templated)
- Matches tone to rank position

## Compute Compliance

| Constraint | Our Approach |
|-----------|-------------|
| ≤5 min runtime | Ranking uses pre-computed artifacts; ~60-120s |
| 16GB RAM | Peak ~4GB (embeddings + features) |
| CPU only | All operations are CPU-based |
| No network | Zero API calls during ranking |
| ≤5GB disk | ~1.5GB for embeddings + index |

## AI Tools Declaration

- Antigravity (Gemini-based coding assistant) for architecture discussion and code generation
- All engineering decisions were made by the team
- No candidate data was fed to any external LLM
