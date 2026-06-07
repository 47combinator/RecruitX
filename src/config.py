"""
RecruitX Configuration
======================
Central configuration for the Intelligent Candidate Discovery & Ranking Engine.
All JD requirements, scoring weights, file paths, and thresholds in one place.
"""

import os
from pathlib import Path
from datetime import date

# ==============================================================================
# File Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"
CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
PRECOMPUTED_DIR = PROJECT_ROOT / "data" / "precomputed"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Model Configuration
# ==============================================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 256  # For embedding computation

# ==============================================================================
# Reference Date (for recency calculations)
# ==============================================================================
REFERENCE_DATE = date(2026, 6, 1)  # Approximate current date

# ==============================================================================
# JD: Extracted Requirements
# ==============================================================================
JD_TITLE = "Senior AI Engineer — Founding Team"
JD_COMPANY = "Redrob AI"
JD_LOCATION = "Pune/Noida, India (Hybrid)"
JD_EXPERIENCE_RANGE = (5, 9)  # Years, but flexible
JD_EXPERIENCE_HARD_MIN = 2    # Below this is disqualifying

# The full JD text for embedding (key paragraphs only)
JD_EMBEDDING_TEXT = """
Senior AI Engineer at an AI-native talent intelligence platform.
Own the intelligence layer: ranking, retrieval, and matching systems.
Ship embeddings-based retrieval, hybrid search, LLM-based re-ranking.
Build evaluation infrastructure: offline benchmarks, online A/B testing.
Production experience with sentence-transformers, vector databases, hybrid search.
Strong Python, ranking evaluation frameworks, NDCG, MRR, MAP.
Experience with FAISS, Qdrant, Pinecone, Weaviate, Milvus, OpenSearch, Elasticsearch.
NLP, information retrieval, text mining, semantic search, dense retrieval.
Learning-to-rank, XGBoost, LightGBM, recommendation systems.
LLM fine-tuning, LoRA, QLoRA, PEFT. Machine learning engineering in production.
Data pipelines, feature engineering, model deployment, MLOps.
"""

# ==============================================================================
# Skill Taxonomy — Must-Have Skills
# ==============================================================================
# Each entry: canonical_name -> list of aliases/synonyms (case-insensitive matching)
MUST_HAVE_SKILLS = {
    # Embeddings & Retrieval
    "Sentence Transformers": [
        "sentence-transformers", "sentence transformers", "sbert",
        "bi-encoder", "bi encoder"
    ],
    "Embeddings": [
        "embeddings", "word embeddings", "text embeddings",
        "BGE", "E5", "OpenAI embeddings", "embedding models",
        "dense retrieval", "vector embeddings", "semantic embeddings"
    ],
    "Semantic Search": [
        "semantic search", "neural search", "vector search",
        "similarity search", "approximate nearest neighbor", "ANN"
    ],
    # Vector Databases
    "FAISS": ["FAISS", "faiss"],
    "Qdrant": ["Qdrant", "qdrant"],
    "Pinecone": ["Pinecone", "pinecone"],
    "Weaviate": ["Weaviate", "weaviate"],
    "Milvus": ["Milvus", "milvus"],
    "OpenSearch": ["OpenSearch", "opensearch"],
    "Elasticsearch": ["Elasticsearch", "elasticsearch", "elastic search", "ELK"],
    # Python
    "Python": ["Python", "python", "CPython"],
    # NLP / IR
    "NLP": [
        "NLP", "natural language processing", "text mining",
        "text classification", "NER", "named entity recognition",
        "information retrieval", "IR", "text analytics",
        "language models", "transformer models", "BERT", "GPT",
        "tokenization", "text processing"
    ],
    # Ranking & Evaluation
    "Ranking Systems": [
        "ranking", "search ranking", "recommendation system",
        "recommender system", "learning to rank", "learning-to-rank",
        "LTR", "re-ranking", "reranking"
    ],
    "ML Evaluation": [
        "NDCG", "MRR", "MAP", "precision", "recall",
        "A/B testing", "evaluation framework", "offline evaluation",
        "model evaluation", "ranking evaluation"
    ],
}

# ==============================================================================
# Skill Taxonomy — Nice-to-Have Skills
# ==============================================================================
NICE_TO_HAVE_SKILLS = {
    "LLM Fine-tuning": [
        "LoRA", "QLoRA", "PEFT", "fine-tuning", "fine tuning",
        "finetuning", "model fine-tuning", "instruction tuning",
        "RLHF", "DPO", "adapter tuning"
    ],
    "LLM/GenAI": [
        "LLM", "large language model", "GPT", "ChatGPT",
        "generative AI", "GenAI", "prompt engineering",
        "RAG", "retrieval augmented generation",
        "LangChain", "LlamaIndex", "Hugging Face",
        "transformers", "Transformer"
    ],
    "Learning-to-Rank Models": [
        "XGBoost", "LightGBM", "gradient boosting",
        "LambdaMART", "LambdaRank", "RankNet",
        "CatBoost", "random forest"
    ],
    "Deep Learning": [
        "deep learning", "neural network", "PyTorch", "TensorFlow",
        "Keras", "CNN", "RNN", "LSTM", "attention mechanism",
        "transformer architecture"
    ],
    "MLOps": [
        "MLOps", "ML pipeline", "model deployment",
        "model serving", "ML infrastructure", "feature store",
        "experiment tracking", "MLflow", "Weights & Biases", "W&B",
        "Kubeflow", "model monitoring"
    ],
    "Data Engineering": [
        "Spark", "PySpark", "Airflow", "Kafka",
        "data pipeline", "ETL", "data warehouse",
        "Snowflake", "BigQuery", "Databricks", "dbt"
    ],
    "Cloud Platforms": [
        "AWS", "GCP", "Azure", "cloud",
        "SageMaker", "Vertex AI", "EC2", "S3", "Lambda"
    ],
    "Distributed Systems": [
        "distributed systems", "microservices", "Kubernetes",
        "Docker", "containerization", "scalability",
        "high availability", "load balancing"
    ],
    "Open Source": [
        "open source", "open-source", "GitHub contributions",
        "OSS", "contributor"
    ],
}

# ==============================================================================
# Title Relevance Tiers
# ==============================================================================
# Maps title patterns (lowercased) to relevance scores 0-1
TITLE_RELEVANCE = {
    # Tier 1: Direct match (0.9-1.0)
    "ai engineer": 1.0,
    "machine learning engineer": 1.0,
    "ml engineer": 1.0,
    "senior ai engineer": 1.0,
    "senior ml engineer": 1.0,
    "senior machine learning engineer": 1.0,
    "lead ai engineer": 0.95,
    "lead ml engineer": 0.95,
    "staff ai engineer": 0.95,
    "staff ml engineer": 0.95,
    "principal ai engineer": 0.95,
    "principal ml engineer": 0.95,
    "applied scientist": 0.95,
    "applied ml scientist": 0.95,
    "nlp engineer": 0.95,
    "search engineer": 0.95,
    "ranking engineer": 0.95,
    "recommendation engineer": 0.95,

    # Tier 2: Strong adjacent (0.7-0.89)
    "data scientist": 0.85,
    "senior data scientist": 0.85,
    "lead data scientist": 0.85,
    "research engineer": 0.80,
    "research scientist": 0.75,
    "deep learning engineer": 0.85,
    "software engineer": 0.70,  # If they have ML context
    "senior software engineer": 0.72,
    "backend engineer": 0.65,
    "senior backend engineer": 0.67,
    "full stack engineer": 0.55,
    "platform engineer": 0.60,

    # Tier 3: Weak match (0.3-0.55)
    "data engineer": 0.50,
    "data analyst": 0.40,
    "analytics engineer": 0.45,
    "devops engineer": 0.35,
    "frontend engineer": 0.30,

    # Tier 4: Non-match (0.0-0.2)
    "product manager": 0.15,
    "project manager": 0.10,
    "business analyst": 0.10,
    "marketing manager": 0.05,
    "sales executive": 0.02,
    "hr manager": 0.02,
    "accountant": 0.01,
    "operations manager": 0.05,
    "customer support": 0.02,
    "content writer": 0.05,
    "graphic designer": 0.03,
    "mechanical engineer": 0.05,
    "civil engineer": 0.03,
    "electrical engineer": 0.10,
}

# ==============================================================================
# Consulting / Services Firms (Disqualifier if entire career)
# ==============================================================================
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "tech mahindra",
    "hcl", "hcl technologies", "mindtree", "l&t infotech",
    "lti", "ltimindtree", "persistent systems", "mphasis",
    "hexaware", "cyient", "zensar", "birlasoft", "sonata software",
    "niit technologies", "coforge", "larsen & toubro infotech",
}

# Known non-consulting product companies (positive signal)
PRODUCT_COMPANIES = {
    "google", "microsoft", "amazon", "apple", "meta", "facebook",
    "netflix", "uber", "airbnb", "stripe", "spotify",
    "flipkart", "zomato", "swiggy", "razorpay", "cred",
    "paytm", "phonepe", "meesho", "zerodha", "groww",
    "ola", "byju's", "unacademy", "dream11", "freshworks",
    "zoho", "atlassian", "salesforce", "adobe", "oracle",
    "sap", "ibm", "intel", "nvidia", "openai", "anthropic",
    "hugging face", "deepmind", "databricks", "snowflake",
    "elastic", "confluent", "datadog", "mongodb",
}

# ==============================================================================
# Education Field Relevance
# ==============================================================================
EDUCATION_FIELD_RELEVANCE = {
    "computer science": 1.0,
    "computer engineering": 1.0,
    "artificial intelligence": 1.0,
    "machine learning": 1.0,
    "data science": 0.95,
    "information technology": 0.85,
    "software engineering": 0.90,
    "statistics": 0.80,
    "mathematics": 0.75,
    "applied mathematics": 0.80,
    "computational linguistics": 0.85,
    "electrical engineering": 0.60,
    "electronics": 0.55,
    "electronics and communication": 0.55,
    "information systems": 0.70,
    "physics": 0.50,
    "mechanical engineering": 0.15,
    "civil engineering": 0.10,
    "chemical engineering": 0.10,
    "biotechnology": 0.15,
    "commerce": 0.05,
    "business administration": 0.10,
    "marketing": 0.03,
    "finance": 0.05,
}

EDUCATION_TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
    "unknown": 0.35,
}

DEGREE_LEVEL_SCORES = {
    "ph.d": 1.0,
    "phd": 1.0,
    "m.tech": 0.85,
    "mtech": 0.85,
    "m.e.": 0.85,
    "ms": 0.80,
    "m.sc": 0.75,
    "msc": 0.75,
    "m.s.": 0.80,
    "mba": 0.40,
    "b.tech": 0.60,
    "btech": 0.60,
    "b.e.": 0.60,
    "be": 0.60,
    "b.sc": 0.50,
    "bsc": 0.50,
    "bca": 0.50,
    "mca": 0.70,
    "diploma": 0.30,
}

# ==============================================================================
# India Preferred Locations
# ==============================================================================
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi",
    "delhi ncr", "gurgaon", "gurugram", "bengaluru", "bangalore",
    "chennai", "kolkata",
}

INDIA_KEYWORDS = {"india"}

# ==============================================================================
# Scoring Weights (for weighted ensemble ranker)
# ==============================================================================
SCORING_WEIGHTS = {
    "skill_match":         0.35,
    "career_relevance":    0.20,
    "semantic_similarity": 0.15,
    "behavioral_score":    0.10,
    "experience_fit":      0.08,
    "education_quality":   0.05,
    "location_fit":        0.05,
    "certification_bonus": 0.02,
}

# Penalty multipliers (applied post-score)
PENALTY_HONEYPOT = 0.0
PENALTY_CONSULTING_ONLY = 0.30
PENALTY_KEYWORD_STUFFER = 0.35
PENALTY_NO_ML_SIGNAL = 0.45
PENALTY_INACTIVE_6_MONTHS = 0.65
PENALTY_VERY_LONG_NOTICE = 0.80  # notice_period > 90 days

# ==============================================================================
# Retrieval Configuration
# ==============================================================================
BM25_TOP_K = 500
DENSE_TOP_K = 500
HYBRID_TOP_K = 300  # After RRF fusion
RRF_K = 60  # RRF smoothing constant
FINAL_TOP_K = 100  # Final output

# ==============================================================================
# Honeypot Detection Thresholds
# ==============================================================================
HONEYPOT_EXPERT_MIN_DURATION = 6       # Expert proficiency needs at least 6 months
HONEYPOT_MAX_SKILLS_WITH_ZERO_DUR = 3  # Max skills with 0 months before flagging
HONEYPOT_SCORE_THRESHOLD = 0.55        # Above this → flagged as honeypot

# Career-description mismatch: title says X but description says Y
TITLE_DESCRIPTION_MISMATCH_KEYWORDS = {
    "marketing manager": ["CAD", "SolidWorks", "mechanical", "FEA", "ANSYS", "warehouse", "fulfillment"],
    "accountant": ["machine learning", "deep learning", "neural network"],
    "hr manager": ["Spark", "Kafka", "data pipeline", "model training"],
    "customer support": ["algorithm", "distributed systems", "Kubernetes"],
    "civil engineer": ["NLP", "transformer", "fine-tuning", "embedding"],
    "mechanical engineer": ["NLP", "transformer", "fine-tuning", "embedding", "LLM"],
    "graphic designer": ["machine learning", "deep learning", "neural network", "ranking"],
    "content writer": ["production ML", "model serving", "Kubernetes"],
    "operations manager": ["machine learning", "deep learning", "model training"],
    "sales executive": ["machine learning", "deep learning", "model training"],
}
