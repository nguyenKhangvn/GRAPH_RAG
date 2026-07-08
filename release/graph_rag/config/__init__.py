"""Centralized configuration loader for schema-guided GraphRAG.

All domain-specific configuration (keywords, thresholds, scoring weights,
schema metadata, intent policy, business rules) is loaded from JSON files
in this directory.

This module also re-exports all legacy config.py constants for backward
compatibility so that `from graph_rag.config import NEO4J_URI` continues
to work.

Usage:
    from graph_rag.config import cfg, NEO4J_URI, RELATIONSHIP_MAP
    from graph_rag.config.loader import cfg
"""

import os
from pathlib import Path

from .loader import ConfigLoader

# ── Domain config loader (JSON-backed) ─────────────────────────────
cfg = ConfigLoader()

# ── Legacy constants (from original config.py) ─────────────────────
# These are kept for backward compatibility. All modules that do
# `from graph_rag.config import X` will find them here.

from dotenv import load_dotenv

# 1. Load root .env (primary configuration)
root_env = Path(__file__).resolve().parent.parent.parent / ".env"
if root_env.exists():
    load_dotenv(root_env, override=True)

# 2. Load graph_rag/.env as fallback defaults (do NOT override)
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# =============================================================================
#  SECTION 1: DATABASE CONFIGURATION (NEO4J)
# =============================================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
if not NEO4J_PASSWORD:
    import logging as _logging
    _logging.getLogger(__name__).warning("NEO4J_PASSWORD is not set. Connection may fail.")

# =============================================================================
#  SECTION 2: AI & MODEL CONFIGURATION
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MIMO_API_KEY = os.getenv("MIMO_API_KEY")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
PIPELINE_LLM_MODEL_NAME = os.getenv("PIPELINE_LLM_MODEL_NAME", LLM_MODEL_NAME)
QUERY_ANALYZER_LLM_MODEL_NAME = os.getenv("QUERY_ANALYZER_LLM_MODEL_NAME", PIPELINE_LLM_MODEL_NAME)
ENABLE_LLM_FALLBACKS = os.getenv("ENABLE_LLM_FALLBACKS", "false").lower() == "true"
LLM_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",")
    if m.strip()
]

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
TOP_K = int(os.getenv("TOP_K", 10))

# =============================================================================
#  SECTION 3: SEARCH INDEXES (NEO4J)
# =============================================================================
_VECTOR_INDEXES_DEFAULT = [
    "accommodation_vec_idx",
    "dish_vec_idx",
    "event_vec_idx",
    "restaurant_vec_idx",
    "tour_vec_idx",
    "tourist_vec_idx"
]
_vector_indexes_env = os.getenv("VECTOR_INDEXES_STR", "").strip()
VECTOR_INDEXES = (
    [s.strip() for s in _vector_indexes_env.split(",") if s.strip()]
    if _vector_indexes_env
    else _VECTOR_INDEXES_DEFAULT
)

FULLTEXT_INDEXES = [
    "accommodation_ft_idx",
    "agency_ft_idx",
    "dish_ft_idx",
    "event_ft_idx",
    "restaurant_ft_idx",
    "tour_ft_idx",
    "tourist_ft_idx",
    "travelinfo_ft_idx"
]

# =============================================================================
#  SECTION 4: GRAPH BUSINESS LOGIC (schema-guided)
# =============================================================================
RELATIONSHIP_MAP = cfg.relationship_map()

# =============================================================================
#  SECTION 5: APP SETTINGS
# =============================================================================
DEBUG_MODE = os.getenv("DEBUG", "True").lower() == "true"

ENABLE_EXPANDED_NAMES_IN_SEARCH_QUERY = (
    os.getenv("ENABLE_EXPANDED_NAMES_IN_SEARCH_QUERY", "false").lower() == "true"
)
MAX_EXPANDED_GROUNDED_NAMES = int(os.getenv("MAX_EXPANDED_GROUNDED_NAMES", "3"))

ENABLE_AGENTIC_RETRIEVAL = os.getenv("ENABLE_AGENTIC_RETRIEVAL", "true").lower() == "true"
AGENTIC_MAX_ITERATIONS = int(os.getenv("AGENTIC_MAX_ITERATIONS", "2"))
AGENTIC_MAX_SUB_QUERIES = int(os.getenv("AGENTIC_MAX_SUB_QUERIES", "3"))

# Web search fallback for missing attributes
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true"
WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY", "")
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")  # duckduckgo (free) or tavily

# Text-to-Cypher: LLM generates Cypher queries for complex retrieval
ENABLE_TEXT_TO_CYPHER = os.getenv("ENABLE_TEXT_TO_CYPHER", "true").lower() == "true"
TEXT_TO_CYPHER_MAX_RESULTS = int(os.getenv("TEXT_TO_CYPHER_MAX_RESULTS", "20"))
TEXT_TO_CYPHER_TIMEOUT_MS = int(os.getenv("TEXT_TO_CYPHER_TIMEOUT_MS", "5000"))

RAW_CONTEXT_DEFAULT_MAX_ITEMS = int(os.getenv("RAW_CONTEXT_DEFAULT_MAX_ITEMS", "40"))

# Intent constants — import here to avoid circular dependency
from graph_rag.core.intents import IntentType as _IT

RAW_CONTEXT_MAX_ITEMS_BY_INTENT = {
    _IT.TOUR_PLAN: int(os.getenv("RAW_CONTEXT_MAX_ITEMS_TOUR_PLAN", "40")),
    _IT.DISCOVERY: int(os.getenv("RAW_CONTEXT_MAX_ITEMS_DISCOVERY", "60")),
    _IT.TOURISM: int(os.getenv("RAW_CONTEXT_MAX_ITEMS_TOURISM", "40")),
    _IT.FOOD: int(os.getenv("RAW_CONTEXT_MAX_ITEMS_FOOD", "40")),
    _IT.ACCOMMODATION: int(os.getenv("RAW_CONTEXT_MAX_ITEMS_ACCOMMODATION", "40")),
    _IT.EVENT: int(os.getenv("RAW_CONTEXT_MAX_ITEMS_EVENT", "40")),
    _IT.ENTITY_FACT: int(os.getenv("RAW_CONTEXT_MAX_ITEMS_ENTITY_FACT", "40")),
    _IT.DISTANCE: int(os.getenv("RAW_CONTEXT_MAX_ITEMS_DISTANCE", "40")),
}

# Schema-guided traversal policy: intent → evidence_types → relation names
INTENT_TRAVERSAL_POLICY = {}
for _intent_name in cfg.all_intent_names():
    _evidence_types = cfg.intent_evidence_types(_intent_name)
    _relations = cfg.relations_for_evidence_types(_evidence_types)
    INTENT_TRAVERSAL_POLICY[_intent_name] = [r["name"] for r in _relations]

TRAVERSAL_WHITELIST = cfg.traversal_whitelist()

INTENT_ATTRIBUTE_POLICY = {}
for _intent_name in cfg.all_intent_names():
    _attrs = cfg.attribute_policy(_intent_name)
    if _attrs:
        INTENT_ATTRIBUTE_POLICY[_intent_name] = [
            (a["property"], a["display"]) for a in _attrs
        ]

# Context GraphRAG V2 rollout flags
CONTEXT_BUILDER_VERSION = os.getenv("CONTEXT_BUILDER_VERSION", "v1").strip().lower()
ENABLE_CONTEXT_ORGANIZER = os.getenv("ENABLE_CONTEXT_ORGANIZER", "false").lower() == "true"
ENABLE_STRUCTURAL_CONTEXT = os.getenv("ENABLE_STRUCTURAL_CONTEXT", "true").lower() == "true"
ENABLE_HARD_KEEP_1HOP = os.getenv("ENABLE_HARD_KEEP_1HOP", "true").lower() == "true"
ENABLE_TEXTUAL_MMR = os.getenv("ENABLE_TEXTUAL_MMR", "true").lower() == "true"
ENABLE_CONTEXT_DEBUG_LOG = os.getenv("ENABLE_CONTEXT_DEBUG_LOG", "true").lower() == "true"
ENABLE_CROSS_ENCODER_RERANKER = os.getenv("ENABLE_CROSS_ENCODER_RERANKER", "false").lower() == "true"
CROSS_ENCODER_RERANKER_MODEL = os.getenv("CROSS_ENCODER_RERANKER_MODEL", "BAAI/bge-reranker-base")
CROSS_ENCODER_RERANK_TOP_N = int(os.getenv("CROSS_ENCODER_RERANK_TOP_N", "12"))
CROSS_ENCODER_RERANK_TIMEOUT_SEC = float(os.getenv("CROSS_ENCODER_RERANK_TIMEOUT_SEC", "3.0"))

# BGE candidate scoring — semantic relevance for PolicyRanker
ENABLE_BGE_CANDIDATE_SCORING = os.getenv("ENABLE_BGE_CANDIDATE_SCORING", "false").lower() == "true"
BGE_CANDIDATE_SCORING_MODEL = os.getenv("BGE_CANDIDATE_SCORING_MODEL", "BAAI/bge-reranker-base")
BGE_CANDIDATE_SCORE_WEIGHT = float(os.getenv("BGE_CANDIDATE_SCORE_WEIGHT", "2.0"))
BGE_CANDIDATE_SCORING_TIMEOUT_SEC = float(os.getenv("BGE_CANDIDATE_SCORING_TIMEOUT_SEC", "15.0"))

ENABLE_COMMUNITY_SUMMARY = os.getenv("ENABLE_COMMUNITY_SUMMARY", "false").lower() == "true"
COMMUNITY_SUMMARY_PATH = os.getenv(
    "COMMUNITY_SUMMARY_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "community_summaries.json"),
)
COMMUNITY_SUMMARY_TOP_K = int(os.getenv("COMMUNITY_SUMMARY_TOP_K", "2"))
COMMUNITY_SUMMARY_MIN_SCORE = float(os.getenv("COMMUNITY_SUMMARY_MIN_SCORE", "0.18"))

ENABLE_QUERY_FRAME_V2 = os.getenv("ENABLE_QUERY_FRAME_V2", "false").lower() == "true"
ENABLE_ROLE_AWARE_GROUNDING = os.getenv("ENABLE_ROLE_AWARE_GROUNDING", "false").lower() == "true"
QUERY_FRAME_MIN_CONFIDENCE = float(os.getenv("QUERY_FRAME_MIN_CONFIDENCE", "0.60"))
QUERY_FRAME_DEBUG_LOG = os.getenv("QUERY_FRAME_DEBUG_LOG", "true").lower() == "true"

GRAPH_RAG_V3_ENABLED = os.getenv("GRAPH_RAG_V3_ENABLED", "false").lower() == "true"
GRAPH_RAG_V3_MAX_FACTS_PER_ANCHOR = int(os.getenv("GRAPH_RAG_V3_MAX_FACTS_PER_ANCHOR", "18"))

FREEZE_METADATA_AFTER_STEP1 = os.getenv("FREEZE_METADATA_AFTER_STEP1", "true").lower() == "true"

__all__ = [
    "cfg", "ConfigLoader",
    # Database
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
    # AI
    "GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "XAI_API_KEY",
    "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "MIMO_API_KEY", "MIMO_BASE_URL",
    "LLM_MODEL_NAME", "PIPELINE_LLM_MODEL_NAME", "QUERY_ANALYZER_LLM_MODEL_NAME",
    "ENABLE_LLM_FALLBACKS", "LLM_FALLBACK_MODELS",
    "EMBEDDING_MODEL_NAME", "EMBEDDING_DIMENSION", "TOP_K",
    # Indexes
    "VECTOR_INDEXES", "FULLTEXT_INDEXES",
    # Graph
    "RELATIONSHIP_MAP", "INTENT_TRAVERSAL_POLICY", "TRAVERSAL_WHITELIST",
    "INTENT_ATTRIBUTE_POLICY", "RAW_CONTEXT_MAX_ITEMS_BY_INTENT",
    "RAW_CONTEXT_DEFAULT_MAX_ITEMS",
    # Settings
    "DEBUG_MODE", "ENABLE_EXPANDED_NAMES_IN_SEARCH_QUERY",
    "MAX_EXPANDED_GROUNDED_NAMES",
    "ENABLE_AGENTIC_RETRIEVAL", "AGENTIC_MAX_ITERATIONS", "AGENTIC_MAX_SUB_QUERIES",
    "ENABLE_TEXT_TO_CYPHER", "TEXT_TO_CYPHER_MAX_RESULTS", "TEXT_TO_CYPHER_TIMEOUT_MS",
    "CONTEXT_BUILDER_VERSION", "ENABLE_CONTEXT_ORGANIZER",
    "ENABLE_STRUCTURAL_CONTEXT", "ENABLE_HARD_KEEP_1HOP",
    "ENABLE_TEXTUAL_MMR", "ENABLE_CONTEXT_DEBUG_LOG",
    "ENABLE_CROSS_ENCODER_RERANKER", "CROSS_ENCODER_RERANKER_MODEL",
    "CROSS_ENCODER_RERANK_TOP_N", "CROSS_ENCODER_RERANK_TIMEOUT_SEC",
    "ENABLE_BGE_CANDIDATE_SCORING", "BGE_CANDIDATE_SCORING_MODEL",
    "BGE_CANDIDATE_SCORE_WEIGHT", "BGE_CANDIDATE_SCORING_TIMEOUT_SEC",
    "ENABLE_COMMUNITY_SUMMARY", "COMMUNITY_SUMMARY_PATH",
    "COMMUNITY_SUMMARY_TOP_K", "COMMUNITY_SUMMARY_MIN_SCORE",
    "ENABLE_QUERY_FRAME_V2", "ENABLE_ROLE_AWARE_GROUNDING",
    "QUERY_FRAME_MIN_CONFIDENCE", "QUERY_FRAME_DEBUG_LOG",
    "GRAPH_RAG_V3_ENABLED", "GRAPH_RAG_V3_MAX_FACTS_PER_ANCHOR",
    "FREEZE_METADATA_AFTER_STEP1",
]
