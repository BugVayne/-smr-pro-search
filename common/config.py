"""Centralised configuration for all services.

Reads from environment variables with sensible defaults. All paths are
resolved relative to the project root so any service can be launched
from any working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
INDICES_DIR = DATA_DIR / "indices"

DB_PATH = DATA_DIR / "smr_pro.db"

# Make sure all directories exist on import
for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, INDICES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Service endpoints -------------------------------------------------------

PREPROCESS_URL = os.getenv("PREPROCESS_URL", "http://localhost:5001")
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://localhost:5002")
RANKING_URL = os.getenv("RANKING_URL", "http://localhost:5003")
POSTPROCESS_URL = os.getenv("POSTPROCESS_URL", "http://localhost:5004")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:5000")

# --- Service ports (used by run scripts) -------------------------------------

PORTS = {
    "preprocessing": 5001,
    "retrieval": 5002,
    "ranking": 5003,
    "postprocessing": 5004,
    "gateway": 5000,
    "ui": 8080,
}

# --- Model files -------------------------------------------------------------

SBERT_MODEL_DIR = MODELS_DIR / "sbert"
FASTTEXT_MODEL_PATH = MODELS_DIR / "fasttext.model"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_ranker.cbm"
QTYPE_MODEL_PATH = MODELS_DIR / "query_type_classifier.pkl"
SYMSPELL_DICT_PATH = MODELS_DIR / "symspell_dict.txt"

# --- Index files -------------------------------------------------------------

BM25_INDEX_PATH = INDICES_DIR / "bm25_index.pkl"
FAISS_INDEX_PATH = INDICES_DIR / "faiss.index"
FAISS_ID_MAP_PATH = INDICES_DIR / "faiss_id_map.pkl"

# --- Training hyper-parameters ----------------------------------------------

FP_SUPPORT_MIN = 0.03
FP_CONFIDENCE_MIN = 0.6
FP_LIFT_MIN = 1.0
FP_MAX_LEN = 8

SBERT_BASE_MODEL = "DeepPavlov/rubert-base-cased-sentence"
SBERT_EMBED_DIM = 768

RETRIEVAL_TOP_K = 100
RANKING_TOP_K = 20
RRF_K = 60

# --- Domain rules ------------------------------------------------------------

# Only these tip values are kept when forming transactions from XML
ALLOWED_TIP = {"100", "101", "103"}
