from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRUNGLAP_DATA", str(ROOT / "data")))
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "indexes"
DB_PATH = DATA_DIR / "app.db"

# Force models to be saved in the tool's folder (models/) for portability
MODEL_DIR = Path(os.environ.get("TRUNGLAP_MODELS", str(ROOT / "models")))
HF_HOME = MODEL_DIR

# Set environment variables immediately so any lazy imports (like HF) pick it up
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_DIR)
APP_NAME = "Check trùng lặp"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBED_DIM = 768
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
EMBED_BATCH_GPU = 32
EMBED_BATCH_CPU = 8
MINHASH_PERM = 96
MINHASH_NGRAM = 4
DEFAULT_WARN = 70
DEFAULT_BLOCK = 90
MATCH_KEEP_MIN = 28.0
SEARCH_K = 24
COS_FLOOR = 0.58
COS_CEIL = 0.995
COS_COVER = 0.70
COS_HIT = 0.50

SUPPORTED_EXT = {
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".docx",
    ".pdf",
    ".rtf",
    ".odt",
    ".html",
    ".htm",
    ".epub",
}


def ensure_dirs() -> None:
    for p in (DATA_DIR, DOCS_DIR, INDEX_DIR, MODEL_DIR, HF_HOME):
        p.mkdir(parents=True, exist_ok=True)
