from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
MEMORY_FILE = BASE_DIR / "memory" / "rag_notes.md"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "220"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.45"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.55"))

    @property
    def active_model_name(self) -> str:
        return self.openai_model
