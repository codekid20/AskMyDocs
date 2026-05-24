"""Typed settings loaded once at process start from .env + environment.

Everything tunable in the pipeline lives here, so later phases don't sprout
magic numbers across modules. Import `settings` anywhere you need them.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Paths ---
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    sections_path: Path = PROJECT_ROOT / "data" / "processed" / "sections.jsonl"
    chunks_path: Path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    chroma_dir: Path = PROJECT_ROOT / "data" / "processed" / "chroma_db"

    # --- Models (used in later phases, declared now to keep config in one place) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"

    # --- Chunking ---
    chunk_size: int = 800  # characters; we'll revisit when we see the text
    min_chunk_chars: int = 50  # drop fragments too small to carry meaning
    chunk_overlap: int = 120

    # --- Retrieval ---
    top_k: int = 5
    retrieve_n_before_rerank: int = 20

    # --- Secrets (pulled from .env via pydantic-settings) ---
    groq_api_key: str = Field(default="")


settings = Settings()
