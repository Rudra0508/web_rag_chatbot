"""
config/settings.py
Central configuration — loaded once, imported everywhere.
All values come from the .env file or their defaults below.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"        # updated: llama3-8b-8192 is decommissioned

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"       # no prefix — matches SentenceTransformer usage

    # ── Scraper ───────────────────────────────────────────────────────────────
    scraper_request_timeout: int = 30
    scraper_max_retries: int = 3
    scraper_delay_seconds: float = 1.5

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # ── RAG ───────────────────────────────────────────────────────────────────
    chunk_size: int = 400           # updated: matches DEFAULT_CHUNK_SIZE in embedder.py
    chunk_overlap: int = 40         # updated: matches DEFAULT_CHUNK_OVERLAP in embedder.py
    top_k_results: int = 5
    similarity_threshold: float = 0.7

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_dir: str = "./logs"

    # ── Derived paths (not from .env) ─────────────────────────────────────────
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def chroma_dir(self) -> Path:
        return self.base_dir / "chroma_db"


# Singleton — import this everywhere: `from config.settings import settings`
settings = Settings()