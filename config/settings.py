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
    groq_model: str = "llama3-8b-8192"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Vector Store ──────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/vector_store"
    chroma_collection_name: str = "web_rag_collection"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_ttl_seconds: int = 86400

    # ── Scraper ───────────────────────────────────────────────────────────────
    scraper_request_timeout: int = 30
    scraper_max_retries: int = 3
    scraper_delay_seconds: float = 1.5
    scraper_max_depth: int = 2
    scraper_max_pages: int = 50

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_secret_key: str = "change-me-in-production"

    # ── RAG ───────────────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50
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


# Singleton — import this everywhere: `from config.settings import settings`
settings = Settings()