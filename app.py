"""
web-rag-chatbot — Entry Point
Run with: python app.py
"""

import sys
from pathlib import Path

# Make sure the src/ package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env (if it exists)
load_dotenv()


def main() -> None:
    logger.info("Starting Web RAG Chatbot…")

    # ── Quick environment sanity-check ────────────────────────────────────────
    import os

    required_vars = ["GROQ_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]

    if missing:
        logger.warning(
            f"Missing environment variables: {missing}. "
            "Copy .env.example to .env and fill in your values."
        )
    else:
        logger.success("Environment variables loaded successfully.")

    # ── Confirm project structure ─────────────────────────────────────────────
    expected_dirs = [
        "src/scraper", "src/processor", "src/embeddings",
        "src/retrieval", "src/llm", "src/api", "src/frontend",
        "src/utils", "data/raw", "data/processed", "data/vector_store",
        "logs", "tests", "scripts", "config",
    ]
    base = Path(__file__).parent
    all_ok = all((base / d).exists() for d in expected_dirs)

    if all_ok:
        logger.success("Project structure verified.")
    else:
        missing_dirs = [d for d in expected_dirs if not (base / d).exists()]
        logger.warning(f"Missing directories: {missing_dirs}")

    print("\n" + "=" * 50)
    print("  ✅  Project setup complete!")
    print("=" * 50 + "\n")
    print("Next steps:")
    print("  1. Copy .env.example → .env and add your GROQ_API_KEY")
    print("  2. pip install -r requirements.txt")
    print("  3. python -m playwright install chromium")
    print("  4. Start building Phase 2: Web Scraper\n")


if __name__ == "__main__":
    main()