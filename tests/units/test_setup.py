"""tests/unit/test_setup.py — Verify the project structure is correct."""

import importlib
from pathlib import Path


BASE = Path(__file__).parent.parent.parent


def test_project_root_exists():
    assert BASE.exists()


def test_required_directories_exist():
    dirs = [
        "src/scraper", "src/processor", "src/embeddings",
        "src/retrieval", "src/llm", "src/api", "src/frontend", "src/utils",
        "config", "tests/unit", "tests/integration",
        "scripts", "notebooks", "docs",
    ]
    for d in dirs:
        assert (BASE / d).is_dir(), f"Missing directory: {d}"


def test_required_files_exist():
    files = [
        "requirements.txt", ".env.example", ".gitignore",
        "app.py", "Makefile", "pyproject.toml", "README.md",
        "config/settings.py",
    ]
    for f in files:
        assert (BASE / f).is_file(), f"Missing file: {f}"


def test_dotenv_example_has_groq_key():
    env_example = (BASE / ".env.example").read_text()
    assert "GROQ_API_KEY" in env_example


def test_gitignore_excludes_dotenv():
    gitignore = (BASE / ".gitignore").read_text()
    assert ".env" in gitignore