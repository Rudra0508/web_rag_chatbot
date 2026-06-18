"""
src/preprocessor/cleaner.py
==============================
PHASE 3 — Data Cleaning

This file takes the messy raw data we scraped in Phase 2 and turns it
into clean, readable text that's ready for chunking and embeddings later.

Beginner notes:
- Scraped web pages are full of "noise": cookie banners, "Share on
  Facebook" buttons, broken characters (like "Ã©" instead of "é"), and
  inconsistent spacing. None of that is useful to an AI chatbot.
- This file removes all of that junk in clear stages, one function per
  stage, so each step is easy to test and understand on its own.
"""

import json                  # for saving our cleaned result as a .json file
import re                    # regex — pattern matching to find/remove junk text
import unicodedata            # for normalizing unicode characters
from datetime import datetime, timezone  # to timestamp when cleaning happened
from pathlib import Path     # modern way to work with file paths

import ftfy                          # fixes garbled/broken text encoding
import trafilatura                   # extracts clean main content from raw HTML
from langdetect import detect, LangDetectException  # detects the language of text
from loguru import logger            # pretty, structured logging


# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────

# Folder where every cleaned result gets saved as JSON.
# parents[2] goes: cleaner.py -> preprocessor/ -> src/ -> project root.
CLEAN_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "clean"

# If the final cleaned text has fewer words than this, we log a warning —
# it probably means the scrape/clean didn't capture real content.
MIN_WORD_COUNT_WARNING = 100


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — trafilatura_clean
# ──────────────────────────────────────────────────────────────────────────

def trafilatura_clean(raw_html: str) -> str:
    """
    Removes boilerplate (ads, navigation menus, footers, sidebars) from
    raw HTML and keeps only the main article/content text.

    Args:
        raw_html: the full HTML source of a webpage (as a string).

    Returns:
        Clean main-content text. If trafilatura can't find anything
        (returns None), we fall back to the original raw_html string
        so we never lose data completely.
    """
    # trafilatura.extract() reads the HTML structure and uses heuristics
    # to guess which part of the page is the "real" article content,
    # discarding menus/ads/footers automatically.
    extracted = trafilatura.extract(raw_html)

    # trafilatura returns None if it couldn't confidently find content
    # (e.g. the HTML was too short, malformed, or had no clear article).
    if extracted is None:
        logger.warning("[trafilatura_clean] Extraction failed, falling back to raw text.")
        # Fallback: return the original input so downstream steps still
        # have *something* to work with, instead of crashing on None.
        return raw_html

    # Success — return the clean extracted text.
    return extracted


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — fix_encoding
# ──────────────────────────────────────────────────────────────────────────

def fix_encoding(text: str) -> str:
    """
    Fixes broken/garbled characters (a problem called "mojibake", e.g.
    "café" showing up as "cafÃ©") and normalizes unicode so equivalent
    characters are represented the same way internally.

    Args:
        text: possibly broken text.

    Returns:
        Text with encoding issues fixed and unicode normalized.
    """
    # ftfy ("fixes text for you") detects common encoding mistakes and
    # repairs them automatically — this is the single best tool for this.
    fixed = ftfy.fix_text(text)

    # NFKC normalization converts visually-similar unicode characters
    # into a single standard form, e.g. full-width "Ａ" becomes regular
    # "A", and combined accent characters become their standard form.
    # This keeps later text processing (chunking, embeddings) consistent.
    normalized = unicodedata.normalize("NFKC", fixed)

    return normalized


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — remove_noise
# ──────────────────────────────────────────────────────────────────────────

def remove_noise(text: str) -> str:
    """
    Uses regex to strip out common junk patterns that survive even
    after trafilatura's extraction: cookie banners, share buttons,
    newsletter prompts, and messy whitespace.

    Args:
        text: text to clean.

    Returns:
        Text with noise patterns removed and whitespace collapsed.
    """
    # Split the text into individual lines so we can check each one
    # separately — easier than trying to match across the whole blob.
    lines = text.split("\n")

    # Patterns (lowercase) that mark a line as junk we want to drop
    # entirely. We check for these as substrings, not exact matches,
    # so "Please accept all cookies to continue" still gets caught.
    junk_substrings = [
        "cookie",                    # cookie consent banners
        "privacy policy",            # privacy policy notices
        "accept all",                # "Accept All" cookie buttons
        "share on twitter",          # social share prompts
        "share on facebook",         # social share prompts
        "subscribe to our newsletter",  # email subscription prompts
    ]

    cleaned_lines = []  # will hold only the lines we decide to KEEP

    for line in lines:
        # Step 1: skip lines that are empty or only whitespace —
        # strip() removes spaces/tabs, so "" or "   " both become "".
        if not line.strip():
            continue

        # Step 2: lowercase a copy of the line just for comparison,
        # so matching is case-insensitive ("Cookie" == "cookie").
        lowered = line.lower()

        # Step 3: if ANY junk substring appears anywhere in this line,
        # skip the whole line — any() stops as soon as one match is found.
        if any(junk in lowered for junk in junk_substrings):
            continue

        # Step 4: line survived all checks — keep it.
        cleaned_lines.append(line)

    # Rejoin the surviving lines back into one block of text.
    text = "\n".join(cleaned_lines)

    # Step 5: collapse repeated whitespace.
    # re.sub(pattern, replacement, text) finds all matches of `pattern`
    # and replaces them with `replacement`.
    # r"[ \t]+"  -> one or more spaces/tabs in a row -> replace with one space.
    text = re.sub(r"[ \t]+", " ", text)
    # r"\n{2,}"  -> two or more newlines in a row -> replace with one newline.
    text = re.sub(r"\n{2,}", "\n", text)

    # Step 6: trim any leading/trailing whitespace left on the whole text.
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — detect_language
# ──────────────────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detects what language the given text is written in.

    Args:
        text: the text to analyse.

    Returns:
        A short language code like 'en' (English), 'fr' (French), etc.
        Returns 'unknown' if detection fails (e.g. text too short/empty).
    """
    try:
        # langdetect.detect() analyses character/word patterns and
        # returns the most likely ISO 639-1 language code.
        return detect(text)
    except LangDetectException as e:
        # This happens most often when `text` is empty or has no
        # detectable language features (e.g. just numbers/symbols).
        logger.warning(f"[detect_language] Could not detect language: {e}")
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — clean_text  (the main pipeline)
# ──────────────────────────────────────────────────────────────────────────

def clean_text(raw_data_dict: dict) -> dict:
    """
    The main function you call from outside this file.
    Takes the dictionary produced by Phase 2's scraper and runs it
    through the full cleaning pipeline:
        trafilatura_clean -> fix_encoding -> remove_noise -> detect_language

    Args:
        raw_data_dict: dict from the scraper. Expected to contain
            either "raw_html" (preferred, for trafilatura) or
            "raw_text" (fallback, already-extracted text).

    Returns:
        A new dict: {url, clean_text, word_count, language, cleaned_at}
        This is also saved to disk as a JSON file in data/clean/.
    """
    url = raw_data_dict.get("url", "unknown_url")
    logger.info(f"[clean_text] Starting cleaning for: {url}")

    # Step 1: Run trafilatura on raw HTML if we have it (most accurate).
    # If only raw_text (already tag-extracted) is available, we skip
    # straight to that, since trafilatura needs real HTML to work on.
    if raw_data_dict.get("raw_html"):
        text = trafilatura_clean(raw_data_dict["raw_html"])
    else:
        logger.warning(
            f"[clean_text] No 'raw_html' found for {url}; "
            "using 'raw_text' from the scraper as-is."
        )
        text = raw_data_dict.get("raw_text", "")

    # Step 2: Fix any broken characters / normalize unicode.
    text = fix_encoding(text)

    # Step 3: Strip out cookie banners, share buttons, extra whitespace.
    text = remove_noise(text)

    # Step 4: Detect what language the cleaned text is in.
    language = detect_language(text)

    # Count words by splitting on whitespace — a simple but effective
    # word count for sanity-checking the result.
    word_count = len(text.split())

    # Step 5: Warn (but don't crash) if the result looks suspiciously short
    # — this usually means scraping or cleaning lost most of the content.
    if word_count < MIN_WORD_COUNT_WARNING:
        logger.warning(
            f"[clean_text] '{url}' only has {word_count} words after "
            f"cleaning (expected at least {MIN_WORD_COUNT_WARNING}). "
            "Check if scraping/cleaning worked correctly."
        )

    # Build the final result dictionary in the exact shape requested.
    result = {
        "url": url,
        "clean_text": text,
        "word_count": word_count,
        "language": language,
        # isoformat() gives a standard, sortable timestamp string like
        # "2026-06-16T12:34:56.789012+00:00" — timezone-aware (UTC).
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.success(
        f"[clean_text] Finished cleaning {url}: "
        f"{word_count} words, language='{language}'."
    )

    # Save it to disk so Phase 4 (chunking/embeddings) can pick it up.
    _save_clean_result(result)

    return result


def _save_clean_result(result: dict) -> None:
    """
    Helper function: saves a cleaned result dictionary as a JSON file
    inside data/clean/, named using the same kind of short hash we used
    in Phase 2, so filenames stay short and filesystem-safe.
    """
    import hashlib  # local import — only needed inside this helper

    # Make sure data/clean/ exists (creates it if missing).
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Hash the URL the same way Phase 2 did, so it's easy to match a
    # cleaned file back to its raw counterpart later if needed.
    url_hash = hashlib.md5(result["url"].encode("utf-8")).hexdigest()[:16]

    file_path = CLEAN_DATA_DIR / f"{url_hash}.json"

    # Write the dictionary to disk as nicely-formatted, readable JSON.
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"[_save_clean_result] Saved cleaned result to: {file_path}")


# ──────────────────────────────────────────────────────────────────────────
# CLI — lets you run this file directly on an already-saved raw JSON file
# from Phase 2, e.g.:
#   python src/preprocessor/cleaner.py data/raw/<hash>.json
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse  # built-in library for reading command-line arguments

    parser = argparse.ArgumentParser(
        description="Clean a raw scraped JSON file (from Phase 2) and save the result to data/clean/."
    )
    parser.add_argument(
        "raw_json_path",
        help="Path to a raw JSON file saved by the Phase 2 scraper (e.g. data/raw/abc123.json).",
    )
    args = parser.parse_args()

    # Load the raw scraped dictionary from disk.
    with open(args.raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Run it through the full cleaning pipeline.
    result = clean_text(raw_data)

    # Print a preview so you can visually confirm the output looks right.
    print("\n--- CLEANED TEXT PREVIEW ---")
    print("URL:", result["url"])
    print("Language:", result["language"])
    print("Word count:", result["word_count"])
    print("Cleaned at:", result["cleaned_at"])
    print("\nFirst 500 characters of clean_text:\n")
    print(result["clean_text"][:500])


if __name__ == "__main__":
    main()