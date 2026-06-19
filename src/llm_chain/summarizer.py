"""
src/llm_chain/summarizer.py
===============================
PHASE 7 — Auto Summary

This file turns one big cleaned document (from Phase 3) into a compact
"knowledge card" — a short, scannable summary of what the page is
about, who/what it mentions, and its key points. Three different
techniques are combined:

  1. EXTRACTIVE summary (sumy + LexRank) — picks the most important
     SENTENCES that already exist in the text. Fully offline, free,
     no API calls. Good as a fast, reliable fallback.
  2. ENTITY extraction (spaCy) — finds names of people, organizations,
     places, and dates mentioned in the text. Also fully offline.
  3. LLM-structured summary (Groq) — asks an actual LLM to read the
     text and produce a clean title, one-line summary, key points,
     sentiment, and topic, as structured JSON. Needs Groq API key.
"""

import httpx                  # explicit HTTP transport with timeout control
import json                  # for parsing the LLM's JSON response safely
import re                    # for filtering citations and fixing JSON

import httpx                 # for transport-level timeout on Groq client

import spacy                                         # NLP library for entity extraction
from groq import Groq                                 # type hint — real client is passed in
from loguru import logger                             # pretty, structured logging
from sumy.nlp.tokenizers import Tokenizer             # splits text into sentences for sumy
from sumy.parsers.plaintext import PlaintextParser    # wraps raw text for sumy to process
from sumy.summarizers.lex_rank import LexRankSummarizer  # the summarization algorithm


# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────

# Same Groq model Phase 6's rag_chain.py uses — consistent across the project.
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# How many characters spaCy processes for entity extraction.
# Entities repeat throughout a document, so 5000 chars gives good coverage fast.
ENTITY_TEXT_LIMIT = 5000

# How many WORDS we send to the LLM. 800 words is enough for a good
# summary and avoids timeouts on Groq's free tier — 3000 words can
# cause the request to hang, especially for complex Wikipedia articles.
LLM_SUMMARY_WORD_LIMIT = 800

# LexRank is O(n^2) in sentence count — cap input to keep it fast.
# 20000 chars takes ~0.3 seconds vs 22+ seconds for a full Wikipedia article.
EXTRACTIVE_SUMMARY_TEXT_LIMIT = 20000

# Average adult reading speed — used to estimate read time.
AVERAGE_READING_SPEED_WPM = 200

# Max entities kept per category after deduplication.
MAX_ENTITIES_PER_CATEGORY = 10

# Lazily-loaded spaCy model — loaded once on first use, reused after.
_nlp = None


def _get_spacy_model():
    """Loads spaCy model on first call, returns cached model on every call after."""
    global _nlp
    if _nlp is None:
        logger.info("[summarizer] Loading spaCy model: en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
        logger.success("[summarizer] spaCy model loaded.")
    return _nlp


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — extractive_summary
# ──────────────────────────────────────────────────────────────────────────

def extractive_summary(clean_text: str, sentence_count: int = 5) -> str:
    """
    Picks the most important existing sentences from the text using
    the LexRank algorithm. Fully offline, no API needed.

    Args:
        clean_text: the full cleaned document text (from Phase 3).
        sentence_count: how many top sentences to extract.

    Returns:
        The extracted sentences joined into one string.
    """
    # Cap input length — LexRank is O(n^2) in sentence count,
    # so a full Wikipedia article would take 20+ seconds without this.
    clean_text = clean_text[:EXTRACTIVE_SUMMARY_TEXT_LIMIT]

    # Filter out Wikipedia citation lines like "^ Einstein (1926b)."
    # These confuse LexRank by being repetitively short and structured.
    citation_pattern = re.compile(r"^\s*\^\s")
    lines = clean_text.split("\n")
    filtered_lines = [line for line in lines if not citation_pattern.match(line)]
    filtered_text = "\n".join(filtered_lines)

    # PlaintextParser wraps raw text into a structure sumy understands.
    parser = PlaintextParser.from_string(filtered_text, Tokenizer("english"))

    # LexRankSummarizer ranks sentences by how "central" they are —
    # similar to how PageRank ranks web pages by links.
    summarizer = LexRankSummarizer()

    # Run the summarizer — returns Sentence objects, not plain strings.
    summary_sentences = summarizer(parser.document, sentence_count)

    # Convert Sentence objects to strings and join into one paragraph.
    summary_text = " ".join(str(sentence) for sentence in summary_sentences)

    logger.info(
        f"[extractive_summary] Extracted {len(summary_sentences)} sentences "
        f"from {len(clean_text)} characters of text."
    )

    return summary_text


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — extract_entities
# ──────────────────────────────────────────────────────────────────────────

def extract_entities(clean_text: str) -> dict:
    """
    Finds named entities (people, organizations, places, dates)
    using spaCy's pre-trained English model. Fully offline.

    Args:
        clean_text: the full cleaned document text (from Phase 3).

    Returns:
        {persons: [...], organizations: [...], locations: [...], dates: [...]}
    """
    nlp = _get_spacy_model()

    # Only process the first ENTITY_TEXT_LIMIT chars for speed.
    doc = nlp(clean_text[:ENTITY_TEXT_LIMIT])

    persons, organizations, locations, dates = [], [], [], []

    # doc.ents has every entity spaCy found, with .text and .label_.
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text)
        elif ent.label_ == "ORG":
            organizations.append(ent.text)
        elif ent.label_ == "GPE":   # Geo-Political Entity
            locations.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)

    def _dedupe_and_limit(items: list[str]) -> list[str]:
        """Remove duplicates (keep first occurrence) and trim to limit."""
        return list(dict.fromkeys(items))[:MAX_ENTITIES_PER_CATEGORY]

    entities = {
        "persons": _dedupe_and_limit(persons),
        "organizations": _dedupe_and_limit(organizations),
        "locations": _dedupe_and_limit(locations),
        "dates": _dedupe_and_limit(dates),
    }

    logger.info(
        f"[extract_entities] Found {len(entities['persons'])} persons, "
        f"{len(entities['organizations'])} orgs, "
        f"{len(entities['locations'])} locations, "
        f"{len(entities['dates'])} dates."
    )

    return entities


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — llm_structured_summary
# ──────────────────────────────────────────────────────────────────────────

def llm_structured_summary(clean_text: str, groq_client: Groq) -> dict:
    """
    Asks the LLM to produce a structured JSON summary of the text.
    Returns a safe default dict (never raises) if anything goes wrong.

    Args:
        clean_text: the full cleaned document text (from Phase 3).
        groq_client: an already-created Groq client from rag_chain.py.

    Returns:
        Dict with keys: title, one_line_summary, key_points, sentiment, topic.
    """
    # Take only the first LLM_SUMMARY_WORD_LIMIT words of the real text.
    # THIS was the bug in your debug version — it had a hardcoded string
    # instead of the actual clean_text content.
    limited_text = " ".join(clean_text.split()[:LLM_SUMMARY_WORD_LIMIT])

    # Prompt instructs the model to return ONLY valid JSON — no preamble,
    # no explanation, no markdown fences.
    prompt = f"""Summarize this text. Reply ONLY with valid JSON. No extra text before or after. Use this exact format:
{{
  "title": "one sentence title",
  "one_line_summary": "one sentence that explains the whole page",
  "key_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "sentiment": "positive or negative or neutral",
  "topic": "main topic in 2-3 words"
}}

TEXT:
{limited_text}"""

    # Safe fallback returned whenever anything goes wrong.
    default_result = {
        "title": "Unknown",
        "one_line_summary": "Summary unavailable.",
        "key_points": [],
        "sentiment": "unknown",
        "topic": "unknown",
        "error": None,
    }

    try:
        # Use the passed-in groq_client directly but wrap the call in a
        # thread with a hard timeout — this is the only reliable way to
        # kill a hanging network call on Windows, where DNS resolution
        # can block indefinitely before httpx's connect timeout even fires.
        import threading

        logger.info("[llm_structured_summary] Calling Groq API...")
        print("  → Calling Groq API for structured summary...")

        result_holder = {}  # shared dict to pass result back from thread

        def _call_groq():
            """Runs the Groq API call in a background thread."""
            try:
                resp = groq_client.chat.completions.create(
                    model=GROQ_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                )
                result_holder["response"] = resp
            except Exception as e:
                result_holder["error"] = e

        # Start the API call in a daemon thread (dies if main thread exits)
        t = threading.Thread(target=_call_groq, daemon=True)
        t.start()
        t.join(timeout=45)  # wait max 45 seconds — hard wall clock timeout

        if t.is_alive():
            # Thread is still running after 45s — it's hung on DNS or connect
            raise Exception(
                "Groq API call timed out after 45 seconds. "
                "Check your internet connection or try again."
            )

        if "error" in result_holder:
            raise result_holder["error"]

        response = result_holder["response"]

        # Extract the raw text response from the nested response object.
        raw_reply = response.choices[0].message.content

        # Strip markdown code fences if the model added them anyway.
        cleaned_reply = (
            raw_reply.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        # Extract just the {...} block — handles any stray text before/after.
        json_match = re.search(r"\{.*\}", cleaned_reply, re.DOTALL)
        if json_match:
            cleaned_reply = json_match.group(0)

        # Try parsing the JSON directly.
        try:
            parsed = json.loads(cleaned_reply)
        except json.JSONDecodeError:
            # COMMON LLAMA FAILURE: model writes a literal " inside a JSON
            # string value (e.g. He developed the "theory of relativity".)
            # instead of escaping it as \". Repair that specific pattern
            # then retry — if this also fails the outer except catches it.
            repaired = re.sub(
                r'(?<=[a-zA-Z,\s])"(?=[a-zA-Z\s])',
                r'\\"',
                cleaned_reply,
            )
            parsed = json.loads(repaired)
            logger.info("[llm_structured_summary] Repaired unescaped-quote JSON issue.")

        # Merge parsed values into the default dict so all keys are present
        # even if the model omitted some.
        default_result.update(parsed)
        default_result["error"] = None

        logger.success("[llm_structured_summary] Successfully parsed LLM JSON response.")

    except json.JSONDecodeError as e:
        # Couldn't parse JSON even after repair attempt — log and return defaults.
        logger.warning(f"[llm_structured_summary] Failed to parse LLM JSON: {e}")
        default_result["error"] = f"JSON parsing failed: {e}"

    except Exception as e:
        # Any other failure (network, auth, rate limit, etc.).
        logger.warning(f"[llm_structured_summary] LLM call failed: {e}")
        default_result["error"] = f"LLM call failed: {e}"

    return default_result


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — generate_knowledge_card (the main pipeline)
# ──────────────────────────────────────────────────────────────────────────

def generate_knowledge_card(clean_data_dict: dict, groq_client: Groq) -> dict:
    """
    Main function. Takes a cleaned document dict (from Phase 3) and
    runs all three summary techniques, returning one knowledge card dict.

    Args:
        clean_data_dict: dict with "url" and "clean_text" keys
            (exact shape produced by src/preprocessor/cleaner.py).
        groq_client: an already-created Groq client.

    Returns:
        Dict: {url, title, one_line_summary, key_points, entities,
        sentiment, extractive_summary, word_count,
        estimated_read_time_minutes, topic}
    """
    url = clean_data_dict.get("url", "unknown_url")
    clean_text = clean_data_dict.get("clean_text", "")

    logger.info(f"[generate_knowledge_card] Starting Phase 7 pipeline for: {url}")

    # Use word_count from Phase 3 if available, else recompute.
    word_count = clean_data_dict.get("word_count") or len(clean_text.split())

    # Step 1 — offline extractive summary.
    print("\n[Step 1/3] Running extractive summary (offline)...")
    extractive = extractive_summary(clean_text)
    print(f"  ✓ Done — extracted top sentences")

    # Step 2 — offline entity extraction.
    print("[Step 2/3] Extracting named entities (offline)...")
    entities = extract_entities(clean_text)
    print(f"  ✓ Done — found persons, orgs, locations, dates")

    # Step 3 — LLM structured summary (needs Groq + internet).
    print("[Step 3/3] Generating structured summary via Groq LLM...")
    llm_summary = llm_structured_summary(clean_text, groq_client)
    if llm_summary.get("error"):
        print(f"  ⚠ LLM step had an issue: {llm_summary['error']}")
    else:
        print(f"  ✓ Done — got title, key points, sentiment")

    # reading_time = total words / average reading speed (words per minute).
    estimated_read_time_minutes = round(word_count / AVERAGE_READING_SPEED_WPM, 1)

    knowledge_card = {
        "url": url,
        "title": llm_summary["title"],
        "one_line_summary": llm_summary["one_line_summary"],
        "key_points": llm_summary["key_points"],
        "entities": entities,
        "sentiment": llm_summary["sentiment"],
        "extractive_summary": extractive,
        "word_count": word_count,
        "estimated_read_time_minutes": estimated_read_time_minutes,
        "topic": llm_summary["topic"],
    }

    logger.success(f"[generate_knowledge_card] Finished Phase 7 pipeline for: {url}")

    return knowledge_card


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import os
    import httpx

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        description="Generate a knowledge card from a Phase 3 cleaned JSON file."
    )
    parser.add_argument(
        "clean_json_path",
        help="Path to a cleaned JSON file (e.g. data/clean/abc123.json).",
    )
    args = parser.parse_args()

    load_dotenv()

    # httpx transport timeout prevents silent hanging on Windows —
    # plain Groq() with no timeout will freeze indefinitely if the
    # TCP connection stalls before any data is sent.
    groq_client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
        http_client=httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)
        ),
    )

    with open(args.clean_json_path, "r", encoding="utf-8") as f:
        clean_data = json.load(f)

    card = generate_knowledge_card(clean_data, groq_client)

    print("\n" + "=" * 60)
    print("KNOWLEDGE CARD")
    print("=" * 60)
    print(f"URL:               {card['url']}")
    print(f"Title:             {card['title']}")
    print(f"Topic:             {card['topic']}")
    print(f"Sentiment:         {card['sentiment']}")
    print(f"Word count:        {card['word_count']}")
    print(f"Est. read time:    {card['estimated_read_time_minutes']} minutes")
    print(f"\nOne-line summary:\n  {card['one_line_summary']}")
    print("\nKey points:")
    for i, point in enumerate(card["key_points"], start=1):
        print(f"  {i}. {point}")
    print("\nEntities found:")
    for category, items in card["entities"].items():
        print(f"  {category.capitalize()}: {', '.join(items) if items else '(none found)'}")
    print(f"\nExtractive summary:\n  {card['extractive_summary']}")
    print("=" * 60)


if __name__ == "__main__":
    main()