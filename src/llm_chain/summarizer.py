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
     sentiment, and topic, as structured JSON. This needs internet +
     your Groq API key, unlike the two methods above.

Beginner notes:
- "Extractive" summarization = picking existing sentences (like
  highlighting a textbook). "Abstractive" summarization = writing NEW
  sentences that didn't exist before (like the LLM does here) — this
  file actually uses both styles, one offline and one via the LLM.
- "Entities" just means specific NAMED things in text: people,
  companies, places, dates — spaCy is a library trained to recognise these.
"""

import json                 # for parsing the LLM's JSON response safely

import spacy                                        # NLP library used for entity extraction
from groq import Groq                                # type hint only — the real client is passed in
from loguru import logger                            # pretty, structured logging
from sumy.nlp.tokenizers import Tokenizer            # splits text into sentences for sumy
from sumy.parsers.plaintext import PlaintextParser   # wraps raw text for sumy to process
from sumy.summarizers.lex_rank import LexRankSummarizer  # the actual summarization algorithm


# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────

# Same Groq model Phase 6's rag_chain.py uses — kept consistent so the
# whole project only ever talks to one model unless you deliberately
# change it in both places.
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# How many characters of text spaCy processes for entity extraction.
# Limiting this keeps it fast — entity extraction on a 100,000-character
# article would be slow for little extra benefit, since the same kinds
# of names/places/dates tend to repeat throughout a page anyway.
ENTITY_TEXT_LIMIT = 5000

# How many WORDS of text we send to the LLM for the structured summary.
# Keeps the prompt small (faster, cheaper, fits comfortably in context).
LLM_SUMMARY_WORD_LIMIT = 3000

# Average adult reading speed in words per minute — used to estimate
# how long the full article would take to read.
AVERAGE_READING_SPEED_WPM = 200

# Maximum number of entities kept per category, after de-duplication.
MAX_ENTITIES_PER_CATEGORY = 10

# Lazily-loaded spaCy model — loaded once on first use, not at import
# time, since loading it takes a moment and not every script that
# imports this file necessarily needs entity extraction.
_nlp = None


def _get_spacy_model():
    """
    Returns the loaded spaCy English model, loading it on first call
    and reusing it on every call after that (a simple cache).
    """
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
    Picks the most important EXISTING sentences from the text using the
    LexRank algorithm (similar idea to how Google ranks important web
    pages, but applied to ranking sentences by importance within one
    document). Fully offline — no API calls, no cost, always available
    even if Groq is down or your API key runs out of quota.

    Args:
        clean_text: the full cleaned document text (from Phase 3).
        sentence_count: how many top sentences to extract.

    Returns:
        The extracted sentences joined into one string.
    """
    # PlaintextParser reads our raw text and breaks it into a structure
    # sumy understands (it needs to know about sentences/words, which is
    # why we give it a Tokenizer set to "english" rules).
    parser = PlaintextParser.from_string(clean_text, Tokenizer("english"))

    # LexRankSummarizer implements the actual ranking algorithm: it builds
    # a graph of how similar every sentence is to every other sentence,
    # then picks the sentences most "central" to the overall document
    # (similar to how important web pages get many other pages linking to them).
    summarizer = LexRankSummarizer()

    # __call__ on the summarizer does the work: it returns sentence_count
    # Sentence objects, in the order they appeared in the ORIGINAL text
    # (not necessarily in "importance order").
    summary_sentences = summarizer(parser.document, sentence_count)

    # Each item is a sumy Sentence object, not a plain string — str()
    # converts it to its actual text content. " ".join(...) glues all
    # the chosen sentences into one readable paragraph.
    summary_text = " ".join(str(sentence) for sentence in summary_sentences)

    logger.info(f"[extractive_summary] Extracted {len(summary_sentences)} sentences "
                f"from {len(clean_text)} characters of text.")

    return summary_text


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — extract_entities
# ──────────────────────────────────────────────────────────────────────────

def extract_entities(clean_text: str) -> dict:
    """
    Finds named entities (people, organizations, places, dates) that
    appear in the text, using spaCy's pre-trained statistical model.

    Args:
        clean_text: the full cleaned document text (from Phase 3).

    Returns:
        A dict: {persons: [...], organizations: [...], locations: [...], dates: [...]}
        Each list has at most MAX_ENTITIES_PER_CATEGORY unique items.
    """
    nlp = _get_spacy_model()

    # Only process the first ENTITY_TEXT_LIMIT characters — entity types
    # tend to repeat throughout a document (the same people/places get
    # mentioned multiple times), so this keeps things fast without
    # meaningfully losing coverage.
    doc = nlp(clean_text[:ENTITY_TEXT_LIMIT])

    # These will temporarily hold ALL matches (including duplicates);
    # we deduplicate and trim each one further down.
    persons, organizations, locations, dates = [], [], [], []

    # doc.ents is spaCy's list of every entity it found, each with a
    # .text (the actual words) and a .label_ (what TYPE of entity it is).
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text)
        elif ent.label_ == "ORG":
            organizations.append(ent.text)
        elif ent.label_ == "GPE":  # GPE = Geo-Political Entity (countries, cities, states)
            locations.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)
        # Any other entity type (MONEY, PERCENT, etc.) is intentionally
        # ignored — we only care about these 4 categories per the spec.

    def _dedupe_and_limit(items: list[str]) -> list[str]:
        """Removes duplicates while keeping the first-seen order, then
        trims to at most MAX_ENTITIES_PER_CATEGORY items."""
        # dict.fromkeys() is a common Python trick: it keeps only the
        # first occurrence of each item while preserving order (a plain
        # set() would also dedupe, but loses the original ordering).
        unique_items = list(dict.fromkeys(items))
        return unique_items[:MAX_ENTITIES_PER_CATEGORY]

    entities = {
        "persons": _dedupe_and_limit(persons),
        "organizations": _dedupe_and_limit(organizations),
        "locations": _dedupe_and_limit(locations),
        "dates": _dedupe_and_limit(dates),
    }

    logger.info(
        f"[extract_entities] Found {len(entities['persons'])} persons, "
        f"{len(entities['organizations'])} orgs, {len(entities['locations'])} locations, "
        f"{len(entities['dates'])} dates."
    )

    return entities


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — llm_structured_summary
# ──────────────────────────────────────────────────────────────────────────

def llm_structured_summary(clean_text: str, groq_client: Groq) -> dict:
    """
    Asks the LLM to read the text and return a structured JSON summary:
    title, one-line summary, key points, sentiment, and topic.

    Args:
        clean_text: the full cleaned document text (from Phase 3).
        groq_client: an already-created Groq client (e.g. the same
            `client` object built in src/llm_chain/rag_chain.py) — this
            function does NOT create its own client, so the whole
            project shares one client/API key consistently.

    Returns:
        A dict with keys: title, one_line_summary, key_points,
        sentiment, topic. If the LLM's response isn't valid JSON, a
        safe default structure is returned instead (with an "error" key)
        so callers never crash on a malformed LLM reply.
    """
    # Take only the first LLM_SUMMARY_WORD_LIMIT words — .split() breaks
    # text into words, [:n] takes the first n, " ".join(...) glues them
    # back into one string. This keeps the prompt a manageable size.
    limited_text = " ".join(clean_text.split()[:LLM_SUMMARY_WORD_LIMIT])

    # The exact prompt format requested — triple-quoted so we can write
    # it across multiple lines exactly as it will be sent to the model.
    prompt = f"""Summarize this text. Reply ONLY with valid JSON in this exact format:
{{
  "title": "one sentence title",
  "one_line_summary": "one sentence that explains the whole page",
  "key_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "sentiment": "positive or negative or neutral",
  "topic": "main topic in 2-3 words"
}}

TEXT:
{limited_text}"""

    # This default gets returned if ANYTHING goes wrong below — a
    # network error, a bad API response, or unparseable JSON — so
    # generate_knowledge_card() never crashes just because the LLM
    # step had a problem.
    default_result = {
        "title": "Unknown",
        "one_line_summary": "Summary unavailable.",
        "key_points": [],
        "sentiment": "unknown",
        "topic": "unknown",
        "error": None,
    }

    try:
        # Send our prompt to Groq exactly like rag_chain.py's get_answer()
        # does — one user message, asking for a completion back.
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )

        # Pull the actual generated text out of the response object.
        raw_reply = response.choices[0].message.content

        # LLMs sometimes wrap JSON in markdown code fences like ```json
        # ... ``` even when told not to — strip those off if present,
        # so json.loads() below doesn't choke on the ``` characters.
        cleaned_reply = raw_reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        # Attempt to parse the cleaned text as actual JSON.
        parsed = json.loads(cleaned_reply)

        # Merge into default_result so even if the LLM's JSON is missing
        # a key, we still return every key generate_knowledge_card() expects.
        default_result.update(parsed)
        default_result["error"] = None

        logger.success("[llm_structured_summary] Successfully parsed LLM JSON response.")

    except json.JSONDecodeError as e:
        # The LLM's reply wasn't valid JSON — log it and fall back to
        # the safe default instead of crashing the whole pipeline.
        logger.warning(f"[llm_structured_summary] Failed to parse LLM JSON: {e}")
        default_result["error"] = f"JSON parsing failed: {e}"

    except Exception as e:
        # Catches anything else (network error, API error, missing key,
        # rate limit, etc.) so this function NEVER raises an exception
        # that could crash generate_knowledge_card().
        logger.warning(f"[llm_structured_summary] LLM call failed: {e}")
        default_result["error"] = f"LLM call failed: {e}"

    return default_result


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — generate_knowledge_card (the main pipeline)
# ──────────────────────────────────────────────────────────────────────────

def generate_knowledge_card(clean_data_dict: dict, groq_client: Groq) -> dict:
    """
    The main function you call from outside this file. Takes ONE
    cleaned document dictionary (exactly what Phase 3's cleaner.py
    saves into data/clean/*.json) and runs all three summary techniques
    on it, combining everything into one "knowledge card" dictionary.

    Args:
        clean_data_dict: dict with at least "url" and "clean_text" keys
            (the exact shape produced by src/preprocessor/cleaner.py).
        groq_client: an already-created Groq client to use for the LLM step.

    Returns:
        A dict: {url, title, one_line_summary, key_points, entities,
        sentiment, extractive_summary, word_count,
        estimated_read_time_minutes, topic}
    """
    url = clean_data_dict.get("url", "unknown_url")
    clean_text = clean_data_dict.get("clean_text", "")

    logger.info(f"[generate_knowledge_card] Starting Phase 7 pipeline for: {url}")

    # word_count: prefer the value Phase 3 already calculated (so we
    # don't disagree with cleaner.py's own count), but recompute as a
    # fallback if it's missing for any reason.
    word_count = clean_data_dict.get("word_count") or len(clean_text.split())

    # Step 1 — offline extractive summary (always works, no API needed).
    extractive = extractive_summary(clean_text)

    # Step 2 — offline entity extraction (always works, no API needed).
    entities = extract_entities(clean_text)

    # Step 3 — LLM-generated structured summary (needs Groq + internet).
    llm_summary = llm_structured_summary(clean_text, groq_client)

    # Reading time: word_count divided by average reading speed gives
    # minutes. round(..., 1) keeps it to one decimal place (e.g. 4.3 min).
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
# CLI — lets you run this file directly on an already-cleaned JSON file
# from Phase 3, e.g.:
#   python src/llm_chain/summarizer.py data/clean/<hash>.json
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse  # built-in library for reading command-line arguments
    import os        # for reading GROQ_API_KEY from the environment

    from dotenv import load_dotenv  # loads variables from a .env file

    parser = argparse.ArgumentParser(
        description="Generate a knowledge card (Phase 7) from a cleaned JSON file (Phase 3)."
    )
    parser.add_argument(
        "clean_json_path",
        help="Path to a cleaned JSON file saved by Phase 3's cleaner.py (e.g. data/clean/abc123.json).",
    )
    args = parser.parse_args()

    # Load .env so GROQ_API_KEY is available, same as rag_chain.py does.
    load_dotenv()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Load the cleaned dictionary exactly as cleaner.py saved it.
    with open(args.clean_json_path, "r", encoding="utf-8") as f:
        clean_data = json.load(f)

    # Run the full Phase 7 pipeline on it.
    card = generate_knowledge_card(clean_data, groq_client)

    # Print the knowledge card in a clean, readable format.
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

    print(f"\nExtractive summary (top sentences from the original text):\n  {card['extractive_summary']}")
    print("=" * 60)


if __name__ == "__main__":
    main()