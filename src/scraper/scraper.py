"""
src/scraper/scraper.py
========================
PHASE 2 — Web Scraper

This file scrapes any webpage and decides automatically whether it needs
a simple HTTP request (fast, "static") or a real browser (slower, "dynamic"
— for pages that load content using JavaScript, like React/Vue apps).

Beginner notes:
- "static" page = the HTML you get from requests.get() ALREADY contains
  the text you want to read (e.g. a Wikipedia article).
- "dynamic" page = the HTML you get from requests.get() is mostly EMPTY
  divs, and the real content is injected later by JavaScript running in
  a browser (e.g. many modern single-page apps). For these we need
  Playwright, which actually opens a browser and waits for JS to run.
"""

import hashlib          # used to turn a URL into a short, safe filename
import json             # used to save our scraped data as a .json file
import time             # used for time.sleep() — being polite to servers
from pathlib import Path  # modern way to work with file paths

import requests                       # makes simple HTTP GET requests
from bs4 import BeautifulSoup         # parses HTML so we can extract tags
from fake_useragent import UserAgent  # generates a realistic browser identity
from playwright.sync_api import sync_playwright  # controls a real browser
from loguru import logger             # pretty, structured logging


# ──────────────────────────────────────────────────────────────────────────
# CONFIG — simple constants so we don't repeat "magic numbers" everywhere
# ──────────────────────────────────────────────────────────────────────────

# Folder where every scraped page gets saved as raw JSON.
# parents[2] goes: scraper.py -> scraper/ -> src/ -> project root.
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# How long to wait (in seconds) for a request before giving up.
REQUEST_TIMEOUT = 15

# Tags we WANT to read text from (the "real content" of a page).
CONTENT_TAGS = ["article", "main", "p", "h1", "h2", "h3"]

# Tags we want to IGNORE completely (menus, ads, footers, code, etc.).
SKIP_TAGS = ["nav", "footer", "header", "script", "style", "aside"]


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — detect_page_type
# ──────────────────────────────────────────────────────────────────────────

def detect_page_type(url: str) -> str:
    """
    Quickly checks if a URL is a 'static' page (content already in the
    HTML) or a 'dynamic' page (content loaded later by JavaScript).

    Returns:
        "static"  -> safe to scrape with requests + BeautifulSoup
        "dynamic" -> needs Playwright (a real browser) to render JS
    """
    # Tell the website we are a normal Chrome browser, not a bot script.
    # Some sites block requests that don't have a User-Agent header.
    headers = {"User-Agent": UserAgent().random}

    try:
        # Step 1: Try a normal, fast GET request (no browser, no JS).
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

        # Step 2: If the server responded with an error (404, 500, etc.),
        # raise an exception so our except block below can catch it.
        response.raise_for_status()

        # Step 3: Parse the raw HTML text into a BeautifulSoup object
        # so we can search through its tags easily.
        soup = BeautifulSoup(response.text, "html.parser")

        # Step 4: Look for the tags that usually hold the "real" content.
        main_content = soup.find_all(["article", "main", "p"])

        # Step 5: Join all the text found inside those tags into one
        # string, and strip leading/trailing whitespace.
        combined_text = " ".join(tag.get_text(strip=True) for tag in main_content)

        # Step 6: Decide based on how much text we actually found.
        # If there's barely any text (less than 200 characters), the
        # real content is probably injected later by JavaScript ->
        # this is a sign of a "dynamic" page.
        if len(combined_text) < 200:
            logger.info(f"[detect_page_type] '{url}' looks DYNAMIC (little/no text in raw HTML).")
            return "dynamic"

        # Otherwise, we already found plenty of readable text -> "static".
        logger.info(f"[detect_page_type] '{url}' looks STATIC (text found in raw HTML).")
        return "static"

    except requests.RequestException as e:
        # If the simple request fails completely (timeout, connection
        # refused, DNS error, etc.), we assume it might need a real
        # browser, so we fall back to "dynamic" as the safer guess.
        logger.warning(f"[detect_page_type] Request failed for '{url}': {e}. Assuming DYNAMIC.")
        return "dynamic"


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — scrape_static
# ──────────────────────────────────────────────────────────────────────────

def scrape_static(url: str) -> dict:
    """
    Scrapes a 'static' page using requests + BeautifulSoup.
    Extracts only text from useful tags (article, main, p, h1, h2, h3)
    and skips junk tags (nav, footer, header, script, style, aside).

    Returns:
        dict with keys: url, raw_text, raw_html, title, scrape_method
    """
    # Generate a random, realistic User-Agent so the site treats us
    # like a normal browser visit instead of an obvious bot.
    headers = {"User-Agent": UserAgent().random}

    # Download the page's HTML. timeout stops us waiting forever.
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

    # Raise an error if the page returned a bad status code (404 etc.).
    response.raise_for_status()
 
    # Keep the FULL, untouched HTML before we start deleting tags below.
    # Phase 3's cleaner (trafilatura) needs this raw HTML to do its own,
    # more accurate boilerplate removal — if we only kept our own
    # tag-extracted raw_text, trafilatura would have nothing useful to work on.
    raw_html = response.text

    # Parse the downloaded HTML string into a navigable BeautifulSoup tree.
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove all junk tags FIRST so their text never leaks into our output.
    # .decompose() deletes the tag and everything inside it from the tree.
    for tag_name in SKIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Now find only the tags we actually care about, in document order.
    content_elements = soup.find_all(CONTENT_TAGS)

    # Extract clean text from each tag and join with newlines so
    # paragraphs/headings stay visually separated in the saved text.
    raw_text = "\n".join(
        el.get_text(strip=True) for el in content_elements if el.get_text(strip=True)
    )

    # Grab the page <title> tag text if it exists, otherwise use a fallback.
    title = soup.title.get_text(strip=True) if soup.title else "No title found"

    # Return everything in one dictionary — easy to save as JSON later.
    return {
        "url": url,                  # the page we scraped
        "raw_text": raw_text,        # all extracted text, cleaned
        "raw_html": raw_html,        # full untouched HTML, for Phase 3's trafilatura
        "title": title,              # the page's <title> tag
        "scrape_method": "static",   # how we scraped it (for debugging)
    }


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — scrape_dynamic
# ──────────────────────────────────────────────────────────────────────────

def scrape_dynamic(url: str) -> dict:
    """
    Scrapes a 'dynamic' (JavaScript-heavy) page using Playwright.
    Opens a real, invisible (headless) Chromium browser, waits for all
    network activity to settle, then reads the fully-rendered text.

    Returns:
        dict with keys: url, raw_text, raw_html, title, scrape_method
    """
    # sync_playwright() starts the Playwright engine. The "with" block
    # makes sure everything is cleaned up automatically when we're done.
    with sync_playwright() as p:

        # Launch a Chromium browser in headless mode (no visible window —
        # faster and works on servers without a screen).
        browser = p.chromium.launch(headless=True)

        # Open a new blank browser tab/page.
        page = browser.new_page()

        # Navigate the tab to our target URL.
        page.goto(url, timeout=30000)  # 30000 ms = 30 second max wait

        # Wait until there have been no network requests for 500ms —
        # this means the page has (most likely) finished loading its
        # JavaScript-driven content.
        page.wait_for_load_state("networkidle")

        # Pull the page's <title> text directly from the rendered DOM.
        title = page.title()

        # page.content() returns the FULLY RENDERED HTML — i.e. the DOM
        # AFTER all the JavaScript has run and injected its content.
        # This is what Phase 3's trafilatura needs, since the original
        # server response would have been mostly empty for dynamic pages.
        raw_html = page.content()

        # inner_text() on the <body> grabs ALL visible text on the page,
        # exactly as a human would see it (hidden elements are excluded).
        raw_text = page.inner_text("body")

        # Always close the browser to free up memory — even though the
        # "with" block helps, closing explicitly is best practice.
        browser.close()

    # Return the same dictionary shape as scrape_static for consistency.
    return {
        "url": url,
        "raw_text": raw_text,
        "raw_html": raw_html,
        "title": title,
        "scrape_method": "dynamic",
    }


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — scrape_url  (the main entry point)
# ──────────────────────────────────────────────────────────────────────────

def scrape_url(url: str) -> dict:
    """
    The main function you call from outside this file.
    1. Detects whether the page is static or dynamic.
    2. Scrapes it with the right method.
    3. Saves the raw result as JSON inside data/raw/.
    4. Returns the scraped dictionary (or an error dictionary).
    """
    logger.info(f"[scrape_url] Starting scrape for: {url}")

    try:
        # Step 1: Figure out which scraping strategy to use.
        page_type = detect_page_type(url)
        logger.info(f"[scrape_url] Detected page_type='{page_type}' for {url}")

        # Step 2: Call the matching scraper function.
        if page_type == "static":
            result = scrape_static(url)
        else:
            result = scrape_dynamic(url)

        logger.success(
            f"[scrape_url] Successfully scraped {url} "
            f"({len(result['raw_text'])} characters of text extracted)."
        )

        # Step 3: Be polite — pause 1 second so we don't hammer servers
        # with rapid back-to-back requests if scraping many URLs.
        time.sleep(1)

        # Step 4: Save this result to disk as a JSON file for Phase 3.
        _save_raw_result(result)

        return result

    except Exception as e:
        # Catch ANY error (network issue, parsing issue, browser crash,
        # etc.) so one bad URL never crashes the whole scraping run.
        logger.error(f"[scrape_url] Failed to scrape {url}: {e}")
        return {"error": str(e), "url": url}


def _save_raw_result(result: dict) -> None:
    """
    Helper function: saves a scraped result dictionary as a JSON file
    inside data/raw/, named using a short hash of the URL so filenames
    stay short, unique, and filesystem-safe (URLs contain characters
    like ':' and '/' that aren't allowed in filenames).
    """
    # Make sure the data/raw/ folder exists (creates it if missing).
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # md5 turns the URL string into a fixed-length hex string, e.g.
    # "a1b2c3d4e5f6...". We only need the first 16 characters for a
    # filename that's unique enough for this project.
    url_hash = hashlib.md5(result["url"].encode("utf-8")).hexdigest()[:16]

    # Build the final file path, e.g. data/raw/a1b2c3d4e5f6abcd.json
    file_path = RAW_DATA_DIR / f"{url_hash}.json"

    # Write the dictionary to disk as nicely-formatted JSON.
    # ensure_ascii=False keeps special characters (é, ñ, etc.) readable.
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"[_save_raw_result] Saved raw result to: {file_path}")


# ──────────────────────────────────────────────────────────────────────────
# CLI (Command-Line Interface) — runs only when you execute this file
# directly (i.e. `python src/scraper/scraper.py ...`), not when it's
# imported elsewhere. This is the REAL dynamic entry point: you pass
# whatever URL(s) you want straight on the command line — nothing is
# hardcoded inside this file anymore.
# ──────────────────────────────────────────────────────────────────────────

import argparse  # built-in library for reading command-line arguments


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Defines what this script accepts on the command line.
    Keeping this in its own function makes it easy to test or reuse.
    """
    parser = argparse.ArgumentParser(
        description="Scrape any website(s) and save the result as JSON in data/raw/."
    )

    # nargs="+" means "one or more values" — so you can pass a single
    # URL or many, separated by spaces, e.g.:
    #   python src/scraper/scraper.py https://example.com
    #   python src/scraper/scraper.py https://a.com https://b.com
    parser.add_argument(
        "urls",
        nargs="+",
        help="One or more URLs to scrape, separated by spaces.",
    )

    return parser


def main() -> None:
    """
    Entry point for running this file as a script.
    Reads URL(s) from the command line (NOT hardcoded), scrapes each
    one using the same scrape_url() pipeline, and prints a short
    preview of every result.
    """
    # Parse whatever the user typed after "python src/scraper/scraper.py"
    parser = _build_arg_parser()
    args = parser.parse_args()

    # args.urls is a list — even if the user only typed one URL — because
    # we set nargs="+" above. This is what makes the script truly dynamic:
    # the list of URLs comes from YOU, at runtime, not from the file.
    for url in args.urls:
        output = scrape_url(url)

        print("\n--- SCRAPE RESULT PREVIEW ---")
        print("URL:", output.get("url"))

        # If scrape_url() returned an error dict, print it and move on
        # to the next URL instead of crashing the whole script.
        if "error" in output:
            print("ERROR:", output["error"])
            continue

        print("Title:", output.get("title"))
        print("Method:", output.get("scrape_method"))
        print("Text length:", len(output.get("raw_text", "")))
        print("First 300 chars:\n", output.get("raw_text", "")[:300])


if __name__ == "__main__":
    main()