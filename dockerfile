# ──────────────────────────────────────────────────────────
# Dockerfile for the FastAPI backend (api/main.py)
# ──────────────────────────────────────────────────────────

# Start from a lightweight official Python 3.11 image.
# "slim" = stripped-down Linux + Python, much smaller than the
# full python:3.11 image, which keeps build/deploy times down.
FROM python:3.11-slim

# All following commands (COPY, RUN, CMD) happen inside this folder
# inside the container. It's created automatically if missing.
WORKDIR /app

# Playwright's headless Chromium needs several system-level
# libraries (for fonts, rendering, etc.) that aren't in the slim
# image by default. We install them now, before copying our code,
# so this slow step gets cached by Docker and doesn't re-run on
# every code change (only if this line itself changes).
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements.txt first (not the whole project yet).
# Docker caches each instruction as a "layer" — by copying just
# this file and installing it before copying the rest of the code,
# Docker can reuse this slow "pip install" layer on future builds
# as long as requirements.txt hasn't changed, even if your code has.
COPY requirements.txt .

# --no-cache-dir keeps the image smaller by not storing pip's
# download cache inside the container (we don't need it after install).
RUN pip install --no-cache-dir -r requirements.txt

# Download the actual Chromium browser binary that Playwright's
# Python package needs at runtime (the pip package alone is just
# the controller, not the browser itself). --with-deps also grabs
# any remaining OS libraries Chromium needs.
RUN playwright install --with-deps chromium

# Download spaCy's small English model, used by summarizer.py for
# named-entity extraction (persons/orgs/locations/dates).
RUN python -m spacy download en_core_web_sm

# Download the NLTK tokenizer data sumy needs to split text into
# sentences for the extractive (LexRank) summary.
RUN python -m nltk.downloader punkt punkt_tab

# NOW copy the rest of your project files (src/, api/, etc.) into
# the image. This is done last, after the slow installs above,
# so editing your own code doesn't force those installs to re-run.
COPY . .

# Documents which port the container listens on. This is metadata
# for humans/tools (like docker run -p) — it does NOT actually
# publish the port by itself.
EXPOSE 8000

# The command that runs when the container starts.
# --host 0.0.0.0 means "listen on all network interfaces" (required
# so traffic from outside the container can reach it — 127.0.0.1
# would only accept connections from inside the container itself).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]