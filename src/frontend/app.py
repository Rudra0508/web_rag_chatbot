"""
frontend/app.py  —  PHASE 9: Streamlit Frontend

Run from project root:
    streamlit run frontend/app.py

Requires the FastAPI backend (Phase 8) running at http://localhost:8000
    uvicorn src.api.main:app --reload --port 8000
"""

import time           # for simulating streaming with small delays
import requests       # makes HTTP calls to the FastAPI backend
import streamlit as st  # turns this Python script into a web app

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call in the file)
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WebChat AI",                # browser tab title
    page_icon="💬",                         # browser tab icon
    layout="wide",                          # use full browser width
)

# ── Custom CSS ───────────────────────────────────────────────────────────
# st.markdown with unsafe_allow_html injects raw HTML/CSS into the page.
st.markdown("""
<style>
    /* Remove default Streamlit top padding */
    .block-container { padding-top: 1.5rem; }

    /* Knowledge card box */
    .card-box {
        background: #f8f9fa;
        border-left: 4px solid #4C72B0;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* Source citation line below AI messages */
    .source-line {
        font-size: 0.78rem;
        color: #888;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }

    /* Sidebar section header */
    .sidebar-header {
        font-weight: 700;
        font-size: 0.9rem;
        color: #444;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"   # FastAPI backend address

# ──────────────────────────────────────────────────────────────────────────
# SESSION STATE  (Streamlit's memory between re-runs)
# ──────────────────────────────────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction. st.session_state
# is how we remember data (like the session_id and chat history) between
# those re-runs. We initialise every key we'll use here so nothing crashes
# on the very first run when the keys don't exist yet.

if "session_id"      not in st.session_state: st.session_state.session_id      = None
if "knowledge_card"  not in st.session_state: st.session_state.knowledge_card  = None
if "scraped_url"     not in st.session_state: st.session_state.scraped_url     = None
if "messages"        not in st.session_state: st.session_state.messages        = []
if "chunks_stored"   not in st.session_state: st.session_state.chunks_stored   = 0


# ──────────────────────────────────────────────────────────────────────────
# HELPER — check backend is alive
# ──────────────────────────────────────────────────────────────────────────

def backend_is_alive() -> bool:
    """Returns True if the FastAPI server is reachable, False otherwise."""
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛠 Controls")

    # ── Backend status ────────────────────────────────────────────────────
    if backend_is_alive():
        st.success("✅ Backend connected")       # green = server is running
    else:
        st.error(
            "❌ Backend offline\n\n"
            "Start it with:\n"
            "```\nuvicorn src.api.main:app --reload --port 8000\n```"
        )

    st.divider()

    # ── How it works ──────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-header">How it works</p>', unsafe_allow_html=True)
    st.markdown("""
1. 🌐 **Paste a URL** — any webpage
2. ⚙️ **Click Analyze** — scrapes, cleans, embeds it
3. 💬 **Ask questions** — RAG answers from that page only
""")

    st.divider()

    # ── Session info ──────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-header">Session info</p>', unsafe_allow_html=True)

    if st.session_state.session_id:
        # Show details of the current active session
        st.write("**URL:**", st.session_state.scraped_url)
        st.write("**Chunks stored:**", st.session_state.chunks_stored)
        st.write("**Session ID:**")
        st.code(st.session_state.session_id[:16] + "…", language=None)
    else:
        st.info("No active session yet.")

    st.divider()

    # ── New chat button ───────────────────────────────────────────────────
    if st.button("🔄 New Chat", use_container_width=True):
        # Clear ALL session state keys so everything resets to scratch
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()   # force an immediate re-run so the UI clears instantly


# ──────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────

st.title("💬 Chat with Any Webpage")
st.caption("Paste any URL and ask questions about it")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1 — URL INPUT
# ──────────────────────────────────────────────────────────────────────────

col_input, col_btn = st.columns([5, 1])   # wide text box + narrow button

with col_input:
    url_input = st.text_input(
        label="Enter a webpage URL",
        placeholder="https://en.wikipedia.org/wiki/Albert_Einstein",
        label_visibility="collapsed",     # hide label, placeholder does the job
    )

with col_btn:
    analyze_clicked = st.button(
        "🔍 Analyze",
        type="primary",                    # makes it blue
        use_container_width=True,
    )

# ── Handle Analyze button click ───────────────────────────────────────────

if analyze_clicked:
    if not url_input.strip():
        st.warning("⚠️ Please enter a URL first.")

    elif not url_input.startswith(("http://", "https://")):
        st.error("⚠️ URL must start with http:// or https://")

    elif not backend_is_alive():
        st.error("❌ Backend is not running. Start it first (see sidebar).")

    else:
        # st.spinner shows a loading animation while the pipeline runs.
        # Everything inside the 'with' block runs before the spinner hides.
        with st.spinner("🔄 Scraping and analyzing the page… (this takes ~30–60 seconds)"):
            try:
                # POST the URL to the FastAPI /api/scrape endpoint
                response = requests.post(
                    f"{API_BASE}/api/scrape",
                    json={"url": url_input.strip()},
                    timeout=300,          # allow up to 5 minutes for scrape + embed
                )

                if response.status_code == 200:
                    data = response.json()

                    # Store everything we'll need in session_state
                    st.session_state.session_id     = data["session_id"]
                    st.session_state.knowledge_card = data["knowledge_card"]
                    st.session_state.scraped_url    = url_input.strip()
                    st.session_state.messages       = []   # fresh chat for new URL

                    # Chunks stored is in the session endpoint, fetch it
                    sess_resp = requests.get(
                        f"{API_BASE}/api/session/{data['session_id']}", timeout=5
                    )
                    if sess_resp.status_code == 200:
                        st.session_state.chunks_stored = (
                        sess_resp.json().get("chunks_stored", 0)
                    )

                    st.success("✅ Page analyzed! Ask your questions below.")
                    st.rerun()   # re-run so the knowledge card + chat appear immediately

                else:
                    # FastAPI returned an error status
                    detail = response.json().get("detail", response.text)
                    st.error(f"❌ Error {response.status_code}: {detail}")

            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The page may be too large. Try again.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# SECTION 2 — KNOWLEDGE CARD
# Only visible after a successful scrape
# ──────────────────────────────────────────────────────────────────────────

if st.session_state.session_id and st.session_state.knowledge_card:
    card = st.session_state.knowledge_card   # shorthand

    st.divider()

    with st.container():
        st.markdown('<div class="card-box">', unsafe_allow_html=True)

        # Title
        st.markdown(f"### 📄 {card.get('title', 'Untitled')}")

        # One-line summary (italic)
        summary = card.get("one_line_summary", "")
        if summary and summary != "Summary unavailable.":
            st.markdown(f"*{summary}*")

        st.markdown("")   # small spacer

        # Three metric columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment",      card.get("sentiment",                  "—").capitalize())
        col2.metric("Read time",      f"{card.get('estimated_read_time_minutes', '—')} min")
        col3.metric("Word count",     card.get("word_count",                 "—"))

        # Key points
        key_points = card.get("key_points", [])
        if key_points:
            st.markdown("**Key points:**")
            for point in key_points:
                st.markdown(f"- {point}")

        # People & organisations mentioned
        entities = card.get("entities", {})
        persons  = entities.get("persons", [])
        orgs     = entities.get("organizations", [])

        if persons or orgs:
            st.markdown("**Mentioned:**")
            if persons:
                st.markdown(f"👤 **People:** {', '.join(persons)}")
            if orgs:
                st.markdown(f"🏢 **Orgs:** {', '.join(orgs)}")

        st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# SECTION 3 — CHAT INTERFACE
# Only visible after a successful scrape
# ──────────────────────────────────────────────────────────────────────────

if st.session_state.session_id:
    st.divider()
    st.subheader("💬 Ask questions about this page")

    # ── Replay chat history ───────────────────────────────────────────────
    # Loop through every past message and render it in the correct bubble.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):    # "user" or "assistant"
            st.write(msg["content"])

            # Show source URLs below every assistant message (small grey text)
            if msg["role"] == "assistant" and msg.get("sources"):
                sources_text = " · ".join(msg["sources"])
                st.markdown(
                    f'<p class="source-line">📎 Sources: {sources_text}</p>',
                    unsafe_allow_html=True,
                )

    # ── Chat input ────────────────────────────────────────────────────────
    # st.chat_input creates the message box that stays fixed at the bottom.
    # It returns the typed text when the user presses Enter, else None.
    if prompt := st.chat_input("Ask anything about this page…"):

        # ── Show user message immediately ─────────────────────────────────
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.write(prompt)

        # ── Call backend and stream response ──────────────────────────────
        with st.chat_message("assistant"):
            # Build a flat list of past Q&A for the chat_history parameter.
            # rag_chain.py's get_answer() expects [{question, answer}, ...]
            history_for_api = [
                {"question": m["content"], "answer": ""}
                if m["role"] == "user"
                else {"question": "", "answer": m["content"]}
                for m in st.session_state.messages[:-1]   # exclude current user message
            ]

            response_placeholder = st.empty()   # blank slot we'll fill progressively
            full_answer = ""
            sources     = []

            try:
                api_response = requests.post(
                    f"{API_BASE}/api/chat",
                    json={
                        "session_id":   st.session_state.session_id,
                        "question":     prompt,
                        "chat_history": history_for_api,
                    },
                    timeout=60,
                )

                if api_response.status_code == 200:
                    data        = api_response.json()
                    full_answer = data["answer"]
                    sources     = data.get("sources", [])

                    # Simulate streaming by revealing the answer word by word —
                    # the real answer is already available but this looks much
                    # nicer than having the whole block appear at once.
                    words       = full_answer.split()
                    streamed    = ""
                    for word in words:
                        streamed += word + " "
                        response_placeholder.write(streamed)
                        time.sleep(0.025)           # 25ms per word ≈ natural reading pace

                else:
                    full_answer = f"❌ Error {api_response.status_code}: {api_response.json().get('detail', '')}"
                    response_placeholder.error(full_answer)

            except requests.exceptions.Timeout:
                full_answer = "❌ Request timed out. Try a shorter question."
                response_placeholder.error(full_answer)
            except Exception as e:
                full_answer = f"❌ Unexpected error: {e}"
                response_placeholder.error(full_answer)

            # Show sources below the answer
            if sources:
                sources_text = " · ".join(sources)
                st.markdown(
                    f'<p class="source-line">📎 Sources: {sources_text}</p>',
                    unsafe_allow_html=True,
                )

        # Save the completed assistant message to history
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_answer,
            "sources": sources,
        })