"""
src/llm_chain/rag_chain.py
=============================
PHASE 6 — LLM + RAG Chain

This file is where everything from Phases 2-5 finally gets USED to
answer real questions. Given a user's question, it:
  1. Finds the most relevant chunks we stored in ChromaDB (Phase 5).
  2. Builds a prompt that forces the LLM to only answer using those
     chunks (this is what makes it "RAG" — Retrieval-Augmented Generation
     — instead of the LLM just making things up from its own training).
  3. Sends that prompt to Groq's free LLM API and returns the answer,
     along with which source URLs it came from.

Beginner notes:
- "Retrieval" = searching our own stored documents for relevant pieces.
- "Augmented Generation" = giving the LLM that retrieved text as context
  before it generates an answer, instead of letting it answer from
  memory alone (which could be wrong or made up — "hallucination").
- Groq is a company that runs open LLMs (like Llama 3) very fast, and
  offers a generous free tier — perfect for a learning project like this.
"""

import os                    # used to read environment variables (the API key)
from pathlib import Path     # modern way to work with file paths

import chromadb                              # the same local vector database from Phase 5
from dotenv import load_dotenv               # loads variables from a .env file
from groq import Groq                        # official Groq API client
from loguru import logger                    # pretty, structured logging
from sentence_transformers import SentenceTransformer  # same embedding model as Phase 5


# ──────────────────────────────────────────────────────────────────────────
# CONFIG — must match Phase 5's embedder.py EXACTLY, otherwise this file
# would be searching the wrong database folder, wrong collection, or
# encoding questions with a DIFFERENT model than the one used to encode
# the stored chunks (which would make similarity search meaningless).
# ──────────────────────────────────────────────────────────────────────────

# Same persistent ChromaDB folder Phase 5's embedder.py wrote to.
# parents[2] goes: rag_chain.py -> llm_chain/ -> src/ -> project root.
CHROMA_PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_db"

# Same embedding model Phase 5 used to create the stored chunk vectors.
# Using a different model here would produce embeddings that aren't
# comparable to the ones already stored, breaking similarity search.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Same default collection name Phase 5's embedder.py used when storing.
DEFAULT_COLLECTION_NAME = "rag_docs"

# Which Groq-hosted model we ask to generate answers.
# NOTE: llama3-8b-8192 was deprecated by Groq — see
# https://console.groq.com/docs/deprecations. Their official
# replacement is llama-3.1-8b-instant (same size class, faster).
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# How many of the last Q&A pairs we keep and show the LLM for context.
CHAT_HISTORY_LIMIT = 3


# ──────────────────────────────────────────────────────────────────────────
# SETUP — runs once when this file is imported or executed
# ──────────────────────────────────────────────────────────────────────────

# Reads the .env file (same one Phase 1 set up) and loads its contents
# into the environment, so os.getenv() below can find GROQ_API_KEY.
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create the Groq client with an explicit httpx transport timeout so it
# never hangs silently on Windows — connect=10s prevents indefinite
# blocking when the TCP connection stalls before any data is sent.
import httpx as _httpx
client = Groq(
    api_key=GROQ_API_KEY,
    http_client=_httpx.Client(
        timeout=_httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)
    ),
)

# Connect to the SAME persistent ChromaDB folder Phase 5 wrote chunks
# into. PersistentClient means it reads/writes real files on disk,
# so everything Phase 5 stored is still here, even in a new script run.
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

# Load the SAME embedding model Phase 5 used. This is loaded once, here,
# at import time — not inside every function call — so repeated
# questions in chat_session() don't reload an 80MB model every time.
logger.info(f"[setup] Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
logger.success("[setup] Embedding model loaded successfully.")


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — retrieve_chunks
# ──────────────────────────────────────────────────────────────────────────

def retrieve_chunks(question: str, collection_name: str = DEFAULT_COLLECTION_NAME, top_k: int = 5) -> list[dict]:
    """
    Searches ChromaDB for the chunks most similar in MEANING to the
    user's question (not just matching keywords — actual semantic
    similarity, thanks to the embeddings).

    Args:
        question: the user's question, as plain text.
        collection_name: which ChromaDB collection to search (must match
            the name used when storing, in Phase 5's store_in_chromadb()).
        top_k: how many of the best-matching chunks to return.

    Returns:
        A list of dicts: {chunk_text, source_url, chunk_index, similarity_score}
        ordered from most to least relevant.
    """
    # get_or_create_collection is safe even if the collection somehow
    # doesn't exist yet — it just returns an empty one instead of crashing.
    #
    # IMPORTANT: this MUST match the "hnsw:space": "cosine" setting used
    # when the collection was first created in embedder.py's
    # store_in_chromadb(). The space is locked in at creation time, so
    # this metadata argument here only takes effect if the collection
    # doesn't exist yet — for an EXISTING collection it's ignored, and
    # whatever space it was created with is what's actually used.
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Convert the question into the SAME kind of embedding vector the
    # stored chunks have, using the SAME model — this is what makes the
    # two comparable. .encode() returns a numpy array; ChromaDB's query()
    # wants a plain list, so we wrap the single question in a list and
    # take [0] back out further down where needed.
    question_embedding = embedding_model.encode([question])[0]

    # query() does the actual similarity search: it compares our question
    # embedding against every stored chunk embedding and returns the
    # closest n_results matches. include=["distances", ...] asks Chroma
    # to also give us a numeric similarity/distance score per result.
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB's query() returns results wrapped in an extra outer list
    # (because you COULD search multiple questions at once) — since we
    # only searched ONE question, we always take index [0] of each list.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []  # will hold one clean dict per matched chunk

    # zip() lets us walk through three same-length lists together, one
    # item from each per loop — so item i of all three lists belong together.
    for chunk_text, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append({
            "chunk_text": chunk_text,                       # the actual matched text
            "source_url": metadata.get("source_url", "unknown"),  # which page it came from
            "chunk_index": metadata.get("chunk_index", -1),  # its position in that document
            # ChromaDB's default distance is smaller = more similar.
            # We convert it to a 0-1 "similarity_score" (bigger = better
            # match) so it's more intuitive to read and display.
            "similarity_score": round(1 - distance, 4),
        })

    print(f"Found {len(retrieved)} relevant chunks for this question")
    logger.info(f"[retrieve_chunks] '{question}' -> {len(retrieved)} chunks from '{collection_name}'.")

    return retrieved


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — build_prompt
# ──────────────────────────────────────────────────────────────────────────

def build_prompt(question: str, retrieved_chunks: list[dict], chat_history: list | None = None) -> str:
    """
    Builds the exact text we send to the LLM. This is the single most
    important part of a RAG system: by putting strict instructions at
    the top and ONLY the retrieved chunks as "CONTEXT", we stop the LLM
    from answering using its own general training knowledge, and force
    it to stick to what's actually on the scraped webpage(s).

    Args:
        question: the user's question.
        retrieved_chunks: list of dicts from retrieve_chunks().
        chat_history: optional list of past {question, answer} dicts,
            so the LLM has some memory of the recent conversation.

    Returns:
        One complete prompt string, ready to send to the LLM.
    """
    # Default argument trap: never use a mutable list ([]) as a default
    # value directly in the function signature — Python would reuse the
    # SAME list across every call. Using None + this line is the safe way.
    if chat_history is None:
        chat_history = []

    # Build the CONTEXT section: one line per retrieved chunk, numbered
    # and labelled with its source URL, exactly as specified.
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append(f"[SOURCE {i} - {chunk['source_url']}]: {chunk['chunk_text']}")
    context_block = "\n".join(context_lines)

    # Only keep the most recent CHAT_HISTORY_LIMIT exchanges — older
    # messages get dropped so the prompt doesn't grow unbounded over a
    # long conversation. [-3:] takes the last 3 items of a list.
    recent_history = chat_history[-CHAT_HISTORY_LIMIT:]

    # Format chat history as readable "Q: ... / A: ..." lines instead of
    # raw Python dict syntax, so the LLM reads it like a real conversation.
    if recent_history:
        history_lines = [f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent_history]
        history_block = "\n".join(history_lines)
    else:
        history_block = "(no previous messages)"

    # Assemble the full prompt using an f-string. Triple-quoted strings
    # let us write multi-line text exactly as it will be sent.
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: 'I could not find this in the provided webpage.'
Do NOT use any knowledge outside the context. Always mention which source you used.
---
CONTEXT:
{context_block}
---
CHAT HISTORY (last 3 messages): {history_block}
---
QUESTION: {question}
ANSWER:"""

    return prompt


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — get_answer
# ──────────────────────────────────────────────────────────────────────────

def get_answer(question: str, collection_name: str = DEFAULT_COLLECTION_NAME, chat_history: list | None = None) -> dict:
    """
    The main function that turns a question into a grounded answer.
    Runs the full RAG pipeline: retrieve -> build prompt -> ask the LLM.

    Args:
        question: the user's question.
        collection_name: which ChromaDB collection to search.
        chat_history: optional list of past {question, answer} dicts.

    Returns:
        A dict: {answer, sources, chunks_used}
    """
    if chat_history is None:
        chat_history = []

    # Step 1 — find the chunks most relevant to this question.
    retrieved_chunks = retrieve_chunks(question, collection_name=collection_name)

    # Guard clause: if NOTHING was retrieved (e.g. empty/wrong collection),
    # there's no context to answer from — return early instead of sending
    # an empty-context prompt to the LLM and getting a confusing reply.
    if not retrieved_chunks:
        logger.warning(f"[get_answer] No chunks found for '{question}' in '{collection_name}'.")
        return {
            "answer": "I could not find this in the provided webpage.",
            "sources": [],
            "chunks_used": 0,
        }

    # Step 2 — build the strict, context-only prompt.
    prompt = build_prompt(question, retrieved_chunks, chat_history)

    # Step 3 — send the prompt to Groq's hosted Llama 3 model.
    # messages follows the standard chat format: a list of role/content
    # dicts. We only need a single "user" message containing our full prompt.
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    # The actual generated text lives inside this nested structure —
    # choices[0] is the first (and only, since we didn't ask for more)
    # completion the model produced.
    answer_text = response.choices[0].message.content

    # Build a de-duplicated list of source URLs actually used, preserving
    # the order they first appeared in. dict.fromkeys() is a common
    # Python trick for "unique items, order preserved" (sets don't keep order).
    sources = list(dict.fromkeys(chunk["source_url"] for chunk in retrieved_chunks))

    logger.success(f"[get_answer] Answered '{question}' using {len(retrieved_chunks)} chunks from {len(sources)} source(s).")

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(retrieved_chunks),
    }


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — chat_session
# ──────────────────────────────────────────────────────────────────────────

def chat_session(collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    """
    A simple terminal chat loop. Type a question, get an answer with
    sources, repeat. Type 'quit' to stop.

    Args:
        collection_name: which ChromaDB collection to chat against.
    """
    print(f"\n💬 RAG Chat — collection: '{collection_name}'")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    # This list grows as the conversation goes on; get_answer() and
    # build_prompt() only ever look at the last CHAT_HISTORY_LIMIT items.
    chat_history: list[dict] = []

    while True:
        # input() pauses the program and waits for the user to type
        # something and press Enter — this is what makes it interactive.
        question = input("You: ").strip()

        # Let the user exit the loop cleanly instead of using Ctrl+C.
        if question.lower() == "quit":
            print("\n👋 Ending chat session. Goodbye!")
            break

        # Skip empty submissions (just pressing Enter) without crashing.
        if not question:
            continue

        # Run the full RAG pipeline for this one question.
        result = get_answer(question, collection_name=collection_name, chat_history=chat_history)

        # Display the answer and which source(s) it came from.
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print(f"(used {result['chunks_used']} chunks)\n")

        # Save this exchange into chat_history so future questions in
        # this same session can refer back to it.
        chat_history.append({"question": question, "answer": result["answer"]})


# ──────────────────────────────────────────────────────────────────────────
# CLI — running this file directly starts an interactive chat session.
# e.g.: python src/llm_chain/rag_chain.py
#       python src/llm_chain/rag_chain.py --collection rag_docs
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse  # built-in library for reading command-line arguments

    parser = argparse.ArgumentParser(
        description="Start an interactive RAG chat session against a ChromaDB collection."
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"ChromaDB collection name to chat against (default: '{DEFAULT_COLLECTION_NAME}').",
    )
    args = parser.parse_args()

    chat_session(collection_name=args.collection)


if __name__ == "__main__":
    main()