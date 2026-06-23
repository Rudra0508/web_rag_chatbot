"""
src/embedder/embedder.py
============================
PHASE 5 — Chunking and Embedding

This file takes the CLEAN text from Phase 3 (data/clean/*.json) and:
  PART A: splits it into small overlapping "chunks" (because LLMs and
          vector search work much better on small pieces of text than
          on one giant document).
  PART B: converts each chunk into a list of numbers ("embedding") that
          captures its MEANING — similar sentences get similar numbers.
  PART C: stores everything in ChromaDB, a local vector database, so
          Phase 6 can search "which chunks are most relevant to this
          user question?" later.

Beginner notes:
- A "chunk" is just a smaller piece of a bigger document — like cutting
  a book into individual paragraphs.
- An "embedding" is a list of a few hundred numbers (a "vector") that
  represents what a piece of text MEANS. Texts about similar topics end
  up with similar numbers, which is how semantic search works.
- ChromaDB is a database built specifically to store and search these
  number-lists efficiently.
"""

import hashlib                  # used to create a unique chunk_id per chunk
from datetime import datetime, timezone  # to timestamp each chunk
from pathlib import Path        # modern way to work with file paths

import chromadb                                  # local vector database
import numpy as np                                # numpy arrays (embeddings are numpy arrays)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # smart text splitter
from loguru import logger                         # pretty, structured logging
from sentence_transformers import SentenceTransformer  # turns text into embeddings


# ──────────────────────────────────────────────────────────────────────────
# CONFIG — simple constants so we don't repeat "magic numbers" everywhere
# ──────────────────────────────────────────────────────────────────────────

# Defaults based on the EDA notebook's recommendation (Phase 4).
# You can override these per-call if a future EDA run suggests new values.
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 40

# Folder where ChromaDB stores its database files on disk. Using the
# same project-root pattern as scraper.py / cleaner.py so every Phase
# saves its output inside the project, not somewhere random.
CHROMA_PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_db"

# The free, local embedding model. ~80MB, downloads automatically the
# first time it's used, then gets cached for every run after that.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Name of the ChromaDB "collection" — think of a collection like a
# table in a normal database; all our chunks live inside this one.
DEFAULT_COLLECTION_NAME = "rag_docs"


# ──────────────────────────────────────────────────────────────────────────
# PART A — CHUNKING
# ──────────────────────────────────────────────────────────────────────────

# FUNCTION 1 — chunk_text
def chunk_text(
    clean_text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits one big piece of clean text into smaller overlapping chunks.

    "Recursive" means: it FIRST tries to split on paragraph breaks
    ("\\n\\n"). If a resulting piece is still too big, it tries splitting
    on single newlines, then sentences (". "), then finally on spaces
    between words — always trying to keep whole sentences/paragraphs
    together where possible, instead of cutting text mid-word.

    Args:
        clean_text: the full cleaned document text (from Phase 3).
        chunk_size: target MAXIMUM characters per chunk.
        chunk_overlap: how many characters from the end of one chunk
            are repeated at the start of the next — this helps the AI
            avoid losing context that fell right on a chunk boundary.

    Returns:
        A list of chunk strings, e.g. ["chunk 1 text...", "chunk 2 text...", ...]
    """
    # Build the splitter. The separators list, in order, is exactly
    # what makes this "recursive": try paragraph breaks first, fall
    # back to single newlines, then sentence-ending periods, then
    # finally plain spaces if nothing bigger fits.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # split_text() does all the actual work and returns a plain list
    # of strings — one string per chunk.
    chunks = splitter.split_text(clean_text)

    # Let the user/log see how many pieces this document became.
    print(f"Created {len(chunks)} chunks from {len(clean_text)} characters of text.")
    logger.info(f"[chunk_text] Split {len(clean_text)} chars into {len(chunks)} chunks "
                f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}).")

    return chunks


# FUNCTION 2 — add_metadata
def add_metadata(chunks: list[str], source_url: str, doc_title: str) -> list[dict]:
    """
    Attaches useful metadata to each chunk so we always know WHERE a
    chunk came from, even after it's stored in the vector database.

    Args:
        chunks: list of chunk strings from chunk_text().
        source_url: the URL this document was scraped from.
        doc_title: the page's title (from the scraper's "title" field).

    Returns:
        A list of dictionaries, one per chunk, each with:
        chunk_id, chunk_text, source_url, doc_title, chunk_index,
        word_count, created_at.
    """
    chunks_with_metadata = []  # will hold one dict per chunk

    # enumerate() gives us both the chunk's text AND its position
    # (0, 1, 2, ...) in the original list, in one loop.
    for chunk_index, chunk_text_value in enumerate(chunks):
        # Build a string that's unique per (document, position) pair —
        # e.g. two different chunks at index 0 from two different URLs
        # will never collide, because the URL is part of the input.
        id_source = f"{source_url}_{chunk_index}"

        # md5 turns that string into a fixed-length, unique-enough hex
        # string we can safely use as a database ID.
        chunk_id = hashlib.md5(id_source.encode("utf-8")).hexdigest()

        chunks_with_metadata.append({
            "chunk_id": chunk_id,                       # unique ID for this chunk
            "chunk_text": chunk_text_value,              # the actual chunk text
            "source_url": source_url,                    # which page this came from
            "doc_title": doc_title,                       # that page's title
            "chunk_index": chunk_index,                  # position within the document
            "word_count": len(chunk_text_value.split()), # how many words in this chunk
            "created_at": datetime.now(timezone.utc).isoformat(),  # timestamp
        })

    return chunks_with_metadata


# ──────────────────────────────────────────────────────────────────────────
# PART B — EMBEDDING
# ──────────────────────────────────────────────────────────────────────────

# FUNCTION 3 — load_embedding_model
def load_embedding_model() -> SentenceTransformer:
    """
    Loads the sentence-transformers model that turns text into
    embeddings. The very first time this runs, it downloads the model
    (~80MB) from the internet; every run after that loads it instantly
    from a local cache folder.

    Returns:
        A ready-to-use SentenceTransformer model object.
    """
    logger.info(f"[load_embedding_model] Loading model: {EMBEDDING_MODEL_NAME}")

    # This single line downloads (if needed) and loads the model into
    # memory, ready to convert text into embeddings.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding model loaded successfully")
    logger.success(f"[load_embedding_model] '{EMBEDDING_MODEL_NAME}' ready to use.")

    return model


# FUNCTION 4 — embed_chunks
def embed_chunks(chunks_with_metadata: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Converts every chunk's text into an embedding (a list of numbers).

    Args:
        chunks_with_metadata: list of dicts from add_metadata().
        model: the loaded SentenceTransformer model from load_embedding_model().

    Returns:
        A numpy array of shape (num_chunks, embedding_dimensions) —
        e.g. for 'all-MiniLM-L6-v2', each embedding has 384 numbers.
    """
    # Pull out JUST the text from every chunk dictionary — model.encode()
    # only needs plain strings, not our extra metadata fields.
    texts = [chunk["chunk_text"] for chunk in chunks_with_metadata]

    logger.info(f"[embed_chunks] Embedding {len(texts)} chunks...")

    # encode() runs ALL texts through the model in batches (32 at a
    # time here) instead of one-by-one — this is much faster, especially
    # on a GPU, but still helps a lot even on a normal CPU.
    # show_progress_bar=True prints a live progress bar in the terminal.
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    logger.success(f"[embed_chunks] Produced embeddings with shape {embeddings.shape}.")

    return embeddings


# ──────────────────────────────────────────────────────────────────────────
# PART C — STORING IN CHROMADB
# ──────────────────────────────────────────────────────────────────────────

# FUNCTION 5 — store_in_chromadb
def store_in_chromadb(
    chunks_with_metadata: list[dict],
    embeddings: np.ndarray,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> None:
    """
    Saves chunks + their embeddings into a local, persistent ChromaDB
    database on disk, so they survive between separate script runs.

    Args:
        chunks_with_metadata: list of dicts from add_metadata().
        embeddings: numpy array from embed_chunks(), same length/order
            as chunks_with_metadata.
        collection_name: which "table" inside ChromaDB to store these in.
    """
    # PersistentClient saves data to a real folder on disk (CHROMA_PERSIST_DIR),
    # as opposed to an in-memory client that would lose everything when
    # the script ends. str() because chromadb expects a plain string path.
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    # get_or_create_collection: if "rag_docs" already exists (from a
    # previous run), reuse it and just add more chunks to it. If it
    # doesn't exist yet, create it fresh. This makes the function safe
    # to call repeatedly across many documents.
    #
    # IMPORTANT: ChromaDB defaults to "l2" (Euclidean) distance, but
    # all-MiniLM-L6-v2 (our embedding model) produces NORMALIZED vectors
    # that are specifically designed to be compared with COSINE distance.
    # Using the wrong distance metric here makes retrieval pick worse
    # chunks even though it "runs" with no errors — a quiet correctness
    # bug, not a crash. We explicitly request cosine to match the model.
    # NOTE: this setting is locked in at CREATION time and can't be
    # changed on an existing collection — if you already have a
    # collection from before this fix, delete the chroma_db/ folder
    # and re-run your embedding step so it gets created fresh with
    # the correct setting.
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB's .add() wants four separate lists, all the SAME LENGTH
    # and in the SAME ORDER (item 0 in each list belongs together):
    #   ids         -> unique string ID per chunk
    #   embeddings  -> the number-vectors for each chunk
    #   metadatas   -> extra info dicts per chunk (for filtering later)
    #   documents   -> the actual chunk text (so Chroma can return it directly)
    ids = [chunk["chunk_id"] for chunk in chunks_with_metadata]

    # ChromaDB's add() expects embeddings as plain Python lists, not a
    # numpy array directly — .tolist() converts each row safely.
    embeddings_list = embeddings.tolist()

    # Metadata dictionaries can't contain every Python type ChromaDB
    # supports oddly shaped data, so we pass a clean dict of simple
    # strings/numbers per chunk (excluding the chunk_text itself,
    # since that goes in `documents` instead, not duplicated here).
    metadatas = [
        {
            "source_url": chunk["source_url"],
            "doc_title": chunk["doc_title"],
            "chunk_index": chunk["chunk_index"],
            "word_count": chunk["word_count"],
            "created_at": chunk["created_at"],
        }
        for chunk in chunks_with_metadata
    ]

    documents = [chunk["chunk_text"] for chunk in chunks_with_metadata]

    # upsert() = insert if new, update if already exists. This replaces
    # add() which crashes or spams warnings when the same chunk IDs are
    # stored again (e.g. running the embedder twice on the same document).
    # Safe to call any number of times on the same data.
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=documents,
    )

    print(f"Stored {len(ids)} chunks in ChromaDB")
    logger.success(
        f"[store_in_chromadb] Stored {len(ids)} chunks in collection "
        f"'{collection_name}' at {CHROMA_PERSIST_DIR}."
    )


# ──────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — process_document (the main pipeline)
# ──────────────────────────────────────────────────────────────────────────

def process_document(clean_data_dict: dict,collection_name: str =DEFAULT_COLLECTION_NAME) -> dict:
    """
    The main function you call from outside this file.
    Takes ONE cleaned document dictionary (exactly what Phase 3's
    cleaner.py saves into data/clean/*.json) and runs the FULL Phase 5
    pipeline on it, end to end:
        chunk_text -> add_metadata -> load_embedding_model
        -> embed_chunks -> store_in_chromadb

    Args:
        clean_data_dict: dict with at least "url" and "clean_text" keys
            (the exact shape produced by src/preprocessor/cleaner.py's
            clean_text() function). "doc_title" is optional — if your
            cleaned JSON doesn't have one yet, we fall back to the URL.

    Returns:
        A dict with: chunks_with_metadata, embeddings — so the caller
        (or our own test block below) can inspect what happened without
        needing to re-read anything from ChromaDB.
    """
    source_url = clean_data_dict.get("url", "unknown_url")

    # Phase 3's cleaner.py does not currently save a "doc_title" field,
    # so we fall back to the URL itself if it's missing — this keeps
    # process_document() working even on data/clean/*.json files exactly
    # as cleaner.py produces them today, with no required changes there.
    doc_title = clean_data_dict.get("doc_title", source_url)

    clean_text_value = clean_data_dict.get("clean_text", "")

    logger.info(f"[process_document] Starting Phase 5 pipeline for: {source_url}")

    # Guard clause: an empty document has nothing to chunk/embed, so we
    # stop early with a clear warning instead of crashing deeper in the
    # pipeline (e.g. inside the embedding model on an empty list).
    if not clean_text_value.strip():
        logger.warning(f"[process_document] '{source_url}' has no clean_text. Skipping.")
        return {"chunks_with_metadata": [], "embeddings": None}

    # Step 1 — split the document into overlapping chunks.
    chunks = chunk_text(clean_text_value)

    # Step 2 — attach metadata (id, url, title, etc.) to every chunk.
    chunks_with_metadata = add_metadata(chunks, source_url, doc_title)

    # Step 3 — load the embedding model (downloads on first run only).
    model = load_embedding_model()

    # Step 4 — convert every chunk's text into an embedding vector.
    embeddings = embed_chunks(chunks_with_metadata, model)

    # Step 5 — persist everything into the local ChromaDB database.
    store_in_chromadb(chunks_with_metadata, embeddings,collection_name=collection_name)

    print(f"✅ Successfully processed '{source_url}' — {len(chunks)} chunks embedded and stored.")
    logger.success(f"[process_document] Finished Phase 5 pipeline for: {source_url}")

    return {
        "chunks_with_metadata": chunks_with_metadata,
        "embeddings": embeddings,
        "total_chunks": len(chunks_with_metadata)
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI — lets you run this file directly on an already-cleaned JSON file
# from Phase 3, e.g.:
#   python src/embedder/embedder.py data/clean/<hash>.json
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse  # built-in library for reading command-line arguments
    import json      # for loading the cleaned JSON file from disk

    parser = argparse.ArgumentParser(
        description="Chunk, embed, and store a cleaned JSON file (from Phase 3) into ChromaDB."
    )
    parser.add_argument(
        "clean_json_path",
        help="Path to a cleaned JSON file saved by Phase 3's cleaner.py (e.g. data/clean/abc123.json).",
    )
    args = parser.parse_args()

    # Load the cleaned dictionary exactly as cleaner.py saved it.
    with open(args.clean_json_path, "r", encoding="utf-8") as f:
        clean_data = json.load(f)

    # Run the full Phase 5 pipeline on it.
    result = process_document(clean_data)

    chunks_with_metadata = result["chunks_with_metadata"]
    embeddings = result["embeddings"]

    # If process_document() bailed early (empty document), there's
    # nothing meaningful to preview — say so and stop.
    if not chunks_with_metadata:
        print("\nNo chunks were created (document was empty). Nothing to preview.")
        return

    # Print exactly what was asked for: the first chunk's text, and the
    # first 5 numbers of its embedding, so you can SEE this is working.
    print("\n--- FIRST CHUNK PREVIEW ---")
    print("Chunk text:\n")
    print(chunks_with_metadata[0]["chunk_text"])

    print("\n--- FIRST CHUNK'S EMBEDDING (first 5 numbers) ---")
    print(embeddings[0][:5])


if __name__ == "__main__":
    main()