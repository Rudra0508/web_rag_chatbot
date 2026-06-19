# api/main.py

# ================================
# PHASE 8 - FASTAPI BACKEND
# ================================

# FastAPI framework
from fastapi import FastAPI, HTTPException

# Pydantic models
from pydantic import BaseModel

# CORS support
from fastapi.middleware.cors import CORSMiddleware

# Environment variables
from dotenv import load_dotenv

# UUID for session IDs
from uuid import uuid4

# Standard Python modules
import os

# Groq client
from groq import Groq

# ================================
# LOAD ENVIRONMENT VARIABLES
# ================================

load_dotenv()

# ================================
# IMPORT YOUR PHASES
# ================================

from src.scraper.scraper import scrape_url
from src.processor.cleaner import clean_text
from src.embeddings.embeddings import process_document
from src.llm.llm_rag import get_answer
from src.llm_chain.summarizer import generate_knowledge_card

# ================================
# CREATE FASTAPI APP
# ================================

app = FastAPI(
    title="Web Scraping RAG Chatbot",
    description="Phase 8 FastAPI Backend",
    version="1.0"
)

# ================================
# ENABLE CORS
# ================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# GROQ CLIENT
# ================================

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ================================
# IN-MEMORY SESSION STORE
# ================================

sessions = {}

# Example:
#
# sessions = {
#     "abc123": {
#         "url": "...",
#         "status": "ready",
#         "collection_name": "rag_docs"
#     }
# }

# ================================
# PYDANTIC MODELS
# ================================

class ScrapeRequest(BaseModel):
    url: str


class ScrapeResponse(BaseModel):
    session_id: str
    status: str
    knowledge_card: dict
    message: str


class ChatRequest(BaseModel):
    session_id: str
    question: str
    chat_history: list = []


class ChatResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: int


class StatusResponse(BaseModel):
    session_id: str
    status: str
    progress_message: str


# ================================
# HEALTH CHECK
# ================================

@app.get("/api/health")
def health_check():
    """
    Simple endpoint to verify the server is running.
    """

    return {
        "status": "ok",
        "message": "Server is running"
    }


# ================================
# SCRAPE ENDPOINT
# ================================

@app.post(
    "/api/scrape",
    response_model=ScrapeResponse
)
def scrape_endpoint(request: ScrapeRequest):
    """
    Full pipeline:

    URL
        ↓
    Scraper
        ↓
    Cleaner
        ↓
    Embeddings
        ↓
    Knowledge Card
    """

    try:

        # Validate URL

        if not (
            request.url.startswith("http://")
            or
            request.url.startswith("https://")
        ):
            raise HTTPException(
                status_code=400,
                detail="URL must start with http:// or https://"
            )

        # Generate unique session

        session_id = str(uuid4())

        # --------------------------------
        # PHASE 2
        # SCRAPE
        # --------------------------------

        raw_data = scrape_url(request.url)

        if "error" in raw_data:
            raise Exception(raw_data["error"])

        # --------------------------------
        # PHASE 3
        # CLEAN
        # --------------------------------

        clean_data = clean_text(raw_data)

        # --------------------------------
        # PHASE 5
        # EMBEDDINGS
        # --------------------------------

        process_document(clean_data)

        # --------------------------------
        # PHASE 7
        # KNOWLEDGE CARD
        # --------------------------------

        knowledge_card = generate_knowledge_card(
            clean_data,
            groq_client
        )

        # Save session

        sessions[session_id] = {
            "url": request.url,
            "status": "ready",
            "collection_name": "rag_docs"
        }

        return ScrapeResponse(
            session_id=session_id,
            status="ready",
            knowledge_card=knowledge_card,
            message="Website processed successfully"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ================================
# CHAT ENDPOINT
# ================================

@app.post(
    "/api/chat",
    response_model=ChatResponse
)
def chat_endpoint(request: ChatRequest):
    """
    Ask questions against stored embeddings.
    """

    if request.session_id not in sessions:

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    session = sessions[request.session_id]

    collection_name = session["collection_name"]

    result = get_answer(
        question=request.question,
        collection_name=collection_name,
        chat_history=request.chat_history
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"]
    )


# ================================
# SESSION STATUS ENDPOINT
# ================================

@app.get(
    "/api/session/{session_id}",
    response_model=StatusResponse
)
def session_status(session_id: str):
    """
    Return information about a session.
    """

    if session_id not in sessions:

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    session = sessions[session_id]

    return StatusResponse(
        session_id=session_id,
        status=session["status"],
        progress_message="Session is ready"
    )