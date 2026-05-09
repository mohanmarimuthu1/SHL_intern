"""
FastAPI application — SHL Conversational Assessment Recommender.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → stateless conversation, returns reply + recommendations
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from agent.conversation import process_chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TURNS = 8  # Hard cap per spec (user + assistant messages combined)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load FAISS index and embedding model at startup."""
    logger.info("Loading retriever (FAISS + embedding model)...")
    from retrieval.retriever import get_retriever
    get_retriever()  # Warm up singleton
    logger.info("Retriever ready.")
    yield


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL Individual Test Solutions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Validate roles
    for msg in messages:
        if msg["role"] not in ("user", "assistant"):
            raise HTTPException(status_code=422, detail=f"Invalid role: {msg['role']!r}")

    # Must start with a user message
    if not messages or messages[0]["role"] != "user":
        raise HTTPException(status_code=422, detail="Conversation must start with a user message.")

    # Enforce turn cap — truncate gracefully rather than error
    if len(messages) > MAX_TURNS:
        messages = messages[-MAX_TURNS:]

    try:
        result = process_chat(messages)
    except Exception as e:
        logger.exception("Unhandled error in process_chat")
        raise HTTPException(status_code=500, detail="Internal server error.")

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"],
            )
            for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )
