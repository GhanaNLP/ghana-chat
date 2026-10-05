"""FastAPI Server for Ghana Chat API & Web Interface."""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__, config
from .retriever import HeadKGRetriever
from .generator import GroundedGenerator

app = FastAPI(
    title="Ghana Chat API",
    description="Head-focused Knowledge Graph Grounded QA powered by Qwen 2B on NVIDIA H200",
    version=__version__
)

# Open CORS for external web calls & HF Space frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
_retriever: Optional[HeadKGRetriever] = None
_generator: Optional[GroundedGenerator] = None


def get_retriever() -> HeadKGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HeadKGRetriever()
    return _retriever


def get_generator() -> GroundedGenerator:
    global _generator
    if _generator is None:
        _generator = GroundedGenerator()
    return _generator


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QuestionRequest(BaseModel):
    question: Optional[str] = Field(default=None, description="The user query or question")
    message: Optional[str] = Field(default=None, description="Alternative message field")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Prior conversation turns")
    history_entities: Optional[List[str]] = Field(default=[], description="Entities accumulated in conversation")
    conversation_triples: Optional[List[Dict[str, Any]]] = Field(default=[], description="Triples accumulated so far")
    max_triples: Optional[int] = Field(default=8, ge=1, le=25)
    country_filter: Optional[str] = Field(default="Ghana")


class QuestionResponse(BaseModel):
    question: str
    answer: str
    reasoning: Optional[str] = Field(default="", description="The model's internal step-by-step reasoning process")
    is_followup: bool
    new_entities: List[str]
    all_entities: List[str]
    new_triples: List[Dict[str, Any]]
    all_triples: List[Dict[str, Any]]
    triples: List[Dict[str, Any]]
    entities: List[str]
    latency_s: float
    tokens_generated: int
    tokens_per_sec: float
    model: str


@app.on_event("startup")
async def startup_event():
    print("Pre-loading KG retriever and Qwen 2B generator...")
    try:
        get_retriever()
        get_generator()
    except Exception as e:
        print(f"Warning: Startup loading encountered: {e}. Will lazily initialize on first request.")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the web interface or fallback to API information."""
    html_path = Path(__file__).parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Ghana Chat API</h1><p>Visit <a href='/docs'>/docs</a> for interactive API documentation.</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint providing status and hardware info."""
    import torch
    r = get_retriever()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "status": "healthy",
        "service": "ghana-chat",
        "version": __version__,
        "model": config.MODEL_ID,
        "device": config.DEVICE,
        "device_name": device_name,
        "indexed_heads": len(r.heads_map) if r else 0
    }


@app.post("/retrieve")
async def retrieve_triples(req: QuestionRequest):
    """Retrieve matching Knowledge Graph triples where the query entity is strictly the HEAD."""
    retriever = get_retriever()
    result = retriever.retrieve(req.question, max_triples=req.max_triples, country_filter=req.country_filter)
    return result


@app.post("/ask", response_model=QuestionResponse)
@app.post("/chat", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    """Ask a question or continue a multi-turn conversation with Ghana Chat.

    Follow-up questions referencing established context are automatically detected:
    - If no new entities are introduced: answers from existing context without new retrieval.
    - If new entities are introduced: retrieves new head facts and merges them into context.
    """
    query = req.question or req.message
    if not query or len(query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    query = query.strip()
    retriever = get_retriever()
    generator = get_generator()

    # 1. Multi-turn aware retrieval
    ret_res = retriever.process_turn(
        query=query,
        history_entities=req.history_entities,
        existing_triples=req.conversation_triples,
        max_triples=req.max_triples or 8,
        country_filter=req.country_filter
    )

    active_triples = ret_res.get("active_triples", ret_res["all_triples"])
    all_triples = ret_res["all_triples"]
    new_triples = ret_res["new_triples"]
    is_followup = ret_res["is_followup"]

    # 2. Topic-Aware History Filtering:
    # If this is a FOLLOW-UP (referencing established context), include prior conversation turns.
    # If this is a NEW TOPIC (new entities introduced), prune previous unrelated turns from the prompt
    # so unrelated past entities (e.g. Asiedu Nketia) do not contaminate the new topic (e.g. cassava farming).
    if is_followup and req.history:
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    else:
        history_dicts = []

    # 3. Generate grounded response with Qwen 2B in friendly narrative prose
    gen_res = generator.generate(query=query, triples=active_triples, history=history_dicts)

    return QuestionResponse(
        question=query,
        answer=gen_res["answer"],
        reasoning=gen_res.get("reasoning", ""),
        is_followup=is_followup,
        new_entities=ret_res["new_entities"],
        all_entities=ret_res["all_entities"],
        new_triples=new_triples,
        all_triples=all_triples,
        triples=active_triples,
        entities=ret_res["all_entities"],
        latency_s=gen_res["latency_s"],
        tokens_generated=gen_res["tokens_generated"],
        tokens_per_sec=gen_res["tokens_per_sec"],
        model=config.MODEL_ID
    )
