"""FastAPI Server for Ghana Chat API & Web Interface."""

import os
import re
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__, config
from .kg_retriever import VerbalizedKGRetriever
from .generator import GroundedGenerator

logger = logging.getLogger("ghana_chat.server")

app = FastAPI(
    title="Ghana Chat API",
    description="Knowledge Graph Grounded QA powered by Gemma 4 2B on NVIDIA H200",
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
_retriever: Optional[VerbalizedKGRetriever] = None
_generator: Optional[GroundedGenerator] = None


def get_retriever() -> VerbalizedKGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VerbalizedKGRetriever()
    return _retriever


def get_generator() -> GroundedGenerator:
    global _generator
    if _generator is None:
        _generator = GroundedGenerator()
    return _generator


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


# Pronouns / elided references that mean the query depends on prior turns.
_FOLLOWUP_CUES = re.compile(
    r"\b(he|she|they|him|her|them|his|hers|their|it|its|that|this|those|these|the same)\b|"
    r"\b(also|and then|what about|how many|how much|when|where|who)\b",
    re.I,
)


def is_followup_query(query: str, history: Optional[List[ChatMessage]]) -> bool:
    """True when the query leans on prior context rather than standing alone."""
    if not history:
        return False
    if len(query.split()) <= 6 and _FOLLOWUP_CUES.search(query):
        return True
    return bool(re.search(r"\b(he|she|him|her|they|them|his|hers|their)\b", query, re.I))


def retrieval_query(query: str, history: Optional[List[ChatMessage]]) -> str:
    """Expand an anaphoric follow-up so retrieval has something to match on."""
    if not is_followup_query(query, history):
        return query
    prior = [m.content for m in history if m.role == "user"]
    if not prior:
        return query
    return f"{prior[-1]} {query}"


class QuestionRequest(BaseModel):
    question: Optional[str] = Field(default=None, description="The user query or question")
    message: Optional[str] = Field(default=None, description="Alternative message field")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Prior conversation turns")
    max_sources: Optional[int] = Field(default=300, ge=1, le=500)
    country_filter: Optional[str] = Field(default="Ghana")
    stream: Optional[bool] = Field(default=False, description="Stream response tokens via Server-Sent Events (SSE)")


class SourceHit(BaseModel):
    sid: Any = None
    doc_id: Any = None
    sentence: str = ""
    context: str = ""
    head: Optional[str] = ""
    relation: Optional[str] = ""
    tail: Optional[str] = ""
    score: float = 0.0
    matched_candidate: Optional[str] = ""


class QuestionResponse(BaseModel):
    question: str
    answer: str
    reasoning: Optional[str] = Field(default="")
    is_followup: bool
    grounded: bool
    sources: List[SourceHit] = Field(default=[])
    latency_s: float
    tokens_generated: int
    tokens_per_sec: float
    model: str


@app.on_event("startup")
async def startup_event():
    print("Pre-loading verbalized knowledge graph retriever and Gemma 4 2B generator...")
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
        "indexed_facts": r.n if r else 0,
        "retrieval": "verbalized knowledge graph (deduplicated entity-pairs)"
    }


@app.post("/retrieve")
async def retrieve_sources(req: QuestionRequest):
    """Retrieve knowledge graph facts relevant to the query."""
    retriever = get_retriever()
    return retriever.retrieve(req.question or req.message or "", top_k=req.max_sources or 30)


@app.post("/ask", response_model=QuestionResponse)
@app.post("/chat", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    """Answer a question from retrieved corpus passages using noun-phrase retrieval."""
    query = req.question or req.message
    if not query or len(query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    query = query.strip()

    try:
        retriever = get_retriever()
        generator = get_generator()
    except Exception as exc:
        logger.error("Model warm-up failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Model is still starting up. Please retry.")

    # 1. Multi-turn aware noun-phrase retrieval
    try:
        history_prompts = [m.content for m in req.history if m.role == "user"] if req.history else []
        ret_res = retriever.process_turn(
            query=query,
            history_entities=history_prompts,
            max_sources=req.max_sources or 300
        )
    except Exception as exc:
        logger.error("Retrieval failed for %r: %s", query, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Retrieval failed. Please retry.")

    sources = ret_res.get("sources", [])
    grounded = bool(ret_res.get("grounded", True))
    is_followup = bool(ret_res.get("is_followup", False))

    # 2. Keep prior turns only for follow-ups, so unrelated topics don't contaminate.
    history_dicts = ([{"role": m.role, "content": m.content} for m in req.history]
                     if is_followup and req.history else [])

    # Streaming Branch: stream tokens directly to frontend via SSE
    if req.stream:
        def sse_event_stream():
            import json
            meta_data = {
                "sources": [SourceHit(**s).dict() for s in sources],
                "grounded": grounded,
                "is_followup": is_followup,
                "model": config.MODEL_ID
            }
            yield f"event: meta\ndata: {json.dumps(meta_data)}\n\n"

            try:
                for token_chunk in generator.generate_stream(
                    query=query,
                    sources=sources,
                    grounded=grounded,
                    history=history_dicts,
                    abbreviations=ret_res.get("abbreviations", [])
                ):
                    yield f"event: token\ndata: {json.dumps({'token': token_chunk})}\n\n"
            except Exception as e:
                logger.error("Error during streaming generation: %s", e)
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                return

            yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

        return StreamingResponse(
            sse_event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # 3. Non-streaming fallback: Generate complete grounded response
    try:
        gen_res = generator.generate(
            query=query,
            sources=sources,
            grounded=grounded,
            history=history_dicts
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        logger.error("CUDA OOM during generation: %s", exc)
        raise HTTPException(status_code=503, detail="Server is busy. Please retry in a moment.")
    except Exception as exc:
        logger.error("Generation failed for %r: %s", query, exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Answer generation failed. Please retry.")

    return QuestionResponse(
        question=query,
        answer=gen_res["answer"],
        reasoning="",
        is_followup=is_followup,
        grounded=grounded,
        sources=[SourceHit(**s) for s in sources],
        latency_s=gen_res["latency_s"],
        tokens_generated=gen_res["tokens_generated"],
        tokens_per_sec=gen_res["tokens_per_sec"],
        model=config.MODEL_ID
    )
