"""
VitalX Pathway Live RAG Service
================================
This service is the second Pathway pipeline in the VitalX architecture.

  Pipeline 1 (pathway-engine):  vitals  → vitals_enriched  (feature engineering)
  Pipeline 2 (this service):    vitals_enriched → Pathway BaseRAGQuestionAnswerer
                                                → /v1/retrieve         (live KNN search)
                                                → /v1/pw_ai_answer     (full LLM chain inside Pathway)
                                                → /api/handoff/query   (FastAPI proxy → Pathway QA)

Pathway primitives used:
  • pw.io.kafka.read()                      — streaming ingest
  • pw.apply(format_vital_chunk)            — text chunking with real deltas
  • SentenceTransformerEmbedder             — incremental embedding inside Pathway
  • VectorStoreServer                       — always-updated KNN index
  • BaseRAGQuestionAnswerer                 — retrieve + LiteLLMChat (Groq) inside Pathway
"""

import logging
import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from app.config import (
    API_HOST, API_PORT,
    GROQ_API_KEY, GROQ_MODEL,
    PATHWAY_HOST, PATHWAY_PORT,
)
from app.pathway_pipeline import launch_in_background

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Internal URL to talk to the Pathway VectorStoreServer running in-process
PATHWAY_BASE = f"http://localhost:{PATHWAY_PORT}"


def _pathway_retrieve(query: str, k: int = 8, metadata_filter: Optional[dict] = None) -> list:
    """Call Pathway VectorStoreServer /v1/retrieve synchronously."""
    payload = {"query": query, "k": k}
    if metadata_filter:
        payload["metadata_filter"] = metadata_filter
    try:
        resp = httpx.post(f"{PATHWAY_BASE}/v1/retrieve", json=payload, timeout=10.0)
        resp.raise_for_status()
        return resp.json().get("response", [])
    except Exception as e:
        logger.warning(f"Pathway retrieve failed: {e}")
        return []


def _pathway_statistics() -> dict:
    """Call Pathway VectorStoreServer /v1/statistics."""
    try:
        resp = httpx.post(f"{PATHWAY_BASE}/v1/statistics", json={}, timeout=5.0)
        resp.raise_for_status()
        return resp.json().get("response", {})
    except Exception as e:
        logger.warning(f"Pathway statistics failed: {e}")
        return {"num_indexed_texts": 0, "status": "starting"}


def _pathway_answer(question: str, k: int = 8) -> Optional[str]:
    """
    Call Pathway BaseRAGQuestionAnswerer /v1/pw_ai_answer.
    The full retrieve → embed → LLM chain runs INSIDE Pathway (not here in FastAPI).
    Returns the LLM answer string, or None if the endpoint is unavailable.
    """
    try:
        resp = httpx.post(
            f"{PATHWAY_BASE}/v1/pw_ai_answer",
            json={"query": question, "k": k},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result") or data.get("response") or data.get("answer")
        return str(result).strip() if result else None
    except Exception as e:
        logger.debug(f"Pathway QA endpoint unavailable (falling back to direct Groq): {e}")
        return None


def _call_groq(context: str, question: str) -> str:
    """Fallback: call Groq LLM directly with retrieved Pathway context."""
    if not GROQ_API_KEY:
        return f"[LLM unavailable — raw Pathway context]\n\n{context}"
    try:
        from groq import Groq as GroqClient
        client = GroqClient(api_key=GROQ_API_KEY)
        prompt = f"""You are an AI assistant helping ICU nurses during shift handoffs.
Use ONLY the patient data below to answer the question. Be concise (3-5 lines).
Highlight any anomalies or deteriorating trends.

Patient Data (retrieved live from Pathway vector index):
{context}

Nurse's Question: {question}

Answer:"""
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return f"LLM error: {e}\n\nRaw context:\n{context}"


# ─── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("VitalX Pathway Live RAG Service starting")
    logger.info("Launching Pathway VectorStoreServer in background thread...")

    # Start the Pathway pipeline (non-blocking — runs in daemon thread)
    launch_in_background()

    # Give Pathway a moment to initialise before serving requests
    time.sleep(3)
    logger.info(f"FastAPI ready. Pathway VectorStore at {PATHWAY_BASE}")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down Pathway RAG Service")
    # Pathway thread is a daemon — it will exit with the process


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="VitalX Pathway Live RAG Service",
    description=(
        "Real-time ICU patient RAG powered by Pathway. "
        "Full pipeline runs inside Pathway: "
        "Kafka vitals_enriched → pw.io.kafka.read → pw.apply → "
        "SentenceTransformerEmbedder → VectorStoreServer → BaseRAGQuestionAnswerer (LiteLLMChat/Groq)"
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    patient_id: Optional[str] = None
    k: int = 8

class SummaryRequest(BaseModel):
    patient_id: str
    k: int = 10

class TrendRequest(BaseModel):
    patient_id: str
    vital_name: str
    k: int = 6


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    stats = _pathway_statistics()
    return {
        "service": "VitalX Pathway Live RAG",
        "version": "3.0.0",
        "pathway_primitives": [
            "pw.io.kafka.read()         — streaming ingest",
            "pw.apply(format_chunk)     — text chunking with real deltas",
            "SentenceTransformerEmbedder— incremental embedding inside Pathway",
            "VectorStoreServer          — live KNN index",
            "BaseRAGQuestionAnswerer    — retrieve + LiteLLMChat (Groq) inside Pathway",
        ],
        "endpoints": {
            "/v1/retrieve":     "live KNN similarity search",
            "/v1/pw_ai_answer": "full RAG answer — LLM chain runs inside Pathway",
            "/v1/statistics":   "live indexed document count",
        },
        "indexed_chunks": stats.get("num_indexed_texts", 0),
        "vector_store_url": PATHWAY_BASE,
    }


@app.get("/health")
async def health_check():
    """Health check — includes live Pathway index stats."""
    stats = _pathway_statistics()
    indexed = stats.get("num_indexed_texts", 0)
    return {
        "status": "healthy",
        "pathway_pipeline": {
            "url":            PATHWAY_BASE,
            "indexed_chunks": indexed,
            "ready":          indexed > 0,
            "rag_endpoint":   f"{PATHWAY_BASE}/v1/pw_ai_answer",
        },
        "groq_configured": bool(GROQ_API_KEY),
        "embedding_model":  "all-MiniLM-L6-v2",
        "kafka_topic":      "vitals_enriched",
        "pipeline":         "pw.io.kafka.read → pw.apply → SentenceTransformer → VectorStoreServer → BaseRAGQuestionAnswerer",
    }


@app.post("/api/handoff/query")
async def query_patient_data(request: QueryRequest):
    """
    Natural-language nurse query answered by Pathway RAG + Groq.

    Primary flow (full chain inside Pathway):
      1. FastAPI calls Pathway BaseRAGQuestionAnswerer /v1/pw_ai_answer
      2. Pathway retrieves top-k chunks from VectorStoreServer
      3. Pathway calls LiteLLMChat (Groq) — LLM answering inside Pathway

    Fallback (if Pathway QA endpoint not ready):
      1. FastAPI retrieves chunks from /v1/retrieve
      2. FastAPI calls Groq directly
    """
    try:
        # ── Primary: full LLM chain inside Pathway ────────────────────────
        answer = _pathway_answer(request.question, k=request.k)
        if answer:
            return {
                "answer":  answer,
                "success": True,
                "engine":  "Pathway BaseRAGQuestionAnswerer + Groq (LLM inside Pathway)",
            }

        # ── Fallback: retrieve via Pathway, LLM via FastAPI ───────────────
        metadata_filter = None
        if request.patient_id:
            metadata_filter = {"patient_id": request.patient_id}

        chunks = _pathway_retrieve(request.question, k=request.k,
                                   metadata_filter=metadata_filter)

        if not chunks:
            return {
                "answer": "No recent data found in Pathway index. Is the Kafka pipeline running?",
                "context_used": [],
                "success": False,
                "sources": 0,
            }

        context = "\n\n---\n\n".join(
            c.get("text", c.get("data", str(c))) for c in chunks
        )
        answer = _call_groq(context, request.question)

        return {
            "answer":       answer,
            "context_used": [c.get("text", c.get("data", "")) for c in chunks],
            "success":      True,
            "sources":      len(chunks),
            "engine":       "Pathway VectorStoreServer + Groq (fallback)",
        }

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/handoff/summary")
async def get_patient_summary(request: SummaryRequest):
    """Generate a shift-handoff summary for one patient using Pathway RAG."""
    try:
        question = (
            f"Generate a concise shift handoff summary for patient {request.patient_id}. "
            "Include current status, key trends, any anomalies, and recommended actions."
        )

        # Primary: full LLM chain inside Pathway
        summary = _pathway_answer(question, k=request.k)
        if summary:
            return {"summary": summary, "success": True,
                    "engine": "Pathway BaseRAGQuestionAnswerer + Groq"}

        # Fallback
        chunks = _pathway_retrieve(
            f"patient {request.patient_id} vital signs status trends alerts",
            k=request.k,
            metadata_filter={"patient_id": request.patient_id},
        )
        if not chunks:
            return {"summary": f"No data in Pathway index for patient {request.patient_id}.",
                    "success": False}

        context = "\n\n---\n\n".join(c.get("text", c.get("data", str(c))) for c in chunks)
        summary = _call_groq(context, question)
        return {"summary": summary, "success": True, "sources": len(chunks),
                "engine": "Pathway VectorStoreServer + Groq (fallback)"}

    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/handoff/trend")
async def get_vital_trend(request: TrendRequest):
    """Get recent trend for a specific vital sign using Pathway RAG."""
    try:
        question = (
            f"Describe the {request.vital_name} trend for patient {request.patient_id} "
            "over the available timeframe. Note any concerning changes."
        )

        # Primary: full LLM chain inside Pathway
        trend = _pathway_answer(question, k=request.k)
        if trend:
            return {"trend": trend, "success": True,
                    "engine": "Pathway BaseRAGQuestionAnswerer + Groq"}

        # Fallback
        query = f"patient {request.patient_id} {request.vital_name} trend history"
        chunks = _pathway_retrieve(query, k=request.k,
                                   metadata_filter={"patient_id": request.patient_id})
        if not chunks:
            return {"trend": f"No trend data for {request.vital_name} / patient {request.patient_id}.",
                    "success": False}

        context = "\n\n---\n\n".join(c.get("text", c.get("data", str(c))) for c in chunks)
        trend = _call_groq(context, question)
        return {"trend": trend, "success": True, "sources": len(chunks),
                "engine": "Pathway VectorStoreServer + Groq (fallback)"}

    except Exception as e:
        logger.error(f"Trend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    """Live Pathway VectorStore statistics."""
    stats = _pathway_statistics()
    return {
        "pathway_vector_store": stats,
        "endpoint":             PATHWAY_BASE,
        "pipeline":             "vitals_enriched → Pathway → VectorStoreServer",
    }


# ─── Pathway pass-through endpoints (for judges / demo) ───────────────────────

@app.post("/v1/retrieve")
async def pathway_retrieve_passthrough(body: dict):
    """
    Proxy to Pathway VectorStoreServer /v1/retrieve.
    Judges can call this directly to see live vector similarity search results.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{PATHWAY_BASE}/v1/retrieve", json=body, timeout=10.0)
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pathway server error: {e}")


@app.post("/v1/statistics")
async def pathway_statistics_passthrough():
    """
    Proxy to Pathway VectorStoreServer /v1/statistics.
    Watch num_indexed_texts grow in real time as vitals stream in.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{PATHWAY_BASE}/v1/statistics", json={}, timeout=5.0)
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pathway server error: {e}")


@app.post("/v1/inputs")
async def pathway_inputs_passthrough():
    """Proxy to Pathway VectorStoreServer /v1/inputs — list all indexed documents."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{PATHWAY_BASE}/v1/inputs", json={}, timeout=10.0)
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pathway server error: {e}")


@app.post("/v1/pw_ai_answer")
async def pathway_qa_passthrough(body: dict):
    """
    Proxy to Pathway BaseRAGQuestionAnswerer /v1/pw_ai_answer.

    The ENTIRE RAG chain (retrieve → LiteLLMChat/Groq) runs inside Pathway.
    Judges can call this directly to see the full Pathway LLM pipeline.

    Body: {"query": "...", "k": 8}
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{PATHWAY_BASE}/v1/pw_ai_answer",
                json=body,
                timeout=30.0,
            )
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pathway QA server error: {e}")
