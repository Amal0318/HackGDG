"""
Pathway Live RAG Pipeline — VitalX ICU Monitoring
==================================================
Pipeline 2 of the Pathway backbone:

  vitals_enriched (Kafka)
       │
       ▼  pw.io.kafka.read()              ← live streaming ingest
  Pathway Table (enriched vitals)
       │
       ▼  pw.apply(format_vital_chunk)    ← text chunking per row (with real deltas)
  Natural-language chunks
       │
       ▼  SentenceTransformerEmbedder     ← incremental embedding inside Pathway
  Embeddings (auto-updated as new vitals stream in)
       │
       ▼  VectorStoreServer               ← live KNN index
       │
       ▼  BaseRAGQuestionAnswerer         ← retrieve + LiteLLMChat (Groq) inside Pathway
  /v1/retrieve       ← similarity search
  /v1/statistics     ← live document count
  /v1/inputs         ← list all indexed chunks
  /v1/pw_ai_answer   ← full RAG: retrieve → LLM answer (entire chain inside Pathway)
"""

import logging
import os
import threading

import pathway as pw
from pathway.xpacks.llm import embedders, llms
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer

from app.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    KAFKA_GROUP_ID,
    PATHWAY_HOST,
    PATHWAY_PORT,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)

logger = logging.getLogger(__name__)

# ─── Pathway Schema ────────────────────────────────────────────────────────────

class VitalsEnrichedSchema(pw.Schema):
    """Schema matching the vitals_enriched Kafka topic produced by Pathway Engine"""
    patient_id:          str   = pw.column_definition(default_value="unknown")
    timestamp:           str   = pw.column_definition(default_value="")
    state:               str   = pw.column_definition(default_value="UNKNOWN")
    # Raw current vitals
    heart_rate:          float = pw.column_definition(default_value=0.0)
    systolic_bp:         float = pw.column_definition(default_value=0.0)
    diastolic_bp:        float = pw.column_definition(default_value=0.0)
    spo2:                float = pw.column_definition(default_value=0.0)
    respiratory_rate:    float = pw.column_definition(default_value=0.0)
    temperature:         float = pw.column_definition(default_value=0.0)
    shock_index:         float = pw.column_definition(default_value=0.0)
    # Rolling baselines (running avg per patient)
    rolling_hr:          float = pw.column_definition(default_value=0.0)
    rolling_spo2:        float = pw.column_definition(default_value=0.0)
    rolling_sbp:         float = pw.column_definition(default_value=0.0)
    # Trends
    hr_trend:            float = pw.column_definition(default_value=0.0)
    sbp_trend:           float = pw.column_definition(default_value=0.0)
    # Real deltas: current − running_average
    hr_delta:            float = pw.column_definition(default_value=0.0)
    sbp_delta:           float = pw.column_definition(default_value=0.0)
    spo2_delta:          float = pw.column_definition(default_value=0.0)
    shock_index_delta:   float = pw.column_definition(default_value=0.0)
    # Risk + anomaly flags
    computed_risk:       float = pw.column_definition(default_value=0.0)
    triage_level:        str   = pw.column_definition(default_value="STABLE")
    anomaly_flag:        bool  = pw.column_definition(default_value=False)
    hr_anomaly:          bool  = pw.column_definition(default_value=False)
    sbp_anomaly:         bool  = pw.column_definition(default_value=False)
    spo2_anomaly:        bool  = pw.column_definition(default_value=False)
    shock_index_anomaly: bool  = pw.column_definition(default_value=False)


# ─── Text Formatting ────────────────────────────────────────────────────────────

def _delta_symbol(delta: float) -> str:
    if delta > 0.5:  return "↑"
    if delta < -0.5: return "↓"
    return "→"


def format_vital_chunk(
    patient_id: str,
    timestamp: str,
    state: str,
    triage_level: str,
    heart_rate: float,
    systolic_bp: float,
    diastolic_bp: float,
    spo2: float,
    respiratory_rate: float,
    temperature: float,
    shock_index: float,
    rolling_hr: float,
    rolling_spo2: float,
    rolling_sbp: float,
    hr_delta: float,
    sbp_delta: float,
    spo2_delta: float,
    shock_index_delta: float,
    anomaly_flag: bool,
    hr_anomaly: bool,
    sbp_anomaly: bool,
    spo2_anomaly: bool,
    shock_index_anomaly: bool,
) -> str:
    """
    Convert one enriched vital-sign row into a rich natural-language chunk.
    Deltas are REAL values (current − running_average), not zeros.
    triage_level is computed by pw.if_else() inside Pathway (no Python UDF).

    Example output:
      '[14:32] Patient P003 | State EARLY_DETERIORATION | Triage: HIGH
       HR 108 ↑(+9.3 vs baseline 98.7) | SBP 88 ↓(-11.2 vs baseline 99.2)
       SpO2 91 ↓(-2.1 vs baseline 93.1) | RR 24 | Temp 38.4°C
       ShockIndex 1.23 ↑(+0.18 vs baseline 1.05)
       Baseline — AvgHR: 98.7 | AvgSBP: 99.2 | AvgSpO2: 93.1
       ⚠ ANOMALY: Hypotension, Hypoxia, High shock index'
    """
    time_str = ""
    if timestamp and "T" in timestamp:
        time_str = timestamp.split("T")[1][:5]
    elif timestamp:
        time_str = timestamp[:5]

    anomalies = []
    if hr_anomaly:          anomalies.append("Tachycardia")
    if sbp_anomaly:         anomalies.append("Hypotension")
    if spo2_anomaly:        anomalies.append("Hypoxia")
    if shock_index_anomaly: anomalies.append("High shock index")

    pulse_pressure = systolic_bp - diastolic_bp

    lines = [
        f"[{time_str}] Patient {patient_id} | State {state} | Triage: {triage_level}",
        (
            f"HR {heart_rate:.0f} {_delta_symbol(hr_delta)}({hr_delta:+.1f} vs baseline {rolling_hr:.1f}) | "
            f"SBP {systolic_bp:.0f} {_delta_symbol(sbp_delta)}({sbp_delta:+.1f} vs baseline {rolling_sbp:.1f}) | "
            f"SpO2 {spo2:.0f} {_delta_symbol(spo2_delta)}({spo2_delta:+.1f} vs baseline {rolling_spo2:.1f})"
        ),
        f"RR {respiratory_rate:.0f} | Temp {temperature:.1f}\u00b0C | PulsePressure {pulse_pressure:.0f}",
        (
            f"ShockIndex {shock_index:.2f} {_delta_symbol(shock_index_delta)}"
            f"({shock_index_delta:+.2f} vs baseline)"
        ),
        f"Baseline \u2014 AvgHR: {rolling_hr:.1f} | AvgSBP: {rolling_sbp:.1f} | AvgSpO2: {rolling_spo2:.1f}",
    ]

    if anomaly_flag and anomalies:
        lines.append(f"\u26a0 ANOMALY: {', '.join(anomalies)}")
    else:
        lines.append("Status: STABLE \u2014 all vitals within normal range")

    return "\n".join(lines)


def build_metadata(
    patient_id: str,
    timestamp: str,
    state: str,
    triage_level: str,
    anomaly_flag: bool,
    shock_index: float,
    spo2: float,
) -> dict:
    return {
        "patient_id":   patient_id,
        "timestamp":    timestamp,
        "state":        state,
        "triage_level": triage_level,
        "anomaly_flag": anomaly_flag,
        "shock_index":  shock_index,
        "spo2":         spo2,
    }


# ─── Pipeline ──────────────────────────────────────────────────────────────────

_pipeline_started = False
_pipeline_lock    = threading.Lock()


def start_pathway_rag_pipeline():
    """
    Build and run the Pathway Live RAG pipeline.
    This is BLOCKING — call it inside a daemon thread.

    Full Pathway pipeline (everything inside Pathway):
      Kafka vitals_enriched
        → pw.io.kafka.read()              [streaming ingest]
        → pw.apply(format_vital_chunk)    [text chunking + real deltas]
        → SentenceTransformerEmbedder     [incremental embedding]
        → VectorStoreServer               [live KNN index]
        → BaseRAGQuestionAnswerer         [retrieve + LiteLLMChat Groq inside Pathway]
          └── /v1/retrieve
          └── /v1/pw_ai_answer            ← NEW: full LLM chain inside Pathway
          └── /v1/statistics
          └── /v1/inputs
    """
    global _pipeline_started

    with _pipeline_lock:
        if _pipeline_started:
            logger.warning("Pathway RAG pipeline already requested, skipping duplicate start")
            return
        _pipeline_started = True

    logger.info("=" * 60)
    logger.info("  Pathway Live RAG Pipeline starting")
    logger.info(f"  Kafka:  {KAFKA_BOOTSTRAP_SERVERS}  topic={KAFKA_TOPIC}")
    logger.info(f"  Model:  {EMBEDDING_MODEL}")
    logger.info(f"  Server: {PATHWAY_HOST}:{PATHWAY_PORT}")
    logger.info("=" * 60)

    # ── 1. Ingest from Kafka ────────────────────────────────────────────────
    vitals_stream = pw.io.kafka.read(
        rdkafka_settings={
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id":          KAFKA_GROUP_ID,
            "auto.offset.reset": "latest",
            "enable.auto.commit": "true",
            "session.timeout.ms": "30000",
        },
        topic=KAFKA_TOPIC,
        format="json",
        schema=VitalsEnrichedSchema,
        autocommit_duration_ms=500,
    )

    # ── 2. Format each row as a rich text chunk ─────────────────────────────
    docs = vitals_stream.select(
        # 'data' is the column VectorStoreServer reads for embedding
        data=pw.apply(
            format_vital_chunk,
            pw.this.patient_id,
            pw.this.timestamp,
            pw.this.state,
            pw.this.triage_level,
            pw.this.heart_rate,
            pw.this.systolic_bp,
            pw.this.diastolic_bp,
            pw.this.spo2,
            pw.this.respiratory_rate,
            pw.this.temperature,
            pw.this.shock_index,
            pw.this.rolling_hr,
            pw.this.rolling_spo2,
            pw.this.rolling_sbp,
            pw.this.hr_delta,
            pw.this.sbp_delta,
            pw.this.spo2_delta,
            pw.this.shock_index_delta,
            pw.this.anomaly_flag,
            pw.this.hr_anomaly,
            pw.this.sbp_anomaly,
            pw.this.spo2_anomaly,
            pw.this.shock_index_anomaly,
        ),
        # '_metadata' is returned alongside retrieved chunks
        _metadata=pw.apply(
            build_metadata,
            pw.this.patient_id,
            pw.this.timestamp,
            pw.this.state,
            pw.this.triage_level,
            pw.this.anomaly_flag,
            pw.this.shock_index,
            pw.this.spo2,
        ),
    )

    # ── 3. Embed + index ──────────────────────────────────────────────────────
    embedder = embedders.SentenceTransformerEmbedder(EMBEDDING_MODEL)

    vector_server = VectorStoreServer(
        docs,
        embedder=embedder,
    )

    # ── 4. Pathway-native LLM answering (BaseRAGQuestionAnswerer) ─────────────
    # This exposes /v1/pw_ai_answer so the ENTIRE retrieve → embed → LLM chain
    # runs inside Pathway — nothing delegated to FastAPI for QA.
    if GROQ_API_KEY:
        # Make key available to LiteLLM (used internally by Pathway)
        os.environ.setdefault("GROQ_API_KEY", GROQ_API_KEY)

        llm = llms.LiteLLMChat(
            model=f"groq/{GROQ_MODEL}",
            temperature=0.1,
            max_tokens=512,
        )

        rag_app = BaseRAGQuestionAnswerer(
            llm=llm,
            indexer=vector_server,
            search_topk=8,
        )

        logger.info("=" * 60)
        logger.info("  Pathway BaseRAGQuestionAnswerer ACTIVE")
        logger.info("  Full RAG chain runs inside Pathway:")
        logger.info("    Kafka → pw.apply → SentenceTransformer → VectorStore → LiteLLMChat")
        logger.info("  Endpoints:")
        logger.info(f"    /v1/retrieve       — live KNN similarity search")
        logger.info(f"    /v1/pw_ai_answer   — full RAG: retrieve + Groq LLM (inside Pathway)")
        logger.info(f"    /v1/statistics     — live index stats")
        logger.info("=" * 60)

        # Blocks — Pathway drives the full streaming + LLM pipeline
        rag_app.run_server(
            host=PATHWAY_HOST,
            port=PATHWAY_PORT,
            with_cache=False,
            threaded=False,
        )
    else:
        # No LLM key — fall back to vector store only
        logger.warning("GROQ_API_KEY not set — running VectorStoreServer only (no /v1/pw_ai_answer)")
        vector_server.run_server(
            host=PATHWAY_HOST,
            port=PATHWAY_PORT,
            with_cache=False,
            threaded=False,
        )


def launch_in_background():
    """Launch the Pathway pipeline in a daemon thread (non-blocking for FastAPI)."""
    t = threading.Thread(target=start_pathway_rag_pipeline, daemon=True, name="pathway-rag")
    t.start()
    logger.info(f"Pathway RAG pipeline thread launched (daemon={t.daemon})")
    return t
