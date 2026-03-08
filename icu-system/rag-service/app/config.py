import os

# ─── Kafka ────────────────────────────────────────────────────────────────────
# RAG reads vitals_enriched (already feature-engineered by Pathway Pipeline 1)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "vitals_enriched")
KAFKA_GROUP_ID          = os.getenv("KAFKA_GROUP_ID", "pathway-rag-indexer")

# ─── Pathway VectorStoreServer (internal) ────────────────────────────────────
PATHWAY_HOST = os.getenv("PATHWAY_HOST", "0.0.0.0")
PATHWAY_PORT = int(os.getenv("PATHWAY_PORT", "8666"))

# ─── LLM (Groq) ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── FastAPI ──────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
