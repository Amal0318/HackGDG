import os
from typing import Optional

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPICS = os.getenv("KAFKA_TOPICS", "vitals").split(",")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "rag-indexer-group")

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "patient_records")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq or ollama
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Set this in environment
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Data retention (in hours)
DATA_RETENTION_HOURS = int(os.getenv("DATA_RETENTION_HOURS", "24"))

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
