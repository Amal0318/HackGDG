import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from app.config import API_HOST, API_PORT
from app.vector_store import VectorStoreManager
from app.kafka_indexer import KafkaDataIndexer
from app.rag_chain import PatientHandoffRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
vector_store = None
kafka_indexer = None
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global vector_store, kafka_indexer, rag_system
    
    # Startup
    logger.info("Starting RAG Service...")
    
    try:
        # Initialize vector store
        vector_store = VectorStoreManager()
        logger.info("Vector store initialized")
        
        # Initialize RAG system
        rag_system = PatientHandoffRAG(vector_store)
        logger.info("RAG system initialized")
        
        # Initialize and start Kafka indexer
        kafka_indexer = KafkaDataIndexer(vector_store)
        kafka_indexer.start()
        logger.info("Kafka indexer started")
        
        logger.info("RAG Service ready!")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Service...")
    
    if kafka_indexer:
        kafka_indexer.stop()
    
    logger.info("RAG Service stopped")

# Create FastAPI app
app = FastAPI(
    title="ICU Patient Handoff RAG Service",
    description="AI-powered patient handoff assistant using live Kafka data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    patient_id: Optional[str] = None
    time_window_hours: int = 4

class SummaryRequest(BaseModel):
    patient_id: str
    hours: int = 4

class TrendRequest(BaseModel):
    patient_id: str
    vital_name: str
    hours: int = 2

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "ICU Patient Handoff RAG",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if not vector_store or not kafka_indexer or not rag_system:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    indexer_stats = kafka_indexer.get_stats()
    vector_stats = vector_store.get_collection_stats()
    
    return {
        "status": "healthy",
        "kafka_indexer": indexer_stats,
        "vector_store": vector_stats
    }

@app.post("/api/handoff/query")
async def query_patient_data(request: QueryRequest):
    """
    Query patient data using natural language
    
    Examples:
    - "What was the heart rate for patient P1-001 in the last hour?"
    - "Show me vital trends for bed 23"
    - "Any alerts on Floor 2?"
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(
            question=request.question,
            patient_id=request.patient_id,
            time_window_hours=request.time_window_hours
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/handoff/summary")
async def get_patient_summary(request: SummaryRequest):
    """
    Generate a shift handoff summary for a patient
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.get_patient_summary(
            patient_id=request.patient_id,
            hours=request.hours
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/handoff/trend")
async def get_vital_trend(request: TrendRequest):
    """
    Get trending information for a specific vital sign
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.get_trending_vitals(
            patient_id=request.patient_id,
            vital_name=request.vital_name,
            hours=request.hours
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Trend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_statistics():
    """Get service statistics"""
    if not vector_store or not kafka_indexer:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "indexer": kafka_indexer.get_stats(),
        "vector_store": vector_store.get_collection_stats()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False
    )
