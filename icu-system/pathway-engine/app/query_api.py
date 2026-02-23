"""
Query API - HTTP Interface for Streaming RAG
Exposes retrieval endpoint (NO LLM inside Pathway)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Global reference to RAG index (set by main.py)
rag_index = None

# Create FastAPI app
app = FastAPI(
    title="Pathway RAG Query API",
    description="Streaming vector index query interface",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    """Query request schema"""
    patient_id: str = Field(..., description="Patient identifier")
    query_text: str = Field(..., description="Natural language query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "query_text": "Why is shock index increasing?",
                "top_k": 5
            }
        }

class QueryResponse(BaseModel):
    """Query response schema"""
    patient_id: str
    query_text: str
    retrieved_context: List[Dict]
    result_count: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    rag_index_initialized: bool
    total_patients: int
    total_embeddings: int

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query patient's streaming RAG index
    
    Returns relevant context chunks based on semantic similarity
    NO LLM inference here - pure retrieval only
    """
    
    if rag_index is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG index not initialized"
        )
    
    try:
        # Retrieve relevant context
        results = rag_index.query(
            patient_id=request.patient_id,
            query_text=request.query_text,
            top_k=request.top_k
        )
        
        return QueryResponse(
            patient_id=request.patient_id,
            query_text=request.query_text,
            retrieved_context=results,
            result_count=len(results)
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Query failed: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns RAG index statistics
    """
    
    if rag_index is None:
        return HealthResponse(
            status="unhealthy",
            rag_index_initialized=False,
            total_patients=0,
            total_embeddings=0
        )
    
    try:
        stats = rag_index.get_stats()
        
        return HealthResponse(
            status="healthy",
            rag_index_initialized=True,
            total_patients=stats['total_patients'],
            total_embeddings=stats['total_embeddings']
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="degraded",
            rag_index_initialized=True,
            total_patients=0,
            total_embeddings=0
        )

@app.get("/stats")
async def get_stats():
    """Get detailed RAG index statistics"""
    
    if rag_index is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index not initialized"
        )
    
    try:
        return rag_index.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/patients/{patient_id}/context")
async def get_patient_context(patient_id: str, limit: int = 10):
    """
    Get recent context for a specific patient
    Useful for debugging and monitoring
    """
    
    if rag_index is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index not initialized"
        )
    
    try:
        # Get patient's index directly
        if patient_id not in rag_index.patient_indices:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for patient {patient_id}"
            )
        
        entries = rag_index.patient_indices[patient_id]
        
        # Return most recent entries
        recent_entries = entries[-limit:] if len(entries) > limit else entries
        
        context = []
        for entry in recent_entries:
            context.append({
                'text': entry['text'],
                'timestamp': entry['timestamp'].isoformat(),
                'has_anomaly': entry['raw_data'].get('anomaly_flag', False)
            })
        
        return {
            'patient_id': patient_id,
            'total_entries': len(entries),
            'returned_entries': len(context),
            'context': context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def set_rag_index(index):
    """Set RAG index reference (called by main.py)"""
    global rag_index
    rag_index = index
    logger.info("RAG index reference set in query API")
