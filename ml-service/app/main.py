"""
ML Service - Machine Learning Inference Engine
Phase 0: Minimal FastAPI service without ML logic
"""

import os
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml-service")

# Initialize FastAPI app
app = FastAPI(
    title="ICU ML Service",
    description="Machine Learning Inference Engine for ICU Deterioration Prediction",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    """Service startup event"""
    kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092')
    logger.info("ML Service started successfully")
    logger.info(f"Connected to Kafka servers: {kafka_servers}")
    logger.info("Service is ready for ML inference requests")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "ml-service", 
            "message": "ML Service is running"
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "service": "ICU ML Service",
            "status": "operational",
            "version": "0.1.0"
        }
    )