"""
Backend API Service - REST + WebSocket API Gateway
Phase 0: Minimal FastAPI service without business logic
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
logger = logging.getLogger("backend-api")

# Initialize FastAPI app
app = FastAPI(
    title="ICU Backend API", 
    description="REST + WebSocket API Gateway for ICU Digital Twin System",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    """Service startup event"""
    kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092')
    ml_service_url = os.getenv('ML_SERVICE_URL', 'http://localhost:8001')
    logger.info("Backend API Service started successfully")
    logger.info(f"Connected to Kafka servers: {kafka_servers}")
    logger.info(f"ML Service URL: {ml_service_url}")
    logger.info("Service is ready for API requests")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "backend-api",
            "message": "Backend API Service is running"
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        status_code=200, 
        content={
            "service": "ICU Backend API",
            "status": "operational",
            "version": "0.1.0"
        }
    )