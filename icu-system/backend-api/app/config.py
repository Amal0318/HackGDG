"""
Configuration settings for Backend API
"""
import os
from typing import List

class Settings:
    """Application settings"""
    
    # API Settings
    API_TITLE = "ICU Monitoring System API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Real-time ICU patient monitoring with ML predictions"
    
    # Authentication
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours for demo
    
    # Kafka Settings
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    KAFKA_TOPIC_ENRICHED = "vitals_enriched"
    KAFKA_CONSUMER_GROUP = "backend-api-consumer"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "*"  # Allow all for demo (restrict in production)
    ]
    
    # Hospital Configuration
    FLOORS = [
        {"id": "1F", "name": "1st Floor ICU", "capacity": 8},
        {"id": "2F", "name": "2nd Floor ICU", "capacity": 8},
        {"id": "3F", "name": "3rd Floor ICU", "capacity": 8},
    ]
    
    # Demo Users (in production, use database)
    DEMO_USERS = {
        "admin": {
            "username": "admin",
            "password": "admin123",  # In production, use hashed passwords
            "role": "admin",
            "full_name": "System Administrator"
        },
        "doctor": {
            "username": "doctor",
            "password": "doctor123",
            "role": "doctor",
            "full_name": "Dr. Sarah Johnson"
        },
        "nurse": {
            "username": "nurse",
            "password": "nurse123",
            "role": "nurse",
            "full_name": "Nurse Emily Chen"
        }
    }

settings = Settings()
