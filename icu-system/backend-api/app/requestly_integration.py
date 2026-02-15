"""
Requestly Integration Module
Provides API monitoring, mock server fallback, and request interception
"""
import logging
import json
from typing import Optional, Dict
from datetime import datetime
import random

logger = logging.getLogger("requestly-integration")

class RequestlyService:
    """
    Requestly integration for:
    1. API monitoring and traffic visualization
    2. Mock server fallback when Kafka is unavailable
    3. Request/Response interception for debugging
    """
    
    def __init__(self):
        self.enabled = True
        self.mock_mode = False
        self.request_log = []
        logger.info("🔧 Requestly Service initialized (Sponsor Integration)")
    
    def log_api_request(self, endpoint: str, method: str, user: Optional[str] = None):
        """
        Log API requests for Requestly monitoring
        In production, this would send to Requestly API
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "user": user,
            "service": "backend-api"
        }
        self.request_log.append(log_entry)
        
        # Keep only last 1000 requests
        if len(self.request_log) > 1000:
            self.request_log = self.request_log[-1000:]
        
        logger.debug(f"📊 Requestly: {method} {endpoint} by {user}")
    
    def enable_mock_mode(self):
        """Enable mock mode for fallback"""
        self.mock_mode = True
        logger.info("🎭 Requestly Mock Mode ENABLED - Using mock patient data")
    
    def disable_mock_mode(self):
        """Disable mock mode"""
        self.mock_mode = False
        logger.info("✅ Requestly Mock Mode DISABLED - Using live Kafka data")
    
    def get_mock_patient_data(self, patient_id: str, floor_id: str) -> dict:
        """
        Generate mock patient data for demo/testing
        Useful when Kafka is down or for testing specific scenarios
        """
        # Generate realistic mock data based on patient_id
        seed = sum(ord(c) for c in patient_id)
        random.seed(seed)
        
        base_hr = random.randint(60, 100)
        base_sbp = random.randint(110, 140)
        
        return {
            "patient_id": patient_id,
            "floor_id": floor_id,
            "timestamp": datetime.utcnow().isoformat(),
            "rolling_hr": base_hr + random.uniform(-5, 5),
            "rolling_sbp": base_sbp + random.uniform(-10, 10),
            "rolling_spo2": random.uniform(92, 98),
            "hr_trend": random.uniform(-10, 10),
            "sbp_trend": random.uniform(-15, 15),
            "shock_index": random.uniform(0.4, 0.8),
            "spo2": random.uniform(92, 99),
            "state": random.choice(["stable", "stable", "unstable"]),
            "computed_risk": random.uniform(0.2, 0.8),
            "anomaly_flag": random.choice([0, 0, 0, 1]),  # 25% anomaly rate
            "is_high_risk": random.choice([False, False, False, True]),
            "_mock": True,  # Flag to indicate mock data
            "_requestly_source": "mock-server"
        }
    
    def get_mock_patients_for_floor(self, floor_id: str, count: int = 8) -> Dict[str, dict]:
        """Generate mock patients for a floor"""
        patients = {}
        floor_num = floor_id[0]  # Extract floor number (1, 2, 3)
        
        for i in range(1, count + 1):
            patient_id = f"P{floor_num}-{i:03d}"
            patients[patient_id] = self.get_mock_patient_data(patient_id, floor_id)
        
        return patients
    
    def intercept_response(self, data: dict, endpoint: str) -> dict:
        """
        Intercept and potentially modify API responses
        Useful for adding metadata or transforming data
        """
        if self.enabled:
            # Add Requestly metadata to response
            data["_requestly"] = {
                "intercepted": True,
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": endpoint,
                "mock_mode": self.mock_mode
            }
        
        return data
    
    def get_request_analytics(self) -> dict:
        """Get API request analytics for monitoring dashboard"""
        if not self.request_log:
            return {
                "total_requests": 0,
                "endpoints": {},
                "users": {}
            }
        
        endpoints = {}
        users = {}
        
        for log in self.request_log:
            endpoint = log["endpoint"]
            user = log.get("user", "anonymous")
            
            endpoints[endpoint] = endpoints.get(endpoint, 0) + 1
            users[user] = users.get(user, 0) + 1
        
        return {
            "total_requests": len(self.request_log),
            "endpoints": endpoints,
            "users": users,
            "time_range": {
                "start": self.request_log[0]["timestamp"],
                "end": self.request_log[-1]["timestamp"]
            }
        }


# Global Requestly service instance
requestly_service = RequestlyService()
