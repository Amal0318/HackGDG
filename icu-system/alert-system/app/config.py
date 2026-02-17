"""
Configuration for Alert System
"""
import os
from typing import Optional

class Settings:
    """Alert System Configuration"""
    
    # Service Info
    SERVICE_NAME = "ICU Alert System"
    VERSION = "1.0.0"
    
    # LLM Provider (choose: "gemini" or "openai")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # Default to Gemini (free tier)
    
    # Gemini Model (use 2.5-flash for latest, fastest, cheapest)
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Latest Flash model
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Kafka Settings
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    KAFKA_TOPIC = "vitals_predictions"
    KAFKA_GROUP_ID = "alert-system-consumer-v2"  # New group ID to read from earliest
    
    # Alert Thresholds
    HIGH_RISK_THRESHOLD = 0.7  # Trigger alert if computed_risk > 0.7
    CRITICAL_SHOCK_INDEX = 1.0  # Critical if shock_index > 1.0
    CRITICAL_SPO2 = 90  # Critical if SpO2 < 90%
    
    # Notification Channels
    ENABLE_CONSOLE_ALERTS = True
    ENABLE_WEBHOOK_ALERTS = os.getenv("ENABLE_WEBHOOK_ALERTS", "false").lower() == "true"
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # Slack/Discord webhook
    
    # Email Notifications for Doctor Alerts
    ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")  # Your Gmail account
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # Gmail App Password
    ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "icu-alerts@hospital.com")
    ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")  # Doctor's email (comma-separated for multiple)
    
    # Backend API Integration
    BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
    
    # Alert Rate Limiting (avoid spam)
    MIN_ALERT_INTERVAL_SECONDS = 300  # 5 minutes between alerts for same patient
    
    # LLM Settings
    LLM_TEMPERATURE = 0.3  # Lower = more consistent
    LLM_MAX_TOKENS = 500

settings = Settings()
