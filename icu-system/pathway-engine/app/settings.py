"""
VitalX Pathway Engine Settings
Production-ready configuration management for the ICU Digital Twin streaming pipeline
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class KafkaSettings(BaseSettings):
    """Kafka connection and topic configuration"""
    
    bootstrap_servers: str = Field(
        default="kafka:9092",
        env="KAFKA_BOOTSTRAP_SERVERS",
        description="Kafka broker addresses"
    )
    
    input_topic: str = Field(
        default="vitals",
        env="KAFKA_INPUT_TOPIC", 
        description="Input topic for raw vital signs"
    )
    
    output_topic: str = Field(
        default="vitals_enriched",
        env="KAFKA_OUTPUT_TOPIC",
        description="Output topic for enriched vital signs with risk analysis"
    )
    
    consumer_group: str = Field(
        default="vitalx-pathway-engine",
        env="KAFKA_CONSUMER_GROUP",
        description="Kafka consumer group ID"
    )
    
    # Consumer settings
    auto_offset_reset: str = Field(
        default="latest",
        env="KAFKA_AUTO_OFFSET_RESET",
        description="Auto offset reset policy"
    )
    
    # Producer settings for reliability
    acks: str = Field(
        default="all",
        env="KAFKA_ACKS", 
        description="Acknowledgment policy for producers"
    )
    
    retries: int = Field(
        default=10,
        env="KAFKA_RETRIES",
        description="Number of retries for failed messages"
    )
    
    retry_backoff_ms: int = Field(
        default=300,
        env="KAFKA_RETRY_BACKOFF_MS",
        description="Retry backoff time in milliseconds"
    )

class PathwaySettings(BaseSettings):
    """Pathway streaming engine configuration"""
    
    window_duration_seconds: int = Field(
        default=30,
        env="PATHWAY_WINDOW_DURATION", 
        description="Sliding window duration in seconds for trend analysis"
    )
    
    monitoring_enabled: bool = Field(
        default=True,
        env="PATHWAY_MONITORING_ENABLED",
        description="Enable Pathway monitoring and metrics"
    )

class RiskEngineSettings(BaseSettings):  
    """Risk calculation configuration"""
    
    # Risk calculation weights (must sum to 1.0)
    shock_weight: float = Field(
        default=0.35,
        env="RISK_SHOCK_WEIGHT",
        description="Weight for shock index component in risk calculation"
    )
    
    hr_weight: float = Field(
        default=0.25, 
        env="RISK_HR_WEIGHT",
        description="Weight for heart rate trend component"
    )
    
    sbp_weight: float = Field(
        default=0.20,
        env="RISK_SBP_WEIGHT",
        description="Weight for systolic BP trend component"
    )
    
    spo2_weight: float = Field(
        default=0.20,
        env="RISK_SPO2_WEIGHT", 
        description="Weight for SpO2 component"
    )
    
    # Anomaly detection thresholds
    hr_anomaly_threshold: float = Field(
        default=160.0,
        env="ANOMALY_HR_THRESHOLD",
        description="Heart rate threshold for anomaly detection"
    )
    
    spo2_anomaly_threshold: float = Field(
        default=88.0,
        env="ANOMALY_SPO2_THRESHOLD", 
        description="SpO2 threshold for anomaly detection"
    )
    
    sbp_anomaly_threshold: float = Field(
        default=80.0,
        env="ANOMALY_SBP_THRESHOLD",
        description="Systolic BP threshold for anomaly detection"
    )
    
    shock_index_anomaly_threshold: float = Field(
        default=1.3,
        env="ANOMALY_SHOCK_INDEX_THRESHOLD",
        description="Shock index threshold for anomaly detection"
    )

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT", 
        description="Log message format"
    )

class Settings(BaseSettings):
    """Main application settings container"""
    
    app_name: str = "VitalX Pathway Engine"
    version: str = "1.0.0"
    
    # Component settings
    kafka: KafkaSettings = KafkaSettings()
    pathway: PathwaySettings = PathwaySettings()
    risk_engine: RiskEngineSettings = RiskEngineSettings()
    logging: LoggingSettings = LoggingSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

def validate_risk_weights() -> bool:
    """Validate that risk calculation weights sum to 1.0"""
    total_weight = (
        settings.risk_engine.shock_weight +
        settings.risk_engine.hr_weight +
        settings.risk_engine.sbp_weight +
        settings.risk_engine.spo2_weight
    )
    
    tolerance = 0.001
    if abs(total_weight - 1.0) > tolerance:
        raise ValueError(
            f"Risk weights must sum to 1.0, got {total_weight}. "
            f"Current weights: shock={settings.risk_engine.shock_weight}, "
            f"hr={settings.risk_engine.hr_weight}, "
            f"sbp={settings.risk_engine.sbp_weight}, "
            f"spo2={settings.risk_engine.spo2_weight}"
        )
    
    return True