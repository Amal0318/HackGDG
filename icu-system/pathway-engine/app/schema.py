"""
VitalX Pathway Engine Data Schemas
Production-ready data models for ICU vital signs streaming pipeline
"""

from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, validator
import json

class VitalSignsInput(BaseModel):
    """Input schema for raw vital signs from the vital-simulator"""
    
    patient_id: str = Field(..., description="Unique patient identifier")
    timestamp: str = Field(..., description="ISO8601 timestamp")
    heart_rate: float = Field(..., ge=0, le=300, description="Heart rate in BPM")
    systolic_bp: float = Field(..., ge=0, le=300, description="Systolic blood pressure in mmHg") 
    diastolic_bp: float = Field(..., ge=0, le=200, description="Diastolic blood pressure in mmHg")
    spo2: float = Field(..., ge=0, le=100, description="Oxygen saturation percentage")
    respiratory_rate: float = Field(..., ge=0, le=60, description="Respiratory rate per minute")
    temperature: float = Field(..., ge=30, le=45, description="Body temperature in Celsius")
    shock_index: float = Field(..., ge=0, description="Calculated shock index (HR/SBP)")
    state: str = Field(..., description="Patient state")
    event_type: Optional[str] = Field(None, description="Event type if any")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp format"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {v}")
    
    @validator('shock_index')
    def validate_shock_index(cls, v, values):
        """Validate shock index is reasonable given HR and SBP"""
        if 'heart_rate' in values and 'systolic_bp' in values:
            expected_si = values['heart_rate'] / max(values['systolic_bp'], 1.0)
            # Allow some tolerance for calculation differences
            if abs(v - expected_si) > 0.1:
                # Log warning but don't fail - use calculated value
                v = expected_si
        return v

class VitalSignsEnriched(BaseModel):
    """Output schema for enriched vital signs with risk analysis"""
    
    # Original fields
    patient_id: str
    timestamp: str 
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    spo2: float
    respiratory_rate: float
    temperature: float
    shock_index: float
    state: str
    event_type: Optional[str] = None
    
    # Enriched fields - rolling averages
    rolling_hr: float = Field(..., description="30-second rolling average heart rate")
    rolling_spo2: float = Field(..., description="30-second rolling average SpO2") 
    rolling_sbp: float = Field(..., description="30-second rolling average systolic BP")
    
    # Trend analysis
    hr_trend: float = Field(..., description="Heart rate trend (current - rolling average)")
    sbp_trend: float = Field(..., description="SBP trend (current - rolling average)")
    
    # Risk analysis
    computed_risk: float = Field(..., ge=0, le=1, description="Computed risk score (0-1)")
    anomaly_flag: bool = Field(..., description="True if any anomaly condition is met")
    
    # Additional enrichment metadata
    enriched_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), 
                           description="Timestamp when enrichment was performed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PathwayStreamRecord(BaseModel):
    """Internal Pathway record structure for streaming operations"""
    
    key: str = Field(..., description="Record key (patient_id)")
    value: Dict[str, Any] = Field(..., description="Record value (vital signs data)")
    timestamp_ms: int = Field(..., description="Event timestamp in milliseconds")
    
    @classmethod
    def from_kafka_message(cls, key: bytes, value: bytes, timestamp_ms: int) -> 'PathwayStreamRecord':
        """Create PathwayStreamRecord from Kafka message"""
        try:
            key_str = key.decode('utf-8') if key else ""
            value_dict = json.loads(value.decode('utf-8'))
            
            return cls(
                key=key_str,
                value=value_dict,
                timestamp_ms=timestamp_ms
            )
        except Exception as e:
            raise ValueError(f"Failed to parse Kafka message: {e}")

class RiskComponents(BaseModel):
    """Detailed risk calculation components for debugging and monitoring"""
    
    shock_component: float = Field(..., ge=0, le=1, description="Normalized shock index component")
    hr_component: float = Field(..., ge=0, le=1, description="Heart rate trend component")
    sbp_component: float = Field(..., ge=0, le=1, description="SBP trend component") 
    spo2_component: float = Field(..., ge=0, le=1, description="SpO2 component")
    
    # Component weights used
    shock_weight: float = Field(..., description="Weight applied to shock component")
    hr_weight: float = Field(..., description="Weight applied to HR component")
    sbp_weight: float = Field(..., description="Weight applied to SBP component")
    spo2_weight: float = Field(..., description="Weight applied to SpO2 component")
    
    # Final risk score
    final_risk: float = Field(..., ge=0, le=1, description="Final computed risk score")

class AnomalyFlags(BaseModel):
    """Detailed anomaly detection flags"""
    
    high_hr: bool = Field(..., description="Heart rate exceeds threshold")
    low_spo2: bool = Field(..., description="SpO2 below threshold") 
    low_sbp: bool = Field(..., description="Systolic BP below threshold")
    high_shock_index: bool = Field(..., description="Shock index above threshold")
    any_anomaly: bool = Field(..., description="True if any anomaly flag is set")

def parse_vital_signs_input(json_data: str) -> VitalSignsInput:
    """Safely parse JSON string to VitalSignsInput with error handling"""
    try:
        data = json.loads(json_data)
        return VitalSignsInput(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Invalid vital signs data: {e}")

def serialize_vital_signs_output(enriched_data: VitalSignsEnriched) -> str:
    """Serialize VitalSignsEnriched to JSON string"""
    try:
        return enriched_data.json()
    except Exception as e:
        raise ValueError(f"Failed to serialize enriched data: {e}")

# Schema validation utilities
def validate_input_schema(data: dict) -> bool:
    """Validate if data conforms to input schema"""
    try:
        VitalSignsInput(**data)
        return True
    except Exception:
        return False

def validate_output_schema(data: dict) -> bool:
    """Validate if data conforms to output schema"""
    try:
        VitalSignsEnriched(**data)
        return True
    except Exception:
        return False