"""
LangChain Agent for Intelligent Alert Generation
Uses Gemini or OpenAI to generate contextual medical alerts
"""
import logging
from typing import Dict, Optional
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .config import settings

logger = logging.getLogger("langchain-agent")


class MedicalAlertAgent:
    """
    LangChain-powered agent for generating intelligent medical alerts
    """
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.alert_chain = self._create_alert_chain()
        logger.info(f"🧠 Medical Alert Agent initialized with {settings.LLM_PROVIDER.upper()}")
    
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        if settings.LLM_PROVIDER == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set in environment")
            
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,  # Use Flash model for fast, cheap generation
                google_api_key=settings.GEMINI_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_TOKENS
            )
        
        elif settings.LLM_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment")
            
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )
        
        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
    
    def _create_alert_chain(self) -> LLMChain:
        """Create LangChain prompt chain for alert generation"""
        
        prompt_template = """You are an ICU monitoring AI assistant. Generate a clear, concise medical alert for a doctor.

Patient Data:
- Patient ID: {patient_id}
- Floor: {floor_id}
- Current State: {state}

Vital Signs:
- Heart Rate: {heart_rate} bpm (rolling avg)
- Blood Pressure (Systolic): {sbp} mmHg (rolling avg)
- SpO2: {spo2}%
- Shock Index: {shock_index}

Trends & Analysis:
- Heart Rate Trend: {hr_trend}
- Blood Pressure Trend: {sbp_trend}
- Risk Score: {risk_score}
- Anomaly Detected: {anomaly_flag}

Current Time: {timestamp}

Generate a professional medical alert with:
1. Severity assessment (CRITICAL / HIGH / MEDIUM)
2. Key findings (max 3 bullet points)
3. Recommended immediate action
4. Brief clinical reasoning

Keep it concise and actionable for emergency response.
"""
        
        prompt = PromptTemplate(
            input_variables=[
                "patient_id", "floor_id", "state",
                "heart_rate", "sbp", "spo2", "shock_index",
                "hr_trend", "sbp_trend", "risk_score", "anomaly_flag",
                "timestamp"
            ],
            template=prompt_template
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def generate_alert(self, patient_data: Dict) -> Optional[Dict]:
        """
        Generate intelligent alert using LangChain
        
        Args:
            patient_data: Patient vital signs and predictions
            
        Returns:
            Alert dictionary with generated content and metadata
        """
        # Check if alert is warranted
        if not self._should_alert(patient_data):
            return None
        
        # TEMPORARY: Skip LLM and use fallback to avoid API key leak issues
        logger.info(f"🧠 Generating rule-based alert for patient {patient_data.get('patient_id')}")
        return self._generate_fallback_alert(patient_data)
    
    def _should_alert(self, data: Dict) -> bool:
        """Determine if alert should be triggered"""
        # Extract risk_score from actual Kafka message format
        risk_score = data.get("risk_score", 0)
        
        # Extract vitals from nested object
        vitals = data.get("vitals", {})
        spo2 = vitals.get("spo2", 100)
        heart_rate = vitals.get("heart_rate", 0)
        systolic_bp = vitals.get("systolic_bp", 0)
        
        # Calculate shock index (HR / SBP)
        shock_index = heart_rate / systolic_bp if systolic_bp > 0 else 0
        
        # Trigger conditions
        if risk_score > settings.HIGH_RISK_THRESHOLD:
            logger.info(f"🚨 HIGH RISK DETECTED: {data.get('patient_id')} - Risk={risk_score:.3f}")
            return True
        if shock_index > settings.CRITICAL_SHOCK_INDEX:
            logger.info(f"🚨 CRITICAL SHOCK INDEX: {data.get('patient_id')} - SI={shock_index:.2f}")
            return True
        if spo2 < settings.CRITICAL_SPO2:
            logger.info(f"🚨 LOW SPO2: {data.get('patient_id')} - SpO2={spo2}")
            return True
        
        return False
    
    def _prepare_llm_input(self, data: Dict) -> Dict:
        """Prepare data for LLM input - adapted for actual Kafka message format"""
        vitals = data.get("vitals", {})
        heart_rate = vitals.get("heart_rate", 0)
        sbp = vitals.get("systolic_bp", 0)
        spo2 = vitals.get("spo2", 0)
        shock_index = heart_rate / sbp if sbp > 0 else 0
        
        return {
            "patient_id": data.get("patient_id", "Unknown"),
            "floor_id": "ICU Floor 1",  # Default since not in message
            "state": "monitoring",
            "heart_rate": round(heart_rate, 1),
            "sbp": round(sbp, 1),
            "spo2": round(spo2, 1),
            "shock_index": round(shock_index, 2),
            "hr_trend": "stable",  # Not available in message
            "sbp_trend": "stable",  # Not available in message
            "risk_score": round(data.get("risk_score", 0), 3),
            "anomaly_flag": "NO",  # Not available in message
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    
    def _determine_severity(self, data: Dict) -> str:
        """Determine alert severity level"""
        vitals = data.get("vitals", {})
        heart_rate = vitals.get("heart_rate", 0)
        sbp = vitals.get("systolic_bp", 0)
        spo2 = vitals.get("spo2", 100)
        risk_score = data.get("risk_score", 0)
        
        # Calculate shock index
        shock_index = heart_rate / sbp if sbp > 0 else 0
        
        # CRITICAL conditions
        if shock_index > 1.5 or spo2 < 85 or risk_score > 0.9:
            return "CRITICAL"
        
        # HIGH conditions
        if shock_index > 1.0 or spo2 < 90 or risk_score > 0.7:
            return "HIGH"
        
        # MEDIUM (default)
        return "MEDIUM"
    
    def _generate_fallback_alert(self, data: Dict) -> Dict:
        """Generate rule-based alert if LLM fails"""
        patient_id = data.get("patient_id", "Unknown")
        severity = self._determine_severity(data)
        vitals = data.get("vitals", {})
        
        heart_rate = vitals.get("heart_rate", 0)
        sbp = vitals.get("systolic_bp", 0)
        spo2 = vitals.get("spo2", 0)
        shock_index = heart_rate / sbp if sbp > 0 else 0
        
        fallback_message = f"""
🚨 {severity} PRIORITY ALERT - ICU Floor 1

Patient: {patient_id}
State: MONITORING

Key Vitals:
- Heart Rate: {heart_rate:.1f} bpm
- Blood Pressure: {sbp:.1f} mmHg
- SpO2: {spo2:.1f}%
- Shock Index: {shock_index:.2f}

Risk Score: {data.get('risk_score', 0):.3f}
Model: {data.get('model_type', 'Unknown')}

⚠️ Immediate medical review recommended
(Fallback alert - LLM unavailable)
"""
        
        return {
            "patient_id": patient_id,
            "floor_id": "ICU Floor 1",
            "severity": severity,
            "alert_message": fallback_message.strip(),
            "raw_data": data,
            "generated_at": datetime.utcnow().isoformat(),
            "llm_provider": "fallback_rules",
            "alert_type": "high_risk_patient"
        }
