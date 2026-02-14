"""
VitalX Risk Engine
Production-ready risk calculation and anomaly detection for ICU patients
"""

import logging
from typing import Tuple, Dict, Any
import pathway as pw
from .settings import settings
from .schema import VitalSignsInput, VitalSignsEnriched, RiskComponents, AnomalyFlags

logger = logging.getLogger(__name__)

class VitalXRiskEngine:
    """Advanced risk calculation engine for ICU patient monitoring"""
    
    def __init__(self):
        self.shock_weight = settings.risk_engine.shock_weight
        self.hr_weight = settings.risk_engine.hr_weight
        self.sbp_weight = settings.risk_engine.sbp_weight
        self.spo2_weight = settings.risk_engine.spo2_weight
        
        # Anomaly thresholds
        self.hr_threshold = settings.risk_engine.hr_anomaly_threshold
        self.spo2_threshold = settings.risk_engine.spo2_anomaly_threshold
        self.sbp_threshold = settings.risk_engine.sbp_anomaly_threshold
        self.shock_index_threshold = settings.risk_engine.shock_index_anomaly_threshold
        
        logger.info(f"Risk engine initialized with weights: "
                   f"shock={self.shock_weight}, hr={self.hr_weight}, "
                   f"sbp={self.sbp_weight}, spo2={self.spo2_weight}")
    
    def calculate_rolling_averages(self, vitals_stream: pw.Table) -> pw.Table:
        """Calculate 30-second rolling averages using Pathway windowing"""
        
        # Parse timestamp and create time-based windows
        vitals_with_time = vitals_stream.select(
            *vitals_stream,
            event_time=pw.this.timestamp.dt.strptime("%Y-%m-%dT%H:%M:%S.%fZ")
        )
        
        # Group by patient_id and create sliding windows
        windowed = vitals_with_time.windowby(
            vitals_with_time.event_time,
            window=pw.temporal.sliding(
                hop=pw.Duration.seconds(1),  # Update every second
                duration=pw.Duration.seconds(settings.pathway.window_duration_seconds)
            ),
            behavior=pw.temporal.exactly_once_behavior(),  # Process each record exactly once
        ).reduce(
            patient_id=pw.this._pw_window_start_time,  # Group key
            rolling_hr=pw.reducers.avg(pw.this.heart_rate),
            rolling_spo2=pw.reducers.avg(pw.this.spo2),
            rolling_sbp=pw.reducers.avg(pw.this.systolic_bp),
            record_count=pw.reducers.count(),
        )
        
        return windowed
    
    def calculate_trends(self, current_vitals: pw.Table, rolling_averages: pw.Table) -> pw.Table:
        """Calculate vital sign trends (current - rolling average)"""
        
        # Join current vitals with rolling averages by patient_id
        trends = current_vitals.join(
            rolling_averages,
            current_vitals.patient_id == rolling_averages.patient_id,
            how=pw.JoinMode.INNER
        ).select(
            *current_vitals,
            rolling_hr=rolling_averages.rolling_hr,
            rolling_spo2=rolling_averages.rolling_spo2, 
            rolling_sbp=rolling_averages.rolling_sbp,
            hr_trend=current_vitals.heart_rate - rolling_averages.rolling_hr,
            sbp_trend=current_vitals.systolic_bp - rolling_averages.rolling_sbp,
        )
        
        return trends
    
    def normalize_shock_component(self, shock_index: float) -> float:
        """Normalize shock index component (0-1)"""
        return min(1.0, shock_index / 2.0)
    
    def normalize_hr_component(self, hr_trend: float) -> float:
        """Normalize heart rate trend component (0-1)"""
        return min(1.0, abs(hr_trend) / 40.0)
    
    def normalize_sbp_component(self, sbp_trend: float) -> float:
        """Normalize systolic BP trend component (0-1)"""
        return min(1.0, abs(sbp_trend) / 40.0)
    
    def normalize_spo2_component(self, spo2: float) -> float:
        """Normalize SpO2 component (0-1) - lower SpO2 = higher risk"""
        return min(1.0, (100.0 - spo2) / 20.0)
    
    def calculate_risk_score(self, 
                           shock_index: float,
                           hr_trend: float, 
                           sbp_trend: float,
                           spo2: float) -> Tuple[float, RiskComponents]:
        """Calculate comprehensive risk score with component breakdown"""
        
        # Normalize each component
        shock_component = self.normalize_shock_component(shock_index)
        hr_component = self.normalize_hr_component(hr_trend)
        sbp_component = self.normalize_sbp_component(sbp_trend)
        spo2_component = self.normalize_spo2_component(spo2)
        
        # Calculate weighted risk score
        risk_score = (
            self.shock_weight * shock_component +
            self.hr_weight * hr_component +
            self.sbp_weight * sbp_component +
            self.spo2_weight * spo2_component
        )
        
        # Ensure risk score is bounded [0, 1]
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Create detailed risk components for monitoring
        components = RiskComponents(
            shock_component=shock_component,
            hr_component=hr_component,
            sbp_component=sbp_component,
            spo2_component=spo2_component,
            shock_weight=self.shock_weight,
            hr_weight=self.hr_weight,
            sbp_weight=self.sbp_weight,
            spo2_weight=self.spo2_weight,
            final_risk=risk_score
        )
        
        return risk_score, components
    
    def detect_anomalies(self,
                        heart_rate: float,
                        spo2: float, 
                        systolic_bp: float,
                        shock_index: float) -> AnomalyFlags:
        """Detect critical anomalies in vital signs"""
        
        high_hr = heart_rate > self.hr_threshold
        low_spo2 = spo2 < self.spo2_threshold
        low_sbp = systolic_bp < self.sbp_threshold
        high_shock_index = shock_index > self.shock_index_threshold
        
        any_anomaly = any([high_hr, low_spo2, low_sbp, high_shock_index])
        
        return AnomalyFlags(
            high_hr=high_hr,
            low_spo2=low_spo2,
            low_sbp=low_sbp,
            high_shock_index=high_shock_index,
            any_anomaly=any_anomaly
        )
    
    def enrich_vitals_stream(self, vitals_stream: pw.Table) -> pw.Table:
        """Main function to enrich vital signs stream with risk analysis"""
        
        # Calculate rolling averages
        logger.info("Calculating rolling averages for trend analysis")
        rolling_averages = self.calculate_rolling_averages(vitals_stream)
        
        # Calculate trends
        logger.info("Calculating vital sign trends")
        trends = self.calculate_trends(vitals_stream, rolling_averages)
        
        # Apply risk calculations and anomaly detection
        enriched = trends.select(
            # Original fields
            patient_id=pw.this.patient_id,
            timestamp=pw.this.timestamp,
            heart_rate=pw.this.heart_rate,
            systolic_bp=pw.this.systolic_bp,
            diastolic_bp=pw.this.diastolic_bp,
            spo2=pw.this.spo2,
            respiratory_rate=pw.this.respiratory_rate,
            temperature=pw.this.temperature,
            shock_index=pw.this.shock_index,
            state=pw.this.state,
            event_type=pw.this.event_type,
            
            # Rolling averages
            rolling_hr=pw.this.rolling_hr,
            rolling_spo2=pw.this.rolling_spo2,
            rolling_sbp=pw.this.rolling_sbp,
            
            # Trends
            hr_trend=pw.this.hr_trend,
            sbp_trend=pw.this.sbp_trend,
            
            # Risk calculations using UDFs
            computed_risk=pw.apply(
                self._calculate_risk_udf,
                pw.this.shock_index,
                pw.this.hr_trend,
                pw.this.sbp_trend,
                pw.this.spo2
            ),
            
            # Anomaly detection
            anomaly_flag=pw.apply(
                self._detect_anomalies_udf,
                pw.this.heart_rate,
                pw.this.spo2,
                pw.this.systolic_bp,
                pw.this.shock_index
            ),
            
            # Metadata
            enriched_at=pw.apply(lambda: pw.temporal.now().isoformat()),
        )
        
        return enriched
    
    def _calculate_risk_udf(self, 
                          shock_index: float,
                          hr_trend: float,
                          sbp_trend: float, 
                          spo2: float) -> float:
        """User-defined function for risk calculation"""
        try:
            risk_score, _ = self.calculate_risk_score(shock_index, hr_trend, sbp_trend, spo2)
            return risk_score
        except Exception as e:
            logger.warning(f"Risk calculation failed: {e}, returning 0.0")
            return 0.0
    
    def _detect_anomalies_udf(self,
                            heart_rate: float,
                            spo2: float,
                            systolic_bp: float, 
                            shock_index: float) -> bool:
        """User-defined function for anomaly detection"""
        try:
            anomalies = self.detect_anomalies(heart_rate, spo2, systolic_bp, shock_index)
            return anomalies.any_anomaly
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}, returning False")
            return False

def create_risk_engine() -> VitalXRiskEngine:
    """Factory function to create risk engine instance"""
    try:
        # Validate weights before creating engine
        from .settings import validate_risk_weights
        validate_risk_weights()
        
        engine = VitalXRiskEngine()
        logger.info("VitalX Risk Engine successfully created")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create risk engine: {e}")
        raise RuntimeError(f"Risk engine initialization failed: {e}")

def get_risk_engine_health() -> Dict[str, Any]:
    """Get risk engine health status for monitoring"""
    try:
        engine = VitalXRiskEngine()
        
        return {
            "risk_engine_healthy": True,
            "shock_weight": engine.shock_weight,
            "hr_weight": engine.hr_weight, 
            "sbp_weight": engine.sbp_weight,
            "spo2_weight": engine.spo2_weight,
            "hr_threshold": engine.hr_threshold,
            "spo2_threshold": engine.spo2_threshold,
            "sbp_threshold": engine.sbp_threshold,
            "shock_index_threshold": engine.shock_index_threshold,
        }
    except Exception as e:
        return {
            "risk_engine_healthy": False,
            "error": str(e)
        }