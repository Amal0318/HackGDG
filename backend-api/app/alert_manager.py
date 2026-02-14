"""
Alert Manager - Phase 3.3
Implements intelligent alert suppression to reduce false positives
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
from pydantic import BaseModel
import logging

logger = logging.getLogger("alert-manager")


class Alert(BaseModel):
    """Single alert event"""
    alert_id: str
    patient_id: str
    alert_type: str  # "HIGH_RISK", "CRITICAL_RISK", "DETERIORATION", "TREATMENT_FAILURE"
    risk_score: float
    timestamp: datetime
    message: str
    suppressed: bool = False
    suppression_reason: Optional[str] = None
    ml_details: Optional[Dict] = None
    
    # Phase 6: Alert acknowledgment tracking
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None  # Clinician ID
    acknowledged_at: Optional[datetime] = None
    outcome: Optional[str] = None  # "true_positive", "false_positive", "intervention_needed", "no_action"
    outcome_notes: Optional[str] = None


class AlertHistory(BaseModel):
    """Track alert history for a patient"""
    patient_id: str
    recent_alerts: deque = deque(maxlen=50)  # Keep last 50 alerts
    last_alert_time: Optional[datetime] = None
    last_alert_type: Optional[str] = None
    last_risk_score: Optional[float] = None
    false_positive_count: int = 0
    true_positive_count: int = 0
    suppressed_count: int = 0


class AlertManager:
    """
    Manages intelligent alert suppression to reduce alarm fatigue.
    
    Rules:
    1. Intervention masking: Suppress alerts for expected physiological responses
    2. Risk progression: Require risk increase, not repeated same-level alerts
    3. Temporal smoothing: Use 3-prediction moving average before alerting
    4. Alert fatigue: Limit alert frequency per patient
    """
    
    # Alert configuration
    RISK_THRESHOLDS = {
        "LOW": 0.3,
        "MODERATE": 0.5,
        "HIGH": 0.7,
        "CRITICAL": 0.85
    }
    
    MIN_ALERT_INTERVAL_MINUTES = 5  # Minimum time between same-type alerts
    MOVING_AVERAGE_WINDOW = 3  # Smooth over 3 predictions
    RISK_INCREASE_THRESHOLD = 0.05  # Require 5% risk increase to re-alert
    
    def __init__(self):
        self.alert_history: Dict[str, AlertHistory] = {}
        self.risk_history: Dict[str, deque] = {}  # Patient → recent risk scores
        self._alert_counter = 0
    
    def _get_or_create_history(self, patient_id: str) -> AlertHistory:
        """Get or create alert history for patient"""
        if patient_id not in self.alert_history:
            self.alert_history[patient_id] = AlertHistory(
                patient_id=patient_id,
                recent_alerts=deque(maxlen=50)
            )
        return self.alert_history[patient_id]
    
    def record_risk_score(self, patient_id: str, risk_score: float):
        """Record a risk score for temporal smoothing"""
        if patient_id not in self.risk_history:
            self.risk_history[patient_id] = deque(maxlen=self.MOVING_AVERAGE_WINDOW)
        
        self.risk_history[patient_id].append(risk_score)
    
    def get_smoothed_risk(self, patient_id: str) -> Optional[float]:
        """
        Get smoothed risk score using moving average.
        
        Returns None if not enough samples yet.
        """
        if patient_id not in self.risk_history:
            return None
        
        scores = list(self.risk_history[patient_id])
        if len(scores) < self.MOVING_AVERAGE_WINDOW:
            return None  # Not enough samples yet
        
        return sum(scores) / len(scores)
    
    def should_suppress_alert(
        self,
        patient_id: str,
        alert_type: str,
        risk_score: float,
        active_intervention_masks: Optional[Dict] = None,
        current_time: Optional[datetime] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if an alert should be suppressed.
        
        Args:
            patient_id: Patient identifier
            alert_type: Type of alert being triggered
            risk_score: Current risk score
            active_intervention_masks: Active intervention masks
            current_time: Current timestamp
        
        Returns:
            (should_suppress, suppression_reason)
        """
        if current_time is None:
            current_time = datetime.now()
        
        history = self._get_or_create_history(patient_id)
        
        # Rule 1: Intervention masking
        # If patient has active interventions, be more lenient with alerts
        if active_intervention_masks and len(active_intervention_masks) > 0:
            # Check if alert is about vitals that are currently masked
            masked_vitals = set(active_intervention_masks.keys())
            
            if len(masked_vitals) >= 2:  # Multiple vitals masked
                # Require higher risk to alert during intervention response period
                if risk_score < self.RISK_THRESHOLDS["HIGH"]:
                    return True, f"Suppressed: {len(masked_vitals)} vitals masked by active interventions"
        
        # Rule 2: Temporal smoothing
        # Use moving average to avoid alerting on single spike
        smoothed_risk = self.get_smoothed_risk(patient_id)
        
        if smoothed_risk is not None:
            # If current risk is high but smoothed risk is lower, suppress
            if risk_score >= self.RISK_THRESHOLDS["HIGH"]:
                if smoothed_risk < self.RISK_THRESHOLDS["MODERATE"]:
                    return True, f"Suppressed: Smoothed risk ({smoothed_risk:.3f}) below moderate threshold"
        else:
            # Not enough samples for smoothing - suppress to avoid early false positives
            if alert_type in ["HIGH_RISK", "CRITICAL_RISK"]:
                return True, "Suppressed: Insufficient data for temporal smoothing"
        
        # Rule 3: Alert fatigue - minimum interval between same-type alerts
        if history.last_alert_type == alert_type and history.last_alert_time:
            time_since_last = (current_time - history.last_alert_time).total_seconds() / 60
            
            if time_since_last < self.MIN_ALERT_INTERVAL_MINUTES:
                return True, f"Suppressed: Same alert type within {self.MIN_ALERT_INTERVAL_MINUTES} minutes"
        
        # Rule 4: Risk progression - require increasing risk, not flat-line alerts
        if history.last_risk_score is not None:
            risk_increase = risk_score - history.last_risk_score
            
            # If risk hasn't increased significantly, suppress
            if risk_increase < self.RISK_INCREASE_THRESHOLD:
                # Exception: Always alert on CRITICAL regardless
                if alert_type != "CRITICAL_RISK":
                    return True, f"Suppressed: Risk increase ({risk_increase:.3f}) below threshold"
        
        # Rule 5: Avoid alert spam - if many recent alerts, require higher threshold
        recent_alert_count = len([
            a for a in history.recent_alerts
            if (current_time - a.timestamp).total_seconds() < 600  # Last 10 minutes
        ])
        
        if recent_alert_count >= 5:
            # Already 5+ alerts in last 10 minutes - require CRITICAL level
            if risk_score < self.RISK_THRESHOLDS["CRITICAL"]:
                return True, f"Suppressed: Alert fatigue ({recent_alert_count} alerts in 10 min)"
        
        # Alert should fire
        return False, None
    
    def create_alert(
        self,
        patient_id: str,
        alert_type: str,
        risk_score: float,
        message: str,
        ml_details: Optional[Dict] = None,
        active_intervention_masks: Optional[Dict] = None,
        current_time: Optional[datetime] = None
    ) -> Alert:
        """
        Create an alert with suppression logic applied.
        
        Args:
            patient_id: Patient identifier
            alert_type: Type of alert
            risk_score: Current risk score
            message: Alert message
            ml_details: ML prediction details
            active_intervention_masks: Active intervention masks
            current_time: Current timestamp
        
        Returns:
            Alert object (may be suppressed)
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Check if alert should be suppressed
        should_suppress, suppression_reason = self.should_suppress_alert(
            patient_id=patient_id,
            alert_type=alert_type,
            risk_score=risk_score,
            active_intervention_masks=active_intervention_masks,
            current_time=current_time
        )
        
        # Create alert
        self._alert_counter += 1
        alert = Alert(
            alert_id=f"ALERT_{patient_id}_{self._alert_counter:06d}",
            patient_id=patient_id,
            alert_type=alert_type,
            risk_score=risk_score,
            timestamp=current_time,
            message=message,
            suppressed=should_suppress,
            suppression_reason=suppression_reason,
            ml_details=ml_details
        )
        
        # Record in history
        history = self._get_or_create_history(patient_id)
        history.recent_alerts.append(alert)
        
        if not should_suppress:
            history.last_alert_time = current_time
            history.last_alert_type = alert_type
            history.last_risk_score = risk_score
            logger.info(f"Alert created: {patient_id} - {alert_type} (risk: {risk_score:.3f})")
        else:
            history.suppressed_count += 1
            logger.debug(f"Alert suppressed: {patient_id} - {suppression_reason}")
        
        return alert
    
    def mark_alert_outcome(
        self,
        patient_id: str,
        alert_id: str,
        was_true_positive: bool
    ):
        """
        Record whether an alert was a true or false positive.
        Used to measure alert accuracy.
        
        Args:
            patient_id: Patient identifier
            alert_id: Alert identifier
            was_true_positive: Whether alert correctly predicted deterioration
        """
        history = self._get_or_create_history(patient_id)
        
        if was_true_positive:
            history.true_positive_count += 1
        else:
            history.false_positive_count += 1
        
        # Find and mark the alert
        for alert in history.recent_alerts:
            if alert.alert_id == alert_id:
                # Could add outcome field to Alert model if needed
                logger.info(f"Alert outcome recorded: {alert_id} - {'TP' if was_true_positive else 'FP'}")
                break
    
    def get_alert_statistics(self, patient_id: str) -> Dict:
        """
        Get alert statistics for a patient.
        
        Returns:
            Dictionary with alert metrics
        """
        if patient_id not in self.alert_history:
            return {
                "total_alerts": 0,
                "suppressed_alerts": 0,
                "true_positives": 0,
                "false_positives": 0,
                "precision": None,
                "suppression_rate": None
            }
        
        history = self.alert_history[patient_id]
        total_alerts = len(history.recent_alerts)
        evaluated_alerts = history.true_positive_count + history.false_positive_count
        
        precision = None
        if evaluated_alerts > 0:
            precision = history.true_positive_count / evaluated_alerts
        
        suppression_rate = None
        if total_alerts > 0:
            suppression_rate = history.suppressed_count / total_alerts
        
        return {
            "patient_id": patient_id,
            "total_alerts": total_alerts,
            "suppressed_alerts": history.suppressed_count,
            "true_positives": history.true_positive_count,
            "false_positives": history.false_positive_count,
            "precision": precision,
            "suppression_rate": suppression_rate,
            "last_alert_time": history.last_alert_time.isoformat() if history.last_alert_time else None,
            "last_alert_type": history.last_alert_type
        }
    
    def get_recent_alerts(
        self,
        patient_id: str,
        include_suppressed: bool = False,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get recent alerts for a patient.
        
        Args:
            patient_id: Patient identifier
            include_suppressed: Whether to include suppressed alerts
            limit: Maximum number of alerts to return
        
        Returns:
            List of alert dictionaries
        """
        if patient_id not in self.alert_history:
            return []
        
        history = self.alert_history[patient_id]
        alerts = list(history.recent_alerts)
        
        # Filter suppressed if requested
        if not include_suppressed:
            alerts = [a for a in alerts if not a.suppressed]
        
        # Sort by timestamp descending
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        # Limit results
        alerts = alerts[:limit]
        
        return [
            {
                "alert_id": a.alert_id,
                "alert_type": a.alert_type,
                "risk_score": a.risk_score,
                "timestamp": a.timestamp.isoformat(),
                "message": a.message,
                "suppressed": a.suppressed,
                "suppression_reason": a.suppression_reason,
                "ml_details": a.ml_details
            }
            for a in alerts
        ]
    
    def clear_patient_alerts(self, patient_id: str):
        """Clear alert history for a patient (e.g., on discharge)"""
        if patient_id in self.alert_history:
            del self.alert_history[patient_id]
        if patient_id in self.risk_history:
            del self.risk_history[patient_id]
        logger.info(f"Cleared alerts for patient {patient_id}")
    
    def get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level string"""
        if risk_score >= self.RISK_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif risk_score >= self.RISK_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif risk_score >= self.RISK_THRESHOLDS["MODERATE"]:
            return "MODERATE"
        else:
            return "LOW"
    
    def should_create_alert(self, risk_score: float, risk_level: str) -> bool:
        """Determine if risk level warrants an alert"""
        if risk_level in ["HIGH", "CRITICAL"]:
            return True
        if risk_level == "MODERATE" and risk_score > 0.6:
            return True
        return False
    
    # ========================================================================
    # Phase 6: Alert Acknowledgment & Analytics
    # ========================================================================
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Retrieve a specific alert by ID"""
        for patient_id, history in self.alert_history.items():
            for alert in history.recent_alerts:
                if alert.alert_id == alert_id:
                    return alert
        return None
    
    def acknowledge_alert(
        self,
        alert_id: str,
        clinician_id: str,
        outcome: str,
        outcome_notes: Optional[str] = None
    ) -> Optional[Alert]:
        """
        Acknowledge an alert and record clinical outcome.
        
        Args:
            alert_id: Alert identifier
            clinician_id: ID of acknowledging clinician
            outcome: One of ["true_positive", "false_positive", "intervention_needed", "no_action"]
            outcome_notes: Optional notes from clinician
        
        Returns:
            Updated alert or None if not found
        """
        alert = self.get_alert_by_id(alert_id)
        
        if not alert:
            logger.warning(f"Alert {alert_id} not found for acknowledgment")
            return None
        
        # Update alert
        alert.acknowledged = True
        alert.acknowledged_by = clinician_id
        alert.acknowledged_at = datetime.now()
        alert.outcome = outcome
        alert.outcome_notes = outcome_notes
        
        # Update patient history statistics
        history = self.alert_history.get(alert.patient_id)
        if history:
            if outcome == "true_positive":
                history.true_positive_count += 1
            elif outcome == "false_positive":
                history.false_positive_count += 1
        
        logger.info(
            f"Alert {alert_id} acknowledged by {clinician_id} as {outcome}"
        )
        
        return alert
    
    def get_analytics(
        self,
        patient_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """
        Calculate alert analytics and accuracy metrics.
        
        Args:
            patient_id: Specific patient or None for all patients
            start_time: Start of time range
            end_time: End of time range
        
        Returns:
            Analytics dictionary with metrics
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)  # Default: last 7 days
        if end_time is None:
            end_time = datetime.now()
        
        # Collect relevant patients
        if patient_id:
            patient_ids = [patient_id] if patient_id in self.alert_history else []
        else:
            patient_ids = list(self.alert_history.keys())
        
        # Initialize metrics
        total_alerts = 0
        suppressed_alerts = 0
        acknowledged_alerts = 0
        true_positives = 0
        false_positives = 0
        unacknowledged_alerts = 0
        
        alerts_by_type = {}
        alerts_by_risk_level = {
            "LOW": 0,
            "MODERATE": 0,
            "HIGH": 0,
            "CRITICAL": 0
        }
        
        # Collect metrics
        for pid in patient_ids:
            history = self.alert_history[pid]
            
            for alert in history.recent_alerts:
                # Filter by time range
                if not (start_time <= alert.timestamp <= end_time):
                    continue
                
                total_alerts += 1
                
                # Count by type
                if alert.alert_type not in alerts_by_type:
                    alerts_by_type[alert.alert_type] = 0
                alerts_by_type[alert.alert_type] += 1
                
                # Count by risk level
                risk_level = self.get_risk_level(alert.risk_score)
                alerts_by_risk_level[risk_level] += 1
                
                # Suppression tracking
                if alert.suppressed:
                    suppressed_alerts += 1
                
                # Acknowledgment tracking
                if alert.acknowledged:
                    acknowledged_alerts += 1
                    if alert.outcome == "true_positive":
                        true_positives += 1
                    elif alert.outcome == "false_positive":
                        false_positives += 1
                else:
                    unacknowledged_alerts += 1
        
        # Calculate rates
        acknowledgment_rate = (
            acknowledged_alerts / total_alerts if total_alerts > 0 else 0
        )
        
        suppression_rate = (
            suppressed_alerts / total_alerts if total_alerts > 0 else 0
        )
        
        # Calculate accuracy (of acknowledged alerts)
        total_with_outcome = true_positives + false_positives
        accuracy = (
            true_positives / total_with_outcome if total_with_outcome > 0 else None
        )
        
        false_positive_rate = (
            false_positives / total_with_outcome if total_with_outcome > 0 else None
        )
        
        return {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "patient_count": len(patient_ids),
            "total_alerts": total_alerts,
            "suppressed_alerts": suppressed_alerts,
            "unsuppressed_alerts": total_alerts - suppressed_alerts,
            "acknowledged_alerts": acknowledged_alerts,
            "unacknowledged_alerts": unacknowledged_alerts,
            "acknowledgment_rate": round(acknowledgment_rate, 3),
            "suppression_rate": round(suppression_rate, 3),
            "outcomes": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "total_with_outcome": total_with_outcome,
                "accuracy": round(accuracy, 3) if accuracy is not None else None,
                "false_positive_rate": round(false_positive_rate, 3) if false_positive_rate is not None else None
            },
            "alerts_by_type": alerts_by_type,
            "alerts_by_risk_level": alerts_by_risk_level
        }
    
    def get_patient_statistics(self, patient_id: str) -> Optional[Dict]:
        """
        Get detailed statistics for a specific patient.
        
        Returns:
            Statistics dictionary or None if patient not found
        """
        if patient_id not in self.alert_history:
            return None
        
        history = self.alert_history[patient_id]
        
        # Calculate metrics
        total_alerts = len(history.recent_alerts)
        acknowledged = sum(1 for a in history.recent_alerts if a.acknowledged)
        suppressed = sum(1 for a in history.recent_alerts if a.suppressed)
        
        return {
            "patient_id": patient_id,
            "total_alerts": total_alerts,
            "acknowledged_alerts": acknowledged,
            "suppressed_alerts": suppressed,
            "true_positives": history.true_positive_count,
            "false_positives": history.false_positive_count,
            "last_alert_time": history.last_alert_time.isoformat() if history.last_alert_time else None,
            "last_alert_type": history.last_alert_type,
            "last_risk_score": history.last_risk_score
        }
