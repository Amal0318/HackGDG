"""
Intervention Tracker - Phase 3.1
Tracks clinical interventions and masks expected physiological responses
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger("intervention-tracker")


class InterventionType(str, Enum):
    """Clinical intervention types with expected effects"""
    VASOPRESSORS = "vasopressors"
    NEBULIZER = "nebulizer"
    DIURETIC = "diuretic"
    INSULIN = "insulin"
    FLUIDS = "fluids"
    OXYGEN = "oxygen"
    BETA_BLOCKER = "beta_blocker"


class VitalMask(BaseModel):
    """Defines which vitals to mask and expected direction"""
    vital_name: str  # HR, SpO2, SBP, RR, Temp
    expected_direction: str  # "increase", "decrease", "stabilize"
    mask_duration_minutes: int
    threshold_change: float  # Expected magnitude of change


class Intervention(BaseModel):
    """Single intervention event"""
    intervention_id: str
    patient_id: str
    intervention_type: InterventionType
    timestamp: datetime
    expected_effects: List[VitalMask]
    dosage: Optional[str] = None
    administered_by: Optional[str] = None
    notes: Optional[str] = None
    
    # Tracking
    is_active: bool = True  # Whether masks are still applied
    effectiveness_checked: bool = False
    treatment_effective: Optional[bool] = None
    actual_changes: Dict[str, float] = Field(default_factory=dict)


class InterventionTracker:
    """
    Tracks clinical interventions and applies intelligent masking to prevent
    false alarms when expected physiological responses occur.
    
    Example: After vasopressor administration, rising BP is expected and should
    not trigger deterioration alerts.
    """
    
    # Intervention definitions with expected effects
    INTERVENTION_PROFILES = {
        InterventionType.VASOPRESSORS: {
            "masks": [
                VitalMask(
                    vital_name="SBP",
                    expected_direction="increase",
                    mask_duration_minutes=10,
                    threshold_change=15.0  # Expect SBP to rise by ~15 mmHg
                ),
                VitalMask(
                    vital_name="HR",
                    expected_direction="increase",
                    mask_duration_minutes=10,
                    threshold_change=10.0  # May see reflex tachycardia
                )
            ],
            "response_window_minutes": 10,
            "expected_response": "BP should increase within 10 minutes"
        },
        
        InterventionType.NEBULIZER: {
            "masks": [
                VitalMask(
                    vital_name="RR",
                    expected_direction="decrease",
                    mask_duration_minutes=15,
                    threshold_change=5.0  # Expect RR to decrease by ~5 bpm
                ),
                VitalMask(
                    vital_name="SpO2",
                    expected_direction="increase",
                    mask_duration_minutes=15,
                    threshold_change=3.0  # Expect SpO2 improvement
                ),
                VitalMask(
                    vital_name="HR",
                    expected_direction="decrease",
                    mask_duration_minutes=20,
                    threshold_change=10.0  # Less work of breathing → HR down
                )
            ],
            "response_window_minutes": 15,
            "expected_response": "RR should decrease and SpO2 improve within 15 minutes"
        },
        
        InterventionType.DIURETIC: {
            "masks": [
                VitalMask(
                    vital_name="SBP",
                    expected_direction="decrease",
                    mask_duration_minutes=180,  # 3 hours
                    threshold_change=10.0
                ),
                VitalMask(
                    vital_name="RR",
                    expected_direction="decrease",
                    mask_duration_minutes=240,  # 4 hours
                    threshold_change=3.0
                ),
                VitalMask(
                    vital_name="HR",
                    expected_direction="stabilize",
                    mask_duration_minutes=180,
                    threshold_change=5.0
                )
            ],
            "response_window_minutes": 240,
            "expected_response": "BP should decrease gradually over 2-4 hours"
        },
        
        InterventionType.INSULIN: {
            "masks": [
                VitalMask(
                    vital_name="HR",
                    expected_direction="stabilize",
                    mask_duration_minutes=60,
                    threshold_change=5.0
                )
            ],
            "response_window_minutes": 60,
            "expected_response": "Blood glucose should decrease within 30-60 minutes"
        },
        
        InterventionType.FLUIDS: {
            "masks": [
                VitalMask(
                    vital_name="HR",
                    expected_direction="decrease",
                    mask_duration_minutes=30,
                    threshold_change=15.0
                ),
                VitalMask(
                    vital_name="SBP",
                    expected_direction="increase",
                    mask_duration_minutes=30,
                    threshold_change=10.0
                )
            ],
            "response_window_minutes": 30,
            "expected_response": "HR should decrease and BP increase within 30 minutes"
        },
        
        InterventionType.OXYGEN: {
            "masks": [
                VitalMask(
                    vital_name="SpO2",
                    expected_direction="increase",
                    mask_duration_minutes=10,
                    threshold_change=5.0
                ),
                VitalMask(
                    vital_name="RR",
                    expected_direction="decrease",
                    mask_duration_minutes=15,
                    threshold_change=3.0
                )
            ],
            "response_window_minutes": 10,
            "expected_response": "SpO2 should increase within 10 minutes"
        },
        
        InterventionType.BETA_BLOCKER: {
            "masks": [
                VitalMask(
                    vital_name="HR",
                    expected_direction="decrease",
                    mask_duration_minutes=60,
                    threshold_change=15.0
                ),
                VitalMask(
                    vital_name="SBP",
                    expected_direction="decrease",
                    mask_duration_minutes=60,
                    threshold_change=10.0
                )
            ],
            "response_window_minutes": 60,
            "expected_response": "HR and BP should decrease within 60 minutes"
        }
    }
    
    def __init__(self):
        self.intervention_log: Dict[str, List[Intervention]] = {}
        self._intervention_counter = 0
    
    def log_intervention(
        self,
        patient_id: str,
        intervention_type: InterventionType,
        timestamp: Optional[datetime] = None,
        dosage: Optional[str] = None,
        administered_by: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Intervention:
        """
        Log a clinical intervention and setup expected effect masks.
        
        Args:
            patient_id: Patient identifier
            intervention_type: Type of intervention
            timestamp: When intervention was given (default: now)
            dosage: Optional dosage information
            administered_by: Clinician identifier
            notes: Additional notes
        
        Returns:
            Intervention object with active masks
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get intervention profile
        profile = self.INTERVENTION_PROFILES.get(intervention_type)
        if not profile:
            logger.warning(f"Unknown intervention type: {intervention_type}")
            expected_effects = []
        else:
            expected_effects = profile["masks"]
        
        # Create intervention record
        self._intervention_counter += 1
        intervention = Intervention(
            intervention_id=f"INT_{patient_id}_{self._intervention_counter:04d}",
            patient_id=patient_id,
            intervention_type=intervention_type,
            timestamp=timestamp,
            expected_effects=expected_effects,
            dosage=dosage,
            administered_by=administered_by,
            notes=notes
        )
        
        # Add to log
        if patient_id not in self.intervention_log:
            self.intervention_log[patient_id] = []
        self.intervention_log[patient_id].append(intervention)
        
        logger.info(
            f"Intervention logged: {patient_id} - {intervention_type.value} "
            f"at {timestamp.isoformat()}, masking {len(expected_effects)} vitals"
        )
        
        return intervention
    
    def get_active_masks(
        self,
        patient_id: str,
        current_time: Optional[datetime] = None
    ) -> Dict[str, List[VitalMask]]:
        """
        Get currently active vital masks for a patient.
        
        Args:
            patient_id: Patient identifier
            current_time: Current timestamp (default: now)
        
        Returns:
            Dictionary mapping vital names to active masks
        """
        if current_time is None:
            current_time = datetime.now()
        
        if patient_id not in self.intervention_log:
            return {}
        
        active_masks: Dict[str, List[VitalMask]] = {}
        
        for intervention in self.intervention_log[patient_id]:
            if not intervention.is_active:
                continue
            
            # Check if any masks are still within their duration
            time_since_intervention = (current_time - intervention.timestamp).total_seconds() / 60
            
            for mask in intervention.expected_effects:
                if time_since_intervention <= mask.mask_duration_minutes:
                    # Mask is still active
                    if mask.vital_name not in active_masks:
                        active_masks[mask.vital_name] = []
                    active_masks[mask.vital_name].append(mask)
                else:
                    # Mask expired, deactivate intervention if all masks expired
                    all_expired = all(
                        time_since_intervention > m.mask_duration_minutes
                        for m in intervention.expected_effects
                    )
                    if all_expired:
                        intervention.is_active = False
        
        return active_masks
    
    def apply_mask(
        self,
        patient_id: str,
        vital_deviations: Dict[str, float],
        current_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Apply intervention masks to adjust vital deviations.
        
        Expected vital changes are suppressed to prevent false alarms.
        
        Args:
            patient_id: Patient identifier
            vital_deviations: Dictionary of vital names to deviation magnitudes
            current_time: Current timestamp
        
        Returns:
            Adjusted vital deviations with masks applied
        """
        active_masks = self.get_active_masks(patient_id, current_time)
        
        if not active_masks:
            return vital_deviations  # No masking needed
        
        adjusted_deviations = vital_deviations.copy()
        
        for vital_name, deviation in vital_deviations.items():
            if vital_name not in active_masks:
                continue
            
            # Check if deviation is in expected direction
            for mask in active_masks[vital_name]:
                if mask.expected_direction == "increase" and deviation > 0:
                    # Expected increase - reduce alarm sensitivity
                    adjusted_deviations[vital_name] = max(0, deviation - mask.threshold_change)
                    logger.debug(f"Masked {vital_name} increase: {deviation:.2f} → {adjusted_deviations[vital_name]:.2f}")
                
                elif mask.expected_direction == "decrease" and deviation < 0:
                    # Expected decrease - reduce alarm sensitivity
                    adjusted_deviations[vital_name] = min(0, deviation + mask.threshold_change)
                    logger.debug(f"Masked {vital_name} decrease: {deviation:.2f} → {adjusted_deviations[vital_name]:.2f}")
                
                elif mask.expected_direction == "stabilize":
                    # Expect stability - reduce any deviation
                    if abs(deviation) < mask.threshold_change:
                        adjusted_deviations[vital_name] = 0
                        logger.debug(f"Masked {vital_name} stabilization: {deviation:.2f} → 0")
        
        return adjusted_deviations
    
    def check_intervention_effectiveness(
        self,
        patient_id: str,
        current_vitals: Dict[str, float],
        baseline_vitals: Dict[str, Dict[str, float]],
        current_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Check if interventions produced expected effects within response window.
        
        Args:
            patient_id: Patient identifier
            current_vitals: Current vital signs [HR, SpO2, SBP, RR, Temp]
            baseline_vitals: Patient baseline metrics
            current_time: Current timestamp
        
        Returns:
            List of treatment failure alerts
        """
        if current_time is None:
            current_time = datetime.now()
        
        if patient_id not in self.intervention_log:
            return []
        
        alerts = []
        vital_names = ['HR', 'SpO2', 'SBP', 'RR', 'Temp']
        vitals_dict = dict(zip(vital_names, current_vitals))
        
        for intervention in self.intervention_log[patient_id]:
            if intervention.effectiveness_checked:
                continue
            
            # Check if response window has passed
            profile = self.INTERVENTION_PROFILES.get(intervention.intervention_type)
            if not profile:
                continue
            
            time_since = (current_time - intervention.timestamp).total_seconds() / 60
            response_window = profile["response_window_minutes"]
            
            if time_since < response_window:
                continue  # Still within response window
            
            # Response window passed - check effectiveness
            intervention.effectiveness_checked = True
            treatment_effective = True
            failed_vitals = []
            
            for mask in intervention.expected_effects:
                vital_name = mask.vital_name
                if vital_name not in vitals_dict or vital_name not in baseline_vitals:
                    continue
                
                current_value = vitals_dict[vital_name]
                baseline_value = baseline_vitals[vital_name]['mean']
                actual_change = current_value - baseline_value
                
                # Store actual change
                intervention.actual_changes[vital_name] = actual_change
                
                # Check if change is in expected direction
                if mask.expected_direction == "increase":
                    if actual_change < mask.threshold_change * 0.5:  # At least 50% of expected
                        treatment_effective = False
                        failed_vitals.append(f"{vital_name} (expected ↑{mask.threshold_change}, got {actual_change:+.1f})")
                
                elif mask.expected_direction == "decrease":
                    if actual_change > -mask.threshold_change * 0.5:
                        treatment_effective = False
                        failed_vitals.append(f"{vital_name} (expected ↓{mask.threshold_change}, got {actual_change:+.1f})")
            
            intervention.treatment_effective = treatment_effective
            
            if not treatment_effective:
                alert = {
                    "alert_type": "TREATMENT_FAILURE",
                    "patient_id": patient_id,
                    "intervention_id": intervention.intervention_id,
                    "intervention_type": intervention.intervention_type.value,
                    "timestamp": current_time.isoformat(),
                    "message": f"{intervention.intervention_type.value} did not produce expected response",
                    "expected_response": profile["expected_response"],
                    "failed_vitals": failed_vitals,
                    "time_since_intervention": f"{time_since:.1f} minutes"
                }
                alerts.append(alert)
                logger.warning(f"Treatment failure detected: {patient_id} - {intervention.intervention_type.value}")
        
        return alerts
    
    def get_intervention_history(
        self,
        patient_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get intervention history for a patient.
        
        Args:
            patient_id: Patient identifier
            limit: Maximum number of interventions to return
        
        Returns:
            List of intervention records with effectiveness data
        """
        if patient_id not in self.intervention_log:
            return []
        
        interventions = self.intervention_log[patient_id]
        
        if limit:
            interventions = interventions[-limit:]
        
        return [
            {
                "intervention_id": inv.intervention_id,
                "type": inv.intervention_type.value,
                "timestamp": inv.timestamp.isoformat(),
                "dosage": inv.dosage,
                "administered_by": inv.administered_by,
                "notes": inv.notes,
                "is_active": inv.is_active,
                "effectiveness_checked": inv.effectiveness_checked,
                "treatment_effective": inv.treatment_effective,
                "actual_changes": inv.actual_changes,
                "masked_vitals": [m.vital_name for m in inv.expected_effects]
            }
            for inv in interventions
        ]
    
    def clear_patient_interventions(self, patient_id: str):
        """Clear all interventions for a patient (e.g., on discharge)"""
        if patient_id in self.intervention_log:
            del self.intervention_log[patient_id]
            logger.info(f"Cleared interventions for patient {patient_id}")
