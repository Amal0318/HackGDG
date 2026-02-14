"""
Phase 3.2: Intervention Endpoints
Add these to backend-api/app/main.py
"""

# Add this Pydantic model near the top with other models
class InterventionRequest(BaseModel):
    """Request to log a clinical intervention"""
    type: str = Field(..., description="Intervention type (e.g., vasopressors, nebulizer)")
    dosage: Optional[str] = Field(None, description="Dosage information")
    administered_by: Optional[str] = Field(None, description="Clinician identifier")
    notes: Optional[str] = Field(None, description="Additional notes")
    timestamp: Optional[datetime] = Field(None, description="When intervention was given (default: now)")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "vasopressors",
                "dosage": "5mcg/min norepinephrine",
                "administered_by": "DR_SMITH",
                "notes": "Started for septic shock",
                "timestamp": "2026-02-14T10:30:00"
            }
        }


# Add these endpoints after the discharge_patient endpoint

@app.post("/patients/{patient_id}/interventions")
async def log_intervention(patient_id: str, intervention: InterventionRequest):
    """
    Log a clinical intervention for a patient.
    
    This activates intervention masks to prevent false alarms from expected
    physiological responses.
    
    Args:
        patient_id: Patient identifier
        intervention: Intervention details
    
    Returns:
        Intervention record with active masks
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found. Admit patient first."
        )
    
    # Validate intervention type
    try:
        intervention_type = InterventionType(intervention.type.lower())
    except ValueError:
        valid_types = [t.value for t in InterventionType]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid intervention type. Valid types: {valid_types}"
        )
    
    # Log intervention
    intervention_record = state.intervention_tracker.log_intervention(
        patient_id=patient_id,
        intervention_type=intervention_type,
        timestamp=intervention.timestamp,
        dosage=intervention.dosage,
        administered_by=intervention.administered_by,
        notes=intervention.notes
    )
    
    logger.info(f"Intervention logged for {patient_id}: {intervention_type.value}")
    
    return {
        "message": "Intervention logged successfully",
        "intervention_id": intervention_record.intervention_id,
        "patient_id": patient_id,
        "type": intervention_record.intervention_type.value,
        "timestamp": intervention_record.timestamp.isoformat(),
        "active_masks": [
            {
                "vital": mask.vital_name,
                "expected_direction": mask.expected_direction,
                "duration_minutes": mask.mask_duration_minutes,
                "threshold_change": mask.threshold_change
            }
            for mask in intervention_record.expected_effects
        ],
        "response_window_minutes": state.intervention_tracker.INTERVENTION_PROFILES[intervention_type]["response_window_minutes"],
        "expected_response": state.intervention_tracker.INTERVENTION_PROFILES[intervention_type]["expected_response"]
    }


@app.get("/patients/{patient_id}/interventions")
async def get_interventions(patient_id: str, limit: Optional[int] = 10):
    """
    Get intervention history for a patient.
    
    Args:
        patient_id: Patient identifier
        limit: Maximum number of interventions to return
    
    Returns:
        List of interventions with effectiveness data
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    interventions = state.intervention_tracker.get_intervention_history(
        patient_id=patient_id,
        limit=limit
    )
    
    return {
        "patient_id": patient_id,
        "intervention_count": len(interventions),
        "interventions": interventions
    }


@app.get("/patients/{patient_id}/interventions/active")
async def get_active_masks(patient_id: str):
    """
    Get currently active intervention masks for a patient.
    
    Shows which vitals are currently masked due to recent interventions.
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        Active intervention masks
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    active_masks = state.intervention_tracker.get_active_masks(patient_id)
    
    return {
        "patient_id": patient_id,
        "has_active_masks": len(active_masks) > 0,
        "masked_vitals": list(active_masks.keys()),
        "active_masks": {
            vital: [
                {
                    "expected_direction": mask.expected_direction,
                    "duration_minutes": mask.mask_duration_minutes,
                    "threshold_change": mask.threshold_change
                }
                for mask in masks
            ]
            for vital, masks in active_masks.items()
        }
    }


@app.get("/patients/{patient_id}/alerts")
async def get_patient_alerts(
    patient_id: str,
    include_suppressed: bool = False,
    limit: int = 10
):
    """
    Get recent alerts for a patient.
    
    Args:
        patient_id: Patient identifier
        include_suppressed: Include suppressed alerts
        limit: Maximum number of alerts to return
    
    Returns:
        List of recent alerts
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    alerts = state.alert_manager.get_recent_alerts(
        patient_id=patient_id,
        include_suppressed=include_suppressed,
        limit=limit
    )
    
    stats = state.alert_manager.get_alert_statistics(patient_id)
    
    return {
        "patient_id": patient_id,
        "alert_count": len(alerts),
        "alerts": alerts,
        "statistics": stats
    }


@app.post("/patients/{patient_id}/alerts/{alert_id}/outcome")
async def record_alert_outcome(
    patient_id: str,
    alert_id: str,
    was_true_positive: bool
):
    """
    Record the outcome of an alert (true or false positive).
    
    Used to track alert accuracy and improve suppression logic.
    
    Args:
        patient_id: Patient identifier
        alert_id: Alert identifier
        was_true_positive: Whether alert correctly predicted deterioration
    
    Returns:
        Confirmation
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    state.alert_manager.mark_alert_outcome(
        patient_id=patient_id,
        alert_id=alert_id,
        was_true_positive=was_true_positive
    )
    
    return {
        "message": "Alert outcome recorded",
        "alert_id": alert_id,
        "outcome": "true_positive" if was_true_positive else "false_positive"
    }
