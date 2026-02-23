"""
Physiological Drift Model - Realistic Vital Sign Simulation
No state machines, no spikes - just gradual physiological changes
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class PatientBaseline:
    """Individual patient baseline parameters"""
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    spo2: float
    respiratory_rate: float
    temperature: float
    lactate: float

class PhysiologicalDriftModel:
    """
    Models realistic physiological changes using mean-reversion drift
    with Gaussian noise and probabilistic stress episodes
    """
    
    def __init__(self, patient_id: str, seed: int = None):
        self.patient_id = patient_id
        
        # Set seed for reproducibility if provided
        if seed:
            random.seed(seed + hash(patient_id))
            np.random.seed(seed + hash(patient_id))
        
        # Initialize realistic baselines
        self.baseline = PatientBaseline(
            heart_rate=random.uniform(65, 85),
            systolic_bp=random.uniform(110, 130),
            diastolic_bp=random.uniform(60, 75),
            spo2=random.uniform(96, 99),
            respiratory_rate=random.uniform(12, 18),
            temperature=random.uniform(36.3, 37.1),
            lactate=random.uniform(0.5, 1.2)
        )
        
        # Drift rates (change per minute)
        self.hr_drift_rate = 0.0
        self.sbp_drift_rate = 0.0
        self.lactate_drift_rate = 0.0
        
        # Current values (start at baseline)
        self.current_hr = self.baseline.heart_rate
        self.current_sbp = self.baseline.systolic_bp
        self.current_dbp = self.baseline.diastolic_bp
        self.current_spo2 = self.baseline.spo2
        self.current_rr = self.baseline.respiratory_rate
        self.current_temp = self.baseline.temperature
        self.current_lactate = self.baseline.lactate
        
        # Stress episode state
        self.is_under_stress = False
        self.stress_duration = 0
        self.stress_severity = 0.0
        
        # Mean reversion strength (pulls back to baseline)
        self.reversion_strength = 0.05
        
    def trigger_stress_episode(self):
        """
        Probabilistic stress episode trigger (5% chance per call)
        Causes gradual deterioration over time
        """
        if not self.is_under_stress and random.random() < 0.05:
            self.is_under_stress = True
            self.stress_severity = random.uniform(0.3, 0.8)
            self.stress_duration = random.randint(300, 1200)  # 5-20 minutes
            
            # Set drift rates based on stress severity
            self.hr_drift_rate = self.stress_severity * random.uniform(0.15, 0.45)
            self.sbp_drift_rate = -self.stress_severity * random.uniform(0.08, 0.25)
            self.lactate_drift_rate = self.stress_severity * random.uniform(0.015, 0.04)
            
    def update(self, dt: float = 1.0) -> Dict:
        """
        Update vital signs using drift model
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Dictionary of current vital signs
        """
        dt_minutes = dt / 60.0
        
        # Check for stress episode trigger
        self.trigger_stress_episode()
        
        # Handle ongoing stress episode
        if self.is_under_stress:
            self.stress_duration -= dt
            if self.stress_duration <= 0:
                # Stress episode ending - begin recovery
                self.is_under_stress = False
                self.hr_drift_rate *= 0.5
                self.sbp_drift_rate *= 0.5
                self.lactate_drift_rate *= 0.3
        
        # Apply drift with mean reversion
        hr_reversion = (self.baseline.heart_rate - self.current_hr) * self.reversion_strength
        sbp_reversion = (self.baseline.systolic_bp - self.current_sbp) * self.reversion_strength
        
        self.current_hr += (self.hr_drift_rate + hr_reversion) * dt_minutes
        self.current_sbp += (self.sbp_drift_rate + sbp_reversion) * dt_minutes
        self.current_lactate += self.lactate_drift_rate * dt_minutes
        
        # Gradual decay of drift rates (return to baseline)
        if not self.is_under_stress:
            self.hr_drift_rate *= 0.98
            self.sbp_drift_rate *= 0.98
            self.lactate_drift_rate *= 0.95
        
        # Add Gaussian noise for realistic variability
        noise_hr = np.random.normal(0, 1.8)
        noise_sbp = np.random.normal(0, 2.5)
        noise_dbp = np.random.normal(0, 1.5)
        noise_spo2 = np.random.normal(0, 0.6)
        noise_rr = np.random.normal(0, 1.0)
        noise_temp = np.random.normal(0, 0.12)
        
        # Apply noise to current values
        hr = self.current_hr + noise_hr
        sbp = self.current_sbp + noise_sbp
        dbp = self.current_dbp + noise_dbp
        spo2 = self.current_spo2 + noise_spo2
        rr = self.current_rr + noise_rr
        temp = self.current_temp + noise_temp
        lactate = self.current_lactate
        
        # SpO2 decreases with shock
        shock_index = hr / max(sbp, 1.0)
        if shock_index > 1.0:
            spo2 -= (shock_index - 1.0) * 2.0
        
        # Calculate derived values
        map_value = dbp + (sbp - dbp) / 3.0
        
        # Enforce clinical bounds
        hr = np.clip(hr, 50, 160)
        sbp = np.clip(sbp, 70, 180)
        dbp = np.clip(dbp, 40, 110)
        spo2 = np.clip(spo2, 80, 100)
        rr = np.clip(rr, 8, 35)
        temp = np.clip(temp, 35.0, 40.0)
        lactate = np.clip(lactate, 0.5, 8.0)
        map_value = np.clip(map_value, 50, 150)
        
        # Recalculate shock index
        shock_index = hr / max(sbp, 1.0)
        
        return {
            'patient_id': self.patient_id,
            'timestamp': None,  # Set by caller
            'heart_rate': round(hr, 1),
            'systolic_bp': round(sbp, 1),
            'diastolic_bp': round(dbp, 1),
            'map': round(map_value, 1),
            'spo2': round(spo2, 1),
            'respiratory_rate': round(rr, 1),
            'temperature': round(temp, 2),
            'lactate': round(lactate, 2),
            'shock_index': round(shock_index, 2)
        }
    
    def get_status(self) -> Dict:
        """Get current physiological status"""
        return {
            'patient_id': self.patient_id,
            'is_under_stress': self.is_under_stress,
            'stress_severity': round(self.stress_severity, 2) if self.is_under_stress else 0.0,
            'hr_drift_rate': round(self.hr_drift_rate, 3),
            'sbp_drift_rate': round(self.sbp_drift_rate, 3),
            'current_shock_index': round(self.current_hr / max(self.current_sbp, 1.0), 2)
        }
