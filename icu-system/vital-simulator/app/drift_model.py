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
    wbc: float  # White Blood Cell Count (4-11 K/µL)
    creatinine: float  # Serum Creatinine (0.6-1.2 mg/dL)
    platelets: float  # Platelet Count (150-400 K/µL)
    bilirubin_total: float  # Total Bilirubin (0.3-1.2 mg/dL)
    etco2: float  # End-tidal CO2 (35-45 mmHg)
    age: int  # Patient age (years)
    gender: int  # Gender (0=Female, 1=Male)

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
            lactate=random.uniform(0.5, 1.2),
            wbc=random.uniform(4.5, 10.5),  # Normal range
            creatinine=random.uniform(0.7, 1.1),  # Normal range
            platelets=random.uniform(180, 350),  # Normal range
            bilirubin_total=random.uniform(0.4, 1.0),  # Normal range
            etco2=random.uniform(36, 42),  # Normal range
            age=random.randint(45, 85),  # ICU typical age range
            gender=random.randint(0, 1)  # 0=Female, 1=Male
        )
        
        # Drift rates (change per minute)
        self.hr_drift_rate = 0.0
        self.sbp_drift_rate = 0.0
        self.lactate_drift_rate = 0.0
        self.wbc_drift_rate = 0.0
        self.creatinine_drift_rate = 0.0
        
        # Current values (start at baseline)
        self.current_hr = self.baseline.heart_rate
        self.current_sbp = self.baseline.systolic_bp
        self.current_dbp = self.baseline.diastolic_bp
        self.current_spo2 = self.baseline.spo2
        self.current_rr = self.baseline.respiratory_rate
        self.current_temp = self.baseline.temperature
        self.current_lactate = self.baseline.lactate
        self.current_wbc = self.baseline.wbc
        self.current_creatinine = self.baseline.creatinine
        self.current_platelets = self.baseline.platelets
        self.current_bilirubin = self.baseline.bilirubin_total
        self.current_etco2 = self.baseline.etco2
        
        # Stress episode state
        self.is_under_stress = False
        self.stress_duration = 0
        self.stress_severity = 0.0
        
        # Mean reversion strength (pulls back to baseline)
        self.reversion_strength = 0.05
        
    def trigger_stress_episode(self):
        """
        Probabilistic stress episode trigger (30% chance per call)
        Causes rapid deterioration to demonstrate LSTM predictions
        """
        if not self.is_under_stress and random.random() < 0.30:  # Increased from 5% to 30%
            self.is_under_stress = True
            self.stress_severity = random.uniform(0.7, 1.0)  # High severity for visible changes
            self.stress_duration = random.randint(180, 600)  # 3-10 minutes
            
            # Set MUCH stronger drift rates for rapid deterioration (10x multiplier)
            self.hr_drift_rate = self.stress_severity * random.uniform(3.0, 8.0)  # HR increases rapidly
            self.sbp_drift_rate = -self.stress_severity * random.uniform(2.0, 5.0)  # BP drops rapidly
            self.lactate_drift_rate = self.stress_severity * random.uniform(0.3, 0.8)  # Lactate spikes quickly
            self.wbc_drift_rate = self.stress_severity * random.uniform(0.8, 2.0)  # WBC rises with infection
            self.creatinine_drift_rate = self.stress_severity * random.uniform(0.08, 0.25)  # Kidney stress visible
            
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
                self.wbc_drift_rate *= 0.4
                self.creatinine_drift_rate *= 0.4
        
        # Apply drift with mean reversion
        hr_reversion = (self.baseline.heart_rate - self.current_hr) * self.reversion_strength
        sbp_reversion = (self.baseline.systolic_bp - self.current_sbp) * self.reversion_strength
        
        self.current_hr += (self.hr_drift_rate + hr_reversion) * dt_minutes
        self.current_sbp += (self.sbp_drift_rate + sbp_reversion) * dt_minutes
        self.current_lactate += self.lactate_drift_rate * dt_minutes
        self.current_wbc += self.wbc_drift_rate * dt_minutes
        self.current_creatinine += self.creatinine_drift_rate * dt_minutes
        
        # Gradual decay of drift rates (return to baseline)
        if not self.is_under_stress:
            self.hr_drift_rate *= 0.98
            self.sbp_drift_rate *= 0.98
            self.lactate_drift_rate *= 0.95
            self.wbc_drift_rate *= 0.96
            self.creatinine_drift_rate *= 0.96
        
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
        
        # Lab values (minimal noise for realistic stability)
        wbc = self.current_wbc + np.random.normal(0, 0.3)
        creatinine = self.current_creatinine + np.random.normal(0, 0.05)
        platelets = self.current_platelets + np.random.normal(0, 5.0)
        bilirubin = self.current_bilirubin + np.random.normal(0, 0.05)
        etco2 = self.current_etco2 + np.random.normal(0, 1.5)
        
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
        wbc = np.clip(wbc, 2.0, 25.0)  # 2-25 K/µL
        creatinine = np.clip(creatinine, 0.3, 5.0)  # 0.3-5.0 mg/dL
        platelets = np.clip(platelets, 50, 500)  # 50-500 K/µL
        bilirubin = np.clip(bilirubin, 0.2, 8.0)  # 0.2-8.0 mg/dL
        etco2 = np.clip(etco2, 20, 60)  # 20-60 mmHg
        
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
            'wbc': round(wbc, 2),
            'creatinine': round(creatinine, 2),
            'platelets': round(platelets, 1),
            'bilirubin_total': round(bilirubin, 2),
            'etco2': round(etco2, 1),
            'age': self.baseline.age,
            'gender': self.baseline.gender,
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
    
    # ==================== DEVELOPER TOOLS: Manual Scenario Triggers ====================
    
    def trigger_sepsis_scenario(self, severity: float = 0.8, duration_seconds: int = 300):
        """
        Manually trigger sepsis scenario: high HR, low BP, high lactate, high WBC
        severity: 0.0-1.0 (mild to severe)
        duration_seconds: how long the episode lasts
        """
        self.is_under_stress = True
        self.stress_severity = max(0.5, min(1.0, severity))
        self.stress_duration = duration_seconds
        
        # Sepsis pattern: tachycardia, hypotension, elevated lactate & WBC
        self.hr_drift_rate = self.stress_severity * random.uniform(4.0, 10.0)  # Rapid HR increase
        self.sbp_drift_rate = -self.stress_severity * random.uniform(3.0, 6.0)  # Significant BP drop
        self.lactate_drift_rate = self.stress_severity * random.uniform(0.4, 1.0)  # High lactate buildup
        self.wbc_drift_rate = self.stress_severity * random.uniform(1.0, 2.5)  # Infection marker
        self.creatinine_drift_rate = self.stress_severity * random.uniform(0.1, 0.3)  # Kidney dysfunction
    
    def trigger_shock_scenario(self, severity: float = 0.9, duration_seconds: int = 240):
        """
        Manually trigger shock scenario: very high HR, very low BP, high lactate
        More severe than sepsis - critical condition
        """
        self.is_under_stress = True
        self.stress_severity = max(0.7, min(1.0, severity))
        self.stress_duration = duration_seconds
        
        # Shock pattern: extreme tachycardia, severe hypotension
        self.hr_drift_rate = self.stress_severity * random.uniform(8.0, 15.0)  # Very rapid HR
        self.sbp_drift_rate = -self.stress_severity * random.uniform(5.0, 10.0)  # Severe BP drop
        self.lactate_drift_rate = self.stress_severity * random.uniform(0.6, 1.5)  # Very high lactate
        self.wbc_drift_rate = self.stress_severity * random.uniform(1.5, 3.0)  # High WBC
        self.creatinine_drift_rate = self.stress_severity * random.uniform(0.15, 0.4)  # Kidney failure
    
    def trigger_mild_deterioration(self, duration_seconds: int = 180):
        """
        Mild deterioration - subtle changes for early warning demonstration
        """
        self.is_under_stress = True
        self.stress_severity = 0.3
        self.stress_duration = duration_seconds
        
        self.hr_drift_rate = random.uniform(1.0, 2.5)
        self.sbp_drift_rate = -random.uniform(0.8, 1.5)
        self.lactate_drift_rate = random.uniform(0.1, 0.25)
        self.wbc_drift_rate = random.uniform(0.3, 0.6)
        self.creatinine_drift_rate = random.uniform(0.02, 0.05)
    
    def trigger_rapid_recovery(self, recovery_speed: float = 2.0):
        """
        Trigger rapid recovery - returns to baseline quickly
        recovery_speed: multiplier for how fast (1.0 = normal, 2.0 = 2x faster)
        """
        self.is_under_stress = False
        
        # Apply stronger reversion to pull back to baseline faster
        self.reversion_strength = 0.05 * recovery_speed
        
        # Reduce drift rates more aggressively
        self.hr_drift_rate *= 0.1
        self.sbp_drift_rate *= 0.1
        self.lactate_drift_rate *= 0.1
        self.wbc_drift_rate *= 0.1
        self.creatinine_drift_rate *= 0.1
    
    def reset_to_baseline(self):
        """
        Immediately reset patient to healthy baseline - for demo purposes
        """
        self.is_under_stress = False
        self.stress_severity = 0.0
        self.stress_duration = 0
        
        # Reset current values to baseline
        self.current_hr = self.baseline.heart_rate
        self.current_sbp = self.baseline.systolic_bp
        self.current_dbp = self.baseline.diastolic_bp
        self.current_spo2 = self.baseline.spo2
        self.current_rr = self.baseline.respiratory_rate
        self.current_temp = self.baseline.temperature
        self.current_lactate = self.baseline.lactate
        self.current_wbc = self.baseline.wbc
        self.current_creatinine = self.baseline.creatinine
        
        # Clear drift rates
        self.hr_drift_rate = 0.0
        self.sbp_drift_rate = 0.0
        self.lactate_drift_rate = 0.0
        self.wbc_drift_rate = 0.0
        self.creatinine_drift_rate = 0.0
        
        # Reset reversion strength
        self.reversion_strength = 0.05
    
    def set_critical_condition(self):
        """
        Set patient to critical septic shock condition immediately
        For demonstration of high-risk detection
        """
        self.is_under_stress = True
        self.stress_severity = 1.0
        self.stress_duration = 600  # 10 minutes
        
        # Set extreme values
        self.current_hr = random.uniform(130, 150)
        self.current_sbp = random.uniform(70, 85)
        self.current_lactate = random.uniform(4.0, 7.0)
        self.current_wbc = random.uniform(16.0, 22.0)
        self.current_creatinine = random.uniform(2.5, 4.0)
        self.current_spo2 = random.uniform(88, 92)
        
        # High drift rates to maintain critical state
        self.hr_drift_rate = random.uniform(2.0, 5.0)
        self.sbp_drift_rate = -random.uniform(1.0, 2.0)
        self.lactate_drift_rate = random.uniform(0.2, 0.4)
        self.wbc_drift_rate = random.uniform(0.5, 1.0)
        self.creatinine_drift_rate = random.uniform(0.05, 0.15)
