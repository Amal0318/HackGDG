"""
Vital Simulator Service - ICU Telemetry Data Generator
Simulates realistic physiological data for 8 ICU patients with state transitions
"""

import asyncio
import logging
import os
import json
import random
import time
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    # Fallback for development without Kafka
    logging.warning("Kafka library not available - using mock producer")
    KAFKA_AVAILABLE = False
    class KafkaProducer:
        def __init__(self, *args, **kwargs):
            pass
        def send(self, topic, value):
            pass
        def flush(self):
            pass
        def close(self):
            pass

# Debug mode configuration
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vital-simulator")

class PatientState(Enum):
    """Patient physiological states"""
    STABLE = "STABLE"
    EARLY_DETERIORATION = "EARLY_DETERIORATION"
    LATE_DETERIORATION = "LATE_DETERIORATION"
    CRITICAL = "CRITICAL"
    INTERVENTION = "INTERVENTION"
    RECOVERING = "RECOVERING"

class AcuteEventType(Enum):
    """Acute medical events that can override baseline vitals"""
    NONE = "NONE"
    SEPSIS_SPIKE = "SEPSIS_SPIKE"
    HYPOXIA_EVENT = "HYPOXIA_EVENT"
    HYPOTENSION_DROP = "HYPOTENSION_DROP"
    MEDICATION_RESPONSE = "MEDICATION_RESPONSE"
    RAPID_RECOVERY = "RAPID_RECOVERY"

@dataclass
class VitalSigns:
    """Container for patient vital signs"""
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    spo2: float
    respiratory_rate: float
    temperature: float
    
    @property
    def shock_index(self) -> float:
        """Calculated shock index (HR/SBP)"""
        return self.heart_rate / max(self.systolic_bp, 1.0)

class Patient:
    """ICU Patient simulator with hospital-grade clinical realism"""
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.state = PatientState.STABLE
        self.state_duration = 0
        
        # Individual patient baselines for clinical realism
        self.baseline_hr = random.uniform(65, 85)
        self.baseline_sbp = random.uniform(110, 130)
        self.baseline_spo2 = random.uniform(96, 99)
        self.baseline_rr = random.uniform(12, 18)
        self.baseline_temp = random.uniform(36.2, 37.2)
        
        self.last_vitals = self._initialize_stable_vitals()
        
        # Acute Event Engine with cooldown
        self.active_acute_event = None
        self.acute_event_duration = 0
        self.baseline_vitals = None
        self.acute_event_cooldown = 0  # 2 minute cooldown
        
        # Shock index monitoring (5 consecutive seconds rule)
        self.shock_index_high_count = 0  # Count of consecutive high SI > 1.3
        self.shock_index_normal_count = 0  # Count of consecutive normal SI < 1.0
        
        # Clinical realism tracking for validation
        self.critical_state_time = 0
        self.max_shock_index_recorded = 0.0
        self.acute_event_count = 0
        
        # State transition (minimal duration before transitions)
        self.min_state_duration = 30

        # Scripted gradual deterioration spike state (controlled by simulator scheduler)
        self.scripted_spike_active = False
        self.scripted_spike_phase = "idle"
        self.scripted_spike_timer = 0
        self.scripted_spike_ramp_up = 0
        self.scripted_spike_hold = 0
        self.scripted_spike_ramp_down = 0
        self.scripted_spike_deltas = {
            "heart_rate": 0.0,
            "systolic_bp": 0.0,
            "spo2": 0.0,
            "respiratory_rate": 0.0,
            "temperature": 0.0,
        }
        
        logger.info(f"Initialized patient {patient_id} with baseline HR:{self.baseline_hr:.1f}, SBP:{self.baseline_sbp:.1f}")

    def has_scripted_spike_active(self) -> bool:
        """Return whether the patient is currently in a scripted spike episode."""
        return self.scripted_spike_active

    def start_scripted_spike(
        self,
        ramp_up_seconds: int,
        hold_seconds: int,
        ramp_down_seconds: int,
        deltas: Dict[str, float]
    ) -> bool:
        """Start a gradual, deterministic spike profile for this patient."""
        if self.scripted_spike_active:
            return False

        self.scripted_spike_active = True
        self.scripted_spike_phase = "ramp_up"
        self.scripted_spike_timer = 0
        self.scripted_spike_ramp_up = max(1, ramp_up_seconds)
        self.scripted_spike_hold = max(1, hold_seconds)
        self.scripted_spike_ramp_down = max(1, ramp_down_seconds)
        self.scripted_spike_deltas = {
            "heart_rate": float(deltas.get("heart_rate", 0.0)),
            "systolic_bp": float(deltas.get("systolic_bp", 0.0)),
            "spo2": float(deltas.get("spo2", 0.0)),
            "respiratory_rate": float(deltas.get("respiratory_rate", 0.0)),
            "temperature": float(deltas.get("temperature", 0.0)),
        }

        # Nudge trajectory so the ML pipeline sees a meaningful trend quickly
        if self.state == PatientState.STABLE:
            self.state = PatientState.EARLY_DETERIORATION
            self.state_duration = 0

        logger.warning(
            f"Patient {self.patient_id}: SCRIPTED SPIKE STARTED "
            f"(ramp_up={self.scripted_spike_ramp_up}s, hold={self.scripted_spike_hold}s, "
            f"ramp_down={self.scripted_spike_ramp_down}s, deltas={self.scripted_spike_deltas})"
        )
        return True

    def _scripted_spike_intensity(self) -> float:
        """Current spike intensity in range [0, 1], evolves gradually by phase."""
        if not self.scripted_spike_active:
            return 0.0

        if self.scripted_spike_phase == "ramp_up":
            return min(1.0, self.scripted_spike_timer / max(1, self.scripted_spike_ramp_up))
        if self.scripted_spike_phase == "hold":
            return 1.0
        if self.scripted_spike_phase == "ramp_down":
            remaining = max(0, self.scripted_spike_ramp_down - self.scripted_spike_timer)
            return max(0.0, remaining / max(1, self.scripted_spike_ramp_down))
        return 0.0

    def _tick_scripted_spike(self) -> None:
        """Advance scripted spike phase machine by one second."""
        if not self.scripted_spike_active:
            return

        self.scripted_spike_timer += 1

        if self.scripted_spike_phase == "ramp_up" and self.scripted_spike_timer >= self.scripted_spike_ramp_up:
            self.scripted_spike_phase = "hold"
            self.scripted_spike_timer = 0
            return

        if self.scripted_spike_phase == "hold" and self.scripted_spike_timer >= self.scripted_spike_hold:
            self.scripted_spike_phase = "ramp_down"
            self.scripted_spike_timer = 0
            return

        if self.scripted_spike_phase == "ramp_down" and self.scripted_spike_timer >= self.scripted_spike_ramp_down:
            self.scripted_spike_active = False
            self.scripted_spike_phase = "idle"
            self.scripted_spike_timer = 0
            logger.info(f"Patient {self.patient_id}: SCRIPTED SPIKE COMPLETED")

    def _apply_scripted_spike_modifiers(self, vitals: VitalSigns) -> VitalSigns:
        """Apply gradual spike deltas to produce a trend-like deterioration curve."""
        intensity = self._scripted_spike_intensity()
        if intensity <= 0:
            return vitals

        return VitalSigns(
            heart_rate=vitals.heart_rate + (self.scripted_spike_deltas["heart_rate"] * intensity),
            systolic_bp=vitals.systolic_bp - (self.scripted_spike_deltas["systolic_bp"] * intensity),
            diastolic_bp=vitals.diastolic_bp,
            spo2=vitals.spo2 - (self.scripted_spike_deltas["spo2"] * intensity),
            respiratory_rate=vitals.respiratory_rate + (self.scripted_spike_deltas["respiratory_rate"] * intensity),
            temperature=vitals.temperature + (self.scripted_spike_deltas["temperature"] * intensity)
        )
    
    def _initialize_stable_vitals(self) -> VitalSigns:
        """Initialize patient with individual baseline vitals"""
        return VitalSigns(
            heart_rate=self.baseline_hr + random.gauss(0, 2),
            systolic_bp=self.baseline_sbp + random.gauss(0, 3), 
            diastolic_bp=self.baseline_sbp - random.uniform(30, 50),  # Realistic pulse pressure
            spo2=self.baseline_spo2 + random.gauss(0, 0.5),
            respiratory_rate=self.baseline_rr + random.gauss(0, 1),
            temperature=self.baseline_temp + random.gauss(0, 0.2)
        )
    
    def _apply_gaussian_noise(self, value: float, noise_std: float) -> float:
        """Apply Gaussian noise for smooth physiological evolution (hospital-grade)"""
        return value + random.gauss(0, noise_std)
    
    def _stability_bias_mechanism(self, current: float, baseline: float, noise_std: float) -> float:
        """Clinical stability bias: restoring force toward individual baseline"""
        # PART 2 REQUIREMENT: value = prev_value + (baseline_value - prev_value) * 0.05 + gaussian_noise
        restoring_force = (baseline - current) * 0.05
        noise = random.gauss(0, noise_std)
        return current + restoring_force + noise
    
    def _clamp_vitals(self, vitals: VitalSigns) -> VitalSigns:
        """PART 1: Enforce absolute hard physiological caps"""
        # PART 1 ABSOLUTE PHYSIOLOGICAL CAPS
        hr = max(40, min(180, vitals.heart_rate))      # 40-180 bpm
        sbp = max(70, min(180, vitals.systolic_bp))    # 70-180 mmHg
        spo2 = max(85, min(100, vitals.spo2))         # 85-100% (unless acute hypoxia)
        rr = max(10, min(38, vitals.respiratory_rate)) # 10-38 breaths/min
        temp = max(35.5, min(40.5, vitals.temperature))# 35.5-40.5°C
        
        # Diastolic BP: 40-110 mmHg with realistic pulse pressure
        pulse_pressure = random.uniform(35, 55)
        dbp = max(40, min(110, sbp - pulse_pressure))
        
        # Ensure DBP < SBP always
        if dbp >= sbp:
            dbp = sbp - 25
        
        # PART 5: Shock Index max cap 2.0 (NEVER allow > 2.0)
        calculated_hr = hr
        shock_index = hr / max(sbp, 1.0)
        if shock_index > 2.0:
            # Adjust HR to maintain shock index <= 2.0
            calculated_hr = min(hr, sbp * 2.0)
        
        # PART 4: Multi-organ collapse prevention
        # Rule 1: If HR > 160, don't allow SBP < 70 unless acute shock event
        if calculated_hr > 160 and sbp < 70:
            if not (self.active_acute_event == AcuteEventType.HYPOTENSION_DROP):
                sbp = 70  # Prevent simultaneous extreme values
        
        # Rule 2: If SpO2 < 88, don't allow RR > 35 for prolonged duration
        if spo2 < 88 and rr > 35:
            if not (self.active_acute_event == AcuteEventType.HYPOXIA_EVENT):
                rr = 35  # Cap RR during hypoxemia
        
        return VitalSigns(
            heart_rate=calculated_hr,
            systolic_bp=sbp,
            diastolic_bp=dbp,
            spo2=spo2,
            respiratory_rate=rr,
            temperature=temp
        )
    
    def _apply_physiological_correlations(self, vitals: VitalSigns, state: PatientState) -> VitalSigns:
        """Apply physiological correlations based on patient state"""
        if state == PatientState.EARLY_DETERIORATION:
            # HR increases should correlate with SBP decreases
            if vitals.heart_rate > 90:  # If HR is elevated
                # Reduce SBP slightly more
                vitals = VitalSigns(
                    heart_rate=vitals.heart_rate,
                    systolic_bp=vitals.systolic_bp * 0.995,  # Extra 0.5% reduction
                    diastolic_bp=vitals.diastolic_bp,
                    spo2=vitals.spo2 * 0.998 if vitals.respiratory_rate > 18 else vitals.spo2,  # RR/SpO2 correlation
                    respiratory_rate=vitals.respiratory_rate,
                    temperature=vitals.temperature
                )
        
        elif state == PatientState.LATE_DETERIORATION:
            # Stronger correlations in late deterioration
            if vitals.systolic_bp < 100:  # Low BP triggers compensatory tachycardia
                vitals = VitalSigns(
                    heart_rate=vitals.heart_rate * 1.02,  # 2% increase
                    systolic_bp=vitals.systolic_bp,
                    diastolic_bp=vitals.diastolic_bp,
                    spo2=vitals.spo2 * 0.995,  # Hypotension affects oxygenation
                    respiratory_rate=vitals.respiratory_rate * 1.01,  # Compensatory tachypnea
                    temperature=vitals.temperature
                )
        
        return vitals
    
    def maybe_trigger_acute_event(self) -> None:
        """PART 6: Rare acute events with 2-minute cooldown (0.03% per second)"""
        # Don't trigger new events if one is already active OR in cooldown
        if self.active_acute_event is not None or self.acute_event_cooldown > 0:
            return
        
        # PART 6: 0.03% per second = 0.0003 probability
        if random.random() < 0.0003:
            # Select event type
            event_types = [
                AcuteEventType.SEPSIS_SPIKE,
                AcuteEventType.HYPOXIA_EVENT,
                AcuteEventType.HYPOTENSION_DROP,
            ]
            
            # Only medication response if critical
            if self.state == PatientState.CRITICAL:
                event_types.append(AcuteEventType.MEDICATION_RESPONSE)
            
            selected_event = random.choice(event_types)
            self._trigger_acute_event(selected_event)
    
    def _trigger_acute_event(self, event_type: AcuteEventType) -> None:
        """PART 6: Trigger acute event with 10-30 second duration"""
        self.active_acute_event = event_type
        self.baseline_vitals = VitalSigns(
            heart_rate=self.last_vitals.heart_rate,
            systolic_bp=self.last_vitals.systolic_bp,
            diastolic_bp=self.last_vitals.diastolic_bp,
            spo2=self.last_vitals.spo2,
            respiratory_rate=self.last_vitals.respiratory_rate,
            temperature=self.last_vitals.temperature
        )
        
        # PART 6: Duration 10-30 seconds
        self.acute_event_duration = random.randint(10, 30)
        self.acute_event_count += 1
        
        logger.warning(f"Patient {self.patient_id}: ACUTE EVENT #{self.acute_event_count} → {event_type.value} (Duration: {self.acute_event_duration}s)")
    
    def apply_acute_event_modifiers(self, vitals: VitalSigns) -> VitalSigns:
        """Apply acute event modifications to baseline vitals"""
        if self.active_acute_event is None:
            return vitals
        
        # Apply event-specific modifications
        modified_vitals = VitalSigns(
            heart_rate=vitals.heart_rate,
            systolic_bp=vitals.systolic_bp,
            diastolic_bp=vitals.diastolic_bp,
            spo2=vitals.spo2,
            respiratory_rate=vitals.respiratory_rate,
            temperature=vitals.temperature
        )
        
        event = self.active_acute_event
        
        if event == AcuteEventType.SEPSIS_SPIKE:
            # Sepsis: High HR, low BP, low SpO2, high RR, fever
            modified_vitals.heart_rate += random.uniform(30, 50)
            modified_vitals.systolic_bp -= random.uniform(20, 40)
            modified_vitals.spo2 -= random.uniform(5, 10)
            modified_vitals.respiratory_rate += random.uniform(8, 15)
            modified_vitals.temperature += 1.5
            
        elif event == AcuteEventType.HYPOXIA_EVENT:
            # Hypoxia: Low SpO2, compensatory tachypnea and tachycardia
            modified_vitals.spo2 = random.uniform(80, 88)
            modified_vitals.respiratory_rate += random.uniform(5, 12)
            modified_vitals.heart_rate += random.uniform(10, 20)
            
        elif event == AcuteEventType.HYPOTENSION_DROP:
            # Hypotension: Low BP with compensatory tachycardia
            modified_vitals.systolic_bp = random.uniform(75, 90)
            modified_vitals.heart_rate += random.uniform(20, 30)
            
        elif event == AcuteEventType.MEDICATION_RESPONSE:
            # Recovery: Gradual improvement towards baseline
            recovery_factor = min(self.acute_event_duration / 15.0, 1.0)
            if self.baseline_vitals:
                modified_vitals.systolic_bp += 5 * recovery_factor
                modified_vitals.heart_rate -= 3 * recovery_factor
                modified_vitals.spo2 += 2 * recovery_factor
                modified_vitals.temperature -= 0.2 * recovery_factor
                
        elif event == AcuteEventType.RAPID_RECOVERY:
            # Rapid return to stable ranges
            if self.baseline_vitals:
                modified_vitals.heart_rate = self.baseline_vitals.heart_rate + random.uniform(-5, 5)
                modified_vitals.systolic_bp = self.baseline_vitals.systolic_bp + random.uniform(-5, 5)
                modified_vitals.spo2 = min(100, self.baseline_vitals.spo2 + random.uniform(0, 3))
                modified_vitals.respiratory_rate = self.baseline_vitals.respiratory_rate + random.uniform(-1, 1)
                modified_vitals.temperature = self.baseline_vitals.temperature + random.uniform(-0.1, 0.1)
        
        return modified_vitals
    
    def _update_acute_event_status(self) -> None:
        """PART 6: Update acute event with 2-minute cooldown after resolution"""
        # Decrease cooldown timer
        if self.acute_event_cooldown > 0:
            self.acute_event_cooldown -= 1
        
        if self.active_acute_event is None:
            return
            
        self.acute_event_duration -= 1
        
        if self.acute_event_duration <= 0:
            logger.info(f"Patient {self.patient_id}: EVENT RESOLVED → {self.active_acute_event.value}")
            
            # PART 6: Force RECOVERY state after event ends
            if self.active_acute_event in [AcuteEventType.SEPSIS_SPIKE, AcuteEventType.HYPOXIA_EVENT, AcuteEventType.HYPOTENSION_DROP]:
                self.state = PatientState.RECOVERING
                logger.info(f"Patient {self.patient_id}: POST-EVENT → FORCED RECOVERY")
            elif self.active_acute_event == AcuteEventType.MEDICATION_RESPONSE:
                self.state = PatientState.RECOVERING
                logger.info(f"Patient {self.patient_id}: MEDICATION → RECOVERY")
            
            # Clear acute event and set 2-minute cooldown
            self.active_acute_event = None
            self.baseline_vitals = None
            self.acute_event_cooldown = 120  # 2 minutes = 120 seconds
            self.state_duration = 0  # Reset for new state
            
            logger.info(f"Patient {self.patient_id}: 2-minute acute event cooldown started")
    
    def _check_shock_index_escalation(self, vitals: VitalSigns) -> None:
        """PART 5: Shock index escalation/de-escalation with clinical criteria"""
        shock_index = vitals.shock_index
        
        # Track max shock index for validation
        self.max_shock_index_recorded = max(self.max_shock_index_recorded, shock_index)
        
        # PART 5: Si > 1.3 AND persists 5 consecutive seconds → escalate to CRITICAL
        if shock_index > 1.3:
            self.shock_index_high_count += 1
            self.shock_index_normal_count = 0  # Reset normal counter
            
            if (self.shock_index_high_count >= 5 and 
                self.state not in [PatientState.CRITICAL, PatientState.INTERVENTION]):
                logger.warning(
                    f"Patient {self.patient_id}: SHOCK INDEX ESCALATION → CRITICAL "
                    f"(SI: {shock_index:.3f}, sustained for {self.shock_index_high_count}s)"
                )
                self.state = PatientState.CRITICAL
                self.state_duration = 0
                
        # PART 5: Si < 1.0 for 20 consecutive seconds → de-escalate
        elif shock_index < 1.0:
            self.shock_index_normal_count += 1
            self.shock_index_high_count = 0  # Reset high counter
            
            if (self.shock_index_normal_count >= 20 and 
                self.state in [PatientState.CRITICAL, PatientState.LATE_DETERIORATION]):
                logger.info(
                    f"Patient {self.patient_id}: SHOCK INDEX DE-ESCALATION → RECOVERING "
                    f"(SI: {shock_index:.3f}, stable for {self.shock_index_normal_count}s)"
                )
                self.state = PatientState.RECOVERING
                self.state_duration = 0
        else:
            # SI between 1.0 and 1.3 - reset both counters
            self.shock_index_high_count = 0
            self.shock_index_normal_count = 0
    
    def _transition_state(self) -> None:
        """PART 3: Realistic state distribution with exact probabilities"""
        self.state_duration += 1
        
        # Track time in critical state for validation
        if self.state == PatientState.CRITICAL:
            self.critical_state_time += 1
        
        # Don't transition too quickly
        if self.state_duration < self.min_state_duration:
            return
        
        # PART 3: Exact state transition probabilities for realistic distribution
        # Target: 60-75% STABLE, 15-25% EARLY_DETERIORATION, 3-8% CRITICAL
        
        if self.state == PatientState.STABLE:
            # PART 3: 98% remain STABLE, 1.5% → EARLY_DETERIORATION, 0.5% natural variance
            roll = random.random()
            if roll < 0.015:  # 1.5% chance to deteriorate
                self.state = PatientState.EARLY_DETERIORATION
                self.state_duration = 0
                logger.info(f"Patient {self.patient_id}: STABLE → EARLY_DETERIORATION")
                
        elif self.state == PatientState.EARLY_DETERIORATION:
            # PART 3: 75% recover → STABLE, 20% remain EARLY, 5% worsen → CRITICAL
            roll = random.random()
            if roll < 0.75:  # 75% recover to stable
                self.state = PatientState.STABLE
                self.state_duration = 0
                logger.info(f"Patient {self.patient_id}: EARLY_DETERIORATION → STABLE")
            elif roll < 0.95:  # 20% remain (95% - 75% = 20%)
                pass  # Stay in current state
            else:  # 5% move to critical
                self.state = PatientState.CRITICAL
                self.state_duration = 0
                logger.warning(f"Patient {self.patient_id}: EARLY_DETERIORATION → CRITICAL")
                
        elif self.state == PatientState.LATE_DETERIORATION:
            # Transition to CRITICAL or INTERVENTION quickly
            if random.random() < 0.8:  # 80% chance for intervention
                self.state = PatientState.INTERVENTION
                self.state_duration = 0
                logger.info(f"Patient {self.patient_id}: LATE_DETERIORATION → INTERVENTION")
            else:  # 20% become critical
                self.state = PatientState.CRITICAL
                self.state_duration = 0
                logger.warning(f"Patient {self.patient_id}: LATE_DETERIORATION → CRITICAL")
                
        elif self.state == PatientState.CRITICAL:
            # PART 3: 60% → INTERVENTION, 30% remain CRITICAL, 10% slow improvement
            roll = random.random()
            if roll < 0.60:  # 60% get intervention
                self.state = PatientState.INTERVENTION
                self.state_duration = 0
                logger.warning(f"Patient {self.patient_id}: CRITICAL → INTERVENTION")
            elif roll < 0.90:  # 30% remain critical (90% - 60% = 30%)
                pass  # Stay critical
            else:  # 10% gradual improvement
                self.state = PatientState.RECOVERING
                self.state_duration = 0
                logger.info(f"Patient {self.patient_id}: CRITICAL → RECOVERING")
                
        elif self.state == PatientState.INTERVENTION:
            # PART 3: 80% move toward RECOVERY, 20% remain
            if self.state_duration > 90:  # After 1.5 minutes of intervention
                roll = random.random()
                if roll < 0.80:  # 80% gradual recovery
                    self.state = PatientState.RECOVERING
                    self.state_duration = 0
                    logger.info(f"Patient {self.patient_id}: INTERVENTION → RECOVERING")
                # 20% remain in intervention
                    
        elif self.state == PatientState.RECOVERING:
            # PART 3: 90% → STABLE, 10% remain
            if self.state_duration > 150:  # After 2.5 minutes of recovery
                if random.random() < 0.90:  # 90% chance to stabilize
                    self.state = PatientState.STABLE
                    self.state_duration = 0
                    logger.info(f"Patient {self.patient_id}: RECOVERING → STABLE")
    
    def _update_vitals_by_state(self) -> VitalSigns:
        """Update vital signs based on current patient state"""
        prev = self.last_vitals
        
        if self.state == PatientState.STABLE:
            #MIMIC-IV REALISM: Very small noise for stable patients (like real ICU monitors)
            vitals = VitalSigns(
                heart_rate=self._stability_bias_mechanism(
                    prev.heart_rate, self.baseline_hr, 0.5),  # HR noise: std 0.5 (was 1.5)
                systolic_bp=self._stability_bias_mechanism(
                    prev.systolic_bp, self.baseline_sbp, 0.8),  # SBP noise: std 0.8 (was 2.0)
                diastolic_bp=prev.diastolic_bp,  # Calculated in clamp_vitals
                spo2=self._stability_bias_mechanism(
                    prev.spo2, self.baseline_spo2, 0.1),  # SpO2 noise: std 0.1 (was 0.3)
                respiratory_rate=self._stability_bias_mechanism(
                    prev.respiratory_rate, self.baseline_rr, 0.3),  # RR noise: std 0.3 (was 0.8)
                temperature=self._stability_bias_mechanism(
                    prev.temperature, self.baseline_temp, 0.02)  # Temp noise: 0.02 (was 0.05)
            )
            return self._clamp_vitals(vitals)
            
        elif self.state == PatientState.EARLY_DETERIORATION:
            # EARLY: HR 90-120, SBP 90-105, SpO2 92-95, RR 20-26, Temp 37.8-39.0
            target_hr = min(120, max(90, self.baseline_hr + 25))
            target_sbp = min(105, max(90, self.baseline_sbp - 10))
            target_spo2 = min(95, max(92, self.baseline_spo2 - 3))
            target_rr = min(26, max(20, self.baseline_rr + 6))
            target_temp = min(39.0, max(37.8, self.baseline_temp + 0.8))
            
            vitals = VitalSigns(
                heart_rate=self._stability_bias_mechanism(
                    prev.heart_rate, target_hr, 0.7),  # Reduced from 1.2
                systolic_bp=self._stability_bias_mechanism(
                    prev.systolic_bp, target_sbp, 1.0),  # Reduced from 1.5
                diastolic_bp=prev.diastolic_bp,  # Calculated in clamp_vitals
                spo2=self._stability_bias_mechanism(
                    prev.spo2, target_spo2, 0.2),  # Reduced from 0.4
                respiratory_rate=self._stability_bias_mechanism(
                    prev.respiratory_rate, target_rr, 0.4),  # Reduced from 0.6
                temperature=self._stability_bias_mechanism(
                    prev.temperature, target_temp, 0.03)  # Reduced from 0.05
            )
            
            return self._clamp_vitals(vitals)
            
        elif self.state == PatientState.LATE_DETERIORATION:
            # LATE DETERIORATION: Similar to early but more pronounced - transition to critical quickly
            target_hr = min(140, max(100, self.baseline_hr + 40))
            target_sbp = min(95, max(80, self.baseline_sbp - 20))
            target_spo2 = min(90, max(88, self.baseline_spo2 - 6))
            target_rr = min(30, max(24, self.baseline_rr + 8))
            target_temp = min(39.5, max(38.0, self.baseline_temp + 1.2))
            
            vitals = VitalSigns(
                heart_rate=self._stability_bias_mechanism(
                    prev.heart_rate, target_hr, 0.8),  # Reduced from 1.2
                systolic_bp=self._stability_bias_mechanism(
                    prev.systolic_bp, target_sbp, 1.0),  # Reduced from 1.5
                diastolic_bp=prev.diastolic_bp,  # Calculated in clamp_vitals
                spo2=self._stability_bias_mechanism(
                    prev.spo2, target_spo2, 0.3),  # Reduced from 0.4
                respiratory_rate=self._stability_bias_mechanism(
                    prev.respiratory_rate, target_rr, 0.5),  # Reduced from 0.6
                temperature=self._stability_bias_mechanism(
                    prev.temperature, target_temp, 0.04)  # Reduced from 0.05
            )
            
            return self._clamp_vitals(vitals)
            
        elif self.state == PatientState.CRITICAL:
            # CRITICAL: HR 110-160, SBP 70-90, SpO2 85-92, RR 26-35, Temp 39-40
            target_hr = min(160, max(110, self.baseline_hr + 55))
            target_sbp = min(90, max(70, self.baseline_sbp - 30))
            target_spo2 = min(92, max(85, self.baseline_spo2 - 8))
            target_rr = min(35, max(26, self.baseline_rr + 12))
            target_temp = min(40, max(39, self.baseline_temp + 2.0))
            
            vitals = VitalSigns(
                heart_rate=self._stability_bias_mechanism(
                    prev.heart_rate, target_hr, 1.0),  # Reduced from 1.2 (more variability in critical)
                systolic_bp=self._stability_bias_mechanism(
                    prev.systolic_bp, target_sbp, 1.2),  # Reduced from 1.5
                diastolic_bp=prev.diastolic_bp,  # Calculated in clamp_vitals
                spo2=self._stability_bias_mechanism(
                    prev.spo2, target_spo2, 0.5),  # Slightly increased from 0.4 (unstable in critical)
                respiratory_rate=self._stability_bias_mechanism(
                    prev.respiratory_rate, target_rr, 0.7),  # Slightly increased from 0.6
                temperature=self._stability_bias_mechanism(
                    prev.temperature, target_temp, 0.04)  # Reduced from 0.05
            )
            return self._clamp_vitals(vitals)
            
        elif self.state == PatientState.INTERVENTION:
            # PART 7: Recovery dynamics with gradual improvement slopes (MIMIC-IV realism)
            # HR decrease slope: max 3 bpm per second
            # SBP increase slope: max 3 mmHg per second  
            # SpO2 increase slope: max 1% per second
            
            # Calculate improvement direction
            hr_improvement = -3 if prev.heart_rate > self.baseline_hr else 3
            sbp_improvement = 3 if prev.systolic_bp < self.baseline_sbp else -3
            spo2_improvement = 1 if prev.spo2 < self.baseline_spo2 else -1
            
            vitals = VitalSigns(
                heart_rate=prev.heart_rate + hr_improvement + random.gauss(0, 0.5),  # Reduced from 1.0
                systolic_bp=prev.systolic_bp + sbp_improvement + random.gauss(0, 0.8),  # Reduced from 1.5
                diastolic_bp=prev.diastolic_bp,  # Calculated in clamp_vitals
                spo2=prev.spo2 + spo2_improvement + random.gauss(0, 0.2),  # Reduced from 0.3
                respiratory_rate=prev.respiratory_rate + (self.baseline_rr - prev.respiratory_rate) * 0.05 + random.gauss(0, 0.4),  # Reduced from 0.6
                temperature=prev.temperature + (self.baseline_temp - prev.temperature) * 0.03 + random.gauss(0, 0.03)  # Reduced from 0.05
            )
            return self._clamp_vitals(vitals)
            
        elif self.state == PatientState.RECOVERING:
            # PART 7: Gradual recovery dynamics - no instant corrections
            # Use stability bias mechanism to gradually return to baseline
            vitals = VitalSigns(
                heart_rate=self._stability_bias_mechanism(
                    prev.heart_rate, self.baseline_hr, 1.2),
                systolic_bp=self._stability_bias_mechanism(
                    prev.systolic_bp, self.baseline_sbp, 1.5),
                diastolic_bp=prev.diastolic_bp,  # Calculated in clamp_vitals
                spo2=self._stability_bias_mechanism(
                    prev.spo2, self.baseline_spo2, 0.4),
                respiratory_rate=self._stability_bias_mechanism(
                    prev.respiratory_rate, self.baseline_rr, 0.6),
                temperature=self._stability_bias_mechanism(
                    prev.temperature, self.baseline_temp, 0.05)
            )
            return self._clamp_vitals(vitals)
        
        return self.last_vitals
    
    def _validate_vitals(self, vitals: VitalSigns, prev_vitals: VitalSigns) -> None:
        """Validate vital signs for unrealistic values or jumps"""
        # Check for unrealistic absolute values
        if vitals.systolic_bp < 50 or vitals.systolic_bp > 200:
            logger.warning(f"Patient {self.patient_id}: Unrealistic SBP: {vitals.systolic_bp}")
        if vitals.heart_rate < 40 or vitals.heart_rate > 180:
            logger.warning(f"Patient {self.patient_id}: Unrealistic HR: {vitals.heart_rate}")
        if vitals.spo2 < 70 or vitals.spo2 > 100:
            logger.warning(f"Patient {self.patient_id}: Unrealistic SpO2: {vitals.spo2}")
            
        # Check for unrealistic jumps (>20% change in 1 second)
        def check_jump(current, previous, name):
            if previous > 0:  # Avoid division by zero
                change_pct = abs((current - previous) / previous) * 100
                if change_pct > 20:
                    logger.warning(
                        f"Patient {self.patient_id}: Large {name} jump: "
                        f"{previous:.1f} -> {current:.1f} ({change_pct:.1f}%)"
                    )
        
        check_jump(vitals.heart_rate, prev_vitals.heart_rate, "HR")
        check_jump(vitals.systolic_bp, prev_vitals.systolic_bp, "SBP")
        check_jump(vitals.spo2, prev_vitals.spo2, "SpO2")

    def update(self) -> Dict:
        """Update patient state and generate new vital signs reading with acute event engine"""
        prev_state = self.state
        prev_vitals = self.last_vitals
        
        # === BASELINE LOGIC ===
        self._transition_state()
        baseline_vitals = self._update_vitals_by_state()
        
        # === ACUTE EVENT ENGINE ===
        # 1. Check for new acute events
        self.maybe_trigger_acute_event()
        
        # 2. Apply acute event modifiers
        modified_vitals = self.apply_acute_event_modifiers(baseline_vitals)

        # 2b. Apply scripted gradual spike modifiers (if active)
        trended_vitals = self._apply_scripted_spike_modifiers(modified_vitals)
        
        # 3. Apply physiological correlations
        correlated_vitals = self._apply_physiological_correlations(trended_vitals, self.state)
        
        # 4. Clamp to safe ranges
        final_vitals = self._clamp_vitals(correlated_vitals)
        
        # 5. Check shock index and escalate if needed
        self._check_shock_index_escalation(final_vitals)
        
        # 6. Update acute event status
        self._update_acute_event_status()

        # 7. Advance scripted spike state machine
        self._tick_scripted_spike()
        
        # === VALIDATION ===
        self._validate_vitals(final_vitals, prev_vitals)
        
        # Store final vitals
        self.last_vitals = final_vitals
        
        # Log significant state transitions
        if self.state != prev_state:
            logger.info(f"Patient {self.patient_id}: State transition {prev_state.value} → {self.state.value}")
        
        # === EVENT PAYLOAD ===
        event = {
            "patient_id": self.patient_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "heart_rate": round(self.last_vitals.heart_rate, 1),
            "systolic_bp": round(self.last_vitals.systolic_bp, 1),
            "diastolic_bp": round(self.last_vitals.diastolic_bp, 1),
            "spo2": round(self.last_vitals.spo2, 1),
            "respiratory_rate": round(self.last_vitals.respiratory_rate, 1),
            "temperature": round(self.last_vitals.temperature, 2),
            "shock_index": round(self.last_vitals.shock_index, 3),
            "state": self.state.value,
            "event_type": self.active_acute_event.value if self.active_acute_event else AcuteEventType.NONE.value
        }
        
        return event

class VitalSimulator:
    """Main vital signs simulator coordinating multiple patients"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092')
        self.kafka_topic = 'vitals'
        self.patients = {}
        self.producer = None
        self.running = False
        self.debug_mode = DEBUG_MODE
        self.event_count = 0

        # Predictable-random gradual spike scheduler configuration
        self.scripted_spike_enabled = os.getenv('SCRIPTED_SPIKE_ENABLED', 'true').lower() == 'true'
        self.scripted_spike_seed = int(os.getenv('SCRIPTED_SPIKE_SEED', '2026'))
        self.scripted_spike_min_cycle = int(os.getenv('SCRIPTED_SPIKE_MIN_CYCLE_SECONDS', '300'))
        self.scripted_spike_max_cycle = int(os.getenv('SCRIPTED_SPIKE_MAX_CYCLE_SECONDS', '480'))
        self.scripted_spike_ramp_up = int(os.getenv('SCRIPTED_SPIKE_RAMP_UP_SECONDS', '120'))
        self.scripted_spike_hold = int(os.getenv('SCRIPTED_SPIKE_HOLD_SECONDS', '45'))
        self.scripted_spike_ramp_down = int(os.getenv('SCRIPTED_SPIKE_RAMP_DOWN_SECONDS', '120'))
        self.scripted_spike_rng = random.Random(self.scripted_spike_seed)
        self._scripted_spike_task: Optional[asyncio.Task] = None
        
        # PART 8: Realism validation counters
        self.total_simulation_time = 0
        self.critical_state_warnings_logged = 0
        
        if self.debug_mode:
            logger.info("🔧 DEBUG MODE ENABLED - Events will be printed to console instead of Kafka")
        else:
            logger.info(f"Initialized with Kafka servers: {self.kafka_servers}")

        if self.scripted_spike_enabled:
            logger.info(
                "🎯 Scripted gradual spikes ENABLED "
                f"(seed={self.scripted_spike_seed}, cycle={self.scripted_spike_min_cycle}-{self.scripted_spike_max_cycle}s, "
                f"shape={self.scripted_spike_ramp_up}/{self.scripted_spike_hold}/{self.scripted_spike_ramp_down}s)"
            )
        else:
            logger.info("🎯 Scripted gradual spikes DISABLED")

    async def _scripted_spike_scheduler_loop(self) -> None:
        """Pick a random patient on a deterministic schedule and trigger gradual spikes."""
        if not self.scripted_spike_enabled:
            return

        # Initial short delay so baseline histories populate first
        next_trigger_in = self.scripted_spike_rng.randint(45, 90)

        while self.running:
            try:
                await asyncio.sleep(1)

                if next_trigger_in > 0:
                    next_trigger_in -= 1
                    continue

                # Avoid overlapping scripted spikes
                if any(patient.has_scripted_spike_active() for patient in self.patients.values()):
                    next_trigger_in = 5
                    continue

                target_patient = self.scripted_spike_rng.choice(list(self.patients.values()))
                deltas = {
                    "heart_rate": self.scripted_spike_rng.uniform(24.0, 42.0),
                    "systolic_bp": self.scripted_spike_rng.uniform(16.0, 30.0),
                    "spo2": self.scripted_spike_rng.uniform(2.5, 6.5),
                    "respiratory_rate": self.scripted_spike_rng.uniform(4.0, 10.0),
                    "temperature": self.scripted_spike_rng.uniform(0.4, 1.1),
                }

                started = target_patient.start_scripted_spike(
                    ramp_up_seconds=self.scripted_spike_ramp_up,
                    hold_seconds=self.scripted_spike_hold,
                    ramp_down_seconds=self.scripted_spike_ramp_down,
                    deltas=deltas,
                )

                if started:
                    logger.warning(
                        f"🎯 Scheduled gradual spike assigned to {target_patient.patient_id}; "
                        "risk should show ramp-up trend instead of abrupt jump"
                    )

                next_trigger_in = self.scripted_spike_rng.randint(
                    max(60, self.scripted_spike_min_cycle),
                    max(max(60, self.scripted_spike_min_cycle), self.scripted_spike_max_cycle)
                )

            except asyncio.CancelledError:
                logger.info("Scripted spike scheduler cancelled")
                break
            except Exception as exc:
                logger.error(f"Error in scripted spike scheduler: {exc}")
                next_trigger_in = 10
    
    def _setup_kafka_producer(self) -> Optional[KafkaProducer]:
        """Initialize Kafka producer with error handling"""
        if self.debug_mode:
            logger.info("Skipping Kafka producer initialization (DEBUG_MODE=true)")
            return None
            
        if not KAFKA_AVAILABLE:
            logger.error("Kafka library not available - install kafka-python")
            return None
            
        try:
            logger.info(f"Connecting to Kafka at {self.kafka_servers}...")
            producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=16384,
                linger_ms=10,
                acks='all',
                retries=3,
                retry_backoff_ms=100,
                request_timeout_ms=30000,
                api_version=(0, 10, 1)
            )
            logger.info("✅ Kafka producer initialized successfully")
            return producer
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kafka producer: {e}")
            logger.error("Check that Kafka is running and KAFKA_BOOTSTRAP_SERVERS is correct")
            return None
    
    def _initialize_patients(self) -> None:
        """Initialize 8 ICU patients"""
        for i in range(1, 9):
            patient_id = f"P{i}"
            self.patients[patient_id] = Patient(patient_id)
        
        logger.info(f"Initialized {len(self.patients)} patients: {list(self.patients.keys())}")
    
    async def _patient_simulation_loop(self, patient: Patient) -> None:
        """Async loop for individual patient data generation"""
        logger.info(f"Starting simulation loop for patient {patient.patient_id}")
        
        while self.running:
            try:
                # Generate new vital signs reading
                event = patient.update()
                self.event_count += 1
                
                # Send to Kafka or print to console
                if self.debug_mode:
                    # Debug mode - print to console
                    print(f"\n[DEBUG] {json.dumps(event, indent=2)}")
                    if self.event_count % 10 == 0:
                        logger.info(f"📊 Generated {self.event_count} events so far")
                else:
                    # Production mode - send to Kafka
                    if self.producer:
                        try:
                            future = self.producer.send(self.kafka_topic, value=event)
                            # Log successful publish occasionally
                            if self.event_count % 50 == 0:
                                logger.info(f"📤 Published {self.event_count} events to Kafka topic '{self.kafka_topic}'")
                        except Exception as e:
                            logger.error(f"❌ Failed to send to Kafka: {e}")
                    else:
                        logger.warning("No Kafka producer available - event not sent")
                
                # PART 8: Realism validation logging
                self.total_simulation_time += 1
                
                # Check for realism violations and log warnings
                if patient.max_shock_index_recorded > 2.0:
                    if not hasattr(patient, '_si_warning_logged'):
                        logger.warning(f"⚠️ REALISM VIOLATION: Patient {patient.patient_id} shock index exceeded 2.0 (max: {patient.max_shock_index_recorded:.3f})")
                        patient._si_warning_logged = True
                
                # Log critical states and acute events
                if patient.active_acute_event:
                    logger.warning(
                        f"🚨 Patient {patient.patient_id} [ACUTE: {patient.active_acute_event.value}] "
                        f"[{patient.state.value}]: HR={event['heart_rate']}, SBP={event['systolic_bp']}, "
                        f"SpO2={event['spo2']}, SI={event['shock_index']} (Duration: {patient.acute_event_duration}s)"
                    )
                elif patient.state in [PatientState.CRITICAL, PatientState.INTERVENTION]:
                    logger.warning(
                        f"🚨 Patient {patient.patient_id} [{patient.state.value}]: "
                        f"HR={event['heart_rate']}, SBP={event['systolic_bp']}, "
                        f"SpO2={event['spo2']}, SI={event['shock_index']}"
                    )
                elif random.random() < 0.05:  # Log 5% of normal readings
                    logger.info(
                        f"📈 Patient {patient.patient_id} [{patient.state.value}]: "
                        f"HR={event['heart_rate']}, SBP={event['systolic_bp']}, SpO2={event['spo2']}"
                    )
                
                # PART 8: Check if CRITICAL percentage > 10% every 5 minutes
                if self.total_simulation_time % 300 == 0:  # Every 5 minutes
                    critical_patients = sum(1 for p in self.patients.values() if p.state == PatientState.CRITICAL)
                    critical_percentage = (critical_patients / len(self.patients)) * 100
                    
                    if critical_percentage > 10:
                        logger.warning(
                            f"⚠️ REALISM WARNING: {critical_percentage:.1f}% of patients are CRITICAL (>10% threshold). "
                            f"Critical patients: {critical_patients}/{len(self.patients)}"
                        )
                        self.critical_state_warnings_logged += 1
                    else:
                        logger.info(
                            f"✅ Population Distribution Check: {critical_percentage:.1f}% CRITICAL "
                            f"({critical_patients}/{len(self.patients)} patients) - Within realistic range"
                        )
                
                # Wait for next reading (1 second)
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                logger.info(f"Patient {patient.patient_id} simulation cancelled")
                break
            except Exception as e:
                logger.error(f"Error in patient {patient.patient_id} simulation: {e}")
                await asyncio.sleep(1.0)  # Continue simulation after error
    
    async def start(self) -> None:
        """Start the vital simulator service"""
        logger.info("Starting Vital Simulator Service")
        
        # Initialize components
        self.producer = self._setup_kafka_producer()
        self._initialize_patients()
        
        self.running = True
        
        # Start patient simulation tasks
        tasks = []

        if self.scripted_spike_enabled:
            self._scripted_spike_task = asyncio.create_task(self._scripted_spike_scheduler_loop())
            tasks.append(self._scripted_spike_task)

        for patient in self.patients.values():
            task = asyncio.create_task(self._patient_simulation_loop(patient))
            tasks.append(task)
        
        logger.info("All patient simulations started - ICU telemetry generation active")
        
        try:
            # Wait for all patient simulations
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Simulation tasks cancelled")
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the service"""
        logger.info("Shutting down Vital Simulator Service")
        self.running = False
        
        # PART 8: Clinical Realism Validation Summary
        logger.info("=" * 60)
        logger.info("CLINICAL REALISM VALIDATION REPORT")
        logger.info("=" * 60)
        
        total_acute_events = sum(p.acute_event_count for p in self.patients.values())
        avg_acute_events_per_patient = total_acute_events / len(self.patients)
        critical_patients = sum(1 for p in self.patients.values() if p.state == PatientState.CRITICAL)
        critical_percentage = (critical_patients / len(self.patients)) * 100
        
        # Calculate max shock indices across all patients
        max_shock_indices = [p.max_shock_index_recorded for p in self.patients.values()]
        highest_shock_index = max(max_shock_indices) if max_shock_indices else 0.0
        
        logger.info(f"📊 Total Simulation Time: {self.total_simulation_time} seconds")
        logger.info(f"📊 Total Events Generated: {self.event_count}")
        logger.info(f"📊 Total Acute Events: {total_acute_events} ({avg_acute_events_per_patient:.1f} per patient)")
        logger.info(f"📊 Current Critical Patients: {critical_patients}/{len(self.patients)} ({critical_percentage:.1f}%)")
        logger.info(f"📊 Highest Shock Index Recorded: {highest_shock_index:.3f}")
        logger.info(f"📊 Critical State Warnings: {self.critical_state_warnings_logged}")
        
        # Validation checks
        if critical_percentage <= 10:
            logger.info("✅ PASS: Critical patient percentage within realistic range (≤10%)")
        else:
            logger.warning(f"❌ FAIL: Critical patient percentage too high ({critical_percentage:.1f}% > 10%)")
            
        if highest_shock_index <= 2.0:
            logger.info("✅ PASS: All shock indices within physiological limits (≤2.0)")
        else:
            logger.warning(f"❌ FAIL: Shock index exceeded limit (max: {highest_shock_index:.3f} > 2.0)")
            
        acute_event_rate = (total_acute_events / self.total_simulation_time) * 100 if self.total_simulation_time > 0 else 0
        if acute_event_rate <= 0.05:  # 0.05% = very rare
            logger.info(f"✅ PASS: Acute event rate realistic ({acute_event_rate:.3f}% ≤ 0.05%)")
        else:
            logger.warning(f"❌ FAIL: Acute event rate too high ({acute_event_rate:.3f}% > 0.05%)")
        
        logger.info("=" * 60)
        
        if self.producer:
            try:
                logger.info("Flushing remaining Kafka messages...")
                self.producer.flush(timeout=10)
                self.producer.close()
                logger.info("✅ Kafka producer closed gracefully")
            except Exception as e:
                logger.error(f"❌ Error closing Kafka producer: {e}")
        
        logger.info(f"📊 Total events generated: {self.event_count}")

async def main():
    """Main entry point"""
    simulator = VitalSimulator()
    
    try:
        await simulator.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        raise
    finally:
        await simulator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())