"""
Phase 2.1: Multivariate Correlation Risk Calculator
Analyzes cross-vital dependencies and temporal patterns for anomaly detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CorrelationType(str, Enum):
    """Types of vital sign correlations"""
    COMPENSATORY = "compensatory"  # Expected: HR↑ when SpO2↓ (body compensates)
    ANTAGONISTIC = "antagonistic"  # Expected: SBP↑ when HR↓ (inverse relationship)
    SYNERGISTIC = "synergistic"    # Expected: SpO2↓ with RR↑ (both change together)
    INDEPENDENT = "independent"    # No expected correlation


@dataclass
class VitalDeviation:
    """Deviation from baseline for a single vital"""
    name: str
    value: float
    baseline_mean: float
    baseline_std: float
    deviation_sigma: float  # How many std deviations from baseline
    in_green_zone: bool
    
    @property
    def deviation_magnitude(self) -> float:
        """Absolute deviation from baseline"""
        return abs(self.value - self.baseline_mean)
    
    @property
    def direction(self) -> str:
        """Direction of deviation: 'high', 'low', or 'normal'"""
        if self.deviation_sigma > 1.5:
            return "high"
        elif self.deviation_sigma < -1.5:
            return "low"
        else:
            return "normal"


@dataclass
class CorrelationPattern:
    """Detected correlation pattern between vitals"""
    vital1: str
    vital2: str
    expected_correlation: CorrelationType
    actual_direction1: str  # 'high', 'low', 'normal'
    actual_direction2: str
    is_anomalous: bool
    risk_contribution: float  # 0.0 to 1.0
    explanation: str


@dataclass
class CorrelationRiskAssessment:
    """Complete multivariate correlation risk assessment"""
    timestamp: str
    overall_risk: float  # 0.0 to 1.0
    deviations: Dict[str, VitalDeviation]
    detected_patterns: List[CorrelationPattern]
    anomaly_count: int
    risk_factors: List[str]
    confidence: float  # Assessment confidence 0.0 to 1.0
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "timestamp": self.timestamp,
            "overall_risk": round(self.overall_risk, 3),
            "deviations": {
                name: {
                    "value": dev.value,
                    "baseline_mean": dev.baseline_mean,
                    "deviation_sigma": round(dev.deviation_sigma, 2),
                    "direction": dev.direction,
                    "in_green_zone": dev.in_green_zone
                }
                for name, dev in self.deviations.items()
            },
            "detected_patterns": [
                {
                    "vitals": f"{p.vital1} ↔ {p.vital2}",
                    "type": p.expected_correlation,
                    "is_anomalous": p.is_anomalous,
                    "risk_contribution": round(p.risk_contribution, 3),
                    "explanation": p.explanation
                }
                for p in self.detected_patterns
            ],
            "anomaly_count": self.anomaly_count,
            "risk_factors": self.risk_factors,
            "confidence": round(self.confidence, 2)
        }


class CorrelationRiskCalculator:
    """
    Multivariate Trend Correlation Engine for VitalX
    
    Analyzes cross-vital dependencies to detect:
    1. Broken compensatory mechanisms (HR↑ but SpO2 not compensating)
    2. Antagonistic breakdown (BP and HR both rising inappropriately)
    3. Synergistic deterioration (Multiple vitals degrading together)
    4. Temporal anomalies (Changes faster/slower than expected)
    """
    
    # Expected physiological correlations
    CORRELATIONS = {
        ("HR", "SpO2"): {
            "type": CorrelationType.COMPENSATORY,
            "rule": "HR should increase when SpO2 drops (tachycardia compensates for hypoxemia)",
            "anomaly": "HR↑ SpO2↓↓ = Decompensation"
        },
        ("HR", "SBP"): {
            "type": CorrelationType.ANTAGONISTIC,
            "rule": "Normally inverse: HR↑ → SBP↓ or HR↓ → SBP↑",
            "anomaly": "Both rising = Cardiac stress or sepsis"
        },
        ("SpO2", "RR"): {
            "type": CorrelationType.COMPENSATORY,
            "rule": "RR should increase when SpO2 drops (respiratory compensation)",
            "anomaly": "SpO2↓ RR→ = Respiratory failure"
        },
        ("RR", "HR"): {
            "type": CorrelationType.SYNERGISTIC,
            "rule": "Both increase together during distress",
            "anomaly": "RR↑ HR→ = Impaired cardiac response"
        },
        ("SBP", "Temp"): {
            "type": CorrelationType.SYNERGISTIC,
            "rule": "Fever can cause vasodilation and BP drop",
            "anomaly": "Temp↑ SBP↓↓ = Septic shock"
        }
    }
    
    def __init__(self, sensitivity: float = 1.0):
        """
        Initialize correlationcalculator
        
        Args:
            sensitivity: Multiplier for risk scoring (1.0 = normal, >1.0 = more sensitive)
        """
        self.sensitivity = sensitivity
    
    def calculate_deviation(
        self,
        vital_name: str,
        value: float,
        baseline_mean: float,
        baseline_std: float,
        green_zone_min: float,
        green_zone_max: float
    ) -> VitalDeviation:
        """Calculate deviation metrics for a single vital"""
        
        # Avoid division by zero
        std = max(baseline_std, 0.01)
        
        deviation_sigma = (value - baseline_mean) / std
        in_green_zone = green_zone_min <= value <= green_zone_max
        
        return VitalDeviation(
            name=vital_name,
            value=value,
            baseline_mean=baseline_mean,
            baseline_std=std,
            deviation_sigma=deviation_sigma,
            in_green_zone=in_green_zone
        )
    
    def detect_correlation_anomaly(
        self,
        vital1: str,
        vital2: str,
        dev1: VitalDeviation,
        dev2: VitalDeviation
    ) -> Optional[CorrelationPattern]:
        """
        Detect if correlation between two vitals is anomalous
        
        Returns CorrelationPattern if anomaly detected, None otherwise
        """
        
        # Get expected correlation rule
        key = (vital1, vital2)
        reverse_key = (vital2, vital1)
        
        if key not in self.CORRELATIONS and reverse_key not in self.CORRELATIONS:
            return None
        
        if key in self.CORRELATIONS:
            corr_info = self.CORRELATIONS[key]
            d1, d2 = dev1, dev2
            v1, v2 = vital1, vital2
        else:
            corr_info = self.CORRELATIONS[reverse_key]
            d1, d2 = dev2, dev1
            v1, v2 = vital2, vital1
        
        corr_type = corr_info["type"]
        is_anomalous = False
        risk = 0.0
        explanation = ""
        
        # COMPENSATORY: Vital1 should increase when Vital2 decreases
        if corr_type == CorrelationType.COMPENSATORY:
            # Check for broken compensation
            if d1.direction == "high" and d2.direction == "low":
                # Expected: HR↑ when SpO2↓ (normal compensation)
                is_anomalous = False
                risk = 0.1  # Low risk (healthy compensation)
                explanation = f"{v1} compensating for {v2} decline (normal)"
            
            elif d1.direction == "normal" and d2.direction == "low":
                # Anomaly: Vital2 dropping but Vital1 not responding
                is_anomalous = True
                risk = 0.6  # High risk (failed compensation)
                explanation = f"{v2}↓ but {v1} not compensating → Decompensation"
            
            elif d1.direction == "high" and d2.direction == "normal":
                # Anomaly: Vital1 rising without Vital2 trigger
                is_anomalous = True
                risk = 0.4
                explanation = f"{v1}↑ without {v2} decline → Inappropriate response"
        
        # ANTAGONISTIC: Inverse relationship expected
        elif corr_type == CorrelationType.ANTAGONISTIC:
            if d1.direction == "high" and d2.direction == "high":
                # Both rising (e.g., HR and BP both up)
                is_anomalous = True
                risk = 0.7  # Very high risk (cardiac stress, sepsis)
                explanation = f"{v1}↑ {v2}↑ → Stress/sepsis (both should not rise)"
            
            elif d1.direction == "low" and d2.direction == "low":
                # Both dropping
                is_anomalous = True
                risk = 0.8  # Critical (hemodynamic collapse)
                explanation = f"{v1}↓ {v2}↓ → Hemodynamic collapse"
        
        # SYNERGISTIC: Should change together
        elif corr_type == CorrelationType.SYNERGISTIC:
            if (d1.direction in ["high", "low"]) and d2.direction == "normal":
                # One changing, other stable
                is_anomalous = True
                risk = 0.5
                explanation = f"{v1} changing but {v2} stable → Impaired coupling"
            
            elif d1.direction != d2.direction and d1.direction != "normal" and d2.direction != "normal":
                # Moving in opposite directions
                is_anomalous = True
                risk = 0.6
                explanation = f"{v1} and {v2} diverging → Dysregulation"
        
        if is_anomalous or risk > 0:
            return CorrelationPattern(
                vital1=v1,
                vital2=v2,
                expected_correlation=corr_type,
                actual_direction1=d1.direction,
                actual_direction2=d2.direction,
                is_anomalous=is_anomalous,
                risk_contribution=risk,
                explanation=explanation
            )
        
        return None
    
    def compute_risk(
        self,
        vitals: Dict[str, float],
        baseline: Dict[str, Dict[str, float]],
        timestamp: str = None
    ) -> CorrelationRiskAssessment:
        """
        Compute multivariate correlation risk assessment
        
        Args:
            vitals: Current vital values {"HR": 85, "SpO2": 94, ...}
            baseline: Baseline metrics from BaselineCalibrator
                      {"HR": {"mean": 75, "std": 3, "green_zone_min": 70, ...}, ...}
            timestamp: ISO timestamp (optional)
        
        Returns:
            CorrelationRiskAssessment with detailed analysis
        """
        
        import datetime
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        
        # Step 1: Calculate deviations for all vitals
        deviations = {}
        for vital_name in ["HR", "SpO2", "SBP", "RR", "Temp"]:
            if vital_name in vitals and vital_name in baseline:
                dev = self.calculate_deviation(
                    vital_name=vital_name,
                    value=vitals[vital_name],
                    baseline_mean=baseline[vital_name]["mean"],
                    baseline_std=baseline[vital_name]["std"],
                    green_zone_min=baseline[vital_name]["green_zone_min"],
                    green_zone_max=baseline[vital_name]["green_zone_max"]
                )
                deviations[vital_name] = dev
        
        # Step 2: Detect correlation anomalies
        detected_patterns = []
        for (v1, v2) in self.CORRELATIONS.keys():
            if v1 in deviations and v2 in deviations:
                pattern = self.detect_correlation_anomaly(
                    v1, v2, deviations[v1], deviations[v2]
                )
                if pattern:
                    detected_patterns.append(pattern)
        
        # Step 3: Compute overall risk
        anomaly_count = sum(1 for p in detected_patterns if p.is_anomalous)
        
        # Base risk from deviations (vitals outside green zone)
        deviation_risk = 0.0
        vitals_outside_green = 0
        for dev in deviations.values():
            if not dev.in_green_zone:
                vitals_outside_green += 1
                # Risk increases with sigma distance
                deviation_risk += min(abs(dev.deviation_sigma) * 0.1, 0.3)
        
        # Correlation risk (weighted by pattern risk contributions)
        correlation_risk = sum(p.risk_contribution for p in detected_patterns) / max(len(detected_patterns), 1)
        
        # Combined risk (weighted average)
        base_risk = (0.4 * deviation_risk + 0.6 * correlation_risk) * self.sensitivity
        
        # Boost risk if multiple anomalies detected
        if anomaly_count >= 2:
            base_risk *= 1.3  # 30% increase for multiple anomalies
        if anomaly_count >= 3:
            base_risk *= 1.5  # 50% increase for critical multi-system failure
        
        overall_risk = min(base_risk, 1.0)  # Cap at 1.0
        
        # Step 4: Generate risk factors summary
        risk_factors = []
        if vitals_outside_green > 0:
            risk_factors.append(f"{vitals_outside_green} vitals outside Green Zone")
        if anomaly_count > 0:
            risk_factors.append(f"{anomaly_count} correlation anomalies detected")
        for pattern in detected_patterns:
            if pattern.is_anomalous and pattern.risk_contribution > 0.5:
                risk_factors.append(pattern.explanation)
        
        # Step 5: Calculate confidence (higher when more vitals available)
        confidence = len(deviations) / 5.0  # 5 vitals maximum
        if len(detected_patterns) > 0:
            confidence *= 1.1  # Boost when correlations detected
        confidence = min(confidence, 1.0)
        
        return CorrelationRiskAssessment(
            timestamp=timestamp,
            overall_risk=overall_risk,
            deviations=deviations,
            detected_patterns=detected_patterns,
            anomaly_count=anomaly_count,
            risk_factors=risk_factors if risk_factors else ["No significant anomalies"],
            confidence=confidence
        )
