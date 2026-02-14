"""
Baseline Calibration Module for VitalX
Implements dynamic patient-specific baseline ("fingerprint") calibration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("baseline-calibrator")


@dataclass
class BaselineMetrics:
    """Patient baseline vital sign ranges"""
    patient_id: str
    hr_mean: float
    hr_std: float
    spo2_mean: float
    spo2_std: float
    sbp_mean: float
    sbp_std: float
    rr_mean: float
    rr_std: float
    temp_mean: float
    temp_std: float
    timestamp: datetime
    stability_confidence: float  # 0.0 to 1.0
    sample_count: int
    
    def get_green_zone(self, vital_name: str) -> Tuple[float, float]:
        """
        Get the 'Green Zone' range for a vital sign (mean ± 1.5×std)
        Returns: (lower_bound, upper_bound)
        """
        vital_map = {
            'HR': (self.hr_mean, self.hr_std),
            'SpO2': (self.spo2_mean, self.spo2_std),
            'SBP': (self.sbp_mean, self.sbp_std),
            'RR': (self.rr_mean, self.rr_std),
            'Temp': (self.temp_mean, self.temp_std)
        }
        
        if vital_name not in vital_map:
            raise ValueError(f"Unknown vital: {vital_name}")
        
        mean, std = vital_map[vital_name]
        lower = mean - 1.5 * std
        upper = mean + 1.5 * std
        return (lower, upper)
    
    def to_dict(self) -> Dict:
        """Convert baseline to dictionary format"""
        return {
            'patient_id': self.patient_id,
            'vitals': {
                'HR': {'mean': self.hr_mean, 'std': self.hr_std},
                'SpO2': {'mean': self.spo2_mean, 'std': self.spo2_std},
                'SBP': {'mean': self.sbp_mean, 'std': self.sbp_std},
                'RR': {'mean': self.rr_mean, 'std': self.rr_std},
                'Temp': {'mean': self.temp_mean, 'std': self.temp_std}
            },
            'green_zones': {
                'HR': self.get_green_zone('HR'),
                'SpO2': self.get_green_zone('SpO2'),
                'SBP': self.get_green_zone('SBP'),
                'RR': self.get_green_zone('RR'),
                'Temp': self.get_green_zone('Temp')
            },
            'timestamp': self.timestamp.isoformat(),
            'stability_confidence': self.stability_confidence,
            'sample_count': self.sample_count
        }


class BaselineCalibrator:
    """
    Implements patient-specific baseline calibration for VitalX.
    
    Phase 1: "Fingerprint" Calibration
    - Cold Start: Collects first 10-30 seconds of vitals
    - Sanitizes artifacts using MAD filter
    - Projects dynamic baseline B_0 for each patient
    - Stores in patient_baselines registry
    
    Example: Post-stroke patient with resting HR=90 → sets 90 as "Green Zone"
    """
    
    def __init__(self, min_samples: int = 10, max_samples: int = 30):
        """
        Initialize baseline calibrator
        
        Args:
            min_samples: Minimum timesteps for stable baseline (default: 10 = 10 seconds at 1Hz)
            max_samples: Maximum timesteps for cold start (default: 30 = 30 seconds)
        """
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.patient_baselines: Dict[str, BaselineMetrics] = {}
        self.cold_start_buffers: Dict[str, List[List[float]]] = {}
        
        logger.info(f"BaselineCalibrator initialized (min={min_samples}, max={max_samples})")
    
    def ingest_cold_start(
        self, 
        patient_id: str, 
        vitals: List[float]
    ) -> Optional[BaselineMetrics]:
        """
        Ingest vital signs during cold-start phase.
        Collects first 10-30 timesteps to establish baseline.
        
        Args:
            patient_id: Unique patient identifier
            vitals: [HR, SpO2, SBP, RR, Temp] at current timestep
        
        Returns:
            BaselineMetrics if calibration complete, None if still collecting
        
        Example:
            >>> calibrator = BaselineCalibrator()
            >>> for timestep in range(30):
            ...     baseline = calibrator.ingest_cold_start("PT001", [85, 98, 120, 16, 37.0])
            >>> # baseline is now computed after 30 timesteps
        """
        # Validate input format
        if len(vitals) != 5:
            raise ValueError(f"Expected 5 vitals [HR, SpO2, SBP, RR, Temp], got {len(vitals)}")
        
        # Initialize buffer for new patient
        if patient_id not in self.cold_start_buffers:
            self.cold_start_buffers[patient_id] = []
            logger.info(f"Starting cold-start calibration for patient {patient_id}")
        
        # Add vitals to buffer
        self.cold_start_buffers[patient_id].append(vitals)
        buffer_size = len(self.cold_start_buffers[patient_id])
        
        # Check if we have enough samples
        if buffer_size < self.min_samples:
            logger.debug(f"Patient {patient_id}: Collecting samples ({buffer_size}/{self.min_samples})")
            return None
        
        # If we hit max samples or have sufficient data, compute baseline
        if buffer_size >= self.max_samples or buffer_size >= self.min_samples:
            logger.info(f"Patient {patient_id}: Computing baseline from {buffer_size} samples")
            
            # Sanitize artifacts
            sanitized_vitals = self.sanitize_artifacts(
                patient_id, 
                self.cold_start_buffers[patient_id]
            )
            
            # Project baseline
            baseline = self.project_baseline(patient_id, sanitized_vitals)
            
            # Store baseline
            self.patient_baselines[patient_id] = baseline
            
            # Clear buffer
            del self.cold_start_buffers[patient_id]
            
            logger.info(f"Patient {patient_id}: Baseline calibrated successfully "
                       f"(confidence={baseline.stability_confidence:.2f})")
            logger.info(f"  HR: {baseline.hr_mean:.1f} ± {baseline.hr_std:.1f} bpm")
            logger.info(f"  SpO2: {baseline.spo2_mean:.1f} ± {baseline.spo2_std:.1f} %")
            logger.info(f"  SBP: {baseline.sbp_mean:.1f} ± {baseline.sbp_std:.1f} mmHg")
            logger.info(f"  RR: {baseline.rr_mean:.1f} ± {baseline.rr_std:.1f} /min")
            logger.info(f"  Temp: {baseline.temp_mean:.2f} ± {baseline.temp_std:.2f} °C")
            
            return baseline
        
        return None
    
    def sanitize_artifacts(
        self, 
        patient_id: str, 
        vitals_stream: List[List[float]]
    ) -> np.ndarray:
        """
        Remove connection noise and artifacts using Median Absolute Deviation (MAD) filter.
        
        The MAD filter is more robust to outliers than standard deviation:
        MAD = median(|X - median(X)|)
        
        Outliers are defined as values where |X - median| > k × MAD, typically k=3
        
        Args:
            patient_id: Patient identifier (for logging)
            vitals_stream: List of [HR, SpO2, SBP, RR, Temp] readings
        
        Returns:
            Numpy array of sanitized vitals (outliers replaced with median)
        
        Example:
            Input:  [[85, 98, 120, 16, 37.0], [250, 99, 121, 16, 37.1], ...] # 250 is artifact
            Output: [[85, 98, 120, 16, 37.0], [85, 99, 121, 16, 37.1], ...] # 250 → 85 (median)
        """
        vitals_array = np.array(vitals_stream)  # Shape: (timesteps, 5)
        sanitized = vitals_array.copy()
        
        # Feature names for logging
        feature_names = ['HR', 'SpO2', 'SBP', 'RR', 'Temp']
        
        # Apply MAD filter to each vital sign independently
        for i in range(5):
            feature_data = vitals_array[:, i]
            
            # Calculate median and MAD
            median = np.median(feature_data)
            mad = np.median(np.abs(feature_data - median))
            
            # Avoid division by zero (if all values identical)
            if mad == 0:
                mad = 1.0
            
            # Identify outliers (k=3 standard threshold)
            k = 3.0
            z_score = np.abs(feature_data - median) / mad
            outliers = z_score > k
            
            # Replace outliers with median
            if np.any(outliers):
                outlier_count = np.sum(outliers)
                logger.warning(f"Patient {patient_id}: Removed {outlier_count} artifact(s) "
                             f"from {feature_names[i]} (median={median:.1f})")
                sanitized[outliers, i] = median
        
        return sanitized
    
    def project_baseline(
        self, 
        patient_id: str, 
        sanitized_vitals: np.ndarray
    ) -> BaselineMetrics:
        """
        Calculate patient-specific baseline B_0 as mean ± 1.5×std for each vital.
        
        This establishes the "Green Zone" for this patient. Deviations beyond
        1.5 standard deviations from their personal baseline trigger risk assessment.
        
        Args:
            patient_id: Patient identifier
            sanitized_vitals: Clean vitals array (timesteps, 5)
        
        Returns:
            BaselineMetrics object with computed ranges
        
        Example:
            Post-stroke patient with elevated resting HR=90:
            - HR baseline: 90 ± 5 bpm → Green Zone [82.5, 97.5]
            - Normal patient with HR=70:
            - HR baseline: 70 ± 5 bpm → Green Zone [62.5, 77.5]
        """
        # Calculate mean and std for each vital
        means = np.mean(sanitized_vitals, axis=0)
        stds = np.std(sanitized_vitals, axis=0)
        
        # Calculate stability confidence score
        # Low variability (small std) → high confidence
        # High variability → low confidence (unstable baseline)
        # Normalize by typical ranges for each vital
        typical_ranges = np.array([20, 5, 20, 5, 1.0])  # [HR, SpO2, SBP, RR, Temp]
        normalized_stds = stds / typical_ranges
        variability_score = np.mean(normalized_stds)
        
        # Confidence: 1.0 = very stable, 0.0 = very unstable
        stability_confidence = max(0.0, min(1.0, 1.0 - variability_score))
        
        # Create baseline metrics
        baseline = BaselineMetrics(
            patient_id=patient_id,
            hr_mean=float(means[0]),
            hr_std=float(stds[0]),
            spo2_mean=float(means[1]),
            spo2_std=float(stds[1]),
            sbp_mean=float(means[2]),
            sbp_std=float(stds[2]),
            rr_mean=float(means[3]),
            rr_std=float(stds[3]),
            temp_mean=float(means[4]),
            temp_std=float(stds[4]),
            timestamp=datetime.now(),
            stability_confidence=stability_confidence,
            sample_count=len(sanitized_vitals)
        )
        
        return baseline
    
    def get_baseline(self, patient_id: str) -> Optional[BaselineMetrics]:
        """
        Retrieve stored baseline for a patient.
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            BaselineMetrics if exists, None otherwise
        """
        return self.patient_baselines.get(patient_id)
    
    def is_calibrated(self, patient_id: str) -> bool:
        """Check if patient has completed calibration"""
        return patient_id in self.patient_baselines
    
    def get_calibration_progress(self, patient_id: str) -> Tuple[int, int]:
        """
        Get cold-start calibration progress.
        
        Returns:
            (current_samples, required_samples)
        """
        if patient_id in self.cold_start_buffers:
            current = len(self.cold_start_buffers[patient_id])
            return (current, self.min_samples)
        elif patient_id in self.patient_baselines:
            return (self.min_samples, self.min_samples)  # Complete
        else:
            return (0, self.min_samples)  # Not started
    
    def update_baseline(
        self,
        patient_id: str,
        vitals_window: List[List[float]],
        alpha: float = 0.1
    ) -> Optional[BaselineMetrics]:
        """
        Update existing baseline using exponential moving average.
        Should be called every 4 hours during stable periods.
        
        Formula: B_new = α × B_current + (1 - α) × B_old
        
        Args:
            patient_id: Patient identifier
            vitals_window: Recent vitals during stable period
            alpha: EMA smoothing factor (default: 0.1 for slow adaptation)
        
        Returns:
            Updated BaselineMetrics, or None if patient not calibrated
        
        Note:
            This should only be called during stable periods (risk < 0.3 for 30+ min)
            to prevent baseline drift during deterioration or interventions.
        """
        if patient_id not in self.patient_baselines:
            logger.warning(f"Cannot update baseline for uncalibrated patient {patient_id}")
            return None
        
        if len(vitals_window) < self.min_samples:
            logger.warning(f"Insufficient samples for baseline update: {len(vitals_window)} < {self.min_samples}")
            return None
        
        # Get current baseline
        old_baseline = self.patient_baselines[patient_id]
        
        # Sanitize new window
        sanitized = self.sanitize_artifacts(patient_id, vitals_window)
        
        # Calculate new statistics
        new_means = np.mean(sanitized, axis=0)
        new_stds = np.std(sanitized, axis=0)
        
        # Apply exponential moving average
        updated_means = alpha * new_means + (1 - alpha) * np.array([
            old_baseline.hr_mean, old_baseline.spo2_mean, old_baseline.sbp_mean,
            old_baseline.rr_mean, old_baseline.temp_mean
        ])
        
        updated_stds = alpha * new_stds + (1 - alpha) * np.array([
            old_baseline.hr_std, old_baseline.spo2_std, old_baseline.sbp_std,
            old_baseline.rr_std, old_baseline.temp_std
        ])
        
        # Calculate new confidence
        typical_ranges = np.array([20, 5, 20, 5, 1.0])
        normalized_stds = updated_stds / typical_ranges
        variability_score = np.mean(normalized_stds)
        stability_confidence = max(0.0, min(1.0, 1.0 - variability_score))
        
        # Update baseline
        updated_baseline = BaselineMetrics(
            patient_id=patient_id,
            hr_mean=float(updated_means[0]),
            hr_std=float(updated_stds[0]),
            spo2_mean=float(updated_means[1]),
            spo2_std=float(updated_stds[1]),
            sbp_mean=float(updated_means[2]),
            sbp_std=float(updated_stds[2]),
            rr_mean=float(updated_means[3]),
            rr_std=float(updated_stds[3]),
            temp_mean=float(updated_means[4]),
            temp_std=float(updated_stds[4]),
            timestamp=datetime.now(),
            stability_confidence=stability_confidence,
            sample_count=len(sanitized)
        )
        
        self.patient_baselines[patient_id] = updated_baseline
        
        logger.info(f"Patient {patient_id}: Baseline updated (α={alpha}, confidence={stability_confidence:.2f})")
        
        return updated_baseline
    
    def reset_patient(self, patient_id: str):
        """
        Reset calibration for a patient (e.g., on discharge or recalibration request)
        
        Args:
            patient_id: Patient identifier
        """
        if patient_id in self.patient_baselines:
            del self.patient_baselines[patient_id]
            logger.info(f"Patient {patient_id}: Baseline reset")
        
        if patient_id in self.cold_start_buffers:
            del self.cold_start_buffers[patient_id]
    
    def get_all_patients(self) -> List[str]:
        """Get list of all calibrated patients"""
        return list(self.patient_baselines.keys())
