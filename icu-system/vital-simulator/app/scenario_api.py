"""
Developer Tools API - Scenario Control for Presentations
Allows manual triggering of clinical scenarios for demonstration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import threading

logger = logging.getLogger("scenario-api")

class ScenarioControlAPI:
    """REST API for controlling patient scenarios"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for frontend access
        
        # Register routes
        self.app.route('/api/dev/scenarios/trigger', methods=['POST'])(self.trigger_scenario)
        self.app.route('/api/dev/scenarios/reset', methods=['POST'])(self.reset_patient)
        self.app.route('/api/dev/scenarios/status', methods=['GET'])(self.get_status)
        self.app.route('/health', methods=['GET'])(self.health_check)
        
    def trigger_scenario(self):
        """
        Trigger a specific scenario for a patient
        POST /api/dev/scenarios/trigger
        Body: {
            "patient_id": "P001",
            "scenario_type": "sepsis|shock|mild_deterioration|critical",
            "severity": 0.8,  // 0.0-1.0
            "duration": 300   // seconds
        }
        """
        try:
            data = request.get_json()
            patient_id = data.get('patient_id')
            scenario_type = data.get('scenario_type', 'sepsis')
            severity = float(data.get('severity', 0.8))
            duration = int(data.get('duration', 300))
            
            if not patient_id:
                return jsonify({"error": "patient_id required"}), 400
            
            if patient_id not in self.simulator.patients:
                return jsonify({"error": f"Patient {patient_id} not found"}), 404
            
            patient = self.simulator.patients[patient_id]
            
            # Apply scenario based on type
            if scenario_type == 'sepsis':
                self._apply_sepsis_scenario(patient, severity, duration)
            elif scenario_type == 'shock':
                self._apply_shock_scenario(patient, severity, duration)
            elif scenario_type == 'mild_deterioration':
                self._apply_mild_scenario(patient, duration)
            elif scenario_type == 'critical':
                self._apply_critical_scenario(patient)
            elif scenario_type == 'recovery':
                self._apply_recovery_scenario(patient)
            else:
                return jsonify({"error": f"Unknown scenario type: {scenario_type}"}), 400
            
            logger.info(f"Triggered {scenario_type} scenario for {patient_id} (severity={severity}, duration={duration}s)")
            
            return jsonify({
                "status": "success",
                "message": f"Triggered {scenario_type} for {patient_id}",
                "patient_id": patient_id,
                "scenario_type": scenario_type,
                "severity": severity,
                "duration": duration
            })
            
        except Exception as e:
            logger.error(f"Error triggering scenario: {e}")
            return jsonify({"error": str(e)}), 500
    
    def reset_patient(self):
        """
        Reset patient to healthy baseline
        POST /api/dev/scenarios/reset
        Body: {"patient_id": "P001"}
        """
        try:
            data = request.get_json()
            patient_id = data.get('patient_id')
            
            if not patient_id:
                return jsonify({"error": "patient_id required"}), 400
            
            if patient_id not in self.simulator.patients:
                return jsonify({"error": f"Patient {patient_id} not found"}), 404
            
            patient = self.simulator.patients[patient_id]
            
            # Reset to baseline
            patient.last_vitals = patient._initialize_stable_vitals()
            patient.state = patient.PatientState.STABLE if hasattr(patient, 'PatientState') else None
            patient.active_acute_event = None
            patient.acute_event_duration = 0
            patient.scripted_spike_active = False
            
            logger.info(f"Reset {patient_id} to healthy baseline")
            
            return jsonify({
                "status": "success",
                "message": f"Reset {patient_id} to baseline",
                "patient_id": patient_id
            })
            
        except Exception as e:
            logger.error(f"Error resetting patient: {e}")
            return jsonify({"error": str(e)}), 500
    
    def get_status(self):
        """Get status of all patients"""
        try:
            patient_statuses = {}
            for patient_id, patient in self.simulator.patients.items():
                patient_statuses[patient_id] = {
                    "patient_id": patient_id,
                    "hr": round(patient.last_vitals.heart_rate, 1),
                    "sbp": round(patient.last_vitals.systolic_bp, 1),
                    "spo2": round(patient.last_vitals.spo2, 1),
                    "shock_index": round(patient.last_vitals.shock_index, 2),
                    "active_spike": patient.scripted_spike_active if hasattr(patient, 'scripted_spike_active') else False
                }
            
            return jsonify({
                "status": "success",
                "patients": patient_statuses,
                "total_patients": len(patient_statuses)
            })
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return jsonify({"error": str(e)}), 500
    
    def health_check(self):
        """Health check endpoint"""
        return jsonify({"status": "healthy", "service": "vital-simulator-scenario-api"})
    
    # Scenario implementation helpers
    
    def _apply_sepsis_scenario(self, patient, severity, duration):
        """Trigger sepsis: increased HR, decreased BP, elevated lactate markers"""
        # Trigger scripted spike for sepsis pattern
        deltas = {
            "heart_rate": severity * 30.0,  # HR increases significantly
            "systolic_bp": severity * 25.0,  # BP drops
            "spo2": severity * 4.0,  # SpO2 slightly decreases
            "respiratory_rate": severity * 8.0,  # RR increases
            "temperature": severity * 1.2,  # Fever
        }
        
        patient.start_scripted_spike(
            ramp_up_seconds=int(duration * 0.4),  # 40% of duration for ramp up
            hold_seconds=int(duration * 0.3),  # 30% hold at peak
            ramp_down_seconds=int(duration * 0.3),  # 30% recovery
            deltas=deltas
        )
    
    def _apply_shock_scenario(self, patient, severity, duration):
        """Trigger shock: severe tachycardia, severe hypotension"""
        deltas = {
            "heart_rate": severity * 50.0,  # Severe tachycardia
            "systolic_bp": severity * 40.0,  # Severe hypotension
            "spo2": severity * 8.0,  # Significant hypoxia
            "respiratory_rate": severity * 12.0,  # Severe tachypnea
            "temperature": severity * 0.5,  # Hypothermia in shock
        }
        
        patient.start_scripted_spike(
            ramp_up_seconds=int(duration * 0.3),  # Rapid onset
            hold_seconds=int(duration * 0.5),  # Long critical period
            ramp_down_seconds=int(duration * 0.2),  # Slow recovery
            deltas=deltas
        )
    
    def _apply_mild_scenario(self, patient, duration):
        """Trigger mild deterioration"""
        deltas = {
            "heart_rate": 15.0,
            "systolic_bp": 12.0,
            "spo2": 2.0,
            "respiratory_rate": 4.0,
            "temperature": 0.6,
        }
        
        patient.start_scripted_spike(
            ramp_up_seconds=int(duration * 0.5),
            hold_seconds=int(duration * 0.25),
            ramp_down_seconds=int(duration * 0.25),
            deltas=deltas
        )
    
    def _apply_critical_scenario(self, patient):
        """Set patient to critical condition immediately"""
        # Manually set critical vitals
        from main import VitalSigns
        patient.last_vitals = VitalSigns(
            heart_rate=140.0,
            systolic_bp=75.0,
            diastolic_bp=50.0,
            spo2=89.0,
            respiratory_rate=30.0,
            temperature=38.5
        )
        logger.warning(f"WARNING: Set {patient.patient_id} to CRITICAL condition")
    
    def _apply_recovery_scenario(self, patient):
        """Trigger rapid recovery"""
        # Stop any active spikes
        patient.scripted_spike_active = False
        patient.scripted_spike_phase = "idle"
        
        # Reset to baseline with slight variations
        patient.last_vitals = patient._initialize_stable_vitals()
        logger.info(f"Triggered recovery for {patient.patient_id}")
    
    def start(self, port=5001, host='0.0.0.0'):
        """Start the API server in a background thread"""
        def run():
            logger.info(f"Developer Tools API starting on {host}:{port}")
            self.app.run(host=host, port=port, debug=False, use_reloader=False)
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        logger.info("Scenario Control API ready")
