"""
Kafka Consumer for real-time patient data
Consumes from vitals_enriched topic and maintains in-memory state
"""
import asyncio
import json
import logging
from typing import Dict, Optional
try:
    from confluent_kafka import Consumer, KafkaError
    CONFLUENT_KAFKA_AVAILABLE = True
except ImportError:
    CONFLUENT_KAFKA_AVAILABLE = False
    Consumer = None
    KafkaError = None
from .config import settings

logger = logging.getLogger("kafka-consumer")

class PatientDataStore:
    """In-memory store for live patient data"""
    
    def __init__(self):
        self.patients: Dict[str, dict] = {}
        self.predictions: Dict[str, dict] = {}  # Store ML predictions separately
        self.last_update: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._update_callback = None  # WebSocket notification callback
    
    def set_update_callback(self, callback):
        """Set callback function to notify listeners of data updates"""
        self._update_callback = callback
    
    async def _notify_update(self, patient_id: str, data: dict):
        """Notify listeners of patient data update"""
        if self._update_callback:
            try:
                await self._update_callback(patient_id, data)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
    
    async def update_patient(self, patient_id: str, data: dict):
        """Update patient data"""
        async with self._lock:
            # Merge with existing predictions if available
            if patient_id in self.predictions:
                data.update(self.predictions[patient_id])
            self.patients[patient_id] = data
            self.last_update[patient_id] = asyncio.get_event_loop().time()
            
        # Notify WebSocket clients outside the lock
        await self._notify_update(patient_id, data)
    
    async def update_prediction(self, patient_id: str, prediction: dict):
        """Update ML prediction for a patient"""
        updated_data = None
        async with self._lock:
            self.predictions[patient_id] = prediction
            # Merge with existing patient data if available
            if patient_id in self.patients:
                adjusted_prediction = self._apply_clinical_risk_floor(
                    self.patients[patient_id],
                    prediction
                )
                self.patients[patient_id].update(adjusted_prediction)
                updated_data = self.patients[patient_id].copy()
        
        # Notify WebSocket clients if we have complete patient data
        if updated_data:
            await self._notify_update(patient_id, updated_data)

    def _apply_clinical_risk_floor(self, patient: dict, prediction: dict) -> dict:
        """
        Apply a display-side clinical floor to avoid LOW labels during clear instability.
        This acts as a final guardrail if upstream model output is inconsistent.
        """
        merged = {**patient, **prediction}

        def _to_float(value):
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        state = str(merged.get('state') or '').upper()
        heart_rate = _to_float(merged.get('heart_rate'))
        if heart_rate is None:
            heart_rate = _to_float(merged.get('rolling_hr'))

        systolic_bp = _to_float(merged.get('systolic_bp'))
        if systolic_bp is None:
            systolic_bp = _to_float(merged.get('rolling_sbp'))

        spo2 = _to_float(merged.get('spo2'))
        if spo2 is None:
            spo2 = _to_float(merged.get('rolling_spo2'))

        shock_index = _to_float(merged.get('shock_index'))
        if shock_index is None and heart_rate is not None and systolic_bp is not None and systolic_bp > 0:
            shock_index = heart_rate / systolic_bp

        floor = 0.0

        if state == 'CRITICAL':
            floor = max(floor, 0.80)
        elif state in ('INTERVENTION', 'LATE_DETERIORATION'):
            floor = max(floor, 0.60)
        elif state in ('EARLY_DETERIORATION', 'RECOVERING'):
            floor = max(floor, 0.35)

        if shock_index is not None:
            if shock_index >= 1.3:
                floor = max(floor, 0.85)
            elif shock_index >= 1.1:
                floor = max(floor, 0.65)

        if spo2 is not None:
            if spo2 <= 88:
                floor = max(floor, 0.85)
            elif spo2 <= 92:
                floor = max(floor, 0.65)

        if systolic_bp is not None:
            if systolic_bp <= 85:
                floor = max(floor, 0.80)
            elif systolic_bp <= 95:
                floor = max(floor, 0.60)

        if heart_rate is not None:
            if heart_rate >= 130:
                floor = max(floor, 0.70)
            elif heart_rate >= 115:
                floor = max(floor, 0.50)

        raw_score = _to_float(prediction.get('risk_score'))
        if raw_score is None:
            raw_score = _to_float(prediction.get('computed_risk'))
        if raw_score is None:
            return prediction

        adjusted_score = max(raw_score, floor)

        if adjusted_score >= 0.7:
            adjusted_level = 'HIGH'
        elif adjusted_score >= 0.3:
            adjusted_level = 'MEDIUM'
        else:
            adjusted_level = 'LOW'

        adjusted = dict(prediction)
        adjusted['risk_score'] = adjusted_score
        adjusted['computed_risk'] = adjusted_score
        adjusted['risk_level'] = adjusted_level
        adjusted['is_high_risk'] = adjusted_score > 0.7

        return adjusted
    
    async def get_patient(self, patient_id: str) -> Optional[dict]:
        """Get single patient data"""
        async with self._lock:
            return self.patients.get(patient_id)
    
    async def get_all_patients(self) -> Dict[str, dict]:
        """Get all patients"""
        async with self._lock:
            return self.patients.copy()
    
    async def get_patients_by_floor(self, floor_id: str) -> Dict[str, dict]:
        """Get patients for specific floor"""
        async with self._lock:
            return {
                pid: data for pid, data in self.patients.items()
                if data.get("floor_id") == floor_id
            }
    
    def get_patient_count(self) -> int:
        """Get total patient count"""
        return len(self.patients)


class KafkaConsumerService:
    """Background Kafka consumer service"""
    
    def __init__(self, data_store: PatientDataStore):
        self.data_store = data_store
        self.vitals_consumer: Optional[Consumer] = None
        self.predictions_consumer: Optional[Consumer] = None
        self.running = False
        self._vitals_task: Optional[asyncio.Task] = None
        self._predictions_task: Optional[asyncio.Task] = None
    
    def _create_consumer(self, group_id: str):
        """Create Kafka consumer"""
        if not CONFLUENT_KAFKA_AVAILABLE:
            raise RuntimeError("confluent_kafka is not installed")
        conf = {
            'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
            'group.id': group_id,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
        }
        return Consumer(conf)
    
    async def start(self):
        """Start consuming from Kafka"""
        if not CONFLUENT_KAFKA_AVAILABLE:
            raise RuntimeError("confluent_kafka package is not installed (requirements.txt has kafka-python instead)")
        if self.running:
            logger.warning("Kafka consumer already running")
            return
        
        self.running = True
        # Start both consumers
        self._vitals_task = asyncio.create_task(self._consume_vitals_loop())
        self._predictions_task = asyncio.create_task(self._consume_predictions_loop())
        logger.info(f"Kafka consumers started for topics: {settings.KAFKA_TOPIC_ENRICHED} and vitals_predictions")
    
    async def stop(self):
        """Stop consuming"""
        self.running = False
        
        # Cancel tasks
        for task in [self._vitals_task, self._predictions_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close consumers
        if self.vitals_consumer:
            self.vitals_consumer.close()
        if self.predictions_consumer:
            self.predictions_consumer.close()
        
        logger.info("Kafka consumers stopped")
    
    async def _consume_vitals_loop(self):
        """Main consumption loop for vitals_enriched topic"""
        try:
            self.vitals_consumer = self._create_consumer(settings.KAFKA_CONSUMER_GROUP)
            self.vitals_consumer.subscribe([settings.KAFKA_TOPIC_ENRICHED])
            logger.info(f"Subscribed to topic: {settings.KAFKA_TOPIC_ENRICHED}")
            
            while self.running:
                # Poll for messages (non-blocking with timeout)
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, self.vitals_consumer.poll, 1.0
                )
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        continue
                
                # Process message
                try:
                    value = json.loads(msg.value().decode('utf-8'))
                    patient_id = value.get('patient_id')
                    
                    if patient_id:
                        # Assign floor based on patient_id prefix (P1-xxx = Floor 1F, etc.)
                        floor_id = self._assign_floor(patient_id)
                        value['floor_id'] = floor_id
                        
                        # Update data store
                        await self.data_store.update_patient(patient_id, value)
                        
                        logger.info(f"📍 Updated patient {patient_id} on floor {floor_id}")
                
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode message: {msg.value()}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        except Exception as e:
            logger.error(f"Kafka vitals consumer loop error: {e}")
            self.running = False
    
    async def _consume_predictions_loop(self):
        """Consumption loop for vitals_predictions topic"""
        try:
            self.predictions_consumer = self._create_consumer(f"{settings.KAFKA_CONSUMER_GROUP}-predictions")
            self.predictions_consumer.subscribe(['vitals_predictions'])
            logger.info("Subscribed to topic: vitals_predictions")
            
            while self.running:
                # Poll for messages
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, self.predictions_consumer.poll, 1.0
                )
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka predictions error: {msg.error()}")
                        continue
                
                # Process prediction message
                try:
                    prediction = json.loads(msg.value().decode('utf-8'))
                    patient_id = prediction.get('patient_id')
                    
                    if patient_id:
                        # Extract relevant prediction fields
                        pred_data = {
                            'risk_score': prediction.get('risk_score'),
                            'risk_level': prediction.get('risk_level'),
                            'prediction_time': prediction.get('prediction_time'),
                            'computed_risk': prediction.get('risk_score'),  # Alias for frontend
                            'is_high_risk': prediction.get('risk_score', 0) > 0.7
                        }
                        
                        # Update data store with prediction
                        await self.data_store.update_prediction(patient_id, pred_data)
                        
                        logger.debug(f"Updated prediction for {patient_id}: {pred_data['risk_level']}")
                
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode prediction message: {msg.value()}")
                except Exception as e:
                    logger.error(f"Error processing prediction: {e}")
        
        except Exception as e:
            logger.error(f"Kafka predictions consumer loop error: {e}")
            self.running = False
    
    def _assign_floor(self, patient_id: str) -> str:
        """Assign floor based on patient_id pattern - distribute across 3 floors"""
        # Expected pattern: P1, P2, P3, etc.
        try:
            if patient_id.startswith('P'):
                # Extract patient number
                patient_num = int(patient_id[1:]) if patient_id[1:].isdigit() else int(patient_id[1])
                # Distribute across 3 floors using modulo (1-3)
                floor_num = ((patient_num - 1) % 3) + 1
                return f"{floor_num}F"
        except (IndexError, ValueError):
            pass
        
        # Default to 1F if pattern doesn't match
        return "1F"


# Global data store instance
patient_data_store = PatientDataStore()
kafka_consumer_service = KafkaConsumerService(patient_data_store)


# =========================================================
# MOCK DATA SIMULATOR (fallback when Kafka is unavailable)
# =========================================================

import random
import math
from datetime import datetime

_MOCK_PATIENT_BASELINES = [
    {"patient_id": "P1", "name": "James Wilson",    "floor_id": "1F", "bed_number": "1A", "age": 58, "_base_hr": 78,  "_base_sbp": 124, "_base_spo2": 97, "_base_rr": 15, "_base_temp": 37.1, "_state": "STABLE"},
    {"patient_id": "P2", "name": "Maria Garcia",    "floor_id": "2F", "bed_number": "2A", "age": 72, "_base_hr": 88,  "_base_sbp": 108, "_base_spo2": 94, "_base_rr": 19, "_base_temp": 37.6, "_state": "EARLY_DETERIORATION"},
    {"patient_id": "P3", "name": "Robert Chen",     "floor_id": "3F", "bed_number": "3A", "age": 45, "_base_hr": 105, "_base_sbp": 92,  "_base_spo2": 91, "_base_rr": 24, "_base_temp": 38.4, "_state": "LATE_DETERIORATION"},
    {"patient_id": "P4", "name": "Sarah Johnson",   "floor_id": "1F", "bed_number": "1B", "age": 63, "_base_hr": 72,  "_base_sbp": 130, "_base_spo2": 98, "_base_rr": 14, "_base_temp": 36.8, "_state": "STABLE"},
    {"patient_id": "P5", "name": "Michael Brown",   "floor_id": "2F", "bed_number": "2B", "age": 51, "_base_hr": 118, "_base_sbp": 88,  "_base_spo2": 89, "_base_rr": 28, "_base_temp": 38.9, "_state": "CRITICAL"},
    {"patient_id": "P6", "name": "Emily Davis",     "floor_id": "3F", "bed_number": "3B", "age": 66, "_base_hr": 82,  "_base_sbp": 118, "_base_spo2": 96, "_base_rr": 17, "_base_temp": 37.3, "_state": "RECOVERING"},
    {"patient_id": "P7", "name": "David Martinez",  "floor_id": "1F", "bed_number": "1C", "age": 79, "_base_hr": 95,  "_base_sbp": 100, "_base_spo2": 93, "_base_rr": 22, "_base_temp": 38.1, "_state": "EARLY_DETERIORATION"},
    {"patient_id": "P8", "name": "Lisa Thompson",   "floor_id": "2F", "bed_number": "2C", "age": 44, "_base_hr": 68,  "_base_sbp": 135, "_base_spo2": 99, "_base_rr": 13, "_base_temp": 36.7, "_state": "STABLE"},
]

_STATE_RISK = {
    "STABLE":              0.12,
    "RECOVERING":          0.30,
    "EARLY_DETERIORATION": 0.52,
    "LATE_DETERIORATION":  0.73,
    "CRITICAL":            0.88,
    "INTERVENTION":        0.65,
}

def _build_patient_record(base: dict, tick: int) -> dict:
    """Generate a realistic patient record with gentle oscillating vitals."""
    phase = tick * 0.05  # slow drift
    noise = lambda std: random.gauss(0, std)

    hr  = base["_base_hr"]  + 3 * math.sin(phase) + noise(1.5)
    sbp = base["_base_sbp"] + 4 * math.cos(phase) + noise(2.0)
    dbp = sbp * 0.62 + noise(1.5)
    spo2 = min(100, base["_base_spo2"] + 0.5 * math.sin(phase + 1) + noise(0.4))
    rr   = base["_base_rr"]   + 1.5 * math.sin(phase + 2) + noise(0.8)
    temp = base["_base_temp"] + 0.1 * math.sin(phase * 0.3) + noise(0.05)

    risk = _STATE_RISK.get(base["_state"], 0.1) + random.uniform(-0.03, 0.03)
    risk = max(0.0, min(1.0, risk))

    if risk >= 0.7:
        risk_level, is_high = "HIGH", True
    elif risk >= 0.3:
        risk_level, is_high = "MEDIUM", False
    else:
        risk_level, is_high = "LOW", False

    shock_index = hr / max(sbp, 1.0)

    return {
        "patient_id":     base["patient_id"],
        "name":           base["name"],
        "bed_number":     base["bed_number"],
        "floor_id":       base["floor_id"],
        "age":            base["age"],
        "state":          base["_state"],
        "timestamp":      datetime.now().isoformat(),
        # flat vitals (what backend API reads directly)
        "heart_rate":     round(hr, 1),
        "systolic_bp":    round(sbp, 1),
        "diastolic_bp":   round(dbp, 1),
        "spo2":           round(spo2, 1),
        "respiratory_rate": round(rr, 1),
        "temperature":    round(temp, 2),
        "shock_index":    round(shock_index, 3),
        # risk
        "risk_score":     round(risk, 4),
        "computed_risk":  round(risk, 4),
        "risk_level":     risk_level,
        "is_high_risk":   is_high,
        "anomaly_flag":   1 if is_high else 0,
        # features (for frontend anomaly badge logic)
        "features": {
            "anomaly_flag":        is_high,
            "hr_anomaly":          hr > 110 or hr < 50,
            "sbp_anomaly":         sbp < 90 or sbp > 160,
            "spo2_anomaly":        spo2 < 92,
            "shock_index_anomaly": shock_index > 1.0,
            "lactate_anomaly":     False,
        },
    }


async def run_mock_simulator(store: PatientDataStore):
    """Background task: update mock patients every 2 seconds."""
    tick = 0
    logger.info("🔄 Mock patient simulator started (Kafka unavailable)")
    # seed all at tick=0 first
    for base in _MOCK_PATIENT_BASELINES:
        record = _build_patient_record(base, tick)
        await store.update_patient(base["patient_id"], record)
    while True:
        await asyncio.sleep(2)
        tick += 1
        for base in _MOCK_PATIENT_BASELINES:
            record = _build_patient_record(base, tick)
            await store.update_patient(base["patient_id"], record)
