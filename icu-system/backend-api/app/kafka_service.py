"""
Kafka Consumer for real-time patient data
Consumes from vitals_enriched topic and maintains in-memory state
"""
import asyncio
import json
import logging
from typing import Dict, Optional
from confluent_kafka import Consumer, KafkaError
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
    
    def _create_consumer(self, group_id: str) -> Consumer:
        """Create Kafka consumer"""
        conf = {
            'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
            'group.id': group_id,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
        }
        return Consumer(conf)
    
    async def start(self):
        """Start consuming from Kafka"""
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
