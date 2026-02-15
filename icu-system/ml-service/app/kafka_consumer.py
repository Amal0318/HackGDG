"""
Kafka Consumer for ML Service
Consumes enriched vitals from Pathway and makes real-time predictions
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Deque
import numpy as np

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-kafka-consumer")


class PatientBuffer:
    """Buffer to store patient vital sequences for prediction"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        # Store sequences per patient
        self.buffers: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=sequence_length))
        # Feature extractors
        self.feature_names = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2',
            'respiratory_rate', 'temperature', 'shock_index',
            'rolling_hr', 'rolling_spo2', 'rolling_sbp',
            'hr_trend', 'sbp_trend'
        ]
        
    def add_record(self, patient_id: str, record: dict):
        """Add a vital sign record to the patient's buffer"""
        # Extract features in order
        features = []
        for fname in self.feature_names:
            value = record.get(fname, 0.0)
            features.append(float(value))
        
        # Calculate MAP (Mean Arterial Pressure) if not present
        # MAP = DBP + (SBP - DBP)/3
        map_value = record.get('diastolic_bp', 0) + (record.get('systolic_bp', 0) - record.get('diastolic_bp', 0)) / 3
        features.append(float(map_value))
        
        # Add placeholder features for lab values (glucose, ph, lactate, etc.)
        # These will be set to mean/normal values as we don't have them from simulation
        features.append(100.0)  # glucose (normal ~100 mg/dL)
        
        self.buffers[patient_id].append(features)
        
    def get_sequence(self, patient_id: str):
        """Get the current sequence for a patient"""
        buffer = self.buffers.get(patient_id, deque())
        if len(buffer) < self.sequence_length:
            # Pad with zeros if we don't have enough data yet
            padding_needed = self.sequence_length - len(buffer)
            padded = [[0.0] * len(self.feature_names + ['map', 'glucose'])] * padding_needed
            return np.array(list(padded) + list(buffer), dtype=np.float32)
        return np.array(list(buffer), dtype=np.float32)
    
    def is_ready(self, patient_id: str) -> bool:
        """Check if we have enough data for prediction"""
        return len(self.buffers.get(patient_id, [])) >= self.sequence_length


class MLKafkaConsumer:
    """Kafka consumer for real-time ML predictions"""
    
    def __init__(
        self,
        bootstrap_servers='kafka:9092',
        input_topic='vitals_enriched',
        output_topic='vitals_predictions',
        group_id='ml-service-consumer'
    ):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.group_id = group_id
        
        self.consumer = None
        self.producer = None
        self.patient_buffer = PatientBuffer(sequence_length=60)
        self.running = False
        self.thread = None
        
        # Prediction function (will be set externally)
        self.predict_fn = None
        
        logger.info(f"ML Kafka Consumer initialized for topic '{input_topic}'")
    
    def set_predictor(self, predict_fn):
        """Set the prediction function"""
        self.predict_fn = predict_fn
        
    def connect(self):
        """Connect to Kafka with retry logic for production reliability"""
        import time
        
        max_retries = 10
        retry_delay = 5  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"🔄 Connecting to Kafka at {self.bootstrap_servers} (attempt {attempt}/{max_retries})...")
                
                self.consumer = KafkaConsumer(
                    self.input_topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    consumer_timeout_ms=1000,
                    api_version=(0, 10, 1)  # Compatible version
                )
                
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    api_version=(0, 10, 1)
                )
                
                logger.info(f"✅ Connected to Kafka at {self.bootstrap_servers}")
                logger.info(f"📥 Consuming from topic: {self.input_topic}")
                logger.info(f"📤 Publishing predictions to: {self.output_topic}")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️  Kafka connection attempt {attempt} failed: {e}")
                
                if attempt < max_retries:
                    logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ Failed to connect to Kafka after {max_retries} attempts")
                    return False
        
        return False
    
    def process_message(self, message):
        """Process a single Kafka message"""
        try:
            record = message.value
            patient_id = record.get('patient_id')
            
            if not patient_id:
                logger.warning("Received record without patient_id")
                return
            
            # Add to buffer
            self.patient_buffer.add_record(patient_id, record)
            buffer_size = len(self.patient_buffer.buffers.get(patient_id, []))
            
            # Log buffer status periodically
            if buffer_size % 10 == 0:
                logger.info(f"📊 Buffer for {patient_id}: {buffer_size}/60 records")
            
            # Check if we have enough data for prediction
            if self.patient_buffer.is_ready(patient_id):
                logger.info(f"✅ Buffer ready for {patient_id}, making prediction...")
                # Get sequence
                sequence = self.patient_buffer.get_sequence(patient_id)
                
                # Make prediction if predictor is set
                if self.predict_fn:
                    logger.debug(f"🔮 Calling predictor for {patient_id}...")
                    try:
                        raw_risk_score = float(self.predict_fn(sequence))
                        clinical_floor = self._clinical_risk_floor(record)
                        risk_score = max(raw_risk_score, clinical_floor)

                        if risk_score > raw_risk_score:
                            logger.warning(
                                f"⚠️ Clinical risk override for {patient_id}: "
                                f"raw={raw_risk_score:.4f} → adjusted={risk_score:.4f} "
                                f"(floor={clinical_floor:.4f}, state={record.get('state')}, "
                                f"SI={record.get('shock_index')}, SpO2={record.get('spo2')})"
                            )

                        logger.info(
                            f"🎯 Prediction result for {patient_id}: raw={raw_risk_score:.8f}, final={risk_score:.8f}"
                        )
                        
                        # Create prediction result
                        prediction = {
                            'patient_id': patient_id,
                            'timestamp': record.get('timestamp'),
                            'risk_score': float(risk_score),
                            'risk_level': self._classify_risk(risk_score),
                            'raw_risk_score': float(raw_risk_score),
                            'clinical_risk_floor': float(clinical_floor),
                            'prediction_time': datetime.utcnow().isoformat(),
                            'current_vitals': {
                                'heart_rate': record.get('heart_rate'),
                                'systolic_bp': record.get('systolic_bp'),
                                'spo2': record.get('spo2'),
                                'state': record.get('state')
                            }
                        }
                        
                        # Log prediction with full precision
                        logger.info(
                            f"🔮 Prediction for {patient_id}: "
                            f"Risk={risk_score} (raw: {raw_risk_score:.6f}, floor: {clinical_floor:.6f}) ({prediction['risk_level']}) "
                            f"State={record.get('state')}"
                        )
                        
                        # Publish prediction to Kafka
                        self.producer.send(self.output_topic, value=prediction)
                        
                    except Exception as e:
                        logger.error(f"❌ Prediction error for {patient_id}: {e}", exc_info=True)
                else:
                    logger.warning(f"⚠️ No predictor set! Cannot make prediction for {patient_id}")
                        
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}", exc_info=True)
    
    def _classify_risk(self, risk_score: float) -> str:
        """Classify risk score into categories"""
        if risk_score >= 0.7:
            return "HIGH"
        elif risk_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def _clinical_risk_floor(self, record: dict) -> float:
        """
        Derive a minimum risk score from current physiology.
        This keeps obviously unstable/critical vitals from being labeled LOW.
        """
        def _to_float(value):
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        heart_rate = _to_float(record.get('heart_rate'))
        if heart_rate is None:
            heart_rate = _to_float(record.get('rolling_hr'))

        systolic_bp = _to_float(record.get('systolic_bp'))
        if systolic_bp is None:
            systolic_bp = _to_float(record.get('rolling_sbp'))

        spo2 = _to_float(record.get('spo2'))
        if spo2 is None:
            spo2 = _to_float(record.get('rolling_spo2'))

        shock_index = _to_float(record.get('shock_index'))
        if shock_index is None and heart_rate is not None and systolic_bp is not None and systolic_bp > 0:
            shock_index = heart_rate / systolic_bp

        state = str(record.get('state') or '').upper()

        floor = 0.0

        # State-aware baseline floor
        if state == 'CRITICAL':
            floor = max(floor, 0.80)
        elif state in ('LATE_DETERIORATION', 'INTERVENTION'):
            floor = max(floor, 0.60)
        elif state in ('EARLY_DETERIORATION', 'RECOVERING'):
            floor = max(floor, 0.35)

        # Shock index signal
        if shock_index is not None:
            if shock_index >= 1.3:
                floor = max(floor, 0.85)
            elif shock_index >= 1.1:
                floor = max(floor, 0.65)
            elif shock_index >= 0.95:
                floor = max(floor, 0.40)

        # Oxygenation signal
        if spo2 is not None:
            if spo2 <= 88:
                floor = max(floor, 0.85)
            elif spo2 <= 92:
                floor = max(floor, 0.65)
            elif spo2 <= 94:
                floor = max(floor, 0.40)

        # Circulation / perfusion signal
        if systolic_bp is not None:
            if systolic_bp <= 85:
                floor = max(floor, 0.80)
            elif systolic_bp <= 95:
                floor = max(floor, 0.60)

        # Compensatory tachycardia signal
        if heart_rate is not None:
            if heart_rate >= 130:
                floor = max(floor, 0.70)
            elif heart_rate >= 115:
                floor = max(floor, 0.50)

        # Combined severe pattern
        if (
            shock_index is not None and
            spo2 is not None and
            systolic_bp is not None and
            shock_index >= 1.2 and
            spo2 <= 90 and
            systolic_bp <= 95
        ):
            floor = max(floor, 0.90)

        return min(1.0, max(0.0, floor))
    
    def start(self):
        """Start consuming in a background thread"""
        if self.running:
            logger.warning("Consumer already running")
            return
        
        if not self.connect():
            logger.error("Cannot start consumer - connection failed")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._consume_loop, daemon=True)
        self.thread.start()
        logger.info("🚀 ML Kafka Consumer started in background thread")
    
    def _consume_loop(self):
        """Main consumption loop"""
        logger.info("Starting consumption loop...")
        logger.info(f"Predictor function set: {self.predict_fn is not None}")
        message_count = 0
        
        while self.running:
            try:
                for message in self.consumer:
                    if not self.running:
                        break
                    
                    self.process_message(message)
                    message_count += 1
                    
                    if message_count % 50 == 0:
                        logger.info(f"📨 Processed {message_count} messages so far")
                        
            except Exception as e:
                if self.running:
                    logger.error(f"Error in consumption loop: {e}")
                    time.sleep(1)  # Brief pause before retrying
        
        logger.info("Consumption loop stopped")
    
    def stop(self):
        """Stop the consumer"""
        logger.info("Stopping ML Kafka Consumer...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
        
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        logger.info("✅ ML Kafka Consumer stopped")


# Global consumer instance
ml_consumer = None


def start_ml_consumer(predict_fn):
    """Start the ML Kafka consumer with a prediction function"""
    global ml_consumer
    
    # Get bootstrap servers from environment or use default
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
    
    if ml_consumer is None:
        ml_consumer = MLKafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            input_topic='vitals_enriched',
            output_topic='vitals_predictions'
        )
        ml_consumer.set_predictor(predict_fn)
        ml_consumer.start()
    
    return ml_consumer


def stop_ml_consumer():
    """Stop the ML Kafka consumer"""
    global ml_consumer
    
    if ml_consumer:
        ml_consumer.stop()
        ml_consumer = None
