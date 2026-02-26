"""
VitalX ML Service - Real-time Sepsis Risk Prediction
Uses trained LSTM model on MIMIC-IV dataset
Consumes: vitals_enriched from Pathway
Publishes: vitals_predictions to Kafka
"""

import json
import logging
import os
import sys
import time
from collections import deque, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

# Import VitalX inference module
try:
    from inference import SepsisPredictor
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"WARNING: Could not import SepsisPredictor: {e}")
    MODEL_AVAILABLE = False

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeMLService:
    """
    Real-time ML service that:
    1. Consumes enriched vitals from Pathway
    2. Maintains 24-hour buffer per patient
    3. Generates sepsis risk predictions using trained LSTM
    4. Publishes predictions to Kafka
    """
    
    def __init__(self):
        """Initialize ML service"""
        self.kafka_broker = os.getenv('KAFKA_BROKER', 'kafka:9092')
        self.consume_topic = os.getenv('CONSUME_TOPIC', 'vitals_enriched')
        self.publish_topic = os.getenv('PUBLISH_TOPIC', 'vitals_predictions')
        
        # Per-patient data buffers (24-hour sliding window)
        self.patient_buffers = defaultdict(lambda: deque(maxlen=config.SEQUENCE_LENGTH))
        
        # Initialize predictor
        self.predictor = None
        self.use_trained_model = False
        
        if MODEL_AVAILABLE:
            try:
                logger.info("Initializing VitalX Sepsis Predictor...")
                self.predictor = SepsisPredictor()
                self.use_trained_model = True
                logger.info("Trained LSTM model loaded successfully!")
            except Exception as e:
                logger.error(f"WARNING: Failed to load trained model: {e}")
                logger.info("Continuing with fallback heuristic...")
        else:
            logger.warning("WARNING: Trained model not available, using fallback heuristic")
        
        # Kafka consumer for enriched vitals
        self.consumer = None
        self.producer = None
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'predictions_made': 0,
            'errors': 0,
            'start_time': time.time()
        }
    
    def connect_kafka(self):
        """Initialize Kafka consumer and producer"""
        logger.info(f"Connecting to Kafka broker: {self.kafka_broker}")
        
        # Consumer
        self.consumer = KafkaConsumer(
            self.consume_topic,
            bootstrap_servers=[self.kafka_broker],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='ml-service-group',
            max_poll_records=100,
            consumer_timeout_ms=1000
        )
        
        # Producer
        self.producer = KafkaProducer(
            bootstrap_servers=[self.kafka_broker],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks=1,
            compression_type='gzip'
        )
        
        logger.info(f"Connected to Kafka")
        logger.info(f"   Consuming: {self.consume_topic}")
        logger.info(f"   Publishing: {self.publish_topic}")
    
    def extract_features_from_vitals(self, vitals_data):
        """
        Extract 34 features from enriched vitals data
        
        Args:
            vitals_data: Dict containing vitals from simulator + pathway enrichment
        
        Returns:
            dict: Mapped features ready for model input
        """
        features = {}
        
        # Map simulator fields to model features using config mapping
        for sim_field, model_field in config.SIMULATOR_TO_MODEL_MAPPING.items():
            features[model_field] = vitals_data.get(sim_field, 0.0)
        
        # Add derived features if present (Pathway may compute these)
        for derived_feat in config.DERIVED_FEATURES:
            if derived_feat in vitals_data:
                features[derived_feat] = vitals_data[derived_feat]
        
        # Add missingness indicators if present
        for missing_feat in config.MISSINGNESS_FEATURES:
            if missing_feat in vitals_data:
                features[missing_feat] = vitals_data[missing_feat]
        
        return features
    
    def compute_derived_features(self, features_list):
        """
        Compute derived features for a sequence of readings
        
        Args:
            features_list: List of feature dicts (chronological order)
        
        Returns:
            numpy array: (seq_len, n_features) with all 34 features
        """
        df = pd.DataFrame(features_list)
        
        # Ensure all base features exist
        for feat in config.BASE_FEATURES:
            if feat not in df.columns:
                df[feat] = 0.0
        
        # Compute derived features
        # ShockIndex = HR / SBP
        df['ShockIndex'] = df['HR'] / (df['SBP'] + 1e-6)
        
        # Deltas (change from previous reading)
        df['HR_delta'] = df['HR'].diff().fillna(0)
        df['SBP_delta'] = df['SBP'].diff().fillna(0)
        df['ShockIndex_delta'] = df['ShockIndex'].diff().fillna(0)
        
        # Rolling means (last 6 hours)
        df['RollingMean_HR'] = df['HR'].rolling(window=min(6, len(df)), min_periods=1).mean()
        df['RollingMean_SBP'] = df['SBP'].rolling(window=min(6, len(df)), min_periods=1).mean()
        
        # Compute missingness indicators
        for vital in config.CORE_VITALS:
            df[f'{vital}_missing'] = ((df[vital] == 0) | df[vital].isna()).astype(int)
        
        for lab in config.KEY_LABS:
            df[f'{lab}_missing'] = ((df[lab] == 0) | df[lab].isna()).astype(int)
        
        # Fill any remaining NaN
        df = df.fillna(0)
        
        # Extract features in correct order
        feature_matrix = df[config.ALL_FEATURES].values
        
        return feature_matrix
    
    def process_enriched_event(self, event):
        """
        Process enriched vitals event and generate prediction
        
        Args:
            event: Enriched vitals data from Pathway
        """
        try:
            patient_id = event.get('patient_id')
            if not patient_id:
                logger.warning("Event missing patient_id")
                return
            
            # Extract features from event
            features = self.extract_features_from_vitals(event)
            
            # Add to patient buffer
            self.patient_buffers[patient_id].append(features)
            
            # Check if we have enough data for prediction (24 hours)
            buffer_len = len(self.patient_buffers[patient_id])
            
            if buffer_len >= config.SEQUENCE_LENGTH:
                # Build feature matrix
                features_list = list(self.patient_buffers[patient_id])
                feature_matrix = self.compute_derived_features(features_list)
                
                # Generate prediction
                if self.use_trained_model and self.predictor:
                    try:
                        # Use trained LSTM model
                        risk_score = self.predictor.predict(feature_matrix)
                        # Debug: Log prediction details
                        if self.stats['predictions_made'] % 50 == 0:
                            logger.info(f"DEBUG Patient {patient_id}: Risk={risk_score:.4f}, HR={event.get('heart_rate')}, BP={event.get('systolic_bp')}, Lactate={event.get('lactate')}")
                    except Exception as e:
                        logger.error(f"Model prediction error: {e}, using fallback")
                        risk_score = self.fallback_risk_prediction(event)
                else:
                    # Fallback heuristic
                    risk_score = self.fallback_risk_prediction(event)
                
                # Create prediction event
                prediction = {
                    'patient_id': patient_id,
                    'timestamp': event.get('timestamp', datetime.utcnow().isoformat()),
                    'risk_score': float(risk_score),
                    'model_type': 'LSTM_MIMIC_IV' if self.use_trained_model else 'HEURISTIC',
                    'confidence': 0.95 if self.use_trained_model else 0.7,
                    'buffer_length': buffer_len,
                    'vitals': {
                        'heart_rate': event.get('heart_rate', 0),
                        'spo2': event.get('spo2', 0),
                        'temperature': event.get('temperature', 0),
                        'systolic_bp': event.get('systolic_bp', 0),
                        'respiratory_rate': event.get('respiratory_rate', 0),
                        'lactate': event.get('lactate', 0)
                    }
                }
                
                # Publish prediction
                self.producer.send(self.publish_topic, value=prediction)
                
                self.stats['predictions_made'] += 1
                
                # Log high-risk predictions
                if risk_score > 0.7:
                    logger.warning(
                        f"HIGH RISK: Patient {patient_id} - "
                        f"Risk={risk_score:.3f} "
                        f"(HR={event.get('heart_rate')}, "
                        f"BP={event.get('systolic_bp')}, "
                        f"Lactate={event.get('lactate')})"
                    )
            
            self.stats['events_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            self.stats['errors'] += 1
    
    def fallback_risk_prediction(self, vitals):
        """
        Simple heuristic-based risk prediction (fallback)
        
        Args:
            vitals: Current vitals dictionary
        
        Returns:
            float: Risk score 0-1
        """
        risk = 0.0
        
        # Heart rate
        hr = vitals.get('heart_rate', 70)
        if hr > 100:
            risk += 0.15
        if hr > 120:
            risk += 0.15
        
        # Blood pressure
        sbp = vitals.get('systolic_bp', 120)
        if sbp < 90:
            risk += 0.2
        
        # Oxygen saturation
        spo2 = vitals.get('spo2', 98)
        if spo2 < 92:
            risk += 0.15
        if spo2 < 88:
            risk += 0.15
        
        # Temperature
        temp = vitals.get('temperature', 37.0)
        if temp > 38.0 or temp < 36.0:
            risk += 0.1
        
        # Lactate
        lactate = vitals.get('lactate', 1.0)
        if lactate > 2.0:
            risk += 0.15
        if lactate > 4.0:
            risk += 0.2
        
        # Respiratory rate
        rr = vitals.get('respiratory_rate', 16)
        if rr > 22:
            risk += 0.1
        
        return min(risk, 1.0)
    
    def run(self):
        """Main event loop"""
        logger.info("="*80)
        logger.info("VitalX ML Service - Real-time Sepsis Risk Prediction")
        logger.info("="*80)
        logger.info(f"Model type: {'LSTM (trained on MIMIC-IV)' if self.use_trained_model else 'Fallback heuristic'}")
        logger.info(f"Features: {config.get_feature_count()}")
        logger.info(f"Sequence length: {config.SEQUENCE_LENGTH} hours")
        logger.info("="*80)
        
        # Connect to Kafka
        self.connect_kafka()
        
        logger.info("ML service started, listening for vitals...")
        
        try:
            while True:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        self.process_enriched_event(message.value)
                
                # Log stats every 100 events
                if self.stats['events_processed'] % 100 == 0 and self.stats['events_processed'] > 0:
                    elapsed = time.time() - self.stats['start_time']
                    logger.info(
                        f"Stats: Events={self.stats['events_processed']}, "
                        f"Predictions={self.stats['predictions_made']}, "
                        f"Errors={self.stats['errors']}, "
                        f"Uptime={elapsed:.1f}s"
                    )
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()


def main():
    """Entry point"""
    logger.info("Starting VitalX ML Service")
    logger.info("Trained LSTM Model on MIMIC-IV Dataset")
    
    try:
        service = RealtimeMLService()
        service.run()
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
