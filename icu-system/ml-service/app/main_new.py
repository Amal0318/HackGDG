"""
ML Service - Sole Risk Authority
Consumes enriched features, produces risk predictions
"""

import json
import logging
import signal
import sys
from collections import deque, defaultdict
from typing import Dict, Optional
import numpy as np

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError, NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    logging.warning("Kafka not available")
    KAFKA_AVAILABLE = False
    class KafkaConsumer:
        def __init__(self, *args, **kwargs):
            pass
        def __iter__(self):
            return iter([])
    class KafkaProducer:
        def __init__(self, *args, **kwargs):
            pass
        def send(self, *args, **kwargs):
            pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml-service")

class PatientSequenceBuffer:
    """
    Maintain rolling sequence buffer per patient
    Stores feature vectors for time-series prediction
    """
    
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.buffers = defaultdict(lambda: deque(maxlen=window_size))
        
    def add(self, patient_id: str, features: dict):
        """
        Add feature vector to patient's buffer
        
        Args:
            patient_id: Patient identifier
            features: Enriched features dictionary
        """
        
        # Extract feature vector (deterministic order)
        feature_vector = [
            features.get('heart_rate', 0),
            features.get('systolic_bp', 0),
            features.get('diastolic_bp', 0),
            features.get('spo2', 0),
            features.get('shock_index', 0),
            features.get('lactate', 0),
            features.get('rolling_mean_hr', 0),
            features.get('rolling_mean_sbp', 0),
            features.get('rolling_mean_spo2', 0),
            features.get('hr_delta', 0),
            features.get('sbp_delta', 0),
            features.get('shock_index_delta', 0),
            features.get('lactate_delta', 0),
            features.get('rolling_mean_shock_index', 0),
            1.0 if features.get('anomaly_flag', False) else 0.0
        ]
        
        self.buffers[patient_id].append(feature_vector)
    
    def is_ready(self, patient_id: str) -> bool:
        """Check if buffer has enough data for prediction"""
        return len(self.buffers[patient_id]) >= self.window_size
    
    def get_sequence(self, patient_id: str) -> np.ndarray:
        """Get full sequence as numpy array"""
        return np.array(list(self.buffers[patient_id]))
    
    def get_buffer_status(self) -> Dict:
        """Get status of all buffers"""
        return {
            patient_id: len(buffer)
            for patient_id, buffer in self.buffers.items()
        }

def predict_risk_placeholder(sequence: np.ndarray) -> float:
    """
    Placeholder risk prediction function
    
    In production, this would load a trained LSTM model.
    For now, we use a weighted heuristic based on features.
    
    Args:
        sequence: [window_size, num_features] array
        
    Returns:
        risk_score: Float between 0.0 and 1.0
    """
    
    # Extract key features from sequence
    # Indices based on feature_vector order in PatientSequenceBuffer
    heart_rates = sequence[:, 0]
    systolic_bps = sequence[:, 1]
    spo2s = sequence[:, 3]
    shock_indices = sequence[:, 4]
    lactates = sequence[:, 5]
    anomaly_flags = sequence[:, -1]
    
    # Compute weighted risk components
    
    # 1. Shock index trend (40% weight)
    mean_shock_index = np.mean(shock_indices[-10:])  # Last 10 readings
    shock_risk = np.clip((mean_shock_index - 0.5) / 1.5, 0.0, 1.0)
    
    # 2. SpO2 deterioration (25% weight)
    mean_spo2 = np.mean(spo2s[-10:])
    spo2_risk = np.clip((98 - mean_spo2) / 18, 0.0, 1.0)
    
    # 3. Lactate elevation (20% weight)
    mean_lactate = np.mean(lactates[-10:])
    lactate_risk = np.clip((mean_lactate - 1.0) / 3.0, 0.0, 1.0)
    
    # 4. Blood pressure deterioration (10% weight)
    mean_sbp = np.mean(systolic_bps[-10:])
    bp_risk = np.clip((110 - mean_sbp) / 40, 0.0, 1.0)
    
    # 5. Anomaly frequency (5% weight)
    anomaly_rate = np.mean(anomaly_flags[-20:])  # Last 20 readings
    anomaly_risk = anomaly_rate
    
    # Weighted combination
    risk_score = (
        0.40 * shock_risk +
        0.25 * spo2_risk +
        0.20 * lactate_risk +
        0.10 * bp_risk +
        0.05 * anomaly_risk
    )
    
    # Sigmoid-like smoothing
    risk_score = 1.0 / (1.0 + np.exp(-5.0 * (risk_score - 0.5)))
    
    return float(np.clip(risk_score, 0.0, 1.0))

class MLService:
    """
    ML Service - Sole risk authority
    Consumes enriched features, produces risk predictions
    """
    
    def __init__(self):
        self.kafka_servers = 'kafka:9092'
        self.input_topic = 'vitals_enriched'
        self.output_topic = 'vitals_predictions'
        self.consumer_group = 'ml-service'
        
        self.sequence_buffer = PatientSequenceBuffer(window_size=60)
        
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        
        self.is_running = False
        self.predictions_made = 0
        
    def initialize(self) -> bool:
        """Initialize Kafka consumer and producer"""
        
        try:
            logger.info("Initializing ML Service")
            logger.info(f"Kafka servers: {self.kafka_servers}")
            logger.info(f"Input topic: {self.input_topic}")
            logger.info(f"Output topic: {self.output_topic}")
            
            if not KAFKA_AVAILABLE:
                logger.error("Kafka library not available")
                return False
            
            # Initialize consumer
            logger.info("Creating Kafka consumer...")
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                max_poll_records=10
            )
            
            # Initialize producer
            logger.info("Creating Kafka producer...")
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            
            logger.info("Kafka connections established")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def process_enriched_event(self, enriched_data: dict):
        """
        Process enriched event and generate risk prediction if ready
        
        Args:
            enriched_data: Enriched features from Pathway
        """
        
        patient_id = enriched_data.get('patient_id')
        if not patient_id:
            logger.warning("Received event without patient_id")
            return
        
        # Add to sequence buffer
        self.sequence_buffer.add(patient_id, enriched_data)
        
        # Check if buffer is ready for prediction
        if self.sequence_buffer.is_ready(patient_id):
            # Get sequence
            sequence = self.sequence_buffer.get_sequence(patient_id)
            
            # Predict risk score
            risk_score = predict_risk_placeholder(sequence)
            
            # Create prediction message
            prediction = {
                'patient_id': patient_id,
                'timestamp': enriched_data.get('timestamp'),
                'risk_score': round(risk_score, 4)
            }
            
            # Publish to Kafka
            try:
                self.producer.send(self.output_topic, value=prediction)
                self.predictions_made += 1
                
                # Log every 60 predictions
                if self.predictions_made % 60 == 0:
                    logger.info(f"Predictions made: {self.predictions_made}")
                    logger.debug(f"Latest: {patient_id} risk={risk_score:.3f}")
                    
            except Exception as e:
                logger.error(f"Error publishing prediction: {e}")
    
    def run(self):
        """Main processing loop"""
        
        logger.info("Starting ML Service main loop")
        logger.info("Consuming from: vitals_enriched")
        logger.info("Publishing to: vitals_predictions")
        logger.info("ML Service is the SOLE risk authority")
        
        self.is_running = True
        iteration = 0
        
        try:
            for message in self.consumer:
                if not self.is_running:
                    break
                
                iteration += 1
                
                try:
                    enriched_data = message.value
                    self.process_enriched_event(enriched_data)
                    
                    # Log buffer status every 300 iterations (~5 minutes at 1Hz per patient)
                    if iteration % 300 == 0:
                        buffer_status = self.sequence_buffer.get_buffer_status()
                        logger.info(f"Buffer status: {buffer_status}")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    continue
        
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        
        logger.info("Cleaning up ML Service...")
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")
        
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        
        logger.info(f"Total predictions made: {self.predictions_made}")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown"""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.is_running = False
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

def main():
    """Entry point"""
    
    logger.info("=" * 60)
    logger.info("VitalX ML Service - Risk Prediction Authority")
    logger.info("=" * 60)
    
    service = MLService()
    
    # Setup signal handlers
    service.setup_signal_handlers()
    
    # Initialize
    if not service.initialize():
        logger.error("Failed to initialize, exiting")
        sys.exit(1)
    
    # Run main loop
    try:
        service.run()
    except Exception as e:
        logger.error(f"Service failed: {e}", exc_info=True)
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
