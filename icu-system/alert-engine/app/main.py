"""
Alert Engine - Threshold-Based Alerting
Triggers alerts when ML risk score exceeds threshold
NO rule-based risk calculation - relies on ML Service only
"""

import json
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Optional

try:
    from kafka import KafkaConsumer, KafkaProducer
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
logger = logging.getLogger("alert-engine")

class AlertEngine:
    """
    Threshold-based alert engine
    Monitors ML predictions and triggers alerts
    """
    
    def __init__(self):
        self.kafka_servers = 'kafka:9092'
        self.input_topic = 'vitals_predictions'
        self.output_topic = 'alerts_stream'
        self.consumer_group = 'alert-engine'
        
        # Alert threshold
        self.alert_threshold = float(os.getenv('ALERT_THRESHOLD', '0.85'))
        
        # Cooldown to prevent alert spam (seconds)
        self.alert_cooldown = 900  # 15 minutes (increased from 5 min)
        self.last_alert_time = {}
        
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        
        self.is_running = False
        self.alerts_triggered = 0
        
    def initialize(self) -> bool:
        """Initialize Kafka connections"""
        
        try:
            logger.info("Initializing Alert Engine")
            logger.info(f"Kafka servers: {self.kafka_servers}")
            logger.info(f"Alert threshold: {self.alert_threshold}")
            logger.info(f"Alert cooldown: {self.alert_cooldown}s")
            
            if not KAFKA_AVAILABLE:
                logger.error("Kafka not available")
                return False
            
            # Initialize consumer
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            # Initialize producer
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
    
    def should_trigger_alert(self, patient_id: str, risk_score: float) -> bool:
        """
        Determine if alert should be triggered
        
        Args:
            patient_id: Patient identifier
            risk_score: ML-predicted risk score
            
        Returns:
            True if alert should be triggered
        """
        
        # Check threshold
        if risk_score <= self.alert_threshold:
            return False
        
        # Check cooldown
        current_time = datetime.now().timestamp()
        last_alert = self.last_alert_time.get(patient_id, 0)
        
        if current_time - last_alert < self.alert_cooldown:
            logger.debug(f"Alert cooldown active for {patient_id}")
            return False
        
        return True
    
    def generate_alert(self, prediction: dict) -> dict:
        """
        Generate alert message
        
        Args:
            prediction: Prediction data from ML Service
            
        Returns:
            Alert message dictionary
        """
        
        patient_id = prediction['patient_id']
        risk_score = prediction['risk_score']
        timestamp = prediction['timestamp']
        
        alert = {
            'alert_id': f"{patient_id}_{int(datetime.now().timestamp())}",
            'patient_id': patient_id,
            'timestamp': timestamp,
            'alert_type': 'HIGH_RISK',
            'risk_score': risk_score,
            'threshold': self.alert_threshold,
            'severity': self._calculate_severity(risk_score),
            'message': f"Patient {patient_id} risk score ({risk_score:.3f}) exceeds threshold ({self.alert_threshold})",
            'generated_at': datetime.now().isoformat()
        }
        
        return alert
    
    def _calculate_severity(self, risk_score: float) -> str:
        """Calculate alert severity based on risk score"""
        
        if risk_score >= 0.90:
            return 'CRITICAL'
        elif risk_score >= 0.80:
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def process_prediction(self, prediction: dict):
        """
        Process prediction and trigger alert if needed
        
        Args:
            prediction: Prediction from ML Service
        """
        
        patient_id = prediction.get('patient_id')
        risk_score = prediction.get('risk_score')
        
        if not patient_id or risk_score is None:
            logger.warning("Invalid prediction data")
            return
        
        # Check if alert should be triggered
        if self.should_trigger_alert(patient_id, risk_score):
            # Generate alert
            alert = self.generate_alert(prediction)
            
            # Publish to alerts topic
            try:
                self.producer.send(self.output_topic, value=alert)
                self.alerts_triggered += 1
                
                # Update last alert time
                self.last_alert_time[patient_id] = datetime.now().timestamp()
                
                logger.warning(
                    f"ALERT TRIGGERED: {patient_id} | "
                    f"Risk={risk_score:.3f} | "
                    f"Severity={alert['severity']}"
                )
                
            except Exception as e:
                logger.error(f"Error publishing alert: {e}")
    
    def run(self):
        """Main processing loop"""
        
        logger.info("Starting Alert Engine main loop")
        logger.info(f"Consuming from: {self.input_topic}")
        logger.info(f"Publishing to: {self.output_topic}")
        logger.info("Alert engine monitors ML predictions ONLY")
        
        self.is_running = True
        
        try:
            for message in self.consumer:
                if not self.is_running:
                    break
                
                try:
                    prediction = message.value
                    self.process_prediction(prediction)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
        
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        
        logger.info("Cleaning up Alert Engine...")
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
        
        if self.consumer:
            self.consumer.close()
        
        logger.info(f"Total alerts triggered: {self.alerts_triggered}")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown"""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.is_running = False
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

import os

def main():
    """Entry point"""
    
    logger.info("=" * 60)
    logger.info("VitalX Alert Engine - Threshold-Based Alerting")
    logger.info("=" * 60)
    
    engine = AlertEngine()
    
    # Setup signal handlers
    engine.setup_signal_handlers()
    
    # Initialize
    if not engine.initialize():
        logger.error("Failed to initialize, exiting")
        sys.exit(1)
    
    # Run
    try:
        engine.run()
    except Exception as e:
        logger.error(f"Engine failed: {e}", exc_info=True)
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
