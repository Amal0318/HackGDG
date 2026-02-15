"""
Alert System Main Application
LangChain-powered emergency notification system for ICU patients
"""
import logging
import signal
import sys
import json
import time
from typing import Dict, Optional
from datetime import datetime
from confluent_kafka import Consumer, KafkaError

from .config import settings
from .langchain_agent import MedicalAlertAgent
from .notification_service import NotificationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alert-system")


class AlertSystem:
    """
    Main alert system orchestrator
    Monitors Kafka, uses LangChain for alert generation, sends notifications
    """
    
    def __init__(self):
        self.running = False
        self.consumer: Optional[Consumer] = None
        self.alert_agent: Optional[MedicalAlertAgent] = None
        self.notification_service: Optional[NotificationService] = None
        self.last_alert_time: Dict[str, float] = {}  # Rate limiting
    
    def initialize(self):
        """Initialize all components"""
        try:
            logger.info("=" * 80)
            logger.info(f"🚨 {settings.SERVICE_NAME} v{settings.VERSION}")
            logger.info("=" * 80)
            logger.info("Initializing components...")
            
            # Initialize LangChain agent
            logger.info(f"🧠 Initializing LangChain with {settings.LLM_PROVIDER.upper()}...")
            self.alert_agent = MedicalAlertAgent()
            
            # Initialize notification service
            logger.info("📬 Initializing notification service...")
            self.notification_service = NotificationService()
            
            # Initialize Kafka consumer
            logger.info(f"📡 Connecting to Kafka: {settings.KAFKA_BOOTSTRAP_SERVERS}")
            self.consumer = self._create_kafka_consumer()
            self.consumer.subscribe([settings.KAFKA_TOPIC])
            
            logger.info("=" * 80)
            logger.info("✅ All components initialized successfully")
            logger.info(f"📊 Monitoring topic: {settings.KAFKA_TOPIC}")
            logger.info(f"🎯 High-risk threshold: {settings.HIGH_RISK_THRESHOLD}")
            logger.info(f"⏱️  Alert rate limit: {settings.MIN_ALERT_INTERVAL_SECONDS}s")
            logger.info("=" * 80)
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def _create_kafka_consumer(self) -> Consumer:
        """Create Kafka consumer"""
        conf = {
            'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
            'group.id': settings.KAFKA_GROUP_ID,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
        }
        return Consumer(conf)
    
    def start(self):
        """Start the alert system"""
        if not self.initialize():
            logger.error("Failed to initialize. Exiting.")
            sys.exit(1)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        logger.info("🚀 Alert System started - monitoring for high-risk patients")
        logger.info("Press Ctrl+C to stop\n")
        
        # Main monitoring loop
        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        continue
                
                # Process message
                self._process_patient_data(msg)
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
        finally:
            self.stop()
    
    def _process_patient_data(self, msg):
        """Process incoming patient data and generate alerts if needed"""
        try:
            # Parse message
            data = json.loads(msg.value().decode('utf-8'))
            patient_id = data.get('patient_id')
            
            if not patient_id:
                return
            
            # Check rate limiting (avoid alert spam)
            if not self._should_send_alert(patient_id):
                return
            
            # Generate alert using LangChain
            alert = self.alert_agent.generate_alert(data)
            
            if alert:
                # Update last alert time
                self.last_alert_time[patient_id] = time.time()
                
                # Send notifications
                self.notification_service.send_alert(alert)
                
                logger.info(f"🚨 Alert triggered for {patient_id} on {data.get('floor_id')}")
        
        except json.JSONDecodeError:
            logger.error(f"Failed to decode message: {msg.value()}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _should_send_alert(self, patient_id: str) -> bool:
        """Check if alert should be sent (rate limiting)"""
        last_alert = self.last_alert_time.get(patient_id, 0)
        elapsed = time.time() - last_alert
        
        if elapsed < settings.MIN_ALERT_INTERVAL_SECONDS:
            logger.debug(f"⏸️  Rate limit active for {patient_id} ({elapsed:.0f}s / {settings.MIN_ALERT_INTERVAL_SECONDS}s)")
            return False
        
        return True
    
    def stop(self):
        """Stop the alert system gracefully"""
        logger.info("\n" + "=" * 80)
        logger.info("🛑 Shutting down Alert System...")
        logger.info("=" * 80)
        
        self.running = False
        
        if self.consumer:
            self.consumer.close()
            logger.info("✅ Kafka consumer closed")
        
        # Print statistics
        if self.notification_service:
            stats = self.notification_service.get_alert_statistics()
            logger.info(f"📊 Session Statistics:")
            logger.info(f"   Total alerts sent: {stats['total_alerts']}")
            logger.info(f"   By severity: {stats['by_severity']}")
            logger.info(f"   By floor: {stats['by_floor']}")
        
        logger.info("=" * 80)
        logger.info("✅ Alert System shutdown complete")
        logger.info("=" * 80)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"\n⚠️  Received signal {signum}")
        self.running = False


def main():
    """Main entry point"""
    try:
        alert_system = AlertSystem()
        alert_system.start()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
