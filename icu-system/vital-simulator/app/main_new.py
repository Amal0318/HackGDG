"""
Vital Simulator Service - Streaming-First Implementation
Produces realistic physiological data using drift model (no state machine)
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timezone
from typing import Dict, Optional

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError, NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    logging.warning("Kafka library not available")
    KAFKA_AVAILABLE = False
    class KafkaProducer:
        def __init__(self, *args, **kwargs):
            pass
        def send(self, topic, value):
            pass
        def flush(self):
            pass
        def close(self):
            pass

from drift_model import PhysiologicalDriftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vital-simulator")

class VitalSimulator:
    """
    Clean vital signs simulator using physiological drift model
    No state machines, no rule-based transitions
    """
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.kafka_topic = 'vitals_raw'
        self.num_patients = int(os.getenv('NUM_PATIENTS', '8'))
        self.update_interval = float(os.getenv('UPDATE_INTERVAL', '1.0'))
        
        # Initialize drift models for each patient
        self.patients = {
            f'P{str(i).zfill(3)}': PhysiologicalDriftModel(f'P{str(i).zfill(3)}')
            for i in range(1, self.num_patients + 1)
        }
        
        # Initialize Kafka producer
        self.producer = None
        self._setup_kafka_producer()
        
        logger.info(f"Initialized simulator for {self.num_patients} patients")
        logger.info(f"Publishing to topic: {self.kafka_topic}")
        logger.info(f"Kafka servers: {self.kafka_servers}")
        
    def _setup_kafka_producer(self) -> Optional[KafkaProducer]:
        """Initialize Kafka producer with retry logic"""
        
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available, running in mock mode")
            return None
            
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.kafka_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all',
                    retries=3,
                    max_in_flight_requests_per_connection=1,
                    compression_type='gzip'
                )
                logger.info("Kafka producer initialized successfully")
                return self.producer
                
            except NoBrokersAvailable:
                logger.warning(f"Kafka broker not available (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                else:
                    logger.error("Failed to connect to Kafka after max retries")
                    
            except Exception as e:
                logger.error(f"Error initializing Kafka producer: {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
        
        return None
    
    async def run(self):
        """Main simulation loop"""
        
        logger.info("Starting simulation loop")
        logger.info(f"Update interval: {self.update_interval}s per patient")
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                
                # Update all patients
                for patient_id, model in self.patients.items():
                    # Generate vitals
                    vitals = model.update(dt=self.update_interval)
                    vitals['timestamp'] = datetime.now(timezone.utc).isoformat()
                    
                    # Publish to Kafka
                    if self.producer:
                        try:
                            self.producer.send(self.kafka_topic, value=vitals)
                        except Exception as e:
                            logger.error(f"Error sending to Kafka: {e}")
                            # Attempt reconnection
                            self._setup_kafka_producer()
                
                # Log status every 60 iterations (1 minute at 1Hz)
                if iteration % 60 == 0:
                    self._log_system_status()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                break
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}", exc_info=True)
                await asyncio.sleep(self.update_interval)
        
        # Cleanup
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")
    
    def _log_system_status(self):
        """Log system status (clean, no emojis)"""
        
        # Get status for first 3 patients as sample
        sample_patients = list(self.patients.keys())[:3]
        
        status_lines = ["System Status:"]
        for patient_id in sample_patients:
            model = self.patients[patient_id]
            status = model.get_status()
            
            status_lines.append(
                f"  {patient_id}: HR drift={status['hr_drift_rate']:.3f}, "
                f"SBP drift={status['sbp_drift_rate']:.3f}, "
                f"Stress={status['is_under_stress']}, "
                f"SI={status['current_shock_index']:.2f}"
            )
        
        logger.info("\n".join(status_lines))

async def main():
    """Entry point"""
    
    simulator = VitalSimulator()
    
    try:
        await simulator.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
