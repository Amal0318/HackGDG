"""
Pathway Engine - Streaming Feature Engineering + Live RAG
Production-grade streaming pipeline with deterministic features and real-time RAG
"""

import sys
import logging
import signal
import traceback
from typing import Optional
import pathway as pw
from datetime import datetime

# Import application modules
from .feature_engineering import create_feature_pipeline, validate_feature_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class PathwayEngine:
    """
    Main Pathway streaming engine
    Responsibilities:
    1. Feature engineering (deterministic)
    2. Kafka output to vitals_enriched topic
    """
    
    def __init__(self):
        self.engine_name = "VitalX Pathway Engine"
        self.version = "2.0.0"
        self.input_stream: Optional[pw.Table] = None
        self.enriched_stream: Optional[pw.Table] = None
        self.is_running = False
        
        # Kafka configuration - using vitals topic from vital simulator
        self.kafka_servers = 'kafka:9092'
        self.input_topic = 'vitals'
        self.output_topic = 'vitals_enriched'
        
    def initialize_components(self) -> bool:
        """Initialize all engine components"""
        
        try:
            logger.info(f"Initializing {self.engine_name} v{self.version}")
            logger.info("=" * 60)
            
            # Initialize Kafka input stream
            logger.info(f"Connecting to Kafka: {self.kafka_servers}")
            logger.info(f"Input topic: {self.input_topic}")
            
            # Define schema for incoming vitals (matches vitals topic from simulator)
            class VitalsSchema(pw.Schema):
                patient_id: str
                timestamp: str
                heart_rate: float
                systolic_bp: float
                diastolic_bp: float
                spo2: float
                respiratory_rate: float
                temperature: float
                shock_index: float
                state: str
                event_type: str
            
            self.input_stream = pw.io.kafka.read(
                rdkafka_settings={
                    'bootstrap.servers': self.kafka_servers,
                    'group.id': 'pathway-engine',
                    'auto.offset.reset': 'earliest'
                },
                topic=self.input_topic,
                format='json',
                schema=VitalsSchema
            )
            
            logger.info("Kafka input stream configured")
            
            # Add missing fields: calculate MAP and set default lactate
            logger.info("Adding derived fields (MAP, lactate)...")
            self.input_stream = self.input_stream.select(
                patient_id=pw.this.patient_id,
                timestamp=pw.this.timestamp,
                heart_rate=pw.this.heart_rate,
                systolic_bp=pw.this.systolic_bp,
                diastolic_bp=pw.this.diastolic_bp,
                spo2=pw.this.spo2,
                respiratory_rate=pw.this.respiratory_rate,
                temperature=pw.this.temperature,
                shock_index=pw.this.shock_index,
                state=pw.this.state,
                event_type=pw.this.event_type,
                map=pw.this.diastolic_bp + (pw.this.systolic_bp - pw.this.diastolic_bp) / 3.0,
                lactate=pw.cast(float, 1.0)  # Normal lactate default value
            )
            
            # Create feature engineering pipeline
            logger.info("Building feature engineering pipeline...")
            self.enriched_stream = create_feature_pipeline(self.input_stream)
            
            # Validate output (ensure no risk_score field)
            self.enriched_stream = validate_feature_output(self.enriched_stream)
            
            # Configure Kafka output (table, topic_name, settings)
            logger.info(f"Output topic: {self.output_topic}")
            pw.io.kafka.write(
                self.enriched_stream,
                rdkafka_settings={
                    'bootstrap.servers': self.kafka_servers,
                    'enable.idempotence': 'true',
                    'acks': 'all'
                },
                topic_name=self.output_topic
            )
            
            logger.info("All components initialized successfully")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.is_running = False
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def run(self):
        """Main execution loop"""
        
        logger.info(f"Starting {self.engine_name}")
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Initialize components
        if not self.initialize_components():
            logger.error("Failed to initialize components, exiting")
            return 1
        
        # Run Pathway pipeline
        logger.info("Starting Pathway streaming pipeline...")
        self.is_running = True
        
        try:
            pw.run()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            logger.error(traceback.format_exc())
            return 1
        finally:
            logger.info("Pathway engine stopped")
        
        return 0

def main():
    """Entry point"""
    
    logger.info("=" * 60)
    logger.info("VitalX Pathway Engine - Streaming Architecture")
    logger.info("Feature Engineering + Live RAG")
    logger.info("=" * 60)
    
    engine = PathwayEngine()
    exit_code = engine.run()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
