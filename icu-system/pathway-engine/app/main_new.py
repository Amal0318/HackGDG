"""
Pathway Engine - Streaming Feature Engineering + Live RAG
Production-grade streaming pipeline with deterministic features and real-time RAG
"""

import sys
import logging
import signal
import threading
import traceback
from typing import Optional
import pathway as pw
from datetime import datetime
import uvicorn

# Import application modules
from .feature_engineering import create_feature_pipeline, validate_feature_output
from .streaming_rag import StreamingRAGIndex
from . import query_api

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
    2. Streaming RAG index (live embeddings)
    3. Query API (retrieval interface)
    """
    
    def __init__(self):
        self.engine_name = "VitalX Pathway Engine"
        self.version = "2.0.0"
        self.rag_index: Optional[StreamingRAGIndex] = None
        self.input_stream: Optional[pw.Table] = None
        self.enriched_stream: Optional[pw.Table] = None
        self.is_running = False
        
        # Kafka configuration
        self.kafka_servers = 'kafka:9092'
        self.input_topic = 'vitals_raw'
        self.output_topic = 'vitals_enriched'
        
    def initialize_components(self) -> bool:
        """Initialize all engine components"""
        
        try:
            logger.info(f"Initializing {self.engine_name} v{self.version}")
            logger.info("=" * 60)
            
            # Initialize streaming RAG index
            logger.info("Initializing streaming RAG index...")
            self.rag_index = StreamingRAGIndex(
                model_name='all-MiniLM-L6-v2',
                window_hours=3
            )
            
            # Set RAG index reference in query API
            query_api.set_rag_index(self.rag_index)
            
            # Initialize Kafka input stream
            logger.info(f"Connecting to Kafka: {self.kafka_servers}")
            logger.info(f"Input topic: {self.input_topic}")
            
            # Define schema for incoming vitals
            class VitalsSchema(pw.Schema):
                patient_id: str
                timestamp: str
                heart_rate: float
                systolic_bp: float
                diastolic_bp: float
                map: float
                spo2: float
                respiratory_rate: float
                temperature: float
                lactate: float
                shock_index: float
            
            self.input_stream = pw.io.kafka.read(
                rdkafka_settings={
                    'bootstrap.servers': self.kafka_servers,
                    'group.id': 'pathway-engine',
                    'auto.offset.reset': 'latest'
                },
                topic=self.input_topic,
                format='json',
                schema=VitalsSchema
            )
            
            logger.info("Kafka input stream configured")
            
            # Create feature engineering pipeline
            logger.info("Building feature engineering pipeline...")
            self.enriched_stream = create_feature_pipeline(self.input_stream)
            
            # Validate output (ensure no risk_score field)
            self.enriched_stream = validate_feature_output(self.enriched_stream)
            
            # Set up RAG index update (event-driven)
            logger.info("Configuring streaming RAG updates...")
            self._setup_rag_updates()
            
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
    
    def _setup_rag_updates(self):
        """
        Configure RAG index to update on each enriched event
        This maintains the live streaming vector index
        """
        
        def on_event(key, row, time, is_addition):
            """Subscriber callback to add events to RAG index"""
            if not is_addition:
                return  # Skip deletions
            
            try:
                # Convert Pathway row to dictionary
                event = {
                    'patient_id': row['patient_id'],
                    'timestamp': row['timestamp'],
                    'heart_rate': row['heart_rate'],
                    'systolic_bp': row['systolic_bp'],
                    'diastolic_bp': row['diastolic_bp'],
                    'spo2': row['spo2'],
                    'shock_index': row['shock_index'],
                    'lactate': row['lactate'],
                    'rolling_mean_hr': row['rolling_mean_hr'],
                    'hr_delta': row['hr_delta'],
                    'sbp_delta': row['sbp_delta'],
                    'shock_index_delta': row['shock_index_delta'],
                    'lactate_delta': row['lactate_delta'],
                    'anomaly_flag': row['anomaly_flag'],
                    'hr_anomaly': row['hr_anomaly'],
                    'sbp_anomaly': row['sbp_anomaly'],
                    'spo2_anomaly': row['spo2_anomaly'],
                    'shock_index_anomaly': row['shock_index_anomaly'],
                    'lactate_anomaly': row['lactate_anomaly']
                }
                
                # Add to streaming RAG index
                self.rag_index.add_enriched_event(event)
                
            except Exception as e:
                logger.error(f"Error updating RAG index: {e}", exc_info=True)
        
        # Subscribe to enriched stream for RAG updates (non-blocking side effect)
        pw.io.subscribe(self.enriched_stream, on_event)
        
        logger.info("RAG updates enabled via streaming subscriber")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.is_running = False
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def run_query_api(self):
        """Run FastAPI query interface in separate thread"""
        
        logger.info("Starting Query API server on port 8080...")
        
        try:
            uvicorn.run(
                query_api.app,
                host="0.0.0.0",
                port=8080,
                log_level="info"
            )
        except Exception as e:
            logger.error(f"Query API error: {e}")
    
    def run(self):
        """Main execution loop"""
        
        logger.info(f"Starting {self.engine_name}")
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Initialize components
        if not self.initialize_components():
            logger.error("Failed to initialize components, exiting")
            return 1
        
        # Start Query API in background thread
        api_thread = threading.Thread(
            target=self.run_query_api,
            daemon=True,
            name="QueryAPI"
        )
        api_thread.start()
        logger.info("Query API thread started")
        
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
