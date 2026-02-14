#!/usr/bin/env python3
"""
VitalX Pathway Streaming Engine
Production-ready real-time ICU Digital Twin streaming pipeline

This is the main entry point for the Pathway streaming engine that processes
real-time vital signs data from ICU patients, performs risk analysis, and
publishes enriched data for downstream ML services.

Architecture:
Vital Simulator → Kafka (vitals) → Pathway Engine → Kafka (vitals_enriched) → ML Service
"""

import sys
import logging
import signal
import traceback
from typing import Optional
import pathway as pw
from datetime import datetime

# Import application modules
from .settings import settings, validate_risk_weights
from .kafka_config import create_kafka_connections, get_kafka_health_status
from .risk_engine import create_risk_engine, get_risk_engine_health
from .schema import parse_vital_signs_input, serialize_vital_signs_output

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.log_level.upper()),
    format=settings.logging.log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/pathway-engine.log', mode='a')
    ] if sys.stdout.isatty() else [logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

class VitalXPathwayEngine:
    """Main VitalX Pathway streaming engine"""
    
    def __init__(self):
        self.engine_name = "VitalX Pathway Engine"
        self.version = settings.version
        self.risk_engine: Optional[object] = None
        self.input_stream: Optional[pw.Table] = None
        self.enriched_stream: Optional[pw.Table] = None
        self.is_running = False
        
    def initialize_components(self) -> bool:
        """Initialize all engine components"""
        try:
            logger.info(f"Initializing {self.engine_name} v{self.version}")
            
            # Validate configuration
            logger.info("Validating risk calculation weights")
            validate_risk_weights()
            
            # Initialize risk engine
            logger.info("Creating risk analysis engine")
            self.risk_engine = create_risk_engine()
            
            # Initialize Kafka connections
            logger.info("Setting up Kafka connections")
            self.input_stream, self.create_output_sink = create_kafka_connections()
            
            logger.info("All components initialized successfully")
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
    
    def validate_input_data(self, vitals_stream: pw.Table) -> pw.Table:
        """Validate and clean input data stream"""
        
        # Add data validation and cleaning logic
        validated_stream = vitals_stream.filter(
            # Basic validation filters
            (pw.this.heart_rate >= 0) & (pw.this.heart_rate <= 300) &
            (pw.this.systolic_bp >= 0) & (pw.this.systolic_bp <= 300) &
            (pw.this.spo2 >= 0) & (pw.this.spo2 <= 100) &
            (pw.this.patient_id.is_not_none()) &
            (pw.this.timestamp.is_not_none())
        )
        
        return validated_stream
    
    def setup_monitoring_and_logging(self, enriched_stream: pw.Table) -> pw.Table:
        """Setup monitoring and logging for the enriched stream"""
        
        # Add monitoring transformations
        monitored_stream = enriched_stream.select(
            *enriched_stream,
            processing_timestamp=pw.apply(lambda: datetime.utcnow().isoformat())
        )
        
        # Log high-risk patients
        high_risk_alerts = monitored_stream.filter(
            pw.this.computed_risk > 0.7  # High risk threshold
        )
        
        # Log anomalies
        anomaly_alerts = monitored_stream.filter(
            pw.this.anomaly_flag == True
        )
        
        # Debug logging (can be enabled/disabled via settings)
        if logger.isEnabledFor(logging.DEBUG):
            pw.debug.compute_and_print(monitored_stream.select(
                pw.this.patient_id,
                pw.this.computed_risk,
                pw.this.anomaly_flag
            ))
        
        return monitored_stream
    
    def create_streaming_pipeline(self) -> pw.Table:
        """Create the complete streaming data pipeline"""
        try:
            logger.info("Setting up streaming data pipeline")
            
            # Step 1: Validate input data
            logger.info("Setting up input data validation")
            validated_stream = self.validate_input_data(self.input_stream)
            
            # Step 2: Enrich with risk analysis
            logger.info("Setting up risk analysis and enrichment")
            enriched_stream = self.risk_engine.enrich_vitals_stream(validated_stream)
            
            # Step 3: Setup monitoring and logging
            logger.info("Setting up monitoring and alerting")
            monitored_stream = self.setup_monitoring_and_logging(enriched_stream)
            
            # Step 4: Create output sink
            logger.info("Setting up Kafka output sink")
            self.create_output_sink(monitored_stream)
            
            logger.info("Streaming pipeline successfully configured")
            return monitored_stream
            
        except Exception as e:
            logger.error(f"Pipeline setup failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to create streaming pipeline: {e}")
    
    def run_health_checks(self) -> bool:
        """Run comprehensive health checks"""
        try:
            logger.info("Running health checks...")
            
            # Check Kafka connectivity
            kafka_health = get_kafka_health_status()
            if not kafka_health.get("kafka_healthy", False):
                logger.error(f"Kafka health check failed: {kafka_health}")
                return False
            
            # Check risk engine 
            risk_engine_health = get_risk_engine_health()
            if not risk_engine_health.get("risk_engine_healthy", False):
                logger.error(f"Risk engine health check failed: {risk_engine_health}")
                return False
            
            logger.info("All health checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Health checks failed: {e}")
            return False
    
    def start_streaming(self):
        """Start the Pathway streaming engine"""
        try:
            logger.info("=" * 80)
            logger.info(f"🚀 Starting {self.engine_name} v{self.version}")
            logger.info("=" * 80)
            
            # Initialize components
            if not self.initialize_components():
                raise RuntimeError("Component initialization failed")
            
            # Run health checks
            if not self.run_health_checks():
                raise RuntimeError("Health checks failed")
            
            # Setup signal handlers for graceful shutdown
            self.setup_signal_handlers()
            
            # Create streaming pipeline
            self.enriched_stream = self.create_streaming_pipeline()
            
            # Start streaming
            self.is_running = True
            logger.info("🔄 VitalX Pathway Engine started successfully")
            logger.info("📈 Publishing enriched vital signs stream")
            logger.info("🎯 Real-time risk analysis and anomaly detection active")
            logger.info("-" * 80)
            
            # Run the Pathway streaming computation
            pw.run(
                monitoring_level=pw.MonitoringLevel.ALL if settings.pathway.monitoring_enabled else pw.MonitoringLevel.NONE,
                with_http_server=True,  # Enable monitoring HTTP server
            )
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down gracefully...")
        except Exception as e:
            logger.error(f"Streaming engine failed: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown procedure"""
        try:
            logger.info("🛑 Initiating graceful shutdown...")
            self.is_running = False
            
            # Additional cleanup can be added here
            # (close connections, flush buffers, etc.)
            
            logger.info("✅ VitalX Pathway Engine shut down complete")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    try:
        # Create and start the engine
        engine = VitalXPathwayEngine()
        engine.start_streaming()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()