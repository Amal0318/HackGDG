"""
VitalX Pathway Engine Kafka Configuration
Production-ready Kafka connectivity for ICU Digital Twin streaming pipeline
"""

import logging
from typing import Dict, Any, Optional
import pathway as pw
from .settings import settings

logger = logging.getLogger(__name__)

class KafkaConnector:
    """Production-ready Kafka connector for Pathway streaming"""
    
    def __init__(self):
        self.input_topic = settings.kafka.input_topic
        self.output_topic = settings.kafka.output_topic
        self._validate_configuration()
        
    def _validate_configuration(self):
        """Validate Kafka configuration"""
        required_settings = [
            settings.kafka.bootstrap_servers,
            settings.kafka.input_topic,
            settings.kafka.output_topic
        ]
        
        if not all(required_settings):
            raise ValueError("Missing required Kafka configuration")
        
        logger.info(f"Kafka configuration validated:")
        logger.info(f"  Bootstrap servers: {settings.kafka.bootstrap_servers}")
        logger.info(f"  Input topic: {settings.kafka.input_topic}")
        logger.info(f"  Output topic: {settings.kafka.output_topic}")
        logger.info(f"  Consumer group: {settings.kafka.consumer_group}")
    
    def get_consumer_config(self) -> Dict[str, Any]:
        """Get Kafka consumer configuration for Pathway"""
        return {
            "bootstrap.servers": settings.kafka.bootstrap_servers,
            "group.id": settings.kafka.consumer_group,
            "auto.offset.reset": settings.kafka.auto_offset_reset,
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 1000,
            "session.timeout.ms": 30000,
            "heartbeat.interval.ms": 10000,
            "max.poll.interval.ms": 300000,
            "fetch.min.bytes": 1,
            "fetch.max.wait.ms": 500,
            # Error handling
            "api.version.request": True,
            "api.version.fallback.ms": 0,
            "broker.version.fallback": "0.9.0",
            # Security (add as needed)
            # "security.protocol": "SASL_SSL",
            # "sasl.mechanism": "PLAIN",
        }
    
    def get_producer_config(self) -> Dict[str, Any]:
        """Get Kafka producer configuration for Pathway with reliability settings"""
        return {
            "bootstrap.servers": settings.kafka.bootstrap_servers,
            "acks": settings.kafka.acks,  # Wait for all in-sync replicas
            "retries": settings.kafka.retries,
            "retry.backoff.ms": settings.kafka.retry_backoff_ms,
            "delivery.timeout.ms": 120000,  # 2 minutes total timeout
            "request.timeout.ms": 30000,    # 30 seconds per request
            "max.in.flight.requests.per.connection": 5,
            "enable.idempotence": True,     # Prevent duplicates
            # Batching for performance
            "batch.size": 16384,
            "linger.ms": 10,
            "compression.type": "snappy",
            # Error handling
            "api.version.request": True,
            "api.version.fallback.ms": 0,
            "broker.version.fallback": "0.9.0",
        }
    
    def create_input_stream(self) -> pw.Table:
        """Create Pathway input stream from Kafka topic"""
        try:
            logger.info(f"Connecting to Kafka input topic: {self.input_topic}")
            
            # Pathway Kafka input connector
            input_stream = pw.io.kafka.read(
                rdkafka_settings=self.get_consumer_config(),
                topic=self.input_topic,
                format="json",
                mode="streaming",
                autocommit_duration_ms=1000,
            )
            
            logger.info("Successfully connected to Kafka input stream")
            return input_stream
            
        except Exception as e:
            logger.error(f"Failed to create input stream: {e}")
            raise ConnectionError(f"Kafka input connection failed: {e}")
    
    def create_output_sink(self, enriched_stream: pw.Table) -> None:
        """Create Pathway output sink to Kafka topic"""
        try:
            logger.info(f"Setting up Kafka output sink to topic: {self.output_topic}")
            
            # Pathway Kafka output connector  
            pw.io.kafka.write(
                table=enriched_stream,
                rdkafka_settings=self.get_producer_config(),
                topic=self.output_topic,
                format="json",
            )
            
            logger.info("Successfully configured Kafka output sink")
            
        except Exception as e:
            logger.error(f"Failed to create output sink: {e}")
            raise ConnectionError(f"Kafka output connection failed: {e}")

    def test_connectivity(self) -> bool:
        """Test Kafka connectivity (for health checks)"""
        try:
            from confluent_kafka.admin import AdminClient
            from confluent_kafka import TopicPartition
            
            # Test admin connection
            admin_config = {
                "bootstrap.servers": settings.kafka.bootstrap_servers,
                "request.timeout.ms": 10000,
            }
            
            admin_client = AdminClient(admin_config)
            
            # Test if we can get topic metadata
            metadata = admin_client.list_topics(timeout=10)
            
            # Check if our topics exist
            topics = metadata.topics
            input_exists = self.input_topic in topics
            output_exists = self.output_topic in topics
            
            if not input_exists:
                logger.warning(f"Input topic '{self.input_topic}' does not exist")
            
            if not output_exists:
                logger.warning(f"Output topic '{self.output_topic}' does not exist") 
            
            logger.info("Kafka connectivity test successful")
            return True
            
        except Exception as e:
            logger.error(f"Kafka connectivity test failed: {e}")
            return False

def create_kafka_connections() -> tuple[pw.Table, callable]:
    """Factory function to create Kafka input and output connections"""
    connector = KafkaConnector()
    
    # Test connectivity first
    if not connector.test_connectivity():
        logger.warning("Kafka connectivity test failed, but proceeding anyway")
    
    # Create input stream
    input_stream = connector.create_input_stream()
    
    # Return input stream and a function to create output sink
    def create_output_sink(enriched_stream: pw.Table) -> None:
        return connector.create_output_sink(enriched_stream)
    
    return input_stream, create_output_sink

def get_kafka_health_status() -> Dict[str, Any]:
    """Get Kafka health status for monitoring"""
    try:
        connector = KafkaConnector()
        is_healthy = connector.test_connectivity()
        
        return {
            "kafka_healthy": is_healthy,
            "input_topic": connector.input_topic,
            "output_topic": connector.output_topic,
            "bootstrap_servers": settings.kafka.bootstrap_servers,
            "consumer_group": settings.kafka.consumer_group,
        }
    except Exception as e:
        return {
            "kafka_healthy": False,
            "error": str(e),
            "bootstrap_servers": settings.kafka.bootstrap_servers,
        }