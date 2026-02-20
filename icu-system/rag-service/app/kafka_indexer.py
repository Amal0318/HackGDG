import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from kafka import KafkaConsumer
import threading
from app.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPICS, KAFKA_GROUP_ID
from app.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class KafkaDataIndexer:
    """Consumes live data from Kafka and indexes it in ChromaDB"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.consumer = None
        self.running = False
        self.indexing_thread = None
        
    def start(self):
        """Start the Kafka consumer in a background thread"""
        if self.running:
            logger.warning("Indexer already running")
            return
            
        logger.info(f"Starting Kafka consumer for topics: {KAFKA_TOPICS}")
        
        try:
            self.consumer = KafkaConsumer(
                *KAFKA_TOPICS,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=KAFKA_GROUP_ID,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            self.running = True
            self.indexing_thread = threading.Thread(target=self._consume_messages, daemon=True)
            self.indexing_thread.start()
            
            logger.info("Kafka consumer started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise
    
    def stop(self):
        """Stop the Kafka consumer"""
        logger.info("Stopping Kafka consumer...")
        self.running = False
        
        if self.consumer:
            self.consumer.close()
        
        if self.indexing_thread:
            self.indexing_thread.join(timeout=5)
        
        logger.info("Kafka consumer stopped")
    
    def _consume_messages(self):
        """Consume messages from Kafka and index them"""
        logger.info("Starting message consumption...")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    data = message.value
                    topic = message.topic
                    
                    # Process and index the message
                    self._index_message(topic, data)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
    
    def _index_message(self, topic: str, data: Dict[str, Any]):
        """Convert Kafka message to searchable document and index it"""
        
        try:
            # Extract patient_id
            patient_id = data.get('patient_id', 'unknown')
            
            # Create timestamp
            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            
            # Generate human-readable text based on topic
            if topic == 'vitals':
                text = self._format_vitals(data)
            else:
                text = self._format_generic(data)
            
            # Create metadata
            metadata = {
                'patient_id': patient_id,
                'timestamp': timestamp,
                'topic': topic,
                'source': 'kafka_live'
            }
            
            # Add to vector store
            self.vector_store.add_document(
                text=text,
                metadata=metadata,
                doc_id=f"{patient_id}_{timestamp}"
            )
            
            logger.debug(f"Indexed message for patient {patient_id}")
            
        except Exception as e:
            logger.error(f"Error indexing message: {e}")
    
    def _format_vitals(self, data: Dict[str, Any]) -> str:
        """Format vital signs data into readable text"""
        patient_id = data.get('patient_id', 'unknown')
        timestamp = data.get('timestamp', '')
        
        # Extract vital signs from flat structure
        hr = data.get('heart_rate', 'N/A')
        sys_bp = data.get('systolic_bp', 'N/A')
        dia_bp = data.get('diastolic_bp', 'N/A')
        spo2 = data.get('spo2', 'N/A')
        rr = data.get('respiratory_rate', 'N/A')
        temp = data.get('temperature', 'N/A')
        state = data.get('state', 'unknown')
        event_type = data.get('event_type', 'none')
        
        text = f"Patient {patient_id} vital signs at {timestamp}: "
        text += f"Heart Rate: {hr} bpm, "
        text += f"Blood Pressure: {sys_bp}/{dia_bp} mmHg, "
        text += f"SpO2: {spo2}%, "
        text += f"Respiratory Rate: {rr} breaths/min, "
        text += f"Temperature: {temp}°F. "
        text += f"Patient state: {state}"
        
        if event_type and event_type != 'none':
            text += f", Event: {event_type}"
        
        return text
    
    def _format_patient_update(self, data: Dict[str, Any]) -> str:
        """Format patient update data"""
        patient_id = data.get('patient_id', 'unknown')
        timestamp = data.get('timestamp', '')
        update_type = data.get('type', 'update')
        
        text = f"Patient {patient_id} {update_type} at {timestamp}: "
        
        # Add relevant fields
        if 'latest_risk_score' in data:
            text += f"Risk Score: {data['latest_risk_score']}, "
        
        if 'abnormal_vitals' in data and data['abnormal_vitals']:
            abnormal = ', '.join([f"{v['vital']}: {v['value']} {v.get('unit', '')}" 
                                 for v in data['abnormal_vitals']])
            text += f"Abnormal Vitals: {abnormal}, "
        
        if 'floor' in data:
            text += f"Floor: {data['floor']}, "
        
        if 'bed_number' in data:
            text += f"Bed: {data['bed_number']}"
        
        return text
    
    def _format_generic(self, data: Dict[str, Any]) -> str:
        """Format generic data"""
        return f"Patient {data.get('patient_id', 'unknown')} data: {json.dumps(data)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics"""
        return {
            'running': self.running,
            'topics': KAFKA_TOPICS,
            'consumer_connected': self.consumer is not None,
            'vector_store_count': self.vector_store.get_document_count()
        }
