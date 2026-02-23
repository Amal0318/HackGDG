"""
Stream Merger - Join vitals_enriched + vitals_predictions
Maintains unified patient state from multiple Kafka topics
"""

import json
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, List

try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    logging.warning("Kafka not available")
    KAFKA_AVAILABLE = False
    class KafkaConsumer:
        def __init__(self, *args, **kwargs):
            pass
        def __iter__(self):
            return iter([])

logger = logging.getLogger(__name__)

class StreamMerger:
    """
    Merge vitals_enriched + vitals_predictions into unified patient view
    Maintains in-memory state and notifies listeners on updates
    """
    
    def __init__(self, kafka_servers='kafka:9092'):
        self.kafka_servers = kafka_servers
        
        # Patient state: {patient_id: {'vitals': {...}, 'prediction': {...}}}
        self.patient_state = defaultdict(dict)
        
        # History buffer per patient (last 100 entries)
        self.patient_history = defaultdict(lambda: deque(maxlen=100))
        
        # Update listeners (callbacks)
        self.listeners: List[Callable] = []
        
        # Consumer threads
        self.vitals_thread: Optional[threading.Thread] = None
        self.predictions_thread: Optional[threading.Thread] = None
        
        self.is_running = False
        
        logger.info("Stream merger initialized")
        logger.info(f"Kafka servers: {kafka_servers}")
    
    def start(self):
        """Start consumer threads"""
        
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available, stream merger cannot start")
            return
        
        self.is_running = True
        
        # Start vitals consumer thread
        self.vitals_thread = threading.Thread(
            target=self._consume_vitals,
            daemon=True,
            name="VitalsConsumer"
        )
        self.vitals_thread.start()
        logger.info("Vitals consumer thread started")
        
        # Start predictions consumer thread
        self.predictions_thread = threading.Thread(
            target=self._consume_predictions,
            daemon=True,
            name="PredictionsConsumer"
        )
        self.predictions_thread.start()
        logger.info("Predictions consumer thread started")
    
    def stop(self):
        """Stop consumer threads"""
        logger.info("Stopping stream merger...")
        self.is_running = False
    
    def _consume_vitals(self):
        """Consume vitals_enriched topic"""
        
        try:
            consumer = KafkaConsumer(
                'vitals_enriched',
                bootstrap_servers=self.kafka_servers,
                group_id='backend-vitals',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            logger.info("Vitals consumer connected")
            
            for message in consumer:
                if not self.is_running:
                    break
                
                try:
                    data = message.value
                    patient_id = data.get('patient_id')
                    
                    if patient_id:
                        # Update patient state
                        self.patient_state[patient_id]['vitals'] = data
                        self.patient_state[patient_id]['last_vitals_update'] = datetime.now()
                        
                        # Notify listeners
                        self._notify_listeners(patient_id)
                
                except Exception as e:
                    logger.error(f"Error processing vitals message: {e}")
            
            consumer.close()
            
        except Exception as e:
            logger.error(f"Vitals consumer error: {e}", exc_info=True)
    
    def _consume_predictions(self):
        """Consume vitals_predictions topic"""
        
        try:
            consumer = KafkaConsumer(
                'vitals_predictions',
                bootstrap_servers=self.kafka_servers,
                group_id='backend-predictions',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            logger.info("Predictions consumer connected")
            
            for message in consumer:
                if not self.is_running:
                    break
                
                try:
                    data = message.value
                    patient_id = data.get('patient_id')
                    
                    if patient_id:
                        # Update patient state
                        self.patient_state[patient_id]['prediction'] = data
                        self.patient_state[patient_id]['last_prediction_update'] = datetime.now()
                        
                        # Add to history
                        unified = self.get_unified_view(patient_id)
                        if unified:
                            self.patient_history[patient_id].append({
                                'timestamp': data.get('timestamp'),
                                'risk_score': data.get('risk_score'),
                                'vitals': unified.get('vitals', {})
                            })
                        
                        # Notify listeners
                        self._notify_listeners(patient_id)
                
                except Exception as e:
                    logger.error(f"Error processing prediction message: {e}")
            
            consumer.close()
            
        except Exception as e:
            logger.error(f"Predictions consumer error: {e}", exc_info=True)
    
    def get_unified_view(self, patient_id: str) -> Optional[Dict]:
        """
        Get merged view for patient
        
        Returns:
            Unified patient view with vitals + risk_score
        """
        
        state = self.patient_state.get(patient_id, {})
        
        if 'vitals' not in state:
            return None
        
        vitals = state['vitals']
        prediction = state.get('prediction', {})
        
        return {
            'patient_id': patient_id,
            'timestamp': vitals.get('timestamp'),
            'vitals': {
                'heart_rate': vitals.get('heart_rate'),
                'systolic_bp': vitals.get('systolic_bp'),
                'diastolic_bp': vitals.get('diastolic_bp'),
                'map': vitals.get('map'),
                'spo2': vitals.get('spo2'),
                'respiratory_rate': vitals.get('respiratory_rate'),
                'temperature': vitals.get('temperature'),
                'lactate': vitals.get('lactate'),
                'shock_index': vitals.get('shock_index')
            },
            'features': {
                'rolling_mean_hr': vitals.get('rolling_mean_hr'),
                'rolling_mean_sbp': vitals.get('rolling_mean_sbp'),
                'rolling_mean_spo2': vitals.get('rolling_mean_spo2'),
                'hr_delta': vitals.get('hr_delta'),
                'sbp_delta': vitals.get('sbp_delta'),
                'shock_index_delta': vitals.get('shock_index_delta'),
                'lactate_delta': vitals.get('lactate_delta'),
                'anomaly_flag': vitals.get('anomaly_flag', False),
                'hr_anomaly': vitals.get('hr_anomaly', False),
                'sbp_anomaly': vitals.get('sbp_anomaly', False),
                'spo2_anomaly': vitals.get('spo2_anomaly', False),
                'shock_index_anomaly': vitals.get('shock_index_anomaly', False),
                'lactate_anomaly': vitals.get('lactate_anomaly', False)
            },
            'risk_score': prediction.get('risk_score', 0.0),
            'last_updated': vitals.get('timestamp')
        }
    
    def get_all_patients(self) -> List[str]:
        """Get list of all active patient IDs"""
        return list(self.patient_state.keys())
    
    def get_patient_history(self, patient_id: str, hours: int = 4) -> List[Dict]:
        """
        Get patient history for last N hours
        
        Args:
            patient_id: Patient identifier
            hours: Number of hours to retrieve
            
        Returns:
            List of historical data points
        """
        
        if patient_id not in self.patient_history:
            return []
        
        history = list(self.patient_history[patient_id])
        
        # Filter by time window if timestamps available
        cutoff = datetime.now() - timedelta(hours=hours)
        
        filtered_history = []
        for entry in history:
            try:
                if entry.get('timestamp'):
                    ts = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                    if ts > cutoff:
                        filtered_history.append(entry)
                else:
                    filtered_history.append(entry)
            except:
                filtered_history.append(entry)
        
        return filtered_history
    
    def register_listener(self, callback: Callable):
        """
        Register callback for patient updates
        
        Args:
            callback: Function that takes (patient_id, unified_view)
        """
        self.listeners.append(callback)
        logger.info(f"Registered listener: {callback.__name__}")
    
    def _notify_listeners(self, patient_id: str):
        """Notify all listeners of patient update"""
        
        unified_view = self.get_unified_view(patient_id)
        
        if unified_view:
            for callback in self.listeners:
                try:
                    callback(patient_id, unified_view)
                except Exception as e:
                    logger.error(f"Listener error: {e}")
    
    def get_status(self) -> Dict:
        """Get stream merger status"""
        
        return {
            'is_running': self.is_running,
            'active_patients': len(self.patient_state),
            'total_listeners': len(self.listeners),
            'threads_alive': {
                'vitals': self.vitals_thread.is_alive() if self.vitals_thread else False,
                'predictions': self.predictions_thread.is_alive() if self.predictions_thread else False
            }
        }
