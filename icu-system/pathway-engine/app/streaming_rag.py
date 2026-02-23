"""
Streaming RAG - Live Vector Index Inside Pathway
Real-time embeddings with sliding window expiry and patient isolation
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("sentence-transformers not available, using mock embeddings")
    TRANSFORMERS_AVAILABLE = False
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, text):
            # Mock embedding: simple hash-based vector
            import hashlib
            h = hashlib.md5(str(text).encode()).hexdigest()
            return np.array([int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)])

logger = logging.getLogger(__name__)

class StreamingRAGIndex:
    """
    Patient-isolated streaming vector index
    Maintains last N hours of embeddings per patient with automatic expiry
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', window_hours=3):
        """
        Initialize streaming RAG index
        
        Args:
            model_name: Sentence transformer model name
            window_hours: Hours of history to maintain per patient
        """
        self.model = SentenceTransformer(model_name)
        self.window_hours = window_hours
        
        # Per-patient storage: {patient_id: [(text, embedding, timestamp, raw_data), ...]}
        self.patient_indices = defaultdict(list)
        
        # Statistics
        self.total_embeddings_created = 0
        self.total_embeddings_expired = 0
        
        logger.info(f"Initialized streaming RAG index")
        logger.info(f"Model: {model_name}")
        logger.info(f"Window: {window_hours} hours per patient")
        logger.info(f"Patient isolation: ENABLED")
        
    def add_enriched_event(self, event: dict):
        """
        Add enriched vital event to streaming index
        
        Args:
            event: Enriched vitals dictionary
        """
        
        try:
            # Convert event to structured text chunk
            chunk = self._event_to_text(event)
            
            # Generate embedding
            embedding = self.model.encode(chunk)
            
            # Store with timestamp
            patient_id = event['patient_id']
            
            # Parse timestamp (handle both string and datetime)
            if isinstance(event['timestamp'], str):
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            else:
                timestamp = event['timestamp']
            
            # Add to patient's index
            self.patient_indices[patient_id].append({
                'text': chunk,
                'embedding': embedding,
                'timestamp': timestamp,
                'raw_data': event
            })
            
            self.total_embeddings_created += 1
            
            # Cleanup old entries
            self._cleanup_expired(patient_id)
            
        except Exception as e:
            logger.error(f"Error adding event to RAG index: {e}", exc_info=True)
    
    def _event_to_text(self, event: dict) -> str:
        """
        Convert enriched event to structured natural language chunk
        
        Example output:
        "Time 14:32 | Patient P001 | HR 102 (↑3.2 from baseline 98.7) | 
         SBP 92 (↓8.1) | SpO2 94 | ShockIndex 1.11 (↑0.15) | 
         Lactate 2.1 (rising) | ANOMALY DETECTED"
        """
        
        # Extract time (HH:MM format)
        timestamp = event.get('timestamp', '')
        if isinstance(timestamp, str):
            time_str = timestamp.split('T')[1][:5] if 'T' in timestamp else ''
        else:
            time_str = timestamp.strftime('%H:%M') if hasattr(timestamp, 'strftime') else ''
        
        # Build text parts
        parts = [
            f"Time {time_str}",
            f"Patient {event.get('patient_id', 'Unknown')}"
        ]
        
        # Heart rate with delta
        hr = event.get('heart_rate', 0)
        hr_mean = event.get('rolling_mean_hr', hr)
        hr_delta = event.get('hr_delta', 0)
        delta_symbol = "↑" if hr_delta > 0 else "↓" if hr_delta < 0 else "→"
        parts.append(f"HR {hr:.0f} ({delta_symbol}{abs(hr_delta):.1f} from baseline {hr_mean:.1f})")
        
        # Systolic BP with delta
        sbp = event.get('systolic_bp', 0)
        sbp_delta = event.get('sbp_delta', 0)
        delta_symbol = "↓" if sbp_delta < 0 else "↑" if sbp_delta > 0 else "→"
        parts.append(f"SBP {sbp:.0f} ({delta_symbol}{abs(sbp_delta):.1f})")
        
        # SpO2
        spo2 = event.get('spo2', 0)
        parts.append(f"SpO2 {spo2:.0f}")
        
        # Shock index with delta
        si = event.get('shock_index', 0)
        si_delta = event.get('shock_index_delta', 0)
        delta_symbol = "↑" if si_delta > 0 else "↓" if si_delta < 0 else "→"
        parts.append(f"ShockIndex {si:.2f} ({delta_symbol}{abs(si_delta):.2f})")
        
        # Lactate with trend
        lactate = event.get('lactate', 0)
        lactate_delta = event.get('lactate_delta', 0)
        if lactate_delta > 0.1:
            lactate_trend = "rising"
        elif lactate_delta < -0.1:
            lactate_trend = "falling"
        else:
            lactate_trend = "stable"
        parts.append(f"Lactate {lactate:.1f} ({lactate_trend})")
        
        # Anomaly flag
        if event.get('anomaly_flag', False):
            parts.append("ANOMALY DETECTED")
        
        # Specific anomalies
        anomalies = []
        if event.get('hr_anomaly', False):
            anomalies.append("HR anomaly")
        if event.get('sbp_anomaly', False):
            anomalies.append("BP anomaly")
        if event.get('spo2_anomaly', False):
            anomalies.append("Hypoxia")
        if event.get('shock_index_anomaly', False):
            anomalies.append("High shock index")
        if event.get('lactate_anomaly', False):
            anomalies.append("Elevated lactate")
        
        if anomalies:
            parts.append(f"[{', '.join(anomalies)}]")
        
        return " | ".join(parts)
    
    def _cleanup_expired(self, patient_id: str):
        """
        Remove embeddings older than window from patient's index
        
        Args:
            patient_id: Patient identifier
        """
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.window_hours)
        
        original_count = len(self.patient_indices[patient_id])
        
        self.patient_indices[patient_id] = [
            entry for entry in self.patient_indices[patient_id]
            if entry['timestamp'] > cutoff
        ]
        
        expired_count = original_count - len(self.patient_indices[patient_id])
        if expired_count > 0:
            self.total_embeddings_expired += expired_count
            logger.debug(f"Expired {expired_count} embeddings for {patient_id}")
    
    def query(
        self, 
        patient_id: str, 
        query_text: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Query patient's streaming index
        
        Args:
            patient_id: Patient identifier
            query_text: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of retrieved contexts with relevance scores
        """
        
        if patient_id not in self.patient_indices:
            logger.warning(f"No index found for patient {patient_id}")
            return []
        
        if len(self.patient_indices[patient_id]) == 0:
            logger.warning(f"Empty index for patient {patient_id}")
            return []
        
        try:
            # Embed query
            query_embedding = self.model.encode(query_text)
            
            # Compute similarities with all entries
            patient_index = self.patient_indices[patient_id]
            similarities = []
            
            for entry in patient_index:
                similarity = self._cosine_similarity(
                    query_embedding, 
                    entry['embedding']
                )
                similarities.append((similarity, entry))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Return top-k results
            results = []
            for similarity, entry in similarities[:top_k]:
                results.append({
                    'text': entry['text'],
                    'timestamp': entry['timestamp'].isoformat(),
                    'relevance_score': float(similarity),
                    'raw_data': {
                        'heart_rate': entry['raw_data'].get('heart_rate'),
                        'systolic_bp': entry['raw_data'].get('systolic_bp'),
                        'shock_index': entry['raw_data'].get('shock_index'),
                        'lactate': entry['raw_data'].get('lactate'),
                        'anomaly_flag': entry['raw_data'].get('anomaly_flag')
                    }
                })
            
            logger.debug(f"Query '{query_text}' for {patient_id}: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying RAG index: {e}", exc_info=True)
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        
        per_patient_stats = {}
        for patient_id, entries in self.patient_indices.items():
            per_patient_stats[patient_id] = {
                'embeddings': len(entries),
                'oldest_timestamp': entries[0]['timestamp'].isoformat() if entries else None,
                'newest_timestamp': entries[-1]['timestamp'].isoformat() if entries else None
            }
        
        return {
            'total_patients': len(self.patient_indices),
            'total_embeddings': sum(len(idx) for idx in self.patient_indices.values()),
            'total_created': self.total_embeddings_created,
            'total_expired': self.total_embeddings_expired,
            'window_hours': self.window_hours,
            'per_patient': per_patient_stats
        }
    
    def cleanup_patient(self, patient_id: str):
        """Remove all data for a patient (e.g., on discharge)"""
        if patient_id in self.patient_indices:
            count = len(self.patient_indices[patient_id])
            del self.patient_indices[patient_id]
            logger.info(f"Cleaned up {count} embeddings for discharged patient {patient_id}")
