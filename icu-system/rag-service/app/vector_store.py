import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from app.config import DATA_RETENTION_HOURS
import json

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Simple in-memory store for patient data - lightweight version"""
    
    def __init__(self):
        logger.info("Initializing in-memory vector store (lightweight mode)")
        self.documents = []  # Simple list to store documents
        
    def add_document(self, text: str, metadata: Dict[str, Any], doc_id: str):
        """Add a document to the store"""
        try:
            doc = {
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.documents.append(doc)
            
            # Keep only recent documents
            self._cleanup_old_documents()
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
    
    def add_documents_batch(self, texts: List[str], metadatas: List[Dict[str, Any]], doc_ids: List[str]):
        """Add multiple documents in batch"""
        try:
            for text, metadata, doc_id in zip(texts, metadatas, doc_ids):
                self.add_document(text, metadata, doc_id)
            
            logger.info(f"Added batch of {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Error adding batch: {e}")
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        patient_id: Optional[str] = None,
        time_window_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents using simple keyword matching"""
        try:
            results = []
            query_lower = query.lower()
            
            # Calculate time filter if specified
            cutoff_time = None
            if time_window_hours:
                cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # If specific patient requested, filter strictly
            if patient_id:
                # Only get documents for this specific patient
                patient_docs = []
                for doc in reversed(self.documents):  # Most recent first
                    if doc["metadata"].get("patient_id") == patient_id:
                        if cutoff_time:
                            doc_time = datetime.fromisoformat(doc["timestamp"])
                            if doc_time < cutoff_time:
                                continue
                        patient_docs.append({
                            "document": doc["text"],
                            "metadata": doc["metadata"],
                            "id": doc["id"]
                        })
                        if len(patient_docs) >= n_results:
                            break
                return patient_docs
            
            # General search - get unique patients' most recent data
            seen_patients = set()
            for doc in reversed(self.documents):  # Most recent first
                pid = doc["metadata"].get("patient_id")
                
                # Skip if we already have this patient's data
                if pid in seen_patients:
                    continue
                
                if cutoff_time:
                    doc_time = datetime.fromisoformat(doc["timestamp"])
                    if doc_time < cutoff_time:
                        continue
                
                # Simple relevance: check if query words are in text
                if any(word in doc["text"].lower() for word in query_lower.split()):
                    results.append({
                        "document": doc["text"],
                        "metadata": doc["metadata"],
                        "id": doc["id"]
                    })
                    if pid:
                        seen_patients.add(pid)
                
                if len(results) >= n_results:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_recent_records(
        self, 
        patient_id: str, 
        hours: int = 1,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent records for a patient"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            results = []
            for doc in self.documents:
                if doc["metadata"].get("patient_id") == patient_id:
                    doc_time = datetime.fromisoformat(doc["timestamp"])
                    if doc_time >= cutoff_time:
                        results.append({
                            "document": doc["text"],
                            "metadata": doc["metadata"]
                        })
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent records: {e}")
            return []
    
    def _cleanup_old_documents(self):
        """Remove documents older than retention period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=DATA_RETENTION_HOURS)
            self.documents = [
                doc for doc in self.documents
                if datetime.fromisoformat(doc["timestamp"]) >= cutoff_time
            ]
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
    
    def cleanup_old_records(self):
        """Public method to cleanup old records"""
        self._cleanup_old_documents()
        logger.info(f"Cleanup complete. {len(self.documents)} documents remaining")
    
    def get_document_count(self) -> int:
        """Get total document count"""
        return len(self.documents)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'name': 'in-memory-store',
            'count': self.get_document_count(),
            'mode': 'lightweight',
            'retention_hours': DATA_RETENTION_HOURS
        }

