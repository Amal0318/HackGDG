import logging
from typing import List, Dict, Any, Optional
from groq import Groq as GroqClient
from app.config import GROQ_API_KEY, GROQ_MODEL
from app.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class PatientHandoffRAG:
    """RAG system for patient handoff queries"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        
        # Initialize Groq client
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set! RAG will not work properly.")
            self.client = None
        else:
            try:
                self.client = GroqClient(api_key=GROQ_API_KEY)
                logger.info(f"Initialized Groq client with model: {GROQ_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
        
        # Prompt template
        self.prompt_template = """You are an AI assistant helping nurses during shift handoffs in an ICU.
Based on the patient data provided below, answer the nurse's question in a clear, structured format.

Patient Data:
{context}

Nurse's Question: {question}

Important Instructions:
- Keep answers brief and focused (3-5 lines maximum per patient)
- Use bullet points or numbered lists for multiple patients
- Only include the most recent and relevant vitals
- Show patient ID clearly
- Include critical information: state, alerts, key vitals only
- If asking about one patient, only show that patient's data
- Use this format for each patient:
  Patient [ID]: [Status] - HR: [value], BP: [value], SpO2: [value]%

Answer:"""
    
    def query(
        self, 
        question: str,
        patient_id: Optional[str] = None,
        time_window_hours: int = 4,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """Process a nurse's query using RAG"""
        
        try:
            # Adjust n_results based on whether specific patient is queried
            if patient_id:
                n_results = min(n_results, 5)  # Limit to 5 for specific patient
            else:
                n_results = min(n_results, 8)  # Limit to 8 for general queries
            
            # Step 1: Retrieve relevant context from vector store
            search_results = self.vector_store.search(
                query=question,
                n_results=n_results,
                patient_id=patient_id,
                time_window_hours=time_window_hours
            )
            
            if not search_results:
                return {
                    'answer': "No recent patient data found for this query.",
                    'context_used': [],
                    'success': False
                }
            
            # Step 2: Format context for LLM
            context_texts = [result['document'] for result in search_results]
            context = "\n".join([f"- {text}" for text in context_texts])
            
            # Step 3: Generate response with LLM
            if self.client is None:
                # Fallback: return context without LLM processing
                return {
                    'answer': f"RAG service running without LLM. Retrieved context:\n{context}",
                    'context_used': context_texts,
                    'success': True,
                    'note': 'LLM not configured - showing raw context'
                }
            
            # Use Groq client directly
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            answer = response.choices[0].message.content
            
            return {
                'answer': answer.strip(),
                'context_used': context_texts,
                'success': True,
                'sources': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'context_used': [],
                'success': False
            }
    
    def get_patient_summary(
        self, 
        patient_id: str,
        hours: int = 4
    ) -> Dict[str, Any]:
        """Generate a summary of patient's recent status"""
        
        try:
            # Get recent records
            records = self.vector_store.get_recent_records(
                patient_id=patient_id,
                hours=hours,
                limit=20
            )
            
            if not records:
                return {
                    'summary': f"No data found for patient {patient_id} in the last {hours} hours.",
                    'success': False
                }
            
            # Format context
            context = "\n".join([f"- {record['document']}" for record in records])
            
            # Generate summary
            if self.client is None:
                return {
                    'summary': f"Recent activity for patient {patient_id}:\n{context}",
                    'success': True,
                    'note': 'LLM not configured - showing raw data'
                }
            
            summary_prompt = f"""Summarize the patient's status based on this data:

{context}

Provide a concise shift handoff summary including:
1. Overall status (stable/declining/improving)
2. Key vital sign trends
3. Any alerts or concerns
4. Notable changes

Summary:"""
            
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.1
            )
            summary = response.choices[0].message.content
            
            return {
                'summary': summary.strip(),
                'record_count': len(records),
                'time_window_hours': hours,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'summary': f"Error generating summary: {str(e)}",
                'success': False
            }
    
    def get_trending_vitals(
        self,
        patient_id: str,
        vital_name: str,
        hours: int = 2
    ) -> Dict[str, Any]:
        """Get trending information for a specific vital sign"""
        
        query = f"What is the trend for {vital_name} for patient {patient_id} in the last {hours} hours?"
        
        return self.query(
            question=query,
            patient_id=patient_id,
            time_window_hours=hours,
            n_results=15
        )
