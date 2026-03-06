"""
LangChain RAG Chat Agent for Medical Q&A
Uses Gemini or OpenAI to generate contextual medical responses
"""
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

logger = logging.getLogger("rag-chat-agent")


class RAGChatAgent:
    """
    LangChain-powered agent for intelligent medical Q&A with RAG
    """
    
    def __init__(self):
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        self.llm = self._initialize_llm()
        self.chat_chain = self._create_chat_chain()
        logger.info(f"RAG Chat Agent initialized with {self.llm_provider.upper()}")
    
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        if self.llm_provider == "gemini":
            if not self.gemini_api_key:
                logger.warning("GEMINI_API_KEY not set, using mock responses")
                return None
            
            return ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
                google_api_key=self.gemini_api_key,
                temperature=0.3,  # Lower temperature for medical accuracy
                max_output_tokens=500
            )
        
        elif self.llm_provider == "openai":
            if not self.openai_api_key:
                logger.warning("OPENAI_API_KEY not set, using mock responses")
                return None
            
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=self.openai_api_key,
                temperature=0.3,
                max_tokens=500
            )
        
        else:
            logger.warning(f"Unknown LLM provider: {self.llm_provider}, using mock responses")
            return None
    
    def _create_chat_chain(self) -> Optional[LLMChain]:
        """Create LangChain prompt chain for medical Q&A"""
        if self.llm is None:
            return None
        
        prompt_template = """You are an intelligent ICU monitoring assistant helping doctors understand patient data.

Patient Context:
Target: {patient_id}
Question: {question}

Retrieved Monitoring Data:
{retrieved_context}

Instructions:
1. Answer the doctor's question using ONLY the provided monitoring data above
2. If Target is "ALL PATIENTS", you MUST list ALL unique patients mentioned in the data
3. For each patient, include: Patient ID, HR, BP, SpO2, and any alerts
4. Format multi-patient responses as a numbered list (one patient per line)
5. For single patient queries, provide detailed trends and analysis
6. Be precise and clinical - use exact values from the data
7. Keep response under 300 words
8. Do not speculate or make up information not in the data

Response:"""
        
        prompt = PromptTemplate(
            input_variables=["patient_id", "question", "retrieved_context"],
            template=prompt_template
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def generate_response(
        self,
        patient_id: Optional[str],
        question: str,
        retrieved_context: List[Dict]
    ) -> str:
        """
        Generate LLM response based on RAG context
        
        Args:
            patient_id: Patient identifier (optional - None for general queries)
            question: User's question
            retrieved_context: List of retrieved monitoring events from RAG
        
        Returns:
            Generated response string
        """
        
        # If no LLM available, return template response
        if self.llm is None or self.chat_chain is None:
            return self._generate_fallback_response(patient_id, retrieved_context)
        
        # Format retrieved context for LLM
        context_text = self._format_context(retrieved_context)
        
        try:
            # Generate response using LangChain
            response = self.chat_chain.run(
                patient_id=patient_id or "ALL PATIENTS",
                question=question,
                retrieved_context=context_text
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_response(patient_id, retrieved_context)
    
    def _format_context(self, retrieved_context: List[Dict]) -> str:
        """Format retrieved context for LLM prompt"""
        if not retrieved_context:
            return "No recent monitoring data available."
        
        context_lines = []
        for i, item in enumerate(retrieved_context[:10], 1):  # Increased from 5 to 10 for multi-patient queries
            text = item.get('text', '')
            score = item.get('relevance_score', item.get('score', 0))
            patient_id = item.get('patient_id', '')
            
            # Include patient_id in context if present (for multi-patient queries)
            if patient_id:
                context_lines.append(f"[{i}] Patient {patient_id}: {text} (Relevance: {score:.2f})")
            else:
                context_lines.append(f"[{i}] {text} (Relevance: {score:.2f})")
        
        return "\n".join(context_lines)
    
    def _generate_fallback_response(self, patient_id: Optional[str], retrieved_context: List[Dict]) -> str:
        """Fallback response when LLM is not available"""
        if not retrieved_context:
            if patient_id:
                return f"No recent data available for patient {patient_id} to answer this question."
            else:
                return "No recent patient data available to answer this question."
        
        context_summary = retrieved_context[0]['text'] if retrieved_context else "No context"
        
        response = f"Based on recent monitoring data: {context_summary}. "
        response += "This data shows the patient's physiological trends. "
        response += "Clinical interpretation should be performed by qualified medical staff."
        
        return response


# Global instance
_chat_agent = None

def get_chat_agent() -> RAGChatAgent:
    """Get or create global chat agent instance"""
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = RAGChatAgent()
    return _chat_agent
