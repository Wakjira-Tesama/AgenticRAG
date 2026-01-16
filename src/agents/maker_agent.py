"""
Maker agent that generates initial answers
"""
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from ..utils.prompts import MAKER_PROMPTS

logger = logging.getLogger(__name__)

class MakerResponse(BaseModel):
    """Structured response from maker agent"""
    answer: str
    citations: List[str]
    reasoning: Optional[str] = None
    confidence: float

class MakerAgent:
    """Agent responsible for generating initial answers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.get('model', 'gpt-4-turbo-preview'),
            temperature=config.get('temperature', 0.1),
            max_tokens=config.get('max_tokens', 1500)
        )
        
        # Prompts
        self.initial_prompt_template = ChatPromptTemplate.from_template(MAKER_PROMPTS['initial_generation'])
        self.refinement_prompt_template = ChatPromptTemplate.from_template(MAKER_PROMPTS['refinement'])
        
        logger.info("MakerAgent initialized")
    
    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate initial answer based on documents
        """
        logger.info(f"Generating initial answer for query: {query[:50]}...")
        
        # Format context from documents
        context = self._format_documents(documents)
        
        # Prepare prompt
        prompt = self.initial_prompt_template.format_messages(
            query=query,
            context=context,
            num_documents=len(documents)
        )
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        # Parse response
        answer_data = self._parse_response(response.content)
        
        # Calculate confidence
        confidence = self._calculate_confidence(answer_data['answer'], documents)
        
        logger.info(f"Initial answer generated (confidence: {confidence:.2%})")
        return {
            'answer': answer_data['answer'],
            'citations': answer_data.get('citations', []),
            'reasoning': answer_data.get('reasoning', ''),
            'confidence': confidence,
            'source_documents': [doc.metadata.get('source', 'Unknown') for doc in documents]
        }
    
    def refine(self, query: str, documents: List[Document], 
               previous_answer: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine answer based on checker feedback
        """
        logger.info(f"Refining answer based on feedback...")
        
        # Format context
        context = self._format_documents(documents)
        
        # Prepare prompt with feedback
        prompt = self.refinement_prompt_template.format_messages(
            query=query,
            context=context,
            previous_answer=previous_answer.get('answer', ''),
            feedback=feedback.get('feedback', ''),
            issues=feedback.get('failed_checks', [])
        )
        
        # Generate refined response
        response = self.llm.invoke(prompt)
        
        # Parse response
        answer_data = self._parse_response(response.content)
        
        # Calculate confidence (slightly higher for refined answers)
        base_confidence = self._calculate_confidence(answer_data['answer'], documents)
        refined_confidence = min(base_confidence * 1.1, 1.0)  # 10% boost
        
        logger.info(f"Answer refined (confidence: {refined_confidence:.2%})")
        return {
            'answer': answer_data['answer'],
            'citations': answer_data.get('citations', []),
            'reasoning': answer_data.get('reasoning', ''),
            'confidence': refined_confidence,
            'is_refined': True,
            'previous_feedback': feedback
        }
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents into a context string"""
        formatted = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', f'Document {i}')
            year = doc.metadata.get('year', 'Unknown')
            formatted.append(f"[Document {i}] Source: {source} ({year})\nContent: {doc.page_content[:500]}...")
        return "\n\n".join(formatted)
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        # Simple parsing - in production, use structured output
        lines = response_text.split('\n')
        answer_lines = []
        citations = []
        reasoning = []
        
        in_answer = True
        in_citations = False
        in_reasoning = False
        
        for line in lines:
            if line.lower().startswith('citations:'):
                in_answer = False
                in_citations = True
                in_reasoning = False
            elif line.lower().startswith('reasoning:'):
                in_answer = False
                in_citations = False
                in_reasoning = True
            elif in_citations and line.strip():
                citations.append(line.strip())
            elif in_reasoning and line.strip():
                reasoning.append(line.strip())
            elif in_answer and line.strip():
                answer_lines.append(line)
        
        return {
            'answer': '\n'.join(answer_lines).strip(),
            'citations': citations,
            'reasoning': '\n'.join(reasoning).strip() if reasoning else None
        }
    
    def _calculate_confidence(self, answer: str, documents: List[Document]) -> float:
        """Calculate confidence score for the answer"""
        # Simple confidence calculation based on answer length and document count
        if not answer or len(documents) == 0:
            return 0.0
        
        # Base confidence from document relevance
        doc_confidence = min(len(documents) / 5.0, 1.0)  # Max confidence with 5+ docs
        
        # Confidence from answer quality (length and structure)
        answer_length = len(answer.split())
        length_confidence = min(answer_length / 200.0, 1.0)  # Good answer ~200 words
        
        # Check for uncertainty phrases
        uncertainty_phrases = [
            "i'm not sure", "i don't know", "cannot answer", 
            "no information", "not enough context"
        ]
        
        uncertainty_penalty = 0
        for phrase in uncertainty_phrases:
            if phrase in answer.lower():
                uncertainty_penalty += 0.3
        
        # Calculate final confidence
        confidence = (doc_confidence * 0.6 + length_confidence * 0.4) - uncertainty_penalty
        return max(0.0, min(1.0, confidence))