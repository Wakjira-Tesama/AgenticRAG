"""
Checker agent that validates answers for safety, accuracy, and completeness
"""
import logging
from typing import Dict, List, Any, Tuple
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from ..utils.prompts import CHECKER_PROMPTS
from ..safety.moderation import ContentModerator

logger = logging.getLogger(__name__)

class CheckResult(BaseModel):
    """Structured result from checker agent"""
    passes_all_checks: bool
    feedback: str
    passed_checks: List[str]
    failed_checks: List[str]
    safety_score: float
    accuracy_score: float
    completeness_score: float

class CheckerAgent:
    """Agent responsible for checking answers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.get('model', 'gpt-4-turbo-preview'),
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens', 1000)
        )
        
        # Initialize content moderator
        self.moderator = ContentModerator(config.get('moderation', {}))
        
        # Prompts
        self.safety_prompt = ChatPromptTemplate.from_template(CHECKER_PROMPTS['safety_check'])
        self.accuracy_prompt = ChatPromptTemplate.from_template(CHECKER_PROMPTS['accuracy_check'])
        self.completeness_prompt = ChatPromptTemplate.from_template(CHECKER_PROMPTS['completeness_check'])
        self.feedback_prompt = ChatPromptTemplate.from_template(CHECKER_PROMPTS['generate_feedback'])
        
        logger.info("CheckerAgent initialized")
    
    def evaluate(self, query: str, answer: str, documents: List[Document], 
                 citations: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate answer on multiple dimensions
        """
        logger.info(f"Evaluating answer for query: {query[:50]}...")
        
        # Run all checks in parallel (conceptually)
        safety_result = self._check_safety(query, answer)
        accuracy_result = self._check_accuracy(query, answer, documents, citations)
        completeness_result = self._check_completeness(query, answer, documents)
        
        # Compile results
        passed_checks = []
        failed_checks = []
        
        if safety_result['passes']:
            passed_checks.append('safety')
        else:
            failed_checks.append('safety')
        
        if accuracy_result['passes']:
            passed_checks.append('accuracy')
        else:
            failed_checks.append('accuracy')
        
        if completeness_result['passes']:
            passed_checks.append('completeness')
        else:
            failed_checks.append('completeness')
        
        # Generate feedback
        feedback = self._generate_feedback(
            query=query,
            answer=answer,
            safety_result=safety_result,
            accuracy_result=accuracy_result,
            completeness_result=completeness_result
        )
        
        # Calculate overall scores
        safety_score = safety_result['score']
        accuracy_score = accuracy_result['score']
        completeness_score = completeness_result['score']
        overall_score = (safety_score * 0.4 + accuracy_score * 0.4 + completeness_score * 0.2)
        
        result = CheckResult(
            passes_all_checks=len(failed_checks) == 0,
            feedback=feedback,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            safety_score=safety_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score
        )
        
        logger.info(f"Evaluation complete: {len(passed_checks)} passed, {len(failed_checks)} failed")
        return result.dict()
    
    def _check_safety(self, query: str, answer: str) -> Dict[str, Any]:
        """Check for safety violations"""
        # Use content moderator
        moderation_result = self.moderator.moderate(answer)
        
        # LLM-based safety check
        prompt = self.safety_prompt.format_messages(
            query=query,
            answer=answer,
            moderation_result=moderation_result
        )
        
        response = self.llm.invoke(prompt)
        
        # Parse safety check result
        lines = response.content.split('\n')
        passes = False
        score = 0.0
        
        for line in lines:
            if 'PASS' in line.upper():
                passes = True
            if 'SCORE:' in line.upper():
                try:
                    score_str = line.split(':')[1].strip().replace('%', '')
                    score = float(score_str) / 100.0
                except (ValueError, IndexError):
                    score = 0.0
        
        # If content moderation flagged, override
        if moderation_result.get('flagged', False):
            passes = False
            score = 0.0
        
        return {
            'passes': passes,
            'score': score,
            'moderation_result': moderation_result,
            'details': response.content
        }
    
    def _check_accuracy(self, query: str, answer: str, documents: List[Document], 
                       citations: List[str]) -> Dict[str, Any]:
        """Check answer accuracy against documents"""
        # Format document content for comparison
        document_texts = [doc.page_content[:500] for doc in documents]
        
        prompt = self.accuracy_prompt.format_messages(
            query=query,
            answer=answer,
            documents='\n---\n'.join(document_texts),
            citations='\n'.join(citations) if citations else 'No citations'
        )
        
        response = self.llm.invoke(prompt)
        
        # Parse accuracy check result
        lines = response.content.split('\n')
        passes = False
        score = 0.0
        
        for line in lines:
            if 'ACCURATE' in line.upper() and 'NOT' not in line.upper():
                passes = True
            if 'ACCURACY SCORE:' in line.upper():
                try:
                    score_str = line.split(':')[1].strip().replace('%', '')
                    score = float(score_str) / 100.0
                except (ValueError, IndexError):
                    score = 0.0
        
        return {
            'passes': passes,
            'score': score,
            'details': response.content
        }
    
    def _check_completeness(self, query: str, answer: str, documents: List[Document]) -> Dict[str, Any]:
        """Check if answer completely addresses the query"""
        # Extract key terms from query
        query_terms = set(query.lower().split())
        
        prompt = self.completeness_prompt.format_messages(
            query=query,
            answer=answer,
            query_terms=', '.join(list(query_terms)[:10])  # First 10 terms
        )
        
        response = self.llm.invoke(prompt)
        
        # Parse completeness check result
        lines = response.content.split('\n')
        passes = False
        score = 0.0
        
        for line in lines:
            if 'COMPLETE' in line.upper() and 'NOT' not in line.upper():
                passes = True
            if 'COMPLETENESS SCORE:' in line.upper():
                try:
                    score_str = line.split(':')[1].strip().replace('%', '')
                    score = float(score_str) / 100.0
                except (ValueError, IndexError):
                    score = 0.0
        
        return {
            'passes': passes,
            'score': score,
            'details': response.content
        }
    
    def _generate_feedback(self, query: str, answer: str, **check_results) -> str:
        """Generate constructive feedback based on check results"""
        prompt = self.feedback_prompt.format_messages(
            query=query,
            answer=answer,
            safety_result=check_results.get('safety_result', {}),
            accuracy_result=check_results.get('accuracy_result', {}),
            completeness_result=check_results.get('completeness_result', {})
        )
        
        response = self.llm.invoke(prompt)
        return response.content