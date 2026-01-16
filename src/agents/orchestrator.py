"""
Main orchestrator agent that coordinates all other agents
"""
import time
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from ..safety.input_validator import InputValidator
from ..safety.output_sanitizer import OutputSanitizer
from .maker_agent import MakerAgent
from .checker_agent import CheckerAgent
from .retriever_agent import RetrieverAgent
from ..utils.prompts import SYSTEM_PROMPTS
from ..utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

class AgenticResponse(BaseModel):
    """Structured response from agentic system"""
    answer: str = Field(..., description="Final answer")
    citations: List[str] = Field(default_factory=list, description="Source citations")
    safety_score: float = Field(..., ge=0.0, le=1.0, description="Safety score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time: float = Field(..., description="Total processing time in seconds")
    iterations: int = Field(..., description="Number of maker-checker iterations")
    validation_passed: List[str] = Field(default_factory=list, description="Passed validations")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")

class OrchestratorAgent:
    """Main orchestrator that manages the agentic workflow"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatOpenAI(
            model=config['agents']['orchestrator'].get('model', 'gpt-4-turbo-preview'),
            temperature=config['agents']['orchestrator'].get('temperature', 0.1)
        )
        
        # Initialize all agents
        self.input_validator = InputValidator(config['safety'])
        self.output_sanitizer = OutputSanitizer(config['safety'])
        self.retriever = RetrieverAgent(config['agents']['retriever'])
        self.maker = MakerAgent(config['agents']['maker'])
        self.checker = CheckerAgent(config['agents']['checker'])
        
        # System prompts
        self.system_prompt = SYSTEM_PROMPTS['orchestrator']
        
        logger.info("OrchestratorAgent initialized")
    
    def process_query(self, query: str) -> AgenticResponse:
        """
        Main processing pipeline for user queries
        """
        start_time = time.time()
        
        # Step 1: Input validation
        logger.info(f"Processing query: {query[:50]}...")
        validation_result = self.input_validator.validate(query)
        
        if not validation_result['is_valid']:
            logger.warning(f"Query blocked: {validation_result.get('reasons', [])}")
            return AgenticResponse(
                answer=f"Query blocked by safety system. Reason: {', '.join(validation_result.get('reasons', []))}",
                citations=[],
                safety_score=0.0,
                confidence=0.0,
                processing_time=time.time() - start_time,
                iterations=0,
                validation_passed=[],
                warnings=["Query blocked by safety filters"]
            )
        
        # Step 2: Document retrieval
        logger.info("Retrieving relevant documents...")
        documents = self.retriever.retrieve(query)
        
        if not documents:
            logger.warning("No relevant documents found")
            return AgenticResponse(
                answer="I couldn't find enough relevant information to answer your question. Please try rephrasing or asking about a different topic.",
                citations=[],
                safety_score=1.0,
                confidence=0.0,
                processing_time=time.time() - start_time,
                iterations=0,
                validation_passed=["input_validation"],
                warnings=["No relevant documents found"]
            )
        
        # Step 3: Maker-Checker loop
        logger.info("Starting maker-checker loop...")
        final_answer = self._maker_checker_loop(query, documents)
        
        # Step 4: Output sanitization
        logger.info("Applying output sanitization...")
        sanitized_answer = self.output_sanitizer.sanitize(final_answer['answer'])
        
        # Step 5: Calculate metrics
        metrics = calculate_metrics(
            query=query,
            answer=sanitized_answer,
            documents=documents,
            iterations=final_answer['iterations']
        )
        
        # Prepare response
        response = AgenticResponse(
            answer=sanitized_answer,
            citations=final_answer.get('citations', []),
            safety_score=metrics['safety_score'],
            confidence=metrics['confidence'],
            processing_time=time.time() - start_time,
            iterations=final_answer['iterations'],
            validation_passed=validation_result.get('passed_checks', []),
            warnings=final_answer.get('warnings', [])
        )
        
        logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
        return response
    
    def _maker_checker_loop(self, query: str, documents: List[Document], max_iterations: int = 3) -> Dict[str, Any]:
        """
        Execute iterative maker-checker refinement loop
        """
        iteration = 0
        current_answer = None
        feedback_history = []
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Maker-Checker iteration {iteration}/{max_iterations}")
            
            # Maker phase
            if iteration == 1:
                # Initial generation
                maker_result = self.maker.generate(query, documents)
            else:
                # Refinement based on feedback
                maker_result = self.maker.refine(
                    query=query,
                    documents=documents,
                    previous_answer=current_answer,
                    feedback=feedback_history[-1]
                )
            
            # Checker phase
            checker_result = self.checker.evaluate(
                query=query,
                answer=maker_result['answer'],
                documents=documents,
                citations=maker_result.get('citations', [])
            )
            
            # Update current state
            current_answer = maker_result
            feedback_history.append(checker_result)
            
            # Check if answer passes all checks
            if checker_result.get('passes_all_checks', False):
                logger.info(f"Answer passed all checks at iteration {iteration}")
                current_answer['iterations'] = iteration
                current_answer['checker_feedback'] = checker_result
                return current_answer
            
            logger.info(f"Iteration {iteration} failed checks: {checker_result.get('failed_checks', [])}")
        
        # Max iterations reached - return best attempt
        logger.warning(f"Max iterations ({max_iterations}) reached")
        current_answer['iterations'] = max_iterations
        current_answer['checker_feedback'] = feedback_history[-1]
        current_answer['warnings'] = [
            f"Max iterations reached. Failed checks: {feedback_history[-1].get('failed_checks', [])}"
        ]
        return current_answer