"""
Tests for RAG functionality
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.retriever_agent import RetrieverAgent
from src.agents.maker_agent import MakerAgent
from src.agents.checker_agent import CheckerAgent
from langchain.schema import Document

class TestRetrieverAgent:
    """Test retriever agent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = {
            'top_k': 3,
            'similarity_threshold': 0.5,
            'embedding_model': 'text-embedding-3-small'
        }
        self.retriever = RetrieverAgent(config)
    
    def test_retrieve_mock(self):
        """Test document retrieval (mock test)"""
        # This is a mock test since we don't have actual vector store
        query = "artificial intelligence"
        
        # Mock the vector store search
        class MockVectorStore:
            def similarity_search_with_relevance_scores(self, query, k):
                return [
                    (Document(page_content="AI is transforming industries", 
                              metadata={'source': 'doc1'}), 0.9),
                    (Document(page_content="Machine learning is a subset of AI", 
                              metadata={'source': 'doc2'}), 0.8),
                    (Document(page_content="Deep learning uses neural networks", 
                              metadata={'source': 'doc3'}), 0.7)
                ]
        
        self.retriever.vector_store = MockVectorStore()
        documents = self.retriever.retrieve(query)
        
        assert len(documents) == 3
        assert documents[0].metadata['source'] == 'doc1'
    
    def test_empty_query(self):
        """Test retrieval with empty query"""
        query = ""
        documents = self.retriever.retrieve(query)
        
        # Should handle empty query gracefully
        assert isinstance(documents, list)

class TestMakerAgent:
    """Test maker agent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = {
            'temperature': 0.1,
            'max_tokens': 500,
            'model': 'gpt-4-turbo-preview'
        }
        self.maker = MakerAgent(config)
    
    def test_generate_answer(self):
        """Test answer generation"""
        query = "What is AI?"
        documents = [
            Document(page_content="Artificial Intelligence refers to machines that can perform tasks requiring human intelligence.", 
                     metadata={'source': 'AI Textbook'}),
            Document(page_content="AI includes machine learning, natural language processing, and computer vision.", 
                     metadata={'source': 'Research Paper'})
        ]
        
        result = self.maker.generate(query, documents)
        
        assert 'answer' in result
        assert 'citations' in result
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
    
    def test_empty_documents(self):
        """Test generation with empty documents"""
        query = "What is AI?"
        documents = []
        
        result = self.maker.generate(query, documents)
        
        assert 'answer' in result
        assert result['confidence'] == 0.0  # Should have zero confidence

class TestCheckerAgent:
    """Test checker agent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = {
            'temperature': 0.0,
            'max_tokens': 1000,
            'model': 'gpt-4-turbo-preview',
            'moderation': {}
        }
        self.checker = CheckerAgent(config)
    
    def test_evaluation_structure(self):
        """Test evaluation returns correct structure"""
        query = "What is AI?"
        answer = "AI stands for Artificial Intelligence, which is machines performing intelligent tasks."
        documents = [
            Document(page_content="AI means Artificial Intelligence", 
                     metadata={'source': 'doc1'})
        ]
        
        result = self.checker.evaluate(query, answer, documents)
        
        assert 'passes_all_checks' in result
        assert 'feedback' in result
        assert 'passed_checks' in result
        assert 'failed_checks' in result
        assert 'safety_score' in result
        assert 0 <= result['safety_score'] <= 1
    
    def test_empty_answer(self):
        """Test evaluation of empty answer"""
        query = "What is AI?"
        answer = ""
        documents = []
        
        result = self.checker.evaluate(query, answer, documents)
        
        # Should handle empty answer
        assert isinstance(result, dict)
        assert result['safety_score'] == 0.0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])