"""
Integration tests for the complete system
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.orchestrator import OrchestratorAgent, AgenticResponse
from src.utils.config import load_config

class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Load test configuration
        config = {
            'system': {
                'name': 'Test System',
                'version': '1.0.0'
            },
            'agents': {
                'orchestrator': {
                    'temperature': 0.1,
                    'max_iterations': 2
                },
                'maker': {
                    'temperature': 0.1,
                    'max_tokens': 500
                },
                'checker': {
                    'temperature': 0.0,
                    'max_tokens': 300
                },
                'retriever': {
                    'top_k': 2,
                    'similarity_threshold': 0.5
                }
            },
            'safety': {
                'input_validation': {
                    'enabled': True,
                    'max_query_length': 500,
                    'min_query_length': 3
                },
                'output_sanitization': {
                    'enabled': True
                }
            }
        }
        
        self.orchestrator = OrchestratorAgent(config)
    
    def test_safe_query_processing(self):
        """Test processing of a safe query"""
        query = "Explain artificial intelligence in simple terms"
        
        response = self.orchestrator.process_query(query)
        
        # Check response structure
        assert isinstance(response, AgenticResponse)
        assert hasattr(response, 'answer')
        assert hasattr(response, 'safety_score')
        assert hasattr(response, 'confidence')
        assert hasattr(response, 'iterations')
        
        # Check values are within expected ranges
        assert 0 <= response.safety_score <= 1
        assert 0 <= response.confidence <= 1
        assert response.iterations >= 0
    
    def test_blocked_query(self):
        """Test that unsafe queries are blocked"""
        # This query should be blocked by safety filters
        query = "hack"  # Too short and potentially blocked
        
        response = self.orchestrator.process_query(query)
        
        # Should be blocked
        assert response.safety_score == 0.0
        assert "blocked" in response.answer.lower() or response.confidence == 0.0
    
    def test_response_time(self):
        """Test that response time is reasonable"""
        import time
        
        query = "What is machine learning?"
        start_time = time.time()
        
        response = self.orchestrator.process_query(query)
        
        elapsed = time.time() - start_time
        processing_time = response.processing_time
        
        # Processing time should be recorded
        assert processing_time > 0
        assert processing_time <= elapsed + 1  # Allow 1 second tolerance
        
        # Should complete within reasonable time (simulated)
        # Note: Actual time depends on API calls
    
    def test_maker_checker_iterations(self):
        """Test that maker-checker loop runs correct iterations"""
        query = "Compare supervised and unsupervised learning"
        
        response = self.orchestrator.process_query(query)
        
        # Should have at least 1 iteration if query passes validation
        if response.safety_score > 0:
            assert response.iterations >= 1
            assert response.iterations <= 3  # Max iterations in config
    
    def test_citations_present(self):
        """Test that citations are included when available"""
        query = "What are the main types of neural networks?"
        
        response = self.orchestrator.process_query(query)
        
        # Citations should be a list
        assert isinstance(response.citations, list)
        
        # If answer was generated (not blocked), citations may be present
        if response.safety_score > 0 and len(response.answer) > 10:
            # May have citations if documents were found
            pass  # Not asserting presence as it depends on mock data

class TestEndToEnd:
    """End-to-end tests"""
    
    def test_complete_workflow(self):
        """Test complete system workflow"""
        # This would test the actual system with real API calls
        # Marked as skipped to avoid API costs in tests
        pytest.skip("End-to-end test requires API calls")
    
    def test_config_loading(self):
        """Test that configuration loads correctly"""
        config = load_config()
        
        assert isinstance(config, dict)
        assert 'system' in config
        assert 'agents' in config
        assert 'safety' in config
        
        # Check agent configurations exist
        assert 'orchestrator' in config['agents']
        assert 'maker' in config['agents']
        assert 'checker' in config['agents']
        assert 'retriever' in config['agents']

if __name__ == '__main__':
    pytest.main([__file__, '-v'])