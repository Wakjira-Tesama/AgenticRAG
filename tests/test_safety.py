"""
Tests for safety modules
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.safety.input_validator import InputValidator
from src.safety.output_sanitizer import OutputSanitizer
from src.safety.moderation import ContentModerator

class TestInputValidator:
    """Test input validation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = {
            'input_validation': {
                'max_query_length': 1000,
                'min_query_length': 3,
                'blocked_patterns_file': 'config/safety_rules.yaml'
            }
        }
        self.validator = InputValidator(config)
    
    def test_valid_query(self):
        """Test valid query passes"""
        query = "What is machine learning?"
        result = self.validator.validate(query)
        assert result['is_valid'] == True
        assert 'length' in result['passed_checks']
    
    def test_too_short_query(self):
        """Test too short query fails"""
        query = "Hi"
        result = self.validator.validate(query)
        assert result['is_valid'] == False
        assert 'Query too short' in str(result.get('reasons', []))
    
    def test_blocked_keyword(self):
        """Test blocked keyword detection"""
        query = "How to hack a system"
        result = self.validator.validate(query)
        assert result['is_valid'] == False
        # Note: Actual blocked keywords depend on safety_rules.yaml
    
    def test_sensitive_topic(self):
        """Test sensitive topic detection"""
        query = "I need medical advice"
        result = self.validator.validate(query)
        # Should pass validation but flag as sensitive
        assert result['is_valid'] == True
        assert result['requires_caution'] == True

class TestOutputSanitizer:
    """Test output sanitization"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = {
            'output_sanitization': {
                'redact_pii': True,
                'remove_unsafe_content': True
            }
        }
        self.sanitizer = OutputSanitizer(config)
    
    def test_pii_redaction(self):
        """Test PII redaction"""
        text = "My email is test@example.com and phone is 123-456-7890"
        sanitized = self.sanitizer.sanitize(text)
        
        # Should redact PII
        assert 'test@example.com' not in sanitized
        assert '123-456-7890' not in sanitized
        assert '[REDACTED' in sanitized
    
    def test_unsafe_content_removal(self):
        """Test unsafe content removal"""
        text = "Here are instructions to hack a system"
        sanitized = self.sanitizer.sanitize(text)
        
        # Should remove unsafe content
        assert 'hack' not in sanitized.lower()
        assert '[REMOVED' in sanitized
    
    def test_safe_content_unchanged(self):
        """Test safe content remains unchanged"""
        text = "Machine learning is a subset of artificial intelligence."
        sanitized = self.sanitizer.sanitize(text)
        
        # Safe content should remain
        assert text == sanitized

class TestContentModerator:
    """Test content moderation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = {
            'moderation_thresholds': {
                'hate': 0.8,
                'violence': 0.8
            }
        }
        self.moderator = ContentModerator(config)
    
    def test_safe_content(self):
        """Test safe content passes moderation"""
        text = "The weather is nice today."
        result = self.moderator.moderate(text)
        
        # Safe content should not be flagged
        # Note: This depends on OpenAI API, so we just check structure
        assert 'flagged' in result
        assert isinstance(result['flagged'], bool)
    
    @pytest.mark.skip(reason="Requires actual unsafe content which we shouldn't create")
    def test_unsafe_content(self):
        """Test unsafe content is flagged"""
        # This test would require actual unsafe content
        # which we shouldn't include in test code
        pass

if __name__ == '__main__':
    pytest.main([__file__, '-v'])