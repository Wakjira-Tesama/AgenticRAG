"""
Output sanitization for safety
"""
import re
import logging
from typing import Dict, Any, List
import yaml

logger = logging.getLogger(__name__)

class OutputSanitizer:
    """Sanitizes outputs to remove unsafe content"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load PII patterns
        self.pii_patterns = []
        self.load_pii_patterns()
        
        # Sanitization settings
        sanitization_config = config.get('output_sanitization', {})
        self.redact_pii = sanitization_config.get('redact_pii', True)
        self.remove_unsafe = sanitization_config.get('remove_unsafe_content', True)
        
        logger.info("OutputSanitizer initialized")
    
    def load_pii_patterns(self):
        """Load PII patterns from config"""
        try:
            # Try to load from safety rules
            rules_path = self.config.get('input_validation', {}).get('blocked_patterns_file', 'config/safety_rules.yaml')
            with open(rules_path, 'r') as f:
                rules = yaml.safe_load(f)
            
            # If no patterns in rules, use defaults
            pii_config = rules.get('safety_rules', {}).get('pii_patterns', [])
            
            if pii_config:
                self.pii_patterns = pii_config
            else:
                # Default PII patterns
                self.pii_patterns = [
                    {'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'},
                    {'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'},
                    {'ssn': r'\b\d{3}-\d{2}-\d{4}\b'},
                    {'credit_card': r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b'}
                ]
            
            logger.info(f"Loaded {len(self.pii_patterns)} PII patterns")
            
        except Exception as e:
            logger.error(f"Error loading PII patterns: {e}")
            # Fallback to default patterns
            self.pii_patterns = [
                {'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'},
                {'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'}
            ]
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize text by removing PII and unsafe content
        """
        if not text:
            return text
        
        original_text = text
        
        # Step 1: Redact PII
        if self.redact_pii:
            text = self._redact_pii(text)
        
        # Step 2: Remove unsafe content
        if self.remove_unsafe:
            text = self._remove_unsafe_content(text)
        
        # Step 3: Add disclaimer if modified
        if text != original_text:
            disclaimer = "\n\n[Note: Some content has been redacted for safety.]"
            if disclaimer not in text:
                text += disclaimer
        
        logger.info(f"Sanitized output (original: {len(original_text)} chars, sanitized: {len(text)} chars)")
        return text
    
    def _redact_pii(self, text: str) -> str:
        """Redact Personally Identifiable Information"""
        for pattern_dict in self.pii_patterns:
            for pii_type, pattern in pattern_dict.items():
                text = re.sub(pattern, f'[REDACTED_{pii_type.upper()}]', text, flags=re.IGNORECASE)
        return text
    
    def _remove_unsafe_content(self, text: str) -> str:
        """Remove potentially unsafe content"""
        unsafe_patterns = [
            # Dangerous instructions
            (r'(?i)(step\s*\d+[:.]?\s*)?(how\s+to|instructions?\s+to)\s+(make|create|build|hack|breach).*', 
             '[REMOVED: Potentially dangerous instructions]'),
            
            # Explicit content markers
            (r'(?i)(explicit|graphic|violent|disturbing).*(content|material|description)', 
             '[REMOVED: Explicit content]'),
            
            # Self-harm references
            (r'(?i)(self[-\s]?harm|suicide|self[-\s]?injury).*(method|technique|way)', 
             '[REMOVED: Self-harm content]')
        ]
        
        for pattern, replacement in unsafe_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text