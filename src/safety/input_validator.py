"""
Input validation for safety
"""
import re
import logging
from typing import Dict, List, Any
import yaml

logger = logging.getLogger(__name__)

class InputValidator:
    """Validates user input for safety"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blocked_patterns = []
        self.sensitive_topics = []
        self.load_safety_rules()
        
        # Validation parameters
        self.max_length = config.get('input_validation', {}).get('max_query_length', 1000)
        self.min_length = config.get('input_validation', {}).get('min_query_length', 3)
        
        logger.info("InputValidator initialized")
    
    def load_safety_rules(self):
        """Load safety rules from configuration file"""
        try:
            rules_path = self.config.get('input_validation', {}).get('blocked_patterns_file', 'config/safety_rules.yaml')
            with open(rules_path, 'r') as f:
                rules = yaml.safe_load(f)
            
            safety_rules = rules.get('safety_rules', {})
            self.blocked_patterns = safety_rules.get('blocked_queries', {}).get('patterns', [])
            self.blocked_keywords = safety_rules.get('blocked_queries', {}).get('keywords', [])
            self.sensitive_topics = safety_rules.get('sensitive_topics', {}).get('require_caution', [])
            
            logger.info(f"Loaded {len(self.blocked_patterns)} patterns and {len(self.blocked_keywords)} keywords")
            
        except Exception as e:
            logger.error(f"Error loading safety rules: {e}")
            # Fallback to default patterns
            self.blocked_patterns = [
                r".*(hack|break into|unauthorized access).*",
                r".*(bomb|explosive|weapon).*(make|create|build).*"
            ]
            self.blocked_keywords = ["how to hack", "make a bomb"]
    
    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate a user query
        Returns: dict with 'is_valid', 'reasons', and 'passed_checks'
        """
        validation_checks = []
        failed_reasons = []
        
        # Check 1: Length validation
        if len(query) < self.min_length:
            failed_reasons.append(f"Query too short (min {self.min_length} chars)")
        elif len(query) > self.max_length:
            failed_reasons.append(f"Query too long (max {self.max_length} chars)")
        else:
            validation_checks.append("length")
        
        # Check 2: Blocked pattern matching
        for pattern in self.blocked_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                failed_reasons.append(f"Matches blocked pattern: {pattern[:50]}...")
                break
        else:
            validation_checks.append("pattern_matching")
        
        # Check 3: Blocked keyword matching
        for keyword in self.blocked_keywords:
            if keyword.lower() in query.lower():
                failed_reasons.append(f"Contains blocked keyword: {keyword}")
                break
        else:
            validation_checks.append("keyword_check")
        
        # Check 4: Sensitive topic detection
        requires_caution = False
        for topic in self.sensitive_topics:
            if topic.lower() in query.lower():
                requires_caution = True
                validation_checks.append(f"sensitive_topic_{topic}")
                break
        
        # Determine result
        is_valid = len(failed_reasons) == 0
        
        result = {
            'is_valid': is_valid,
            'passed_checks': validation_checks,
            'requires_caution': requires_caution
        }
        
        if not is_valid:
            result['reasons'] = failed_reasons
            logger.warning(f"Query validation failed: {failed_reasons}")
        else:
            
            logger.info(f"Query validation passed with checks: {validation_checks}")
        
        return result