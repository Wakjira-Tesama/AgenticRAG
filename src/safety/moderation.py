"""
Content moderation using OpenAI's moderation API
"""
import logging
from typing import Dict, Any, List
from openai import OpenAI

logger = logging.getLogger(__name__)

class ContentModerator:
    """Moderates content using OpenAI's moderation API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = OpenAI()
        
        # Moderation thresholds
        self.thresholds = config.get('moderation_thresholds', {
            'hate': 0.8,
            'hate/threatening': 0.9,
            'self-harm': 0.7,
            'sexual': 0.8,
            'sexual/minors': 1.0,
            'violence': 0.8,
            'violence/graphic': 0.9
        })
        
        logger.info("ContentModerator initialized")
    
    def moderate(self, text: str) -> Dict[str, Any]:
        """
        Moderate text content using OpenAI's moderation API
        Returns: dict with moderation results
        """
        if not text or len(text.strip()) == 0:
            return {
                'flagged': False,
                'categories': {},
                'scores': {},
                'message': 'Empty text'
            }
        
        try:
            # Call OpenAI moderation API
            response = self.client.moderations.create(input=text)
            result = response.results[0]
            
            # Check if any category exceeds threshold
            flagged_categories = []
            category_scores = {}
            
            for category, score in result.category_scores.items():
                category_scores[category] = score
                threshold = self.thresholds.get(category, 0.8)
                
                if score > threshold:
                    flagged_categories.append(category)
            
            is_flagged = len(flagged_categories) > 0
            
            result_dict = {
                'flagged': is_flagged,
                'categories': result.categories,
                'category_scores': category_scores,
                'flagged_categories': flagged_categories,
                'flagged_thresholds': {cat: self.thresholds.get(cat, 0.8) for cat in flagged_categories}
            }
            
            if is_flagged:
                logger.warning(f"Content flagged for categories: {flagged_categories}")
            else:
                logger.info("Content passed moderation")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error during moderation: {e}")
            # Fallback: conservative approach
            return {
                'flagged': True,  # Be conservative on error
                'categories': {},
                'category_scores': {},
                'flagged_categories': ['moderation_error'],
                'error': str(e)
            }
    
    def check_specific_category(self, text: str, category: str) -> float:
        """Check a specific moderation category score"""
        result = self.moderate(text)
        return result.get('category_scores', {}).get(category, 0.0)