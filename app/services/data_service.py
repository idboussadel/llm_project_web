"""
Data Service - Load and manage results, metrics, and examples
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class DataService:
    """Service for loading pre-computed results and metrics"""
    
    def __init__(self, results_path: Path, final_results_json: Path, 
                 qlora_results_json: Path, test_metrics_json: Path,
                 ablation_studies_json: Path = None):
        self.results_path = Path(results_path)
        self.final_results_json = Path(final_results_json)
        self.qlora_results_json = Path(qlora_results_json)
        self.test_metrics_json = Path(test_metrics_json)
        self.ablation_studies_json = Path(ablation_studies_json) if ablation_studies_json else None
    
    @lru_cache(maxsize=1)
    def get_final_results(self) -> Dict:
        """Load final backtesting results"""
        try:
            with open(self.final_results_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load final results: {e}")
            return {}
    
    @lru_cache(maxsize=1)
    def get_sentiment_metrics(self) -> Dict:
        """Load sentiment model evaluation metrics"""
        try:
            with open(self.qlora_results_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load sentiment metrics: {e}")
            return {}
    
    @lru_cache(maxsize=1)
    def get_tft_metrics(self) -> Dict:
        """Load TFT model test metrics"""
        try:
            with open(self.test_metrics_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load TFT metrics: {e}")
            return {}
    
    @lru_cache(maxsize=1)
    def get_ablation_studies(self) -> Dict:
        """Load ablation studies results"""
        if not self.ablation_studies_json or not self.ablation_studies_json.exists():
            logger.warning("Ablation studies JSON not found")
            return {}
        try:
            with open(self.ablation_studies_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ablation studies: {e}")
            return {}
    
    def get_all_metrics(self) -> Dict:
        """Combine all metrics into single response"""
        return {
            'backtesting': self.get_final_results(),
            'sentiment_model': self.get_sentiment_metrics(),
            'tft_model': self.get_tft_metrics()
        }
    
    @staticmethod
    def get_example_texts() -> List[Dict]:
        """Get predefined example texts for demo"""
        return [
            {
                'id': 'news_positive',
                'category': 'News',
                'text': 'Apple Inc. reported record-breaking quarterly earnings, surpassing analyst expectations with a 25% increase in revenue driven by strong iPhone sales and growing services sector.',
                'source': 'Financial News'
            },
            {
                'id': 'news_negative',
                'category': 'News',
                'text': 'Tesla shares plummet 15% as production delays and regulatory concerns weigh on investor sentiment, with analysts downgrading price targets.',
                'source': 'Market Watch'
            },
            {
                'id': 'news_neutral',
                'category': 'News',
                'text': 'Microsoft announced quarterly dividend of $0.68 per share, maintaining its regular payout schedule in line with previous quarters.',
                'source': 'Corporate Announcement'
            },
            {
                'id': 'social_positive',
                'category': 'Social Media',
                'text': '$NVDA absolutely crushing it! New data center chips showing incredible demand. This stock is going to the moon 🚀📈',
                'source': 'Twitter/X'
            },
            {
                'id': 'social_negative',
                'category': 'Social Media',
                'text': '$AAPL looking weak here. Breaking key support levels, volume declining. Time to trim positions before earnings.',
                'source': 'StockTwits'
            },
            {
                'id': 'earnings_positive',
                'category': 'Earnings Call',
                'text': 'We are extremely pleased with our performance this quarter. Our strategic initiatives have yielded exceptional results, with operating margins expanding and customer acquisition accelerating across all segments.',
                'source': 'CEO Transcript'
            },
            {
                'id': 'earnings_negative',
                'category': 'Earnings Call',
                'text': 'While we faced significant headwinds this quarter, including supply chain disruptions and increased competition, we are taking decisive action to restructure operations and reduce costs.',
                'source': 'CFO Transcript'
            },
            {
                'id': 'mixed',
                'category': 'Analysis',
                'text': 'Amazon delivered mixed results with strong AWS growth offsetting weaker retail performance. Management guidance remains cautious amid macroeconomic uncertainty.',
                'source': 'Analyst Report'
            }
        ]


# Global instance will be created by app factory
data_service = None
