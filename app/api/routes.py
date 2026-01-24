"""
API Routes Blueprint - REST endpoints for sentiment analysis
"""
import logging
import time
import threading
from functools import wraps
from flask import Blueprint, request, jsonify, current_app
from app.services.model_service import sentiment_service
from app.services.data_service import DataService
from app.services.trading_service import trading_signal_service

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Simple in-memory cache for signal results (TTL: 5 minutes)
_signal_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 300  # 5 minutes

def get_cached_result(ticker: str):
    """Get cached result if still valid"""
    with _cache_lock:
        if ticker in _signal_cache:
            result, timestamp = _signal_cache[ticker]
            if time.time() - timestamp < CACHE_TTL:
                return result
            else:
                # Expired, remove it
                del _signal_cache[ticker]
    return None

def set_cached_result(ticker: str, result):
    """Cache a result"""
    with _cache_lock:
        _signal_cache[ticker] = (result, time.time())
        # Clean up old entries if cache gets too large (>100 entries)
        if len(_signal_cache) > 100:
            # Remove oldest 20% of entries
            sorted_items = sorted(_signal_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[:20]:
                del _signal_cache[key]


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_info = sentiment_service.get_model_info()
    return jsonify({
        'status': 'healthy',
        'model': model_info
    })


@api_bp.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """
    Analyze sentiment of input text
    
    Request Body:
        {
            "text": "string" or ["string1", "string2"],
            "return_probs": true/false (optional, default: true)
        }
    
    Response:
        {
            "success": true,
            "result": {
                "label": "positive|neutral|negative",
                "score": -1|0|1,
                "confidence": 0.95,
                "probabilities": {
                    "positive": 0.95,
                    "neutral": 0.03,
                    "negative": 0.02
                }
            }
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "text" field in request body'
            }), 400
        
        text = data['text']
        return_probs = data.get('return_probs', True)
        
        # Validate text
        if isinstance(text, str):
            if not text.strip():
                return jsonify({
                    'success': False,
                    'error': 'Text cannot be empty'
                }), 400
        elif isinstance(text, list):
            if not text or not all(isinstance(t, str) and t.strip() for t in text):
                return jsonify({
                    'success': False,
                    'error': 'Text list cannot be empty and must contain valid strings'
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': 'Text must be string or list of strings'
            }), 400
        
        # Predict sentiment
        result = sentiment_service.predict(text, return_probs=return_probs)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/batch', methods=['POST'])
def batch_analyze():
    """
    Batch sentiment analysis
    
    Request Body:
        {
            "texts": ["text1", "text2", ...],
            "return_probs": true/false (optional)
        }
    
    Response:
        {
            "success": true,
            "results": [{...}, {...}],
            "count": 2
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "texts" field in request body'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'success': False,
                'error': 'texts must be a non-empty list'
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                'success': False,
                'error': 'Maximum 100 texts per batch request'
            }), 400
        
        return_probs = data.get('return_probs', True)
        
        # Predict
        results = sentiment_service.predict(texts, return_probs=return_probs)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch_analyze: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get model evaluation metrics
    
    Response:
        {
            "success": true,
            "metrics": {
                "backtesting": {...},
                "sentiment_model": {...},
                "tft_model": {...}
            }
        }
    """
    try:
        ds = current_app.data_service
        metrics = ds.get_all_metrics()
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Error in get_metrics: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/examples', methods=['GET'])
def get_examples():
    """
    Get predefined example texts
    
    Response:
        {
            "success": true,
            "examples": [{
                "id": "news_positive",
                "category": "News",
                "text": "...",
                "source": "..."
            }, ...]
        }
    """
    try:
        examples = DataService.get_example_texts()
        
        return jsonify({
            'success': True,
            'examples': examples
        })
        
    except Exception as e:
        logger.error(f"Error in get_examples: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """
    Get loaded model information
    
    Response:
        {
            "success": true,
            "info": {
                "status": "loaded",
                "device": "cpu",
                "labels": ["negative", "neutral", "positive"],
                "num_parameters": 66955779,
                "model_type": "distilbert"
            }
        }
    """
    try:
        info = sentiment_service.get_model_info()
        
        return jsonify({
            'success': True,
            'info': info
        })
        
    except Exception as e:
        logger.error(f"Error in get_model_info: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/signals/generate', methods=['POST'])
def generate_trading_signals():
    """
    Generate trading signals using full pipeline
    Multi-source → Sentiment → Aggregation → TFT → Signals
    
    OPTIMIZED: Uses caching and optimized processing to prevent timeouts
    
    Request Body:
        {
            "ticker": "AAPL"
        }
    
    Response:
        {
            "success": true,
            "result": {
                "ticker": "AAPL",
                "sources": {
                    "news": {...},
                    "social": {...},
                    "earnings": {...}
                },
                "aggregated_sentiment": {...},
                "tft_prediction": {...},
                "trading_signal": {...}
            }
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "ticker" field in request body'
            }), 400
        
        ticker = data['ticker'].upper()
        
        # Basic ticker format validation (alphanumeric, 1-5 chars)
        if not ticker or not ticker.isalnum() or len(ticker) > 5:
            return jsonify({
                'success': False,
                'error': 'Invalid ticker format. Must be 1-5 alphanumeric characters.'
            }), 400
        
        # Check cache first (avoids recomputing same ticker within 5 minutes)
        cached_result = get_cached_result(ticker)
        if cached_result:
            logger.info(f"Returning cached result for {ticker}")
            return jsonify({
                'success': True,
                'result': cached_result,
                'cached': True
            })
        
        # Generate signals using full pipeline
        # This can take 30-60 seconds, but we've optimized it:
        # - Removed debug/info logging
        # - Optimized VSN tensor capture (only encoder VSN)
        # - Parallel API calls already implemented
        start_time = time.time()
        result = trading_signal_service.generate_signals(
            ticker=ticker,
            sentiment_service=sentiment_service
        )
        elapsed_time = time.time() - start_time
        
        # Cache the result
        set_cached_result(ticker, result)
        
        logger.info(f"Generated signals for {ticker} in {elapsed_time:.2f}s")
        
        return jsonify({
            'success': True,
            'result': result,
            'cached': False,
            'processing_time': round(elapsed_time, 2)
        })
        
    except TimeoutError as e:
        logger.error(f"Timeout generating signals for {ticker}: {e}")
        return jsonify({
            'success': False,
            'error': f'Request timed out. Please try again or use a different ticker.'
        }), 504
    except Exception as e:
        logger.error(f"Error in generate_trading_signals: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/signals/cache/clear', methods=['POST'])
def clear_signal_cache():
    """Clear the signal cache (admin/debug endpoint)"""
    try:
        with _cache_lock:
            count = len(_signal_cache)
            _signal_cache.clear()
        return jsonify({
            'success': True,
            'message': f'Cache cleared. Removed {count} entries.'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/signals/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        with _cache_lock:
            count = len(_signal_cache)
            tickers = list(_signal_cache.keys())
        return jsonify({
            'success': True,
            'cache_size': count,
            'cached_tickers': tickers,
            'ttl_seconds': CACHE_TTL
        })
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/interpretability/empirical', methods=['POST'])
def compute_empirical_validation():
    """
    Compute empirical feature importance across test period
    
    Request Body:
        {
            "test_tickers": ["AAPL", "MSFT", "GOOGL"],
            "test_dates": ["2024-09-01", "2024-09-02", ...]
        }
    
    Response:
        {
            "success": true,
            "result": {
                "aggregated_ranking": [...],
                "sentiment_rank": 1,
                "validation_passed": true,
                "num_predictions": 150,
                ...
            }
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Missing request body'
            }), 400
        
        test_tickers = data.get('test_tickers', [])
        test_dates = data.get('test_dates', [])
        
        if not test_tickers or not isinstance(test_tickers, list):
            return jsonify({
                'success': False,
                'error': 'Missing or invalid "test_tickers" field. Must be a list of ticker symbols.'
            }), 400
        
        if not test_dates or not isinstance(test_dates, list):
            return jsonify({
                'success': False,
                'error': 'Missing or invalid "test_dates" field. Must be a list of dates (YYYY-MM-DD format).'
            }), 400
        
        # Compute empirical feature importance
        result = trading_signal_service.compute_empirical_feature_importance(
            test_tickers=test_tickers,
            test_dates=test_dates,
            sentiment_service=sentiment_service
        )
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error in compute_empirical_validation: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500