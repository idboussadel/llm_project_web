"""
Flask Application Factory
"""
import logging
from pathlib import Path
from flask import Flask
from flask_cors import CORS
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name=None):
    """
    Application factory pattern
    
    Args:
        config_name: Configuration to use ('development', 'production', 'testing')
    
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    config_class = get_config()
    app.config.from_object(config_class)
    
    # Enable CORS
    CORS(app)
    
    # Initialize services
    from app.services.model_service import sentiment_service
    from app.services.data_service import DataService
    
    # Initialize data service
    app.data_service = DataService(
        results_path=app.config['RESULTS_PATH'],
        final_results_json=app.config['FINAL_RESULTS_JSON'],
        qlora_results_json=app.config['QLORA_RESULTS_JSON'],
        test_metrics_json=app.config['TEST_METRICS_JSON'],
        ablation_studies_json=app.config.get('ABLATION_STUDIES_JSON')
    )
    
    # Load sentiment model
    logger.info("Initializing sentiment model...")
    try:
        # Use fine-tuned QLora Llama-2 model
        model_path = app.config['QLORA_MODEL_PATH']
        logger.info(f"Loading fine-tuned QLora model from {model_path}")
        
        sentiment_service.load_model(
            model_path=model_path,
            device=app.config['DEVICE']
        )
        logger.info("Fine-tuned QLora sentiment model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {e}")
        logger.warning("App will start but sentiment analysis will not work")
    
    # Initialize trading signal service with API keys
    logger.info("Initializing trading signal service...")
    from app.services.trading_service import trading_signal_service
    try:
        trading_signal_service.initialize(
            tft_model_path=app.config.get('TFT_MODEL_PATH'),
            news_api_key=app.config.get('NEWS_API_KEY'),
            finnhub_api_key=app.config.get('FINNHUB_API_KEY'),
            fmp_api_key=app.config.get('FMP_API_KEY')
        )
        logger.info("Trading signal service initialized")
    except Exception as e:
        logger.error(f"Trading signal service initialization failed: {e}")
        raise
    
    # Register blueprints
    from app.api import api_bp
    from app.views import views_bp
    
    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return {'error': 'Internal server error'}, 500
    
    logger.info(f"Application initialized in {app.config['DEBUG'] and 'DEBUG' or 'PRODUCTION'} mode")
    
    return app
