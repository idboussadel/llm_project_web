"""
Flask Application Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Base directory
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR  # Changed: now using local directory instead of "model training"

# Load environment variables from local directory (where .env is located)
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)


class Config:
    """Base configuration"""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # API Keys
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
    FMP_API_KEY = os.getenv('FMP_API_KEY')
    
    # Model Paths (relative to web app directory)
    MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_production"
    QLORA_MODEL_PATH = PROJECT_ROOT / "models" / "qlora_5fold" / "fold_0_final"
    TFT_MODEL_PATH = PROJECT_ROOT / "models" / "tft_checkpoints" / "tft-epoch=37-val_loss=0.0237.ckpt"
    
    # Results Paths
    RESULTS_PATH = PROJECT_ROOT / "results"
    FINAL_RESULTS_JSON = PROJECT_ROOT / "final_results_summary.json"
    QLORA_RESULTS_JSON = RESULTS_PATH / "qlora_5fold_final_results.json"
    TEST_METRICS_JSON = PROJECT_ROOT / "models" / "test_metrics.json"
    ABLATION_STUDIES_JSON = RESULTS_PATH / "ablation_studies_summary.json"
    
    # Model Configuration
    USE_GPU = torch.cuda.is_available() if 'torch' in dir() else False
    DEVICE = "cuda" if USE_GPU else "cpu"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    
    # Caching
    ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'False').lower() == 'true'
    CACHE_TTL = 3600  # 1 hour
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Application Settings
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max request size


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    DEVELOPMENT = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    def __init__(self):
        super().__init__()
        # Validate SECRET_KEY in production
        if not os.getenv('SECRET_KEY'):
            raise ValueError("SECRET_KEY environment variable must be set in production")


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
