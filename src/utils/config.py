
"""Configuration manager for SentiTrade-HMA."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Centralized configuration manager."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.project_root = Path(__file__).parent.parent.parent
        
        # Load environment variables
        load_dotenv(self.project_root / ".env")
        
        # Load all configs
        self.data_config = self._load_config("data_config.yaml")
        self.model_config = self._load_config("model_config.yaml")
        self.training_config = self._load_config("training_config.yaml")
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load and parse YAML config file."""
        config_path = self.project_root / self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variable placeholders
        config = self._replace_env_vars(config)
        return config
    
    def _replace_env_vars(self, config: Any) -> Any:
        """Recursively replace ${ENV_VAR} with actual values."""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            value = os.getenv(env_var)
            if value is None:
                raise ValueError(f"Environment variable {env_var} not set")
            return value
        return config
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.data_config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.model_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.training_config
    
    @property
    def news_api_key(self):
        """Get News API key from environment."""
        return os.getenv("NEWS_API_KEY", "")

    
    @property
    def hf_token(self) -> str:
        """Get HuggingFace token."""
        return self.model_config['llm']['hf_token']
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / "data"
    
    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return self.project_root / "models"
    
    @property
    def results_dir(self) -> Path:
        """Get results directory path."""
        return self.project_root / "results"


# Global config instance (singleton)
_config_instance = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("✅ Configuration loaded successfully!")
    print(f"Project: {config.data_config['project']['name']}")
    print(f"Data dir: {config.data_dir}")
