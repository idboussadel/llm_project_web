"""
Sentiment Analysis Model Service
Singleton pattern for efficient model loading and inference
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

logger = logging.getLogger(__name__)


class SentimentModelService:
    """
    Singleton service for sentiment analysis using DistilBERT + LoRA
    Optimized for production deployment with caching and batch processing
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.model = None
            self.tokenizer = None
            self.device = None
            self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            self.label_map_inv = {'negative': 0, 'neutral': 1, 'positive': 2}
            self._initialized = True
    
    def load_model(self, model_path: Union[str, Path], device: str = 'cpu'):
        """
        Load DistilBERT model with LoRA adapter
        
        Args:
            model_path: Path to model directory (e.g., models/sentiment_production)
            device: 'cpu' or 'cuda'
        """
        if self.model is not None:
            logger.info("Model already loaded, skipping...")
            return
        
        try:
            model_path = Path(model_path)
            logger.info(f"Loading sentiment model from {model_path}")
            
            # Determine device
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = 'cpu'
            
            self.device = torch.device(device)
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            # Check if this is a LoRA model or full model
            if (model_path / 'adapter_config.json').exists():
                # Load LoRA model
                logger.info("Detected LoRA adapter, loading base model + adapter...")
                
                # Load adapter config to get base model name
                with open(model_path / 'adapter_config.json', 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path', 'distilbert-base-uncased')
                
                # Load base model
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_name,
                    num_labels=3,
                    torch_dtype=torch.float32
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, str(model_path))
                self.model = self.model.merge_and_unload()  # Merge for faster inference
            else:
                # Load full model directly
                logger.info("Loading full model...")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(model_path),
                    num_labels=3,
                    torch_dtype=torch.float32
                )
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_probs: bool = True,
        batch_size: int = 16
    ) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment for text input(s)
        
        Args:
            text: Single text string or list of texts
            return_probs: Whether to return probability distribution
            batch_size: Batch size for processing multiple texts
        
        Returns:
            Dictionary with 'label', 'score', 'confidence', and optionally 'probabilities'
            If input is list, returns list of dictionaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._predict_batch(batch, return_probs)
            results.extend(batch_results)
        
        return results[0] if is_single else results
    
    def _predict_batch(self, texts: List[str], return_probs: bool) -> List[Dict]:
        """Internal method for batch prediction"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Convert to results
        results = []
        for prob_dist in probs:
            prob_dict = {
                self.label_map[i]: prob_dist[i].item()
                for i in range(len(prob_dist))
            }
            
            # Get predicted label
            pred_idx = torch.argmax(prob_dist).item()
            label = self.label_map[pred_idx]
            confidence = prob_dist[pred_idx].item()
            
            # Calculate continuous sentiment score from probabilities
            # Use weighted average: negative=-1, neutral=0, positive=+1
            # This gives more nuanced scores than discrete -1/0/1
            neg_prob = prob_dict.get('negative', 0.0)
            neu_prob = prob_dict.get('neutral', 0.0)
            pos_prob = prob_dict.get('positive', 0.0)
            
            # Continuous score: weighted by probabilities
            score = (-1.0 * neg_prob) + (0.0 * neu_prob) + (1.0 * pos_prob)
            
            # Also keep discrete label for compatibility
            result = {
                'label': label,
                'score': float(score),  # Continuous score in range [-1, 1]
                'confidence': confidence
            }
            
            if return_probs:
                result['probabilities'] = prob_dict
            
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        if self.model is None:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'device': str(self.device),
            'labels': list(self.label_map.values()),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_type': self.model.config.model_type
        }


# Global instance
sentiment_service = SentimentModelService()
