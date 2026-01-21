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
        batch_size: Optional[int] = None  # Auto-detect based on environment
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
        
        # Auto-detect batch size if not provided
        if batch_size is None:
            import os
            is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None or os.getenv('RAILWAY_SERVICE_NAME') is not None
            # Use smaller batches on Railway to prevent OOM, larger locally
            batch_size = 32 if is_railway else 64
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._predict_batch(batch, return_probs)
            results.extend(batch_results)
            
            # Clear memory after large batches (especially important on Railway)
            if len(texts) > 50 and i % (batch_size * 2) == 0:
                import gc
                gc.collect()
        
        return results[0] if is_single else results
    
    def _predict_batch(self, texts: List[str], return_probs: bool) -> List[Dict]:
        """Internal method for batch prediction - optimized for speed"""
        if not texts:
            return []
        
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
        
        # Inference with torch.no_grad for better performance
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Vectorized operations for better performance
        # Get predicted labels and confidences in batch
        pred_indices = torch.argmax(probs, dim=-1)  # Shape: [batch_size]
        confidences = probs[torch.arange(len(texts)), pred_indices]  # Shape: [batch_size]
        
        # Calculate sentiment scores vectorized
        # probs shape: [batch_size, 3] where columns are [negative, neutral, positive]
        neg_probs = probs[:, 0]  # negative probabilities
        neu_probs = probs[:, 1]  # neutral probabilities  
        pos_probs = probs[:, 2]  # positive probabilities
        scores = (-1.0 * neg_probs) + (0.0 * neu_probs) + (1.0 * pos_probs)  # Shape: [batch_size]
        
        # Convert to CPU numpy for faster conversion
        pred_indices_np = pred_indices.cpu().numpy()
        confidences_np = confidences.cpu().numpy()
        scores_np = scores.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        # Clear GPU/CPU tensors immediately to free memory
        del inputs, outputs, logits, probs, pred_indices, confidences
        del neg_probs, neu_probs, pos_probs, scores
        
        # Build results list efficiently
        results = []
        for i in range(len(texts)):
            pred_idx = int(pred_indices_np[i])
            label = self.label_map[pred_idx]
            confidence = float(confidences_np[i])
            score = float(scores_np[i])
            
            result = {
                'label': label,
                'score': score,
                'confidence': confidence
            }
            
            if return_probs:
                result['probabilities'] = {
                    self.label_map[j]: float(probs_np[i, j])
                    for j in range(len(self.label_map))
                }
            
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
