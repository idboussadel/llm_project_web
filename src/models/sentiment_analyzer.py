"""
Sentiment Analyzer Module for SentiTrade-HMA-V2
Uses Llama-2-7B fine-tuned with QLoRA for financial sentiment analysis

Performance: 99% accuracy on test set (100 examples)
Author: Mohamed NAJID
Date: December 2025
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from pathlib import Path
import json
import logging
from typing import List, Union, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaSentimentAnalyzer:
    """
    Financial sentiment analyzer using Llama-2-7B fine-tuned with QLoRA.
    
    Features:
    - Manual LoRA fusion (no PEFT dependency)
    - 99% accuracy on financial texts
    - Supports batch processing
    - GPU accelerated
    
    Example:
        >>> analyzer = LlamaSentimentAnalyzer(adapter_path="models/sentiment_llm/llama2_qlora")
        >>> analyzer.predict("The company reported record earnings.")
        'positive'
    """
    
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-2-7b-hf",
        adapter_path: str = None,
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            base_model_name: Hugging Face model ID
            adapter_path: Path to LoRA adapters directory
            device: Device placement ("auto", "cuda", "cpu")
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
        """
        self.base_model_name = base_model_name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device
        self.load_in_8bit = load_in_8bit
        
        logger.info("="*70)
        logger.info("Initializing Llama-2 Sentiment Analyzer")
        logger.info("="*70)
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        
        # Load and fuse LoRA adapters if provided
        if self.adapter_path and self.adapter_path.exists():
            self._load_and_fuse_lora_adapters()
        
        logger.info("✅ Sentiment Analyzer ready")
        logger.info("="*70)
    
    def _load_model(self):
        """Load base Llama-2 model."""
        logger.info("Loading base model...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map=self.device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        self.model.eval()
        logger.info(f"✅ Base model loaded: {self.base_model_name}")
    
    def _load_tokenizer(self):
        """Load tokenizer."""
        logger.info("Loading tokenizer...")
        
        # Try loading from adapter path first, fallback to base model
        if self.adapter_path and (self.adapter_path / "tokenizer.model").exists():
            tokenizer_path = self.adapter_path
        else:
            tokenizer_path = self.base_model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        logger.info(f"✅ Tokenizer loaded from {tokenizer_path}")
    
    def _load_and_fuse_lora_adapters(self):
        """
        Load and manually fuse LoRA adapters into base model.
        
        Uses direct weight fusion: W = W_base + (alpha/r) * B @ A
        This avoids PEFT dependency and bitsandbytes issues.
        """
        logger.info("Loading LoRA adapters...")
        
        # Load config
        config_path = self.adapter_path / "adapter_config.json"
        with open(config_path, 'r') as f:
            adapter_config = json.load(f)
        
        lora_r = adapter_config.get('r', 16)
        lora_alpha = adapter_config.get('lora_alpha', 32)
        scaling = lora_alpha / lora_r
        
        logger.info(f"   Config: r={lora_r}, alpha={lora_alpha}, scaling={scaling:.2f}")
        
        # Load adapter weights
        adapter_file = self.adapter_path / "adapter_model.safetensors"
        adapter_weights = load_file(str(adapter_file))
        
        logger.info(f"   Loaded {len(adapter_weights)} tensors")
        
        # Group LoRA matrices (A and B pairs)
        lora_pairs = {}
        for key in adapter_weights.keys():
            if '.lora_A.weight' in key:
                base_key = key.replace('base_model.model.', '').replace('.lora_A.weight', '')
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]['A'] = adapter_weights[key]
            elif '.lora_B.weight' in key:
                base_key = key.replace('base_model.model.', '').replace('.lora_B.weight', '')
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]['B'] = adapter_weights[key]
        
        logger.info(f"   Found {len(lora_pairs)} LoRA pairs")
        
        # Apply fusion: W = W_base + scaling * B @ A
        applied = 0
        skipped = 0
        
        for module_path, matrices in lora_pairs.items():
            if 'A' in matrices and 'B' in matrices:
                try:
                    # Navigate to module in model
                    parts = module_path.split('.')
                    current = self.model
                    
                    for part in parts:
                        current = getattr(current, part)
                    
                    if hasattr(current, 'weight'):
                        A = matrices['A']  # (r, in_features)
                        B = matrices['B']  # (out_features, r)
                        
                        # Calculate delta: scaling * B @ A
                        delta_W = scaling * (B.float() @ A.float())
                        delta_W = delta_W.to(current.weight.dtype).to(current.weight.device)
                        
                        # Verify dimensions
                        if current.weight.shape == delta_W.shape:
                            # Apply fusion
                            with torch.no_grad():
                                current.weight.add_(delta_W)
                            applied += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1
                except (AttributeError, RuntimeError):
                    skipped += 1
        
        logger.info(f"   ✅ {applied} modules fused")
        if skipped > 0:
            logger.info(f"   ⚠️ {skipped} modules skipped")
    
    def format_prompt(self, text: str) -> str:
        """
        Format text into Llama-2 instruction prompt.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Formatted prompt string
        """
        return f"""<s>[INST] You are a financial sentiment analyzer. Analyze the sentiment of the following financial text and respond with ONLY one word: positive, negative, or neutral.

Financial Text: {text}

Sentiment: [/INST]"""
    
    def predict(self, text: str, max_new_tokens: int = 10) -> str:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Financial text to analyze
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        prompt = self.format_prompt(text)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode full response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after [/INST])
        if "[/INST]" in response:
            generated = response.split("[/INST]")[-1].strip()
        else:
            generated = response.strip()
        
        # Debug logging
        if not generated:
            logger.warning(f"Empty response for text: {text[:50]}...")
            return "neutral"
        
        # Extract first line (before any newline or "Reasoning:")
        first_line = generated.split('\n')[0].strip()
        
        # Clean and normalize
        sentiment = first_line.lower().strip('.,;:!? ')
        
        # Validate
        if sentiment in ['positive', 'negative', 'neutral']:
            return sentiment
        
        # If first word is valid sentiment
        words = first_line.split()
        if words:
            first_word = words[0].lower().strip('.,;:!?')
            if first_word in ['positive', 'negative', 'neutral']:
                return first_word
        
        # Search in full response
        for word in generated.lower().split():
            clean = word.strip('.,;:!?\n')
            if clean in ['positive', 'negative', 'neutral']:
                return clean
        
        logger.warning(f"Could not extract sentiment from: {generated[:100]}")
        return "neutral"  # Default fallback

    
    def predict_batch(
        self, 
        texts: List[str], 
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[str]:
        """
        Predict sentiments for multiple texts (batch processing).
        
        Args:
            texts: List of financial texts
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            List of sentiment labels
        """
        predictions = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Analyzing sentiments")
        else:
            iterator = texts
        
        for text in iterator:
            sentiment = self.predict(text)
            predictions.append(sentiment)
        
        return predictions
    
    def get_sentiment_score(self, sentiment: str) -> float:
        """
        Convert sentiment label to numeric score.
        
        Args:
            sentiment: Sentiment label ('positive', 'negative', 'neutral')
            
        Returns:
            Numeric score: -1.0 (negative), 0.0 (neutral), 1.0 (positive)
        """
        mapping = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0,
        }
        return mapping.get(sentiment, 0.0)
    
    def analyze_with_scores(
        self, 
        texts: Union[str, List[str]]
    ) -> Union[Dict, List[Dict]]:
        """
        Analyze text(s) and return both label and numeric score.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Dictionary or list of dictionaries with 'sentiment' and 'score'
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        results = []
        for text in texts:
            sentiment = self.predict(text)
            score = self.get_sentiment_score(sentiment)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'score': score,
            })
        
        return results[0] if single_input else results


# Convenience function for quick analysis
def analyze_sentiment(
    text: Union[str, List[str]],
    adapter_path: str = "models/sentiment_llm/llama2_qlora"
) -> Union[str, List[str]]:
    """
    Quick sentiment analysis function.
    
    Args:
        text: Text or list of texts to analyze
        adapter_path: Path to LoRA adapters
        
    Returns:
        Sentiment label(s)
        
    Example:
        >>> from src.models.sentiment_analyzer import analyze_sentiment
        >>> analyze_sentiment("The company reported strong earnings.")
        'positive'
    """
    analyzer = LlamaSentimentAnalyzer(adapter_path=adapter_path)
    
    if isinstance(text, str):
        return analyzer.predict(text)
    else:
        return analyzer.predict_batch(text)


# Example usage
if __name__ == "__main__":
    # Test analyzer
    analyzer = LlamaSentimentAnalyzer(
        adapter_path="models/sentiment_llm/llama2_qlora_sentiment_final"
    )
    
    # Test examples
    test_texts = [
        "The company reported record earnings exceeding expectations.",
        "Operating losses widened significantly during the quarter.",
        "The firm has 25 offices across Europe.",
    ]
    
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS TESTS")
    print("="*70 + "\n")
    
    for text in test_texts:
        sentiment = analyzer.predict(text)
        score = analyzer.get_sentiment_score(sentiment)
        print(f"Text      : {text}")
        print(f"Sentiment : {sentiment} (score: {score:+.1f})")
        print()

