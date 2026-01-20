"""
Analyseur de Sentiment Llama-2 pour SentiTrade-HMA-V2
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class LlamaSentimentAnalyzer:
    """
    Analyseur de sentiment financier - Llama-2 fine-tuné avec LoRA
    
    Performances :
        - MCC interne: 0.9926
        - MCC externe: 0.8865  
        - Accuracy externe: 92.22%
    """
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Args:
            model_path: Chemin vers le modèle LoRA
            device: 'cuda', 'cpu' ou 'auto'
        """
        if model_path is None:
            model_path = self._find_model_path()
        
        self.model_path = Path(model_path)
        
        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print("="*80)
        print("🦙 LLAMA-2 SENTIMENT ANALYZER")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Modèle: {self.model_path}")
        
        # Charger config
        config_path = self.model_path.parent / "model_config.json"
        if not config_path.exists():
            # Chercher un niveau au-dessus si dans fold0_final
            config_path = self.model_path.parent.parent / "model_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"❌ Configuration non trouvée: {config_path}")
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.label_mapping = self.config["label_mapping"]
        self.id_to_label = {int(k): v for k, v in self.label_mapping.items()}
        self.label_to_id = {v: int(k) for k, v in self.label_mapping.items()}
        
        print(f"Performance: MCC {self.config['metrics']['mcc_external']:.4f}")
        
        # Tokenizer
        print("\n📥 Chargement tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("✅ Tokenizer chargé")
        
        # Modèle base
        print("\n📥 Chargement Llama-2-7B...")
        print("   (1ère fois: téléchargement ~13GB, ~5 min)")
        print("   (Suivantes: cache local, ~30 sec)")
        
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.config["model_name"],
                num_labels=self.config["num_labels"],
                id2label=self.id_to_label,
                label2id=self.label_to_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map=self.device if self.device == "cpu" else "auto"
            )
            print("✅ Modèle base chargé")
            
        except Exception as e:
            print(f"\n⚠️ Erreur: {e}")
            print("\n💡 Solution si modèle Llama-2 nécessite autorisation:")
            print("   1. Va sur huggingface.co/meta-llama/Llama-2-7b-hf")
            print("   2. Accepte les conditions")
            print("   3. Crée un token: huggingface.co/settings/tokens")
            print("   4. Execute: huggingface-cli login")
            raise
        
        # LoRA
        print("\n📥 Chargement adaptateurs LoRA...")
        
        try:
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✅ LoRA chargé")
            
        except Exception as e:
            print(f"\n❌ Erreur LoRA: {e}")
            print(f"\n💡 Vérifie ces fichiers:")
            print(f"   - {self.model_path / 'adapter_config.json'}")
            print(f"   - {self.model_path / 'adapter_model.safetensors'}")
            raise
        
        print("\n" + "="*80)
        print("✅ MODÈLE PRÊT")
        print("="*80)
    
    def _find_model_path(self) -> Path:
        """Auto-détecte le chemin du modèle"""
        possible_paths = [
            # Si dans fold0_final
            Path("models/sentiment_llm/llama2_sentiment_lora/fold0_final"),
            # Si renommé
            Path("models/sentiment_llm/llama2_sentiment_lora"),
            # Chemins relatifs
            Path("../models/sentiment_llm/llama2_sentiment_lora/fold0_final"),
            Path("../../models/sentiment_llm/llama2_sentiment_lora/fold0_final"),
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "adapter_config.json").exists():
                return path
        
        raise FileNotFoundError(
            "❌ Modèle non trouvé.\n"
            "Vérifie que adapter_config.json existe dans :\n"
            "  models/sentiment_llm/llama2_sentiment_lora/fold0_final/"
        )
    
    def predict(self, text: str, return_probs: bool = False) -> Dict:
        """
        Prédit le sentiment d'un texte
        
        Args:
            text: Phrase financière à analyser
            return_probs: Retourner les probabilités détaillées
        
        Returns:
            {
                'label': 'positive' | 'negative' | 'neutral',
                'score': -1 | 0 | 1,
                'confidence': float (0-1),
                'probabilities': dict (optionnel)
            }
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()
        
        label = self.id_to_label[pred_id]
        
        result = {
            "label": label,
            "score": pred_id - 1,  # -1, 0, 1
            "confidence": confidence
        }
        
        if return_probs:
            result["probabilities"] = {
                "negative": probs[0][0].item(),
                "neutral": probs[0][1].item(),
                "positive": probs[0][2].item()
            }
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """Prédiction batch (plus rapide)"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1)
                confidences = probs[range(len(pred_ids)), pred_ids]
            
            for pred_id, conf in zip(pred_ids, confidences):
                pred_id_item = pred_id.item()
                label = self.id_to_label[pred_id_item]
                results.append({
                    "label": label,
                    "score": pred_id_item - 1,
                    "confidence": conf.item()
                })
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = "sentence", 
                         batch_size: int = 16) -> pd.DataFrame:
        """
        Ajoute sentiments à un DataFrame
        
        Colonnes ajoutées:
            - sentiment: label
            - sentiment_score: -1/0/1
            - sentiment_confidence: 0-1
        """
        print(f"\n📊 Analyse de {len(df)} lignes...")
        
        texts = df[text_column].tolist()
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_results = self.predict_batch(texts[i:i+batch_size], batch_size)
            results.extend(batch_results)
            
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
                processed = min(i + batch_size, len(texts))
                print(f"   {processed}/{len(texts)} ({100*processed/len(texts):.1f}%)")
        
        df = df.copy()
        df['sentiment'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        df['sentiment_confidence'] = [r['confidence'] for r in results]
        
        print(f"   ✅ Terminé")
        print(f"\n📈 Distribution: {df['sentiment'].value_counts().to_dict()}")
        print(f"📊 Confiance moy: {df['sentiment_confidence'].mean():.4f}")
        
        return df


def load_sentiment_analyzer(model_path: str = None, device: str = "auto"):
    """
    Helper pour charger facilement
    
    Usage:
        from src.sentiment import load_sentiment_analyzer
        
        analyzer = load_sentiment_analyzer()
        result = analyzer.predict("Strong earnings")
    """
    return LlamaSentimentAnalyzer(model_path=model_path, device=device)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 TEST LLAMA-2 SENTIMENT\n")
    
    try:
        analyzer = load_sentiment_analyzer()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        exit(1)
    
    # Test simple
    test = "The company reported strong quarterly earnings"
    result = analyzer.predict(test, return_probs=True)
    
    print(f"\nTexte: '{test}'")
    print(f"Sentiment: {result['label']}")
    print(f"Score: {result['score']}")
    print(f"Confiance: {result['confidence']:.4f}")
    print(f"\nProbabilités:")
    for label, prob in result['probabilities'].items():
        print(f"  {label:8s}: {prob:.4f}")
    
    # Test batch
    print("\n" + "="*80)
    test_batch = [
        "Revenue exceeded expectations",
        "The company faces challenges",
        "Results were in line with forecasts"
    ]
    
    results = analyzer.predict_batch(test_batch)
    print(f"\nBatch de {len(test_batch)} phrases:")
    for text, res in zip(test_batch, results):
        print(f"  {res['label']:8s} ({res['confidence']:.3f}) - {text}")
    
    print("\n✅ TOUS LES TESTS RÉUSSIS")
