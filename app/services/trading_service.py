"""
TFT Model Service - Trading Signal Generation
Implements full pipeline: Multi-source → Sentiment → TFT → Signals
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

import torch
from torch.serialization import add_safe_globals

logger = logging.getLogger(__name__)

# Required imports for TFT predictions
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
import pickle
HAS_PYTORCH_LIGHTNING = True

# Add src to path for imports (now using local src directory)
project_root = Path(__file__).parent.parent.parent  # web app directory
sys.path.insert(0, str(project_root))

from src.data.collectors import NewsAPICollector, FinancialDataCollector
from src.data.preprocessors import SentimentAggregator, TechnicalIndicatorCalculator
HAS_DATA_COLLECTORS = True

logger = logging.getLogger(__name__)

# Import yfinance for historical data
import yfinance as yf
from datetime import datetime, timedelta


class TradingSignalService:
    """
    Full pipeline for trading signal generation
    Multi-source data → Sentiment → Aggregation → TFT → Signals
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.tft_model = None
            self.tft_config = None
            self.scalers = None
            self.news_collector = None
            self.financial_collector = None
            self.device = None
            self._initialized = True
    
    def initialize(self, tft_model_path: Path, news_api_key: Optional[str] = None,
                   finnhub_api_key: Optional[str] = None, fmp_api_key: Optional[str] = None):
        """Initialize all components of the pipeline"""
        try:
            # Store API keys
            self.news_api_key = news_api_key
            self.finnhub_api_key = finnhub_api_key
            self.fmp_api_key = fmp_api_key
            
            # Initialize data collectors (required)
            logger.info("Initializing data collectors...")
            self.financial_collector = FinancialDataCollector()
            
            if not news_api_key:
                raise ValueError("NewsAPI key is required")
            
            self.news_collector = NewsAPICollector()
            logger.info("NewsAPI collector initialized")
            
            # Load TFT model (required)
            if not tft_model_path or not tft_model_path.exists():
                raise FileNotFoundError(f"TFT model not found at {tft_model_path}")
            
            logger.info(f"Loading TFT model from {tft_model_path}")
            checkpoint_path = self._prepare_tft_checkpoint(tft_model_path)
            self.tft_model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
            self.tft_model.eval()
            logger.info("✅ TFT model loaded successfully")
            
            # Load TFT configuration
            tft_data_dir = tft_model_path.parent.parent.parent / "data" / "tft"
            config_path = tft_data_dir / "tft_config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"TFT config not found at {config_path}")
            
            import json
            with open(config_path, 'r') as f:
                self.tft_config = json.load(f)
            logger.info(f"✅ TFT config loaded: {len(self.tft_config.get('tickers', []))} tickers")
            
            # Load scalers for denormalization
            scalers_path = tft_data_dir / "scalers.pkl"
            if scalers_path.exists():
                self.scalers = self._load_scalers(scalers_path)
                logger.info("✅ Scalers loaded for price denormalization")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading signal service: {e}")
            raise

    def _prepare_tft_checkpoint(self, checkpoint_path: Path) -> Path:
        """Ensure TFT checkpoint is compatible with current library versions."""
        try:
            # Allow legacy QuantileLoss objects stored in the checkpoint
            try:
                from pytorch_forecasting.metrics.quantile import QuantileLoss
                add_safe_globals([QuantileLoss])
            except ImportError:
                logger.warning("QuantileLoss import failed; attempting checkpoint load anyway")

            checkpoint = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read TFT checkpoint {checkpoint_path}: {e}")

        if not isinstance(checkpoint, dict):
            return checkpoint_path

        if self._remove_invalid_checkpoint_keys(checkpoint):
            patched_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_patched{checkpoint_path.suffix}"
            )
            torch.save(checkpoint, str(patched_path))
            logger.info(f"Patched TFT checkpoint written to {patched_path}")
            return patched_path

        # Prefer previously patched checkpoint if it exists
        patched_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_patched{checkpoint_path.suffix}"
        )
        if patched_path.exists():
            logger.info(f"Using previously patched TFT checkpoint at {patched_path}")
            return patched_path

        return checkpoint_path

    @staticmethod
    def _remove_invalid_checkpoint_keys(obj) -> bool:
        """Recursively drop legacy keys not supported by current libraries."""
        removed = False
        if isinstance(obj, dict):
            if "monotone_constaints" in obj:
                obj.pop("monotone_constaints", None)
                removed = True
            for value in obj.values():
                if TradingSignalService._remove_invalid_checkpoint_keys(value):
                    removed = True
        elif isinstance(obj, list):
            for item in obj:
                if TradingSignalService._remove_invalid_checkpoint_keys(item):
                    removed = True
        return removed

    @staticmethod
    def _load_scalers(scalers_path: Path):
        """Load pickled scalers handling legacy numpy module paths."""
        try:
            import numpy.core as numpy_core
            import sys
            sys.modules.setdefault("numpy._core", numpy_core)
            sys.modules.setdefault("numpy._core.multiarray", numpy_core.multiarray)
            sys.modules.setdefault("numpy._core.numerictypes", numpy_core.numerictypes)
        except Exception:
            logger.warning("Could not alias numpy._core; attempting to load scalers anyway")
        with open(scalers_path, 'rb') as f:
            return pickle.load(f)

    def generate_signals(
        self,
        ticker: str,
        sentiment_service
    ) -> Dict:
        """
        Complete pipeline for signal generation
        
        Args:
            ticker: Stock ticker symbol
            sentiment_service: Sentiment model service instance
        
        Returns:
            Dictionary with multi-source sentiments, aggregated score, and trading signal
        """
        try:
            # Step 1: Collect multi-source data
            logger.info(f"Collecting data for {ticker}...")
            sources_data = self._collect_multi_source_data(ticker)
            
            # Step 2: Analyze sentiment for each source
            logger.info("Analyzing sentiment across sources...")
            sentiment_results = self._analyze_multi_source_sentiment(
                sources_data,
                sentiment_service
            )
            
            # Step 3: Hierarchical aggregation
            logger.info("Performing hierarchical aggregation...")
            aggregated_sentiment = self._hierarchical_aggregation(sentiment_results)
            
            # Step 4: Generate TFT prediction
            logger.info("Generating TFT prediction...")
            tft_prediction = self._generate_tft_prediction(
                ticker,
                aggregated_sentiment
            )
            
            # Step 5: Generate trading signal
            logger.info("Generating trading signal...")
            trading_signal = self._generate_trading_signal(
                aggregated_sentiment,
                tft_prediction
            )
            
            return {
                'ticker': ticker,
                'sources': sentiment_results,
                'aggregated_sentiment': aggregated_sentiment,
                'tft_prediction': tft_prediction,
                'trading_signal': trading_signal
            }
            
        except Exception as e:
            logger.error(f"Error generating signals for {ticker}: {e}")
            raise
    
    def _fetch_newsapi(self, ticker: str) -> List[str]:
        """Fetch news from NewsAPI - helper for parallel execution"""
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': f'{ticker} stock OR {ticker} earnings',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                if articles:
                    result = [a.get('title', '') + '. ' + (a.get('description', '') or '') 
                             for a in articles if a.get('title')]
                    logger.info(f"Collected {len(result)} real news articles from NewsAPI")
                    return result
                else:
                    logger.warning("No articles found from NewsAPI")
                    return []
            else:
                logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Failed to fetch from NewsAPI: {e}")
            return []
    
    def _fetch_finnhub(self, ticker: str) -> List[str]:
        """Fetch social data from Finnhub - helper for parallel execution"""
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            url = f'https://finnhub.io/api/v1/company-news'
            params = {
                'symbol': ticker,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    result = [item.get('headline', '') + '. ' + (item.get('summary', '') or '') 
                             for item in data if item.get('headline')]
                    logger.info(f"Collected {len(result)} items from Finnhub")
                    return result
                else:
                    logger.warning(f"No Finnhub data for {ticker}")
                    return []
            else:
                logger.error(f"Finnhub API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Failed to fetch from Finnhub: {e}")
            return []
    
    def _fetch_fmp(self, ticker: str) -> List[str]:
        """Fetch earnings data from FMP - helper for parallel execution"""
        try:
            base_url = 'https://financialmodelingprep.com/stable'
            endpoints = [
                f'{base_url}/income-statement?symbol={ticker}&period=quarter&limit=1',
                f'{base_url}/profile?symbol={ticker}',
            ]
            
            for endpoint in endpoints:
                try:
                    url = f'{endpoint}&apikey={self.fmp_api_key}'
                    response = requests.get(url, timeout=10)
                    
                    logger.info(f"FMP trying: {endpoint}")
                    logger.info(f"FMP Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'income-statement' in endpoint and data and len(data) > 0:
                            stmt = data[0]
                            date = stmt.get('date', 'Recent quarter')
                            revenue = stmt.get('revenue', 0)
                            net_income = stmt.get('netIncome', 0)
                            eps = stmt.get('eps', 0)
                            period = stmt.get('period', 'Q')
                            
                            result = [
                                f"{ticker} {period} earnings for {date}: Revenue of ${revenue:,.0f}",
                                f"Net income was ${net_income:,.0f} for the quarter",
                                f"Diluted EPS came in at ${eps:.2f} per share"
                            ]
                            logger.info(f"Collected {ticker} earnings data from {date} income statement")
                            return result
                        
                        elif 'profile' in endpoint and data and len(data) > 0:
                            profile = data[0]
                            description = profile.get('description', '')
                            if description:
                                sentences = description.split('. ')
                                result = [s + '.' for s in sentences if len(s) > 20]
                                logger.info(f"Using company profile as earnings context")
                                return result
                    
                    elif response.status_code == 403:
                        logger.warning(f"FMP 403 for {endpoint}: {response.text[:200]}")
                        continue
                    else:
                        logger.warning(f"FMP error {response.status_code} for {endpoint}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed endpoint {endpoint}: {e}")
                    continue
            
            logger.error("FMP earnings endpoints failed - check API key and subscription plan")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch earnings from FMP: {e}")
            return []
    
    def _collect_multi_source_data(self, ticker: str) -> Dict:
        """Collect data from multiple sources in parallel - OPTIMIZED"""
        sources = {}
        
        # Prepare tasks for parallel execution
        tasks = {}
        
        if self.news_api_key:
            tasks['news'] = (self._fetch_newsapi, ticker)
        else:
            logger.error("No NewsAPI key configured - cannot fetch news")
            sources['news'] = []
        
        if self.finnhub_api_key:
            tasks['social'] = (self._fetch_finnhub, ticker)
        else:
            logger.error("No Finnhub API key configured - cannot fetch social data")
            sources['social'] = []
        
        if self.fmp_api_key:
            tasks['earnings'] = (self._fetch_fmp, ticker)
        else:
            logger.error("No FMP API key configured - cannot fetch earnings transcripts")
            sources['earnings'] = []
        
        # Execute all API calls in parallel using ThreadPoolExecutor
        logger.info(f"Fetching data from {len(tasks)} sources in parallel for {ticker}...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(func, *args): key for key, (func, *args) in tasks.items()}
            
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result(timeout=15)  # 15 second timeout per API call
                    sources[key] = result
                except Exception as e:
                    logger.error(f"Error fetching {key} data: {e}")
                    sources[key] = []
        
        return sources
    
    def _analyze_single_source(self, source_name: str, texts: List[str], sentiment_service) -> Dict:
        """Analyze sentiment for a single source - helper for parallel execution"""
        if not texts:
            return None
        
        try:
            # Batch analyze
            sentiments = sentiment_service.predict(texts, return_probs=True)
            
            # Calculate source-level aggregates
            avg_score = np.mean([s['score'] for s in sentiments])
            avg_confidence = np.mean([s['confidence'] for s in sentiments])
            
            # Determine dominant sentiment
            if avg_score > 0.3:
                dominant = 'positive'
            elif avg_score < -0.3:
                dominant = 'negative'
            else:
                dominant = 'neutral'
            
            return {
                'source_name': source_name,
                'texts': texts,
                'individual_sentiments': sentiments,
                'average_score': float(avg_score),
                'average_confidence': float(avg_confidence),
                'dominant_sentiment': dominant,
                'sentiment': dominant,
                'score': float(avg_score),
                'confidence': float(avg_confidence),
                'count': len(texts)
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {source_name}: {e}")
            return None
    
    def _analyze_multi_source_sentiment(
        self,
        sources_data: Dict,
        sentiment_service
    ) -> Dict:
        """Analyze sentiment for each source in parallel - OPTIMIZED"""
        results = {}
        
        # Prepare tasks for parallel execution
        tasks = [(source_name, texts) for source_name, texts in sources_data.items() if texts]
        
        if not tasks:
            return results
        
        # Execute sentiment analysis in parallel
        logger.info(f"Analyzing sentiment for {len(tasks)} sources in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(tasks), 3)) as executor:
            futures = {
                executor.submit(self._analyze_single_source, source_name, texts, sentiment_service): source_name
                for source_name, texts in tasks
            }
            
            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per source
                    if result:
                        results[result['source_name']] = {
                            'texts': result['texts'],
                            'individual_sentiments': result['individual_sentiments'],
                            'average_score': result['average_score'],
                            'average_confidence': result['average_confidence'],
                            'dominant_sentiment': result['dominant_sentiment'],
                            'sentiment': result['sentiment'],
                            'score': result['score'],
                            'confidence': result['confidence'],
                            'count': result['count']
                        }
                except Exception as e:
                    logger.error(f"Error processing sentiment for {source_name}: {e}")
                    # Continue with other sources even if one fails
        
        return results
    
    def _hierarchical_aggregation(self, sentiment_results: Dict) -> Dict:
        """
        Hierarchical sentiment aggregation with source weighting
        News: 40%, Social: 30%, Earnings: 30%
        """
        weights = {
            'news': 0.4,
            'social': 0.3,
            'earnings': 0.3
        }
        
        weighted_score = 0
        weighted_confidence = 0
        total_weight = 0
        
        for source, data in sentiment_results.items():
            weight = weights.get(source, 0.33)
            weighted_score += data['average_score'] * weight
            weighted_confidence += data['average_confidence'] * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0
            final_confidence = 0
        
        # Determine final sentiment
        if final_score > 0.3:
            final_sentiment = 'positive'
        elif final_score < -0.3:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        return {
            'score': float(final_score),
            'confidence': float(final_confidence),
            'sentiment': final_sentiment,
            'source_weights': weights
        }
    
    def _generate_tft_prediction(self, ticker: str, aggregated_sentiment: Dict) -> Dict:
        """
        Generate TFT price prediction using the trained model with REAL data
        Returns a fallback prediction if ticker is not in trained set
        """
        if self.tft_model is None or self.tft_config is None:
            raise RuntimeError("TFT model not loaded")
        
        # Check if ticker is in trained tickers
        trained_tickers = self.tft_config.get('tickers', [])
        if ticker not in trained_tickers:
            logger.warning(f"{ticker} not in trained tickers {trained_tickers}. Using sentiment-only prediction.")
            # Return a sentiment-based prediction without TFT model
            return self._generate_sentiment_only_prediction(ticker, aggregated_sentiment)
        
        try:
            # Step 1: Fetch historical data from yfinance (FREE - no API key needed)
            logger.info(f"Fetching {ticker} price history from yfinance...")
            max_encoder = self.tft_config.get('max_encoder_length', 60)
            max_decoder = self.tft_config.get('max_prediction_length', 5)
            max_lag = 5  # Highest lag we engineer
            
            # Optimize: Use period='2y' for faster download (yfinance optimizes this better)
            # This provides ~500 trading days, which is more than enough for indicators
            df = yf.download(ticker, period='2y', progress=False, show_errors=False)
            
            if df.empty or len(df) < max_encoder:
                raise ValueError(f"Insufficient data for {ticker}: got {len(df)} rows, need {max_encoder}")
            
            # Reset index and prepare DataFrame
            df = df.reset_index()
            
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            df['Ticker'] = ticker
            
            # Step 2: Calculate technical indicators using existing preprocessor
            logger.info(f"Calculating technical indicators for {ticker}...")
            indicator_calc = TechnicalIndicatorCalculator()
            df = indicator_calc.add_all_indicators(df)
            
            # Step 3: Add required features from tft_config
            logger.info(f"Engineering features for {ticker}...")
            df = self._engineer_tft_features(df, ticker, aggregated_sentiment)
            
            # Step 4: Prepare for TFT prediction
            logger.info(f"Preparing TFT dataset for {ticker}...")
            prediction_df = self._prepare_tft_dataset(df, ticker)
            
            # Step 5: Create DataLoader for prediction
            logger.info(f"Creating prediction DataLoader for {ticker}...")
            
            logger.info(
                "Prediction frame rows=%s, time_idx_range=(%s, %s)",
                len(prediction_df),
                prediction_df['time_idx'].min(),
                prediction_df['time_idx'].max()
            )
            logger.info(
                "TFT feature stats: Close[%s,%s] Returns[%s,%s] sentiment_score[%s,%s]",
                prediction_df['Close'].min(),
                prediction_df['Close'].max(),
                prediction_df['Returns'].min(),
                prediction_df['Returns'].max(),
                prediction_df['sentiment_score'].min(),
                prediction_df['sentiment_score'].max()
            )
            
            model_hparams = getattr(self.tft_model, "hparams", {})
            static_reals = list(model_hparams.get("static_reals", []))

            dataset_kwargs = {
                "time_idx": "time_idx",
                "target": "Close",
                "group_ids": ["Ticker"],
                "min_encoder_length": 1,
                "max_encoder_length": self.tft_config['max_encoder_length'],
                "min_prediction_length": 1,
                "max_prediction_length": self.tft_config['max_prediction_length'],
                "time_varying_known_reals": self.tft_config['time_varying_known'],
                "time_varying_unknown_reals": self.tft_config['time_varying_unknown'],
                "static_categoricals": self.tft_config['static_features'],
                "static_reals": static_reals,
                "allow_missing_timesteps": True,
                "add_relative_time_idx": True,
                "add_target_scales": True,
                "add_encoder_length": True,
                "predict_mode": True
            }

            model_target_normalizer = getattr(self.tft_model, "target_normalizer", None)
            if model_target_normalizer is not None:
                dataset_kwargs["target_normalizer"] = model_target_normalizer
            else:
                fallback_normalizer = GroupNormalizer(groups=["Ticker"], transformation="softplus")
                dataset_kwargs["target_normalizer"] = fallback_normalizer
                logger.warning("TFT checkpoint missing target_normalizer; using GroupNormalizer fallback")

            # Create TimeSeriesDataSet for prediction using tft_config
            predict_dataset = TimeSeriesDataSet(
                prediction_df,
                **dataset_kwargs
            )
            
            predict_dataloader = predict_dataset.to_dataloader(
                train=False,
                batch_size=1,
                num_workers=0
            )
            
            # Step 6: Run TFT model prediction
            logger.info(f"Running TFT model prediction for {ticker}...")
            raw_prediction = self.tft_model.predict(
                predict_dataloader,
                mode="raw",
                return_index=True,
                return_decoder_lengths=True
            )
            
            # Step 7: Extract and denormalize predictions
            predictions = raw_prediction.output.prediction.cpu().numpy()
            quantiles = getattr(raw_prediction, "quantiles", None)
            
            # Check for invalid predictions (inf/nan)
            if not np.isfinite(predictions).all():
                logger.warning(f"TFT predictions contain non-finite values. Shape: {predictions.shape}")
                # Replace inf/nan with 0 (normalized space)
                predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get current price before any processing
            current_price = float(df['Close'].iloc[-1])
            
            # Check if raw predictions are clearly wrong (too large - likely denormalization issue)
            raw_max = np.abs(predictions).max()
            if raw_max > 1e6:  # If raw predictions are > 1 million, model is producing wrong values
                logger.error(
                    f"TFT raw predictions for {ticker} are clearly wrong: max={raw_max:.2f} "
                    f"(current_price={current_price:.2f}). Model may be incompatible or data mismatch. "
                    f"Falling back to sentiment-only prediction."
                )
                return self._generate_sentiment_only_prediction(ticker, aggregated_sentiment)
            
            logger.info(
                "TFT raw prediction shape=%s quantiles=%s last_step=%s",
                predictions.shape,
                quantiles,
                np.round(predictions[0, -1, :], 6).tolist() if predictions.size else None
            )
            
            # Extract quantiles properly
            # TFT typically outputs: [q10, q50, q90] or similar
            # Shape is (batch, time_steps, quantiles)
            n_quantiles = predictions.shape[2] if len(predictions.shape) > 2 else 1
            
            # Get median (middle quantile) - typically index 1 for 3 quantiles [0.1, 0.5, 0.9]
            if n_quantiles > 1:
                median_idx = n_quantiles // 2
            else:
                median_idx = 0
            
            # Extract median prediction and quantiles for CI
            median_predictions = predictions[0, :, median_idx]  # First batch, all time steps, median quantile
            
            # For confidence interval, use min/max across quantiles if available
            if n_quantiles > 1:
                lower_quantile = predictions[0, :, 0]  # First quantile (e.g., 0.1)
                upper_quantile = predictions[0, :, -1]  # Last quantile (e.g., 0.9)
            else:
                lower_quantile = median_predictions
                upper_quantile = median_predictions
            
            # Denormalize predictions with validation
            target_stats = prediction_df.attrs.get('target_stats')
            
            # Store original predictions for validation
            original_median = median_predictions.copy()
            
            if self.scalers and ticker in self.scalers:
                try:
                    scaler = self.scalers[ticker]
                    median_predictions = scaler.inverse_transform(median_predictions.reshape(-1, 1)).flatten()
                    lower_quantile = scaler.inverse_transform(lower_quantile.reshape(-1, 1)).flatten()
                    upper_quantile = scaler.inverse_transform(upper_quantile.reshape(-1, 1)).flatten()
                    
                    # Validate scaler output is reasonable
                    if np.abs(median_predictions).max() > current_price * 100:
                        logger.warning(f"Scaler denormalization produced unreasonable values. Using fallback.")
                        raise ValueError("Scaler output invalid")
                except Exception as e:
                    logger.warning(f"Scaler denormalization failed: {e}. Using fallback method.")
                    # Fall through to target_stats method
                    median_predictions = original_median.copy()
            
            if target_stats and np.abs(median_predictions).max() < 1000:
                # Only use target_stats if predictions look normalized (< 1000)
                mean = target_stats.get('mean', current_price)
                std = target_stats.get('std', 1.0)
                if std <= 0 or not np.isfinite(std):
                    std = 1.0
                if not np.isfinite(mean):
                    mean = current_price
                    
                median_predictions = median_predictions * std + mean
                lower_quantile = lower_quantile * std + mean
                upper_quantile = upper_quantile * std + mean
                
                # Validate denormalized values
                if np.abs(median_predictions).max() > current_price * 100:
                    logger.warning(f"Target stats denormalization produced unreasonable values. Using current price as baseline.")
                    # Reset to current price with small variation
                    median_predictions = np.full_like(median_predictions, current_price)
                    lower_quantile = np.full_like(lower_quantile, current_price * 0.98)
                    upper_quantile = np.full_like(upper_quantile, current_price * 1.02)
            else:
                # Predictions might already be in price space or need different handling
                if np.abs(median_predictions).max() > current_price * 100:
                    # Clearly wrong - use current price as baseline
                    logger.warning(f"Predictions appear to be in wrong scale. Using current price as baseline.")
                    median_predictions = np.full_like(median_predictions, current_price)
                    lower_quantile = np.full_like(lower_quantile, current_price * 0.98)
                    upper_quantile = np.full_like(upper_quantile, current_price * 1.02)
                elif np.abs(median_predictions).max() < 0.01 * current_price:
                    # Too small - likely normalized incorrectly
                    logger.warning(f"Predictions appear too small. Assuming they're normalized and scaling.")
                    # Scale to reasonable range around current price
                    scale_factor = current_price / (np.abs(median_predictions).mean() + 1e-6)
                    if scale_factor > 1e6:  # Scale factor too large, likely wrong
                        logger.warning(f"Scale factor too large ({scale_factor}). Using current price.")
                        median_predictions = np.full_like(median_predictions, current_price)
                        lower_quantile = np.full_like(lower_quantile, current_price * 0.98)
                        upper_quantile = np.full_like(upper_quantile, current_price * 1.02)
                    else:
                        median_predictions = median_predictions * scale_factor
                        lower_quantile = lower_quantile * scale_factor
                        upper_quantile = upper_quantile * scale_factor
                # Otherwise assume already in price space (reasonable values)

            # Replace any remaining NaN/inf with current price
            median_predictions = np.nan_to_num(
                median_predictions,
                nan=current_price,
                posinf=current_price,
                neginf=current_price
            )
            lower_quantile = np.nan_to_num(
                lower_quantile,
                nan=current_price,
                posinf=current_price,
                neginf=current_price
            )
            upper_quantile = np.nan_to_num(
                upper_quantile,
                nan=current_price,
                posinf=current_price,
                neginf=current_price
            )

            # Validate predictions are reasonable (within 0.1x to 10x current price)
            # If predictions are clearly wrong, fall back to sentiment-only
            final_prediction = float(median_predictions[-1])
            max_reasonable_price = current_price * 10.0  # 10x current price max
            min_reasonable_price = current_price * 0.1   # 0.1x current price min
            
            if final_prediction > max_reasonable_price or final_prediction < min_reasonable_price:
                logger.error(
                    f"TFT prediction for {ticker} is clearly wrong: "
                    f"predicted={final_prediction:.2f}, current={current_price:.2f} "
                    f"(ratio={final_prediction/current_price:.2f}x). Falling back to sentiment-only."
                )
                return self._generate_sentiment_only_prediction(ticker, aggregated_sentiment)
            
            # Also check if raw predictions are already huge (before denormalization)
            raw_max = np.abs(median_predictions).max()
            if raw_max > 1e6:  # If raw predictions are > 1 million, likely wrong
                logger.error(
                    f"TFT raw predictions for {ticker} are too large: max={raw_max:.2f}. "
                    f"Falling back to sentiment-only."
                )
                return self._generate_sentiment_only_prediction(ticker, aggregated_sentiment)

            # Calculate predicted return (percentage)
            if current_price > 0:
                predicted_return = (final_prediction - current_price) / current_price
                # Clamp predicted return to reasonable range (-50% to +50%)
                predicted_return = np.clip(predicted_return, -0.5, 0.5)
            else:
                predicted_return = 0.0
            predicted_return = float(np.nan_to_num(predicted_return, nan=0.0, posinf=0.0, neginf=0.0))
            
            # Calculate confidence interval as percentage returns (not absolute prices)
            lower_price = float(lower_quantile[-1])  # Use final prediction step
            upper_price = float(upper_quantile[-1])
            
            # Validate CI prices are reasonable
            if lower_price > max_reasonable_price or lower_price < min_reasonable_price:
                lower_price = current_price * 0.95  # Default to -5%
            if upper_price > max_reasonable_price or upper_price < min_reasonable_price:
                upper_price = current_price * 1.05  # Default to +5%
            
            if current_price > 0:
                lower_return = (lower_price - current_price) / current_price
                upper_return = (upper_price - current_price) / current_price
                # Clamp CI to reasonable range
                lower_return = np.clip(lower_return, -0.5, 0.5)
                upper_return = np.clip(upper_return, -0.5, 0.5)
            else:
                lower_return = -0.05
                upper_return = 0.05
            
            # Ensure CI is valid
            lower_return = float(np.nan_to_num(lower_return, nan=-0.05, posinf=0.5, neginf=-0.5))
            upper_return = float(np.nan_to_num(upper_return, nan=0.05, posinf=0.5, neginf=-0.5))
            
            # Ensure lower < upper
            if lower_return > upper_return:
                lower_return, upper_return = upper_return, lower_return
            
            logger.info(
                "TFT denormalized prices=%s final=%.4f current=%.4f return=%.6f CI=[%.4f%%, %.4f%%]",
                np.round(median_predictions, 4).tolist(),
                final_prediction,
                current_price,
                predicted_return,
                lower_return * 100,
                upper_return * 100
            )
            
            return {
                'predicted_return': float(predicted_return),
                'predicted_price': float(final_prediction),
                'current_price': float(current_price),
                'confidence_interval': {
                    'lower': float(lower_return),  # Percentage return, not absolute price
                    'upper': float(upper_return)   # Percentage return, not absolute price
                },
                'horizon_days': max_decoder,
                'model': 'TFT (Real Prediction)',
                'data_points_used': len(df)
            }
            
        except Exception as e:
            logger.error(f"TFT prediction failed for {ticker}: {e}")
            # Fallback to sentiment-only prediction if TFT fails
            logger.warning(f"Falling back to sentiment-only prediction for {ticker}")
            return self._generate_sentiment_only_prediction(ticker, aggregated_sentiment)
    
    def _generate_sentiment_only_prediction(self, ticker: str, aggregated_sentiment: Dict) -> Dict:
        """
        Generate a basic price prediction based on sentiment only (no TFT model)
        Used when ticker is not in trained set or TFT fails
        NO FAKE DATA - uses real current price from yfinance
        """
        try:
            # Fetch current price from yfinance (REAL DATA)
            import yfinance as yf
            from datetime import datetime
            
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="1d")
            
            if hist.empty:
                logger.warning(f"No price data available for {ticker} from yfinance")
                current_price = 0.0
            else:
                current_price = float(hist['Close'].iloc[-1])
            
            # Simple sentiment-based prediction (conservative)
            # Map sentiment score to expected return range
            sentiment_score = aggregated_sentiment.get('score', 0.0)
            sentiment_confidence = aggregated_sentiment.get('confidence', 0.5)
            
            # Conservative mapping: sentiment score -> expected return
            # Max ±3% return based on sentiment
            max_return = 0.03 * sentiment_confidence
            predicted_return = sentiment_score * max_return
            
            # Calculate predicted price
            if current_price > 0:
                predicted_price = current_price * (1 + predicted_return)
            else:
                predicted_price = 0.0
                predicted_return = 0.0
            
            # Confidence interval based on sentiment confidence
            uncertainty = 0.02 * (1 - sentiment_confidence)  # Higher uncertainty if low confidence
            lower_return = predicted_return - uncertainty
            upper_return = predicted_return + uncertainty
            
            logger.info(
                f"Sentiment-only prediction for {ticker}: return={predicted_return:.4f} "
                f"CI=[{lower_return:.4f}, {upper_return:.4f}]"
            )
            
            return {
                'predicted_return': float(predicted_return),
                'predicted_price': float(predicted_price),
                'current_price': float(current_price),
                'confidence_interval': {
                    'lower': float(lower_return),
                    'upper': float(upper_return)
                },
                'horizon_days': 5,
                'model': 'Sentiment-Based (TFT not available)',
                'data_points_used': 0,
                'note': f'TFT model not trained for {ticker}. Using sentiment-based prediction.'
            }
            
        except Exception as e:
            logger.error(f"Sentiment-only prediction failed for {ticker}: {e}")
            # Last resort: return zero prediction (NO FAKE DATA)
            return {
                'predicted_return': 0.0,
                'predicted_price': 0.0,
                'current_price': 0.0,
                'confidence_interval': {
                    'lower': -0.05,
                    'upper': 0.05
                },
                'horizon_days': 5,
                'model': 'Unavailable',
                'data_points_used': 0,
                'note': f'Unable to generate prediction for {ticker}'
            }
    
    def _engineer_tft_features(self, df: pd.DataFrame, ticker: str, current_sentiment: Dict) -> pd.DataFrame:
        """Add all features required by TFT config"""
        df = df.copy()
        
        # Basic features
        df['Returns'] = df['Close'].pct_change()
        df['momentum_5d'] = df['Close'].pct_change(5)
        df['Volatility'] = df['Returns'].rolling(20).std()
        
        # Time features
        df['Date'] = pd.to_datetime(df['Date'])
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['quarter'] = df['Date'].dt.quarter
        
        # Cyclical time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Sentiment features (use current sentiment for recent days, 0 for historical)
        df['sentiment_score'] = 0.0  # Neutral for historical data
        df['sentiment_confidence'] = 0.5
        # Set last few days to current sentiment
        df.loc[df.index[-5:], 'sentiment_score'] = current_sentiment['score']
        df.loc[df.index[-5:], 'sentiment_confidence'] = current_sentiment['confidence']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            if lag in [1, 5]:
                df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            if lag in [1, 2, 3, 5]:
                df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
            if lag in [1, 5]:
                df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
        
        for lag in [1, 3]:
            df[f'sentiment_score_lag_{lag}'] = df['sentiment_score'].shift(lag)
        
        # Moving averages
        for window in [5, 10, 20]:
            df[f'Close_ma_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}'] = df['Close'].rolling(window).std()
        
        for window in [5, 10]:
            df[f'Returns_ma_{window}'] = df['Returns'].rolling(window).mean()
            df[f'Returns_std_{window}'] = df['Returns'].rolling(window).std()
            df[f'Volume_ma_{window}'] = df['Volume'].rolling(window).mean()
            df[f'RSI_ma_{window}'] = df['RSI'].rolling(window).mean()
            df[f'sentiment_score_ma_{window}'] = df['sentiment_score'].rolling(window).mean()
        
        # Momentum features
        for window in [5, 10, 20]:
            df[f'price_momentum_{window}'] = df['Close'].pct_change(window)
        
        # Other features
        df['volume_change'] = df['Volume'].pct_change()
        
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['Returns'].rolling(window).std()
        
        for window in [20, 50]:
            df[f'price_to_max_{window}'] = df['Close'] / df['Close'].rolling(window).max()
            df[f'price_to_min_{window}'] = df['Close'] / df['Close'].rolling(window).min()
        
        df['sentiment_change'] = df['sentiment_score'].diff()
        df['price_hma_distance'] = (df['Close'] - df['HMA']) / df['HMA']
        df['rsi_overbought'] = (df['RSI'] > 70).astype(float)
        df['rsi_oversold'] = (df['RSI'] < 30).astype(float)
        
        # Drop NaN rows created by rolling features
        df = df.dropna()
        
        return df
    
    def _prepare_tft_dataset(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare DataFrame in format expected by TFT model"""
        max_encoder = self.tft_config.get('max_encoder_length', 60)
        max_decoder = self.tft_config.get('max_prediction_length', 5)
        total_needed = max_encoder + max_decoder
        if len(df) < total_needed:
            raise ValueError(
                f"Insufficient engineered rows for {ticker}: need {total_needed}, got {len(df)}"
            )

        # Keep the most recent encoder+decoder window for prediction
        df = df.iloc[-total_needed:].copy()

        df = df.reset_index(drop=True)
        max_lag = 5  # Maximum lag from engineered lag features
        start_idx = max_encoder + max_lag
        df['time_idx'] = range(start_idx, start_idx + len(df))
        logger.info(
            "Prepared TFT dataframe for %s with %s rows (time_idx %s-%s)",
            ticker,
            len(df),
            df['time_idx'].min(),
            df['time_idx'].max()
        )
        
        # Don't add group_id - use Ticker as-is (it's already in the dataframe)
        # Ensure Ticker column exists
        if 'Ticker' not in df.columns:
            df['Ticker'] = ticker
        
        model_static_reals: List[str] = []
        if self.tft_model is not None and hasattr(self.tft_model, "hparams"):
            model_static_reals = list(self.tft_model.hparams.get('static_reals', []))

        # Ensure all required columns exist
        protected_columns = {
            'decoder_length',
            'encoder_length',
            'target',
            'target_scale',
            'Close_center',
            'Close_scale'
        }
        required_cols = (
            self.tft_config.get('static_features', []) +
            self.tft_config.get('time_varying_known', []) +
            self.tft_config.get('time_varying_unknown', []) +
            [col for col in model_static_reals if col not in protected_columns] +
            [self.tft_config.get('target', 'Close')]
        )
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for TFT: {missing_cols}")
            for col in missing_cols:
                df[col] = 0.0

        # Standardize numeric features (excluding target and identifiers) to approximate training scale
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {
            'time_idx',
            'group_id',
            self.tft_config.get('target', 'Close'),
            'encoder_length',
            'Close_center',
            'Close_scale'
        }
        standardization_stats = {}
        for col in numeric_cols:
            if col in exclude_cols:
                continue
            series = df[col].astype(float)
            mean = float(series.mean()) if len(series) > 0 else 0.0
            std = float(series.std()) if len(series) > 1 else 0.0
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            df[col] = (series - mean) / std
            standardization_stats[col] = {'mean': mean, 'std': std}

        # Store target statistics for downstream denormalization
        target_mean = float(df[self.tft_config.get('target', 'Close')].mean())
        target_std = float(df[self.tft_config.get('target', 'Close')].std())
        if not np.isfinite(target_std) or target_std == 0.0:
            target_std = 1.0
        df.attrs['target_stats'] = {'mean': target_mean, 'std': target_std}

        # Ensure protected columns are not present in the dataframe
        for protected_col in protected_columns:
            if protected_col in df.columns:
                df = df.drop(columns=[protected_col])

        df.attrs['standardization_stats'] = standardization_stats
        
        return df
    
    def _generate_trading_signal(
        self,
        aggregated_sentiment: Dict,
        tft_prediction: Dict
    ) -> Dict:
        """
        Generate trading signal based on sentiment + TFT prediction
        
        Rules (improved with flexible thresholds):
        - BUY: (Positive sentiment >0.1 OR predicted return >3%) AND predicted return >1%
        - SELL: (Negative sentiment <-0.1 OR predicted return <-3%) AND predicted return <-1%
        - HOLD: Otherwise (but with calculated confidence, not hardcoded)
        """
        sentiment_score = aggregated_sentiment['score']
        sentiment_confidence = aggregated_sentiment['confidence']
        predicted_return = tft_prediction['predicted_return']
        
        # Calculate signal strength and confidence
        # Combine sentiment and prediction signals
        sentiment_signal_strength = abs(sentiment_score)
        return_signal_strength = abs(predicted_return)
        
        # Weighted confidence: blend sentiment confidence with prediction confidence
        # If both signals agree, confidence is higher
        if (sentiment_score > 0 and predicted_return > 0) or (sentiment_score < 0 and predicted_return < 0):
            # Signals agree - boost confidence
            signal_agreement = 1.0
        else:
            # Signals disagree - reduce confidence
            signal_agreement = 0.7  # Less penalty for disagreement
        
        # Decision logic with more flexible thresholds
        # LONG: Either strong sentiment (>0.1) OR strong return (>3%), and return must be positive (>1%)
        long_condition = (
            (sentiment_score > 0.1 or predicted_return > 0.03) and 
            predicted_return > 0.01
        )
        
        # SHORT: Either negative sentiment (<-0.1) OR negative return (<-3%), and return must be negative (<-1%)
        short_condition = (
            (sentiment_score < -0.1 or predicted_return < -0.03) and 
            predicted_return < -0.01
        )
        
        if long_condition:
            signal = 'BUY'
            signal_color = 'bullish'
            # Use weighted confidence: sentiment confidence * agreement * return strength
            # Higher return = higher confidence
            return_multiplier = min(return_signal_strength / 0.05, 1.2)  # Cap at 1.2x for very high returns
            confidence = sentiment_confidence * signal_agreement * return_multiplier
            confidence = max(0.6, min(0.95, confidence))  # Clamp between 60% and 95%
        elif short_condition:
            signal = 'SELL'
            signal_color = 'bearish'
            # Use weighted confidence
            return_multiplier = min(return_signal_strength / 0.05, 1.2)
            confidence = sentiment_confidence * signal_agreement * return_multiplier
            confidence = max(0.6, min(0.95, confidence))  # Clamp between 60% and 95%
        else:
            signal = 'HOLD'
            signal_color = 'neutral'
            # Calculate HOLD confidence based on how close we are to a signal
            # If signals are very weak/neutral, confidence in HOLD is higher
            signal_strength = max(sentiment_signal_strength, return_signal_strength)
            if signal_strength < 0.05:
                # Very neutral - high confidence in HOLD
                confidence = max(sentiment_confidence, 0.75)
            elif abs(predicted_return) < 0.01 and abs(sentiment_score) < 0.1:
                # Both signals weak - moderate confidence in HOLD
                confidence = sentiment_confidence * 0.7
            else:
                # Mixed signals - lower confidence in HOLD
                confidence = sentiment_confidence * 0.5
            confidence = max(0.4, min(0.85, confidence))  # Clamp between 40% and 85% for HOLD
        
        return {
            'action': signal,
            'color': signal_color,
            'confidence': float(confidence),
            'reasoning': self._get_signal_reasoning(
                signal,
                sentiment_score,
                predicted_return
            )
        }
    
    def _get_signal_reasoning(
        self,
        signal: str,
        sentiment: float,
        predicted_return: float
    ) -> str:
        """Generate human-readable reasoning for the signal"""
        if signal == 'BUY':
            return (
                f"Positive sentiment ({sentiment:.2f}) combined with "
                f"predicted upward price movement ({predicted_return*100:.1f}%) "
                f"suggests favorable buying opportunity."
            )
        elif signal == 'SELL':
            return (
                f"Negative sentiment ({sentiment:.2f}) and "
                f"predicted downward movement ({predicted_return*100:.1f}%) "
                f"indicate bearish outlook suitable for selling."
            )
        else:
            return (
                f"Mixed or neutral signals (sentiment: {sentiment:.2f}, "
                f"predicted return: {predicted_return*100:.1f}%) "
                f"suggest waiting for clearer directional bias."
            )


# Global instance
trading_signal_service = TradingSignalService()
