"""
Data preprocessing for SentiTrade-HMA.
Includes technical indicators calculation (HMA, RSI, MACD).
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


class TechnicalIndicatorCalculator:
    """Calculate technical indicators for financial data."""
    
    def __init__(self):
        self.config = get_config()
        self.data_config = self.config.get_data_config()
    
    def calculate_hma(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Hull Moving Average (HMA).
        
        Args:
            prices: Price series
            period: Period for HMA calculation
            
        Returns:
            HMA series
        """
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = prices.rolling(window=half_period).mean()
        wma_full = prices.rolling(window=period).mean()
        
        raw_hma = 2 * wma_half - wma_full
        hma = raw_hma.rolling(window=sqrt_period).mean()
        
        return hma
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: Period for RSI calculation
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (macd, signal, histogram)
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def add_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        logger.info('Calculating technical indicators...')
        
        data = data.copy()
        
        # Get config parameters
        indicators_config = self.data_config['preprocessing']['technical_indicators']
        
        hma_period = next(i['period'] for i in indicators_config if i['name'] == 'HMA')
        rsi_period = next(i['period'] for i in indicators_config if i['name'] == 'RSI')
        macd_config = next(i for i in indicators_config if i['name'] == 'MACD')
        
        # Calculate for each ticker
        for ticker in data['Ticker'].unique():
            mask = data['Ticker'] == ticker
            
            # HMA
            data.loc[mask, 'HMA'] = self.calculate_hma(
                data.loc[mask, 'Close'],
                period=hma_period
            )
            
            # RSI
            data.loc[mask, 'RSI'] = self.calculate_rsi(
                data.loc[mask, 'Close'],
                period=rsi_period
            )
            
            # MACD
            macd, signal, hist = self.calculate_macd(
                data.loc[mask, 'Close'],
                fast=macd_config['fast'],
                slow=macd_config['slow'],
                signal=macd_config['signal']
            )
            data.loc[mask, 'MACD'] = macd
            data.loc[mask, 'MACD_Signal'] = signal
            data.loc[mask, 'MACD_Hist'] = hist
            
            logger.info(f'   {ticker}: Indicators calculated')
        
        logger.info(f' All indicators calculated')
        return data
    
    def save(self, data: pd.DataFrame, filepath: str):
        """Save processed data to CSV."""
        ensure_dir(Path(filepath).parent)
        data.to_csv(filepath, index=False)
        logger.info(f' Saved to {filepath}')


class SentimentAggregator:
    """Aggregate sentiment scores from news articles."""
    
    def __init__(self):
        self.config = get_config()
    
    def aggregate_daily_sentiment(
        self,
        sentiment_df: pd.DataFrame,
        method: str = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by ticker and date.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            method: Aggregation method ('mean', 'median', 'weighted_mean')
            
        Returns:
            Aggregated sentiment DataFrame
        """
        if method == 'mean':
            agg_df = sentiment_df.groupby(['ticker', 'date'])['sentiment'].mean().reset_index()
        elif method == 'median':
            agg_df = sentiment_df.groupby(['ticker', 'date'])['sentiment'].median().reset_index()
        elif method == 'weighted_mean':
            # Weight by confidence if available
            if 'confidence' in sentiment_df.columns:
                agg_df = sentiment_df.groupby(['ticker', 'date']).apply(
                    lambda x: np.average(x['sentiment'], weights=x['confidence'])
                ).reset_index(name='sentiment')
            else:
                agg_df = sentiment_df.groupby(['ticker', 'date'])['sentiment'].mean().reset_index()
        
        return agg_df
    
    def merge_with_financial(
        self,
        financial_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge sentiment scores with financial data.
        
        Args:
            financial_df: Financial data DataFrame
            sentiment_df: Sentiment data DataFrame
            
        Returns:
            Merged DataFrame
        """
        # Ensure date columns are datetime
        financial_df['Date'] = pd.to_datetime(financial_df['Date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Merge on ticker and date
        merged = financial_df.merge(
            sentiment_df,
            left_on=['Ticker', 'Date'],
            right_on=['ticker', 'date'],
            how='left'
        )
        
        # Fill missing sentiment with neutral (0)
        merged['sentiment'] = merged['sentiment'].fillna(0.0)
        
        # Drop duplicate columns
        merged = merged.drop(columns=['ticker', 'date'], errors='ignore')
        
        return merged


if __name__ == '__main__':
    # Test technical indicators
    import yfinance as yf
    
    # Download sample data
    data = yf.download('AAPL', start='2024-01-01', end='2024-12-19', progress=False)
    data = data.reset_index()
    data['Ticker'] = 'AAPL'
    
    # Calculate indicators
    calculator = TechnicalIndicatorCalculator()
    data_with_indicators = calculator.add_all_indicators(data)
    
    print(' Technical indicators calculated')
    print(data_with_indicators[['Date', 'Close', 'HMA', 'RSI', 'MACD']].tail())
