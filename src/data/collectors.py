"""
Data collectors for financial data and news.
"""
import yfinance as yf
import pandas as pd
from typing import List, Optional
import time
import requests
from pathlib import Path

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


class FinancialDataCollector:
    """Collector for financial market data from Yahoo Finance."""
    
    def __init__(self):
        self.config = get_config()
        self.data_config = self.config.get_data_config()
        
    def collect(self, tickers: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Collect historical price data for multiple tickers."""
        logger.info(f'Collecting data for {len(tickers)} tickers...')
        
        if start_date is None:
            start_date = self.data_config['data_sources']['financial']['start_date']
        if end_date is None:
            end_date = self.data_config['data_sources']['financial']['end_date']
        
        all_data = []
        
        for ticker in tickers:
            try:
                logger.info(f'  Downloading {ticker}...')
                ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                if not ticker_data.empty:
                    # IMPORTANT: Flatten MultiIndex columns if present
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.get_level_values(0)
                    
                    ticker_data = ticker_data.reset_index()
                    ticker_data['Ticker'] = ticker
                    all_data.append(ticker_data)
                    logger.info(f'   {ticker}: {len(ticker_data)} rows')
                else:
                    logger.warning(f'   {ticker}: No data')
                    
            except Exception as e:
                logger.error(f'   {ticker}: {str(e)}')
                continue
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            logger.info(f' Total: {len(df)} rows')
            return df
        else:
            logger.error(' No data collected')
            return pd.DataFrame()
    
    def save(self, data: pd.DataFrame, filepath: str):
        """Save financial data to CSV."""
        ensure_dir(Path(filepath).parent)
        data.to_csv(filepath, index=False)
        logger.info(f' Saved to {filepath}')


class NewsAPICollector:
    """Collector for financial news from News API."""
    
    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.news_api_key
        self.base_url = 'https://newsapi.org/v2/everything'
        
    def collect(self, tickers: List[str], start_date: str, end_date: str, max_articles: int = 100) -> pd.DataFrame:
        """Collect news articles for given tickers."""
        logger.info(f'Collecting news for {len(tickers)} tickers...')
        all_news = []
        
        for ticker in tickers:
            try:
                query = f'{ticker} stock OR {ticker} earnings'
                params = {
                    'q': query,
                    'from': start_date,
                    'to': end_date,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(max_articles, 100),
                    'apiKey': self.api_key
                }
                
                response = requests.get(self.base_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    for article in articles:
                        all_news.append({
                            'ticker': ticker,
                            'date': article['publishedAt'][:10],
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article['source']['name'],
                            'url': article.get('url', '')
                        })
                    logger.info(f'   {ticker}: {len(articles)} articles')
                else:
                    logger.warning(f'   {ticker}: HTTP {response.status_code}')
                time.sleep(1)
            except Exception as e:
                logger.error(f'   {ticker}: {str(e)}')
                continue
        
        if all_news:
            df = pd.DataFrame(all_news)
            logger.info(f' Total: {len(df)} articles')
            return df
        else:
            logger.warning('  No news collected')
            return pd.DataFrame()
    
    def save(self, data: pd.DataFrame, filepath: str):
        """Save news data to CSV."""
        ensure_dir(Path(filepath).parent)
        data.to_csv(filepath, index=False)
        logger.info(f' Saved to {filepath}')
