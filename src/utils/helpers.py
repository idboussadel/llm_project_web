
"""Helper functions for SentiTrade-HMA project."""
import pandas as pd
import numpy as np
from typing import List, Tuple
from pathlib import Path


def load_tickers(filepath: str) -> List[str]:
    """Load list of tickers from file."""
    with open(filepath, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers


def save_tickers(tickers: List[str], filepath: str) -> None:
    """Save list of tickers to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write('\n'.join(tickers))


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage returns."""
    return prices.pct_change(periods=periods)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Sharpe Ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0.0
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate (proportion of positive returns)."""
    return (returns > 0).sum() / len(returns)


def train_val_test_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: DataFrame to split
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


if __name__ == "__main__":
    # Test helpers
    import pandas as pd
    
    # Test Sharpe ratio
    returns = pd.Series(np.random.randn(252) * 0.01)
    sharpe = calculate_sharpe_ratio(returns)
    print(f"✅ Sharpe Ratio: {sharpe:.2f}")
    
    # Test max drawdown
    drawdown = calculate_max_drawdown(returns)
    print(f"✅ Max Drawdown: {drawdown:.2%}")

