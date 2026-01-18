"""
Financial Indices Fetcher
Fetches historical data for S&P 500, BIST 100, and Nasdaq indices
for comparison with boat market performance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialIndicesFetcher:
    """Fetches and processes financial indices data"""
    
    def __init__(self):
        self.indices = {
            'SP500': '^GSPC',  # S&P 500
            'NASDAQ': '^IXIC',  # Nasdaq Composite
            'BIST100': 'XU100.IS'  # BIST 100 (Istanbul Stock Exchange)
        }
        self.cache = {}
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour
    
    def fetch_index_data(self, index_symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a financial index
        
        Args:
            index_symbol: Yahoo Finance symbol for the index
            period: Time period (1y, 2y, 5y, 10y, max)
        
        Returns:
            DataFrame with historical prices or None if error
        """
        try:
            ticker = yf.Ticker(index_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data returned for {index_symbol}")
                return None
            
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {index_symbol}: {e}")
            return None
    
    def calculate_returns(self, data: pd.DataFrame, start_date: Optional[str] = None) -> Dict:
        """
        Calculate returns and performance metrics for an index
        
        Args:
            data: DataFrame with historical prices
            start_date: Start date for calculation (YYYY-MM-DD)
        
        Returns:
            Dictionary with performance metrics
        """
        if data is None or data.empty:
            return {}
        
        try:
            # Filter by start date if provided
            if start_date:
                data = data[data.index >= start_date]
            
            if data.empty:
                return {}
            
            # Get first and last closing prices
            first_price = data['Close'].iloc[0]
            last_price = data['Close'].iloc[-1]
            
            # Calculate total return
            total_return = ((last_price - first_price) / first_price) * 100
            
            # Calculate annualized return
            days = (data.index[-1] - data.index[0]).days
            years = days / 365.25
            if years > 0:
                annualized_return = ((last_price / first_price) ** (1 / years) - 1) * 100
            else:
                annualized_return = 0
            
            # Calculate volatility (standard deviation of daily returns)
            daily_returns = data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
            
            # Get current price
            current_price = last_price
            
            # Calculate max drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            return {
                'total_return_pct': round(total_return, 2),
                'annualized_return_pct': round(annualized_return, 2),
                'volatility_pct': round(volatility, 2),
                'max_drawdown_pct': round(max_drawdown, 2),
                'current_price': round(current_price, 2),
                'start_price': round(first_price, 2),
                'start_date': data.index[0].strftime('%Y-%m-%d'),
                'end_date': data.index[-1].strftime('%Y-%m-%d'),
                'days': days
            }
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return {}
    
    def get_all_indices_performance(self, period: str = "5y", start_date: Optional[str] = None) -> Dict:
        """
        Get performance data for all tracked indices
        
        Args:
            period: Time period for data fetch
            start_date: Start date for calculation (YYYY-MM-DD)
        
        Returns:
            Dictionary with performance data for each index
        """
        results = {}
        
        for index_name, symbol in self.indices.items():
            logger.info(f"Fetching data for {index_name} ({symbol})...")
            
            # Check cache
            cache_key = f"{index_name}_{period}_{start_date}"
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    logger.info(f"Using cached data for {index_name}")
                    results[index_name] = cached_data
                    continue
            
            # Fetch data
            data = self.fetch_index_data(symbol, period)
            
            if data is not None:
                performance = self.calculate_returns(data, start_date)
                performance['symbol'] = symbol
                performance['index_name'] = index_name
                
                # Cache the result
                self.cache[cache_key] = (datetime.now(), performance)
                
                results[index_name] = performance
            else:
                logger.warning(f"Failed to fetch data for {index_name}")
                results[index_name] = {
                    'error': f'Failed to fetch data for {index_name}',
                    'index_name': index_name,
                    'symbol': symbol
                }
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def get_historical_prices(self, index_name: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """
        Get historical price data for a specific index
        
        Args:
            index_name: Name of the index (SP500, NASDAQ, BIST100)
            period: Time period (1y, 2y, 5y, 10y, max)
        
        Returns:
            DataFrame with historical prices
        """
        if index_name not in self.indices:
            logger.error(f"Unknown index: {index_name}")
            return None
        
        symbol = self.indices[index_name]
        return self.fetch_index_data(symbol, period)
    
    def get_comparison_summary(self, period: str = "5y", start_date: Optional[str] = None) -> Dict:
        """
        Get a summary comparison of all indices
        
        Args:
            period: Time period for data fetch
            start_date: Start date for calculation
        
        Returns:
            Dictionary with comparison summary
        """
        performance = self.get_all_indices_performance(period, start_date)
        
        # Calculate averages
        returns = [p.get('total_return_pct', 0) for p in performance.values() if 'error' not in p]
        annualized = [p.get('annualized_return_pct', 0) for p in performance.values() if 'error' not in p]
        
        summary = {
            'indices': performance,
            'average_return_pct': round(sum(returns) / len(returns), 2) if returns else 0,
            'average_annualized_return_pct': round(sum(annualized) / len(annualized), 2) if annualized else 0,
            'best_performer': max(performance.items(), key=lambda x: x[1].get('total_return_pct', -999) if 'error' not in x[1] else -999)[0] if performance else None,
            'period': period,
            'start_date': start_date,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary


if __name__ == "__main__":
    # Test the fetcher
    fetcher = FinancialIndicesFetcher()
    
    print("Fetching financial indices data...")
    summary = fetcher.get_comparison_summary(period="5y")
    
    print("\n=== Financial Indices Performance Summary ===")
    for index_name, data in summary['indices'].items():
        if 'error' not in data:
            print(f"\n{index_name}:")
            print(f"  Total Return: {data.get('total_return_pct', 0)}%")
            print(f"  Annualized Return: {data.get('annualized_return_pct', 0)}%")
            print(f"  Volatility: {data.get('volatility_pct', 0)}%")
            print(f"  Max Drawdown: {data.get('max_drawdown_pct', 0)}%")
        else:
            print(f"\n{index_name}: {data.get('error', 'Unknown error')}")
    
    print(f"\nAverage Return: {summary['average_return_pct']}%")
    print(f"Average Annualized Return: {summary['average_annualized_return_pct']}%")

