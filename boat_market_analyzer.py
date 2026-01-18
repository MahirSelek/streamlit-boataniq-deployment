"""
Boat Market Performance Analyzer
Analyzes boat sales data to calculate market performance metrics
for comparison with financial indices
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoatMarketAnalyzer:
    """Analyzes boat market performance from sales data"""
    
    def __init__(self, boats_df: pd.DataFrame):
        """
        Initialize with boat sales dataframe
        
        Args:
            boats_df: DataFrame with boat sales data
        """
        self.boats_df = boats_df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and clean the boat data"""
        # Extract numeric prices
        self.boats_df['price_numeric'] = self.boats_df['price'].apply(self._extract_price)
        
        # Extract numeric years
        self.boats_df['year_numeric'] = pd.to_numeric(
            self.boats_df['year_built'], 
            errors='coerce'
        )
        
        # Extract length from dimensions
        self.boats_df['length_numeric'] = self.boats_df['dimensions'].apply(self._extract_length)
        
        # Filter out invalid data
        self.boats_df = self.boats_df[
            (self.boats_df['price_numeric'].notna()) & 
            (self.boats_df['price_numeric'] > 0) &
            (self.boats_df['year_numeric'].notna()) &
            (self.boats_df['year_numeric'] >= 1900) &
            (self.boats_df['year_numeric'] <= datetime.now().year)
        ].copy()
        
        logger.info(f"Prepared {len(self.boats_df)} boats with valid price and year data")
    
    def _extract_price(self, price_str) -> Optional[float]:
        """Extract numeric price from price string"""
        if pd.isna(price_str) or price_str == '':
            return None
        
        price_str = str(price_str)
        
        # Skip if price on request or similar
        if any(phrase in price_str.upper() for phrase in ['PRICE ON REQUEST', 'UNDER OFFER', 'N/A', 'ASKING']):
            return None
        
        try:
            # Remove currency symbols and formatting
            price_clean = re.sub(r'[^\d.,]', '', price_str)
            # Handle European format (1.234.567,89) and US format (1,234,567.89)
            price_clean = price_clean.replace('.', '').replace(',', '.')
            # Remove any remaining non-numeric characters
            price_clean = re.sub(r'[^\d.]', '', price_clean)
            
            if price_clean:
                return float(price_clean)
        except:
            pass
        
        return None
    
    def _extract_length(self, dim_str) -> Optional[float]:
        """Extract length from dimensions string"""
        if pd.isna(dim_str) or dim_str == '' or 'N/A' in str(dim_str):
            return None
        
        try:
            # Look for pattern like "12.43 x 4.20 m" or "12.43m x 4.20m"
            match = re.search(r'(\d+\.?\d*)\s*x', str(dim_str))
            if match:
                return float(match.group(1))
        except:
            pass
        
        return None
    
    def calculate_market_performance(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> Dict:
        """
        Calculate boat market performance metrics
        
        Args:
            start_year: Start year for analysis (default: 5 years ago)
            end_year: End year for analysis (default: current year)
        
        Returns:
            Dictionary with performance metrics
        """
        if self.boats_df.empty:
            return {'error': 'No valid boat data available'}
        
        # Set default years
        current_year = datetime.now().year
        if start_year is None:
            start_year = current_year - 5
        if end_year is None:
            end_year = current_year
        
        # Filter by year range
        filtered_df = self.boats_df[
            (self.boats_df['year_numeric'] >= start_year) &
            (self.boats_df['year_numeric'] <= end_year)
        ].copy()
        
        if filtered_df.empty:
            return {'error': f'No data available for years {start_year}-{end_year}'}
        
        # Calculate average prices by year
        yearly_avg_prices = filtered_df.groupby('year_numeric')['price_numeric'].mean()
        yearly_median_prices = filtered_df.groupby('year_numeric')['price_numeric'].median()
        yearly_counts = filtered_df.groupby('year_numeric').size()
        
        # Calculate price appreciation
        if len(yearly_avg_prices) > 1:
            first_avg_price = yearly_avg_prices.iloc[0]
            last_avg_price = yearly_avg_prices.iloc[-1]
            
            first_median_price = yearly_median_prices.iloc[0]
            last_median_price = yearly_median_prices.iloc[-1]
            
            # Total return based on average prices
            total_return_avg = ((last_avg_price - first_avg_price) / first_avg_price) * 100
            
            # Total return based on median prices
            total_return_median = ((last_median_price - first_median_price) / first_median_price) * 100
            
            # Annualized return
            years = len(yearly_avg_prices) - 1
            if years > 0:
                annualized_return_avg = ((last_avg_price / first_avg_price) ** (1 / years) - 1) * 100
                annualized_return_median = ((last_median_price / first_median_price) ** (1 / years) - 1) * 100
            else:
                annualized_return_avg = 0
                annualized_return_median = 0
            
            # Calculate volatility (year-over-year price changes)
            year_over_year_changes = yearly_avg_prices.pct_change().dropna() * 100
            volatility = year_over_year_changes.std() if len(year_over_year_changes) > 1 else 0
            
            # Market size growth
            first_count = yearly_counts.iloc[0]
            last_count = yearly_counts.iloc[-1]
            market_growth = ((last_count - first_count) / first_count) * 100 if first_count > 0 else 0
            
            return {
                'total_return_pct': round(total_return_avg, 2),
                'total_return_median_pct': round(total_return_median, 2),
                'annualized_return_pct': round(annualized_return_avg, 2),
                'annualized_return_median_pct': round(annualized_return_median, 2),
                'volatility_pct': round(volatility, 2),
                'market_growth_pct': round(market_growth, 2),
                'start_year': int(start_year),
                'end_year': int(end_year),
                'years_analyzed': int(years + 1),
                'start_avg_price': round(first_avg_price, 2),
                'end_avg_price': round(last_avg_price, 2),
                'start_median_price': round(first_median_price, 2),
                'end_median_price': round(last_median_price, 2),
                'total_listings': len(filtered_df),
                'yearly_data': {
                    'years': [int(y) for y in yearly_avg_prices.index],
                    'avg_prices': [round(p, 2) for p in yearly_avg_prices.values],
                    'median_prices': [round(p, 2) for p in yearly_median_prices.values],
                    'counts': [int(c) for c in yearly_counts.values]
                }
            }
        else:
            return {
                'error': 'Insufficient data for performance calculation',
                'years_available': [int(y) for y in yearly_avg_prices.index]
            }
    
    def calculate_category_performance(self, category_col: str = 'title', start_year: Optional[int] = None) -> Dict:
        """
        Calculate performance by boat category/brand
        
        Args:
            category_col: Column to group by (e.g., 'title' for brand)
            start_year: Start year for analysis
        
        Returns:
            Dictionary with performance by category
        """
        if self.boats_df.empty:
            return {}
        
        current_year = datetime.now().year
        if start_year is None:
            start_year = current_year - 5
        
        # Extract brand from title (first word)
        self.boats_df['brand'] = self.boats_df['title'].str.split().str[0]
        
        filtered_df = self.boats_df[
            (self.boats_df['year_numeric'] >= start_year) &
            (self.boats_df['year_numeric'] <= current_year)
        ].copy()
        
        if filtered_df.empty:
            return {}
        
        # Group by brand and calculate average prices
        brand_performance = {}
        for brand in filtered_df['brand'].unique():
            brand_df = filtered_df[filtered_df['brand'] == brand]
            if len(brand_df) >= 5:  # Only include brands with at least 5 listings
                yearly_prices = brand_df.groupby('year_numeric')['price_numeric'].mean()
                if len(yearly_prices) > 1:
                    first_price = yearly_prices.iloc[0]
                    last_price = yearly_prices.iloc[-1]
                    return_pct = ((last_price - first_price) / first_price) * 100
                    
                    brand_performance[brand] = {
                        'return_pct': round(return_pct, 2),
                        'avg_price': round(brand_df['price_numeric'].mean(), 2),
                        'count': len(brand_df)
                    }
        
        # Sort by return percentage
        sorted_brands = sorted(brand_performance.items(), key=lambda x: x[1]['return_pct'], reverse=True)
        
        return {
            'top_performers': dict(sorted_brands[:10]),
            'bottom_performers': dict(sorted_brands[-10:]),
            'start_year': start_year,
            'end_year': current_year
        }
    
    def get_market_summary(self, start_year: Optional[int] = None) -> Dict:
        """
        Get comprehensive market summary
        
        Args:
            start_year: Start year for analysis
        
        Returns:
            Dictionary with market summary
        """
        performance = self.calculate_market_performance(start_year=start_year)
        category_perf = self.calculate_category_performance(start_year=start_year)
        
        return {
            'overall_performance': performance,
            'category_performance': category_perf,
            'total_boats': len(self.boats_df),
            'price_range': {
                'min': float(self.boats_df['price_numeric'].min()),
                'max': float(self.boats_df['price_numeric'].max()),
                'median': float(self.boats_df['price_numeric'].median()),
                'mean': float(self.boats_df['price_numeric'].mean())
            },
            'year_range': {
                'min': int(self.boats_df['year_numeric'].min()),
                'max': int(self.boats_df['year_numeric'].max())
            }
        }


if __name__ == "__main__":
    # Test the analyzer
    import sys
    
    # Load sample data
    try:
        boats_df = pd.read_csv('all_boats_data.csv')
        analyzer = BoatMarketAnalyzer(boats_df)
        
        print("Calculating boat market performance...")
        performance = analyzer.calculate_market_performance(start_year=2018)
        
        print("\n=== Boat Market Performance ===")
        print(f"Total Return: {performance.get('total_return_pct', 0)}%")
        print(f"Annualized Return: {performance.get('annualized_return_pct', 0)}%")
        print(f"Volatility: {performance.get('volatility_pct', 0)}%")
        print(f"Market Growth: {performance.get('market_growth_pct', 0)}%")
        
    except Exception as e:
        print(f"Error: {e}")

