"""
Data Collection Module for Indian Stock Market
Collects stock prices, financial news, and fundamental data for Large Cap Indian stocks
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup


class DataCollector:
    """Collects data from various sources for Indian stocks"""
    
    # Top Large Cap Indian stocks (NSE symbols)
    LARGE_CAP_STOCKS = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS',
        'HCLTECH.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'SUNPHARMA.NS'
    ]
    
    def __init__(self):
        """Initialize the data collector"""
        self.stocks = self.LARGE_CAP_STOCKS
        
    def get_stock_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with stock price data
        """
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            df['Symbol'] = symbol
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, symbols: Optional[List[str]] = None, period: str = '1y') -> pd.DataFrame:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols (default: Large Cap stocks)
            period: Time period
            
        Returns:
            Combined DataFrame with all stock data
        """
        if symbols is None:
            symbols = self.stocks
            
        all_data = []
        for symbol in symbols:
            df = self.get_stock_data(symbol, period)
            if not df.empty:
                all_data.append(df)
                
        if all_data:
            return pd.concat(all_data, ignore_index=False)
        return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get company information and fundamental data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract key metrics
            metrics = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'roe': info.get('returnOnEquity', 0),
            }
            return metrics
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def get_financial_news(self, symbol: str, query: Optional[str] = None) -> List[Dict]:
        """
        Fetch financial news related to the stock
        
        Args:
            symbol: Stock symbol
            query: Custom search query (default: company name)
            
        Returns:
            List of news articles with title, description, and URL
        """
        try:
            stock = yf.Ticker(symbol)
            news = stock.news
            
            articles = []
            for article in news[:10]:  # Limit to 10 most recent
                articles.append({
                    'title': article.get('title', ''),
                    'publisher': article.get('publisher', ''),
                    'link': article.get('link', ''),
                    'published': article.get('providerPublishTime', 0),
                    'type': article.get('type', ''),
                })
            return articles
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price
        """
        try:
            stock = yf.Ticker(symbol)
            return stock.info.get('currentPrice', 0)
        except Exception as e:
            print(f"Error fetching latest price for {symbol}: {e}")
            return 0.0
            
    def get_all_large_cap_info(self) -> pd.DataFrame:
        """
        Get information for all large cap stocks
        
        Returns:
            DataFrame with all stock information
        """
        all_info = []
        for symbol in self.stocks:
            info = self.get_stock_info(symbol)
            if info:
                all_info.append(info)
        
        if all_info:
            return pd.DataFrame(all_info)
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Get data for a single stock
    print("Fetching data for RELIANCE.NS...")
    reliance_data = collector.get_stock_data('RELIANCE.NS', period='3mo')
    print(f"Data points: {len(reliance_data)}")
    print(reliance_data.head())
    
    # Get company info
    print("\nFetching company info...")
    info = collector.get_stock_info('RELIANCE.NS')
    for key, value in info.items():
        print(f"{key}: {value}")
