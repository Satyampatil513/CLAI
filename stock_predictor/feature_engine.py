"""
Feature Engineering Module for Stock Prediction
Creates technical indicators, sentiment analysis, and other features for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from textblob import TextBlob


class FeatureEngine:
    """Creates features for stock prediction model"""
    
    def __init__(self):
        """Initialize the feature engine"""
        pass
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to stock price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['Close'], period=14)
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # Stochastic Oscillator
        df['Stoch_K'], df['Stoch_D'] = self._calculate_stochastic(df)
        
        # Average True Range (ATR)
        df['ATR'] = self._calculate_atr(df, period=14)
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Price momentum
        df['Momentum_1d'] = df['Close'].pct_change(1)
        df['Momentum_5d'] = df['Close'].pct_change(5)
        df['Momentum_10d'] = df['Close'].pct_change(10)
        df['Momentum_20d'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility_10d'] = df['Close'].pct_change().rolling(window=10).std()
        df['Volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Price position relative to high/low
        df['High_Low_pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=3).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def analyze_sentiment(self, news_articles: List[Dict]) -> Dict[str, float]:
        """
        Analyze sentiment from news articles
        
        Args:
            news_articles: List of news articles with 'title' field
            
        Returns:
            Dictionary with sentiment scores
        """
        if not news_articles:
            return {
                'sentiment_score': 0.0,
                'sentiment_polarity': 0.0,
                'sentiment_subjectivity': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in news_articles:
            title = article.get('title', '')
            if title:
                blob = TextBlob(title)
                polarity = blob.sentiment.polarity
                sentiments.append(blob.sentiment)
                
                if polarity > 0.1:
                    positive_count += 1
                elif polarity < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1
        
        # Calculate average sentiment
        avg_polarity = np.mean([s.polarity for s in sentiments]) if sentiments else 0.0
        avg_subjectivity = np.mean([s.subjectivity for s in sentiments]) if sentiments else 0.0
        
        # Overall sentiment score (-1 to 1)
        sentiment_score = avg_polarity
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_polarity': avg_polarity,
            'sentiment_subjectivity': avg_subjectivity,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
    
    def create_target_variable(self, df: pd.DataFrame, days_ahead: int = 5, threshold: float = 0.02) -> pd.DataFrame:
        """
        Create target variable for prediction (buy/sell/hold)
        
        Args:
            df: DataFrame with stock data
            days_ahead: Number of days to look ahead
            threshold: Minimum price change percentage for buy/sell signal
            
        Returns:
            DataFrame with target variable
        """
        df = df.copy()
        
        # Calculate future return
        df['Future_Return'] = df['Close'].shift(-days_ahead) / df['Close'] - 1
        
        # Create target: 1 (buy), 0 (hold), -1 (sell)
        df['Target'] = 0
        df.loc[df['Future_Return'] > threshold, 'Target'] = 1  # Buy signal
        df.loc[df['Future_Return'] < -threshold, 'Target'] = -1  # Sell signal
        
        # Create binary target for classification
        df['Target_Binary'] = (df['Target'] == 1).astype(int)
        
        return df
    
    def add_fundamental_features(self, df: pd.DataFrame, fundamental_data: Dict) -> pd.DataFrame:
        """
        Add fundamental analysis features
        
        Args:
            df: DataFrame with stock data
            fundamental_data: Dictionary with fundamental metrics
            
        Returns:
            DataFrame with added fundamental features
        """
        df = df.copy()
        
        # Add fundamental metrics
        df['PE_Ratio'] = fundamental_data.get('pe_ratio', 0)
        df['Forward_PE'] = fundamental_data.get('forward_pe', 0)
        df['PEG_Ratio'] = fundamental_data.get('peg_ratio', 0)
        df['Price_to_Book'] = fundamental_data.get('price_to_book', 0)
        df['Dividend_Yield'] = fundamental_data.get('dividend_yield', 0)
        df['Beta'] = fundamental_data.get('beta', 0)
        df['Revenue_Growth'] = fundamental_data.get('revenue_growth', 0)
        df['Earnings_Growth'] = fundamental_data.get('earnings_growth', 0)
        df['Profit_Margin'] = fundamental_data.get('profit_margin', 0)
        df['ROE'] = fundamental_data.get('roe', 0)
        
        return df
    
    def prepare_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean features for machine learning
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Cleaned DataFrame ready for ML
        """
        df = df.copy()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        return df


if __name__ == "__main__":
    # Example usage
    from data_collector import DataCollector
    
    collector = DataCollector()
    engine = FeatureEngine()
    
    # Get stock data
    print("Fetching stock data...")
    df = collector.get_stock_data('RELIANCE.NS', period='1y')
    
    # Add technical indicators
    print("Adding technical indicators...")
    df = engine.add_technical_indicators(df)
    
    # Create target variable
    print("Creating target variable...")
    df = engine.create_target_variable(df, days_ahead=5, threshold=0.02)
    
    # Get fundamental data
    print("Getting fundamental data...")
    fundamental = collector.get_stock_info('RELIANCE.NS')
    df = engine.add_fundamental_features(df, fundamental)
    
    # Prepare for ML
    print("Preparing features for ML...")
    df = engine.prepare_features_for_ml(df)
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nTarget distribution:")
    print(df['Target'].value_counts())
