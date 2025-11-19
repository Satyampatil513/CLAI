"""
Tests for Stock Prediction System
Simple tests to validate core functionality
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stock_predictor.feature_engine import FeatureEngine


class TestFeatureEngine(unittest.TestCase):
    """Test the feature engineering module"""
    
    def setUp(self):
        """Set up test data"""
        self.engine = FeatureEngine()
        
        # Create sample stock data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        np.random.seed(42)
        prices = 2000 * np.exp(np.cumsum(np.random.normal(0, 0.02, 100)))
        
        self.sample_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100),
        }, index=dates)
    
    def test_add_technical_indicators(self):
        """Test that technical indicators are added correctly"""
        df = self.engine.add_technical_indicators(self.sample_data.copy())
        
        # Check that new columns are added
        self.assertGreater(len(df.columns), len(self.sample_data.columns))
        
        # Check specific indicators exist
        required_indicators = ['SMA_20', 'RSI', 'MACD', 'BB_upper', 'EMA_12']
        for indicator in required_indicators:
            self.assertIn(indicator, df.columns, f"Missing indicator: {indicator}")
        
        # Most values should be finite (some early periods may have NaN/inf)
        finite_ratio = (~df.replace([np.inf, -np.inf], np.nan).isnull()).sum().sum() / (df.shape[0] * df.shape[1])
        self.assertGreater(finite_ratio, 0.5)  # At least 50% should be finite
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        df = self.engine.add_technical_indicators(self.sample_data.copy())
        
        # RSI should be between 0 and 100
        rsi_values = df['RSI'].dropna()
        self.assertTrue((rsi_values >= 0).all() and (rsi_values <= 100).all())
    
    def test_create_target_variable(self):
        """Test target variable creation"""
        df = self.engine.add_technical_indicators(self.sample_data.copy())
        df = self.engine.create_target_variable(df, days_ahead=5, threshold=0.02)
        
        # Check target columns exist
        self.assertIn('Target', df.columns)
        self.assertIn('Target_Binary', df.columns)
        
        # Check target values are valid (-1, 0, 1)
        target_values = df['Target'].dropna().unique()
        valid_values = {-1, 0, 1}
        self.assertTrue(all(val in valid_values for val in target_values))
        
        # Check binary target is 0 or 1
        binary_values = df['Target_Binary'].dropna().unique()
        self.assertTrue(all(val in [0, 1] for val in binary_values))
    
    def test_sentiment_analysis_empty(self):
        """Test sentiment analysis with empty news"""
        sentiment = self.engine.analyze_sentiment([])
        
        self.assertEqual(sentiment['sentiment_score'], 0.0)
        self.assertEqual(sentiment['positive_count'], 0)
        self.assertEqual(sentiment['negative_count'], 0)
    
    def test_sentiment_analysis_positive(self):
        """Test sentiment analysis with positive news"""
        news = [
            {'title': 'Stock surges to new highs on excellent earnings'},
            {'title': 'Company announces amazing growth prospects'},
        ]
        
        sentiment = self.engine.analyze_sentiment(news)
        
        # Sentiment should be positive
        self.assertGreater(sentiment['sentiment_score'], 0)
        self.assertGreater(sentiment['positive_count'], 0)
    
    def test_prepare_features_for_ml(self):
        """Test feature preparation for ML"""
        df = self.engine.add_technical_indicators(self.sample_data.copy())
        df = self.engine.create_target_variable(df)
        df_clean = self.engine.prepare_features_for_ml(df)
        
        # Check no NaN values remain
        self.assertFalse(df_clean.isnull().any().any())
        
        # Check no infinite values
        self.assertFalse(df_clean.replace([np.inf, -np.inf], np.nan).isnull().any().any())


class TestStockDataStructure(unittest.TestCase):
    """Test data structure requirements"""
    
    def test_stock_symbol_format(self):
        """Test that stock symbols are correctly formatted"""
        from stock_predictor.data_collector import DataCollector
        
        collector = DataCollector()
        
        # All symbols should end with .NS for NSE
        for symbol in collector.LARGE_CAP_STOCKS:
            self.assertTrue(symbol.endswith('.NS'), f"Invalid symbol format: {symbol}")
    
    def test_large_cap_stocks_list(self):
        """Test that large cap stocks list is not empty"""
        from stock_predictor.data_collector import DataCollector
        
        collector = DataCollector()
        
        self.assertGreater(len(collector.LARGE_CAP_STOCKS), 0)
        self.assertIsInstance(collector.LARGE_CAP_STOCKS, list)


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("RUNNING STOCK PREDICTION SYSTEM TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestStockDataStructure))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
