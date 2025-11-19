"""
Stock Prediction Model for Large Cap Indian Stocks
Main predictor class that integrates data collection, feature engineering, and ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .data_collector import DataCollector
from .feature_engine import FeatureEngine


class StockPredictor:
    """
    Main predictor class for Indian Large Cap stocks
    Combines technical analysis, fundamental analysis, and news sentiment
    """
    
    def __init__(self):
        """Initialize the stock predictor"""
        self.data_collector = DataCollector()
        self.feature_engine = FeatureEngine()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_trained = False
        
    def prepare_training_data(self, symbol: str, period: str = '2y') -> Tuple[pd.DataFrame, bool]:
        """
        Prepare training data for a specific stock
        
        Args:
            symbol: Stock symbol
            period: Historical period to fetch
            
        Returns:
            Tuple of (prepared DataFrame, success flag)
        """
        print(f"Fetching data for {symbol}...")
        df = self.data_collector.get_stock_data(symbol, period=period)
        
        if df.empty:
            print(f"No data available for {symbol}")
            return pd.DataFrame(), False
        
        print(f"Adding technical indicators...")
        df = self.feature_engine.add_technical_indicators(df)
        
        print(f"Fetching fundamental data...")
        fundamental = self.data_collector.get_stock_info(symbol)
        if fundamental:
            df = self.feature_engine.add_fundamental_features(df, fundamental)
        
        print(f"Fetching news for sentiment analysis...")
        news = self.data_collector.get_financial_news(symbol)
        sentiment = self.feature_engine.analyze_sentiment(news)
        
        # Add sentiment features
        for key, value in sentiment.items():
            df[key] = value
        
        print(f"Creating target variable...")
        df = self.feature_engine.create_target_variable(df, days_ahead=5, threshold=0.02)
        
        print(f"Preparing features for ML...")
        df = self.feature_engine.prepare_features_for_ml(df)
        
        if df.empty:
            print(f"Not enough data after preprocessing for {symbol}")
            return pd.DataFrame(), False
        
        return df, True
    
    def train_model(self, symbols: Optional[list] = None, period: str = '2y'):
        """
        Train the prediction model on multiple stocks
        
        Args:
            symbols: List of stock symbols (default: all large cap stocks)
            period: Historical period to fetch
        """
        if symbols is None:
            symbols = self.data_collector.LARGE_CAP_STOCKS[:5]  # Use first 5 for faster training
        
        print(f"Training model on {len(symbols)} stocks...")
        
        # Collect data from all stocks
        all_data = []
        for symbol in symbols:
            df, success = self.prepare_training_data(symbol, period)
            if success:
                all_data.append(df)
                print(f"Added {len(df)} samples from {symbol}")
        
        if not all_data:
            raise ValueError("No training data available")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal training samples: {len(combined_df)}")
        
        # Select features (exclude target and date-related columns)
        exclude_cols = ['Target', 'Target_Binary', 'Future_Return', 'Symbol']
        self.feature_columns = [col for col in combined_df.columns if col not in exclude_cols]
        
        X = combined_df[self.feature_columns]
        y = combined_df['Target_Binary']  # Binary classification: buy (1) or not buy (0)
        
        print(f"Features: {len(self.feature_columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create ensemble model
        print("\nTraining ensemble model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Voting classifier
        self.model = VotingClassifier(
            estimators=[('rf', rf_model), ('gb', gb_model)],
            voting='soft'
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("\n=== Model Evaluation ===")
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.model_trained = True
        print("\n✓ Model training completed!")
    
    def predict(self, symbol: str) -> Dict:
        """
        Make prediction for a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with prediction results
        """
        if not self.model_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print(f"\nMaking prediction for {symbol}...")
        
        # Get latest data
        df, success = self.prepare_training_data(symbol, period='3mo')
        if not success or df.empty:
            return {
                'symbol': symbol,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'signal': 'HOLD',
                'error': 'Insufficient data'
            }
        
        # Use the latest available data point
        latest_data = df[self.feature_columns].iloc[-1:].copy()
        
        # Scale features
        latest_scaled = self.scaler.transform(latest_data)
        
        # Make prediction
        prediction = self.model.predict(latest_scaled)[0]
        prediction_proba = self.model.predict_proba(latest_scaled)[0]
        
        # Get current price
        current_price = self.data_collector.get_latest_price(symbol)
        
        # Determine signal
        if prediction == 1:
            signal = 'BUY'
            confidence = prediction_proba[1]
        else:
            signal = 'HOLD/SELL'
            confidence = prediction_proba[0]
        
        # Get company info
        info = self.data_collector.get_stock_info(symbol)
        company_name = info.get('company_name', 'N/A')
        
        return {
            'symbol': symbol,
            'company_name': company_name,
            'current_price': current_price,
            'prediction': int(prediction),
            'signal': signal,
            'confidence': float(confidence),
            'buy_probability': float(prediction_proba[1]),
            'hold_probability': float(prediction_proba[0]),
        }
    
    def predict_multiple(self, symbols: Optional[list] = None) -> pd.DataFrame:
        """
        Make predictions for multiple stocks
        
        Args:
            symbols: List of stock symbols (default: all large cap stocks)
            
        Returns:
            DataFrame with predictions for all stocks
        """
        if symbols is None:
            symbols = self.data_collector.LARGE_CAP_STOCKS
        
        results = []
        for symbol in symbols:
            try:
                result = self.predict(symbol)
                results.append(result)
                print(f"✓ {symbol}: {result['signal']} (confidence: {result['confidence']:.2%})")
            except Exception as e:
                print(f"✗ {symbol}: Error - {e}")
        
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    def save_model(self, filepath: str = 'stock_predictor_model.pkl'):
        """Save the trained model"""
        if not self.model_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'stock_predictor_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    predictor = StockPredictor()
    
    # Train model on a few stocks
    print("=" * 60)
    print("TRAINING STOCK PREDICTION MODEL")
    print("=" * 60)
    
    training_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    predictor.train_model(symbols=training_stocks, period='2y')
    
    # Save model
    predictor.save_model('stock_predictor_model.pkl')
    
    # Make predictions
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    
    test_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
    predictions_df = predictor.predict_multiple(test_stocks)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(predictions_df.to_string())
    
    # Show buy recommendations
    buy_signals = predictions_df[predictions_df['signal'] == 'BUY'].sort_values('confidence', ascending=False)
    if not buy_signals.empty:
        print("\n" + "=" * 60)
        print("TOP BUY RECOMMENDATIONS")
        print("=" * 60)
        print(buy_signals[['symbol', 'company_name', 'current_price', 'confidence']].to_string())
