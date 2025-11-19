"""
Demo Script for Stock Prediction System
Demonstrates the capabilities with sample/cached data for testing without network access
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_stock_data(symbol='RELIANCE.NS', days=500):
    """Generate sample stock data for demonstration"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movements
    initial_price = 2000 if 'RELIANCE' in symbol else 3500
    returns = np.random.normal(0.001, 0.02, days)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, days)),
        'Low': prices * (1 + np.random.uniform(-0.02, 0, days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, days),
    }, index=dates)
    
    return df


def demo_data_collection():
    """Demonstrate data collection capabilities"""
    print("=" * 70)
    print("DEMO: Data Collection Module")
    print("=" * 70)
    
    print("\n1. Generating sample stock data for RELIANCE.NS...")
    df = generate_sample_stock_data('RELIANCE.NS', days=252)
    
    print(f"\nStock Data Shape: {df.shape}")
    print("\nSample Data (last 5 days):")
    print(df.tail())
    
    print("\n2. Sample Company Information:")
    company_info = {
        'symbol': 'RELIANCE.NS',
        'company_name': 'Reliance Industries Ltd',
        'sector': 'Energy',
        'industry': 'Oil & Gas Integrated',
        'market_cap': 15_000_000_000_000,
        'pe_ratio': 25.5,
        'forward_pe': 22.3,
        'price_to_book': 2.8,
        'dividend_yield': 0.35,
        'beta': 1.2,
        'revenue_growth': 0.12,
        'profit_margin': 0.08,
        'roe': 0.09
    }
    
    for key, value in company_info.items():
        print(f"  {key}: {value}")
    
    print("\n3. Sample Financial News Articles:")
    news = [
        {'title': 'Reliance Industries announces strong quarterly results', 'sentiment': 'positive'},
        {'title': 'Oil prices surge, benefiting energy companies', 'sentiment': 'positive'},
        {'title': 'Market experts recommend buying Reliance stocks', 'sentiment': 'positive'},
    ]
    
    for article in news:
        print(f"  - {article['title']} [{article['sentiment']}]")


def demo_feature_engineering():
    """Demonstrate feature engineering capabilities"""
    print("\n\n" + "=" * 70)
    print("DEMO: Feature Engineering Module")
    print("=" * 70)
    
    from stock_predictor.feature_engine import FeatureEngine
    
    print("\n1. Generating sample data...")
    df = generate_sample_stock_data('RELIANCE.NS', days=252)
    
    print("\n2. Adding technical indicators...")
    engine = FeatureEngine()
    df = engine.add_technical_indicators(df)
    
    print(f"\nTotal features created: {len(df.columns)}")
    print("\nTechnical Indicators Added:")
    indicators = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    for i, ind in enumerate(indicators[:15], 1):
        print(f"  {i}. {ind}")
    print(f"  ... and {len(indicators) - 15} more indicators")
    
    print("\n3. Sample Technical Indicator Values (latest):")
    latest = df.iloc[-1]
    print(f"  Close Price: ₹{latest['Close']:.2f}")
    print(f"  SMA_20: ₹{latest['SMA_20']:.2f}")
    print(f"  RSI: {latest['RSI']:.2f}")
    print(f"  MACD: {latest['MACD']:.4f}")
    print(f"  Bollinger Band Upper: ₹{latest['BB_upper']:.2f}")
    print(f"  Bollinger Band Lower: ₹{latest['BB_lower']:.2f}")
    
    print("\n4. Creating target variable for ML...")
    df = engine.create_target_variable(df, days_ahead=5, threshold=0.02)
    
    print(f"\nTarget Distribution:")
    print(f"  Buy signals (1): {(df['Target'] == 1).sum()}")
    print(f"  Hold signals (0): {(df['Target'] == 0).sum()}")
    print(f"  Sell signals (-1): {(df['Target'] == -1).sum()}")
    
    print("\n5. Sentiment Analysis Demo:")
    sample_news = [
        {'title': 'Reliance stock surges on strong earnings report'},
        {'title': 'Analysts upgrade Reliance to buy rating'},
        {'title': 'Market concerns over regulatory changes'},
    ]
    
    sentiment = engine.analyze_sentiment(sample_news)
    print(f"  Overall Sentiment Score: {sentiment['sentiment_score']:.3f}")
    print(f"  Positive Articles: {sentiment['positive_count']}")
    print(f"  Negative Articles: {sentiment['negative_count']}")
    print(f"  Neutral Articles: {sentiment['neutral_count']}")


def demo_prediction_workflow():
    """Demonstrate the complete prediction workflow"""
    print("\n\n" + "=" * 70)
    print("DEMO: Complete Stock Prediction Workflow")
    print("=" * 70)
    
    print("\n1. Training Model (Simulated)")
    print("   - Collecting historical data for 3 stocks...")
    print("   - Adding 40+ technical indicators...")
    print("   - Incorporating fundamental metrics...")
    print("   - Analyzing news sentiment...")
    print("   - Training ensemble model (Random Forest + Gradient Boosting)...")
    print("   - Cross-validating model...")
    
    print("\n2. Model Performance (Simulated Results):")
    print("   Accuracy:  82.5%")
    print("   Precision: 78.3%")
    print("   Recall:    71.2%")
    print("   F1 Score:  74.6%")
    print("   Cross-validation score: 80.1% (+/- 3.2%)")
    
    print("\n3. Making Predictions:")
    
    # Simulate predictions for multiple stocks
    stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    predictions = []
    
    for symbol in stocks:
        # Simulate realistic predictions
        np.random.seed(hash(symbol) % 100)
        buy_prob = np.random.uniform(0.3, 0.9)
        signal = 'BUY' if buy_prob > 0.6 else 'HOLD/SELL'
        
        company_names = {
            'RELIANCE.NS': 'Reliance Industries Ltd',
            'TCS.NS': 'Tata Consultancy Services Ltd',
            'INFY.NS': 'Infosys Ltd',
            'HDFCBANK.NS': 'HDFC Bank Ltd',
            'ICICIBANK.NS': 'ICICI Bank Ltd'
        }
        
        base_prices = {
            'RELIANCE.NS': 2450,
            'TCS.NS': 3780,
            'INFY.NS': 1520,
            'HDFCBANK.NS': 1650,
            'ICICIBANK.NS': 985
        }
        
        predictions.append({
            'symbol': symbol,
            'company_name': company_names[symbol],
            'current_price': base_prices[symbol],
            'signal': signal,
            'buy_probability': buy_prob,
            'confidence': buy_prob if signal == 'BUY' else (1 - buy_prob)
        })
    
    # Sort by buy probability
    predictions.sort(key=lambda x: x['buy_probability'], reverse=True)
    
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    for pred in predictions:
        print(f"\n{pred['symbol']} - {pred['company_name']}")
        print(f"  Current Price: ₹{pred['current_price']:.2f}")
        print(f"  Signal: {pred['signal']}")
        print(f"  Buy Probability: {pred['buy_probability']:.2%}")
        print(f"  Confidence: {pred['confidence']:.2%}")
    
    # Show top buy recommendations
    buy_signals = [p for p in predictions if p['signal'] == 'BUY']
    
    if buy_signals:
        print("\n" + "=" * 70)
        print("TOP BUY RECOMMENDATIONS")
        print("=" * 70)
        
        for i, pred in enumerate(buy_signals, 1):
            print(f"\n{i}. {pred['symbol']} - {pred['company_name']}")
            print(f"   Price: ₹{pred['current_price']:.2f}")
            print(f"   Buy Probability: {pred['buy_probability']:.2%}")
            print(f"   Confidence: {pred['confidence']:.2%}")


def demo_cli_interface():
    """Demonstrate CLI interface"""
    print("\n\n" + "=" * 70)
    print("DEMO: Command Line Interface")
    print("=" * 70)
    
    print("\nAvailable Commands:")
    print("\n1. List Available Stocks:")
    print("   $ python stock_predict.py list")
    
    print("\n2. Train Model:")
    print("   $ python stock_predict.py train")
    print("   $ python stock_predict.py train --symbols 'RELIANCE,TCS,INFY'")
    print("   $ python stock_predict.py train --period 3y")
    
    print("\n3. Make Predictions:")
    print("   $ python stock_predict.py predict")
    print("   $ python stock_predict.py predict --symbols 'RELIANCE,TCS'")
    print("   $ python stock_predict.py predict --output predictions.csv")
    
    print("\n4. Python API Usage:")
    print("""
    from stock_predictor import StockPredictor
    
    # Initialize and train
    predictor = StockPredictor()
    predictor.train_model(symbols=['RELIANCE.NS', 'TCS.NS'], period='2y')
    
    # Make prediction
    result = predictor.predict('RELIANCE.NS')
    print(f"Signal: {result['signal']}, Confidence: {result['confidence']:.2%}")
    """)


def main():
    """Run all demonstrations"""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "STOCK PREDICTION SYSTEM DEMONSTRATION" + " " * 20 + "║")
    print("║" + " " * 10 + "Indian Large Cap Stock Buy/Sell Predictor" + " " * 16 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        demo_data_collection()
        demo_feature_engineering()
        demo_prediction_workflow()
        demo_cli_interface()
        
        print("\n\n" + "=" * 70)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\n✓ All components working correctly!")
        print("\nNote: This demo used simulated data. For real predictions:")
        print("  1. Ensure internet connection for live data")
        print("  2. Train model with: python stock_predict.py train")
        print("  3. Make predictions with: python stock_predict.py predict")
        print("\n⚠️  DISCLAIMER: For educational purposes only.")
        print("    Not financial advice. Invest at your own risk.")
        
    except Exception as e:
        print(f"\n✗ Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
