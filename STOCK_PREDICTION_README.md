# Stock Prediction System for Indian Large Cap Stocks

This module provides a comprehensive stock prediction system for Indian Large Cap stocks listed on NSE (National Stock Exchange). It combines technical analysis, fundamental analysis, and news sentiment analysis to generate buy/sell signals.

## 🎯 Features

- **Multi-Source Data Collection**: Fetches real-time stock prices, financial news, and fundamental metrics
- **Technical Analysis**: 30+ technical indicators including:
  - Moving Averages (SMA, EMA)
  - MACD, RSI, Bollinger Bands
  - Stochastic Oscillator, ATR
  - Volume and momentum indicators
- **Sentiment Analysis**: Analyzes financial news to gauge market sentiment
- **Fundamental Analysis**: Incorporates P/E ratio, revenue growth, profit margins, ROE, and more
- **Machine Learning Model**: Ensemble model (Random Forest + Gradient Boosting) for predictions
- **Large Cap Coverage**: Pre-configured with 20+ top Indian large cap stocks

## 📊 Covered Stocks

The system includes major Indian Large Cap stocks such as:
- Reliance Industries (RELIANCE.NS)
- Tata Consultancy Services (TCS.NS)
- HDFC Bank (HDFCBANK.NS)
- Infosys (INFY.NS)
- ICICI Bank (ICICIBANK.NS)
- And 15+ more major stocks

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required NLP data for sentiment analysis:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
python -m textblob.download_corpora
```

### Training the Model

Train the model on historical data:

```bash
# Train on default large cap stocks (first 10)
python stock_predict.py train

# Train on specific stocks
python stock_predict.py train --symbols "RELIANCE,TCS,INFY,HDFCBANK"

# Train with custom time period
python stock_predict.py train --period 3y --symbols "RELIANCE,TCS"
```

### Making Predictions

Generate buy/sell signals:

```bash
# Predict all large cap stocks
python stock_predict.py predict

# Predict specific stocks
python stock_predict.py predict --symbols "RELIANCE,TCS,INFY"

# Save predictions to CSV
python stock_predict.py predict --symbols "RELIANCE,TCS" --output predictions.csv
```

### Listing Available Stocks

```bash
python stock_predict.py list
```

## 📖 Usage Examples

### Example 1: Quick Prediction

```bash
python stock_predict.py train --symbols "RELIANCE,TCS,INFY"
python stock_predict.py predict --symbols "RELIANCE"
```

### Example 2: Comprehensive Analysis

```python
from stock_predictor import StockPredictor

# Initialize predictor
predictor = StockPredictor()

# Train model
predictor.train_model(symbols=['RELIANCE.NS', 'TCS.NS', 'INFY.NS'], period='2y')

# Save model
predictor.save_model('my_model.pkl')

# Make prediction
result = predictor.predict('RELIANCE.NS')
print(f"Signal: {result['signal']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Current Price: ₹{result['current_price']:.2f}")
```

### Example 3: Custom Data Analysis

```python
from stock_predictor import DataCollector, FeatureEngine

# Collect data
collector = DataCollector()
stock_data = collector.get_stock_data('RELIANCE.NS', period='1y')

# Add technical indicators
engine = FeatureEngine()
stock_data = engine.add_technical_indicators(stock_data)

# Create target variable
stock_data = engine.create_target_variable(stock_data, days_ahead=5)

print(stock_data[['Close', 'RSI', 'MACD', 'Target']].tail())
```

## 🔧 Module Structure

```
stock_predictor/
├── __init__.py          # Module initialization
├── data_collector.py    # Data collection from Yahoo Finance and news sources
├── feature_engine.py    # Feature engineering and technical indicators
└── predictor.py         # Main prediction model and training logic
```

## 📊 Model Details

### Input Features

1. **Technical Indicators** (30+ features):
   - Moving Averages (5, 10, 20, 50, 200 days)
   - MACD and Signal Line
   - RSI (Relative Strength Index)
   - Bollinger Bands
   - Stochastic Oscillator
   - ATR (Average True Range)
   - Volume indicators
   - Momentum indicators

2. **Fundamental Metrics**:
   - P/E Ratio, Forward P/E
   - PEG Ratio, Price-to-Book
   - Dividend Yield, Beta
   - Revenue Growth, Earnings Growth
   - Profit Margin, ROE

3. **Sentiment Features**:
   - News sentiment score
   - Positive/negative article counts
   - Sentiment polarity and subjectivity

### Model Architecture

- **Algorithm**: Ensemble Voting Classifier
  - Random Forest Classifier (100 estimators)
  - Gradient Boosting Classifier (100 estimators)
- **Training**: Historical data from multiple stocks
- **Prediction**: Binary classification (Buy vs. Hold/Sell)
- **Confidence**: Probability scores from soft voting

### Performance Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Cross-validation (5-fold)

## ⚙️ Configuration

### Stock Symbols

To modify the list of large cap stocks, edit `data_collector.py`:

```python
LARGE_CAP_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS',
    # Add more stocks here
]
```

### Prediction Parameters

Customize prediction parameters in `predictor.py` or feature engine:

```python
# Change prediction horizon (days ahead)
df = engine.create_target_variable(df, days_ahead=5, threshold=0.02)

# Adjust price change threshold for buy/sell signals
# threshold=0.02 means 2% price change
```

## 📈 Interpretation Guide

### Signal Types

- **BUY**: Model predicts price will increase by >2% in next 5 days
- **HOLD/SELL**: Model predicts price will not increase significantly

### Confidence Scores

- **>80%**: High confidence prediction
- **60-80%**: Moderate confidence
- **<60%**: Low confidence (consider with caution)

### Best Practices

1. **Don't rely solely on predictions**: Use as one input in your analysis
2. **Check multiple timeframes**: Train on different periods for validation
3. **Monitor confidence scores**: Higher confidence = more reliable signals
4. **Combine with other analysis**: Use alongside your own research
5. **Update model regularly**: Retrain with fresh data periodically

## 🛡️ Disclaimer

**IMPORTANT**: This is a predictive model for educational and research purposes only. 

- **Not Financial Advice**: Do not use this as the sole basis for investment decisions
- **No Guarantees**: Past performance does not guarantee future results
- **Market Risk**: Stock markets are inherently unpredictable and risky
- **Do Your Research**: Always conduct thorough research and consult financial advisors
- **Use Responsibly**: The developers are not responsible for any financial losses

## 🔍 Technical Details

### Data Sources

- **Stock Prices**: Yahoo Finance (yfinance library)
- **Financial News**: Yahoo Finance news feed
- **Company Info**: Yahoo Finance fundamentals API

### Dependencies

- `yfinance`: Stock price data
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning models
- `textblob`: Sentiment analysis
- `beautifulsoup4`: Web scraping (if needed)

### System Requirements

- Python 3.8+
- Internet connection (for data fetching)
- ~500MB disk space (for model and data)
- 4GB+ RAM recommended

## 🤝 Contributing

To add new features or improve the model:

1. Add new technical indicators in `feature_engine.py`
2. Implement additional data sources in `data_collector.py`
3. Experiment with different ML models in `predictor.py`
4. Test thoroughly before deployment

## 📝 Example Output

```
PREDICTION RESULTS
======================================================================

RELIANCE.NS - Reliance Industries Ltd
  Current Price: ₹2,456.80
  Signal: BUY
  Buy Probability: 78.50%
  Confidence: 78.50%

TCS.NS - Tata Consultancy Services Ltd
  Current Price: ₹3,789.25
  Signal: HOLD/SELL
  Buy Probability: 42.30%
  Confidence: 57.70%

TOP BUY RECOMMENDATIONS
======================================================================

1. RELIANCE.NS - Reliance Industries Ltd
   Price: ₹2,456.80
   Buy Probability: 78.50%
   Confidence: 78.50%
```

## 📚 Further Reading

- [Technical Analysis Basics](https://www.investopedia.com/technical-analysis-4689657)
- [Fundamental Analysis Guide](https://www.investopedia.com/fundamental-analysis-4689757)
- [Machine Learning for Trading](https://www.quantinsti.com/blog/machine-learning-trading)
- [NSE India](https://www.nseindia.com/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
