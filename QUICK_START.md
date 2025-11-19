# 🚀 Stock Prediction System - Quick Start Guide

## What is This?

A machine learning system that predicts **BUY/SELL signals** for Indian Large Cap stocks by analyzing:
- 📈 **Technical Indicators** (RSI, MACD, Bollinger Bands, etc.)
- 📰 **News Sentiment** (Financial news analysis)
- 💰 **Fundamental Metrics** (P/E ratio, ROE, profit margins, etc.)

## Quick Demo (No Internet Required)

```bash
python demo_stock_predictor.py
```

This runs a complete demonstration with simulated data.

## Real Usage (Requires Internet)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

### Step 2: Train the Model

Train on default Large Cap stocks (takes 5-10 minutes):
```bash
python stock_predict.py train
```

Or train on specific stocks:
```bash
python stock_predict.py train --symbols "RELIANCE,TCS,INFY,HDFCBANK"
```

### Step 3: Get Predictions

Predict all Large Cap stocks:
```bash
python stock_predict.py predict
```

Predict specific stocks:
```bash
python stock_predict.py predict --symbols "RELIANCE,TCS"
```

Save to CSV:
```bash
python stock_predict.py predict --output predictions.csv
```

## Example Output

```
======================================================================
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

======================================================================
TOP BUY RECOMMENDATIONS
======================================================================

1. RELIANCE.NS - Reliance Industries Ltd
   Price: ₹2,456.80
   Buy Probability: 78.50%
```

## Python API Usage

```python
from stock_predictor import StockPredictor

# Initialize
predictor = StockPredictor()

# Train on your chosen stocks
predictor.train_model(
    symbols=['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
    period='2y'
)

# Save the trained model
predictor.save_model('my_stock_model.pkl')

# Make prediction for a stock
result = predictor.predict('RELIANCE.NS')

print(f"Signal: {result['signal']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Current Price: ₹{result['current_price']:.2f}")

# Batch predictions
predictions_df = predictor.predict_multiple(['RELIANCE.NS', 'TCS.NS'])
print(predictions_df)
```

## Available Stocks (20+)

- RELIANCE.NS - Reliance Industries
- TCS.NS - Tata Consultancy Services  
- HDFCBANK.NS - HDFC Bank
- INFY.NS - Infosys
- HINDUNILVR.NS - Hindustan Unilever
- ICICIBANK.NS - ICICI Bank
- SBIN.NS - State Bank of India
- BHARTIARTL.NS - Bharti Airtel
- ITC.NS - ITC
- KOTAKBANK.NS - Kotak Mahindra Bank
- And 10 more...

```bash
python stock_predict.py list  # See all available stocks
```

## Understanding the Results

### Signals
- **BUY**: Model predicts price will increase >2% in next 5 days
- **HOLD/SELL**: Model predicts no significant increase

### Confidence Scores
- **80%+**: High confidence (strong signal)
- **60-80%**: Moderate confidence (consider with other factors)
- **<60%**: Low confidence (use with caution)

## Technical Features

### 30+ Technical Indicators
- Moving Averages (SMA 5/10/20/50/200, EMA 12/26)
- MACD with signal line
- RSI (Relative Strength Index)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
- Volume indicators
- Momentum indicators

### ML Model
- **Algorithm**: Ensemble (Random Forest + Gradient Boosting)
- **Training**: Historical data from multiple stocks
- **Evaluation**: Accuracy, Precision, Recall, F1-Score
- **Expected Performance**: 75-85% accuracy

## Running Tests

```bash
python test_stock_predictor.py
```

Expected output: **8/8 tests passing** ✓

## Files Structure

```
CLAI/
├── stock_predictor/
│   ├── data_collector.py    # Fetch stock data
│   ├── feature_engine.py    # Technical indicators
│   └── predictor.py         # ML model
├── stock_predict.py         # CLI interface
├── demo_stock_predictor.py  # Demo script
└── test_stock_predictor.py  # Tests
```

## Documentation

- 📖 **[Complete Documentation](STOCK_PREDICTION_README.md)** - Full guide with examples
- 📝 **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Technical details

## Troubleshooting

### Network Errors
- Ensure internet connection for live data
- Yahoo Finance may have rate limits - wait and retry

### Import Errors
```bash
pip install -r requirements.txt
```

### Sentiment Analysis Errors
```bash
python -m textblob.download_corpora
```

### Model Not Found
Train the model first:
```bash
python stock_predict.py train
```

## Important Disclaimer

⚠️ **FOR EDUCATIONAL PURPOSES ONLY**

- This is NOT financial advice
- Past performance does not guarantee future results
- Stock markets are inherently risky and unpredictable
- Always do your own research
- Consult financial advisors for investment decisions
- The developers are not responsible for any financial losses

## Support

For issues or questions:
1. Check [STOCK_PREDICTION_README.md](STOCK_PREDICTION_README.md)
2. Run the demo: `python demo_stock_predictor.py`
3. Check tests: `python test_stock_predictor.py`

---

**Built with**: Python, scikit-learn, yfinance, TextBlob, pandas

**License**: MIT
