"""
Stock Prediction CLI Interface
Command-line interface for predicting buy/sell signals for Indian Large Cap stocks
"""

import sys
import argparse
from stock_predictor import StockPredictor, DataCollector


def train_model(args):
    """Train the stock prediction model"""
    print("=" * 70)
    print("TRAINING STOCK PREDICTION MODEL FOR INDIAN LARGE CAP STOCKS")
    print("=" * 70)
    
    predictor = StockPredictor()
    
    # Use specific stocks or default large cap stocks
    if args.symbols:
        symbols = [s.strip().upper() + '.NS' if not s.endswith('.NS') else s.upper() 
                  for s in args.symbols.split(',')]
    else:
        # Default to first 10 large cap stocks for faster training
        symbols = predictor.data_collector.LARGE_CAP_STOCKS[:10]
    
    print(f"\nTraining on {len(symbols)} stocks: {', '.join(symbols)}")
    print(f"Historical period: {args.period}")
    
    try:
        predictor.train_model(symbols=symbols, period=args.period)
        
        # Save model
        model_file = args.output or 'stock_predictor_model.pkl'
        predictor.save_model(model_file)
        print(f"\n✓ Model successfully saved to {model_file}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        sys.exit(1)


def predict_stock(args):
    """Make prediction for specific stocks"""
    print("=" * 70)
    print("PREDICTING BUY/SELL SIGNALS FOR INDIAN STOCKS")
    print("=" * 70)
    
    predictor = StockPredictor()
    
    # Load trained model
    model_file = args.model or 'stock_predictor_model.pkl'
    try:
        predictor.load_model(model_file)
        print(f"✓ Model loaded from {model_file}\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Please train the model first using: python stock_predict.py train")
        sys.exit(1)
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() + '.NS' if not s.endswith('.NS') else s.upper() 
                  for s in args.symbols.split(',')]
    else:
        # Default to all large cap stocks
        symbols = predictor.data_collector.LARGE_CAP_STOCKS
    
    print(f"Analyzing {len(symbols)} stocks...\n")
    
    # Make predictions
    predictions_df = predictor.predict_multiple(symbols)
    
    if predictions_df.empty:
        print("No predictions could be made.")
        return
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    # Sort by confidence
    predictions_df = predictions_df.sort_values('buy_probability', ascending=False)
    
    for idx, row in predictions_df.iterrows():
        print(f"\n{row['symbol']} - {row['company_name']}")
        print(f"  Current Price: ₹{row['current_price']:.2f}")
        print(f"  Signal: {row['signal']}")
        print(f"  Buy Probability: {row['buy_probability']:.2%}")
        print(f"  Confidence: {row['confidence']:.2%}")
    
    # Show top buy recommendations
    buy_signals = predictions_df[predictions_df['signal'] == 'BUY']
    
    if not buy_signals.empty:
        print("\n" + "=" * 70)
        print("TOP BUY RECOMMENDATIONS")
        print("=" * 70)
        
        for idx, row in buy_signals.head(5).iterrows():
            print(f"\n{idx+1}. {row['symbol']} - {row['company_name']}")
            print(f"   Price: ₹{row['current_price']:.2f}")
            print(f"   Buy Probability: {row['buy_probability']:.2%}")
            print(f"   Confidence: {row['confidence']:.2%}")
    else:
        print("\n⚠ No strong buy signals detected at this time.")
    
    # Save results if requested
    if args.output:
        predictions_df.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to {args.output}")


def list_stocks(args):
    """List available large cap stocks"""
    print("=" * 70)
    print("AVAILABLE LARGE CAP INDIAN STOCKS")
    print("=" * 70)
    
    collector = DataCollector()
    
    for i, symbol in enumerate(collector.LARGE_CAP_STOCKS, 1):
        try:
            info = collector.get_stock_info(symbol)
            company_name = info.get('company_name', 'N/A')
            sector = info.get('sector', 'N/A')
            print(f"{i:2d}. {symbol:15s} - {company_name:40s} ({sector})")
        except:
            print(f"{i:2d}. {symbol:15s} - (Unable to fetch info)")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Stock Prediction System for Indian Large Cap Stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model on default stocks
  python stock_predict.py train
  
  # Train model on specific stocks
  python stock_predict.py train --symbols "RELIANCE,TCS,INFY,HDFCBANK"
  
  # Make predictions for all large cap stocks
  python stock_predict.py predict
  
  # Predict specific stocks
  python stock_predict.py predict --symbols "RELIANCE,TCS"
  
  # List available stocks
  python stock_predict.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the prediction model')
    train_parser.add_argument(
        '--symbols', '-s',
        help='Comma-separated list of stock symbols (e.g., RELIANCE,TCS,INFY)'
    )
    train_parser.add_argument(
        '--period', '-p',
        default='2y',
        help='Historical period to use (default: 2y)'
    )
    train_parser.add_argument(
        '--output', '-o',
        help='Output file for trained model (default: stock_predictor_model.pkl)'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make buy/sell predictions')
    predict_parser.add_argument(
        '--symbols', '-s',
        help='Comma-separated list of stock symbols (default: all large cap stocks)'
    )
    predict_parser.add_argument(
        '--model', '-m',
        help='Path to trained model file (default: stock_predictor_model.pkl)'
    )
    predict_parser.add_argument(
        '--output', '-o',
        help='Save predictions to CSV file'
    )
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available large cap stocks')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        predict_stock(args)
    elif args.command == 'list':
        list_stocks(args)


if __name__ == "__main__":
    main()
