#!/usr/bin/env python3
"""
Example script to run the ChatGPT Trading Strategy

Before running this script:
1. Install the OpenAI Python package:
   pip install openai

2. Set your OpenAI API key as an environment variable:
   export OPENAI_API_KEY='your-api-key-here'
   
   Or set it in your shell profile for persistence:
   echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
   source ~/.bashrc

3. Run the script:
   python run_chatgpt_strategy.py
"""

import os
from chatgpt_strategy import ChatGPTStrategy
from backtesting import Backtest
from data_loader import load_yahoo_finance_data

def main():
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OpenAI API key not found!")
        print("\nPlease set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nYou can get an API key from: https://platform.openai.com/api-keys")
        return

    print("✅ OpenAI API key found!")
    
    # Configuration - Test one asset at a time to avoid rate limits
    symbol = "BTC-USD"
    name = "Bitcoin"
    start_date = "2024-01-01"
    end_date = "2025-05-28"  # Shorter period to reduce API calls
    
    print(f"\n{'='*60}")
    print(f"🔍 TESTING: {name} ({symbol})")
    print(f"{'='*60}")
    print(f"📊 Loading data for {symbol}...")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        # Load data
        data = load_yahoo_finance_data(symbol, start_date, end_date)
        print(f"✅ Loaded {len(data)} bars of data")
        
        # Check average price to determine appropriate cash amount
        avg_price = data.Close.mean()
        print(f"💰 Average {symbol} price: ${avg_price:,.2f}")
        
        # Set cash based on asset price
        if avg_price > 1000:
            initial_cash = 1000000
            print(f"🏦 Using ${initial_cash:,} initial cash for high-priced asset")
        else:
            initial_cash = 100000
            print(f"🏦 Using ${initial_cash:,} initial cash")
        
        # Create backtest
        bt = Backtest(
            data,
            ChatGPTStrategy,
            cash=initial_cash,
            commission=.002,
            exclusive_orders=True,
            trade_on_close=True,
        )
        
        print(f"\n🤖 Starting backtest for {name}...")
        print("⚠️  Note: This will make API calls to OpenAI (costs money)")
        print("💰 Each call costs approximately $0.01-0.02")
        print("⏱️  This may take several minutes due to API rate limiting...")
        
        # Run the backtest
        output = bt.run()
        
        print(f"\n📈 RESULTS FOR {name}:")
        print(f"Strategy Return: {output['Return [%]']:.2f}%")
        print(f"Buy & Hold Return: {output['Buy & Hold Return [%]']:.2f}%")
        print(f"Outperformance: {output['Return [%]'] - output['Buy & Hold Return [%]']:+.2f}%")
        print(f"Number of Trades: {output['# Trades']}")
        print(f"Win Rate: {output['Win Rate [%]']:.1f}%")
        print(f"Sharpe Ratio: {output['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {output['Max. Drawdown [%]']:.2f}%")
        print(f"Exposure Time: {output['Exposure Time [%]']:.1f}%")
        
        print("\n" + "="*50)
        print("📈 DETAILED BACKTEST RESULTS")
        print("="*50)
        print(output)
        
        # Plot results
        print("\n📊 Generating plot...")
        bt.plot(
            plot_volume=False,
            plot_pl=True,
            plot_drawdown=True,
            plot_equity=True,
            plot_return=False,
        )
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Analyze the trades and ChatGPT reasoning")
        print("2. Adjust confidence thresholds if needed")
        print("3. Test on different time periods")
        print("4. Try different assets (ETH, AAPL, etc.)")
        
    except Exception as e:
        print(f"❌ Error testing {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPossible solutions:")
        print("- Check your internet connection")
        print("- Verify your OpenAI API key is valid")
        print("- Ensure you have sufficient OpenAI credits")
        print("- Try reducing the date range to make fewer API calls")

if __name__ == "__main__":
    main() 