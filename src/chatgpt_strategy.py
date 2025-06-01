from backtesting import Backtest, Strategy
from openai import OpenAI
import json
import os
import time
from data_loader import load_yahoo_finance_data

class ChatGPTStrategy(Strategy):
    # Parameters
    lookback_periods = 100  # Increased significantly for better context
    min_confidence_threshold = 8.0  # Much higher threshold - only take very confident trades
    max_position_size = 0.25  # Reduced from 0.5 to be more conservative
    
    def init(self):
        # Initialize tracking variables
        self.current_signal = 'hold'
        self.take_profit = None
        self.stop_loss = None
        self.justification = ""
        self.signal_executed = False  # Track if current signal has been executed
        self.next_call_criteria = None  # Store criteria for next ChatGPT call
        self.criteria_reference_price = None  # Reference price for price-based criteria
        self.criteria_reference_volume = None  # Reference volume for volume-based criteria
        self.criteria_reference_bar = None  # Reference bar for time-based criteria
        self.confidence = 5.0  # Default confidence
        self.dynamic_position_size = 0.3  # Default position size
        self.last_trade_bar = -20  # Track when we last traded (start with -20 to allow immediate trading)
        self.market_trend = 'sideways'  # Store ChatGPT's trend assessment
        
        # Set up OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            print("Warning: OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = None
    
    def get_market_data_summary(self):
        """Prepare recent market data for ChatGPT analysis"""
        # Get recent OHLCV data
        current_idx = len(self.data) - 1
        start_idx = max(0, current_idx - self.lookback_periods + 1)
        
        ohlcv_data = []
        for i in range(start_idx, current_idx + 1):
            bar_data = {
                "timestamp": str(self.data.index[i]) if hasattr(self.data.index[i], '__str__') else f"Bar_{i}",
                "open": float(self.data.Open[i]),
                "high": float(self.data.High[i]),
                "low": float(self.data.Low[i]),
                "close": float(self.data.Close[i]),
                "volume": float(self.data.Volume[i]) if hasattr(self.data, 'Volume') else 0
            }
            ohlcv_data.append(bar_data)
        
        return ohlcv_data
    
    def should_call_chatgpt(self):
        """Evaluate if we should call ChatGPT based on current criteria"""
        if self.next_call_criteria is None:
            return True  # First call or no criteria set
        
        current_price = self.data.Close[-1]
        current_volume = self.data.Volume[-1] if hasattr(self.data, 'Volume') else 0
        current_bar = len(self.data) - 1
        
        criteria = self.next_call_criteria
        
        try:
            if criteria.get('type') == 'time_based':
                bars_since_call = current_bar - (self.criteria_reference_bar or 0)
                target_bars = criteria.get('bars', 1)
                if bars_since_call >= target_bars:
                    return True
            
            elif criteria.get('type') == 'volume_change':
                if self.criteria_reference_volume and current_volume > 0:
                    volume_change = (current_volume - self.criteria_reference_volume) / self.criteria_reference_volume * 100
                    target_change = criteria.get('percent_increase', 50)
                    if volume_change >= target_change:
                        return True
            
            elif criteria.get('type') == 'immediate':
                return True
            
        except Exception as e:
            print(f"Error evaluating call criteria: {e}")
            # Simple fallback - check next bar
            return True
        
        return False

    def call_chatgpt_for_signal(self, market_data):
        """Call ChatGPT API to get trading signal"""
        if not self.client:
            return {
                "signal": "hold",
                "take_profit": None,
                "stop_loss": None,
                "justification": "OpenAI client not initialized",
                "next_call_criteria": {"type": "immediate", "description": "Default: monitor market"},
                "confidence": 0.0,
                "position_size": 0.1
            }
            
        try:
            current_price = market_data[-1]['close']
            
            # Calculate basic price changes for context
            prices = [bar['close'] for bar in market_data]
            price_change_24h = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) > 1 else 0
            price_change_7d = (prices[-1] - prices[-8]) / prices[-8] * 100 if len(prices) > 7 else 0
            
            # Calculate volatility
            if len(prices) > 10:
                recent_returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, min(len(prices), 11))]
                volatility = (sum([r**2 for r in recent_returns]) / len(recent_returns)) ** 0.5 * 100
            else:
                volatility = 0
            
            prompt = f"""
            You are a PROFESSIONAL crypto analyst focused on QUALITY over QUANTITY trades.
            
            MARKET CONTEXT:
            - Current Price: ${current_price:,.2f}
            - 24h Change: {price_change_24h:+.2f}%
            - 7d Change: {price_change_7d:+.2f}%
            - Volatility: {volatility:.2f}%
            
            COMPLETE OHLCV DATA (last {len(market_data)} bars for trend analysis):
            {json.dumps(market_data, indent=2)}
            
            TRADING RULES:
            1. ANALYZE THE TREND YOURSELF from the OHLCV data provided
            2. QUALITY OVER QUANTITY - Only trade with very high confidence (8+ out of 10)
            3. FOLLOW MAJOR TRENDS - Don't fight strong momentum
            4. BE PATIENT - Wait for clear setups
            5. RISK MANAGEMENT - Conservative position sizing
            6. HOLD WINNERS - Let profitable trades run
            
            ONLY TRADE IF:
            - Confidence is 8.0 or higher
            - You can clearly identify trend direction from the data
            - Good risk/reward ratio (at least 2:1)
            - Strong momentum confirmation
            
            RESPOND WITH EXACTLY THIS JSON FORMAT:
            {{
                "signal": "buy",
                "confidence": 9.0,
                "position_size": 0.20,
                "take_profit": 48000,
                "stop_loss": 43000,
                "justification": "Clear bullish trend visible in OHLCV data with momentum confirmation. Breakout above resistance.",
                "market_trend": "bullish",
                "next_call_criteria": {{
                    "type": "immediate",
                    "description": "Monitor closely during trend"
                }}
            }}
            
            CONFIDENCE SCORING (BE STRICT):
            - 9-10: Extremely strong signal with multiple confirmations
            - 8-9: Strong signal with clear trend and momentum
            - 7-8: Good signal but some uncertainty
            - Below 7: DO NOT TRADE - use "hold" signal
            
            POSITION SIZING (CONSERVATIVE):
            - Very high confidence (9+): 0.20-0.25 (20-25%)
            - High confidence (8-9): 0.15-0.20 (15-20%)
            - Medium confidence (7-8): 0.10-0.15 (10-15%)
            - Low confidence (<7): 0.05 or "hold"
            
            NEXT CALL CRITERIA - CHOOSE BASED ON YOUR TREND ANALYSIS:
            - TRENDING MARKETS (clear bullish/bearish): Use {{"type": "immediate", "description": "Monitor closely during trend"}}
            - SIDEWAYS/CHOPPY MARKETS: Use {{"type": "time_based", "bars": 10, "description": "Wait for breakout"}}
            
            MARKET_TREND: Include your trend assessment as "bullish", "bearish", or "sideways"
            
            Remember: It's better to miss opportunities than to lose money on bad trades.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a conservative professional trader. Analyze trends carefully from OHLCV data. Quality over quantity. Only recommend trades with very high confidence."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1  # Very low temperature for consistency
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Remove markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    response_text = response_text.replace('```', '').strip()
                
                signal_data = json.loads(response_text)
                
                # Ensure required fields exist with proper types
                signal_data.setdefault('confidence', 5.0)
                signal_data.setdefault('position_size', 0.1)
                signal_data.setdefault('market_trend', 'sideways')
                
                # Cap position size for safety
                signal_data['position_size'] = min(signal_data['position_size'], self.max_position_size)
                
                # Fix next_call_criteria if it's not a dict
                market_trend = signal_data.get('market_trend', 'sideways')
                if 'next_call_criteria' not in signal_data:
                    # Use ChatGPT's trend assessment for default
                    if market_trend == 'sideways':
                        signal_data['next_call_criteria'] = {"type": "time_based", "bars": 10, "description": "Default: wait in sideways market"}
                    else:
                        signal_data['next_call_criteria'] = {"type": "immediate", "description": "Default: monitor trending market"}
                elif isinstance(signal_data['next_call_criteria'], str):
                    description = signal_data['next_call_criteria']
                    # Use ChatGPT's trend assessment for conversion
                    if market_trend == 'sideways':
                        signal_data['next_call_criteria'] = {
                            "type": "time_based", 
                            "bars": 10, 
                            "description": description
                        }
                    else:
                        signal_data['next_call_criteria'] = {
                            "type": "immediate", 
                            "description": description
                        }
                elif not isinstance(signal_data['next_call_criteria'], dict):
                    # Use ChatGPT's trend assessment for default
                    if market_trend == 'sideways':
                        signal_data['next_call_criteria'] = {"type": "time_based", "bars": 10, "description": "Default: wait in sideways market"}
                    else:
                        signal_data['next_call_criteria'] = {"type": "immediate", "description": "Default: monitor trending market"}
                
                return signal_data
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed. Error: {e}")
                print(f"Raw response: {response_text[:500]}...")
                
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "position_size": 0.1,
                    "take_profit": None,
                    "stop_loss": None,
                    "justification": "Failed to parse ChatGPT response",
                    "next_call_criteria": {"type": "immediate", "description": "Parsing error: monitor market"}
                }
                
        except Exception as e:
            print(f"Error calling ChatGPT API: {str(e)}")
            
            return {
                "signal": "hold",
                "confidence": 0.0,
                "position_size": 0.1,
                "take_profit": None,
                "stop_loss": None,
                "justification": f"API Error: {str(e)}",
                "next_call_criteria": {"type": "immediate", "description": "API error: monitor market"}
            }

    def next(self):
        current_bar = len(self.data) - 1
        new_signal_received = False
        
        # Check if we should call ChatGPT based on current criteria
        if self.should_call_chatgpt():
            # Get market data and call ChatGPT
            market_data = self.get_market_data_summary()
            signal_data = self.call_chatgpt_for_signal(market_data)
            
            # Update strategy state
            old_signal = self.current_signal
            self.current_signal = signal_data.get('signal', 'hold')
            self.take_profit = signal_data.get('take_profit')
            self.stop_loss = signal_data.get('stop_loss')
            self.justification = signal_data.get('justification', '')
            self.confidence = signal_data.get('confidence', 5.0)
            self.dynamic_position_size = signal_data.get('position_size', 0.3)
            self.market_trend = signal_data.get('market_trend', 'sideways')  # Store ChatGPT's trend assessment
            
            # Update criteria and reference values
            self.next_call_criteria = signal_data.get('next_call_criteria')
            self.criteria_reference_price = self.data.Close[-1]
            self.criteria_reference_volume = self.data.Volume[-1] if hasattr(self.data, 'Volume') else 0
            self.criteria_reference_bar = current_bar
            
            # Check if we got a new signal
            if old_signal != self.current_signal:
                self.signal_executed = False
                new_signal_received = True
            
            # Print the signal and justification
            current_price = self.data.Close[-1]
            print(f"\n=== ChatGPT Trading Signal ===")
            print(f"Bar: {current_bar}, Price: ${current_price:.2f}")
            print(f"Signal: {self.current_signal.upper()}")
            print(f"Confidence: {self.confidence:.1f}/10")
            print(f"Position Size: {self.dynamic_position_size:.1%}")
            print(f"Market Trend: {self.market_trend.upper()}")
            if self.take_profit:
                print(f"Take Profit: ${self.take_profit:.2f}")
            if self.stop_loss:
                print(f"Stop Loss: ${self.stop_loss:.2f}")
            print(f"Justification: {self.justification}")
            print(f"Next Call: {self.next_call_criteria.get('description', 'No description')}")
            print("=" * 30)
        
        # Execute trades based on current signal
        current_price = self.data.Close[-1]
        
        # Close existing positions if stop loss or take profit hit
        if self.position.is_long:
            if self.stop_loss and current_price <= self.stop_loss:
                print(f"Stop loss hit at ${current_price:.2f}")
                self.position.close()
                self.signal_executed = True  # Reset signal after stop loss
                self.next_call_criteria = {"type": "immediate", "description": "Reassess after stop loss"}
            elif self.take_profit and current_price >= self.take_profit:
                print(f"Take profit hit at ${current_price:.2f}")
                self.position.close()
                self.signal_executed = True  # Reset signal after take profit
                self.next_call_criteria = {"type": "immediate", "description": "Reassess after take profit"}
        elif self.position.is_short:
            if self.stop_loss and current_price >= self.stop_loss:
                print(f"Stop loss hit at ${current_price:.2f}")
                self.position.close()
                self.signal_executed = True  # Reset signal after stop loss
                self.next_call_criteria = {"type": "immediate", "description": "Reassess after stop loss"}
            elif self.take_profit and current_price <= self.take_profit:
                print(f"Take profit hit at ${current_price:.2f}")
                self.position.close()
                self.signal_executed = True  # Reset signal after take profit
                self.next_call_criteria = {"type": "immediate", "description": "Reassess after take profit"}
        
        # Execute new signals only if we haven't already executed this signal
        if not self.signal_executed and new_signal_received:
            current_bar = len(self.data) - 1
            
            # Implement minimum time between trades (prevent overtrading)
            bars_since_last_trade = current_bar - self.last_trade_bar
            min_bars_between_trades = 10  # At least 10 bars between trades
            
            # Only trade if confidence is above high threshold
            confidence_threshold = self.min_confidence_threshold  # 8.0
            
            if self.current_signal == 'buy' and not self.position.is_long and self.confidence >= confidence_threshold:
                # Additional filters for buy signals
                can_trade = (
                    bars_since_last_trade >= min_bars_between_trades and
                    self.market_trend in ['bullish', 'sideways']  # Don't buy in bearish trend (ChatGPT's assessment)
                )
                
                if can_trade:
                    if self.position.is_short:
                        self.position.close()
                    
                    # Use conservative position sizing
                    position_fraction = min(self.dynamic_position_size, self.max_position_size)
                    
                    print(f"Executing BUY at ${current_price:.2f}")
                    print(f"Position size: {position_fraction:.1%} of equity (Confidence: {self.confidence:.1f}/10)")
                    print(f"Market Trend: {self.market_trend.upper()} (ChatGPT assessment)")
                    
                    self.buy(size=position_fraction)
                    self.signal_executed = True
                    self.last_trade_bar = current_bar
                else:
                    print(f"BUY signal filtered out:")
                    print(f"- Bars since last trade: {bars_since_last_trade} (need {min_bars_between_trades})")
                    print(f"- Market Trend: {self.market_trend.upper()} (ChatGPT assessment)")
                    self.signal_executed = True
                
            elif self.current_signal == 'sell' and not self.position.is_short and self.confidence >= confidence_threshold:
                # Additional filters for sell signals
                can_trade = (
                    bars_since_last_trade >= min_bars_between_trades and
                    self.market_trend in ['bearish', 'sideways']  # Don't sell in bullish trend (ChatGPT's assessment)
                )
                
                if can_trade:
                    if self.position.is_long:
                        self.position.close()
                    
                    # Use conservative position sizing
                    position_fraction = min(self.dynamic_position_size, self.max_position_size)
                    
                    print(f"Executing SELL at ${current_price:.2f}")
                    print(f"Position size: {position_fraction:.1%} of equity (Confidence: {self.confidence:.1f}/10)")
                    print(f"Market Trend: {self.market_trend.upper()} (ChatGPT assessment)")
                    
                    self.sell(size=position_fraction)
                    self.signal_executed = True
                    self.last_trade_bar = current_bar
                else:
                    print(f"SELL signal filtered out:")
                    print(f"- Bars since last trade: {bars_since_last_trade} (need {min_bars_between_trades})")
                    print(f"- Market Trend: {self.market_trend.upper()} (ChatGPT assessment)")
                    self.signal_executed = True
            
            elif self.current_signal == 'hold' or self.confidence < confidence_threshold:
                if self.position:
                    print(f"Executing HOLD - Closing position at ${current_price:.2f}")
                    if self.confidence < confidence_threshold:
                        print(f"Reason: Low confidence ({self.confidence:.1f}/10, need {confidence_threshold})")
                    self.position.close()
                    self.last_trade_bar = current_bar
                elif self.confidence < confidence_threshold:
                    print(f"No trade executed - Low confidence ({self.confidence:.1f}/10, need {confidence_threshold})")
                self.signal_executed = True

if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Load data using the data_loader module
    symbol = "AAPL"  # Apple Inc.
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    data = load_yahoo_finance_data(symbol, start_date, end_date)
    
    print(f"Loaded {len(data)} bars of data for {symbol}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    bt = Backtest(
        data,
        ChatGPTStrategy,
        cash=100000,
        commission=.002,
        exclusive_orders=True,
    )
    
    print("\nStarting backtest with ChatGPT strategy...")
    output = bt.run()
    
    print("\n=== Backtest Results ===")
    print(output)
    
    # Plot results
    bt.plot(
        plot_volume=False,
        plot_pl=True,
        plot_drawdown=False,
        plot_equity=True,
        plot_return=False,
    ) 