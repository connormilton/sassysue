#!/usr/bin/env python3
"""
Self-Evolving LLM Forex Trading Bot for EUR/USD
Lightweight implementation with OANDA integration
"""

import os
import json
import logging
import time
import datetime
import pandas as pd
import openai
import requests
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_Trader")

# OANDA API credentials from environment
OANDA_API_TOKEN = os.getenv("OANDA_API_TOKEN")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_PRACTICE = os.getenv("OANDA_PRACTICE", "True").lower() in ["true", "1", "yes"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hard-coded parameters
MAX_DAILY_DRAWDOWN = 0.10  # 10% max daily drawdown
MAX_ACCOUNT_RISK = 0.10  # 10% max total account risk
INITIAL_SAFETY_LEVEL = 0.01  # Starting at 1% risk per trade
TARGET_DAILY_RETURN = 0.10  # 10% daily return target
INCREASE_FACTOR = 0.0075  # Increase safety level by 0.75% after profitable trade

# --- OANDA API Client ---
class OandaClient:
    """Simple OANDA API client for EUR/USD trading"""
    
    def __init__(self):
        """Initialize OANDA API client"""
        self.session = requests.Session()
        
        # Set base URL based on account type
        self.base_url = "https://api-fxpractice.oanda.com" if OANDA_PRACTICE else "https://api-fxtrade.oanda.com"
        
        # Set headers for all requests
        self.headers = {
            "Authorization": f"Bearer {OANDA_API_TOKEN}",
            "Content-Type": "application/json"
        }
        self.session.headers.update(self.headers)
        
        # Verify connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to OANDA API"""
        account = self.get_account()
        logger.info(f"Connected to OANDA. Balance: {account.get('balance')} {account.get('currency')}")
    
    def get_account(self):
        """Get account information"""
        try:
            response = self.session.get(f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/summary")
            response.raise_for_status()
            return response.json().get("account", {})
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
    def get_margin_available(self):
        """Get available margin from the account"""
        try:
            account = self.get_account()
            margin_available = float(account.get('marginAvailable', 0))
            return margin_available
        except Exception as e:
            logger.error(f"Error getting margin available: {e}")
            return 0
    
    def get_eur_usd_data(self, count=100, granularity="H1"):
        """Get EUR/USD price data for a specific timeframe"""
        try:
            params = {
                "count": count,
                "granularity": granularity,
                "price": "M"
            }
            response = self.session.get(
                f"{self.base_url}/v3/instruments/EUR_USD/candles", 
                params=params
            )
            response.raise_for_status()
            candles = response.json().get("candles", [])
            
            # Convert to pandas DataFrame
            data = []
            for candle in candles:
                if candle.get("complete", False):
                    mid = candle.get("mid", {})
                    data.append({
                        "time": candle.get("time"),
                        "open": float(mid.get("o", 0)),
                        "high": float(mid.get("h", 0)),
                        "low": float(mid.get("l", 0)),
                        "close": float(mid.get("c", 0)),
                        "volume": int(candle.get("volume", 0)),
                        "timeframe": granularity
                    })
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error getting EUR/USD data: {e}")
            return pd.DataFrame()
            
    def get_multi_timeframe_data(self):
        """Get EUR/USD data across multiple timeframes"""
        timeframes = {
            "M5": {"count": 100, "name": "5-minute"},
            "M15": {"count": 100, "name": "15-minute"},
            "H1": {"count": 100, "name": "1-hour"},
            "H4": {"count": 50, "name": "4-hour"},
            "D": {"count": 30, "name": "Daily"}
        }
        
        data = {}
        for tf, tf_info in timeframes.items():
            df = self.get_eur_usd_data(count=tf_info["count"], granularity=tf)
            if not df.empty:
                # Convert DataFrame to list of dicts for easier serialization
                data[tf_info["name"]] = df.tail(10).to_dict('records')
                
        return data
    
    def get_economic_calendar(self):
        """Get recent and upcoming economic events that might impact EUR/USD
        Note: OANDA doesn't provide economic calendar data, so this simulates it"""
        try:
            # In a real implementation, this would call an economic calendar API
            # For now, just fetch from ForexFactory or similar sites
            # We'll simulate some data
            import datetime
            
            today = datetime.datetime.now()
            
            # Simulate some economic events
            return {
                "recent_events": [
                    {
                        "date": (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                        "time": "10:00",
                        "currency": "EUR",
                        "impact": "High",
                        "event": "ECB Interest Rate Decision",
                        "actual": "4.00%",
                        "forecast": "4.00%",
                        "previous": "4.00%"
                    },
                    {
                        "date": (today - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                        "time": "14:30",
                        "currency": "USD",
                        "impact": "High",
                        "event": "Nonfarm Payrolls",
                        "actual": "236K",
                        "forecast": "240K",
                        "previous": "315K"
                    }
                ],
                "upcoming_events": [
                    {
                        "date": (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                        "time": "14:30",
                        "currency": "USD",
                        "impact": "High",
                        "event": "CPI m/m",
                        "forecast": "0.3%",
                        "previous": "0.4%"
                    },
                    {
                        "date": (today + datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                        "time": "10:00",
                        "currency": "EUR",
                        "impact": "Medium",
                        "event": "Industrial Production m/m",
                        "forecast": "0.5%",
                        "previous": "0.7%"
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error getting economic calendar: {e}")
            return {"recent_events": [], "upcoming_events": []}
            
    def get_intermarket_data(self):
        """Get intermarket correlation data that affects EUR/USD
        Note: In a production system, this would call real market data APIs"""
        try:
            # Simulate related market data
            import random
            
            # Sample data structure
            related_markets = {
                "currency_pairs": {
                    "GBP/USD": random.uniform(1.25, 1.30),
                    "USD/JPY": random.uniform(110.0, 115.0),
                    "USD/CHF": random.uniform(0.90, 0.95),
                    "AUD/USD": random.uniform(0.65, 0.70)
                },
                "commodities": {
                    "Gold": random.uniform(1800, 2000),
                    "Oil_WTI": random.uniform(70, 85)
                },
                "indices": {
                    "S&P500": random.uniform(4500, 4800),
                    "DAX": random.uniform(15000, 16000),
                    "FTSE": random.uniform(7500, 8000)
                },
                "bonds": {
                    "US_10Y_Yield": random.uniform(3.5, 4.2),
                    "DE_10Y_Yield": random.uniform(2.0, 2.5),
                    "US_DE_Spread": random.uniform(1.2, 1.8)
                },
                "correlations": {
                    "EURUSD_GBPUSD": random.uniform(0.7, 0.9),
                    "EURUSD_Gold": random.uniform(0.3, 0.6),
                    "EURUSD_US_DE_Spread": random.uniform(-0.7, -0.5)
                }
            }
            
            return related_markets
        except Exception as e:
            logger.error(f"Error getting intermarket data: {e}")
            return {}
            
    def get_technical_indicators(self, price_data):
        """Calculate common technical indicators for the price data"""
        try:
            if price_data.empty:
                return {}
                
            # Clone the dataframe to avoid modifying the original
            df = price_data.copy()
            
            # Calculate moving averages
            df['MA_20'] = df['close'].rolling(window=20).mean()
            df['MA_50'] = df['close'].rolling(window=50).mean()
            df['MA_100'] = df['close'].rolling(window=100).mean()
            
            # Calculate RSI (14-period)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands (20-period, 2 standard deviations)
            df['BB_Middle'] = df['close'].rolling(window=20).mean()
            df['BB_StdDev'] = df['close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_StdDev']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_StdDev']
            
            # Calculate MACD
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Calculate ATR (14-period)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # Get the latest values for all indicators
            try:
                latest = df.iloc[-1].to_dict()
            except IndexError:
                logger.warning("Not enough data to calculate indicators, using empty values")
                return {
                    "moving_averages": {},
                    "oscillators": {},
                    "volatility": {},
                    "trend_strength": {},
                    "support_resistance": {"key_resistance_levels": [], "key_support_levels": []}
                }
            
            # Create a simplified structure with just the latest indicator values
            indicators = {
                "moving_averages": {
                    "MA_20": latest.get('MA_20'),
                    "MA_50": latest.get('MA_50'),
                    "MA_100": latest.get('MA_100'),
                },
                "oscillators": {
                    "RSI": latest.get('RSI'),
                    "MACD": latest.get('MACD'),
                    "MACD_Signal": latest.get('MACD_Signal'),
                    "MACD_Histogram": latest.get('MACD_Histogram')
                },
                "volatility": {
                    "ATR": latest.get('ATR'),
                    "BB_Width": latest.get('BB_Upper') - latest.get('BB_Lower') if pd.notna(latest.get('BB_Upper')) and pd.notna(latest.get('BB_Lower')) else None,
                    "BB_Upper": latest.get('BB_Upper'),
                    "BB_Lower": latest.get('BB_Lower')
                },
                "trend_strength": {
                    "Price_vs_MA20": (latest.get('close') / latest.get('MA_20') - 1) * 100 if pd.notna(latest.get('MA_20')) and latest.get('MA_20') > 0 else None,
                    "MA20_vs_MA50": (latest.get('MA_20') / latest.get('MA_50') - 1) * 100 if pd.notna(latest.get('MA_50')) and latest.get('MA_50') > 0 else None,
                }
            }
            
            # Add support/resistance levels
            try:
                highs = df['high'].nlargest(5).tolist()
                lows = df['low'].nsmallest(5).tolist()
                
                indicators["support_resistance"] = {
                    "key_resistance_levels": highs,
                    "key_support_levels": lows
                }
            except Exception as e:
                logger.warning(f"Error calculating support/resistance levels: {e}")
                indicators["support_resistance"] = {
                    "key_resistance_levels": [],
                    "key_support_levels": []
                }
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}

    def get_open_positions(self):
        """Get open positions"""
        try:
            response = self.session.get(f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/openPositions")
            response.raise_for_status()
            return response.json().get("positions", [])
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def execute_trade(self, direction, units, stop_loss=None, take_profit_levels=None, trailing_stop_distance=None):
        """Execute a trade on EUR/USD with advanced risk management"""
        try:
            # Determine units based on direction
            if direction.upper() == "SELL":
                units = -abs(units)
            else:  # BUY
                units = abs(units)
                
            # Build order request
            order_data = {
                "order": {
                    "instrument": "EUR_USD",
                    "units": str(units),
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT",
                    "type": "MARKET"
                }
            }
            
            # Add stop loss if provided
            if stop_loss is not None:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(stop_loss),
                    "timeInForce": "GTC",
                    "triggerMode": "TOP_OF_BOOK"
                }
                
                # Add trailing stop if provided
                if trailing_stop_distance is not None:
                    order_data["order"]["trailingStopLossOnFill"] = {
                        "distance": str(trailing_stop_distance),
                        "timeInForce": "GTC",
                        "triggerCondition": "DEFAULT"
                    }
            
            # Add take profit if provided (just the first level)
            # We'll handle multiple take profit levels after the order is executed
            if take_profit_levels and len(take_profit_levels) > 0:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit_levels[0]),
                    "timeInForce": "GTC"
                }
            
            # Execute the order
            logger.info(f"Executing {direction} order for EUR_USD with {units} units")
            response = self.session.post(
                f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/orders",
                json=order_data
            )
            
            # Handle response
            response.raise_for_status()
            result = response.json()
            
            # Check if the order was filled
            if "orderFillTransaction" in result:
                fill_txn = result["orderFillTransaction"]
                fill_id = fill_txn.get("id", "Unknown ID")
                trade_id = fill_txn.get("tradeOpened", {}).get("tradeID")
                logger.info(f"Order executed and filled: {fill_id}, Trade ID: {trade_id}")
                
                # If we have multiple take profit levels, create partial close orders
                if take_profit_levels and len(take_profit_levels) > 1 and trade_id:
                    self._create_partial_profit_orders(trade_id, direction, units, take_profit_levels)
            else:
                logger.info(f"Order created: {result.get('orderCreateTransaction', {}).get('id', 'Unknown ID')}")
                
            return result
            
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors from the API
            error_response = {}
            try:
                error_response = e.response.json()
                logger.error(f"OANDA API error: {error_response}")
            except:
                logger.error(f"OANDA API HTTP error: {str(e)}")
            
            return {"error": str(e), "details": error_response}
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {"error": str(e)}
            
    def _create_partial_profit_orders(self, trade_id, direction, total_units, take_profit_levels):
        """Create partial take profit orders for staged profit-taking"""
        try:
            # We'll close the position in stages
            # For example, with 3 levels, we'll close 33% at each level
            num_levels = len(take_profit_levels) - 1  # Subtract 1 because first level is handled in initial order
            units_per_level = abs(total_units) // (num_levels + 1)
            
            for i, tp_level in enumerate(take_profit_levels[1:], 1):
                # Calculate units for this level
                if i == num_levels:  # Last level, close whatever remains
                    close_units = "ALL"
                else:
                    close_units = str(units_per_level)
                
                # Create take profit order
                take_profit_order = {
                    "order": {
                        "type": "TAKE_PROFIT",
                        "tradeID": trade_id,
                        "price": str(tp_level),
                        "timeInForce": "GTC",
                        "triggerCondition": "DEFAULT",
                        "clientExtensions": {
                            "comment": f"Take profit level {i+1}"
                        }
                    }
                }
                
                # Submit the order
                response = self.session.post(
                    f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/orders",
                    json=take_profit_order
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Created take profit order for level {i+1} at {tp_level}: {result.get('orderCreateTransaction', {}).get('id', 'Unknown ID')}")
                
        except Exception as e:
            logger.error(f"Error creating partial profit orders: {e}")
            
    def get_market_sentiment(self):
        """Get market sentiment data for EUR/USD
        This would normally come from an external API, but we'll simulate it"""
        try:
            import random
            
            # Simulate sentiment data
            bullish_percent = random.randint(30, 70)
            bearish_percent = 100 - bullish_percent
            
            # Volume indicators
            volume_status = random.choice(["High", "Average", "Low"])
            
            # Positioning data
            positioning = {
                "retail_long_percent": random.randint(30, 70),
                "retail_short_percent": random.randint(30, 70),
                "institutional_bias": random.choice(["Bullish", "Bearish", "Neutral"])
            }
            
            return {
                "pair": "EUR/USD",
                "timestamp": datetime.datetime.now().isoformat(),
                "sentiment": {
                    "bullish_percent": bullish_percent,
                    "bearish_percent": bearish_percent,
                    "overall": "Bullish" if bullish_percent > 55 else "Bearish" if bullish_percent < 45 else "Neutral"
                },
                "volume": {
                    "status": volume_status,
                    "relative_to_average": random.uniform(0.8, 1.2)
                },
                "positioning": positioning
            }
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {}
    
    def update_stop_loss(self, stop_loss):
        """Update stop loss for EUR/USD position"""
        try:
            # Get open trades for EUR_USD
            response = self.session.get(
                f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/trades?instrument=EUR_USD&state=OPEN"
            )
            response.raise_for_status()
            trades = response.json().get("trades", [])
            
            if not trades:
                logger.warning("No open EUR/USD trades found to update stop loss")
                return {"error": "No open trades found"}
            
            results = []
            for trade in trades:
                trade_id = trade["id"]
                update_data = {
                    "stopLoss": {
                        "price": str(stop_loss),
                        "timeInForce": "GTC"
                    }
                }
                
                try:
                    update_response = self.session.put(
                        f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_id}/orders",
                        json=update_data
                    )
                    update_response.raise_for_status()
                    results.append(update_response.json())
                    logger.info(f"Updated stop loss for trade {trade_id} to {stop_loss}")
                except Exception as update_error:
                    logger.error(f"Error updating stop loss for trade {trade_id}: {update_error}")
                    results.append({"error": str(update_error), "trade_id": trade_id})
            
            return results
        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
            return {"error": str(e)}

# --- Memory System ---
class TradingMemory:
    """Maintains trading history and system state"""
    
    def __init__(self):
        """Initialize trading memory"""
        self.memory_file = "data/system_memory.json"
        self.trade_log_file = "data/trade_log.jsonl"
        self.review_log_file = "data/review_log.jsonl"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Initialize or load system memory
        self.load_memory()
    
    def load_memory(self):
        """Load or initialize system memory"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.memory = json.load(f)
            except:
                self.initialize_memory()
        else:
            self.initialize_memory()
    
    def initialize_memory(self):
        """Create initial memory structure"""
        self.memory = {
            "created": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "safety_level": INITIAL_SAFETY_LEVEL,
            "daily_drawdown": 0.0,
            "daily_profit": 0.0,
            "daily_profit_pct": 0.0,
            "daily_high_balance": 0.0,
            "daily_starting_balance": 0.0,
            "total_risk_committed": 0.0,
            "prompt_versions": [],
            "strategy_weights": {
                "trend_following": 1.0,
                "breakout": 1.0,
                "mean_reversion": 1.0
            }
        }
        self.save_memory()
    
    def save_memory(self):
        """Save system memory to disk"""
        self.memory["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)
    
    def log_trade(self, trade_data):
        """Log a trade to the trade history"""
        # Update trade count
        self.memory["trade_count"] += 1
        
        # Update win/loss stats if applicable
        if trade_data.get("is_win"):
            self.memory["win_count"] += 1
            # Increase safety level after win
            self.memory["safety_level"] = min(0.05, self.memory["safety_level"] + INCREASE_FACTOR)
        elif trade_data.get("is_loss"):
            self.memory["loss_count"] += 1
        
        # Append to trade log
        with open(self.trade_log_file, "a") as f:
            f.write(json.dumps(trade_data) + "\n")
        
        # Save updated memory
        self.save_memory()
    
    def log_review(self, review_data):
        """Log a system review"""
        # Save the new prompt version if provided
        if "prompt_version" in review_data:
            self.memory["prompt_versions"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "content": review_data["prompt_version"],
                "reason": review_data.get("reason", "Regular review")
            })
        
        # Update strategy weights if provided
        if "strategy_weights" in review_data:
            self.memory["strategy_weights"] = review_data["strategy_weights"]
        
        # Append to review log
        with open(self.review_log_file, "a") as f:
            f.write(json.dumps(review_data) + "\n")
        
        # Save updated memory
        self.save_memory()
    
    def get_recent_trades(self, limit=10):
        """Get recent trades from log"""
        trades = []
        try:
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, "r") as f:
                    for line in f:
                        trades.append(json.loads(line))
                
                # Sort by timestamp (newest first) and limit
                trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                return trades[:limit]
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
        
        return trades
    
    def reset_daily_stats(self, account_balance):
        """Reset daily statistics"""
        self.memory["daily_drawdown"] = 0.0
        self.memory["daily_profit"] = 0.0
        self.memory["daily_profit_pct"] = 0.0
        self.memory["daily_high_balance"] = account_balance
        self.memory["daily_starting_balance"] = account_balance
        self.memory["total_risk_committed"] = 0.0
        self.save_memory()

# --- LLM Trading Brain ---
class LLMTradingBrain:
    """LLM-powered trading decision maker with self-improvement capabilities"""
    
    def __init__(self, memory):
        """Initialize the LLM trading brain"""
        self.memory = memory
        openai.api_key = OPENAI_API_KEY
        
        # Initial system prompt (will be evolved over time)
        self.system_prompt = """
You are a self-evolving forex trading AI specializing in EUR/USD.

Your goal is to achieve a 10% daily return while respecting strict risk management:
- Maximum account risk: 10%
- Maximum daily drawdown: 10%
- Implement trailing stops and profit taking

Your decisions should be based on:
1. Technical analysis of EUR/USD price action
2. Analysis of past trades (what worked and what didn't)
3. Risk management based on account's current exposure

Always provide a clear trade plan with:
- Direction (BUY/SELL)
- Entry price and acceptable range
- Stop loss level (must be specified)
- Take profit targets (multiple levels recommended)
- Risk percentage (1-5% based on conviction)
- Position sizing logic
- Technical justification

Consider using these strategies based on market conditions:
- Trend following: Identify and follow established trends
- Breakout: Capture price moves from established ranges
- Mean reversion: Trade returns to mean after extreme movements

Your memory and self-improvement:
- Learn from successful and unsuccessful trades
- Adapt your approach based on recent performance
- Evolve your strategies over time for better results
"""
    
    def analyze_market(self, price_data, account_data, positions, market_data=None):
        """Analyze market and make trading decisions with comprehensive data"""
        # Extract recent price data from main timeframe
        recent_candles = price_data.tail(50).to_dict('records')
        simplified_candles = [
            {
                "time": candle["time"][-8:], # Just HH:MM:SS
                "open": round(candle["open"], 5),
                "high": round(candle["high"], 5),
                "low": round(candle["low"], 5),
                "close": round(candle["close"], 5)
            }
            for candle in recent_candles[-10:] # Last 10 candles
        ]
        
        # Get recent trades for context
        recent_trades = self.memory.get_recent_trades(5)
        
        # Calculate win rate
        win_rate = 0
        if self.memory.memory["trade_count"] > 0:
            win_rate = self.memory.memory["win_count"] / self.memory.memory["trade_count"] * 100
        
        # Ensure market_data is a dict
        if market_data is None:
            market_data = {}
            
        # Extract data components for prompt
        multi_timeframe_data = market_data.get("multi_timeframe", {})
        technical_indicators = market_data.get("technical_indicators", {})
        intermarket_data = market_data.get("intermarket", {})
        economic_data = market_data.get("economic", {})
        sentiment_data = market_data.get("sentiment", {})
        
        # Build the prompt
        user_prompt = f"""
## Current Trading Status
- Account Balance: {account_data.get('balance')} {account_data.get('currency')}
- Safety Level: {self.memory.memory['safety_level']:.4f} (risk per trade)
- Daily Profit: {self.memory.memory['daily_profit_pct']:.2f}% (Target: 10%)
- Win Rate: {win_rate:.1f}% ({self.memory.memory['win_count']}/{self.memory.memory['trade_count']} trades)
- Open EUR/USD Positions: {len([p for p in positions if p.get('instrument') == 'EUR_USD'])}

## EUR/USD Recent Price Data (H1 Timeframe)
{json.dumps(simplified_candles, indent=2)}

## Technical Indicators
{json.dumps(technical_indicators, indent=2)}

## Multi-Timeframe Analysis
{json.dumps(multi_timeframe_data, indent=2)}

## Intermarket Correlations
{json.dumps(intermarket_data, indent=2)}

## Economic Calendar
{json.dumps(economic_data, indent=2)}

## Market Sentiment
{json.dumps(sentiment_data, indent=2)}

## Recent Trades
{json.dumps(recent_trades[:3], indent=2)}

## Strategy Weights
{json.dumps(self.memory.memory['strategy_weights'], indent=2)}

Based on this comprehensive market analysis:
1. Analyze EUR/USD across all timeframes and consider correlations with other markets
2. Consider economic events, sentiment data, and technical indicators
3. Decide if we should OPEN a new position, UPDATE an existing position, or WAIT
4. If opening, provide complete trade details with entry, stop, targets and risk
5. If updating, provide updated stop levels for proper risk management
6. Consider multiple take-profit levels for staged profit-taking
7. Explain your reasoning based on multi-timeframe analysis and market conditions

Respond in JSON format with: 
{{
  "market_analysis": "Your comprehensive analysis including multiple timeframes and factors",
  "action": "OPEN, UPDATE, or WAIT",
  "trade_details": {{
    "direction": "BUY or SELL",
    "entry_price": 1.xxxx,
    "entry_range": [min, max],
    "stop_loss": 1.xxxx,
    "take_profit": [level1, level2, level3],
    "risk_percent": "between 1-5",
    "trailing_stop_distance": 0.xxxx,
    "strategy": "trend_following, breakout, or mean_reversion",
    "reasoning": "Technical justification with multi-timeframe context"
  }},
  "update_details": {{
    "new_stop_loss": 1.xxxx,
    "reason": "Why update the stop"
  }},
  "exit_strategy": {{
    "early_exit_conditions": "Conditions to exit early if trade not performing as expected",
    "partial_profit_taking": "Detailed plan for taking profits at different levels"
  }},
  "self_improvement": {{
    "performance_assessment": "How your recent trades performed across different market conditions",
    "strategy_adjustments": "How you would adjust your strategy weights based on all available data",
    "prompt_improvements": "Suggestions to improve your system prompt"
  }}
}}"""
        
        # Call LLM API
        try:
            logger.info("Calling LLM API for market analysis")
            completion = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(completion.choices[0].message.content)
            logger.info(f"LLM response received: {result.get('action', 'unknown action')}")
            return result
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return {"error": str(e), "action": "WAIT"}
    
    def review_and_evolve(self, account_data):
        """Review performance and evolve trading approach"""
        # Get all recent trades
        recent_trades = self.memory.get_recent_trades(20)
        
        # Calculate performance metrics
        win_rate = 0
        if self.memory.memory["trade_count"] > 0:
            win_rate = self.memory.memory["win_count"] / self.memory.memory["trade_count"] * 100
        
        # Calculate daily P&L
        daily_profit_pct = self.memory.memory["daily_profit_pct"]
        
        # Build the review prompt without f-string for the problematic line
        review_prompt = (
            "## Performance Review\n"
            f"- Account Balance: {account_data.get('balance')} {account_data.get('currency')}\n"
            f"- Daily P&L: {daily_profit_pct:.2f}% (Target: 10%)\n"
            f"- Win Rate: {win_rate:.1f}% ({self.memory.memory['win_count']}/{self.memory.memory['trade_count']} trades)\n"
            f"- Current Safety Level: {self.memory.memory['safety_level']:.4f}\n"
            f"- Strategy Weights: {json.dumps(self.memory.memory['strategy_weights'], indent=2)}\n\n"
            "## Current System Prompt\n"
            f"{self.system_prompt}\n\n"
            "## Recent Trades (Last 20)\n"
            f"{json.dumps(recent_trades, indent=2)}\n\n"
            "Based on this comprehensive review of your trading performance:\n\n"
            "1. Analyze what's working and what's not working in your trading approach\n"
            "2. Evaluate which strategies have been most effective\n"
            "3. Suggest modifications to your strategy weights\n"
            "4. Recommend improvements to your system prompt for better results\n"
            "5. Consider changes to your risk management approach\n\n"
            "Provide your evolution recommendations in JSON format:\n"
            "{\n"
            '  "performance_analysis": "Detailed analysis of trading performance",\n'
            '  "strategy_weights": {\n'
            '    "trend_following": "float", \n'
            '    "breakout": "float", \n'
            '    "mean_reversion": "float"\n'
            '  },\n'
            '  "risk_adjustments": "Recommendations for risk management",\n'
            '  "improved_prompt": "Complete improved system prompt",\n'
            '  "reasoning": "Detailed reasoning for all changes"\n'
            "}"
        )
        
        # Call LLM API for review
        try:
            logger.info("Calling LLM API for system evolution review")
            completion = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert trading system reviewer who improves a self-evolving forex trading AI."},
                    {"role": "user", "content": review_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(completion.choices[0].message.content)
            
            # Update system with recommendations
            if "improved_prompt" in result and result["improved_prompt"]:
                self.system_prompt = result["improved_prompt"]
                logger.info("Updated system prompt based on review")
            
            # Parse and convert strategy weights to floats
            if "strategy_weights" in result:
                try:
                    strategy_weights = {}
                    for strategy, weight in result["strategy_weights"].items():
                        if isinstance(weight, (int, float)):
                            strategy_weights[strategy] = float(weight)
                        elif isinstance(weight, str):
                            # Try to convert string to float
                            try:
                                strategy_weights[strategy] = float(weight.replace("float", "").strip())
                            except:
                                strategy_weights[strategy] = 1.0  # Default
                        else:
                            strategy_weights[strategy] = 1.0  # Default
                    
                    # If valid weights were found, use them
                    if strategy_weights:
                        result["strategy_weights"] = strategy_weights
                        logger.info(f"Updated strategy weights: {strategy_weights}")
                except Exception as e:
                    logger.error(f"Error converting strategy weights: {e}")
            
            # Log the review
            self.memory.log_review({
                "timestamp": datetime.datetime.now().isoformat(),
                "review": result["performance_analysis"],
                "strategy_weights": result.get("strategy_weights", self.memory.memory["strategy_weights"]),
                "prompt_version": self.system_prompt,
                "reason": result.get("reasoning", "Scheduled review")
            })
            
            logger.info("Completed system evolution review")
            return result
        except Exception as e:
            logger.error(f"Error in system evolution review: {e}")
            return {"error": str(e)}

# --- Main Trading Bot ---
class EURUSDTradingBot:
    """Self-evolving EUR/USD trading bot powered by LLM"""
    
    def __init__(self):
        """Initialize the trading bot"""
        try:
            self.oanda = OandaClient()
            self.memory = TradingMemory()
            self.brain = LLMTradingBrain(self.memory)
            
            # Initialize account info
            account = self.oanda.get_account()
            balance = float(account.get('balance', 1000))
            
            # Reset daily stats if needed
            self.memory.reset_daily_stats(balance)
            
            logger.info(f"EUR/USD Trading Bot initialized. Balance: {balance} {account.get('currency')}")
        except Exception as e:
            logger.error(f"Error initializing trading bot: {e}")
            raise
    
    def update_account_status(self):
        """Update account status and check daily limits"""
        account = self.oanda.get_account()
        balance = float(account.get('balance', 1000))
        
        # Update daily tracking
        if balance > self.memory.memory["daily_high_balance"]:
            self.memory.memory["daily_high_balance"] = balance
        
        # Calculate daily P&L
        self.memory.memory["daily_profit"] = balance - self.memory.memory["daily_starting_balance"]
        self.memory.memory["daily_profit_pct"] = (balance / self.memory.memory["daily_starting_balance"] - 1) * 100
        
        # Calculate drawdown
        if self.memory.memory["daily_high_balance"] > 0:
            current_drawdown = (self.memory.memory["daily_high_balance"] - balance) / self.memory.memory["daily_high_balance"] * 100
            self.memory.memory["daily_drawdown"] = max(self.memory.memory["daily_drawdown"], current_drawdown)
        
        # Save updated memory
        self.memory.save_memory()
        
        return {
            "balance": balance,
            "daily_profit_pct": self.memory.memory["daily_profit_pct"],
            "daily_drawdown": self.memory.memory["daily_drawdown"],
            "max_drawdown_reached": self.memory.memory["daily_drawdown"] >= MAX_DAILY_DRAWDOWN,
            "target_reached": self.memory.memory["daily_profit_pct"] >= TARGET_DAILY_RETURN
        }
    
    def execute_decision(self, decision):
        """Execute trading decision from LLM with enhanced risk management"""
        action = decision.get("action", "WAIT")
        
        if action == "OPEN":
            # Extract trade details
            trade_details = decision.get("trade_details", {})
            direction = trade_details.get("direction")
            entry_price = trade_details.get("entry_price")
            stop_loss = trade_details.get("stop_loss")
            take_profit_levels = trade_details.get("take_profit", [])
            risk_percent = trade_details.get("risk_percent", 2.0)
            trailing_stop_distance = trade_details.get("trailing_stop_distance")
            
            # Get exit strategy for early management
            exit_strategy = decision.get("exit_strategy", {})
            early_exit_conditions = exit_strategy.get("early_exit_conditions")
            
            # Validate required fields
            if not all([direction, entry_price, stop_loss]):
                logger.warning(f"Missing required trade details: {trade_details}")
                return False
            
            # Get account info for position sizing
            account = self.oanda.get_account()
            balance = float(account.get('balance', 1000))
            
            # Convert values to float
            try:
                entry_price = float(entry_price)
                stop_loss = float(stop_loss)
                
                # Handle risk percent which might be a string like "2.5" or "2"
                if isinstance(risk_percent, str):
                    # Remove any non-numeric characters except decimal point
                    risk_percent = ''.join(c for c in risk_percent if c.isdigit() or c == '.')
                    risk_percent = float(risk_percent) if risk_percent else 2.0
                elif isinstance(risk_percent, (int, float)):
                    risk_percent = float(risk_percent)
                else:
                    risk_percent = 2.0
                
                # Cap risk percent between 1-5%
                risk_percent = max(1.0, min(5.0, risk_percent))
                
                # Make sure take_profit is a list of floats
                if isinstance(take_profit_levels, list):
                    take_profit = [float(tp) for tp in take_profit_levels if tp]
                elif take_profit_levels and isinstance(take_profit_levels, (int, float, str)):
                    take_profit = [float(take_profit_levels)]
                else:
                    take_profit = []
                
                # Process trailing stop if provided
                if trailing_stop_distance:
                    if isinstance(trailing_stop_distance, str):
                        trailing_stop_distance = float(''.join(c for c in trailing_stop_distance if c.isdigit() or c == '.'))
                    elif isinstance(trailing_stop_distance, (int, float)):
                        trailing_stop_distance = float(trailing_stop_distance)
                    else:
                        trailing_stop_distance = None
                    
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting trade values to numeric: {e}")
                logger.error(f"Raw values - entry_price: {entry_price}, stop_loss: {stop_loss}, risk_percent: {risk_percent}")
                return False
            
            # Calculate position size based on risk
            risk_amount = balance * (risk_percent / 100)
            stop_distance = abs(entry_price - stop_loss)
            
            if stop_distance <= 0.0001:  # Minimum stop distance
                logger.error("Stop distance is too small, using minimum value")
                stop_distance = 0.0001
            
            # Basic position sizing (very simplified)
            units = int(risk_amount / stop_distance * 10000)  # Scaled for EUR/USD
            
            # Apply reasonable limits FIRST before checking margin
            units = min(units, 10000)  # Maximum 0.1 standard lot (reduced from 1.0)
            units = max(units, 1000)   # Minimum 0.01 lots (micro)
            
            # THEN check margin with the limited position size
            margin_available = self.oanda.get_margin_available()
            margin_needed = units * entry_price * 0.02  # Approximate margin needed (2% of position value)
            
            if margin_needed > margin_available:
                logger.warning(f"Insufficient margin: needed {margin_needed:.2f}, available {margin_available:.2f}")
                # Reduce position size to fit available margin (leave 20% buffer)
                max_units = int((margin_available * 0.8) / (entry_price * 0.02))
                # Ensure it doesn't go below minimum
                max_units = max(max_units, 1000)
                units = min(units, max_units)
                logger.info(f"Reduced position size to {units} units due to margin constraints")
            
            logger.info(f"Calculated position size: {units} units based on risk {risk_percent}% and stop distance {stop_distance}")
            
            # Execute trade with all risk management parameters
            result = self.oanda.execute_trade(
                direction=direction,
                units=units,
                stop_loss=stop_loss,
                take_profit_levels=take_profit,
                trailing_stop_distance=trailing_stop_distance
            )
            
            # Process the result
            if "orderFillTransaction" in result:
                logger.info(f"Trade executed: {direction} EUR/USD, {units} units")
                fill = result["orderFillTransaction"]
                
                # Log trade details with enhanced information
                self.memory.log_trade({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "OPEN",
                    "direction": direction,
                    "entry_price": float(fill.get("price", entry_price)),
                    "stop_loss": float(stop_loss),
                    "take_profit_levels": take_profit,
                    "trailing_stop_distance": trailing_stop_distance,
                    "units": units,
                    "risk_percent": float(risk_percent),
                    "strategy": trade_details.get("strategy", "unknown"),
                    "reasoning": trade_details.get("reasoning", ""),
                    "early_exit_conditions": early_exit_conditions,
                    "is_win": None,  # Will be updated when closed
                    "is_loss": None  # Will be updated when closed
                })
                return True
            elif "orderCancelTransaction" in result:
                cancel = result["orderCancelTransaction"]
                reason = cancel.get("reason", "Unknown")
                logger.error(f"Order cancelled: {reason}")
                return False
            elif "error" in result:
                logger.error(f"Trade execution failed: {result['error']}")
                if "details" in result:
                    logger.error(f"Error details: {result['details']}")
                return False
            else:
                logger.error(f"Unknown trade execution result: {result}")
                return False
                
        elif action == "UPDATE":
            # Extract update details
            update_details = decision.get("update_details", {})
            new_stop_loss = update_details.get("new_stop_loss")
            
            if new_stop_loss:
                # Ensure numeric
                try:
                    new_stop_loss = float(new_stop_loss)
                except (ValueError, TypeError):
                    logger.error(f"Invalid stop loss value: {new_stop_loss}")
                    return False
                
                # Update stop loss
                result = self.oanda.update_stop_loss(new_stop_loss)
                
                if not isinstance(result, dict) or "error" not in result:
                    logger.info(f"Stop loss updated to {new_stop_loss}")
                    
                    # Log the update
                    self.memory.log_trade({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "action": "UPDATE",
                        "new_stop_loss": float(new_stop_loss),
                        "reason": update_details.get("reason", "")
                    })
                    return True
                else:
                    logger.error(f"Stop loss update failed: {result}")
                    return False
        
        # Implement early exit logic if positions need to be closed based on conditions
        if "exit_strategy" in decision and action != "OPEN":
            positions = self.oanda.get_open_positions()
            eur_usd_positions = [p for p in positions if p.get("instrument") == "EUR_USD"]
            
            if eur_usd_positions and "early_exit_conditions" in decision.get("exit_strategy", {}):
                logger.info(f"Checking early exit conditions for {len(eur_usd_positions)} open positions")
                # Logic to check if early exit conditions are met
                # Implementation would depend on specific conditions provided by LLM
                # For now, just log it
                
        return True  # For WAIT action or successful execution
    
    def trading_cycle(self):
        """Run a complete trading cycle with comprehensive data analysis"""
        try:
            # Update account status
            status = self.update_account_status()
            
            # Check if max drawdown reached
            if status["max_drawdown_reached"]:
                logger.warning(f"Maximum daily drawdown reached: {status['daily_drawdown']:.2f}%. Pausing trading.")
                return False
            
            # Check if daily target reached
            if status["target_reached"]:
                logger.info(f"Daily target reached: {status['daily_profit_pct']:.2f}%. Continuing with increased selectivity.")
            
            # Get market data from multiple timeframes
            price_data = self.oanda.get_eur_usd_data(count=100, granularity="H1")
            if price_data.empty:
                logger.error("Failed to get EUR/USD price data")
                return False
            
            # Get data from multiple timeframes
            logger.info("Getting multi-timeframe data")
            multi_timeframe_data = self.oanda.get_multi_timeframe_data()
            
            # Calculate technical indicators
            logger.info("Calculating technical indicators")
            technical_indicators = self.oanda.get_technical_indicators(price_data)
            
            # Get intermarket data
            logger.info("Getting intermarket correlation data")
            intermarket_data = self.oanda.get_intermarket_data()
            
            # Get economic calendar data
            logger.info("Getting economic calendar data")
            economic_data = self.oanda.get_economic_calendar()
            
            # Get market sentiment data
            logger.info("Getting market sentiment data")
            sentiment_data = self.oanda.get_market_sentiment()
            
            # Get account data
            account_data = self.oanda.get_account()
            
            # Get positions
            positions = self.oanda.get_open_positions()
            
            # Prepare comprehensive market data package
            market_data = {
                "multi_timeframe": multi_timeframe_data,
                "technical_indicators": technical_indicators,
                "intermarket": intermarket_data,
                "economic": economic_data,
                "sentiment": sentiment_data
            }
            
            # Analyze market with comprehensive data and get decision
            logger.info("Analyzing market with comprehensive data")
            decision = self.brain.analyze_market(
                price_data, 
                account_data, 
                positions,
                market_data=market_data
            )
            
            # Generate and save trading report
            self._generate_trading_report(
                status, 
                account_data, 
                positions, 
                price_data, 
                decision,
                market_data
            )
            
            # Execute decision
            if "error" not in decision:
                logger.info(f"Decision: {decision.get('action', 'WAIT')}")
                return self.execute_decision(decision)
            else:
                logger.error(f"Error in market analysis: {decision.get('error')}")
                return False
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return False
            
    def _generate_trading_report(self, status, account_data, positions, price_data, decision, market_data=None):
        """Generate a detailed trading report for human review"""
        try:
            report_dir = "reports"
            os.makedirs(report_dir, exist_ok=True)
            
            # Create report filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(report_dir, f"trading_report_{timestamp}.txt")
            
            with open(report_file, "w") as f:
                # Write header
                f.write("=" * 80 + "\n")
                f.write(f"EUR/USD TRADING REPORT - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Account status
                f.write("ACCOUNT STATUS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Balance: {account_data.get('balance')} {account_data.get('currency')}\n")
                f.write(f"Daily P&L: {status['daily_profit_pct']:.2f}%\n")
                f.write(f"Daily Drawdown: {status['daily_drawdown']:.2f}%\n")
                f.write(f"Safety Level: {self.memory.memory['safety_level']:.4f}\n\n")
                
                # Open Positions
                f.write("OPEN POSITIONS\n")
                f.write("-" * 80 + "\n")
                if positions:
                    for pos in positions:
                        if pos.get("instrument") == "EUR_USD":
                            f.write(f"Direction: {'LONG' if float(pos.get('long', {}).get('units', 0)) > 0 else 'SHORT'}\n")
                            f.write(f"Units: {pos.get('long', {}).get('units') or pos.get('short', {}).get('units')}\n")
                            f.write(f"Avg Price: {pos.get('long', {}).get('averagePrice') or pos.get('short', {}).get('averagePrice')}\n")
                            f.write(f"Unrealized P&L: {pos.get('unrealizedPL')}\n\n")
                else:
                    f.write("No open positions\n\n")
                
                # Technical Summary
                if market_data and "technical_indicators" in market_data:
                    f.write("TECHNICAL INDICATORS\n")
                    f.write("-" * 80 + "\n")
                    tech = market_data["technical_indicators"]
                    
                    # MA status
                    if "moving_averages" in tech:
                        ma = tech["moving_averages"]
                        ma_20 = ma.get('MA_20')
                        ma_50 = ma.get('MA_50')
                        ma_100 = ma.get('MA_100')
                        f.write(f"MA20: {ma_20:.5f if pd.notna(ma_20) else 'N/A'}, "
                                f"MA50: {ma_50:.5f if pd.notna(ma_50) else 'N/A'}, "
                                f"MA100: {ma_100:.5f if pd.notna(ma_100) else 'N/A'}\n")
                        
                    # Oscillators
                    if "oscillators" in tech:
                        osc = tech["oscillators"]
                        rsi = osc.get('RSI')
                        macd = osc.get('MACD')
                        f.write(f"RSI(14): {rsi:.2f if pd.notna(rsi) else 'N/A'}, "
                                f"MACD: {macd:.5f if pd.notna(macd) else 'N/A'}\n")
                        
                    # Volatility
                    if "volatility" in tech:
                        vol = tech["volatility"]
                        atr = vol.get('ATR')
                        bb_width = vol.get('BB_Width')
                        f.write(f"ATR(14): {atr:.5f if pd.notna(atr) else 'N/A'}, "
                                f"BB Width: {bb_width:.5f if pd.notna(bb_width) else 'N/A'}\n")
                        
                    # Support/Resistance
                    if "support_resistance" in tech:
                        sr = tech["support_resistance"]
                        resistance = ', '.join([f'{r:.5f}' for r in sr.get('key_resistance_levels', []) if pd.notna(r)])
                        support = ', '.join([f'{s:.5f}' for s in sr.get('key_support_levels', []) if pd.notna(s)])
                        f.write(f"Resistance: {resistance or 'None'}\n")
                        f.write(f"Support: {support or 'None'}\n\n")
                
                # Intermarket Analysis
                if market_data and "intermarket" in market_data:
                    f.write("INTERMARKET ANALYSIS\n")
                    f.write("-" * 80 + "\n")
                    inter = market_data["intermarket"]
                    
                    if "correlations" in inter:
                        f.write("Key Correlations:\n")
                        for k, v in inter["correlations"].items():
                            f.write(f"{k}: {v:.2f}\n")
                        f.write("\n")
                        
                # Market Analysis
                f.write("MARKET ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(f"{decision.get('market_analysis', 'No analysis available')}\n\n")
                
                # Trading Decision
                f.write("TRADING DECISION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Action: {decision.get('action', 'WAIT')}\n")
                
                if decision.get('action') == "OPEN":
                    trade_details = decision.get('trade_details', {})
                    f.write(f"Direction: {trade_details.get('direction')}\n")
                    f.write(f"Entry Price: {trade_details.get('entry_price')}\n")
                    f.write(f"Stop Loss: {trade_details.get('stop_loss')}\n")
                    f.write(f"Take Profit Levels: {trade_details.get('take_profit')}\n")
                    f.write(f"Risk Percent: {trade_details.get('risk_percent')}%\n")
                    f.write(f"Strategy: {trade_details.get('strategy')}\n")
                    f.write(f"Reasoning: {trade_details.get('reasoning')}\n\n")
                elif decision.get('action') == "UPDATE":
                    update_details = decision.get('update_details', {})
                    f.write(f"New Stop Loss: {update_details.get('new_stop_loss')}\n")
                    f.write(f"Reason: {update_details.get('reason')}\n\n")
                
                # Exit Strategy
                if 'exit_strategy' in decision:
                    f.write("EXIT STRATEGY\n")
                    f.write("-" * 80 + "\n")
                    exit_strategy = decision.get('exit_strategy', {})
                    f.write(f"Early Exit Conditions: {exit_strategy.get('early_exit_conditions', 'None')}\n")
                    f.write(f"Partial Profit Taking: {exit_strategy.get('partial_profit_taking', 'None')}\n\n")
                
                # Self Improvement
                f.write("SELF IMPROVEMENT\n")
                f.write("-" * 80 + "\n")
                self_improvement = decision.get('self_improvement', {})
                f.write(f"Performance Assessment: {self_improvement.get('performance_assessment', 'None')}\n")
                f.write(f"Strategy Adjustments: {self_improvement.get('strategy_adjustments', 'None')}\n\n")
                
                # Recent Performance
                f.write("RECENT PERFORMANCE\n")
                f.write("-" * 80 + "\n")
                recent_trades = self.memory.get_recent_trades(10)
                if recent_trades:
                    for i, trade in enumerate(recent_trades):
                        f.write(f"Trade {i+1}: {trade.get('direction', 'Unknown')} - {'WIN' if trade.get('is_win') else 'LOSS' if trade.get('is_loss') is not None else 'OPEN'}\n")
                else:
                    f.write("No recent trades\n")
                
            logger.info(f"Trading report generated: {report_file}")
            return report_file
        except Exception as e:
            logger.error(f"Error generating trading report: {e}")
            return None
    
    def run(self):
        """Run the trading bot continuously"""
        logger.info("Starting EUR/USD Trading Bot")
        
        review_counter = 0
        try:
            while True:
                try:
                    # Run trading cycle
                    self.trading_cycle()
                    
                    # Periodic system review and evolution
                    review_counter += 1
                    if review_counter >= 12:  # Review after every 12 cycles
                        account_data = self.oanda.get_account()
                        self.brain.review_and_evolve(account_data)
                        review_counter = 0
                    
                    # Sleep between cycles
                    logger.info(f"Cycle complete. Sleeping for 5 minutes.")
                    time.sleep(300)  # 5 minutes between cycles
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    time.sleep(60)  # Wait a minute on error
                    
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")

# --- Margin Calculation Utilities ---
def calculate_forex_margin(units, price, leverage=50):
    """
    Calculate required margin for a forex position
    
    Args:
        units: Number of units (base currency)
        price: Current price
        leverage: Leverage ratio (default: 50:1)
        
    Returns:
        Required margin in account currency
    """
    position_value = units * price
    required_margin = position_value / leverage
    return required_margin

# --- Run the bot ---
if __name__ == "__main__":
    try:
        print("=" * 80)
        print("Self-Evolving LLM Forex Trading Bot for EUR/USD")
        print("=" * 80)
        print("\nInitializing trading bot...")
        bot = EURUSDTradingBot()
        print("\nTrading bot initialized successfully. Starting trading cycles...")
        print("\nPress Ctrl+C to stop the bot\n")
        bot.run()
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user. Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error in trading bot: {e}")
        print(f"\nFatal error: {e}")
        print("\nTrading bot stopped due to an error. Check the logs for details.")