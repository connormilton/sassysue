#!/usr/bin/env python3
"""
Refactored Self-Evolving Forex Trading System
--------------------------------------------
A trading system that uses multiple strategies, learns from past performance,
and evolves its approach to target consistent returns with strict risk management.

This refactored version improves:
1. Separation of concerns
2. Clear interfaces between components
3. Better error handling
4. Improved testability
5. Cleaner code organization
"""

import datetime
import json
import logging
import os
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("forex_trading_system")


# ====================================
# Enums and Constants
# ====================================

class MarketCondition(Enum):
    """Market condition classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT_POTENTIAL = "breakout_potential"
    POST_NEWS = "post_news"
    LOW_VOLATILITY = "low_volatility"


class OrderType(Enum):
    """Order types for trade execution"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderDirection(Enum):
    """Order directions (long/short)"""
    LONG = "long"
    SHORT = "short"


# ====================================
# Core Models
# ====================================

class Trade:
    """
    Represents a single trade with entry/exit details, risk management,
    and performance tracking.
    """
    
    def __init__(self, 
                 direction: OrderDirection,
                 entry_price: float,
                 stop_loss: float,
                 take_profit: float,
                 entry_time: datetime.datetime,
                 position_size: float,
                 strategy_name: str,
                 trade_id: str = None):
        """Initialize a new trade"""
        self.direction = direction
        self.entry_price = entry_price
        self.current_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = entry_time
        self.exit_time = None
        self.position_size = position_size  # In units of base currency
        self.strategy_name = strategy_name
        self.trade_id = trade_id or f"{strategy_name}_{entry_time.strftime('%Y%m%d%H%M%S')}"
        self.status = "open"
        self.exit_price = None
        self.profit_loss = 0
        self.profit_loss_pct = 0
        self.max_favorable_excursion = 0  # Maximum profit during trade
        self.max_adverse_excursion = 0  # Maximum drawdown during trade
        self.partial_exits = []  # Track partial profit taking
        self.notes = []
        self.reason_for_entry = ""
        self.reason_for_exit = ""
        self.market_context = ""
        
    def update(self, current_price: float, current_time: datetime.datetime) -> str:
        """Update trade with current price information"""
        self.current_price = current_price
        
        # Calculate current P&L
        if self.direction == OrderDirection.LONG:
            price_change = current_price - self.entry_price
        else:
            price_change = self.entry_price - current_price
            
        self.profit_loss = price_change * self.position_size
        self.profit_loss_pct = price_change / self.entry_price * 100
        
        # Track maximum favorable/adverse excursions
        if self.profit_loss > 0 and self.profit_loss > self.max_favorable_excursion:
            self.max_favorable_excursion = self.profit_loss
        elif self.profit_loss < 0 and abs(self.profit_loss) > abs(self.max_adverse_excursion):
            self.max_adverse_excursion = self.profit_loss
            
        # Check for stop loss or take profit
        if self.should_stop_out(current_price):
            self.close(current_price, current_time, "stop_loss")
            return "stopped_out"
        
        if self.should_take_profit(current_price):
            self.close(current_price, current_time, "take_profit")
            return "profit_taken"
            
        return "open"
    
    def should_stop_out(self, current_price: float) -> bool:
        """Check if the trade should be stopped out"""
        if self.direction == OrderDirection.LONG:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss
    
    def should_take_profit(self, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        if self.direction == OrderDirection.LONG:
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit
    
    def close(self, exit_price: float, exit_time: datetime.datetime, reason: str) -> None:
        """Close the trade"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = "closed"
        self.reason_for_exit = reason
        
        # Calculate final P&L
        if self.direction == OrderDirection.LONG:
            price_change = exit_price - self.entry_price
        else:
            price_change = self.entry_price - exit_price
            
        self.profit_loss = price_change * self.position_size
        self.profit_loss_pct = price_change / self.entry_price * 100
        
        logger.info(f"Trade {self.trade_id} closed: {self.profit_loss_pct:.2f}% profit/loss")
    
    def partial_exit(self, percentage: float, current_price: float, 
                    current_time: datetime.datetime, reason: str) -> float:
        """Take partial profits on the position"""
        amount_to_close = self.position_size * percentage
        self.position_size -= amount_to_close
        
        if self.direction == OrderDirection.LONG:
            price_change = current_price - self.entry_price
        else:
            price_change = self.entry_price - current_price
            
        profit_on_partial = price_change * amount_to_close
        
        self.partial_exits.append({
            "time": current_time,
            "price": current_price,
            "amount": amount_to_close,
            "profit": profit_on_partial,
            "reason": reason
        })
        
        self.notes.append(f"Partial exit: {reason} at {current_price}")
        logger.info(f"Partial exit on {self.trade_id}: {percentage*100:.0f}% of position closed")
        return profit_on_partial
    
    def move_stop_loss(self, new_stop: float, reason: str) -> None:
        """Move the stop loss level"""
        old_stop = self.stop_loss
        self.stop_loss = new_stop
        self.notes.append(f"Stop moved from {old_stop} to {new_stop}: {reason}")
        
    def to_dict(self) -> Dict:
        """Convert trade to dictionary for logging and analysis"""
        return {
            "trade_id": self.trade_id,
            "strategy": self.strategy_name,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "position_size": self.position_size,
            "status": self.status,
            "exit_price": self.exit_price,
            "profit_loss": self.profit_loss,
            "profit_loss_pct": self.profit_loss_pct,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "partial_exits": self.partial_exits,
            "notes": self.notes,
            "reason_for_entry": self.reason_for_entry,
            "reason_for_exit": self.reason_for_exit,
            "market_context": self.market_context
        }


# ====================================
# Strategy Base Class
# ====================================

class TradingStrategy:
    """
    Abstract base class for all trading strategies.
    Defines the interface that all concrete strategies must implement.
    """
    
    def __init__(self, name: str, description: str):
        """Initialize the strategy with name and description"""
        self.name = name
        self.description = description
        self.trades = []
        self.performance_history = []
        self.success_rate = 0
        self.avg_win = 0
        self.avg_loss = 0
        self.profit_factor = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.weight = 1.0  # Dynamic weight for strategy selection
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze market data to determine if entry conditions are met
        Returns dict with entry signal information or None
        """
        raise NotImplementedError("Each strategy must implement analyze_market")
    
    def should_exit(self, trade: Trade, market_data: pd.DataFrame) -> Dict:
        """
        Determine if an open trade should be exited based on market conditions
        Returns dict with exit information or None
        """
        raise NotImplementedError("Each strategy must implement should_exit")
    
    def should_adjust_stops(self, trade: Trade, market_data: pd.DataFrame) -> Optional[float]:
        """
        Check if stop losses should be adjusted
        Returns new stop level or None
        """
        raise NotImplementedError("Each strategy must implement should_adjust_stops")
    
    def calculate_position_size(self, account_size: float, risk_per_trade: float, 
                                entry_price: float, stop_loss: float, direction: OrderDirection) -> float:
        """Calculate appropriate position size based on risk parameters"""
        risk_amount = account_size * risk_per_trade
        
        if direction == OrderDirection.LONG:
            pip_risk = entry_price - stop_loss
        else:
            pip_risk = stop_loss - entry_price
            
        if pip_risk <= 0:
            raise ValueError("Invalid stop loss level")
            
        position_size = risk_amount / pip_risk
        return position_size
    
    def update_performance_metrics(self) -> None:
        """Update strategy performance metrics based on trade history"""
        closed_trades = [t for t in self.trades if t.status == "closed"]
        self.total_trades = len(closed_trades)
        
        if self.total_trades == 0:
            return
            
        winning_trades = [t for t in closed_trades if t.profit_loss > 0]
        losing_trades = [t for t in closed_trades if t.profit_loss <= 0]
        
        self.winning_trades = len(winning_trades)
        self.success_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        self.avg_win = np.mean([t.profit_loss_pct for t in winning_trades]) if winning_trades else 0
        self.avg_loss = np.mean([abs(t.profit_loss_pct) for t in losing_trades]) if losing_trades else 0
        
        total_gains = sum([t.profit_loss for t in winning_trades]) if winning_trades else 0
        total_losses = sum([abs(t.profit_loss) for t in losing_trades]) if losing_trades else 0
        
        self.profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        # Save performance snapshot
        self.performance_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "success_rate": self.success_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades
        })
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            "name": self.name,
            "success_rate": self.success_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades
        }


# ====================================
# Technical Analysis Utilities
# ====================================

class TechnicalAnalysis:
    """
    Utility class for common technical analysis calculations.
    """
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands - returns middle, upper, lower bands"""
        middle = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return middle, upper, lower
    
    @staticmethod
    def identify_market_condition(market_data: pd.DataFrame) -> MarketCondition:
        """Determine the current market condition"""
        # Calculate indicators
        market_data['sma20'] = market_data['close'].rolling(window=20).mean()
        market_data['sma50'] = market_data['close'].rolling(window=50).mean()
        market_data['atr'] = TechnicalAnalysis.calculate_atr(market_data)
        market_data['atr_pct'] = market_data['atr'] / market_data['close'] * 100
        
        # Get recent data points
        recent = market_data.iloc[-5:]
        current = market_data.iloc[-1]
        
        # Check for volatility
        avg_atr_pct = recent['atr_pct'].mean()
        current_atr_pct = current['atr_pct']
        high_volatility = current_atr_pct > avg_atr_pct * 1.5
        low_volatility = current_atr_pct < avg_atr_pct * 0.5
        
        # Check for trend
        trend_up = (current['sma20'] > current['sma50']) and (current['close'] > current['sma20'])
        trend_down = (current['sma20'] < current['sma50']) and (current['close'] < current['sma20'])
        
        # Check for range
        recent_range = recent['high'].max() - recent['low'].min()
        range_condition = recent_range < avg_atr_pct * 3 * recent['close'].mean() / 100
        
        # Determine condition
        if high_volatility:
            return MarketCondition.VOLATILE
        elif trend_up and not low_volatility:
            return MarketCondition.TRENDING_UP
        elif trend_down and not low_volatility:
            return MarketCondition.TRENDING_DOWN
        elif range_condition:
            return MarketCondition.RANGING
        elif low_volatility:
            return MarketCondition.LOW_VOLATILITY
        else:
            return MarketCondition.BREAKOUT_POTENTIAL


# ====================================
# Concrete Strategy Implementations
# ====================================

class TrendFollowingStrategy(TradingStrategy):
    """
    Strategy that identifies and follows established market trends
    using moving average crossovers.
    """
    
    def __init__(self):
        """Initialize the trend following strategy"""
        super().__init__("Trend_Following", "Identifies and follows established trends")
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """Identify trend direction and potential entry points"""
        # Calculate indicators
        market_data['sma20'] = market_data['close'].rolling(window=20).mean()
        market_data['sma50'] = market_data['close'].rolling(window=50).mean()
        market_data['atr'] = TechnicalAnalysis.calculate_atr(market_data)
        
        # Get latest values
        current = market_data.iloc[-1]
        previous = market_data.iloc[-2]
        
        # Check for trend conditions
        uptrend = (current['sma20'] > current['sma50']) and (previous['sma20'] > previous['sma50'])
        downtrend = (current['sma20'] < current['sma50']) and (previous['sma20'] < previous['sma50'])
        
        # Entry conditions
        entry_signal = None
        if uptrend and current['close'] > current['sma20'] and previous['close'] <= previous['sma20']:
            # Long entry: Price crosses above 20 SMA in uptrend
            stop_level = current['close'] - current['atr'] * 1.5
            target_level = current['close'] + current['atr'] * 3
            
            entry_signal = {
                "direction": OrderDirection.LONG,
                "entry_price": current['close'],
                "stop_loss": stop_level,
                "take_profit": target_level,
                "reason": "Price crossed above 20 SMA in established uptrend"
            }
            
        elif downtrend and current['close'] < current['sma20'] and previous['close'] >= previous['sma20']:
            # Short entry: Price crosses below 20 SMA in downtrend
            stop_level = current['close'] + current['atr'] * 1.5
            target_level = current['close'] - current['atr'] * 3
            
            entry_signal = {
                "direction": OrderDirection.SHORT,
                "entry_price": current['close'],
                "stop_loss": stop_level,
                "take_profit": target_level,
                "reason": "Price crossed below 20 SMA in established downtrend"
            }
            
        return entry_signal
    
    def should_exit(self, trade: Trade, market_data: pd.DataFrame) -> Dict:
        """Check for trend reversal exit conditions"""
        # Calculate indicators
        market_data['sma20'] = market_data['close'].rolling(window=20).mean()
        
        # Get latest values
        current = market_data.iloc[-1]
        previous = market_data.iloc[-2]
        
        exit_signal = None
        
        # Exit long if price crosses below 20 SMA
        if (trade.direction == OrderDirection.LONG and 
            current['close'] < current['sma20'] and 
            previous['close'] >= previous['sma20']):
            
            exit_signal = {
                "exit_price": current['close'],
                "reason": "Price crossed below 20 SMA, potential trend reversal"
            }
            
        # Exit short if price crosses above 20 SMA
        elif (trade.direction == OrderDirection.SHORT and 
              current['close'] > current['sma20'] and 
              previous['close'] <= previous['sma20']):
            
            exit_signal = {
                "exit_price": current['close'],
                "reason": "Price crossed above 20 SMA, potential trend reversal"
            }
            
        return exit_signal
    
    def should_adjust_stops(self, trade: Trade, market_data: pd.DataFrame) -> Optional[float]:
        """Move stop loss to break-even or trailing stop"""
        current = market_data.iloc[-1]
        
        # Calculate the price movement in favor of our trade
        if trade.direction == OrderDirection.LONG:
            favorable_move = current['close'] - trade.entry_price
            risk = trade.entry_price - trade.stop_loss
        else:
            favorable_move = trade.entry_price - current['close']
            risk = trade.stop_loss - trade.entry_price
            
        # If price has moved in our favor by at least the initial risk
        if favorable_move >= risk:
            # Move stop to breakeven + buffer
            if trade.direction == OrderDirection.LONG:
                new_stop = trade.entry_price + (risk * 0.1)  # Breakeven + small buffer
                
                # Only adjust if it's a tighter stop
                if new_stop > trade.stop_loss:
                    return new_stop
            else:
                new_stop = trade.entry_price - (risk * 0.1)  # Breakeven + small buffer
                
                # Only adjust if it's a tighter stop
                if new_stop < trade.stop_loss:
                    return new_stop
                    
        return None


class BreakoutStrategy(TradingStrategy):
    """
    Strategy that identifies and trades price breakouts from ranges.
    """
    
    def __init__(self):
        """Initialize the breakout strategy"""
        super().__init__("Breakout", "Identifies and trades price breakouts from ranges")
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """Identify potential breakout opportunities"""
        # Look back period for range identification
        lookback = 20
        
        # Calculate indicators
        market_data['atr'] = TechnicalAnalysis.calculate_atr(market_data)
        
        # Calculate recent range
        recent_data = market_data.iloc[-lookback:]
        resistance = recent_data['high'].max()
        support = recent_data['low'].min()
        range_height = resistance - support
        
        # Get latest values
        current = market_data.iloc[-1]
        previous = market_data.iloc[-2]
        
        # Check if range is well-defined (not too wide relative to ATR)
        range_atr_ratio = range_height / current['atr']
        is_valid_range = 2 <= range_atr_ratio <= 10  # Range should be 2-10x ATR
        
        entry_signal = None
        
        # Breakout above resistance
        if (is_valid_range and 
            current['close'] > resistance and 
            previous['close'] <= resistance):
            
            # Use ATR for stop and target levels
            stop_level = resistance - current['atr'] * 0.5
            target_level = current['close'] + range_height
            
            entry_signal = {
                "direction": OrderDirection.LONG,
                "entry_price": current['close'],
                "stop_loss": stop_level,
                "take_profit": target_level,
                "reason": f"Breakout above resistance at {resistance:.5f}"
            }
            
        # Breakdown below support
        elif (is_valid_range and 
              current['close'] < support and 
              previous['close'] >= support):
            
            # Use ATR for stop and target levels
            stop_level = support + current['atr'] * 0.5
            target_level = current['close'] - range_height
            
            entry_signal = {
                "direction": OrderDirection.SHORT,
                "entry_price": current['close'],
                "stop_loss": stop_level,
                "take_profit": target_level,
                "reason": f"Breakdown below support at {support:.5f}"
            }
            
        return entry_signal
    
    def should_exit(self, trade: Trade, market_data: pd.DataFrame) -> Dict:
        """Check for failed breakout or pullback"""
        # Failed breakout is when price returns into the range
        # We'll use entry reason to extract the breakout level
        current = market_data.iloc[-1]
        
        exit_signal = None
        
        if trade.direction == OrderDirection.LONG and "resistance at " in trade.reason_for_entry:
            # Extract resistance level from entry reason
            resistance_level = float(trade.reason_for_entry.split("resistance at ")[1])
            
            # If price moves back below resistance, exit as failed breakout
            if current['close'] < resistance_level:
                exit_signal = {
                    "exit_price": current['close'],
                    "reason": "Failed breakout, price returned below resistance"
                }
                
        elif trade.direction == OrderDirection.SHORT and "support at " in trade.reason_for_entry:
            # Extract support level from entry reason
            support_level = float(trade.reason_for_entry.split("support at ")[1])
            
            # If price moves back above support, exit as failed breakdown
            if current['close'] > support_level:
                exit_signal = {
                    "exit_price": current['close'],
                    "reason": "Failed breakdown, price returned above support"
                }
                
        return exit_signal
    
    def should_adjust_stops(self, trade: Trade, market_data: pd.DataFrame) -> Optional[float]:
        """Tighten stop when breakout gains momentum"""
        current = market_data.iloc[-1]
        
        # Calculate recent volatility
        market_data['atr'] = TechnicalAnalysis.calculate_atr(market_data)
        lookback_volatility = market_data['atr'].iloc[-5:].mean()
        
        # If we've moved in favor by at least 2x volatility
        if trade.direction == OrderDirection.LONG:
            favorable_move = current['close'] - trade.entry_price
            if favorable_move >= 2 * lookback_volatility:
                # Move stop to entry + small buffer
                new_stop = trade.entry_price + (lookback_volatility * 0.2)
                if new_stop > trade.stop_loss:
                    return new_stop
        else:
            favorable_move = trade.entry_price - current['close']
            if favorable_move >= 2 * lookback_volatility:
                # Move stop to entry - small buffer
                new_stop = trade.entry_price - (lookback_volatility * 0.2)
                if new_stop < trade.stop_loss:
                    return new_stop
                    
        return None


class MeanReversionStrategy(TradingStrategy):
    """
    Strategy that trades return to mean after extreme price movements.
    Uses Bollinger Bands and RSI to identify overbought/oversold conditions.
    """
    
    def __init__(self):
        """Initialize the mean reversion strategy"""
        super().__init__("Mean_Reversion", "Trades return to mean after extreme moves")
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """Identify overbought/oversold conditions for mean reversion"""
        # Calculate indicators
        market_data['sma20'] = market_data['close'].rolling(window=20).mean()
        market_data['stdev20'] = market_data['close'].rolling(window=20).std()
        market_data['upper_band'] = market_data['sma20'] + (market_data['stdev20'] * 2)
        market_data['lower_band'] = market_data['sma20'] - (market_data['stdev20'] * 2)
        market_data['rsi'] = TechnicalAnalysis.calculate_rsi(market_data)
        market_data['atr'] = TechnicalAnalysis.calculate_atr(market_data)
        
        # Get latest values
        current = market_data.iloc[-1]
        previous = market_data.iloc[-2]
        
        entry_signal = None
        
        # Oversold condition: Price below lower band and RSI < 30
        if (current['close'] < current['lower_band'] and 
            current['rsi'] < 30 and 
            previous['rsi'] <= 30):
            
            stop_level = current['close'] - current['atr'] * 1.5
            target_level = current['sma20'].iloc[-1]  # Target is the mean (20 SMA)
            
            entry_signal = {
                "direction": OrderDirection.LONG,
                "entry_price": current['close'],
                "stop_loss": stop_level,
                "take_profit": target_level,
                "reason": f"Oversold condition: Price below lower band, RSI {current['rsi']:.1f}"
            }
            
        # Overbought condition: Price above upper band and RSI > 70
        elif (current['close'] > current['upper_band'] and 
              current['rsi'] > 70 and
              previous['rsi'] >= 70):
            
            stop_level = current['close'] + current['atr'] * 1.5
            target_level = current['sma20'].iloc[-1]  # Target is the mean (20 SMA)
            
            entry_signal = {
                "direction": OrderDirection.SHORT,
                "entry_price": current['close'],
                "stop_loss": stop_level,
                "take_profit": target_level,
                "reason": f"Overbought condition: Price above upper band, RSI {current['rsi']:.1f}"
            }
            
        return entry_signal
    
    def should_exit(self, trade: Trade, market_data: pd.DataFrame) -> Dict:
        """Check if mean reversion has completed"""
        # Calculate 20 SMA
        market_data['sma20'] = market_data['close'].rolling(window=20).mean()
        
        # Get latest values
        current = market_data.iloc[-1]
        
        exit_signal = None
        
        # For long trades, exit when price reaches the mean
        if (trade.direction == OrderDirection.LONG and 
            current['close'] >= current['sma20']):
            
            exit_signal = {
                "exit_price": current['close'],
                "reason": "Price reached the mean (20 SMA)"
            }
            
        # For short trades, exit when price reaches the mean
        elif (trade.direction == OrderDirection.SHORT and 
              current['close'] <= current['sma20']):
            
            exit_signal = {
                "exit_price": current['close'],
                "reason": "Price reached the mean (20 SMA)"
            }
            
        return exit_signal
    
    def should_adjust_stops(self, trade: Trade, market_data: pd.DataFrame) -> Optional[float]:
        """Gradually tighten stop as price moves toward mean"""
        # Calculate 20 SMA
        market_data['sma20'] = market_data['close'].rolling(window=20).mean()
        
        # Get latest values
        current = market_data.iloc[-1]
        
        # Calculate distance to mean
        mean = current['sma20']
        
        if trade.direction == OrderDirection.LONG:
            # How far has price moved toward the mean as a percentage of total distance
            initial_distance = mean - trade.entry_price
            current_distance = mean - current['close']
            
            # If we've moved at least 50% toward the mean
            if initial_distance > 0 and current_distance < 0.5 * initial_distance:
                # Move stop to entry + 25% of the move
                move_so_far = current['close'] - trade.entry_price
                new_stop = trade.entry_price + (move_so_far * 0.25)
                
                if new_stop > trade.stop_loss:
                    return new_stop
                    
        else:  # SHORT
            # How far has price moved toward the mean as a percentage of total distance
            initial_distance = trade.entry_price - mean
            current_distance = current['close'] - mean
            
            # If we've moved at least 50% toward the mean
            if initial_distance > 0 and current_distance < 0.5 * initial_distance:
                # Move stop to entry - 25% of the move
                move_so_far = trade.entry_price - current['close']
                new_stop = trade.entry_price - (move_so_far * 0.25)
                
                if new_stop < trade.stop_loss:
                    return new_stop
                    
        return None


# ====================================
# System Evolution Framework
# ====================================

class SystemEvolution:
    """
    Handles the self-improvement and adaptation of the trading system.
    Maintains history of performance, learned lessons, and generates
    recommendations for system modifications.
    """
    
    def __init__(self):
        """Initialize the evolution system"""
        self.prompt_versions = []
        self.strategy_performance_history = []
        self.lesson_database = []
        self.market_condition_history = []
        
    def record_prompt_version(self, version_text: str, reason: str) -> None:
        """Record a new prompt version"""
        self.prompt_versions.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt": version_text,
            "reason": reason
        })
        
    def record_lesson(self, lesson_text: str, category: str, trade_id: str = None) -> None:
        """Record a lesson learned"""
        self.lesson_database.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "lesson": lesson_text,
            "category": category,
            "trade_id": trade_id
        })
        
    def record_strategy_performance(self, strategy_metrics: Dict) -> None:
        """Record strategy performance metrics"""
        self.strategy_performance_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": strategy_metrics
        })
        
    def record_market_condition(self, condition: MarketCondition, metrics: Dict) -> None:
        """Record observed market condition and associated metrics"""
        self.market_condition_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "condition": condition.value,
            "metrics": metrics
        })
        
    def analyze_strategy_performance(self) -> Dict:
        """Analyze strategy performance to determine which are working best"""
        if not self.strategy_performance_history:
            return {}
            
        # Get most recent performance data
        recent_data = self.strategy_performance_history[-10:]  # Last 10 records
        
        # Group by strategy
        strategy_data = {}
        
        for record in recent_data:
            for strategy in record["metrics"]:
                if strategy["name"] not in strategy_data:
                    strategy_data[strategy["name"]] = []
                    
                strategy_data[strategy["name"]].append({
                    "timestamp": record["timestamp"],
                    "success_rate": strategy["success_rate"],
                    "profit_factor": strategy["profit_factor"]
                })
                
        # Calculate average performance for each strategy
        strategy_rankings = {}
        
        for strategy_name, data in strategy_data.items():
            avg_success_rate = np.mean([d["success_rate"] for d in data]) if data else 0
            avg_profit_factor = np.mean([d["profit_factor"] for d in data]) if data else 0
            
            # Combined score (adjust weights as needed)
            combined_score = (avg_success_rate * 0.4) + (min(avg_profit_factor, 5) * 0.12)
            
            strategy_rankings[strategy_name] = {
                "avg_success_rate": avg_success_rate,
                "avg_profit_factor": avg_profit_factor,
                "combined_score": combined_score
            }
            
        return strategy_rankings
    
    def analyze_market_condition_performance(self) -> Dict:
        """Analyze which strategies perform best in different market conditions"""
        if not self.market_condition_history or not self.strategy_performance_history:
            return {}
            
        # Match market conditions with strategy performance by timestamp
        condition_strategy_map = {}
        
        for condition_record in self.market_condition_history:
            condition = condition_record["condition"]
            timestamp = condition_record["timestamp"]
            
            # Find the closest strategy performance record
            closest_record = min(
                self.strategy_performance_history,
                key=lambda x: abs(datetime.datetime.fromisoformat(x["timestamp"]) - 
                                 datetime.datetime.fromisoformat(timestamp))
            )
            
            if condition not in condition_strategy_map:
                condition_strategy_map[condition] = []
                
            condition_strategy_map[condition].append(closest_record["metrics"])
            
        # Analyze which strategies perform best in each condition
        condition_strategy_performance = {}
        
        for condition, performance_records in condition_strategy_map.items():
            strategy_scores = {}
            
            for record in performance_records:
                for strategy in record:
                    name = strategy["name"]
                    
                    if name not in strategy_scores:
                        strategy_scores[name] = []
                        
                    # Calculate a performance score
                    perf_score = (strategy["success_rate"] * 0.4) + (min(strategy["profit_factor"], 5) * 0.12)
                    strategy_scores[name].append(perf_score)
                    
            # Calculate average score for each strategy in this condition
            avg_scores = {
                name: np.mean(scores) if scores else 0
                for name, scores in strategy_scores.items()
            }
            
            # Sort strategies by performance in this condition
            sorted_strategies = sorted(
                avg_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            condition_strategy_performance[condition] = sorted_strategies
            
        return condition_strategy_performance
        
    def generate_daily_review(self, day_trades: List[Trade], 
                             account_performance: Dict,
                             strategies_used: List[TradingStrategy]) -> Dict:
        """Generate comprehensive daily review"""
        # Calculate daily statistics
        if not day_trades:
            return {
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "trades_taken": 0,
                "performance": account_performance,
                "summary": "No trades taken today"
            }
            
        # Trade statistics
        total_trades = len(day_trades)
        winning_trades = len([t for t in day_trades if t.profit_loss > 0])
        losing_trades = len([t for t in day_trades if t.profit_loss <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.profit_loss_pct for t in day_trades if t.profit_loss > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([abs(t.profit_loss_pct) for t in day_trades if t.profit_loss <= 0]) if losing_trades > 0 else 0
        
        best_trade = max(day_trades, key=lambda t: t.profit_loss_pct) if day_trades else None
        worst_trade = min(day_trades, key=lambda t: t.profit_loss_pct) if day_trades else None
        
        # Strategy performance
        strategy_performance = {}
        for strategy in strategies_used:
            strategy_trades = [t for t in day_trades if t.strategy_name == strategy.name]
            if not strategy_trades:
                continue
                
            wins = len([t for t in strategy_trades if t.profit_loss > 0])
            strategy_performance[strategy.name] = {
                "trades": len(strategy_trades),
                "wins": wins,
                "losses": len(strategy_trades) - wins,
                "win_rate": wins / len(strategy_trades) if strategy_trades else 0,
                "avg_profit": np.mean([t.profit_loss_pct for t in strategy_trades])
            }
            
        # Identify top performing strategies
        if strategy_performance:
            top_strategies = sorted(
                strategy_performance.items(),
                key=lambda x: x[1]["avg_profit"],
                reverse=True
            )
        else:
            top_strategies = []
        
        # Generate insights
        insights = []
        
        if avg_win and avg_loss:
            reward_risk_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            if reward_risk_ratio < 1:
                insights.append("Risk-reward ratio is unfavorable. Consider tightening stops or extending targets.")
            elif reward_risk_ratio > 2:
                insights.append("Excellent risk-reward ratio. Consider increasing position size on high-confidence setups.")
                
        if win_rate < 0.4:
            insights.append("Low win rate today. Review entry criteria for potential refinements.")
        elif win_rate > 0.6:
            insights.append("High win rate today. System is identifying high-probability setups effectively.")
            
        # Generate report
        daily_review = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "trades_taken": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "reward_risk_ratio": avg_win / avg_loss if avg_loss > 0 else float('inf'),
            "best_trade": {
                "id": best_trade.trade_id,
                "strategy": best_trade.strategy_name,
                "profit_pct": best_trade.profit_loss_pct
            } if best_trade else None,
            "worst_trade": {
                "id": worst_trade.trade_id,
                "strategy": worst_trade.strategy_name,
                "profit_pct": worst_trade.profit_loss_pct
            } if worst_trade else None,
            "strategy_performance": strategy_performance,
            "top_strategies": [s[0] for s in top_strategies[:2]] if top_strategies else [],
            "performance": account_performance,
            "insights": insights,
            "summary": self._generate_summary(day_trades, account_performance)
        }
        
        return daily_review
    
    def _generate_summary(self, day_trades: List[Trade], account_performance: Dict) -> str:
        """Generate natural language summary of the trading day"""
        if not day_trades:
            return "No trades were taken today."
            
        # Basic stats
        total_trades = len(day_trades)
        winning_trades = len([t for t in day_trades if t.profit_loss > 0])
        net_profit_pct = account_performance.get("daily_profit_pct", 0)
        
        # Generate summary
        if net_profit_pct > 0:
            tone = "profitable" if net_profit_pct < 5 else "very profitable" if net_profit_pct < 10 else "exceptionally profitable"
        else:
            tone = "slightly negative" if net_profit_pct > -3 else "negative" if net_profit_pct > -7 else "very challenging"
            
        summary = f"Today was a {tone} trading day with {net_profit_pct:.2f}% net profit. "
        summary += f"A total of {total_trades} trades were taken with {winning_trades} winners. "
        
        # Add strategy info
        strategies_used = set([t.strategy_name for t in day_trades])
        if len(strategies_used) == 1:
            summary += f"All trades were taken using the {list(strategies_used)[0]} strategy. "
        else:
            top_strategy = max(
                strategies_used,
                key=lambda s: sum(1 for t in day_trades if t.strategy_name == s and t.profit_loss > 0)
            )
            summary += f"Multiple strategies were employed, with {top_strategy} showing the best performance. "
            
        # Add market context
        if any("trending" in t.market_context.lower() for t in day_trades if t.market_context):
            summary += "The market showed strong directional trends today. "
        elif any("range" in t.market_context.lower() for t in day_trades if t.market_context):
            summary += "The market traded in a range for most of the day. "
        elif any("volatile" in t.market_context.lower() for t in day_trades if t.market_context):
            summary += "Market volatility was elevated today. "
            
        # Add insights
        if net_profit_pct < account_performance.get("target_daily_return", 10) * 0.5:
            summary += "Performance was below the daily target. "
        elif net_profit_pct >= account_performance.get("target_daily_return", 10):
            summary += "The daily target was successfully achieved. "
            
        return summary
        
    def generate_weekly_review(self, daily_reviews: List[Dict], 
                              account_performance: Dict,
                              strategies_used: List[TradingStrategy]) -> Dict:
        """Generate comprehensive weekly review and system evolution recommendations"""
        if not daily_reviews:
            return {
                "week_ending": datetime.datetime.now().strftime("%Y-%m-%d"),
                "summary": "No trading data available for this week"
            }
            
        # Aggregate daily stats
        total_trades = sum(dr["trades_taken"] for dr in daily_reviews)
        winning_trades = sum(dr["winning_trades"] for dr in daily_reviews if "winning_trades" in dr)
        losing_trades = sum(dr["losing_trades"] for dr in daily_reviews if "losing_trades" in dr)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate weekly P&L
        weekly_profit_pct = sum(dr["performance"].get("daily_profit_pct", 0) for dr in daily_reviews)
        
        # Aggregated strategy performance
        strategy_performance = {}
        
        for dr in daily_reviews:
            for strategy_name, perf in dr.get("strategy_performance", {}).items():
                if strategy_name not in strategy_performance:
                    strategy_performance[strategy_name] = {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "total_profit_pct": 0
                    }
                    
                strategy_performance[strategy_name]["trades"] += perf["trades"]
                strategy_performance[strategy_name]["wins"] += perf["wins"]
                strategy_performance[strategy_name]["losses"] += perf["losses"]
                strategy_performance[strategy_name]["total_profit_pct"] += perf["avg_profit"] * perf["trades"]
                
        # Calculate averages and win rates
        for strategy_name, perf in strategy_performance.items():
            if perf["trades"] > 0:
                perf["win_rate"] = perf["wins"] / perf["trades"]
                perf["avg_profit_pct"] = perf["total_profit_pct"] / perf["trades"]
            else:
                perf["win_rate"] = 0
                perf["avg_profit_pct"] = 0
                
        # Rank strategies
        ranked_strategies = sorted(
            strategy_performance.items(),
            key=lambda x: (x[1]["win_rate"], x[1]["avg_profit_pct"]),
            reverse=True
        )
        
        # System evolution recommendations
        recommendations = []
        
        # Check if we're meeting target
        daily_target = account_performance.get("target_daily_return", 10)
        weekly_target = daily_target * 5  # 5 trading days
        
        if weekly_profit_pct < weekly_target:
            shortfall = weekly_target - weekly_profit_pct
            recommendations.append({
                "type": "performance_gap",
                "description": f"Weekly performance is {shortfall:.2f}% below target",
                "action": "Increase focus on top-performing strategies and optimize entry timing"
            })
            
        # Strategy-specific recommendations
        for strategy_name, perf in strategy_performance.items():
            if perf["trades"] >= 5:  # Only consider strategies with enough data
                if perf["win_rate"] < 0.4:
                    recommendations.append({
                        "type": "strategy_adjustment",
                        "strategy": strategy_name,
                        "description": f"Low win rate ({perf['win_rate']:.2f}) for {strategy_name}",
                        "action": "Tighten entry criteria or reduce allocation to this strategy"
                    })
                elif perf["win_rate"] > 0.6 and perf["avg_profit_pct"] > 0:
                    recommendations.append({
                        "type": "strategy_adjustment",
                        "strategy": strategy_name,
                        "description": f"High win rate ({perf['win_rate']:.2f}) for {strategy_name}",
                        "action": "Increase allocation to this strategy"
                    })
                    
        # Check for strategy correlation
        if len(strategy_performance) > 1:
            recommendations.append({
                "type": "diversification",
                "description": "Analyze strategy correlation to ensure diversification",
                "action": "Monitor simultaneous drawdowns to detect correlation"
            })
            
        # Generate summary
        weeks_profitable_days = sum(1 for dr in daily_reviews if dr["performance"].get("daily_profit_pct", 0) > 0)
        total_days = len(daily_reviews)
        
        summary = f"Week completed with {weekly_profit_pct:.2f}% total return ({weeks_profitable_days}/{total_days} profitable days). "
        
        if ranked_strategies:
            top_strategy = ranked_strategies[0][0]
            summary += f"The {top_strategy} strategy showed the best overall performance. "
            
        if weekly_profit_pct >= weekly_target:
            summary += f"Weekly target of {weekly_target:.2f}% was successfully achieved. "
        else:
            summary += f"Performance was {weekly_target - weekly_profit_pct:.2f}% below the weekly target. "
            
        # Generate evolution proposal
        evolution_proposal = "Based on this week's performance, the system should "
        
        if recommendations:
            if any(r["type"] == "strategy_adjustment" for r in recommendations):
                adjust_recs = [r for r in recommendations if r["type"] == "strategy_adjustment"]
                reduce_strategies = [r["strategy"] for r in adjust_recs if "reduce" in r["action"]]
                increase_strategies = [r["strategy"] for r in adjust_recs if "increase" in r["action"]]
                
                if reduce_strategies:
                    evolution_proposal += f"reduce allocation to {', '.join(reduce_strategies)} and "
                if increase_strategies:
                    evolution_proposal += f"increase allocation to {', '.join(increase_strategies)}. "
            else:
                evolution_proposal += "maintain the current strategy allocation but "
                
            if any(r["type"] == "performance_gap" for r in recommendations):
                evolution_proposal += "focus on optimizing entry timing and position sizing to close the performance gap. "
        else:
            evolution_proposal += "continue with the current approach as it's meeting performance targets. "
            
        # Compile weekly review
        weekly_review = {
            "week_ending": datetime.datetime.now().strftime("%Y-%m-%d"),
            "days_analyzed": len(daily_reviews),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "weekly_profit_pct": weekly_profit_pct,
            "target_achieved": weekly_profit_pct >= weekly_target,
            "strategy_performance": strategy_performance,
            "ranked_strategies": [s[0] for s in ranked_strategies],
            "recommendations": recommendations,
            "summary": summary,
            "evolution_proposal": evolution_proposal
        }
        
        return weekly_review


# ====================================
# Risk Management
# ====================================

class RiskManager:
    """
    Manages trading risk, position sizing, and account exposure.
    Implements various risk control mechanisms and monitors account health.
    """
    
    def __init__(self, 
                max_daily_risk: float = 0.10,  # 10% max account risk
                max_daily_drawdown: float = 0.10,  # 10% max daily drawdown
                initial_safety_level: float = 0.01  # 1% risk per trade initially
                ):
        """Initialize the risk manager"""
        self.max_daily_risk = max_daily_risk
        self.max_daily_drawdown = max_daily_drawdown
        self.safety_level = initial_safety_level
        self.current_risk_exposure = 0.0
        
    def can_take_trade(self, daily_profit_pct: float, target_daily_return: float, 
                       current_drawdown: float) -> bool:
        """Determine if a new trade can be taken based on risk parameters"""
        # Check if we've exceeded maximum risk exposure
        if self.current_risk_exposure >= self.max_daily_risk:
            logger.info("Maximum daily risk exposure reached - no new trades")
            return False
            
        # Check if we've exceeded maximum drawdown
        if current_drawdown >= self.max_daily_drawdown:
            logger.info("Maximum daily drawdown reached - no new trades")
            return False
            
        # Check if daily target has been met
        if daily_profit_pct >= target_daily_return:
            logger.info(f"Daily target of {target_daily_return:.2f}% achieved. Being selective with new trades.")
            # We still allow trades, but will be more selective
            return True
            
        return True
        
    def calculate_position_size(self, account_balance: float, strategy: TradingStrategy,
                              entry_price: float, stop_loss: float, direction: OrderDirection) -> float:
        """Calculate the appropriate position size based on risk parameters"""
        # Use the strategy's position sizing method with our safety level
        return strategy.calculate_position_size(
            account_balance, 
            self.safety_level,
            entry_price,
            stop_loss,
            direction
        )
        
    def update_risk_exposure(self, active_trades: List[Trade], account_balance: float) -> None:
        """Update the current risk exposure based on active trades"""
        self.current_risk_exposure = sum(self._calculate_trade_risk(t, account_balance) for t in active_trades)
        
    def adjust_safety_level(self, trade_profitable: bool) -> None:
        """Adjust safety level based on trade outcome"""
        if trade_profitable:
            # Increase safety level after profitable trade (up to max of 5%)
            self.safety_level = min(0.05, self.safety_level + 0.0075)
            logger.info(f"Safety level increased to {self.safety_level:.4f} after profitable trade")
            
    def _calculate_trade_risk(self, trade: Trade, account_balance: float) -> float:
        """Calculate the current risk exposure of a trade"""
        if trade.direction == OrderDirection.LONG:
            risk_pct = (trade.entry_price - trade.stop_loss) / trade.entry_price
        else:
            risk_pct = (trade.stop_loss - trade.entry_price) / trade.entry_price
            
        # Calculate actual account risk based on position size
        account_risk = risk_pct * (trade.position_size * trade.entry_price) / account_balance
        
        return account_risk


# ====================================
# Trading System
# ====================================

class TradingSystem:
    """
    Core trading system that integrates all components and manages the
    trading workflow.
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """Initialize the trading system"""
        # Account parameters
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.target_daily_return = 0.10  # 10% daily return target
        
        # Performance tracking
        self.daily_high_balance = initial_balance
        self.daily_low_balance = initial_balance
        self.daily_starting_balance = initial_balance
        self.current_daily_drawdown = 0.0
        self.daily_profit = 0.0
        self.daily_profit_pct = 0.0
        
        # Trading state
        self.active_trades = []
        self.closed_trades = []
        self.daily_trades = []
        self.daily_reviews = []
        self.weekly_reviews = []
        
        # Components
        self.risk_manager = RiskManager()
        self.evolution = SystemEvolution()
        
        # Trading strategies
        self.strategies = [
            TrendFollowingStrategy(),
            BreakoutStrategy(),
            MeanReversionStrategy()
        ]
        self.strategy_weights = {s.name: 1.0 for s in self.strategies}
        
        # Market condition tracking
        self.current_market_condition = None
        
        # Initialize tracking
        self.last_daily_reset = datetime.datetime.now().date()
        self.last_weekly_review = datetime.datetime.now().date()
        
        logger.info(f"Trading System initialized with balance: ${initial_balance:.2f}")
        
    def update(self, market_data: pd.DataFrame, current_time: datetime.datetime) -> str:
        """
        Main update function - analyzes market, manages trades, evolves system
        """
        # Check if we need to reset daily stats
        self._check_daily_reset(current_time)
        
        # Check if we need to perform weekly review
        self._check_weekly_review(current_time)
        
        # Update all active trades with current prices
        current_price = market_data.iloc[-1]['close']
        self._update_active_trades(current_price, current_time)
        
        # Update strategies based on market data
        self.update_strategies_based_on_market(market_data)
        
        # Check for partial profit taking opportunities
        self.check_for_partial_profits(market_data)
        
        # Analyze market condition
        market_condition = TechnicalAnalysis.identify_market_condition(market_data)
        self.current_market_condition = market_condition
        
        # Record market condition for evolution
        self.evolution.record_market_condition(
            market_condition,
            {
                "price": current_price,
                "time": current_time.isoformat(),
                "active_trades": len(self.active_trades)
            }
        )
        
        # Update risk exposure
        self.risk_manager.update_risk_exposure(self.active_trades, self.current_balance)
        
        # Check if we can take new trades
        if self.risk_manager.can_take_trade(
            self.daily_profit_pct, 
            self.target_daily_return,
            self.current_daily_drawdown
        ):
            # Evaluate strategies for potential new trades
            self._evaluate_strategies(market_data, current_time)
            
        # Return status
        if self.current_daily_drawdown >= self.risk_manager.max_daily_drawdown:
            return "max_drawdown_reached"
        else:
            return "normal_update"
    
    def _update_active_trades(self, current_price: float, current_time: datetime.datetime) -> None:
        """Update all active trades with current market data"""
        for trade in self.active_trades[:]:  # Use a copy to allow removal during iteration
            result = trade.update(current_price, current_time)
            
            if result != "open":
                # Trade has closed
                self.risk_manager.update_risk_exposure(self.active_trades, self.current_balance)
                
                if trade.profit_loss > 0:
                    # Increase safety level on profitable trade
                    self.risk_manager.adjust_safety_level(True)
                
                # Update account balance
                self.current_balance += trade.profit_loss
                
                # Update daily tracking
                if self.current_balance > self.daily_high_balance:
                    self.daily_high_balance = self.current_balance
                if self.current_balance < self.daily_low_balance:
                    self.daily_low_balance = self.current_balance
                    
                # Update daily drawdown
                self.current_daily_drawdown = max(
                    self.current_daily_drawdown,
                    (self.daily_high_balance - self.current_balance) / self.daily_high_balance * 100
                )
                
                # Update daily profit
                self.daily_profit = self.current_balance - self.daily_starting_balance
                self.daily_profit_pct = (self.daily_profit / self.daily_starting_balance) * 100
                
                # Move to closed trades
                self.active_trades.remove(trade)
                self.closed_trades.append(trade)
                self.daily_trades.append(trade)
                
                # Perform post-trade review
                self._post_trade_review(trade)
    
    def update_strategies_based_on_market(self, market_data: pd.DataFrame) -> None:
        """Update strategies with current market data for exit checks"""
        # Check for exit signals on active trades
        for trade in self.active_trades[:]:  # Use a copy to allow removal during iteration
            # Find the strategy for this trade
            strategy = None
            for s in self.strategies:
                if s.name == trade.strategy_name:
                    strategy = s
                    break
                    
            if strategy:
                # Check if we should exit
                exit_signal = strategy.should_exit(trade, market_data)
                if exit_signal:
                    current_price = market_data.iloc[-1]['close']
                    current_time = market_data.index[-1]
                    
                    if isinstance(current_time, pd.Timestamp):
                        current_time = current_time.to_pydatetime()
                    
                    # Close the trade
                    logger.info(f"Exit signal for {trade.trade_id}: {exit_signal['reason']}")
                    trade.close(
                        exit_price=exit_signal.get('exit_price', current_price),
                        exit_time=current_time,
                        reason=exit_signal['reason']
                    )
                    
                    # Update account balance and risk
                    self.current_balance += trade.profit_loss
                    self.risk_manager.update_risk_exposure(self.active_trades, self.current_balance)
                    
                    # Update trade lists
                    self.active_trades.remove(trade)
                    self.closed_trades.append(trade)
                    self.daily_trades.append(trade)
                    
                    # Perform post-trade review
                    self._post_trade_review(trade)
                
                # Check if we should adjust stops
                new_stop = strategy.should_adjust_stops(trade, market_data)
                if new_stop:
                    old_stop = trade.stop_loss
                    trade.move_stop_loss(new_stop, f"Adjusted stop from {old_stop:.5f} to {new_stop:.5f}")
                    logger.info(f"Adjusted stop for {trade.trade_id} from {old_stop:.5f} to {new_stop:.5f}")
    
    def check_for_partial_profits(self, market_data: pd.DataFrame) -> None:
        """Check if we should take partial profits on any trades"""
        current_price = market_data.iloc[-1]['close']
        current_time = market_data.index[-1]
        
        if isinstance(current_time, pd.Timestamp):
            current_time = current_time.to_pydatetime()
        
        for trade in self.active_trades:
            # Check profit level
            if trade.direction == OrderDirection.LONG:
                profit_pct = (current_price - trade.entry_price) / trade.entry_price
            else:
                profit_pct = (trade.entry_price - current_price) / trade.entry_price
                
            # Take partial profits at different levels
            if profit_pct >= 0.03 and not any("25% partial exit" in note for note in trade.notes):
                # Take 25% profits at 3% gain
                profit = trade.partial_exit(0.25, current_price, current_time, "25% partial exit at 3% gain")
                self.current_balance += profit
                logger.info(f"Took 25% partial profits on {trade.trade_id} at {current_price:.5f}")
                
            elif profit_pct >= 0.05 and not any("50% partial exit" in note for note in trade.notes):
                # Take another 25% (50% total) at 5% gain
                profit = trade.partial_exit(0.25, current_price, current_time, "50% partial exit at 5% gain")
                self.current_balance += profit
                logger.info(f"Took additional 25% partial profits on {trade.trade_id} at {current_price:.5f}")
                
            elif profit_pct >= 0.07 and not any("75% partial exit" in note for note in trade.notes):
                # Take another 25% (75% total) at 7% gain
                profit = trade.partial_exit(0.25, current_price, current_time, "75% partial exit at 7% gain")
                self.current_balance += profit
                logger.info(f"Took additional 25% partial profits on {trade.trade_id} at {current_price:.5f}")
    
    def _evaluate_strategies(self, market_data: pd.DataFrame, current_time: datetime.datetime) -> None:
        """Evaluate all strategies for potential trades"""
        # Adjust strategy weights based on market condition
        self._adjust_strategy_weights()
        
        trade_signals = []
        
        # Get signals from all strategies
        for strategy in self.strategies:
            signal = strategy.analyze_market(market_data)
            if signal:
                signal["strategy"] = strategy
                trade_signals.append(signal)
                
        # If we have signals, rank and execute best one
        if trade_signals:
            # Rank signals by strategy weight
            ranked_signals = sorted(
                trade_signals,
                key=lambda s: self.strategy_weights[s["strategy"].name],
                reverse=True
            )
            
            # Try to execute the best signal
            best_signal = ranked_signals[0]
            trade = self._execute_trade(best_signal, market_data, current_time)
            
            if trade:
                logger.info(f"New trade opened: {trade.trade_id} using {trade.strategy_name}")
    
    def _adjust_strategy_weights(self) -> None:
        """Adjust strategy weights based on market condition and performance"""
        # Get strategy performance analysis
        strategy_rankings = self.evolution.analyze_strategy_performance()
        if not strategy_rankings:
            return
            
        # Get market condition-specific performance
        condition_strategy_performance = self.evolution.analyze_market_condition_performance()
        
        # Start with base weights
        base_weights = {s.name: 1.0 for s in self.strategies}
        
        # Adjust based on overall performance
        for strategy_name, metrics in strategy_rankings.items():
            base_weights[strategy_name] = max(0.5, min(2.0, metrics["combined_score"] * 2))
            
        # Further adjust based on current market condition
        if self.current_market_condition and self.current_market_condition.value in condition_strategy_performance:
            condition_rankings = condition_strategy_performance[self.current_market_condition.value]
            
            for strategy_name, score in condition_rankings:
                # Boost weight for strategies that work well in current condition
                multiplier = 1.0 + (score * 0.5)  # Boost by up to 50%
                if strategy_name in base_weights:
                    base_weights[strategy_name] *= multiplier
                    
        # Normalize weights to prevent excessively large differences
        max_weight = max(base_weights.values()) if base_weights else 1.0
        for strategy_name in base_weights:
            base_weights[strategy_name] /= max_weight
            
        # Update system weights
        self.strategy_weights = base_weights
        
        logger.info(f"Strategy weights adjusted: {self.strategy_weights}")
    
    def _execute_trade(self, signal: Dict, market_data: pd.DataFrame, current_time: datetime.datetime) -> Optional[Trade]:
        """Execute a trade based on a strategy signal"""
        # Calculate position size
        try:
            position_size = self.risk_manager.calculate_position_size(
                self.current_balance,
                signal["strategy"],
                signal["entry_price"],
                signal["stop_loss"],
                signal["direction"]
            )
        except ValueError as e:
            logger.warning(f"Invalid trade signal: {e}")
            return None
            
        # Create the trade
        trade = Trade(
            direction=signal["direction"],
            entry_price=signal["entry_price"],
            stop_loss=signal["stop_loss"],
            take_profit=signal["take_profit"],
            entry_time=current_time,
            position_size=position_size,
            strategy_name=signal["strategy"].name
        )
        
        # Add reason and context
        trade.reason_for_entry = signal["reason"]
        trade.market_context = f"Market condition: {self.current_market_condition.value}"
        
        # Add to active trades
        self.active_trades.append(trade)
        
        # Update risk exposure
        self.risk_manager.update_risk_exposure(self.active_trades, self.current_balance)
        
        # Add trade to strategy's trade list
        for strategy in self.strategies:
            if strategy.name == trade.strategy_name:
                strategy.trades.append(trade)
                break
                
        return trade
    
    def _post_trade_review(self, trade: Trade) -> None:
        """Analyze a completed trade and extract lessons"""
        # Record basic trade info
        logger.info(f"Post-trade review: {trade.trade_id}, Result: {trade.profit_loss_pct:.2f}%")
        
        # Extract lessons
        lessons = []
        
        # Check entry timing
        if trade.profit_loss < 0:
            lessons.append(f"Trade {trade.trade_id} resulted in loss. Review entry timing and stop placement.")
        else:
            lessons.append(f"Successful trade with {trade.profit_loss_pct:.2f}% profit.")
            
        # Check max adverse excursion
        if trade.max_adverse_excursion < 0:
            mae_pct = abs(trade.max_adverse_excursion) / (trade.position_size * trade.entry_price) * 100
            if mae_pct > 75:  # If drawdown was more than 75% of risk
                lessons.append(f"Trade experienced significant drawdown ({mae_pct:.1f}% of risk) before turning profitable.")
                
        # Check partial exits
        if trade.partial_exits:
            total_partial = sum(pe["profit"] for pe in trade.partial_exits)
            lessons.append(f"Partial exits captured {total_partial:.2f} profit.")
            
        # Record lessons in evolution system
        for lesson in lessons:
            self.evolution.record_lesson(
                lesson_text=lesson,
                category="trade_execution" if trade.profit_loss < 0 else "trade_success",
                trade_id=trade.trade_id
            )
            
        # Update strategy performance metrics
        for strategy in self.strategies:
            if strategy.name == trade.strategy_name:
                strategy.update_performance_metrics()
                break
                
        # Record strategy performance
        strategy_metrics = [s.get_performance_metrics() for s in self.strategies]
        self.evolution.record_strategy_performance(strategy_metrics)
    
    def _check_daily_reset(self, current_time: datetime.datetime) -> None:
        """Check if we need to reset daily stats"""
        current_date = current_time.date()
        
        if current_date > self.last_daily_reset:
            # It's a new day - perform daily reset and review
            logger.info(f"Performing daily reset for {self.last_daily_reset}")
            
            # Generate daily review for previous day
            daily_review = self.evolution.generate_daily_review(
                self.daily_trades,
                {
                    "starting_balance": self.daily_starting_balance,
                    "ending_balance": self.current_balance,
                    "high_balance": self.daily_high_balance,
                    "low_balance": self.daily_low_balance,
                    "daily_profit": self.daily_profit,
                    "daily_profit_pct": self.daily_profit_pct,
                    "max_drawdown": self.current_daily_drawdown,
                    "target_daily_return": self.target_daily_return
                },
                self.strategies
            )
            
            # Store the review
            self.daily_reviews.append(daily_review)
            
            # Log summary
            logger.info(f"Daily review: {daily_review['summary']}")
            
            # Reset daily tracking
            self.daily_starting_balance = self.current_balance
            self.daily_high_balance = self.current_balance
            self.daily_low_balance = self.current_balance
            self.daily_profit = 0.0
            self.daily_profit_pct = 0.0
            self.current_daily_drawdown = 0.0
            self.daily_trades = []
            
            # Reset risk tracking
            self.risk_manager.update_risk_exposure(self.active_trades, self.current_balance)
            
            # Update last reset date
            self.last_daily_reset = current_date
    
    def _check_weekly_review(self, current_time: datetime.datetime) -> None:
        """Check if we need to perform weekly review"""
        current_date = current_time.date()
        
        # Check if it's a Friday and we haven't done a weekly review yet
        is_friday = current_time.weekday() == 4  # 0 is Monday, 4 is Friday
        
        if is_friday and current_date > self.last_weekly_review:
            # Perform weekly review
            logger.info("Performing weekly review")
            
            # Get recent daily reviews
            recent_reviews = self.daily_reviews[-5:]  # Up to 5 most recent days
            
            if recent_reviews:
                # Generate weekly review
                weekly_review = self.evolution.generate_weekly_review(
                    recent_reviews,
                    {
                        "starting_balance": recent_reviews[0]["performance"]["starting_balance"],
                        "ending_balance": self.current_balance,
                        "weekly_profit": self.current_balance - recent_reviews[0]["performance"]["starting_balance"],
                        "weekly_profit_pct": (self.current_balance / recent_reviews[0]["performance"]["starting_balance"] - 1) * 100,
                        "target_daily_return": self.target_daily_return
                    },
                    self.strategies
                )
                
                # Store the review
                self.weekly_reviews.append(weekly_review)
                
                # Log summary
                logger.info(f"Weekly review: {weekly_review['summary']}")
                
                # Implement recommendations
                self._implement_weekly_recommendations(weekly_review)
                
            # Update last weekly review date
            self.last_weekly_review = current_date
    
    def _implement_weekly_recommendations(self, weekly_review: Dict) -> None:
        """Implement recommendations from weekly review"""
        if not weekly_review.get("recommendations"):
            return
            
        logger.info("Implementing weekly recommendations")
        
        for recommendation in weekly_review["recommendations"]:
            if recommendation["type"] == "strategy_adjustment":
                strategy_name = recommendation["strategy"]
                action = recommendation["action"]
                
                # Find the strategy
                for strategy in self.strategies:
                    if strategy.name == strategy_name:
                        if "increase" in action.lower():
                            # Boost the base weight
                            self.strategy_weights[strategy_name] *= 1.25
                            logger.info(f"Increased weight for {strategy_name} strategy")
                        elif "reduce" in action.lower():
                            # Reduce the base weight
                            self.strategy_weights[strategy_name] *= 0.75
                            logger.info(f"Reduced weight for {strategy_name} strategy")
                        break
                        
        # Record the system evolution
        self.evolution.record_prompt_version(
            f"Strategy weights adjusted to {self.strategy_weights}",
            "Weekly review recommendations"
        )
        
    def self_modify(self, modification_proposal: Dict) -> None:
        """Apply self-modification based on system evolution"""
        if "strategy_weights" in modification_proposal:
            self.strategy_weights = modification_proposal["strategy_weights"]
            logger.info(f"Self-modified strategy weights: {self.strategy_weights}")
            
        if "safety_level" in modification_proposal:
            # Ensure safety level stays within reasonable bounds
            new_safety = min(0.05, max(0.005, modification_proposal["safety_level"]))
            self.risk_manager.safety_level = new_safety
            logger.info(f"Self-modified safety level to {new_safety}")
            
        # Record the modification
        self.evolution.record_prompt_version(
            f"System self-modification: {modification_proposal}",
            "Self-evolution based on performance analysis"
        )
        
    def generate_self_modification_proposal(self) -> Dict:
        """Generate a proposal for system self-modification"""
        # Analyze recent performance
        strategy_rankings = self.evolution.analyze_strategy_performance()
        condition_strategy_map = self.evolution.analyze_market_condition_performance()
        
        # Start with current parameters
        proposal = {
            "strategy_weights": dict(self.strategy_weights),
            "safety_level": self.risk_manager.safety_level
        }
        
        # Check if we're consistently meeting or missing targets
        recent_reviews = self.daily_reviews[-7:]  # Last week of data
        if recent_reviews:
            targets_hit = sum(1 for r in recent_reviews if r["performance"]["daily_profit_pct"] >= self.target_daily_return)
            hit_rate = targets_hit / len(recent_reviews)
            
            if hit_rate >= 0.7:
                # We're consistently hitting targets - consider increasing risk
                proposal["safety_level"] = min(0.05, self.risk_manager.safety_level * 1.1)
            elif hit_rate <= 0.3:
                # We're consistently missing targets - adjust strategy weights more aggressively
                if strategy_rankings:
                    # Increase contrast between strategy weights
                    best_strategy = max(strategy_rankings.items(), key=lambda x: x[1]["combined_score"])[0]
                    worst_strategy = min(strategy_rankings.items(), key=lambda x: x[1]["combined_score"])[0]
                    
                    proposal["strategy_weights"][best_strategy] *= 1.2
                    proposal["strategy_weights"][worst_strategy] *= 0.8
        
        # Check for opportunities to specialize in market conditions
        if condition_strategy_map and self.current_market_condition:
            condition = self.current_market_condition.value
            if condition in condition_strategy_map:
                # Get top strategies for current condition
                top_strategies = condition_strategy_map[condition][:2]  # Top 2
                
                # Boost weights for strategies that excel in current condition
                for strategy_name, _ in top_strategies:
                    if strategy_name in proposal["strategy_weights"]:
                        proposal["strategy_weights"][strategy_name] *= 1.15
        
        # Normalize weights
        total_weight = sum(proposal["strategy_weights"].values())
        if total_weight > 0:
            for strategy in proposal["strategy_weights"]:
                proposal["strategy_weights"][strategy] /= total_weight
                proposal["strategy_weights"][strategy] *= len(proposal["strategy_weights"])  # Rescale to average of 1.0
        
        return proposal
        
    def get_system_status(self) -> Dict:
        """Get current system status for monitoring"""
        return {
            "balance": self.current_balance,
            "daily_profit_pct": self.daily_profit_pct,
            "target_daily_return": self.target_daily_return,
            "daily_drawdown": self.current_daily_drawdown,
            "max_daily_drawdown": self.risk_manager.max_daily_drawdown,
            "safety_level": self.risk_manager.safety_level,
            "total_risk_committed": self.risk_manager.current_risk_exposure,
            "max_daily_risk": self.risk_manager.max_daily_risk,
            "active_trades": len(self.active_trades),
            "active_trade_ids": [t.trade_id for t in self.active_trades],
            "market_condition": self.current_market_condition.value if self.current_market_condition else None,
            "strategy_weights": self.strategy_weights
        }
        
    def get_performance_history(self) -> Dict:
        """Get performance history for analysis"""
        return {
            "daily_reviews": self.daily_reviews,
            "weekly_reviews": self.weekly_reviews,
            "strategy_performance": {s.name: s.get_performance_metrics() for s in self.strategies},
            "balance_history": self._get_balance_history(),
            "trade_history": [t.to_dict() for t in self.closed_trades[-100:]]  # Last 100 trades
        }
        
    def _get_balance_history(self) -> List[Dict]:
        """Construct balance history from closed trades"""
        if not self.closed_trades:
            return []
            
        # Sort trades by close time
        sorted_trades = sorted(
            self.closed_trades,
            key=lambda t: t.exit_time if t.exit_time else datetime.datetime.max
        )
        
        # Construct balance timeline
        balance = self.initial_balance
        balance_history = [{
            "timestamp": sorted_trades[0].entry_time.isoformat(),
            "balance": balance
        }]
        
        for trade in sorted_trades:
            if trade.exit_time:
                balance += trade.profit_loss
                balance_history.append({
                    "timestamp": trade.exit_time.isoformat(),
                    "balance": balance,
                    "trade_id": trade.trade_id,
                    "profit_loss": trade.profit_loss
                })
                
# ====================================
# Broker Interface
# ====================================

class BrokerInterface:
    """
    Handles communication with the broker for data retrieval and
    trade execution. In this implementation, provides simulated
    market data and order execution.
    """
    
    def __init__(self, symbol: str = "EUR/USD"):
        """Initialize the broker interface"""
        self.symbol = symbol
        self.data = None
        self.current_price = 1.0
        self.history_length = 500  # Number of candles to keep
        
    def fetch_market_data(self, timeframe: str = "M5") -> pd.DataFrame:
        """Fetch market data from broker (simulated for this example)"""
        # In a real implementation, this would call the broker API
        # For now, we'll generate simulated data
        
        if self.data is None:
            # Initialize with random walk data
            self.data = self._generate_initial_data()
        else:
            # Add a new bar
            self.data = self._add_new_bar(self.data)
            
        # Return a copy to prevent modification
        return self.data.copy()
    
    def _generate_initial_data(self) -> pd.DataFrame:
        """Generate initial price data"""
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price
        base_price = 1.1000
        
        # Generate a random walk
        returns = np.random.normal(0, 0.0003, self.history_length)
        prices = base_price * (1 + np.cumsum(returns))
        
        # Generate OHLC data
        dates = pd.date_range(
            end=datetime.datetime.now(), 
            periods=self.history_length, 
            freq="5min"
        )
        
        data = pd.DataFrame(index=dates)
        data["close"] = prices
        
        # Generate realistic OHLC
        daily_volatility = 0.008  # 0.8% daily volatility
        candle_volatility = daily_volatility / np.sqrt(288)  # 288 5-min candles per day
        
        data["high"] = data["close"] * (1 + np.random.uniform(0, candle_volatility, self.history_length))
        data["low"] = data["close"] * (1 - np.random.uniform(0, candle_volatility, self.history_length))
        data["open"] = data["close"].shift(1)
        
        # Fill first open price
        data.iloc[0, data.columns.get_loc("open")] = data.iloc[0, data.columns.get_loc("close")] * 0.9995
        
        # Add volume (not used in our strategies, but included for completeness)
        data["volume"] = np.random.randint(50, 200, self.history_length)
        
        return data
    
    def _add_new_bar(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add a new price bar to the data"""
        last_close = data.iloc[-1]["close"]
        
        # Generate a random return
        ret = np.random.normal(0, 0.0003)
        new_close = last_close * (1 + ret)
        
        # Generate OHLC
        candle_volatility = 0.008 / np.sqrt(288)  # Same as in _generate_initial_data
        new_high = new_close * (1 + np.random.uniform(0, candle_volatility))
        new_low = new_close * (1 - np.random.uniform(0, candle_volatility))
        new_open = last_close
        
        # Create new bar
        new_index = data.index[-1] + pd.Timedelta(minutes=5)
        new_bar = pd.DataFrame(
            [[new_open, new_high, new_low, new_close, np.random.randint(50, 200)]],
            columns=["open", "high", "low", "close", "volume"],
            index=[new_index]
        )
        
        # Append to data and keep only the last history_length bars
        updated_data = pd.concat([data, new_bar])
        if len(updated_data) > self.history_length:
            updated_data = updated_data.iloc[-self.history_length:]
            
        self.current_price = new_close
        return updated_data
    
    def get_current_price(self) -> float:
        """Get the current market price"""
        return self.current_price
        
    def execute_market_order(self, direction: OrderDirection, volume: float) -> Dict:
        """Execute a market order (simulated)"""
        # In a real system, this would call the broker API
        return {
            "order_id": f"order_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "symbol": self.symbol,
            "direction": direction.value,
            "volume": volume,
            "price": self.current_price,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def close_order(self, order_id: str) -> Dict:
        """Close an existing order (simulated)"""
        # In a real system, this would call the broker API
        return {
            "order_id": order_id,
            "close_price": self.current_price,
            "timestamp": datetime.datetime.now().isoformat()
        }


# ====================================
# Simulation Runner
# ====================================

def run_trading_system(days_to_run: int = 30, update_interval_seconds: int = 5) -> Dict:
    """Run the trading system for a specified number of days"""
    # Initialize components
    trading_system = TradingSystem(initial_balance=10000.0)
    broker = BrokerInterface(symbol="EUR/USD")
    
    # Set up simulation time
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(days=days_to_run)
    
    logger.info(f"Starting trading simulation from {start_time} to {end_time}")
    
    # Main trading loop
    current_time = start_time
    while current_time < end_time:
        # Fetch latest market data
        market_data = broker.fetch_market_data()
        
        # Update trading system
        result = trading_system.update(market_data, current_time)
        
        if result == "max_drawdown_reached":
            logger.warning("Maximum daily drawdown reached - pausing trading until next day")
        
        # Get system status for monitoring
        if current_time.minute % 30 == 0 and current_time.second < 10:  # Every 30 minutes
            status = trading_system.get_system_status()
            logger.info(f"System status: Balance=${status['balance']:.2f}, Daily P&L: {status['daily_profit_pct']:.2f}%")
        
        # Check for self-evolution opportunity
        if current_time.hour == 0 and current_time.minute == 0 and current_time.second < 10:  # Midnight
            # Generate and apply self-modification
            modification_proposal = trading_system.generate_self_modification_proposal()
            trading_system.self_modify(modification_proposal)
            logger.info(f"System self-evolved: {modification_proposal}")
        
        # Increment simulation time
        current_time += datetime.timedelta(seconds=update_interval_seconds)
        
        # In a real system, we would sleep here
        # time.sleep(update_interval_seconds)
    
    # Simulation complete - generate final report
    performance = trading_system.get_performance_history()
    
    logger.info("Simulation complete")
    logger.info(f"Final balance: ${trading_system.current_balance:.2f}")
    logger.info(f"Total profit: ${trading_system.current_balance - trading_system.initial_balance:.2f}")
    logger.info(f"Return: {(trading_system.current_balance / trading_system.initial_balance - 1) * 100:.2f}%")
    
    # Return final performance data
    return {
        "final_balance": trading_system.current_balance,
        "total_profit": trading_system.current_balance - trading_system.initial_balance,
        "total_return_pct": (trading_system.current_balance / trading_system.initial_balance - 1) * 100,
        "performance_history": performance,
        "prompt_versions": trading_system.evolution.prompt_versions
    }


# ====================================
# Main Entry Point
# ====================================

if __name__ == "__main__":
    # Run the trading system
    results = run_trading_system(days_to_run=30)
    
    print(f"Simulation complete")
    print(f"Final balance: ${results['final_balance']:.2f}")
    print(f"Total profit: ${results['total_profit']:.2f}")
    print(f"Return: {results['total_return_pct']:.2f}%")
    
    # Save results to file
    with open("trading_results.json", "w") as f:
        json.dump(results, f, indent=2)