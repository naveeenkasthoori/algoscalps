"""
Risk management system for the options trading system.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple

from config.strategy_params import (
    TOTAL_CAPITAL, TRADE_CAPITAL, DAILY_LOSS_LIMIT,
    INITIAL_STOP_LOSS_PERCENT, KELLY_FRACTION,
    NIFTY_LOT_SIZE,
    BROKERAGE_PER_LOT, EXCHANGE_CHARGES_RATE, GST_RATE,
    STAMP_DUTY_RATE, SEBI_CHARGES_RATE, STT_RATE_SELL, SLIPPAGE_PERCENT
)

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Manages risk for the trading system including:
    - Position size limits
    - Daily loss limits
    - Drawdown tracking
    - Kelly-based position sizing
    - Transaction cost calculation
    """
    
    def __init__(self, total_capital: float = TOTAL_CAPITAL):
        """
        Initialize the risk manager.
        
        Args:
            total_capital: Total capital for trading
        """
        self.total_capital = total_capital
        self.trade_capital = TRADE_CAPITAL
        self.daily_loss_limit = DAILY_LOSS_LIMIT
        
        # Performance tracking
        self.daily_pnl = {}  # Dictionary to track daily P&L
        self.current_day = None
        self.current_day_pnl = 0
        
        # Historical performance for Kelly calculation
        self.win_count = 0
        self.loss_count = 0
        self.total_win_amount = 0
        self.total_loss_amount = 0
        
        logger.info(f"Risk manager initialized with {total_capital} capital")
    
    def reset(self):
        """Reset the risk manager state."""
        self.daily_pnl = {}
        self.current_day = None
        self.current_day_pnl = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_win_amount = 0
        self.total_loss_amount = 0
        logger.info("Risk manager reset")
    
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is allowed based on daily limits.
        
        Returns:
            True if trading is allowed, False otherwise
        """
        # Check if we've hit the daily loss limit
        if self.current_day_pnl <= -self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {self.current_day_pnl}")
            return False
        
        return True
    
    def can_take_trade(self, signal: Dict[str, Any]) -> bool:
        """
        Check if a specific trade is allowed based on risk rules.
        
        Args:
            signal: Signal dictionary with trade details
            
        Returns:
            True if trade is allowed, False otherwise
        """
        # Check if trading is generally allowed
        if not self.is_trading_allowed():
            return False
        
        # Check if we have enough capital for this trade
        if self.trade_capital > self.get_available_capital():
            logger.warning("Insufficient capital for trade")
            return False
        
        # Additional trade-specific risk checks can be added here
        # For example, checking if the delta is within acceptable range
        
        return True
    
    def get_available_capital(self) -> float:
        """
        Calculate available capital for trading.
        
        Returns:
            Available capital
        """
        # Simple implementation: total capital minus daily losses
        return self.total_capital + min(0, self.current_day_pnl)
    
    def calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """
        Calculate position size for a trade with realistic limits.
        
        Args:
            signal: Signal dictionary with trade details
            
        Returns:
            Position size (number of contracts) rounded to NIFTY lot size
        """
        try:
            option_data = signal['option']
            option_price = option_data['tr_close']
            
            # Calculate win rate and profit/loss ratio from historical data
            win_rate = self.calculate_win_rate()
            profit_loss_ratio = self.calculate_profit_loss_ratio()
            
            # Apply Kelly formula
            kelly_pct = self.calculate_kelly_percentage(win_rate, profit_loss_ratio)
            
            # Always use at least 100,000 per trade (1 lakh)
            # Only reduce if we're in drawdown (defined as current_day_pnl < 0)
            if self.current_day_pnl < 0:
                # If in drawdown, use Kelly-based position sizing but with minimum
                position_capital = max(100000, kelly_pct * self.trade_capital)
            else:
                # If not in drawdown, use full trade capital
                position_capital = self.trade_capital
            
            # Calculate number of contracts
            raw_contracts = position_capital / option_price
            
            # ENHANCEMENT: Add position size limits
            # 1. For very low-priced options (< 10), cap at 1000 contracts
            # 2. For moderately priced options (10-50), cap at 500 contracts
            # 3. For high-priced options (> 50), cap at 300 contracts
            if option_price < 10:
                max_contracts = 1000
            elif option_price < 50:
                max_contracts = 500
            else:
                max_contracts = 300
                
            # Cap the raw contracts at the maximum allowed
            raw_contracts = min(raw_contracts, max_contracts)
            
            # Round down to nearest multiple of NIFTY_LOT_SIZE
            num_contracts = int(raw_contracts / NIFTY_LOT_SIZE) * NIFTY_LOT_SIZE
            
            # Ensure at least one lot if we have enough capital for it
            if num_contracts == 0 and position_capital >= option_price * NIFTY_LOT_SIZE:
                num_contracts = NIFTY_LOT_SIZE
            
            # Log sizing details
            logger.info(
                f"Position sizing for {option_data.get('otype', 'option')} {option_data.get('strike_price', 'unknown')}: "
                f"Price={option_price:.2f}, Capital={position_capital:.2f}, "
                f"Raw contracts={raw_contracts:.2f}, Final contracts={num_contracts} "
                f"(max allowed: {max_contracts})"
            )
            
            return num_contracts
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            
            # Default to a conservative position size
            # ENHANCEMENT: Add a safety cap for the default size too
            try:
                default_size = int(self.trade_capital * 0.8 / option_data['tr_close'])
                # Limit default size to 300 contracts
                default_size = min(default_size, 300)
                # Round down to nearest multiple of NIFTY_LOT_SIZE
                default_size = int(default_size / NIFTY_LOT_SIZE) * NIFTY_LOT_SIZE
                
                logger.info(f"Using default position size (with limits): {default_size} contracts")
                return default_size
            except:
                # Ultimate fallback
                logger.warning("Using minimal fallback position size: 75 contracts")
                return NIFTY_LOT_SIZE  # Return 1 lot as fallback
    
    def calculate_win_rate(self) -> float:
        """
        Calculate historical win rate for Kelly formula.
        Uses weighted recent performance when available,
        falls back to conservative default for initial trades.

        Returns:
            Win rate as a decimal (0.0 to 1.0)
        """
        total_trades = self.win_count + self.loss_count
        
        # Use conservative default for initial trades
        if total_trades < 10:  # Minimum trades threshold
            default_rate = 0.35  # Conservative initial estimate
            logger.debug(f"Insufficient trade history ({total_trades} trades), using default win rate: {default_rate}")
            return default_rate
        
        win_rate = self.win_count / total_trades
        logger.debug(f"Calculated win rate: {win_rate:.2f} from {total_trades} trades ({self.win_count} wins)")
        
        return win_rate
    
    def calculate_profit_loss_ratio(self) -> float:
        """
        Calculate historical profit/loss ratio for Kelly formula.
        
        Returns:
            Profit/loss ratio
        """
        if self.loss_count == 0 or self.total_loss_amount == 0:
            # Default ratio if no historical data
            return 1.5
        
        avg_win = self.total_win_amount / self.win_count if self.win_count > 0 else 0
        avg_loss = abs(self.total_loss_amount / self.loss_count)
        
        if avg_loss == 0:
            return 1.5
        
        return avg_win / avg_loss
    
    def calculate_kelly_percentage(self, win_rate: float, profit_loss_ratio: float) -> float:
        """
        Calculate Kelly percentage based on win rate and profit/loss ratio.
        
        Args:
            win_rate: Historical win rate
            profit_loss_ratio: Historical profit/loss ratio
            
        Returns:
            Kelly percentage as a decimal (0.0 to 1.0)
        """
        # Kelly formula: f* = (p*b - q) / b
        # where f* is optimal fraction, p is win rate, q is loss rate (1-p),
        # and b is profit/loss ratio
        
        loss_rate = 1.0 - win_rate
        
        kelly_pct = (win_rate * profit_loss_ratio - loss_rate) / profit_loss_ratio
        
        # Limit to reasonable range and apply half-Kelly for safety
        kelly_pct = max(0.1, min(0.5, kelly_pct))
        
        # Apply fraction for conservative sizing
        kelly_pct *= KELLY_FRACTION
        
        logger.debug(f"Kelly calculation: win_rate={win_rate}, profit_loss_ratio={profit_loss_ratio}, kelly_pct={kelly_pct}")
        
        return kelly_pct
    
    def update_daily_performance(self, timestamp: datetime, daily_pnl: float):
        """
        Update daily performance tracking.
        
        Args:
            timestamp: Current timestamp
            daily_pnl: Current day's P&L
        """
        current_date = timestamp.date()
        
        # Initialize new day if needed
        if self.current_day != current_date:
            if self.current_day is not None:
                # Store previous day's P&L
                self.daily_pnl[self.current_day] = self.current_day_pnl
                
                # Update win/loss statistics for Kelly
                if self.current_day_pnl > 0:
                    self.win_count += 1
                    self.total_win_amount += self.current_day_pnl
                else:
                    self.loss_count += 1
                    self.total_loss_amount += abs(self.current_day_pnl)
            
            # Reset for new day
            self.current_day = current_date
            self.current_day_pnl = 0
            
            logger.info(f"New trading day: {current_date}")
        
        # Update current day's P&L
        self.current_day_pnl = daily_pnl
    
    def get_daily_performance(self) -> Dict[date, float]:
        """
        Get daily performance history.
        
        Returns:
            Dictionary of daily P&L by date
        """
        # Make a copy of daily P&L
        result = self.daily_pnl.copy()
        
        # Add current day if available
        if self.current_day:
            result[self.current_day] = self.current_day_pnl
        
        return result
    
    def get_max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown from daily performance.
        
        Returns:
            Tuple of (max_drawdown_percentage, peak_date, valley_date)
        """
        daily_pnl = self.get_daily_performance()
        if not daily_pnl:
            return 0.0, None, None
        
        # Convert to Series for calculations
        pnl_series = pd.Series(daily_pnl)
        pnl_series = pnl_series.sort_index()
        
        # Calculate cumulative P&L
        cumulative = self.total_capital + pnl_series.cumsum()
        
        # Calculate drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find peak and valley dates
        valley_date = drawdown.idxmin()
        peak_mask = (running_max.loc[:valley_date] == running_max.loc[valley_date])
        peak_date = peak_mask.index[peak_mask][-1]
        
        return abs(max_drawdown), peak_date, valley_date

    def calculate_transaction_costs(self, option_price: float, position_size: int, is_entry: bool = True) -> Dict[str, float]:
        """
        Calculate transaction costs for a trade.
        
        Args:
            option_price: Price of the option
            position_size: Number of contracts
            is_entry: True if entry trade, False if exit trade
            
        Returns:
            Dictionary with breakdown of transaction costs
        """
        try:
            # Calculate number of lots
            num_lots = position_size / NIFTY_LOT_SIZE
            
            # Calculate total premium value
            premium_value = option_price * position_size
            
            # Calculate brokerage (flat fee per lot)
            brokerage = BROKERAGE_PER_LOT * num_lots
            
            # Calculate exchange transaction charges
            etc = premium_value * EXCHANGE_CHARGES_RATE
            
            # Calculate GST on (brokerage + exchange charges)
            gst = (brokerage + etc) * GST_RATE
            
            # Calculate stamp duty (only on buy/entry)
            stamp_duty = premium_value * STAMP_DUTY_RATE if is_entry else 0
            
            # Calculate SEBI charges
            sebi_charges = premium_value * SEBI_CHARGES_RATE
            
            # Calculate STT (only on sell/exit)
            stt = premium_value * STT_RATE_SELL if not is_entry else 0
            
            # Calculate slippage
            slippage = premium_value * SLIPPAGE_PERCENT
            
            # Calculate total transaction costs
            total_costs = brokerage + etc + gst + stamp_duty + sebi_charges + stt + slippage
            
            costs = {
                'brokerage': brokerage,
                'exchange_charges': etc,
                'gst': gst,
                'stamp_duty': stamp_duty,
                'sebi_charges': sebi_charges,
                'stt': stt,
                'slippage': slippage,
                'total': total_costs
            }
            
            logger.debug(f"Transaction costs for {'entry' if is_entry else 'exit'}: {costs}")
            return costs
            
        except Exception as e:
            logger.error(f"Error calculating transaction costs: {e}")
            return {'total': 0.0}  # Return zero costs on error

    def reset_daily_metrics(self):
        """
        Reset daily metrics without resetting historical performance data.
        Called during day transitions to prepare for a new trading day.
        """
        logger.info(f"Resetting daily metrics for new trading day")
        
        # Store previous day's P&L for historical tracking
        if self.current_day is not None:
            self.daily_pnl[self.current_day] = self.current_day_pnl
            
            # Update win/loss statistics for Kelly
            if self.current_day_pnl > 0:
                self.win_count += 1
                self.total_win_amount += self.current_day_pnl
            else:
                self.loss_count += 1
                self.total_loss_amount += abs(self.current_day_pnl)
        
        # Reset current day's P&L to zero
        self.current_day_pnl = 0
        
        # Update current day
        self.current_day = date.today()
        
        logger.info(f"Daily metrics reset for {self.current_day}")