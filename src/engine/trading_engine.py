"""
Main trading engine for options trading system.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing
import gc

from config.settings import MARKET_OPEN_TIME, MARKET_CLOSE_TIME
from src.strategy.signal_generator import SignalGenerator
from src.engine.position_manager import PositionManager
from src.engine.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Main trading engine that coordinates all components of the trading system.
    
    The trading engine:
    1. Processes time series data timestamp by timestamp
    2. Requests signals from the strategy
    3. Manages position entry/exit through the position manager
    4. Enforces risk limits through the risk manager
    """
    
    def __init__(self, 
                 signal_generator: SignalGenerator,
                 position_manager: PositionManager,
                 risk_manager: RiskManager,
                 backtest_mode: bool = True,
                 max_trades_per_day: int = 5):
        """
        Initialize the trading engine.
        
        Args:
            signal_generator: Strategy signal generator
            position_manager: Position manager
            risk_manager: Risk manager
            backtest_mode: Whether running in backtest mode (default: True)
            max_trades_per_day: Maximum number of trades allowed per day (default: 5)
        """
        self.signal_generator = signal_generator
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.backtest_mode = backtest_mode
        
        # Connect position manager to risk manager
        self.position_manager.set_risk_manager(self.risk_manager)
        
        self.market_open = datetime.strptime(MARKET_OPEN_TIME, "%H:%M:%S").time()
        self.market_close = datetime.strptime(MARKET_CLOSE_TIME, "%H:%M:%S").time()
        
        # Trade history
        self.trade_log = []
        
        # Initialize tracking sets
        self._logged_missing_oi = set()
        
        # Add multiprocessing settings
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        self.chunk_size = 500  # Process timestamps in chunks of this size
        
        # Add error tracking counters
        self.error_count = 0
        self.signal_generation_count = 0
        self.successful_signals_count = 0
        
        self.max_trades_per_day = max_trades_per_day
        
        # New: Track daily trades and active trades by strike
        self.daily_trades = {}  # Format: {date: count}
        self.active_trades_by_strike = {}  # Format: {(option_type, strike, expiry): position_id}
        
        # ADDED: Daily trade registry for tracking trades by strike price
        self.daily_trade_registry = {}  # Format: {date: {(option_type, strike_price): count}}
        
        # Add regime tracking
        self.current_regime = "unknown"
        self.regime_history = []
        
        # ADDED: Global trade registry for deduplication
        self.trade_registry = set()  # Stores (timestamp, option_type, strike_price, expiry_date) tuples
        
        # ADDED: Global trade history that persists across entire backtest
        self.global_trade_history = set()  # Stores (timestamp, option_type, strike, price) tuples
        
        logger.info("Trading engine initialized")
    
    def _is_market_open(self, timestamp: datetime) -> bool:
        """
        Check if the market is open at the given timestamp.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            True if market is open, False otherwise
        """
        if not isinstance(timestamp, datetime):
            return False
            
        current_time = timestamp.time()
        return self.market_open <= current_time <= self.market_close
    
    def _is_duplicate_trade(self, timestamp: datetime, option_type: str, strike_price: float, entry_price: float) -> bool:
        """
        Check against all historical trades to prevent duplicates with enhanced time window and price checks.
        
        Args:
            timestamp: Trade timestamp
            option_type: Type of option (call/put)
            strike_price: Strike price of option
            entry_price: Entry price of trade
            
        Returns:
            bool: True if trade is a duplicate, False otherwise
        """
        try:
            # Normalize timestamp to minute-level granularity
            minute_ts = timestamp.replace(second=0, microsecond=0)
            
            # Dynamic price tolerance based on entry price
            # Use tighter tolerance for higher-priced options
            base_tolerance = 0.005  # 0.5% base tolerance
            if entry_price > 10.0:
                price_tolerance = base_tolerance * 0.5  # 0.25% for higher-priced options
            else:
                price_tolerance = base_tolerance  # 0.5% for lower-priced options
            
            # Check nearby time windows for similar trades
            for historic_ts, historic_type, historic_strike, historic_price in self.global_trade_history:
                try:
                    # Skip if different option type
                    if historic_type != option_type:
                        continue
                    
                    # Check time window (15 minutes before and after)
                    time_diff = abs((minute_ts - historic_ts).total_seconds() / 60)
                    if time_diff > 15:  # 15 minute window
                        continue
                    
                    # Check strike price match (using small absolute tolerance)
                    if abs(historic_strike - strike_price) > 0.01:
                        continue
                    
                    # Check price similarity using dynamic tolerance
                    price_diff_pct = abs(historic_price - entry_price) / entry_price
                    if price_diff_pct <= price_tolerance:
                        logger.warning(
                            f"Duplicate trade detected:\n"
                            f"Current: {option_type} {strike_price} @ {entry_price} ({minute_ts})\n"
                            f"Historic: {historic_type} {historic_strike} @ {historic_price} ({historic_ts})\n"
                            f"Time diff: {time_diff:.1f}min, Price diff: {price_diff_pct:.2%}"
                        )
                        return True
                    
                except Exception as e:
                    logger.debug(f"Error comparing with historic trade: {e}")
                    continue
            
            # If we get here, no duplicates found
            # Add this trade to history for future checks
            self.global_trade_history.add((minute_ts, option_type, strike_price, round(entry_price, 2)))
            return False
            
        except Exception as e:
            logger.error(f"Error checking for duplicate trade: {e}", exc_info=True)
            return True  # Fail safe - treat as duplicate if error occurs

    def _can_open_new_trade(self, timestamp: datetime, option_type: str, strike_price: float, 
                           expiry_date: datetime) -> bool:
        """
        Check if a new trade can be opened based on daily limits and existing positions.
        
        Args:
            timestamp: Current timestamp
            option_type: Type of option (call/put)
            strike_price: Strike price of option
            expiry_date: Expiry date of option
            
        Returns:
            bool: True if new trade can be opened, False otherwise
        """
        current_date = timestamp.date()
        
        # Initialize daily registry if needed
        if current_date not in self.daily_trade_registry:
            self.daily_trade_registry[current_date] = {}
        
        # Track by option characteristics
        option_key = (option_type, strike_price)
        
        # ABSOLUTE LIMIT: Never take more than 1 trade per option key per day
        if option_key in self.daily_trade_registry[current_date]:
            logger.warning(f"Already traded {option_type} {strike_price} today, skipping duplicate")
            return False
        
        # CHECK TOTAL DAILY LIMIT: Count all trades for this date
        total_trades_today = sum(self.daily_trade_registry[current_date].values())
        if total_trades_today >= self.max_trades_per_day:
            logger.warning(f"Daily limit of {self.max_trades_per_day} trades reached for {current_date}")
            return False
        
        # Check existing strike position
        strike_key = (option_type, strike_price, expiry_date.date())
        if strike_key in self.active_trades_by_strike:
            logger.info(f"Already have position on {option_type} {strike_price} {expiry_date.date()}, skipping trade")
            return False
        
        # Check global trade registry for same-minute duplicates
        registry_key = (timestamp.replace(second=0, microsecond=0), option_type, strike_price, expiry_date.date())
        if registry_key in self.trade_registry:
            logger.warning(f"Duplicate trade detected at {timestamp} for {option_type} {strike_price}, skipping")
            return False
        
        # If we get here, it's a valid new trade
        if option_key not in self.daily_trade_registry[current_date]:
            self.daily_trade_registry[current_date][option_key] = 0
        self.daily_trade_registry[current_date][option_key] += 1
        
        # Add to global registry
        self.trade_registry.add(registry_key)
        
        return True

    def _update_trade_tracking(self, timestamp: datetime, position: Dict[str, Any], is_opening: bool = True) -> None:
        """
        Update trade tracking for daily limits and active trades.
        
        Args:
            timestamp: Current timestamp
            position: Position dictionary containing trade details
            is_opening: True if opening a position, False if closing
            
        The function tracks:
        - Daily trade counts
        - Active trades by strike
        - Trade durations
        - Performance metrics by strike
        """
        try:
            current_date = timestamp.date()
            option_type = position.get('option_type')
            strike_price = position.get('strike_price')
            expiry_date = position.get('expiry_date')
            position_id = position.get('id')

            # Validate inputs
            if not all([option_type, strike_price, expiry_date, position_id]):
                logger.error(f"Invalid position data for tracking: {position}")
                return

            expiry_date = expiry_date.date() if isinstance(expiry_date, datetime) else expiry_date
            strike_key = (option_type, strike_price, expiry_date)

            if is_opening:
                # REMOVED: Trade registry addition (now handled in _can_open_new_trade)
                
                # Verify no duplicate trade
                if strike_key in self.active_trades_by_strike:
                    existing_id = self.active_trades_by_strike[strike_key]
                    if existing_id == position_id:
                        logger.warning(f"Duplicate tracking attempt for position {position_id}")
                        return
                    else:
                        logger.error(f"Strike {strike_key} already tracked with different ID: {existing_id}")
                        return

                # Update daily trade count
                if current_date not in self.daily_trades:
                    self.daily_trades[current_date] = 0
                self.daily_trades[current_date] += 1
                
                # Initialize trade tracking
                self.active_trades_by_strike[strike_key] = {
                    'position_id': position_id,
                    'entry_time': timestamp,
                    'entry_price': position.get('entry_price', 0),
                    'quantity': position.get('quantity', 0),
                    'market_regime': self.current_regime
                }
                
                logger.info(f"Added trade tracking for {strike_key} with ID {position_id}")
                
                # Cleanup old daily tracking data (keep last 30 days)
                cutoff_date = current_date - timedelta(days=30)
                self.daily_trades = {
                    date: count for date, count in self.daily_trades.items() 
                    if date > cutoff_date
                }

            else:  # Closing position
                # MODIFIED: Remove from trade registry using minute-level precision
                registry_key = (timestamp.replace(second=0, microsecond=0), option_type, strike_price, expiry_date)
                self.trade_registry.discard(registry_key)
                
                if strike_key in self.active_trades_by_strike:
                    trade_data = self.active_trades_by_strike[strike_key]
                    
                    # Calculate trade duration and P&L
                    duration = timestamp - trade_data['entry_time']
                    exit_price = position.get('exit_price', 0)
                    quantity = trade_data['quantity']
                    pnl = (exit_price - trade_data['entry_price']) * quantity
                    
                    # Update trade history for this strike
                    if not hasattr(self, 'trade_history_by_strike'):
                        self.trade_history_by_strike = {}
                    
                    if strike_key not in self.trade_history_by_strike:
                        self.trade_history_by_strike[strike_key] = []
                    
                    self.trade_history_by_strike[strike_key].append({
                        'position_id': position_id,
                        'entry_time': trade_data['entry_time'],
                        'exit_time': timestamp,
                        'duration': duration,
                        'pnl': pnl,
                        'market_regime': trade_data['market_regime']
                    })
                    
                    # Remove from active tracking
                    del self.active_trades_by_strike[strike_key]
                    logger.info(f"Removed trade tracking for {strike_key} after {duration}")
                    
                    # Cleanup old trade history (keep last 100 trades per strike)
                    if len(self.trade_history_by_strike[strike_key]) > 100:
                        self.trade_history_by_strike[strike_key] = \
                            self.trade_history_by_strike[strike_key][-100:]
                else:
                    logger.warning(f"No active trade found for {strike_key} when closing")

        except Exception as e:
            logger.error(f"Error in trade tracking: {e}", exc_info=True)

    def detect_market_regime(self, futures_data: pd.DataFrame) -> str:
        """
        Detect current market regime (trending, range-bound, volatile)
        with careful handling for limited data scenarios.
        
        Args:
            futures_data: DataFrame with futures market data
            
        Returns:
            String indicating the detected regime
        """
        try:
            # Make sure futures_data is not empty
            if futures_data.empty:
                logger.debug("Empty futures data for market regime detection")
                return "range"  # Default to range-bound when no data
            
            # Check if we have enough data points for calculation
            atr_period = 14
            required_points_min = 5  # Absolute minimum to do any calculation
            required_points_full = 2 * atr_period  # Need at least 2*atr_period points for full ADX
            
            data_points = len(futures_data)
            
            if data_points < required_points_min:
                logger.debug(f"Insufficient data ({data_points} points) for market regime detection, using default")
                return "range"  # Default to range-bound when not enough data
            
            # Prepare data arrays safely
            high = futures_data['tr_high'].values
            low = futures_data['tr_low'].values
            close = futures_data['tr_close'].values
            
            # Basic volatility assessment with limited data
            if required_points_min <= data_points < required_points_full:
                logger.debug(f"Limited data ({data_points} points) for regime detection, using simplified approach")
                
                # Calculate simple price range as percentage of price
                recent_range = (high[-1] - low[-1]) / close[-1] * 100
                
                # Calculate simple trend direction
                if data_points >= 3:
                    if close[-1] > close[-2] > close[-3]:
                        return "uptrend"
                    elif close[-1] < close[-2] < close[-3]:
                        return "downtrend"
                    elif recent_range > 1.0:  # More than 1% range
                        return "volatile"
                
                # Default when no clear pattern detected
                return "range"
            
            # Full calculation when we have enough data
            logger.debug(f"Sufficient data ({data_points} points) for full regime detection")
            
            # 1. ATR calculation - safely handle array dimensions
            if data_points > 1:  # Need at least 2 points for TR
                # Calculate True Range components
                tr1 = np.abs(high[1:] - low[1:])
                tr2 = np.abs(high[1:] - close[:-1])
                tr3 = np.abs(low[1:] - close[:-1])
                
                # Safety check for array shapes
                min_len = min(len(tr1), len(tr2), len(tr3))
                if min_len == 0:
                    return "range"  # Not enough data for TR calculation
                    
                # Ensure all arrays have the same shape before stacking
                tr1, tr2, tr3 = tr1[:min_len], tr2[:min_len], tr3[:min_len]
                
                # Get the maximum of the three components for each period
                tr = np.maximum.reduce([tr1, tr2, tr3])
                
                # Initialize ATR array
                atr = np.zeros(data_points)
                
                # Calculate ATR using a rolling window approach
                for i in range(atr_period, data_points):
                    if i - atr_period >= 0 and i < len(tr) + 1:
                        atr[i] = np.mean(tr[max(0, i-atr_period):min(i, len(tr))])
            else:
                # Not enough data for TR calculation
                return "range"
            
            # 2. Directional Movement - with careful array handling
            plus_dm = np.zeros(data_points)
            minus_dm = np.zeros(data_points)
            
            # Calculate directional movement safely
            for i in range(1, data_points):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            # 3. Calculate ADX components with careful shape handling
            plus_di = np.zeros(data_points)
            minus_di = np.zeros(data_points)
            
            # Only calculate if we have enough data for the smoothed ATR
            for i in range(atr_period, data_points):
                # Prevent division by zero
                denominator = atr[i] * atr_period
                if denominator > 0:  # Only calculate when we have a valid denominator
                    # Make sure our slice indices are valid
                    start_idx = max(0, i-atr_period+1)
                    end_idx = i+1
                    
                    if start_idx < end_idx <= data_points:
                        plus_di[i] = 100 * np.sum(plus_dm[start_idx:end_idx]) / denominator
                        minus_di[i] = 100 * np.sum(minus_dm[start_idx:end_idx]) / denominator
            
            # Calculate DX (Directional Index)
            dx = np.zeros(data_points)
            for i in range(atr_period, data_points):
                # Prevent division by zero
                denominator = plus_di[i] + minus_di[i]
                if denominator > 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / denominator
            
            # Calculate ADX (Average Directional Index) - only if we have enough data
            adx = np.zeros(data_points)
            
            if data_points >= 2*atr_period:
                for i in range(2*atr_period, data_points):
                    # Make sure our slice indices are valid
                    start_idx = max(0, i-atr_period)
                    end_idx = i
                    
                    if start_idx < end_idx <= data_points:
                        # Use only valid DX values (non-zero)
                        valid_dx = dx[start_idx:end_idx]
                        valid_dx = valid_dx[valid_dx > 0]
                        
                        if len(valid_dx) > 0:
                            adx[i] = np.mean(valid_dx)
            
            # 4. Get the latest values for regime determination
            latest_atr = atr[-1]
            latest_adx = adx[-1]
            latest_plus_di = plus_di[-1]
            latest_minus_di = minus_di[-1]
            
            # 5. Calculate average ATR for volatility comparison
            if data_points >= 20:
                lookback = min(20, data_points-1)  # Use shorter lookback if needed
                avg_atr = np.mean(atr[-lookback:])
                atr_ratio = latest_atr / avg_atr if avg_atr > 0 else 1.0
            else:
                atr_ratio = 1.0  # Default to neutral
            
            # 6. Determine market regime based on indicators
            if latest_adx > 25:  # Strong trend
                logger.debug(f"Detected trending market - ADX: {latest_adx:.2f}, +DI: {latest_plus_di:.2f}, -DI: {latest_minus_di:.2f}")
                if latest_plus_di > latest_minus_di:
                    return "uptrend"
                else:
                    return "downtrend"
            elif atr_ratio > 1.5:  # High volatility
                logger.debug(f"Detected volatile market - ATR ratio: {atr_ratio:.2f}")
                return "volatile"
            else:  # Range-bound
                logger.debug(f"Detected range-bound market - ADX: {latest_adx:.2f}, ATR ratio: {atr_ratio:.2f}")
                return "range"
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}", exc_info=True)
            return "range"  # Return safe default on error
    
    def _validate_no_duplicates(self, option_type: str, strike_price: float) -> bool:
        """
        Validate we truly have no duplicate trades before proceeding.
        
        Args:
            option_type: Type of option (call/put)
            strike_price: Strike price of option
            
        Returns:
            bool: True if no duplicates found, False otherwise
        """
        try:
            # Check open positions
            for pos_id, pos in self.position_manager.get_open_positions().items():
                if (pos.get('option_type') == option_type and 
                    abs(pos.get('strike_price', 0) - strike_price) < 0.01):  # Use small tolerance for float comparison
                    logger.error(
                        f"DUPLICATE CHECK FAILED: {option_type} {strike_price} "
                        f"already exists in open positions (ID: {pos_id})!"
                    )
                    return False
            
            # Check active trades tracking
            for key, trade_data in self.active_trades_by_strike.items():
                tracked_type, tracked_strike, _ = key
                if (tracked_type == option_type and 
                    abs(tracked_strike - strike_price) < 0.01):
                    logger.error(
                        f"DUPLICATE CHECK FAILED: {option_type} {strike_price} "
                        f"already exists in active_trades_by_strike! "
                        f"Position ID: {trade_data.get('position_id')}"
                    )
                    return False
            
            # Check global trade history for recent duplicates
            current_minute = datetime.now().replace(second=0, microsecond=0)
            for ts, trade_type, trade_strike, _ in self.global_trade_history:
                # Only check trades in last 5 minutes
                if (current_minute - ts).total_seconds() <= 300:  # 5 minutes
                    if (trade_type == option_type and 
                        abs(trade_strike - strike_price) < 0.01):
                        logger.error(
                            f"DUPLICATE CHECK FAILED: {option_type} {strike_price} "
                            f"found in recent global trade history at {ts}!"
                        )
                        return False
            
            logger.debug(f"Duplicate validation passed for {option_type} {strike_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error in duplicate validation: {e}", exc_info=True)
            return False  # Fail safe - treat as duplicate if error occurs

    def _debug_position_execution(self, 
                                timestamp: datetime, 
                                signal: Dict[str, Any],
                                position_size: float) -> None:
        """
        Debug helper to diagnose why a signal isn't becoming a trade.
        
        Args:
            timestamp: Current timestamp
            signal: Trading signal dictionary
            position_size: Calculated position size
        """
        logger.debug(f"DEBUG POSITION EXECUTION for signal at {timestamp}:")
        
        # Check signal basics
        if 'option' not in signal:
            logger.debug(" - FAIL: Signal is missing 'option' field")
            return
        
        option = signal['option']
        
        # Check required fields in option data
        required_fields = ['otype', 'strike_price', 'tr_close', 'expiry_date', 'week_expiry_date']
        missing_fields = [field for field in required_fields 
                         if field not in option and field != 'week_expiry_date']
        
        if 'expiry_date' not in option and 'week_expiry_date' not in option:
            missing_fields.append('expiry_date/week_expiry_date')
        
        if missing_fields:
            logger.debug(f" - FAIL: Option data missing required fields: {missing_fields}")
            logger.debug(f" - Available fields: {list(option.keys())}")
            return
        
        # Check expiry date
        try:
            if 'expiry_date' in option:
                expiry_date = pd.to_datetime(option['expiry_date'])
            elif 'week_expiry_date' in option:
                expiry_date = pd.to_datetime(option['week_expiry_date'])
            else:
                logger.debug(f" - FAIL: Could not determine expiry date")
                return
            
            # Validate expiry is in the future
            if expiry_date.date() <= timestamp.date():
                logger.debug(f" - FAIL: Expired option: {expiry_date}")
                return
                
        except Exception as e:
            logger.debug(f" - FAIL: Error processing expiry date: {e}")
            return
        
        # Check price validity
        if not isinstance(option['tr_close'], (int, float)) or option['tr_close'] <= 0:
            logger.debug(f" - FAIL: Invalid option price: {option['tr_close']}")
            return
        
        # Check strike price validity
        if not isinstance(option['strike_price'], (int, float)) or option['strike_price'] <= 0:
            logger.debug(f" - FAIL: Invalid strike price: {option['strike_price']}")
            return
        
        # Check position size
        if position_size <= 0:
            logger.debug(f" - FAIL: Invalid position size: {position_size}")
            return
        
        # Check trade limits
        if not self._can_open_new_trade(timestamp, option['otype'], option['strike_price'], expiry_date):
            logger.debug(" - FAIL: Trade limit checks failed")
            return
        
        # Check for duplicates
        if self._is_duplicate_trade(timestamp, option['otype'], option['strike_price'], option['tr_close']):
            logger.debug(" - FAIL: Duplicate trade detected")
            return
        
        # All checks passed
        logger.debug(f" - PASS: Signal validation successful")
        logger.debug(f"   Option: {option['otype']} Strike: {option['strike_price']} Price: {option['tr_close']}")
        logger.debug(f"   Expiry: {expiry_date} Size: {position_size}")

    def _process_options_signals(self, timestamp: datetime, current_price: float, options_data: Dict[str, pd.DataFrame]) -> None:
        """
        Process options signals from the signal generator and manage positions.
        
        Args:
            timestamp: Current timestamp
            current_price: Current price of the underlying asset
            options_data: Dictionary of options data by expiry
        """
        try:
            # Get current market regime
            market_regime = self.current_regime if hasattr(self, 'current_regime') else 'unknown'
            
            # Get open positions
            open_positions = self.position_manager.get_open_positions()
            
            # Generate signals using the signal generator
            signals = self.signal_generator.generate_signals(
                timestamp=timestamp,
                spot_data=pd.DataFrame(),  # We're using futures data instead
                futures_data=pd.DataFrame({'tr_close': [current_price]}),
                options_data=options_data,
                open_positions=open_positions,
                market_regime=market_regime
            )
            
            # Process entry signals
            if 'entries' in signals and signals['entries']:
                for entry in signals['entries']:
                    try:
                        # Check if we can open a new trade
                        if not self._can_open_new_trade(
                            timestamp, 
                            entry.get('option_type', ''), 
                            entry.get('strike_price', 0.0),
                            entry.get('expiry_date', timestamp)
                        ):
                            continue
                        
                        # Calculate position size
                        position_size = self.risk_manager.calculate_position_size(
                            entry.get('price', 0.0),
                            entry.get('stop_loss', 0.0)
                        )
                        
                        # Debug position execution
                        self._debug_position_execution(timestamp, entry, position_size)
                        
                        # Open position
                        position = self.position_manager.open_position(
                            timestamp=timestamp,
                            symbol=entry.get('symbol', ''),
                            option_type=entry.get('option_type', ''),
                            strike_price=entry.get('strike_price', 0.0),
                            expiry_date=entry.get('expiry_date', timestamp),
                            entry_price=entry.get('price', 0.0),
                            position_size=position_size,
                            stop_loss=entry.get('stop_loss', 0.0),
                            take_profit=entry.get('take_profit', 0.0),
                            signal_strength=entry.get('signal_strength', 0.0),
                            conditions_met=entry.get('conditions_met', [])
                        )
                        
                        # Update trade tracking
                        if position:
                            self._update_trade_tracking(timestamp, position, is_opening=True)
                    except Exception as e:
                        logger.error(f"Error processing entry signal: {e}")
            
            # Process exit signals
            if 'exits' in signals and signals['exits']:
                for exit_signal in signals['exits']:
                    try:
                        # Close position
                        position_id = exit_signal.get('position_id', '')
                        if position_id and position_id in open_positions:
                            closed_position = self.position_manager.close_position(
                                position_id=position_id,
                                timestamp=timestamp,
                                exit_price=exit_signal.get('price', 0.0),
                                exit_reason=exit_signal.get('reason', 'signal')
                            )
                            
                            # Update trade tracking
                            if closed_position:
                                self._update_trade_tracking(timestamp, closed_position, is_opening=False)
                    except Exception as e:
                        logger.error(f"Error processing exit signal: {e}")
            
            # Update signal count for monitoring
            if hasattr(self, 'signal_count'):
                self.signal_count += len(signals.get('entries', [])) + len(signals.get('exits', []))
            else:
                self.signal_count = len(signals.get('entries', [])) + len(signals.get('exits', []))
                
        except Exception as e:
            logger.error(f"Error in _process_options_signals: {e}")
            if hasattr(self, 'error_count'):
                self.error_count += 1
            else:
                self.error_count = 1

    def process_data_snapshot(self, 
                             timestamp: datetime,
                             spot_data: pd.DataFrame,
                             futures_data: pd.DataFrame,
                             options_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process a snapshot of market data at a specific timestamp.
        
        Args:
            timestamp: Current timestamp
            spot_data: Spot market data
            futures_data: Futures market data
            options_data: Options market data by expiry
        """
        # Reduce logging during intensive processing
        rules_logger = logging.getLogger('src.strategy.rules')
        original_level = rules_logger.level
        rules_logger.setLevel(logging.INFO)  # Temporarily reduce to INFO level
        
        try:
            # Get current market state
            if not futures_data.empty:
                current_price = futures_data['tr_close'].iloc[-1]
                
                # Process options data in batches to avoid memory issues
                processed_options = {}
                
                # Track options that meet mandatory conditions to avoid redundant checks
                valid_options_cache = {}
                
                for expiry, expiry_data in options_data.items():
                    try:
                        # Skip if no data
                        if expiry_data.empty:
                            continue
                        
                        # Check if we have OI velocity data and add if missing
                        if 'oi_velocity' not in expiry_data.columns or expiry_data['oi_velocity'].isna().all():
                            # Log only once per expiry
                            if expiry not in self._logged_missing_oi:
                                logger.warning(f"Missing OI velocity data for expiry {expiry}, attempting to calculate")
                                self._logged_missing_oi.add(expiry)
                            
                            # Try to calculate OI velocity
                            from src.strategy.rules import handle_missing_oi_velocity
                            expiry_data = handle_missing_oi_velocity(expiry_data, timestamp)
                        
                        # Process this expiry's options
                        processed_options[expiry] = expiry_data
                        
                    except Exception as e:
                        logger.error(f"Error processing options for expiry {expiry}: {e}")
                
                # Continue with normal processing using processed options
                self._process_options_signals(timestamp, current_price, processed_options)
            else:
                logger.warning(f"No futures data available at {timestamp}, skipping options processing")
        
        except Exception as e:
            logger.error(f"Error in process_data_snapshot: {e}")
        
        finally:
            # Restore original log level
            rules_logger.setLevel(original_level)
            
            # Force garbage collection after processing
            gc.collect()

    def _handle_day_transition(self, current_timestamp: datetime, previous_timestamp: Optional[datetime]) -> None:
        """
        Handle transition between trading days with comprehensive state reset and verification.
        
        Args:
            current_timestamp: Current processing timestamp
            previous_timestamp: Previous processed timestamp
        """
        if previous_timestamp is None or current_timestamp.date() <= previous_timestamp.date():
            return
        
        prev_date = previous_timestamp.date()
        new_date = current_timestamp.date()
        logger.info(f"=== DAY TRANSITION: {prev_date} -> {new_date} ===")
        
        try:
            # 1. Record initial state
            initial_state = {
                'open_positions': self.position_manager.get_open_positions(),
                'active_trades': dict(self.active_trades_by_strike),
                'daily_trades': dict(self.daily_trades),
                'trade_registry': set(self.trade_registry)
            }
            
            logger.info(f"Initial state: {len(initial_state['open_positions'])} positions, "
                       f"{len(initial_state['active_trades'])} tracked trades")
            
            # 2. Close all positions from previous day
            try:
                closed_positions = self.position_manager.close_all_positions(previous_timestamp)
                logger.info(f"Closed {len(closed_positions)} positions from {prev_date}")
                
                # Verify positions are closed
                remaining = self.position_manager.get_open_positions()
                if remaining:
                    logger.error(f"ERROR: {len(remaining)} positions still open after closing!")
                    # Force reset as last resort
                    self.position_manager.open_positions = {}
                    logger.info("Forced position manager reset")
            except Exception as e:
                logger.error(f"Error closing positions: {e}", exc_info=True)
                self.position_manager.open_positions = {}
            
            # 3. Reset all tracking mechanisms
            try:
                # Clear active trade tracking
                prev_active = len(self.active_trades_by_strike)
                self.active_trades_by_strike.clear()
                
                # Reset trade registry for just the previous day
                old_registry_size = len(self.trade_registry)
                self.trade_registry = {
                    key for key in self.trade_registry 
                    if key[0].date() >= new_date
                }
                
                # Initialize new day's trade count
                self.daily_trades[new_date] = 0
                
                # Cleanup old daily trade records (keep last 30 days)
                cutoff_date = new_date - timedelta(days=30)
                self.daily_trades = {
                    date: count for date, count in self.daily_trades.items() 
                    if date > cutoff_date
                }
                
                logger.info(f"Cleared {prev_active} tracked trades and "
                           f"{old_registry_size - len(self.trade_registry)} registry entries")
                
            except Exception as e:
                logger.error(f"Error resetting trade tracking: {e}", exc_info=True)
                # Emergency cleanup
                self.active_trades_by_strike.clear()
                self.trade_registry.clear()
                self.daily_trades = {new_date: 0}
            
            # 4. Reset risk manager state
            try:
                self.risk_manager.reset_daily_metrics()
                logger.info("Reset risk manager daily metrics")
            except Exception as e:
                logger.error(f"Error resetting risk metrics: {e}", exc_info=True)
            
            # 5. Verify final state
            final_state = {
                'open_positions': len(self.position_manager.get_open_positions()),
                'active_trades': len(self.active_trades_by_strike),
                'trade_registry': len(self.trade_registry),
                'daily_trades': self.daily_trades.get(new_date, 0)
            }
            
            logger.info(f"=== Day Transition Complete ===\n"
                       f"Initial: {len(initial_state['open_positions'])} positions, "
                       f"{len(initial_state['active_trades'])} tracked trades\n"
                       f"Final: {final_state['open_positions']} positions, "
                       f"{final_state['active_trades']} tracked trades\n"
                       f"New day {new_date} initialized")
            
            # 6. Verify clean state
            if final_state['open_positions'] > 0:
                logger.error(f"ERROR: {final_state['open_positions']} positions still open after reset!")
            if final_state['active_trades'] > 0:
                logger.error(f"ERROR: {final_state['active_trades']} trades still tracked after reset!")
            
        except Exception as e:
            logger.error(f"Critical error during day transition: {e}", exc_info=True)
            # Emergency cleanup of all state
            self.position_manager.open_positions = {}
            self.active_trades_by_strike.clear()
            self.trade_registry.clear()
            self.daily_trades = {new_date: 0}
            self.risk_manager.reset_daily_metrics()
            logger.info("Completed emergency state cleanup after critical error")

    def _preserve_state_between_chunks(self) -> Dict[str, Any]:
        """
        Save and validate the current state of trade tracking to ensure consistency between chunks.
        Returns a state summary for logging.
        """
        try:
            logger.debug("Preserving trading state between chunks")
            
            # Get current open positions
            open_positions = self.position_manager.get_open_positions()
            
            # Verify consistency between open_positions and active_trades_by_strike
            open_position_keys = set()
            for pos_id, pos in open_positions.items():
                try:
                    # Extract key components with validation
                    option_type = pos.get('option_type')
                    strike_price = pos.get('strike_price')
                    expiry_date = pos.get('expiry_date')
                    
                    if not all([option_type, strike_price, expiry_date]):
                        logger.error(f"Invalid position data for {pos_id}: {pos}")
                        continue
                        
                    # Convert expiry to date if needed
                    if isinstance(expiry_date, datetime):
                        expiry_date = expiry_date.date()
                    
                    strike_key = (option_type, strike_price, expiry_date)
                    open_position_keys.add(strike_key)
                    
                    # Ensure position is properly tracked
                    if strike_key not in self.active_trades_by_strike:
                        logger.warning(f"Found untracked position {pos_id} for {strike_key}, adding to tracking")
                        self.active_trades_by_strike[strike_key] = {
                            'position_id': pos_id,
                            'entry_time': pos.get('entry_time', datetime.now()),
                            'entry_price': pos.get('entry_price', 0),
                            'quantity': pos.get('quantity', 0),
                            'market_regime': self.current_regime
                        }
                    
                except Exception as e:
                    logger.error(f"Error processing position {pos_id} during state preservation: {e}")
                    continue
            
            # Clean up stale tracking entries
            tracked_keys = list(self.active_trades_by_strike.keys())
            for strike_key in tracked_keys:
                trade_data = self.active_trades_by_strike[strike_key]
                pos_id = trade_data.get('position_id')
                
                if pos_id not in open_positions:
                    logger.warning(f"Removing stale tracking entry for {strike_key}, position {pos_id} no longer exists")
                    del self.active_trades_by_strike[strike_key]
            
            # Validate daily trades tracking
            current_date = datetime.now().date()
            stale_dates = [date for date in self.daily_trades.keys() 
                          if date < current_date - timedelta(days=30)]
            
            for date in stale_dates:
                del self.daily_trades[date]
            
            # Return state summary
            state_summary = {
                'open_positions': len(open_positions),
                'tracked_strikes': len(self.active_trades_by_strike),
                'daily_trades_dates': len(self.daily_trades),
                'current_regime': self.current_regime,
                'error_count': self.error_count,
                'signal_count': self.signal_generation_count
            }
            
            logger.debug(f"State preserved: {state_summary}")
            return state_summary
            
        except Exception as e:
            logger.error(f"Error during state preservation: {e}", exc_info=True)
            return {
                'error': str(e),
                'status': 'failed'
            }

    def _process_timestamp_chunk(self, 
                               timestamps: List[datetime], 
                               spot_data: pd.DataFrame,
                               futures_data: pd.DataFrame,
                               options_data: Dict[str, pd.DataFrame],
                               start_idx: int,
                               total_len: int) -> None:
            """Process a chunk of timestamps with state preservation."""
            try:
                # Sort timestamps to ensure chronological processing
                timestamps = sorted(timestamps)
                
                # Group timestamps by date for better handling of day transitions
                timestamps_by_date = {}
                for ts in timestamps:
                    date = ts.date()
                    if date not in timestamps_by_date:
                        timestamps_by_date[date] = []
                    timestamps_by_date[date].append(ts)
                
                # Create historical buffer
                historical_buffer = {
                    'spot': None,
                    'futures': None,
                    'options': {}
                }
                
                # Pre-calculate indices
                spot_index = {ts: i for i, ts in enumerate(spot_data.index)}
                futures_index = {ts: i for i, ts in enumerate(futures_data.index)}
                options_indices = {
                    expiry: {ts: i for i, ts in enumerate(data.index)}
                    for expiry, data in options_data.items()
                }
                
                processed_count = 0
                error_count = 0
                previous_timestamp = None
                previous_date = None
                
                # Process each date chronologically
                for date in sorted(timestamps_by_date.keys()):
                    day_timestamps = timestamps_by_date[date]
                    logger.info(f"Processing {len(day_timestamps)} timestamps for date {date}")
                    
                    if previous_date is not None and date > previous_date:
                        logger.info(f"Day transition detected: {previous_date} -> {date}")
                        self._handle_day_transition(day_timestamps[0], previous_timestamp)
                        
                        if historical_buffer['spot'] is not None:
                            logger.info("Using historical buffer for day transition")
                    
                    # Initialize data slices
                    current_spot_slice = historical_buffer['spot'] if historical_buffer['spot'] is not None else spot_data.iloc[:1]
                    current_futures_slice = historical_buffer['futures'] if historical_buffer['futures'] is not None else futures_data.iloc[:1]
                    
                    current_options_slices = {}
                    for expiry, buffer_data in historical_buffer['options'].items():
                        if expiry in options_data:
                            current_options_slices[expiry] = buffer_data
                            
                    # Add missing expiries
                    for expiry in options_data.keys():
                        if expiry not in current_options_slices:
                            current_options_slices[expiry] = options_data[expiry].iloc[:1] if not options_data[expiry].empty else pd.DataFrame()
                    
                    # Process timestamps for this day
                    for timestamp in day_timestamps:
                        try:
                            # Update data slices
                            try:
                                valid_spot_ts = [ts for ts in spot_index.keys() if ts <= timestamp]
                                if valid_spot_ts:
                                    spot_idx = spot_index[max(valid_spot_ts)]
                                    if spot_idx >= len(current_spot_slice):
                                        current_spot_slice = spot_data.iloc[:spot_idx+1]
                        
                                valid_futures_ts = [ts for ts in futures_index.keys() if ts <= timestamp]
                                if valid_futures_ts:
                                    futures_idx = futures_index[max(valid_futures_ts)]
                                    if futures_idx >= len(current_futures_slice):
                                        current_futures_slice = futures_data.iloc[:futures_idx+1]
                    
                            except ValueError as e:
                                logger.error(f"Invalid timestamp index for {timestamp}: {e}")
                                error_count += 1
                                continue
                            
                            # Update options slices
                            for expiry, expiry_data in options_data.items():
                                if expiry in options_indices:
                                    try:
                                        valid_option_ts = [ts for ts in options_indices[expiry].keys() if ts <= timestamp]
                                        if valid_option_ts:
                                            options_idx = options_indices[expiry][max(valid_option_ts)]
                                            if expiry not in current_options_slices or options_idx >= len(current_options_slices[expiry]):
                                                current_options_slices[expiry] = expiry_data.iloc[:options_idx+1]
                                    except (ValueError, KeyError) as e:
                                        logger.debug(f"No valid option data for {expiry} at {timestamp}: {e}")
                                        if expiry not in current_options_slices:
                                            current_options_slices[expiry] = pd.DataFrame()
                        
                            # Process snapshot
                            self.process_data_snapshot(
                                timestamp, current_spot_slice, current_futures_slice, current_options_slices
                            )
                            
                            processed_count += 1
                            previous_timestamp = timestamp
                            
                            if processed_count % 100 == 0:
                                logger.debug(f"Processed {start_idx + processed_count}/{total_len} timestamps (errors: {error_count})")
                        
                        except Exception as e:
                            error_count += 1
                            logger.error(f"Error processing timestamp {timestamp}: {e}")
                    
                    # Update historical buffer
                    historical_buffer['spot'] = current_spot_slice
                    historical_buffer['futures'] = current_futures_slice
                    historical_buffer['options'] = {k: v for k, v in current_options_slices.items()}
                    
                    previous_date = date
                
                logger.info(f"Chunk complete - Processed: {processed_count}, Errors: {error_count}")
                
                # After processing chunk, preserve state
                state_summary = self._preserve_state_between_chunks()
                
                if state_summary.get('status') == 'failed':
                    logger.error("State preservation failed after chunk processing")
                    # Consider additional error handling here
                else:
                    logger.info(f"Chunk processed successfully. State summary: {state_summary}")
                
            except Exception as e:
                logger.error(f"Error in chunk processing: {e}", exc_info=True)
                # Attempt state preservation even on error
                self._preserve_state_between_chunks()

    def run_backtest(self, 
                    spot_data: pd.DataFrame,
                    futures_data: pd.DataFrame,
                    options_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run backtest with improved state management."""
        try:
            logger.info("Starting backtest with optimized performance")
            
            # Reset state
            self.trade_log = []
            self.position_manager.reset()
            self.risk_manager.reset()
            
            # ADDED: Reset trade registry
            self.trade_registry.clear()
            
            # ADDED: Clear global trade history at start of backtest
            self.global_trade_history.clear()
            
            # Reset error counters
            self.error_count = 0
            self.signal_generation_count = 0
            self.successful_signals_count = 0
            
            # IMPORTANT FIX: Add validation for input data
            if spot_data.empty:
                logger.error("Empty spot data, cannot run backtest")
                return {"status": "error", "error": "Empty spot data"}
            
            if futures_data.empty:
                logger.error("Empty futures data, cannot run backtest")
                return {"status": "error", "error": "Empty futures data"}
            
            if not options_data:
                logger.error("No options data, cannot run backtest")
                return {"status": "error", "error": "No options data"}
            
            # Get all unique timestamps from futures data
            timestamps = futures_data.index.tolist()
            total_timestamps = len(timestamps)
            
            logger.info(f"Processing {total_timestamps} timestamps")
            
            # Early filter for market hours
            market_hours_timestamps = []
            for ts in timestamps:
                if (self._is_market_open(ts)) or (
                    # Special case: include pre-market timestamps if we need them
                    # for proper data initialization
                    (len(market_hours_timestamps) == 0 and ts.time() < self.market_open)
                ):
                    market_hours_timestamps.append(ts)
            
            logger.info(f"Filtered to {len(market_hours_timestamps)} timestamps during market hours")
            
            # Debug: Print first few timestamps for diagnostics
            if market_hours_timestamps:
                first_ts = market_hours_timestamps[0]
                last_ts = market_hours_timestamps[-1]
                logger.info(f"First timestamp: {first_ts} ({first_ts.time()}), Last: {last_ts} ({last_ts.time()})")
                logger.debug(f"Market hours: {self.market_open} to {self.market_close}")
            
            # FIX: If no market hours timestamps, use all timestamps
            if not market_hours_timestamps:
                logger.warning("No market hours timestamps found! Using all timestamps instead.")
                market_hours_timestamps = timestamps
            
            # Process in chunks
            chunk_size = min(self.chunk_size, len(market_hours_timestamps))
            chunks = [market_hours_timestamps[i:i+chunk_size] for i in range(0, len(market_hours_timestamps), chunk_size)]
            
            logger.info(f"Processing in {len(chunks)} chunks of up to {chunk_size} timestamps each")
            
            # Process chunks sequentially with day tracking
            previous_timestamp = None
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                try:
                    # Pass previous timestamp to chunk processor
                    self._process_timestamp_chunk(
                        chunk, spot_data, futures_data, options_data,
                        i * chunk_size, len(market_hours_timestamps)
                    )
                    
                    # ADDED: Preserve state between chunks
                    state_summary = self._preserve_state_between_chunks()
                    if state_summary.get('status') == 'failed':
                        logger.error(f"State preservation failed after chunk {i+1}")
                        # Consider additional error handling
                    
                    # Update previous timestamp for next chunk
                    if chunk:
                        previous_timestamp = chunk[-1]
                    
                    # Force garbage collection after each chunk
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    # Attempt state preservation even on chunk error
                    self._preserve_state_between_chunks()
                    continue
            
            # Close remaining positions
            try:
                final_timestamp = timestamps[-1]
                self.position_manager.close_all_positions(final_timestamp)
            except Exception as e:
                logger.error(f"Error closing positions: {e}")
            
            # Generate results
            results = self.generate_backtest_results()
            
            # Add error statistics to results
            results['signal_generation_count'] = self.signal_generation_count
            results['successful_signals_count'] = self.successful_signals_count
            results['error_count'] = self.error_count
            results['signal_generation_success_rate'] = (
                self.successful_signals_count / self.signal_generation_count 
                if self.signal_generation_count > 0 else 0
            )
            
            # Log final statistics
            logger.info("Backtest completed")
            trade_count = len(self.position_manager.get_closed_positions())
            logger.info(f"Generated {trade_count} trades during backtest")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def generate_backtest_results(self) -> Dict[str, Any]:
        """
        Generate backtest performance results.
        
        Returns:
            Dictionary with performance metrics
        """
        def get_trade_pnl(trade: Dict) -> float:
            """Helper function to get net PnL from a trade consistently"""
            # Prioritize total_net_pnl which includes all costs
            return trade.get('total_net_pnl', 0.0)
        
        # Get trade history
        trades = self.position_manager.get_closed_positions()
        
        # Calculate basic metrics
        num_trades = len(trades)
        if num_trades == 0:
            logger.warning("No trades executed in backtest")
            return {
                "status": "completed",
                "num_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "trades": []
            }
        
        # Calculate performance metrics using net PnL (with costs)
        winning_trades = [t for t in trades if get_trade_pnl(t) > 0]
        losing_trades = [t for t in trades if get_trade_pnl(t) <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        avg_win = np.mean([get_trade_pnl(t) for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([get_trade_pnl(t) for t in losing_trades])) if losing_trades else 0
        
        total_profit = sum([get_trade_pnl(t) for t in winning_trades]) if winning_trades else 0
        total_loss = abs(sum([get_trade_pnl(t) for t in losing_trades])) if losing_trades else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate total costs
        total_entry_costs = sum([t.get('entry_costs', {}).get('total', 0) for t in trades])
        total_exit_costs = sum([t.get('exit_costs', {}).get('total', 0) for t in trades])
        total_costs = total_entry_costs + total_exit_costs
        
        # Calculate equity curve and drawdown
        daily_equity = self.calculate_equity_curve()
        max_drawdown = self.calculate_max_drawdown(daily_equity)
        
        # Get daily P&L from risk manager
        daily_pnl = self.risk_manager.get_daily_performance()
        
        results = {
            "status": "completed",
            "num_trades": num_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "total_pnl": total_profit - total_loss,
            "total_costs": total_costs,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "equity_curve": daily_equity,
            "daily_pnl": daily_pnl,
            "trades": trades
        }
        
        # Add regime analysis to results
        regime_stats = {}
        for trade in trades:
            regime = trade.get('market_regime', 'unknown')
            if regime not in regime_stats:
                regime_stats[regime] = {
                    'count': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0
                }
            
            stats = regime_stats[regime]
            stats['count'] += 1
            pnl = trade.get('total_net_pnl', 0)
            stats['total_pnl'] += pnl
            if pnl > 0:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
        
        # Calculate win rates by regime
        for regime, stats in regime_stats.items():
            if stats['count'] > 0:
                stats['win_rate'] = stats['wins'] / stats['count']
            else:
                stats['win_rate'] = 0
        
        results['regime_statistics'] = regime_stats
        results['regime_history'] = self.regime_history
        
        return results
    
    def calculate_equity_curve(self) -> pd.Series:
        """
        Calculate daily equity curve from trade log.
        
        Returns:
            Series with daily equity values
        """
        try:
            # Extract daily P&L from trade log
            daily_pnl = {}
            
            for entry in self.trade_log:
                date = entry['timestamp'].date()
                pnl = entry['daily_pnl']
                
                # Store the last P&L value for each day
                daily_pnl[date] = pnl
            
            # Convert to series and calculate cumulative equity
            if not daily_pnl:
                logger.warning("No daily P&L data available for equity curve calculation")
                return pd.Series()
            
            # Convert to series and sort by date
            equity_series = pd.Series(daily_pnl)
            equity_series.index = pd.to_datetime(equity_series.index)
            equity_series = equity_series.sort_index()
            
            # Calculate running equity starting from initial capital
            starting_capital = self.risk_manager.total_capital
            equity_curve = starting_capital + equity_series.cumsum()
            
            # Validate equity curve
            if len(equity_curve) > 0:
                min_equity = equity_curve.min()
                max_equity = equity_curve.max()
                logger.info(f"Equity curve generated: {len(equity_curve)} points, "
                           f"Min: {min_equity:.2f}, Max: {max_equity:.2f}")
                
                # Calculate and log max drawdown for verification
                running_max = equity_curve.cummax()
                drawdown = (equity_curve - running_max) / running_max
                max_dd = abs(drawdown.min()) * 100
                logger.info(f"Max drawdown calculated: {max_dd:.2f}%")
            
            return equity_curve
            
        except Exception as e:
            logger.error(f"Error calculating equity curve: {e}", exc_info=True)
            return pd.Series()
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: Series with equity values
            
        Returns:
            Maximum drawdown as a percentage
        """
        if len(equity_curve) < 2:
            return 0.0
            
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown in percentage terms
        drawdown = (equity_curve - running_max) / running_max
        
        # Get maximum drawdown
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)