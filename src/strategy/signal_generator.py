"""
Signal generator for the options trading strategy.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import traceback

from src.strategy.rules import generate_entry_rules, generate_exit_rules

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Generates trading signals based on market data and strategy rules.
    """
    
    def __init__(self, options_sentiment: Dict[str, Any] = None, 
                 mandatory_conditions: List[str] = None,
                 min_conditions_required: int = 5,
                 trading_engine = None,
                 weighted_scoring: bool = False,
                 market_regime_adaptation: bool = False,
                 optimal_strikes: bool = False):
        """
        Initialize the signal generator.
        
        Args:
            options_sentiment: Dictionary with options sentiment indicators
            mandatory_conditions: List of condition names that must be satisfied
            min_conditions_required: Minimum number of conditions required for a valid signal
            trading_engine: Reference to the trading engine for market regime detection
            weighted_scoring: Whether to use weighted condition scoring instead of binary checks
            market_regime_adaptation: Whether to adapt strategy to different market regimes
            optimal_strikes: Whether to use optimized strike selection
        """
        self.options_sentiment = options_sentiment or {}
        self.signals_history = []
        self.error_count = 0  # Add error tracking
        self.signal_attempts = 0
        self.debug_mode = False  # Add debug mode
        self.mandatory_conditions = mandatory_conditions or ["cvd", "vwap", "gamma"]  # Added "cvd"
        self.min_conditions_required = min_conditions_required
        self.trading_engine = trading_engine
        # Add caches for day transition
        self._pcr_cache = {}
        self._oi_cache = {}
        # Store new feature flags
        self.weighted_scoring = weighted_scoring
        self.market_regime_adaptation = market_regime_adaptation
        self.optimal_strikes = optimal_strikes
        logger.info(f"Signal generator initialized with mandatory conditions: {self.mandatory_conditions}")
        if weighted_scoring:
            logger.info("Weighted scoring enabled")
        if market_regime_adaptation:
            logger.info("Market regime adaptation enabled")
        if optimal_strikes:
            logger.info("Optimal strike selection enabled")
    
    def update_sentiment(self, options_sentiment: Dict[str, Any]):
        """
        Update options sentiment data.
        
        Args:
            options_sentiment: Dictionary with options sentiment indicators
        """
        self.options_sentiment = options_sentiment
        logger.debug("Options sentiment updated")
    
    def enable_debug_mode(self):
        """Enable detailed debug logging."""
        self.debug_mode = True
        logger.info("Signal generator debug mode enabled")
    
    def disable_debug_mode(self):
        """Disable detailed debug logging."""
        self.debug_mode = False
    
    def check_mandatory_conditions(self, conditions: Dict[str, bool]) -> bool:
        """
        Check if all mandatory conditions are satisfied.
        
        Args:
            conditions: Dictionary mapping condition names to boolean results
            
        Returns:
            True if all mandatory conditions are satisfied, False otherwise
        """
        # If no mandatory conditions specified, default to requiring any one
        if not self.mandatory_conditions:
            return True
        
        # Check each mandatory condition
        for condition_name in self.mandatory_conditions:
            if condition_name in conditions and conditions[condition_name]:
                # Return True if any mandatory condition is met
                return True
        
        # No mandatory conditions were met
        return False
    
    def set_mandatory_conditions(self, mandatory_conditions: List[str]):
        """
        Set the mandatory conditions that must be satisfied.
        
        Args:
            mandatory_conditions: List of condition names that must be satisfied
        """
        self.mandatory_conditions = mandatory_conditions
        logger.info(f"Updated mandatory conditions: {self.mandatory_conditions}")
    
    def set_min_conditions_required(self, min_conditions: int):
        """
        Set the minimum number of conditions required for a valid signal.
        
        Args:
            min_conditions: Minimum number of conditions
        """
        self.min_conditions_required = min_conditions
        logger.info(f"Updated minimum conditions required: {self.min_conditions_required}")

    def filter_top_signals(self, signals: Dict[str, Any], max_signals: int = 3) -> Dict[str, Any]:
        """
        Filter to keep only the top N signals with the highest quality score.
        Prevents duplicate signals for the same option parameters.
        
        Args:
            signals: Dictionary with signal information
            max_signals: Maximum number of signals to keep
            
        Returns:
            Filtered signals dictionary
        """
        if 'entry_signals' not in signals or not signals['entry_signals']:
            return signals
        
        # Get entry signals
        entry_signals = signals['entry_signals']
        
        # Calculate quality scores for each signal
        for signal in entry_signals:
            option = signal.get('option', {})
            conditions = signal.get('conditions', {})
            conditions_met = signal.get('conditions_met', 0)
            
            # Quality factors
            delta_score = abs(option.get('delta', 0)) * 10  # 0-10 points for delta strength
            gamma_score = min(option.get('gamma', 0) * 100, 5)  # 0-5 points for gamma
            vega_score = min(option.get('vega', 0), 3)  # 0-3 points for vega
            
            # Condition-based score
            condition_score = conditions_met * 2  # 2 points per condition
            
            # CVD confirmation adds value
            cvd_bonus = 3 if conditions.get('cvd', False) else 0
            
            # Calculate total quality score
            signal['quality_score'] = delta_score + gamma_score + vega_score + condition_score + cvd_bonus
        
        # Deduplicate signals with the same option parameters
        unique_signals = {}
        for signal in entry_signals:
            # Create a unique key for each option
            option_key = (
                signal.get('option_type', ''), 
                signal.get('strike_price', 0),
                str(signal.get('expiry_date', ''))
            )
            
            # Keep only the highest quality signal for each unique option
            if option_key not in unique_signals or signal.get('quality_score', 0) > unique_signals[option_key].get('quality_score', 0):
                unique_signals[option_key] = signal
                
                if self.debug_mode:
                    logger.debug(f"Signal for {option_key}: Quality={signal.get('quality_score', 0):.1f}")
        
        # Convert back to list and sort by quality score
        deduplicated_signals = list(unique_signals.values())
        sorted_signals = sorted(deduplicated_signals, key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Log the top signals for debugging
        if sorted_signals and self.debug_mode:
            top_signal = sorted_signals[0]
            logger.debug(f"Top signal quality: {top_signal.get('quality_score', 0):.1f}, "
                        f"Strike: {top_signal.get('strike_price')}, "
                        f"Type: {top_signal.get('option_type')}, "
                        f"Conditions met: {top_signal.get('conditions_met', 0)}")
        
        # Keep only top N signals
        signals['entry_signals'] = sorted_signals[:max_signals]
        
        return signals

    def generate_signals(self,
                       timestamp: datetime,
                       spot_data: pd.DataFrame,
                       futures_data: pd.DataFrame,
                       options_data: Dict[str, pd.DataFrame],
                       open_positions: Optional[Dict[str, Dict[str, Any]]] = None,
                       market_regime: str = None) -> Dict[str, Any]:
        """
        Generate trading signals for the current timestamp with improved error handling.
        
        Args:
            timestamp: Current timestamp
            spot_data: Spot market data DataFrame
            futures_data: Futures market data DataFrame
            options_data: Dictionary of options data DataFrames
            open_positions: Dictionary of currently open positions
            market_regime: Optional market regime override. If not provided, will be auto-detected
            
        Returns:
            Dictionary containing generated signals
        """
        logger.debug(f"Generating signals for {timestamp}")
        self.signal_attempts += 1
        
        try:
            # Input validation
            if (spot_data is None or spot_data.empty) and (futures_data is None or futures_data.empty):
                logger.warning(f"Empty spot and futures data at {timestamp}")
                return self._create_empty_signals(timestamp)
            
            # If spot data is empty but futures data is available, use futures data as a substitute
            if spot_data is None or spot_data.empty:
                if futures_data is not None and not futures_data.empty:
                    logger.debug(f"Using futures data as substitute for empty spot data at {timestamp}")
                    
                    # Determine the column names in futures_data
                    futures_columns = futures_data.columns.tolist()
                    
                    # Map standard column names to possible alternatives
                    column_mapping = {
                        'tr_open': ['tr_open', 'open', 'Open', 'OPEN'],
                        'tr_high': ['tr_high', 'high', 'High', 'HIGH'],
                        'tr_low': ['tr_low', 'low', 'Low', 'LOW'],
                        'tr_close': ['tr_close', 'close', 'Close', 'CLOSE']
                    }
                    
                    # Find the actual column names in the futures data
                    actual_columns = {}
                    for standard_col, alternatives in column_mapping.items():
                        for alt_col in alternatives:
                            if alt_col in futures_columns:
                                actual_columns[standard_col] = alt_col
                                break
                        if standard_col not in actual_columns:
                            logger.warning(f"Could not find a suitable column for {standard_col} in futures data")
                            # Default to the last column as a fallback
                            if futures_columns:
                                actual_columns[standard_col] = futures_columns[-1]
                    
                    # Create a copy of spot data with futures price
                    try:
                        spot_data = pd.DataFrame({
                            'tr_open': [futures_data[actual_columns.get('tr_open', futures_columns[-1])].iloc[-1]],
                            'tr_high': [futures_data[actual_columns.get('tr_high', futures_columns[-1])].iloc[-1]],
                            'tr_low': [futures_data[actual_columns.get('tr_low', futures_columns[-1])].iloc[-1]],
                            'tr_close': [futures_data[actual_columns.get('tr_close', futures_columns[-1])].iloc[-1]]
                        }, index=[timestamp])
                        logger.debug(f"Successfully created spot data from futures data with columns: {actual_columns}")
                    except Exception as e:
                        logger.error(f"Error creating spot data from futures: {e}")
                        # Create a simple DataFrame with a single value as a last resort
                        if len(futures_data.columns) > 0:
                            value = futures_data.iloc[-1, 0]  # Use the first column's last value
                            spot_data = pd.DataFrame({
                                'tr_open': [value],
                                'tr_high': [value],
                                'tr_low': [value],
                                'tr_close': [value]
                            }, index=[timestamp])
                            logger.debug(f"Created fallback spot data with value: {value}")
                else:
                    logger.warning(f"Empty spot data at {timestamp}")
                    return self._create_empty_signals(timestamp)
                
            if futures_data is None or futures_data.empty:
                logger.warning(f"Empty futures data at {timestamp}")
                return self._create_empty_signals(timestamp)
                
            if not options_data:
                logger.warning(f"No options data at {timestamp}")
                return self._create_empty_signals(timestamp)

            # Pre-validate and standardize options data
            valid_options_data = {}
            required_columns = {'strike', 'delta', 'gamma', 'vega', 'theta', 'rho', 'iv'}
            column_mappings = {
                'otype': ['option_type', 'type'],
                'expiry_date': ['week_expiry_date', 'expiration', 'expiry'],
                'strike': ['strike_price'],
                'iv': ['implied_volatility'],
                'oi': ['open_interest'],
                'vol': ['volume', 'tr_volume']
            }

            for expiry, expiry_data in options_data.items():
                if expiry_data.empty:
                    logger.debug(f"Skipping empty options data for expiry {expiry}")
                    continue

                try:
                    # Create a copy to avoid modifying original data
                    expiry_df = expiry_data.copy()

                    # Standardize column names
                    for target_col, source_cols in column_mappings.items():
                        if target_col not in expiry_df.columns:
                            for source_col in source_cols:
                                if source_col in expiry_df.columns:
                                    expiry_df[target_col] = expiry_df[source_col]
                                    break

                    # Validate required columns
                    missing_cols = required_columns - set(expiry_df.columns)
                    if missing_cols:
                        logger.warning(f"Missing required columns for expiry {expiry}: {missing_cols}")
                        continue

                    # Additional data validation
                    expiry_df = expiry_df[~expiry_df['strike'].isna()]  # Remove rows with missing strikes
                    expiry_df = expiry_df[expiry_df['strike'] > 0]  # Remove invalid strikes

                    if self.debug_mode:
                        logger.debug(f"Validated options data for expiry {expiry}: {len(expiry_df)} rows")
                        logger.debug(f"Columns: {expiry_df.columns.tolist()}")

                    valid_options_data[expiry] = expiry_df

                except Exception as e:
                    logger.error(f"Error processing options data for expiry {expiry}: {e}")
                    if self.debug_mode:
                        logger.error(traceback.format_exc())

            if not valid_options_data:
                logger.warning("No valid options data after validation")
                return self._create_empty_signals(timestamp)

            # Use provided market regime or detect if not provided
            if market_regime is None:
                market_regime = "normal"
                if self.trading_engine and self.market_regime_adaptation:
                    try:
                        market_regime = self.trading_engine.detect_market_regime(futures_data)
                        logger.debug(f"Detected market regime: {market_regime}")
                    except Exception as e:
                        logger.warning(f"Failed to detect market regime: {e}")
            else:
                logger.debug(f"Using provided market regime: {market_regime}")
            
            # Adapt conditions based on market regime
            original_conditions = self.mandatory_conditions.copy()
            original_min_conditions = self.min_conditions_required
            
            try:
                if self.market_regime_adaptation:
                    if market_regime == "uptrend" or market_regime == "downtrend":
                        self.mandatory_conditions = ["cvd", "delta"]
                        self.min_conditions_required = 4
                        logger.debug("Using trending market conditions configuration")
                    elif market_regime == "volatile":
                        self.mandatory_conditions = ["gamma", "vwap", "delta"]
                        self.min_conditions_required = 6
                        logger.debug("Using volatile market conditions configuration")
                    elif market_regime == "range":
                        self.mandatory_conditions = ["oi", "pcr"]
                        self.min_conditions_required = 5
                        logger.debug("Using range-bound market conditions configuration")
                
                # Filter data up to current timestamp
                current_spot = spot_data.loc[:timestamp]
                current_futures = futures_data.loc[:timestamp]
                
                current_options = {}
                for expiry, expiry_data in valid_options_data.items():
                    current_options[expiry] = expiry_data.loc[:timestamp]
                
                # Generate entry signals with regime information
                try:
                    entry_rules = generate_entry_rules(
                        timestamp, futures_data, valid_options_data, self.options_sentiment,
                        mandatory_conditions=self.mandatory_conditions,
                        min_conditions_required=self.min_conditions_required,
                        market_regime=market_regime  # Pass regime to rules
                    )
                    
                    # Add quality scoring for entry signals
                    condition_weights = {
                        'delta': 2.0,   # Strong directional bias
                        'gamma': 1.5,   # Price sensitivity
                        'oi': 1.0,      # Open interest changes 
                        'cvd': 2.0,     # Volume-price relationship
                        'vwap': 1.5,    # Price vs VWAP
                        'pcr': 1.0,     # Put-call ratio trend
                        'iv': 1.0       # Implied volatility
                    }
                    
                    scored_signals = []
                    for signal in entry_rules.get('signals', []):
                        conditions = signal.get('conditions', {})
                        total_score = 0
                        total_weight = sum(condition_weights.values())
                        
                        # Calculate weighted score
                        for condition, is_met in conditions.items():
                            weight = condition_weights.get(condition, 1.0)
                            score = 10 if is_met else 0  # 10 points if condition met
                            total_score += score * weight
                        
                        # Normalize to 0-100 scale
                        signal_quality = (total_score / (total_weight * 10)) * 100
                        
                        # Only include signals above quality threshold
                        if signal_quality >= 60:
                            signal['quality_score'] = signal_quality
                            scored_signals.append(signal)
                        elif self.debug_mode:
                            logger.debug(f"Signal rejected - quality score {signal_quality:.1f} below threshold")
                    
                    entry_rules['signals'] = scored_signals
                    
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Error generating entry rules at {timestamp}: {e}")
                    if self.debug_mode:
                        logger.error(traceback.format_exc())
                    entry_rules = {'signals': [], 'direction': 0}
                
                # Generate exit signals with regime information
                exit_signals = []
                if open_positions:
                    exit_signals = generate_exit_rules(
                        timestamp, open_positions, current_futures, current_options, 
                        self.options_sentiment, market_regime=market_regime  # Pass regime to exit rules
                    )
                
                # Combine signals
                signals = {
                    'timestamp': timestamp,
                    'entry_signals': entry_rules.get('signals', []),
                    'exit_signals': exit_signals,
                    'direction': entry_rules.get('direction', 0)
                }
                
                # Filter to top signals (now using quality scores)
                signals = self.filter_top_signals(signals, max_signals=3)
                
                # Add debug logging for quality scores
                if self.debug_mode and signals['entry_signals']:
                    for signal in signals['entry_signals']:
                        logger.debug(f"Signal quality score: {signal.get('quality_score', 0):.1f}, "
                                   f"Strike: {signal.get('strike_price')}, "
                                   f"Type: {signal.get('option_type')}")
                
                # Log signal count
                logger.debug(f"Generated {len(signals['entry_signals'])} entry signals and {len(signals['exit_signals'])} exit signals")
                
                # Add to history
                self.signals_history.append(signals)
                
                # Add detailed signal logging in debug mode
                if self.debug_mode and len(signals['entry_signals']) > 0:
                    for i, signal in enumerate(signals['entry_signals']):
                        logger.debug(f"Entry signal {i+1}: {signal['option_type']} {signal['strike_price']} with {signal['conditions_met']} conditions")
                        logger.debug(f"Conditions: {signal['conditions']}")
                
                return signals
                
            finally:
                # Restore original conditions
                self.mandatory_conditions = original_conditions
                self.min_conditions_required = original_min_conditions
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Unhandled error in generate_signals at {timestamp}: {e}")
            logger.error(traceback.format_exc())
            return self._create_empty_signals(timestamp)

    def _create_empty_signals(self, timestamp: datetime) -> Dict[str, Any]:
        """Create an empty signals dictionary for error cases."""
        return {
            'timestamp': timestamp,
            'entry_signals': [],
            'exit_signals': [],
            'direction': 0
        }

    def get_signals_history(self) -> List[Dict[str, Any]]:
        """
        Get history of generated signals.
        
        Returns:
            List of signal dictionaries
        """
        return self.signals_history
    
    def get_latest_signals(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get the most recent signals.
        
        Args:
            count: Number of recent signals to retrieve
            
        Returns:
            List of recent signal dictionaries
        """
        return self.signals_history[-count:] if self.signals_history else []
    
    def get_error_rate(self) -> float:
        """
        Get the error rate for signal generation.
        
        Returns:
            Error rate as a decimal (0.0 to 1.0)
        """
        if self.signal_attempts == 0:
            return 0.0
        return self.error_count / self.signal_attempts

    def analyze_signals_performance(self) -> Dict[str, Any]:
        """
        Analyze performance of generated signals with enhanced metrics.
        """
        if not self.signals_history:
            return {
                'signal_count': 0,
                'entry_signal_count': 0,
                'exit_signal_count': 0,
                'error_count': self.error_count,
                'error_rate': self.get_error_rate()
            }
        
        entry_signals_count = sum(len(s.get('entry_signals', [])) for s in self.signals_history)
        exit_signals_count = sum(len(s.get('exit_signals', [])) for s in self.signals_history)
        
        # Calculate average conditions met for entry signals
        total_conditions = 0
        condition_count = 0
        
        for signal in self.signals_history:
            for entry in signal.get('entry_signals', []):
                if 'conditions_met' in entry:
                    total_conditions += entry['conditions_met']
                    condition_count += 1
        
        avg_conditions = total_conditions / condition_count if condition_count > 0 else 0
        
        # Calculate signal generation by day
        signals_by_day = {}
        for signal in self.signals_history:
            day = signal['timestamp'].date()
            if day not in signals_by_day:
                signals_by_day[day] = {'entry': 0, 'exit': 0}
            signals_by_day[day]['entry'] += len(signal.get('entry_signals', []))
            signals_by_day[day]['exit'] += len(signal.get('exit_signals', []))
        
        return {
            'signal_count': entry_signals_count + exit_signals_count,
            'entry_signal_count': entry_signals_count,
            'exit_signal_count': exit_signals_count,
            'avg_conditions_met': avg_conditions,
            'error_count': self.error_count,
            'error_rate': self.get_error_rate(),
            'signals_by_day': signals_by_day
        }

    def on_new_trading_day(self, previous_date, current_date):
        """
        Handle the transition to a new trading day.
        This method preserves indicators and sentiment data across trading days.
        
        Args:
            previous_date: Date of the previous trading day
            current_date: Date of the new trading day
        """
        logger.info(f"Signal generator handling day transition: {previous_date} -> {current_date}")
        
        # We want to preserve sentiment data across days, but we need to avoid any
        # stale timestamps that might incorrectly influence the current trading day
        
        # 1. Preserve PCR data but ensure it's properly carried forward
        if 'pcr' in self.options_sentiment and not self.options_sentiment['pcr'].empty:
            pcr_df = self.options_sentiment['pcr']
            
            # Store the last values of each MA to use at the start of the new day
            pcr_last_value = pcr_df['pcr'].iloc[-1] if not pcr_df.empty else 1.0
            pcr_ma5_last = pcr_df['pcr_ma5'].iloc[-1] if 'pcr_ma5' in pcr_df.columns and not pcr_df.empty else pcr_last_value
            pcr_ma_short_last = pcr_df['pcr_ma_short'].iloc[-1] if 'pcr_ma_short' in pcr_df.columns and not pcr_df.empty else pcr_last_value
            pcr_ma_long_last = pcr_df['pcr_ma_long'].iloc[-1] if 'pcr_ma_long' in pcr_df.columns and not pcr_df.empty else pcr_last_value
            
            logger.info(f"Preserving PCR values across days: PCR={pcr_last_value:.2f}, MA5={pcr_ma5_last:.2f}, MA_Short={pcr_ma_short_last:.2f}, MA_Long={pcr_ma_long_last:.2f}")
            
            # Store these values in a cache that can be accessed by the PCR calculator
            self._pcr_cache = {
                'last_pcr': pcr_last_value,
                'last_pcr_ma5': pcr_ma5_last,
                'last_pcr_ma_short': pcr_ma_short_last,
                'last_pcr_ma_long': pcr_ma_long_last,
                'previous_date': previous_date
            }
        
        # 2. Preserve OI velocity data for context
        if 'oi_velocity' in self.options_sentiment:
            # Store significant OI changes from the previous day to provide context
            oi_velocity = self.options_sentiment['oi_velocity']
            if oi_velocity:
                # Find the last timestamp of the previous day
                previous_day_timestamps = [ts for ts in oi_velocity.keys() 
                                         if isinstance(ts, pd.Timestamp) and ts.date() == previous_date]
                
                if previous_day_timestamps:
                    last_ts = max(previous_day_timestamps)
                    significant_changes = oi_velocity[last_ts].get('significant_oi_changes', [])
                    oi_sentiment = oi_velocity[last_ts].get('oi_sentiment', 0)
                    
                    # Store this data to be accessible by the OI calculator
                    self._oi_cache = {
                        'last_significant_changes': significant_changes,
                        'last_oi_sentiment': oi_sentiment,
                        'previous_date': previous_date
                    }
                    
                    logger.info(f"Preserved OI data with {len(significant_changes)} significant changes and sentiment {oi_sentiment:.2f}")
        
        # 3. Log information about signals generated on the previous day
        previous_day_signals = [s for s in self.signals_history 
                              if s.get('timestamp') and s.get('timestamp').date() == previous_date]
        
        entry_count = sum(len(s.get('entry_signals', [])) for s in previous_day_signals)
        exit_count = sum(len(s.get('exit_signals', [])) for s in previous_day_signals)
        
        logger.info(f"Previous day {previous_date} generated {entry_count} entry signals and {exit_count} exit signals")
        
        # 4. Reset any daily counters or state that shouldn't carry over
        # (Keep the signals history and error tracking intact)
        
        # 5. Clear any temporary data structures that should be rebuilt each day
        # but preserve the caches we explicitly want to keep
        
        # Log the transition completion
        logger.info(f"Day transition complete: Ready for {current_date}")