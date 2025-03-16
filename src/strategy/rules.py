"""
Entry and exit rules for the options trading strategy.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from functools import lru_cache

from config.strategy_params import (
    CALL_DELTA_THRESHOLD, PUT_DELTA_THRESHOLD, GAMMA_THRESHOLD, 
    OI_CHANGE_THRESHOLD, PROFIT_TARGET_1, PROFIT_TARGET_2, EXPIRY_DAY_CUTOFF_TIME,
    IV_PERCENTILE_THRESHOLD
)
from src.utils.date_utils import get_next_expiry_date

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1024)
def check_delta_condition_cached(delta: float, option_type: str) -> bool:
    """
    Cached version of delta condition check using primitive values.
    
    Args:
        delta: Option delta value
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        True if delta condition is met, False otherwise
    """
    if delta is None:
        return False
    
    return delta >= CALL_DELTA_THRESHOLD if option_type == 'CE' else delta <= PUT_DELTA_THRESHOLD

@lru_cache(maxsize=1024)
def check_gamma_condition_cached(gamma: float) -> bool:
    """
    Cached version of gamma condition check using primitive values.
    Modified to be more permissive with low gamma values.
    
    Args:
        gamma: Option gamma value
        
    Returns:
        True if gamma condition is met, False otherwise
    """
    if gamma is None:
        return False
    
    # Use a much lower threshold for gamma (1/10th of original)
    # This is more permissive while still filtering extreme low values
    return gamma >= (GAMMA_THRESHOLD / 10)

@lru_cache(maxsize=1024)
def check_price_direction_cached(current_price: float, previous_price: float) -> int:
    """Cache price direction calculation."""
    return 1 if current_price > previous_price else -1

@lru_cache(maxsize=1024)
def check_pcr_threshold_cached(pcr: float, option_type: str) -> bool:
    """Cache PCR threshold check."""
    if option_type == 'CE':
        return pcr < 1.0
    else:  # PE
        return pcr > 1.0

@lru_cache(maxsize=1024)
def check_iv_condition_cached(iv_percentile: Optional[float]) -> bool:
    """
    Cached version of IV condition check.
    
    Args:
        iv_percentile: IV percentile value
        
    Returns:
        True if IV condition is met, False otherwise
    """
    if iv_percentile is None:
        return True  # Be permissive if data is missing
    
    return iv_percentile <= IV_PERCENTILE_THRESHOLD

def check_delta_condition(option_data: Dict[str, Any], option_type: str) -> bool:
    """
    Check if delta meets the entry condition. Uses cached computation.
    
    Args:
        option_data: Dictionary with option data
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        True if delta condition is met, False otherwise
    """
    if 'delta' not in option_data:
        logger.warning("Delta not found in option data")
        return False
    
    return check_delta_condition_cached(option_data['delta'], option_type)

def check_gamma_condition(option_data: Dict[str, Any]) -> bool:
    """
    Check if gamma meets the entry condition. Uses cached computation.
    
    Args:
        option_data: Dictionary with option data
        
    Returns:
        True if gamma condition is met, False otherwise
    """
    if 'gamma' not in option_data:
        logger.warning("Gamma not found in option data")
        return False
    
    return check_gamma_condition_cached(option_data['gamma'])

@lru_cache(maxsize=1024)
def check_oi_threshold_cached(oi_change_pct: float, threshold: float) -> bool:
    """Cache OI threshold check."""
    return oi_change_pct >= threshold

def check_oi_change_condition(option_data: Dict[str, Any], 
                            oi_velocity: Dict[pd.Timestamp, Dict[str, Any]],
                            timestamp: pd.Timestamp) -> bool:
    """
    Check if OI change meets the entry condition with enhanced analysis.
    
    Args:
        option_data: Dictionary with option data
        oi_velocity: Dictionary with OI velocity data
        timestamp: Current timestamp
        
    Returns:
        True if condition met, False otherwise
    """
    # Early validation
    if not isinstance(option_data, dict) or 'otype' not in option_data or 'strike_price' not in option_data:
        logger.warning("Invalid option data format")
        return True  # Be permissive with invalid data
    
    # Check if we have OI velocity data for this timestamp
    if timestamp not in oi_velocity:
        logger.debug(f"No OI velocity data for timestamp {timestamp}")
        return True  # More permissive - assume condition is met when data is missing
    
    # Extract required data
    velocity_data = oi_velocity[timestamp]
    option_type = option_data['otype']
    strike_price = option_data['strike_price']
    
    try:
        # Check OI concentration first (most reliable signal)
        concentration = velocity_data.get('oi_concentration', {})
        if concentration:
            strikes_to_check = concentration.get('calls' if option_type == 'CE' else 'puts', [])
            for strike_info in strikes_to_check:
                if strike_info['strike'] == strike_price:
                    logger.debug(f"Found concentrated {option_type} OI at strike {strike_price}")
                    return True
        
        # Check OI-price divergence
        divergences = velocity_data.get('divergences', [])
        for div in divergences:
            if div['otype'] == option_type and div['strike_price'] == strike_price:
                logger.debug(f"Found OI-price divergence at {option_type} {strike_price}")
                return True
        
        # Check significant OI changes
        significant_changes = velocity_data.get('significant_oi_changes', [])
        if significant_changes:
            from config.strategy_params import OI_CHANGE_THRESHOLD
            oi_threshold = OI_CHANGE_THRESHOLD * 0.5  # 50% of original threshold
            
            for change in significant_changes:
                if (change['otype'] == option_type and 
                    change['strike_price'] == strike_price):
                    # Use cached threshold check
                    if check_oi_threshold_cached(change.get('oi_change_pct', 0), oi_threshold):
                        logger.debug(f"Significant OI change detected: {change.get('oi_change_pct')}%")
                        return True
        
        # Check broader OI sentiment as fallback
        oi_sentiment = velocity_data.get('oi_sentiment', 0)
        if abs(oi_sentiment) > 0:  # Only consider if we have non-zero sentiment
            sentiment_aligned = (option_type == 'CE' and oi_sentiment > 0) or \
                              (option_type == 'PE' and oi_sentiment < 0)
            if sentiment_aligned:
                logger.debug(f"OI sentiment {oi_sentiment} aligns with {option_type}")
                return True
        
        logger.debug(f"No significant OI signals found for {option_type} {strike_price}")
        return False
        
    except Exception as e:
        logger.error(f"Error in OI change condition check: {e}")
        return True  # Be permissive on errors

def check_cvd_breakout_condition(futures_data: pd.DataFrame, option_type: str) -> bool:
    """
    Check if CVD breakout condition is met with improved error handling
    and more permissive behavior during early backtest stages.
    
    Args:
        futures_data: Futures market data
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        True if condition met, False otherwise. Returns True if data is missing
        to be more permissive.
    """
    if futures_data is None or futures_data.empty:
        logger.debug("Empty futures data for CVD check")
        return True  # Be permissive with missing data
        
    if 'cvd_signal' not in futures_data.columns:
        logger.debug("CVD signal not found in futures data")
        return True  # Be permissive when indicator is missing
    
    try:
        # Extract the last (most recent) value as a scalar
        # This is critical to avoid Series in boolean context
        if len(futures_data) > 0:
            cvd_signal = futures_data['cvd_signal'].iloc[-1]
            
            # Handle NaN values
            if pd.isna(cvd_signal):
                logger.debug("NaN value in CVD signal")
                return True  # Be permissive with missing data
            
            # Check data length - be more permissive in the early stages of backtesting
            if len(futures_data) < 30:  # First 30 minutes of data
                logger.debug(f"Limited data points ({len(futures_data)}) for CVD signal, being more permissive")
                # For early backtest stages, use weaker signal threshold or allow trading regardless
                if option_type == 'CE':
                    result = cvd_signal >= -0.5  # Much more permissive threshold
                else:  # PE
                    result = cvd_signal <= 0.5   # Much more permissive threshold
                logger.debug(f"Using permissive early-stage CVD condition: signal={cvd_signal}, result={result}")
                return result
                
            # Standard check for established data periods
            result = check_cvd_breakout_condition_cached(cvd_signal, option_type)
            logger.debug(f"CVD condition: signal={cvd_signal}, result={result}")
            return result
        else:
            logger.debug("Empty futures data after filtering")
            return True  # Be permissive with missing data
    except Exception as e:
        logger.warning(f"Error in CVD condition check: {e}")
        return True  # Be permissive on errors

def check_vwap_condition(futures_data: pd.DataFrame, option_type: str) -> bool:
    """
    Check if VWAP condition is met with improved error handling.
    Considers both price vs VWAP and VWAP signal strength if available.
    
    Args:
        futures_data: Futures market data
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        True if condition met, False otherwise. Returns True if data is missing
        to be more permissive.
    """
    required_columns = ['tr_close', 'vwap']
    
    if futures_data is None or futures_data.empty:
        logger.debug("Empty futures data, VWAP condition check failed")
        return True  # Be permissive with missing data
        
    if not all(col in futures_data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in futures_data.columns]
        logger.debug(f"Missing VWAP columns in futures data: {missing}")
        return True  # Be permissive with missing columns
    
    try:
        # Extract the last (most recent) value as scalars
        if len(futures_data) > 0:
            current_price = futures_data['tr_close'].iloc[-1]
            vwap = futures_data['vwap'].iloc[-1]
            
            # Check for VWAP breakout signal if available
            if 'vwap_signal' in futures_data.columns:
                signal_value = futures_data['vwap_signal'].iloc[-1]
                
                # Handle NaN signal values
                if not pd.isna(signal_value):
                    # Strong signal (1.5/-1.5) indicates VWAP slope alignment
                    if (option_type == 'CE' and signal_value >= 1.0) or \
                       (option_type == 'PE' and signal_value <= -1.0):
                        logger.debug(f"VWAP signal strength: {signal_value}")
                        # Give higher confidence to stronger signals
                        return abs(signal_value) > 1.0  # Returns True for signals with magnitude > 1
            
            # Handle NaN values in price/VWAP
            if pd.isna(current_price) or pd.isna(vwap):
                logger.debug("NaN values in VWAP condition data")
                return True  # Be permissive with NaN values
                
            # Fall back to basic VWAP cross check if no signal or weak signal
            result = check_vwap_condition_cached(current_price, vwap, option_type)
            logger.debug(f"VWAP condition: {current_price} vs {vwap}, result: {result}")
            return result
        else:
            logger.debug("Empty futures data after filtering")
            return True  # Be permissive with missing data
    except Exception as e:
        logger.warning(f"Error in VWAP condition check: {e}")
        return True  # Be permissive on errors

def check_pcr_condition(pcr_data: pd.DataFrame, option_type: str, timestamp: pd.Timestamp) -> bool:
    """
    Check if PCR condition is met using multiple timeframes.
    
    Args:
        pcr_data: DataFrame with PCR data
        option_type: Option type ('CE' or 'PE')
        timestamp: Current timestamp
        
    Returns:
        True if PCR condition is met, False otherwise
    """
    # FIX: More permissive PCR condition checking
    if pcr_data.empty or 'pcr' not in pcr_data.columns:
        logger.debug("PCR data not found or empty")
        # Be permissive - don't block trades just because PCR data is missing
        return True
    
    try:
        # Get PCR value for this timestamp
        pcr_at_time = pcr_data.loc[pcr_data.index <= timestamp]
        
        if pcr_at_time.empty:
            logger.debug("No PCR data for timestamp")
            return True  # Be permissive
        
        # Get the last (most recent) values as scalars
        pcr = pcr_at_time['pcr'].iloc[-1]
        
        # Handle NaN values
        if pd.isna(pcr):
            logger.debug("NaN value in PCR data")
            return True
        
        # Check if we have alignment data
        if 'pcr_aligned' in pcr_at_time.columns:
            aligned = pcr_at_time['pcr_aligned'].iloc[-1]
            if aligned:
                # Strong signal when PCR trend is aligned across timeframes
                logger.debug(f"PCR alignment detected, giving stronger condition signal")
                return True
        
        # Check if we have both MA columns
        if ('pcr_ma_short' in pcr_at_time.columns and 
            'pcr_ma_long' in pcr_at_time.columns and 
            len(pcr_at_time) >= 5):
            
            # Get MA values as scalars
            pcr_ma_short = pcr_at_time['pcr_ma_short'].iloc[-1]
            pcr_ma_long = pcr_at_time['pcr_ma_long'].iloc[-1]
            
            # Handle NaN values
            if pd.isna(pcr_ma_short) or pd.isna(pcr_ma_long):
                logger.debug("NaN value in PCR MAs")
                # Fall back to simple check if MAs have NaN
                if option_type == 'CE':
                    result = pcr < 1.1  # More permissive threshold for calls
                    logger.debug(f"PCR simple condition for CE: {pcr} < 1.1 = {result}")
                    return result
                else:  # PE
                    result = pcr > 0.9  # More permissive threshold for puts
                    logger.debug(f"PCR simple condition for PE: {pcr} > 0.9 = {result}")
                    return result
            
            # Enhanced check using both timeframes
            if option_type == 'CE':
                # For calls, we want PCR < short MA < long MA (bearishness decreasing)
                result = pcr < pcr_ma_short < pcr_ma_long
                logger.debug(f"PCR enhanced condition for CE: {pcr} < {pcr_ma_short} < {pcr_ma_long} = {result}")
                return result
            else:  # PE
                # For puts, we want PCR > short MA > long MA (bearishness increasing)
                result = pcr > pcr_ma_short > pcr_ma_long
                logger.debug(f"PCR enhanced condition for PE: {pcr} > {pcr_ma_short} > {pcr_ma_long} = {result}")
                return result
        else:
            # Simple threshold check if we don't have enough data for trend
            if option_type == 'CE':
                result = pcr < 1.1  # More permissive threshold
                logger.debug(f"PCR simple condition for CE: {pcr} < 1.1 = {result}")
                return result
            else:  # PE
                result = pcr > 0.9  # More permissive threshold
                logger.debug(f"PCR simple condition for PE: {pcr} > 0.9 = {result}")
                return result
    except Exception as e:
        logger.warning(f"Error in PCR condition check: {e}")
        return True  # Be permissive in case of errors

def check_iv_condition(option_data: Dict[str, Any]) -> bool:
    """
    Check if IV meets the entry condition.
    Favors options with lower IV percentile for better potential gain.
    
    Args:
        option_data: Dictionary with option data
        
    Returns:
        True if IV condition is met, False otherwise
    """
    if 'iv_percentile' not in option_data:
        logger.debug("IV percentile not found in option data")
        return True  # Be permissive if data is missing
    
    return check_iv_condition_cached(option_data.get('iv_percentile'))

def check_conditions(conditions: Dict[str, bool]) -> Tuple[int, Dict[str, bool]]:
    """
    Check multiple conditions and return count of conditions met.
    
    Args:
        conditions: Dictionary of condition names and results
        
    Returns:
        Tuple of (conditions_met_count, conditions_dict)
    """
    conditions_met = sum(1 for v in conditions.values() if v)
    return conditions_met, conditions

def is_expiry_day(expiry_date: datetime, timestamp: datetime) -> bool:
    """
    Check if the current date is the expiry date.
    
    Args:
        expiry_date: Option expiry date
        timestamp: Current timestamp
        
    Returns:
        True if it's expiry day, False otherwise
    """
    return expiry_date.date() == timestamp.date()

def find_atm_strike(futures_data: pd.DataFrame, options_data: Dict[str, pd.DataFrame], 
                   expiry: str, option_type: str) -> Optional[float]:
    """
    Find the At-The-Money (ATM) strike price for a given expiry.
    
    Args:
        futures_data: DataFrame with futures market data
        options_data: Dictionary of options data by expiry
        expiry: Expiry date string
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        ATM strike price, or None if not found
    """
    # Get current futures price
    if futures_data.empty:
        return None
    
    current_price = futures_data['tr_close'].iloc[-1]
    
    # Check if we have options data for this expiry
    if expiry not in options_data:
        return None
    
    expiry_data = options_data[expiry]
    
    # Get all strikes for this option type
    strikes = expiry_data[expiry_data['otype'] == option_type]['strike_price'].unique()
    if len(strikes) == 0:
        return None
    
    # Find the closest strike to current price
    atm_strike = min(strikes, key=lambda x: abs(x - current_price))
    
    return atm_strike

def find_next_expiry_options(timestamp: datetime, 
                            options_data: Dict[str, pd.DataFrame]) -> Optional[str]:
    """
    Find the next expiry date after the current timestamp.
    
    Args:
        timestamp: Current timestamp
        options_data: Dictionary of options data by expiry
        
    Returns:
        Next expiry date string, or None if not found
    """
    if not options_data:
        return None
    
    # Try to find the next expiry in the options data
    valid_expiries = []
    for expiry in options_data.keys():
        try:
            expiry_date = pd.to_datetime(expiry).date()
            if expiry_date > timestamp.date():
                valid_expiries.append(expiry)
        except:
            continue
    
    if not valid_expiries:
        # Use date utility as fallback
        next_expiry = get_next_expiry_date(timestamp)
        if next_expiry:
            return str(next_expiry)
        return None
    
    # Get the closest expiry date
    return min(valid_expiries, key=lambda x: pd.to_datetime(x).date() - timestamp.date())

@lru_cache(maxsize=256)
def check_expiry_validity_cached(expiry_date: str, current_date: str) -> bool:
    """Cache expiry date validation."""
    return pd.to_datetime(expiry_date).date() >= pd.to_datetime(current_date).date()

@lru_cache(maxsize=256)
def check_direction_cached(cvd_signal: Optional[float], 
                         price_above_vwap: Optional[bool],
                         price_change: float) -> int:
    """Cache direction calculation."""
    if cvd_signal is not None:
        return 1 if cvd_signal > 0 else -1
    if price_above_vwap is not None:
        return 1 if price_above_vwap else -1
    return 1 if price_change > 0 else -1

def filter_strikes_near_atm(
    options_df: pd.DataFrame, 
    futures_price: float, 
    num_strikes: int = 5
) -> pd.DataFrame:
    """
    Filter options to include only strikes near the current price.
    
    Args:
        options_df: DataFrame with options data
        futures_price: Current futures price
        num_strikes: Number of strikes to include above and below current price
        
    Returns:
        Filtered DataFrame
    """
    if options_df.empty:
        return options_df
        
    # Get unique strikes
    unique_strikes = sorted(options_df['strike_price'].unique())
    
    # Find ATM strike (closest to futures price)
    atm_strike = min(unique_strikes, key=lambda x: abs(x - futures_price))
    
    # Find strikes to include
    strike_indices = unique_strikes.index(atm_strike)
    
    # Get strikes above and below ATM
    start_idx = max(0, strike_indices - num_strikes)
    end_idx = min(len(unique_strikes) - 1, strike_indices + num_strikes)
    
    selected_strikes = unique_strikes[start_idx:end_idx + 1]
    
    # Filter options dataframe
    return options_df[options_df['strike_price'].isin(selected_strikes)]

def optimize_strike_selection(options_df: pd.DataFrame, current_price: float, option_type: str) -> pd.DataFrame:
    """
    Filter options to focus on optimal strike range based on delta and gamma/theta ratio.
    
    Args:
        options_df: DataFrame with options data
        current_price: Current underlying price
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        Filtered DataFrame with optimal strikes
    """
    if options_df.empty:
        return options_df
        
    try:
        # First filter by delta range (sweet spot for risk/reward)
        if option_type == 'CE':
            delta_mask = (options_df['delta'] >= 0.35) & (options_df['delta'] <= 0.65)
        else:  # PE
            delta_mask = (options_df['delta'] >= -0.65) & (options_df['delta'] <= -0.35)
        
        delta_filtered = options_df[delta_mask]
        
        # If we have too few options after delta filter, revert to original
        if len(delta_filtered) < 3:
            logger.debug("Too few options after delta filter, using original set")
            return options_df
        
        # Calculate gamma/theta ratio (higher is better)
        if 'theta' in delta_filtered.columns:
            # Avoid division by zero
            delta_filtered['theta_safe'] = delta_filtered['theta'].apply(
                lambda x: x if abs(x) > 0.001 else -0.001
            )
            
            # Calculate ratio (absolute value of gamma / absolute value of theta)
            delta_filtered['gamma_theta_ratio'] = (
                delta_filtered['gamma'].abs() / delta_filtered['theta_safe'].abs()
            )
            
            # Sort by gamma/theta ratio (descending)
            result = delta_filtered.sort_values('gamma_theta_ratio', ascending=False)
            
            # Take top options
            return result.head(5)
        
        return delta_filtered
        
    except Exception as e:
        logger.error(f"Error in optimize_strike_selection: {e}")
        return options_df

def generate_entry_rules(timestamp: pd.Timestamp,
                        futures_data: pd.DataFrame,
                        options_data: Dict[str, pd.DataFrame],
                        options_sentiment: Dict[str, Any],
                        mandatory_conditions: List[str] = None,
                        min_conditions_required: int = 4,
                        market_regime: str = None) -> Dict[str, Any]:
    """
    Apply entry rules to generate trade signals with improved efficiency.
    
    Args:
        timestamp: Current timestamp
        futures_data: DataFrame with futures market data
        options_data: Dictionary of options data by expiry
        options_sentiment: Dictionary with sentiment data
        mandatory_conditions: List of condition names that must be satisfied (default: None)
        min_conditions_required: Minimum number of conditions required for a valid signal (default: 4)
        market_regime: Current market regime (default: None)
    
    Returns:
        Dictionary with direction and signals
    """
    logger.debug(f"Generating entry rules at {timestamp} with market regime: {market_regime}")
    
    entry_signals = []
    
    # Default mandatory conditions if none provided
    if mandatory_conditions is None:
        mandatory_conditions = ["gamma", "vwap"]
        
        # Add regime-specific mandatory conditions
        if market_regime == "trending":
            mandatory_conditions.extend(["cvd", "pcr"])
        elif market_regime == "volatile":
            mandatory_conditions.extend(["iv"])
    
    # Adjust min conditions based on market regime
    if market_regime == "volatile":
        min_conditions_required = max(5, min_conditions_required)  # More stringent in volatile markets
    
    # Pre-calculate common values once
    current_date = timestamp.strftime('%Y-%m-%d')
    expiry_cutoff = datetime.strptime(EXPIRY_DAY_CUTOFF_TIME, "%H:%M:%S").time()
    
    # Extract and validate futures data
    if futures_data.empty:
        return {'direction': 0, 'signals': []}
        
    # Handle potential empty DataFrame after filtering by timestamp
    futures_at_time = futures_data[futures_data.index <= timestamp]
    if futures_at_time.empty:
        return {'direction': 0, 'signals': []}
        
    futures_latest = futures_at_time.iloc[-1]
    
    # Safety check for required fields
    if 'tr_close' not in futures_latest:
        logger.warning(f"Missing tr_close in futures data at {timestamp}")
        return {'direction': 0, 'signals': []}
    
    # Calculate direction more efficiently using cached function
    cvd_signal = futures_latest.get('cvd_signal') if 'cvd_signal' in futures_latest else None
    price_above_vwap = (
        futures_latest['tr_close'] > futures_latest['vwap'] 
        if 'vwap' in futures_latest else None
    )
    
    # Safe access to previous value for price change calculation
    if len(futures_at_time) > 1:
        previous_close = futures_at_time['tr_close'].iloc[-2]
        price_change = futures_latest['tr_close'] - previous_close
    else:
        price_change = 0
    
    direction = check_direction_cached(cvd_signal, price_above_vwap, price_change)
    option_type = 'CE' if direction > 0 else 'PE'
    
    # Extract sentiment data once
    pcr_data = options_sentiment.get('pcr', pd.DataFrame())
    oi_velocity = options_sentiment.get('oi_velocity', {})
    
    # Process each expiry with early filtering
    valid_expiries = {
        expiry: data for expiry, data in options_data.items()
        if check_expiry_validity_cached(expiry, current_date)
    }
    
    for expiry, expiry_data in valid_expiries.items():
        try:
            expiry_date = pd.to_datetime(expiry)
            
            # Filter data up to current timestamp
            expiry_data_at_time = expiry_data[expiry_data.index <= timestamp]
            
            # Skip if no data for this timestamp
            if expiry_data_at_time.empty:
                continue
            
            # Optimize expiry day handling
            trading_on_expiry = (
                is_expiry_day(expiry_date, timestamp) and 
                timestamp.time() < expiry_cutoff
            )
            
            if trading_on_expiry:
                # Handle expiry day trading with next expiry options
                next_expiry = find_next_expiry_options(timestamp, options_data)
                if next_expiry:
                    atm_strike = find_atm_strike(
                        futures_at_time, options_data, next_expiry, option_type
                    )
                    if atm_strike:
                        process_next_expiry_options(
                            timestamp, next_expiry, atm_strike, option_type,
                            options_data, futures_at_time, oi_velocity, pcr_data,
                            entry_signals, mandatory_conditions, min_conditions_required
                        )
            
            # Filter options by type efficiently using boolean indexing
            options_df = expiry_data_at_time[expiry_data_at_time['otype'] == option_type]
            
            if options_df.empty:
                continue
                
            # Process options in vectorized manner where possible
            process_options_batch(
                options_df, timestamp, option_type, futures_at_time,
                oi_velocity, pcr_data, entry_signals, 
                mandatory_conditions, min_conditions_required
            )
            
        except Exception as e:
            logger.error(f"Error processing expiry {expiry}: {e}")
            continue
    
    # Sort signals by conditions met
    entry_signals.sort(key=lambda x: x['conditions_met'], reverse=True)
    
    return {
        'direction': direction,
        'signals': entry_signals
    }

def process_options_batch(options_df: pd.DataFrame,
                         timestamp: pd.Timestamp,
                         option_type: str,
                         futures_data: pd.DataFrame,
                         oi_velocity: Dict,
                         pcr_data: pd.DataFrame,
                         entry_signals: List,
                         mandatory_conditions: List[str] = None,
                         min_conditions_required: int = 4) -> None:
    """
    Process a batch of options efficiently with mandatory conditions.
    
    Args:
        options_df: DataFrame with options data
        timestamp: Current timestamp
        option_type: Option type ('CE' or 'PE')
        futures_data: DataFrame with futures market data
        oi_velocity: Dictionary with OI velocity data
        pcr_data: DataFrame with PCR data
        entry_signals: List to append signals to
        mandatory_conditions: List of condition names that must be satisfied
        min_conditions_required: Minimum number of conditions required
    """
    try:
        if options_df.empty:
            logger.debug("Empty options DataFrame received")
            return
            
        # Pre-calculate common values with better error handling
        if futures_data.empty:
            logger.warning("Empty futures data in process_options_batch")
            return
            
        try:
            futures_latest = futures_data.iloc[-1]
            current_price = futures_latest['tr_close']
            cvd_signal = futures_latest.get('cvd_signal')
            vwap = futures_latest.get('vwap')
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting futures data: {e}")
            return
        
        # Normalize column names if needed
        column_mappings = {
            'option_type': 'otype',
            'week_expiry_date': 'expiry_date',
            'implied_volatility': 'iv'
        }
        
        for old_col, new_col in column_mappings.items():
            if old_col in options_df.columns and new_col not in options_df.columns:
                options_df[new_col] = options_df[old_col]
        
        # Validate required columns
        required_columns = ['strike_price', 'otype', 'delta', 'gamma']
        missing_columns = [col for col in required_columns if col not in options_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in options data: {missing_columns}")
            return
            
        # MODIFIED: First optimize strike selection
        options_df = optimize_strike_selection(options_df, current_price, option_type)
        
        # Filter strikes near ATM with logging
        logger.debug(f"Processing {len(options_df)} optimized options")
        
        if options_df.empty:
            logger.debug("No options found after optimization")
            return
        
        # Pre-calculate common conditions once
        cvd_condition = check_cvd_breakout_condition(futures_data, option_type)
        vwap_condition = check_vwap_condition(futures_data, option_type)
        pcr_condition = check_pcr_condition(pcr_data, option_type, timestamp)
        
        # Process in batches for better memory efficiency
        for i in range(0, len(options_df), 10):
            options_batch = options_df.iloc[i:i+10]
            
            for _, option in options_batch.iterrows():
                try:
                    # Convert option to dict with normalized field names
                    option_dict = option.to_dict()
                    
                    # Normalize field names in the dictionary
                    for old_field, new_field in column_mappings.items():
                        if old_field in option_dict and new_field not in option_dict:
                            option_dict[new_field] = option_dict[old_field]
                    
                    # Extract and validate expiry date
                    try:
                        expiry_date = None
                        expiry_sources = [
                            ('expiry_date', option_dict.get('expiry_date')),
                            ('week_expiry_date', option_dict.get('week_expiry_date')),
                            ('index', option.name if isinstance(option.name, (datetime, pd.Timestamp)) else None)
                        ]
                        
                        for source_name, source_value in expiry_sources:
                            if source_value is not None:
                                try:
                                    expiry_date = pd.to_datetime(source_value)
                                    logger.debug(f"Using expiry date from {source_name}: {expiry_date}")
                                    break
                                except:
                                    continue
                        
                        if expiry_date is None:
                            logger.error("Could not determine expiry date")
                            continue
                            
                        # Validate expiry is in the future
                        if expiry_date.date() <= timestamp.date():
                            logger.debug(f"Skipping expired option: {expiry_date}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error processing expiry date: {e}")
                        continue
                    
                    # Check conditions with better error handling
                    conditions = {}
                    
                    # Delta condition
                    try:
                        conditions['delta'] = check_delta_condition_cached(
                            option_dict.get('delta'), option_type
                        )
                    except Exception as e:
                        logger.error(f"Error in delta check: {e}")
                        conditions['delta'] = False
                    
                    # Gamma condition
                    try:
                        conditions['gamma'] = check_gamma_condition_cached(
                            option_dict.get('gamma')
                        )
                    except Exception as e:
                        logger.error(f"Error in gamma check: {e}")
                        conditions['gamma'] = False
                    
                    # Add other pre-calculated conditions
                    conditions.update({
                        'cvd': cvd_condition,
                        'vwap': vwap_condition,
                        'pcr': pcr_condition,
                        'oi': check_oi_change_condition(option_dict, oi_velocity, timestamp),
                        'iv': check_iv_condition(option_dict)
                    })
                    
                    # Check mandatory conditions
                    if mandatory_conditions:
                        all_mandatory_met = all(
                            conditions.get(cond, False) for cond in mandatory_conditions
                        )
                        if not all_mandatory_met:
                            logger.debug(f"Skipping option - mandatory conditions not met")
                            continue
                    
                    # Count conditions met
                    conditions_met, conditions = check_conditions(conditions)
                    
                    # Generate signal if enough conditions are met
                    if conditions_met >= min_conditions_required:
                        signal = {
                            'timestamp': timestamp,
                            'option_type': option_type,
                            'strike_price': option_dict['strike_price'],
                            'expiry_date': expiry_date,
                            'option': option_dict,
                            'conditions': conditions,
                            'conditions_met': conditions_met,
                            'conditions_details': {
                                k: {
                                    'result': v,
                                    'value': option_dict.get(k),
                                    'threshold': get_condition_threshold(k, option_type)
                                } 
                                for k, v in conditions.items()
                            }
                        }
                        
                        # Calculate and add condition scores
                        signal['condition_scores'] = calculate_condition_scores(
                            option_dict, futures_data, option_type, 
                            pcr_data, timestamp, oi_velocity
                        )
                        
                        entry_signals.append(signal)
                        
                        logger.debug(
                            f"Signal generated for {option_type} {option_dict['strike_price']}, "
                            f"expiry: {expiry_date}, conditions met: {conditions_met}, "
                            f"scores: {signal['condition_scores']}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing option {option.name}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error in process_options_batch: {e}")

def get_condition_threshold(condition: str, option_type: str) -> Any:
    """Helper function to get threshold values for conditions."""
    thresholds = {
        'delta': CALL_DELTA_THRESHOLD if option_type == 'CE' else PUT_DELTA_THRESHOLD,
        'gamma': GAMMA_THRESHOLD,
        'iv': IV_PERCENTILE_THRESHOLD,
        'oi': OI_CHANGE_THRESHOLD
    }
    return thresholds.get(condition, None)

def process_next_expiry_options(
    timestamp: pd.Timestamp,
    next_expiry: str,
    atm_strike: float,
    option_type: str,
    options_data: Dict[str, pd.DataFrame],
    futures_data: pd.DataFrame,
    oi_velocity: Dict,
    pcr_data: pd.DataFrame,
    entry_signals: List,
    mandatory_conditions: List[str] = None,
    min_conditions_required: int = 4
) -> None:
    """
    Process next expiry ATM options for expiry day trading.
    
    Args:
        timestamp: Current timestamp
        next_expiry: Next expiry date string
        atm_strike: ATM strike price
        option_type: Option type ('CE' or 'PE')
        options_data: Dictionary of options data by expiry
        futures_data: Futures market data
        oi_velocity: OI velocity data
        pcr_data: PCR data
        entry_signals: List to append signals to
        mandatory_conditions: List of condition names that must be satisfied (default: None)
        min_conditions_required: Minimum number of conditions required for a valid signal (default: 4)
    """
    try:
        # Get next expiry options data
        if next_expiry not in options_data:
            return
            
        next_expiry_data = options_data[next_expiry]
        
        # Filter for ATM options of the specified type
        atm_options = next_expiry_data[
            (next_expiry_data['otype'] == option_type) &
            (next_expiry_data['strike_price'] == atm_strike) &
            (next_expiry_data.index <= timestamp)
        ]
        
        if atm_options.empty:
            return
        
        # Default mandatory conditions if none provided
        if mandatory_conditions is None:
            mandatory_conditions = ["cvd", "vwap"]
        
        # Pre-calculate conditions once to avoid Series in boolean context
        cvd_condition = check_cvd_breakout_condition(futures_data, option_type)
        vwap_condition = check_vwap_condition(futures_data, option_type)
        pcr_condition = check_pcr_condition(pcr_data, option_type, timestamp)
            
        # Process ATM options
        for _, option in atm_options.iterrows():
            try:
                option_dict = option.to_dict()
                
                # Skip invalid options
                if not all(k in option_dict for k in ['delta', 'gamma', 'strike_price']):
                    continue
                    
                # Check individual conditions with scalar results
                delta_condition = check_delta_condition_cached(option_dict.get('delta'), option_type)
                gamma_condition = check_gamma_condition_cached(option_dict.get('gamma'))
                oi_condition = check_oi_change_condition(option_dict, oi_velocity, timestamp)
                iv_condition = check_iv_condition(option_dict)
                
                # Create conditions dictionary
                conditions = {
                    'delta': delta_condition,
                    'gamma': gamma_condition,
                    'oi': oi_condition,
                    'iv': iv_condition,
                    'cvd': cvd_condition,
                    'vwap': vwap_condition,
                    'pcr': pcr_condition
                }
                
                # MODIFIED: Check if ALL mandatory conditions are met
                all_mandatory_met = True
                for condition_name in mandatory_conditions:
                    if condition_name not in conditions or not conditions[condition_name]:
                        all_mandatory_met = False
                        break
                
                # Skip if not all mandatory conditions are met
                if not all_mandatory_met:
                    continue
                
                # Count conditions met
                conditions_met, conditions = check_conditions(conditions)
                
                # Generate signal if enough conditions are met
                if conditions_met >= min_conditions_required:
                    signal = {
                        'timestamp': timestamp,
                        'option_type': option_type,
                        'strike_price': option_dict['strike_price'],
                        'expiry_date': pd.to_datetime(option.name).to_pydatetime(),
                        'option': option_dict,
                        'conditions': conditions,
                        'conditions_met': conditions_met
                    }
                    entry_signals.append(signal)
                    logger.debug(
                        f"Next expiry signal generated for {option_type} {option_dict['strike_price']}, "
                        f"conditions met: {conditions_met}"
                    )
                
            except Exception as e:
                logger.error(f"Error processing next expiry option: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error in process_next_expiry_options: {e}")

def generate_exit_rules(timestamp: pd.Timestamp,
                       open_positions: Dict[str, Dict[str, Any]],
                       futures_data: pd.DataFrame,
                       options_data: Dict[str, pd.DataFrame],
                       options_sentiment: Dict[str, Any],
                       market_regime: str = None) -> List[Dict[str, Any]]:
    """
    Apply exit rules to generate exit signals.
    
    Args:
        timestamp: Current timestamp
        open_positions: Dictionary of open positions by ID
        futures_data: DataFrame with futures market data
        options_data: Dictionary of DataFrames with options data by expiry
        options_sentiment: Dictionary with options sentiment indicators
        market_regime: Current market regime (default: None)
        
    Returns:
        List of exit signals
    """
    logger.debug(f"Generating exit rules at {timestamp} with market regime: {market_regime}")
    
    exit_signals = []
    
    # Extract sentiment data
    pcr_data = options_sentiment.get('pcr', pd.DataFrame())
    
    for position_id, position in open_positions.items():
        try:
            # Extract position details
            option_type = position['option_type']
            expiry_date = position['expiry_date']
            strike_price = position['strike_price']
            entry_price = position['entry_price']
            current_price = position['current_price']
            
            # Calculate profit/loss
            pnl_pct = position['pnl_percent']
            
            # Adjust profit targets based on market regime
            profit_target_1 = PROFIT_TARGET_1
            profit_target_2 = PROFIT_TARGET_2
            
            if market_regime == "volatile":
                # More aggressive profit taking in volatile markets
                profit_target_1 *= 0.8
                profit_target_2 *= 0.8
            elif market_regime == "trending":
                # Let profits run in trending markets
                profit_target_1 *= 1.2
                profit_target_2 *= 1.2
            
            # Check stop loss
            stop_loss_hit = position['current_price'] <= position['stop_loss'] if option_type == 'CE' else \
                            position['current_price'] >= position['stop_loss']
            
            # Check profit targets with regime-adjusted values
            first_target_hit = pnl_pct >= profit_target_1
            second_target_hit = pnl_pct >= profit_target_2
            
            # Check signal reversal
            direction_reversed = False
            
            if not futures_data.empty and 'cvd_signal' in futures_data.columns:
                current_signal = futures_data['cvd_signal'].iloc[-1]
                direction_reversed = (option_type == 'CE' and current_signal < 0) or \
                                    (option_type == 'PE' and current_signal > 0)
                
                # More sensitive to reversals in volatile markets
                if market_regime == "volatile" and abs(current_signal) > 1.5:
                    direction_reversed = True
            
            # Check delta reversal
            delta_reversed = False
            if 'current_delta' in position and 'delta' in position:
                current_delta = position['current_delta']
                original_delta = position['delta']
                
                delta_reversed = (option_type == 'CE' and current_delta < 0.4 and original_delta >= 0.4) or \
                               (option_type == 'PE' and current_delta > -0.4 and original_delta <= -0.4)
            
            # Check PCR reversal (using scalar values)
            pcr_reversed = False
            if not pcr_data.empty and 'pcr' in pcr_data.columns:
                pcr_at_time = pcr_data.loc[pcr_data.index <= timestamp]
                
                if not pcr_at_time.empty and 'pcr_ma5' in pcr_at_time.columns:
                    # Get scalar values with iloc
                    pcr = pcr_at_time['pcr'].iloc[-1]
                    pcr_ma5 = pcr_at_time['pcr_ma5'].iloc[-1]
                    
                    pcr_reversed = (option_type == 'CE' and pcr > pcr_ma5) or \
                                 (option_type == 'PE' and pcr < pcr_ma5)
            
            # Generate exit signal if any exit condition is met
            if stop_loss_hit or first_target_hit or second_target_hit or direction_reversed or delta_reversed or pcr_reversed:
                exit_reason = "stop_loss" if stop_loss_hit else \
                              "first_target" if first_target_hit else \
                              "second_target" if second_target_hit else \
                              "direction_reversal" if direction_reversed else \
                              "delta_reversal" if delta_reversed else \
                              "pcr_reversal"
                
                exit_signal = {
                    'position_id': position_id,
                    'timestamp': timestamp,
                    'exit_reason': exit_reason,
                    'pnl_percent': pnl_pct
                }
                
                exit_signals.append(exit_signal)
        
        except Exception as e:
            logger.error(f"Error generating exit rules for position {position_id}: {e}")
    
    return exit_signals

@lru_cache(maxsize=1024)
def check_pcr_condition_cached(pcr: float, pcr_ma5: float, option_type: str) -> bool:
    """
    Cached version of PCR condition check.
    
    Args:
        pcr: Current PCR value
        pcr_ma5: PCR 5-period moving average
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        True if condition met, False otherwise
    """
    if option_type == 'CE':
        # For calls, we want decreasing PCR (bullish)
        return pcr < pcr_ma5
    else:  # PE
        # For puts, we want increasing PCR (bearish)
        return pcr > pcr_ma5

@lru_cache(maxsize=1024)
def check_cvd_breakout_condition_cached(cvd_signal: float, option_type: str) -> bool:
    """
    Cached version of CVD breakout condition check.
    
    Args:
        cvd_signal: CVD signal value
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        True if condition met, False otherwise
    """
    if option_type == 'CE':
        return cvd_signal > 0
    else:  # PE
        return cvd_signal < 0

@lru_cache(maxsize=1024)
def check_vwap_condition_cached(current_price: float, vwap: float, option_type: str) -> bool:
    """
    Cached version of VWAP condition check.
    
    Args:
        current_price: Current price
        vwap: VWAP value
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        True if condition met, False otherwise
    """
    if option_type == 'CE':
        return current_price > vwap
    else:  # PE
        return current_price < vwap

def calculate_condition_scores(option_dict: Dict[str, Any], 
                             futures_data: pd.DataFrame,
                             option_type: str,
                             pcr_data: pd.DataFrame,
                             timestamp: pd.Timestamp,
                             oi_velocity: Dict) -> Dict[str, float]:
    """
    Calculate weighted scores for each condition (0-10 scale).
    
    Args:
        option_dict: Dictionary with option data
        futures_data: DataFrame with futures market data
        option_type: Option type ('CE' or 'PE')
        pcr_data: DataFrame with PCR data
        timestamp: Current timestamp
        oi_velocity: Dictionary with OI velocity data
        
    Returns:
        Dictionary of condition scores (0-10 scale)
    """
    scores = {}
    
    try:
        # Delta score (strength of directional bias)
        delta = option_dict.get('delta', 0)
        if option_type == 'CE':
            delta_score = min(10, max(0, (delta - 0.3) * 20))  # Scale 0.3-0.8 to 0-10
        else:
            delta_score = min(10, max(0, (abs(delta) - 0.3) * 20))
        scores['delta'] = delta_score
        
        # Gamma score (sensitivity to price movements)
        gamma = option_dict.get('gamma', 0)
        gamma_score = min(10, gamma * 200)  # Scale 0-0.05 to 0-10
        scores['gamma'] = gamma_score
        
        # OI change score
        oi_change_score = 0
        if timestamp in oi_velocity:
            oi_info = oi_velocity[timestamp]
            significant_changes = oi_info.get('significant_oi_changes', [])
            
            for change in significant_changes:
                if (change['otype'] == option_type and 
                    change['strike_price'] == option_dict['strike_price']):
                    # Scale change percentage to score
                    change_pct = abs(change.get('oi_change_pct', 0))
                    oi_change_score = min(10, change_pct / 5)  # 5% change = score of 10
                    break
        
        scores['oi'] = oi_change_score
        
        # CVD score based on signal strength
        cvd_score = 0
        if not futures_data.empty and 'cvd_signal' in futures_data.columns:
            cvd_signal = futures_data['cvd_signal'].iloc[-1]
            if (option_type == 'CE' and cvd_signal > 0) or (option_type == 'PE' and cvd_signal < 0):
                cvd_score = min(10, abs(cvd_signal) * 5)  # Scale signal strength to score
        
        scores['cvd'] = cvd_score
        
        # VWAP score based on price vs VWAP distance
        vwap_score = 0
        if not futures_data.empty and 'vwap' in futures_data.columns:
            current_price = futures_data['tr_close'].iloc[-1]
            vwap = futures_data['vwap'].iloc[-1]
            
            # Calculate percentage deviation from VWAP
            pct_from_vwap = (current_price - vwap) / vwap
            
            if (option_type == 'CE' and pct_from_vwap > 0) or (option_type == 'PE' and pct_from_vwap < 0):
                vwap_score = min(10, abs(pct_from_vwap) * 200)  # Scale 0-5% to 0-10
        
        scores['vwap'] = vwap_score
        
        # PCR score based on trend alignment
        pcr_score = 0
        if not pcr_data.empty and 'pcr' in pcr_data.columns:
            pcr_at_time = pcr_data.loc[pcr_data.index <= timestamp]
            
            if not pcr_at_time.empty:
                pcr = pcr_at_time['pcr'].iloc[-1]
                
                if 'pcr_ma_short' in pcr_at_time.columns and 'pcr_ma_long' in pcr_at_time.columns:
                    pcr_ma_short = pcr_at_time['pcr_ma_short'].iloc[-1]
                    pcr_ma_long = pcr_at_time['pcr_ma_long'].iloc[-1]
                    
                    if option_type == 'CE':
                        # For calls, we want PCR < short MA < long MA (bearishness decreasing)
                        if pcr < pcr_ma_short < pcr_ma_long:
                            # Calculate score based on alignment strength
                            alignment = (pcr_ma_long - pcr) / pcr_ma_long
                            pcr_score = min(10, alignment * 20)
                    else:  # PE
                        # For puts, we want PCR > short MA > long MA (bearishness increasing)
                        if pcr > pcr_ma_short > pcr_ma_long:
                            # Calculate score based on alignment strength
                            alignment = (pcr - pcr_ma_long) / pcr_ma_long
                            pcr_score = min(10, alignment * 20)
        
        scores['pcr'] = pcr_score
        
        # IV score based on percentile
        iv_score = 0
        if 'iv_percentile' in option_dict:
            iv_percentile = option_dict['iv_percentile']
            
            if option_type == 'CE':
                # For calls, lower IV percentile is better (room to expand)
                iv_score = min(10, (100 - iv_percentile) / 10)
            else:  # PE
                # For puts, moderate IV is better (not too high, not too low)
                iv_score = 10 - min(10, abs(50 - iv_percentile) / 5)
        
        scores['iv'] = iv_score
        
    except Exception as e:
        logger.error(f"Error calculating condition scores: {e}")
        # Return zero scores on error
        scores = {
            'delta': 0, 'gamma': 0, 'oi': 0, 'cvd': 0,
            'vwap': 0, 'pcr': 0, 'iv': 0
        }
    
    return scores

def handle_missing_oi_velocity(option_data, timestamp):
    """
    Handle missing OI velocity data by estimating it from available data.
    
    Args:
        option_data: Options DataFrame
        timestamp: Current timestamp
        
    Returns:
        Options DataFrame with estimated OI velocity where missing
    """
    if 'oi_velocity' not in option_data.columns:
        # Add the column if it doesn't exist
        option_data['oi_velocity'] = 0.0
        return option_data
    
    # Check if we have any OI velocity data
    if option_data['oi_velocity'].isna().all() or (option_data['oi_velocity'] == 0).all():
        # If we have open_interest data, try to calculate velocity on the fly
        if 'open_interest' in option_data.columns:
            try:
                # Group by strike and option type
                grouped = option_data.groupby(['strike_price', 'otype'])
                
                # Calculate for each group
                for name, group in grouped:
                    if len(group) > 1:
                        # Calculate OI changes
                        oi_change = group['open_interest'].diff()
                        
                        # Use a simple time-based approach for velocity
                        oi_velocity = oi_change / 5  # Assume 5-minute intervals if exact time not available
                        
                        # Fill the values back
                        idx = group.index
                        option_data.loc[idx, 'oi_velocity'] = oi_velocity
                
                # Fill NaN values with 0
                option_data['oi_velocity'] = option_data['oi_velocity'].fillna(0)
            except Exception as e:
                logger.debug(f"Error calculating on-the-fly OI velocity: {e}")
    
    return option_data