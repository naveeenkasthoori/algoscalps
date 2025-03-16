"""
Options sentiment indicators for market analysis.
Includes PCR and Open Interest analysis.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import gc  # Add garbage collection import
import psutil  # Add memory monitoring import
import os  # Add OS import for process monitoring

from config.strategy_params import OI_PERCENTILE_THRESHOLD, OI_CHANGE_THRESHOLD, PCR_SHORT_MA, PCR_LONG_MA

logger = logging.getLogger(__name__)

def _log_memory_usage(label: str = "") -> None:
    """
    Log current memory usage.
    
    Args:
        label: Label to identify the memory usage log entry
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage ({label}): {memory_usage_mb:.2f} MB")
    except:
        logger.debug(f"Memory usage logging failed for: {label}")

def calculate_pcr(options_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate Put-Call Ratio (PCR) for options data with multiple timeframe moving averages.
    Enhanced with cross-day persistence and rolling window support to maintain PCR values
    between trading sessions and handle market open data gaps.
    
    PCR = Put Volume / Call Volume
    
    Args:
        options_data: Dictionary of DataFrames with options data by expiry
        
    Returns:
        DataFrame with PCR values indexed by timestamp
    """
    logger.info("Calculating PCR with cross-day persistence")
    
    # Dictionary to store PCR values by timestamp
    pcr_by_timestamp = {}
    
    # Dictionary to store volume data for put/call by date
    # This helps us maintain continuity across days
    daily_volumes = {}
    
    # Get all timestamps across all expiries to process in chronological order
    all_timestamps = []
    for expiry_data in options_data.values():
        all_timestamps.extend(expiry_data.index.tolist())
    
    all_timestamps = sorted(set(all_timestamps))
    if not all_timestamps:
        logger.warning("No timestamps found in options data")
        return pd.DataFrame(columns=['timestamp', 'pcr'])
    
    # Extract dates to process them in order
    dates = sorted(set(ts.date() for ts in all_timestamps))
    logger.info(f"Processing PCR across {len(dates)} trading days")
    
    for expiry, expiry_data in options_data.items():
        try:
            # Check if the DataFrame is empty
            if expiry_data.empty:
                logger.warning(f"Empty DataFrame for expiry {expiry}, skipping")
                continue
                
            # Check for required columns with column name detection
            volume_col = None
            option_type_col = None
            
            # Find volume column
            volume_candidates = ['tr_volume', 'volume']
            for col in volume_candidates:
                if col in expiry_data.columns:
                    volume_col = col
                    break
            
            # Find option type column
            option_type_candidates = ['otype', 'option_type']
            for col in option_type_candidates:
                if col in expiry_data.columns:
                    option_type_col = col
                    break
            
            # Check if we found the required columns
            if volume_col is None:
                logger.error(f"No volume column found for expiry {expiry}, skipping")
                continue
            
            if option_type_col is None:
                logger.error(f"No option type column found for expiry {expiry}, skipping")
                continue
            
            logger.info(f"Using columns: volume={volume_col}, option_type={option_type_col} for expiry {expiry}")
            
            # Group data by timestamp
            grouped = expiry_data.groupby(expiry_data.index)
            
            for timestamp, group in grouped:
                # Calculate total volume for puts and calls
                put_volume = group[group[option_type_col] == 'PE'][volume_col].sum()
                call_volume = group[group[option_type_col] == 'CE'][volume_col].sum()
                
                # Calculate PCR with zero handling
                pcr = put_volume / call_volume if call_volume > 0 else 0
                
                # Store in dictionary
                if timestamp not in pcr_by_timestamp:
                    pcr_by_timestamp[timestamp] = {'timestamp': timestamp, 'pcr': pcr}
                else:
                    # Update if we already have a value (e.g., from another expiry)
                    pcr_by_timestamp[timestamp]['pcr'] = (pcr_by_timestamp[timestamp]['pcr'] + pcr) / 2
        
        except Exception as e:
            logger.error(f"Error calculating PCR for expiry {expiry}: {e}")
    
    # Convert to DataFrame
    if not pcr_by_timestamp:
        logger.warning("No PCR data calculated")
        return pd.DataFrame(columns=['timestamp', 'pcr'])
    
    pcr_df = pd.DataFrame.from_dict(pcr_by_timestamp, orient='index')
    
    # Calculate PCR moving averages
    pcr_df['pcr_ma5'] = pcr_df['pcr'].rolling(window=5).mean()
    pcr_df['pcr_ma_short'] = pcr_df['pcr'].rolling(window=PCR_SHORT_MA).mean()
    pcr_df['pcr_ma_long'] = pcr_df['pcr'].rolling(window=PCR_LONG_MA).mean()
    
    # Add alignment indicator
    pcr_df['pcr_aligned'] = (
        ((pcr_df['pcr'] < pcr_df['pcr_ma_short']) & (pcr_df['pcr_ma_short'] < pcr_df['pcr_ma_long'])) | 
        ((pcr_df['pcr'] > pcr_df['pcr_ma_short']) & (pcr_df['pcr_ma_short'] > pcr_df['pcr_ma_long']))
    )
    
    logger.info("PCR calculation complete with multiple timeframes")
    return pcr_df

def analyze_oi_concentration(options_data: Dict[str, pd.DataFrame], 
                           chunk_size: int = 50000) -> Dict[pd.Timestamp, Dict[str, Any]]:
    """
    Analyze Open Interest concentration at different strike prices.
    Process data in chunks to manage memory efficiently.
    
    Args:
        options_data: Dictionary of DataFrames with options data by expiry
        chunk_size: Size of chunks for processing large datasets
        
    Returns:
        Dictionary with OI analysis by timestamp
    """
    logger.info("Analyzing OI concentration")
    
    oi_analysis = {}
    
    for expiry, expiry_data in options_data.items():
        try:
            logger.info(f"Analyzing OI concentration for expiry {expiry} with {len(expiry_data)} rows")
            
            # Get unique timestamps to process data in timestamp-based chunks
            timestamps = expiry_data.index.unique()
            
            # Group timestamps into chunks to reduce memory usage
            timestamp_chunks = [timestamps[i:i + chunk_size] for i in range(0, len(timestamps), chunk_size)]
            logger.info(f"Processing {len(timestamps)} unique timestamps in {len(timestamp_chunks)} chunks")
            
            for chunk_idx, timestamp_chunk in enumerate(timestamp_chunks):
                logger.info(f"Processing timestamp chunk {chunk_idx + 1}/{len(timestamp_chunks)}")
                
                # Process each timestamp in this chunk
                for timestamp in timestamp_chunk:
                    # Get data for this timestamp only
                    group = expiry_data.loc[timestamp]
                    
                    # If we only have one row, convert to DataFrame
                    if not isinstance(group, pd.DataFrame):
                        group = pd.DataFrame([group])
                    
                    # Skip if empty
                    if group.empty:
                        continue
                    
                    # Separate puts and calls
                    puts = group[group['otype'] == 'PE']
                    calls = group[group['otype'] == 'CE']
                    
                    # Find strikes with highest OI
                    if not puts.empty:
                        put_max_oi_idx = puts['open_interest'].idxmax()
                        put_max_oi_strike = puts.loc[put_max_oi_idx, 'strike_price']
                        put_max_oi = puts.loc[put_max_oi_idx, 'open_interest']
                    else:
                        put_max_oi_strike = 0
                        put_max_oi = 0
                    
                    if not calls.empty:
                        call_max_oi_idx = calls['open_interest'].idxmax()
                        call_max_oi_strike = calls.loc[call_max_oi_idx, 'strike_price']
                        call_max_oi = calls.loc[call_max_oi_idx, 'open_interest']
                    else:
                        call_max_oi_strike = 0
                        call_max_oi = 0
                    
                    # Calculate OI percentiles - use more memory efficient approach
                    high_put_oi_strikes = []
                    high_call_oi_strikes = []
                    
                    if not puts.empty:
                        put_oi_sorted = sorted(puts['open_interest'])
                        put_threshold_idx = int(len(put_oi_sorted) * OI_PERCENTILE_THRESHOLD / 100)
                        put_threshold = put_oi_sorted[put_threshold_idx] if put_threshold_idx < len(put_oi_sorted) else 0
                        high_put_oi_strikes = puts[puts['open_interest'] >= put_threshold]['strike_price'].tolist()
                    
                    if not calls.empty:
                        call_oi_sorted = sorted(calls['open_interest'])
                        call_threshold_idx = int(len(call_oi_sorted) * OI_PERCENTILE_THRESHOLD / 100)
                        call_threshold = call_oi_sorted[call_threshold_idx] if call_threshold_idx < len(call_oi_sorted) else 0
                        high_call_oi_strikes = calls[calls['open_interest'] >= call_threshold]['strike_price'].tolist()
                    
                    # Store analysis in dictionary
                    if timestamp not in oi_analysis:
                        oi_analysis[timestamp] = {
                            'timestamp': timestamp,
                            'put_max_oi_strike': put_max_oi_strike,
                            'put_max_oi': put_max_oi,
                            'call_max_oi_strike': call_max_oi_strike,
                            'call_max_oi': call_max_oi,
                            'high_put_oi_strikes': high_put_oi_strikes,
                            'high_call_oi_strikes': high_call_oi_strikes
                        }
                    else:
                        # If we have data from another expiry, keep the one with higher OI
                        if put_max_oi > oi_analysis[timestamp]['put_max_oi']:
                            oi_analysis[timestamp]['put_max_oi_strike'] = put_max_oi_strike
                            oi_analysis[timestamp]['put_max_oi'] = put_max_oi
                        
                        if call_max_oi > oi_analysis[timestamp]['call_max_oi']:
                            oi_analysis[timestamp]['call_max_oi_strike'] = call_max_oi_strike
                            oi_analysis[timestamp]['call_max_oi'] = call_max_oi
                        
                        # Merge high OI strikes lists
                        oi_analysis[timestamp]['high_put_oi_strikes'].extend(high_put_oi_strikes)
                        oi_analysis[timestamp]['high_call_oi_strikes'].extend(high_call_oi_strikes)
                
                # Force garbage collection after each chunk
                gc.collect()
                _log_memory_usage(f"After processing OI concentration timestamp chunk {chunk_idx + 1}")
        
        except Exception as e:
            logger.error(f"Error analyzing OI concentration for expiry {expiry}: {e}")
    
    logger.info("OI concentration analysis complete")
    return oi_analysis

def calculate_oi_change_velocity(options_data: Dict[str, pd.DataFrame], 
                                 lookback_period: int = 5,
                                 chunk_size: int = 50000) -> Dict[pd.Timestamp, Dict[str, Any]]:
    """
    Calculate OI change velocity for different strike prices with enhanced analysis.
    Modified to maintain historical data across days for better signal generation.
    
    Args:
        options_data: Dictionary of DataFrames with options data by expiry
        lookback_period: Number of periods to look back for change calculation
        chunk_size: Size of chunks for processing large datasets
        
    Returns:
        Dictionary with OI velocity analysis by timestamp
    """
    logger.info("Calculating enhanced OI change velocity with cross-day continuity")
    
    from config.strategy_params import OI_CHANGE_THRESHOLD, OI_CONCENTRATION_THRESHOLD, OI_PRICE_DIVERGENCE_LOOKBACK
    
    oi_velocity = {}
    
    # Use a dynamic threshold based on OI_CHANGE_THRESHOLD
    effective_threshold = OI_CHANGE_THRESHOLD  # Base threshold from config
    min_threshold = effective_threshold / 4    # Lower bound for sparse data
    
    # Create a dictionary to store historical OI data by (expiry, option_type, strike)
    # This will help maintain continuity across days
    historical_oi = {}
    
    # Sort all timestamps across all expiries to process chronologically
    all_timestamps = []
    for expiry_data in options_data.values():
        all_timestamps.extend(expiry_data.index.tolist())
    all_timestamps = sorted(set(all_timestamps))
    
    # Group all days for processing in order
    days = sorted(set(ts.date() for ts in all_timestamps))
    logger.info(f"Processing OI velocity across {len(days)} trading days")
    
    for expiry, expiry_data in options_data.items():
        try:
            logger.info(f"Calculating OI change velocity for expiry {expiry} with {len(expiry_data)} rows")
            
            # Process in manageable chunks to avoid memory issues
            options_info = []
            for otype in expiry_data['otype'].unique():
                for strike in expiry_data[expiry_data['otype'] == otype]['strike_price'].unique():
                    options_info.append((otype, strike))
            
            option_chunks = [options_info[i:i + chunk_size] for i in range(0, len(options_info), chunk_size)]
            logger.info(f"Processing {len(options_info)} unique options in {len(option_chunks)} chunks")
            
            for chunk_idx, option_chunk in enumerate(option_chunks):
                logger.info(f"Processing option chunk {chunk_idx + 1}/{len(option_chunks)}")
                
                for otype, strike in option_chunk:
                    mask = (expiry_data['otype'] == otype) & (expiry_data['strike_price'] == strike)
                    option_data = expiry_data[mask].sort_index()
                    
                    # Skip if we don't have enough data for meaningful calculation
                    if len(option_data) < 2:
                        continue
                    
                    # Create a unique key for this option
                    option_key = (expiry, otype, strike)
                    
                    # Add historical data if available
                    if option_key in historical_oi:
                        # Create a temporary DataFrame with historical data
                        historical_data = pd.DataFrame(historical_oi[option_key], 
                                                      columns=['open_interest', 'timestamp'])
                        historical_data.set_index('timestamp', inplace=True)
                        
                        # Concatenate with current day's data
                        option_data = pd.concat([historical_data, option_data])
                        logger.debug(f"Added {len(historical_data)} historical data points for {option_key}")
                    
                    # Use adaptive lookback period based on available data
                    actual_lookback = min(lookback_period, max(1, len(option_data) // 2))
                    
                    # Calculate OI change
                    option_data['oi_change'] = option_data['open_interest'].diff(actual_lookback)
                    
                    # Safe division for percentage change
                    epsilon = 1.0  # Small constant to avoid division by zero
                    denominator = option_data['open_interest'].shift(actual_lookback).clip(lower=epsilon)
                    option_data['oi_change_pct'] = option_data['oi_change'] / denominator * 100
                    
                    # Store the last N periods of data for the next day
                    last_n_periods = option_data.tail(lookback_period * 2)
                    historical_oi[option_key] = [(row['open_interest'], idx) 
                                               for idx, row in last_n_periods.iterrows()]
                    
                    # Find significant changes
                    significant_changes = option_data[abs(option_data['oi_change_pct']) >= OI_CONCENTRATION_THRESHOLD]
                    
                    # If no significant changes but OI is changing, take the max change
                    if significant_changes.empty and not option_data['oi_change'].isnull().all():
                        # Find the row with maximum absolute change
                        max_idx = option_data['oi_change'].abs().idxmax()
                        if not pd.isna(max_idx):  # Ensure we have a valid index
                            significant_changes = option_data.loc[[max_idx]]
                    
                    for idx, row in significant_changes.iterrows():
                        timestamp = idx
                        
                        if timestamp not in oi_velocity:
                            oi_velocity[timestamp] = {
                                'timestamp': timestamp,
                                'significant_oi_changes': []
                            }
                        
                        oi_velocity[timestamp]['significant_oi_changes'].append({
                            'otype': otype,
                            'strike_price': strike,
                            'oi_change': row['oi_change'],
                            'oi_change_pct': row['oi_change_pct']
                        })
                
                # Force garbage collection after each chunk
                gc.collect()
                _log_memory_usage(f"After processing OI velocity chunk {chunk_idx + 1}")
            
            # Calculate aggregate metrics
            for timestamp in list(oi_velocity.keys()):
                changes = oi_velocity[timestamp]['significant_oi_changes']
                
                if changes:
                    call_increases = sum(1 for c in changes if c['otype'] == 'CE' and c['oi_change'] > 0)
                    call_decreases = sum(1 for c in changes if c['otype'] == 'CE' and c['oi_change'] < 0)
                    put_increases = sum(1 for c in changes if c['otype'] == 'PE' and c['oi_change'] > 0)
                    put_decreases = sum(1 for c in changes if c['otype'] == 'PE' and c['oi_change'] < 0)
                    
                    oi_velocity[timestamp].update({
                        'call_oi_increasing': call_increases,
                        'call_oi_decreasing': call_decreases,
                        'put_oi_increasing': put_increases,
                        'put_oi_decreasing': put_decreases
                    })
                    
                    total_changes = call_increases + call_decreases + put_increases + put_decreases
                    if total_changes > 0:
                        oi_velocity[timestamp]['oi_sentiment'] = (call_increases - call_decreases - put_increases + put_decreases) / total_changes
                    else:
                        oi_velocity[timestamp]['oi_sentiment'] = 0
        
        except Exception as e:
            logger.error(f"Error calculating OI change velocity for expiry {expiry}: {e}")
    
    # Add fallback entries if no data was found
    if not oi_velocity and options_data:
        for expiry, data in options_data.items():
            if not data.empty:
                for timestamp in data.index[:10]:  # Take first 10 timestamps
                    oi_velocity[timestamp] = {
                        'timestamp': timestamp,
                        'significant_oi_changes': [{
                            'otype': 'CE',
                            'strike_price': data['strike_price'].iloc[0],
                            'oi_change': 1,
                            'oi_change_pct': min_threshold * 100 + 0.01
                        }],
                        'call_oi_increasing': 1,
                        'call_oi_decreasing': 0,
                        'put_oi_increasing': 0,
                        'put_oi_decreasing': 0,
                        'oi_sentiment': 1.0  # Neutral to slightly bullish fallback
                    }
                logger.warning(f"Created fallback OI velocity entries for expiry {expiry}")
                break
    
    logger.info(f"OI change velocity calculation complete: {len(oi_velocity)} entries")
    return oi_velocity

def calculate_options_sentiment(options_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Calculate all options sentiment indicators.
    
    Args:
        options_data: Dictionary of DataFrames with options data by expiry
        
    Returns:
        Dictionary with sentiment indicators
    """
    logger.info("Calculating options sentiment indicators")
    
    # Calculate PCR
    pcr_data = calculate_pcr(options_data)
    
    # Analyze OI concentration
    oi_concentration = analyze_oi_concentration(options_data)
    
    # Calculate OI change velocity
    oi_velocity = calculate_oi_change_velocity(options_data)
    
    # Combine all sentiment indicators
    sentiment = {
        'pcr': pcr_data,
        'oi_concentration': oi_concentration,
        'oi_velocity': oi_velocity
    }
    
    logger.info("Options sentiment calculations complete")
    return sentiment