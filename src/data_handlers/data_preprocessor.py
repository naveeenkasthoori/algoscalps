"""
Data preprocessing module for cleaning and formatting market data.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from config.settings import RESAMPLE_INTERVAL

logger = logging.getLogger(__name__)

def clean_spot_data(spot_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and format spot market data.
    
    Args:
        spot_data: Raw spot market data
        
    Returns:
        Cleaned spot market data
    """
    logger.info("Cleaning spot data")
    
    # Make a copy to avoid modifying original data
    cleaned_data = spot_data.copy()
    
    # Handle missing values
    if cleaned_data.isnull().any().any():
        logger.warning(f"Found {cleaned_data.isnull().sum().sum()} missing values in spot data")
        
        # Forward fill for OHLC data
        for col in ['tr_open', 'tr_high', 'tr_low', 'tr_close']:
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
        
        # Drop any remaining rows with missing values in critical columns
        critical_cols = ['tr_open', 'tr_high', 'tr_low', 'tr_close']
        critical_cols = [col for col in critical_cols if col in cleaned_data.columns]
        
        if critical_cols:
            before_count = len(cleaned_data)
            cleaned_data = cleaned_data.dropna(subset=critical_cols)
            after_count = len(cleaned_data)
            
            if before_count > after_count:
                logger.warning(f"Dropped {before_count - after_count} rows with missing values in critical columns")
    
    # Ensure datetime index
    if not isinstance(cleaned_data.index, pd.DatetimeIndex):
        if 'datetime' in cleaned_data.columns:
            cleaned_data.set_index('datetime', inplace=True)
        elif 'tr_date' in cleaned_data.columns and 'tr_time' in cleaned_data.columns:
            try:
                cleaned_data['datetime'] = pd.to_datetime(
                    cleaned_data['tr_date'] + ' ' + cleaned_data['tr_time']
                )
                cleaned_data.set_index('datetime', inplace=True)
            except Exception as e:
                logger.error(f"Error creating datetime index: {e}")
    
    # Sort by datetime
    cleaned_data = cleaned_data.sort_index()
    
    # Remove duplicates
    if cleaned_data.index.duplicated().any():
        dup_count = cleaned_data.index.duplicated().sum()
        logger.warning(f"Found {dup_count} duplicate timestamps in spot data")
        cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]
    
    logger.info(f"Spot data cleaning complete: {len(cleaned_data)} rows")
    return cleaned_data

def clean_futures_data(futures_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and format futures market data.
    
    Args:
        futures_data: Raw futures market data
        
    Returns:
        Cleaned futures market data
    """
    logger.info("Cleaning futures data")
    
    # Make a copy to avoid modifying original data
    cleaned_data = futures_data.copy()
    
    # Handle missing values
    if cleaned_data.isnull().any().any():
        logger.warning(f"Found {cleaned_data.isnull().sum().sum()} missing values in futures data")
        
        # Forward fill for OHLC and OI data
        for col in ['tr_open', 'tr_high', 'tr_low', 'tr_close', 'open_interest']:
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
        
        # Fill missing volume with zeros
        if 'tr_volume' in cleaned_data.columns:
            cleaned_data['tr_volume'] = cleaned_data['tr_volume'].fillna(0)
        
        # Drop any remaining rows with missing values in critical columns
        critical_cols = ['tr_open', 'tr_high', 'tr_low', 'tr_close', 'open_interest']
        critical_cols = [col for col in critical_cols if col in cleaned_data.columns]
        
        if critical_cols:
            before_count = len(cleaned_data)
            cleaned_data = cleaned_data.dropna(subset=critical_cols)
            after_count = len(cleaned_data)
            
            if before_count > after_count:
                logger.warning(f"Dropped {before_count - after_count} rows with missing values in critical columns")
    
    # Ensure datetime index
    if not isinstance(cleaned_data.index, pd.DatetimeIndex):
        if 'datetime' in cleaned_data.columns:
            cleaned_data.set_index('datetime', inplace=True)
        elif 'tr_datetime' in cleaned_data.columns:
            cleaned_data.set_index('tr_datetime', inplace=True)
        elif 'tr_date' in cleaned_data.columns and 'tr_time' in cleaned_data.columns:
            try:
                cleaned_data['datetime'] = pd.to_datetime(
                    cleaned_data['tr_date'] + ' ' + cleaned_data['tr_time']
                )
                cleaned_data.set_index('datetime', inplace=True)
            except Exception as e:
                logger.error(f"Error creating datetime index: {e}")
    
    # Sort by datetime
    cleaned_data = cleaned_data.sort_index()
    
    # Remove duplicates
    if cleaned_data.index.duplicated().any():
        dup_count = cleaned_data.index.duplicated().sum()
        logger.warning(f"Found {dup_count} duplicate timestamps in futures data")
        cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]
    
    logger.info(f"Futures data cleaning complete: {len(cleaned_data)} rows")
    return cleaned_data

def clean_options_data(options_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean and format options market data.
    
    Args:
        options_data: Dictionary of raw options data by expiry
        
    Returns:
        Dictionary of cleaned options data by expiry
    """
    logger.info("Cleaning options data")
    
    cleaned_options = {}
    
    for expiry, expiry_data in options_data.items():
        logger.info(f"Cleaning options data for expiry {expiry}")
        
        # Make a copy to avoid modifying original data
        cleaned_data = expiry_data.copy()
        
        # Handle missing values
        if cleaned_data.isnull().any().any():
            logger.warning(f"Found {cleaned_data.isnull().sum().sum()} missing values in options data for expiry {expiry}")
            
            # Forward fill for OHLC and OI data
            for col in ['tr_open', 'tr_high', 'tr_low', 'tr_close', 'open_interest']:
                if col in cleaned_data.columns:
                    cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
            
            # Fill missing volume with zeros
            if 'tr_volume' in cleaned_data.columns:
                cleaned_data['tr_volume'] = cleaned_data['tr_volume'].fillna(0)
            
            # For option-specific fields, fill with sensible defaults
            if 'strike_price' in cleaned_data.columns:
                # Group by option type and forward fill strike prices
                cleaned_data['strike_price'] = cleaned_data.groupby('otype')['strike_price'].fillna(method='ffill')
            
            # Drop any remaining rows with missing values in critical columns
            critical_cols = ['tr_open', 'tr_high', 'tr_low', 'tr_close', 'otype', 'strike_price']
            critical_cols = [col for col in critical_cols if col in cleaned_data.columns]
            
            if critical_cols:
                before_count = len(cleaned_data)
                cleaned_data = cleaned_data.dropna(subset=critical_cols)
                after_count = len(cleaned_data)
                
                if before_count > after_count:
                    logger.warning(f"Dropped {before_count - after_count} rows with missing values in critical columns")
        
        # Ensure datetime index
        if not isinstance(cleaned_data.index, pd.DatetimeIndex):
            if 'datetime' in cleaned_data.columns:
                cleaned_data.set_index('datetime', inplace=True)
            elif 'tr_datetime' in cleaned_data.columns:
                cleaned_data.set_index('tr_datetime', inplace=True)
            elif 'tr_date' in cleaned_data.columns and 'tr_time' in cleaned_data.columns:
                try:
                    cleaned_data['datetime'] = pd.to_datetime(
                        cleaned_data['tr_date'] + ' ' + cleaned_data['tr_time']
                    )
                    cleaned_data.set_index('datetime', inplace=True)
                except Exception as e:
                    logger.error(f"Error creating datetime index: {e}")
        
        # Sort by datetime
        cleaned_data = cleaned_data.sort_index()
        
        # Remove duplicates - need to handle multi-index (timestamp, strike, type)
        if 'otype' in cleaned_data.columns and 'strike_price' in cleaned_data.columns:
            # Check for duplicates across timestamp, strike, and option type
            dup_mask = cleaned_data.duplicated(subset=['otype', 'strike_price'], keep='first')
            if dup_mask.any():
                dup_count = dup_mask.sum()
                logger.warning(f"Found {dup_count} duplicate entries in options data for expiry {expiry}")
                cleaned_data = cleaned_data[~dup_mask]
        else:
            # Standard duplicate index check
            if cleaned_data.index.duplicated().any():
                dup_count = cleaned_data.index.duplicated().sum()
                logger.warning(f"Found {dup_count} duplicate timestamps in options data for expiry {expiry}")
                cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]
        
        cleaned_options[expiry] = cleaned_data
        logger.info(f"Options data cleaning complete for expiry {expiry}: {len(cleaned_data)} rows")
    
    return cleaned_options

def resample_data(data: pd.DataFrame, interval: str = RESAMPLE_INTERVAL) -> pd.DataFrame:
    """
    Resample time series data to a specified interval.
    
    Args:
        data: Time series data to resample
        interval: Resampling interval (e.g., '1T' for 1 minute)
        
    Returns:
        Resampled data
    """
    logger.info(f"Resampling data to {interval} interval")
    
    # Check if data has datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.error("Cannot resample data without datetime index")
        return data
    
    # Identify OHLC columns
    ohlc_cols = [col for col in ['tr_open', 'tr_high', 'tr_low', 'tr_close'] if col in data.columns]
    
    if not ohlc_cols:
        logger.warning("No OHLC columns found for resampling")
        return data
    
    # Prepare resampling rules
    agg_dict = {}
    
    for col in ohlc_cols:
        if col == 'tr_open':
            agg_dict[col] = 'first'
        elif col == 'tr_high':
            agg_dict[col] = 'max'
        elif col == 'tr_low':
            agg_dict[col] = 'min'
        elif col == 'tr_close':
            agg_dict[col] = 'last'
    
    # Add rules for other common columns
    if 'tr_volume' in data.columns:
        agg_dict['tr_volume'] = 'sum'
    
    if 'open_interest' in data.columns:
        agg_dict['open_interest'] = 'last'
    
    # Categorical/constant columns
    for col in ['ticker', 'stock_name', 'expiry_date', 'otype', 'strike_price']:
        if col in data.columns:
            agg_dict[col] = 'first'
    
    # Perform resampling
    resampled = data.resample(interval).agg(agg_dict).dropna()
    
    logger.info(f"Resampling complete: {len(data)} rows -> {len(resampled)} rows")
    return resampled

def align_timestamps(spot_data: pd.DataFrame, 
                     futures_data: pd.DataFrame,
                     options_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Align timestamps across different data sources with proper forward-filling.
    
    Args:
        spot_data: Spot market data
        futures_data: Futures market data
        options_data: Dictionary of options data by expiry
        
    Returns:
        Tuple of (aligned_spot, aligned_futures, aligned_options)
    """
    logger.info("Aligning timestamps across data sources with forward-filling")
    
    # Ensure all data has datetime index
    if not isinstance(spot_data.index, pd.DatetimeIndex):
        logger.error("Spot data must have datetime index for alignment")
        return spot_data, futures_data, options_data
    
    if not isinstance(futures_data.index, pd.DatetimeIndex):
        logger.error("Futures data must have datetime index for alignment")
        return spot_data, futures_data, options_data
    
    # Find common time range
    common_start = max(spot_data.index.min(), futures_data.index.min())
    common_end = min(spot_data.index.max(), futures_data.index.max())
    
    logger.info(f"Common time range: {common_start} to {common_end}")
    
    # Check if we have any options data in this range
    valid_options = False
    for expiry_data in options_data.values():
        if not expiry_data.empty and isinstance(expiry_data.index, pd.DatetimeIndex):
            if expiry_data.index.max() >= common_start and expiry_data.index.min() <= common_end:
                valid_options = True
                break
    
    if not valid_options:
        logger.warning("No options data overlaps with spot/futures data range")
        return spot_data, futures_data, options_data
    
    # Filter data to common range
    aligned_spot = spot_data.loc[(spot_data.index >= common_start) & (spot_data.index <= common_end)]
    aligned_futures = futures_data.loc[(futures_data.index >= common_start) & (futures_data.index <= common_end)]
    
    # Align options data
    aligned_options = {}
    for expiry, expiry_data in options_data.items():
        if not isinstance(expiry_data.index, pd.DatetimeIndex):
            logger.warning(f"Options data for expiry {expiry} does not have datetime index, skipping alignment")
            aligned_options[expiry] = expiry_data
            continue
        
        aligned_options[expiry] = expiry_data.loc[
            (expiry_data.index >= common_start) & (expiry_data.index <= common_end)
        ]
    
    # Get unique timestamps across all datasets
    all_timestamps = sorted(set(aligned_spot.index).union(set(aligned_futures.index)))
    
    for expiry_data in aligned_options.values():
        if isinstance(expiry_data.index, pd.DatetimeIndex):
            all_timestamps = sorted(set(all_timestamps).union(set(expiry_data.index)))
    
    logger.info(f"Found {len(all_timestamps)} unique timestamps across all datasets")
    
    # Reindex all data to common timestamps with forward-filling
    logger.info("Reindexing all data to common timestamps with forward-filling")
    aligned_spot = aligned_spot.reindex(all_timestamps, method='ffill')
    aligned_futures = aligned_futures.reindex(all_timestamps, method='ffill')
    
    # Forward-fill options data
    for expiry in aligned_options:
        if isinstance(aligned_options[expiry].index, pd.DatetimeIndex) and not aligned_options[expiry].empty:
            aligned_options[expiry] = aligned_options[expiry].reindex(all_timestamps, method='ffill')
    
    # Add check for gaps after forward-filling
    spot_nulls = aligned_spot.isnull().sum().sum()
    futures_nulls = aligned_futures.isnull().sum().sum()
    
    if spot_nulls > 0:
        logger.warning(f"Spot data has {spot_nulls} null values after forward-filling")
    
    if futures_nulls > 0:
        logger.warning(f"Futures data has {futures_nulls} null values after forward-filling")
    
    logger.info("Timestamp alignment complete with forward-filling")
    return aligned_spot, aligned_futures, aligned_options

def validate_price_continuity(options_data: Dict[str, pd.DataFrame], 
                           max_price_change_pct: float = 30.0) -> Dict[str, pd.DataFrame]:
    """
    Validate price continuity in options data and handle unrealistic price jumps.
    
    Args:
        options_data: Dictionary of options data by expiry
        max_price_change_pct: Maximum allowed price change percentage between consecutive records
        
    Returns:
        Dictionary of validated options data
    """
    logger.info(f"Validating price continuity with max allowed change of {max_price_change_pct}%")
    
    validated_options = {}
    total_anomalies = 0
    
    for expiry, expiry_data in options_data.items():
        if expiry_data.empty:
            validated_options[expiry] = expiry_data
            continue
        
        # Create a copy of the dataframe
        validated_df = expiry_data.copy()
        expiry_anomalies = 0
        
        # Group by option type and strike price
        for (option_type, strike), group in validated_df.groupby(['otype', 'strike_price']):
            if len(group) <= 1:
                continue
                
            # Sort by timestamp
            sorted_group = group.sort_index()
            
            # Calculate percentage price changes
            sorted_group['price_change_pct'] = sorted_group['tr_close'].pct_change() * 100
            
            # Identify unrealistic price changes
            anomalies = sorted_group[abs(sorted_group['price_change_pct']) > max_price_change_pct]
            
            if not anomalies.empty:
                expiry_anomalies += len(anomalies)
                
                for idx in anomalies.index:
                    # Get the row index in the original dataframe
                    df_idx = validated_df.index.get_loc(idx)
                    
                    # Get previous and current prices
                    prev_idx = max(0, df_idx - 1)
                    prev_price = validated_df.iloc[prev_idx]['tr_close']
                    curr_price = validated_df.loc[idx, 'tr_close']
                    
                    # Calculate an acceptable price (limit the change to max_price_change_pct)
                    change_direction = 1 if curr_price > prev_price else -1
                    max_allowed_change = prev_price * (max_price_change_pct / 100)
                    adjusted_price = prev_price + (change_direction * max_allowed_change)
                    
                    logger.warning(
                        f"Unrealistic price jump detected in {expiry} {option_type} {strike}: "
                        f"{prev_price:.2f} -> {curr_price:.2f} ({sorted_group.loc[idx, 'price_change_pct']:.2f}%). "
                        f"Adjusting to {adjusted_price:.2f}"
                    )
                    
                    # Fix the price in the dataframe
                    validated_df.loc[idx, 'tr_close'] = adjusted_price
                    
                    # Also adjust high/low if they're unrealistic
                    if option_type == 'CE' and validated_df.loc[idx, 'tr_high'] > adjusted_price * 1.1:
                        validated_df.loc[idx, 'tr_high'] = adjusted_price * 1.1
                    
                    if option_type == 'PE' and validated_df.loc[idx, 'tr_low'] < adjusted_price * 0.9:
                        validated_df.loc[idx, 'tr_low'] = adjusted_price * 0.9
        
        validated_options[expiry] = validated_df
        total_anomalies += expiry_anomalies
        
        if expiry_anomalies > 0:
            logger.warning(f"Fixed {expiry_anomalies} price anomalies in {expiry}")
    
    logger.info(f"Price continuity validation complete: fixed {total_anomalies} anomalies")
    return validated_options

def preprocess_all_data(spot_data: pd.DataFrame,
                       futures_data: pd.DataFrame,
                       options_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Preprocess all market data.
    
    Args:
        spot_data: Raw spot market data
        futures_data: Raw futures market data
        options_data: Dictionary of raw options data by expiry
        
    Returns:
        Tuple of (processed_spot, processed_futures, processed_options)
    """
    logger.info("Starting data preprocessing")
    
    # Clean data
    cleaned_spot = clean_spot_data(spot_data)
    cleaned_futures = clean_futures_data(futures_data)
    cleaned_options = clean_options_data(options_data)
    
    # Resample if needed
    resampled_spot = resample_data(cleaned_spot)
    resampled_futures = resample_data(cleaned_futures)
    
    resampled_options = {}
    for expiry, expiry_data in cleaned_options.items():
        resampled_options[expiry] = resample_data(expiry_data)
    
    # Validate price continuity in options data
    validated_options = validate_price_continuity(resampled_options)
    
    # Align timestamps
    aligned_spot, aligned_futures, aligned_options = align_timestamps(
        resampled_spot, resampled_futures, validated_options
    )
    
    logger.info("Data preprocessing complete")
    return aligned_spot, aligned_futures, aligned_options