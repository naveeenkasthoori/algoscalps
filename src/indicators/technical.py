"""
Technical indicators for market analysis.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict

from config.settings import VWAP_STD_DEV_PERIODS

logger = logging.getLogger(__name__)

def calculate_cvd(data: pd.DataFrame, volume_col: str = 'tr_volume') -> pd.Series:
    """
    Calculate the Cumulative Volume Delta (CVD) indicator.
    
    CVD measures the net buying or selling pressure by considering candle components
    weighted by the volume during that period. The CVD resets at the start of each
    trading day.
    
    Args:
        data: DataFrame with OHLCV data
        volume_col: Name of the volume column
        
    Returns:
        Series with CVD values that reset daily
    """
    logger.info("Calculating Cumulative Volume Delta")
    
    if volume_col not in data.columns:
        raise ValueError(f"Volume column '{volume_col}' not found in data")
    
    # Add timestamp column from index if not present
    if 'timestamp' not in data.columns:
        data = data.copy()
        data['timestamp'] = data.index
    
    # Add date column for grouping
    data['date'] = pd.to_datetime(data['timestamp']).dt.date
    
    # Calculate candle components
    data.loc[:, 'tw'] = data['tr_high'] - np.maximum(data['tr_open'], data['tr_close'])
    data.loc[:, 'bw'] = np.minimum(data['tr_open'], data['tr_close']) - data['tr_low']
    data.loc[:, 'body'] = np.abs(data['tr_close'] - data['tr_open'])
    
    # Define rate calculation function
    def calculate_rate(row):
        numerator = row['tw'] + row['bw']
        # Add 2*body if condition met (close >= open)
        if row['tr_close'] >= row['tr_open']:
            numerator += 2 * row['body']
        denominator = row['tw'] + row['bw'] + row['body']
        
        if denominator == 0:
            return 0.5
        return 0.5 * numerator / denominator
    
    # Calculate rate for each row
    data.loc[:, 'rate'] = data.apply(calculate_rate, axis=1)
    
    # Calculate delta components
    data.loc[:, 'delta_up'] = data[volume_col] * data['rate']
    data.loc[:, 'delta_down'] = data[volume_col] * (1 - data['rate'])
    
    # Calculate final delta
    data.loc[:, 'delta'] = np.where(data['tr_close'] >= data['tr_open'],
                                   data['delta_up'],
                                   -data['delta_down'])
    
    # Calculate cumulative delta grouped by date
    cvd = data.groupby('date')['delta'].cumsum()
    
    logger.info("CVD calculation complete")
    return cvd

def calculate_vwap(data: pd.DataFrame, volume_col: str = 'tr_volume') -> Dict[str, pd.Series]:
    """
    Calculate Volume Weighted Average Price (VWAP) with standard deviation bands.
    Enhanced with fallback for missing volume data.
    Resets calculations daily.
    
    Args:
        data: DataFrame with OHLCV data
        volume_col: Name of the volume column
        
    Returns:
        Dictionary containing:
            - vwap: VWAP values
            - stdev: Standard deviation
            - prev_day_vwap: Previous day's closing VWAP
            - upper_1 through upper_5: Upper bands (1.28σ to 4.01σ)
            - lower_1 through lower_5: Lower bands (1.28σ to 4.01σ)
            - vwap_upper: Alias for upper_2 (compatibility)
            - vwap_lower: Alias for lower_2 (compatibility)
    """
    logger.info("Calculating VWAP and bands with fallback mechanism")
    
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Check if volume column exists, if not create a proxy
    if volume_col not in df.columns:
        logger.warning(f"Volume column '{volume_col}' not found, creating proxy volume")
        
        # Create proxy volume using price range and change
        df['proxy_volume'] = (df['tr_high'] - df['tr_low']) * abs(df['tr_close'] - df['tr_open']) * 100
        df['proxy_volume'] = df['proxy_volume'].replace(0, 1)  # Ensure no zeros
        volume_col = 'proxy_volume'
    
    # Ensure we have a date column for grouping
    if 'tr_date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df['tr_date'] = df.index.date
    
    # 1. Calculate typical price (using high, low, and close)
    df['typical_price'] = (df['tr_high'] + df['tr_low'] + df['tr_close']) / 3
    
    # 2. Compute typical price * volume and squared terms
    df['tpv'] = df['typical_price'] * df[volume_col]
    df['tpv2'] = (df['typical_price'] ** 2) * df[volume_col]
    
    # Calculate cumulative sums grouped by date with error handling
    try:
        df['cum_tpv'] = df.groupby('tr_date')['tpv'].cumsum()
        df['cum_volume'] = df.groupby('tr_date')[volume_col].cumsum()
        df['cum_tpv2'] = df.groupby('tr_date')['tpv2'].cumsum()
    except Exception as e:
        logger.error(f"Error in VWAP groupby calculation: {e}")
        # Fallback: calculate without grouping by date
        df['cum_tpv'] = df['tpv'].cumsum()
        df['cum_volume'] = df[volume_col].cumsum()
        df['cum_tpv2'] = df['tpv2'].cumsum()
    
    # 3. Calculate VWAP and standard deviation
    df['vwap'] = df['cum_tpv'] / df['cum_volume'].replace(0, 1)  # Prevent division by zero
    df['variance'] = (df['cum_tpv2'] / df['cum_volume'].replace(0, 1)) - (df['vwap'] ** 2)
    
    # Ensure non-negative variance with error handling
    df['variance'] = df['variance'].clip(lower=0)
    df['stdev'] = np.sqrt(df['variance'])
    
    # 4. Generate standard deviation bands
    deviations = [1.28, 2.01, 2.51, 3.09, 4.01]
    for i, dev in enumerate(deviations, 1):
        df[f'upper_{i}'] = df['vwap'] + dev * df['stdev']
        df[f'lower_{i}'] = df['vwap'] - dev * df['stdev']
    
    # 5. Get previous day's VWAP close
    try:
        daily_last_vwap = df.groupby('tr_date')['vwap'].last().reset_index()
        daily_last_vwap['next_date'] = pd.to_datetime(daily_last_vwap['tr_date']) + pd.Timedelta(days=1)
        daily_last_vwap['next_date'] = daily_last_vwap['next_date'].dt.date
        
        # Create a mapping dictionary for faster lookups
        prev_vwap_map = dict(zip(daily_last_vwap['next_date'], daily_last_vwap['vwap']))
        
        # Map previous day's VWAP to current rows
        df['prev_day_vwap'] = df['tr_date'].map(prev_vwap_map)
        
        # Fill missing values with current day's first VWAP value
        if df['prev_day_vwap'].isnull().any():
            first_day_vwap = df.groupby('tr_date')['vwap'].first().to_dict()
            df['prev_day_vwap'] = df.apply(
                lambda row: first_day_vwap.get(row['tr_date']) 
                if pd.isnull(row['prev_day_vwap']) else row['prev_day_vwap'], 
                axis=1
            )
    except Exception as e:
        logger.error(f"Error calculating previous day VWAP: {e}")
        # Fallback: use the first VWAP value
        df['prev_day_vwap'] = df['vwap'].iloc[0] if not df.empty else 0
    
    # Prepare return values
    result = {
        'vwap': df['vwap'],
        'stdev': df['stdev'],
        'prev_day_vwap': df['prev_day_vwap']
    }
    
    # Add bands to result
    for i, _ in enumerate(deviations, 1):
        result[f'upper_{i}'] = df[f'upper_{i}']
        result[f'lower_{i}'] = df[f'lower_{i}']
    
    # For compatibility with existing code
    result['vwap_upper'] = df['upper_2']  # ~2 standard deviations
    result['vwap_lower'] = df['lower_2']
    
    logger.info("VWAP calculation complete")
    return result

def detect_vwap_breakout(data: pd.DataFrame, vwap: pd.Series, upper_band: pd.Series, 
                         lower_band: pd.Series, threshold: float = 0.005, 
                         duration: int = 2) -> pd.Series:
    """
    Detect price breakouts beyond VWAP bands with duration requirement.
    
    Args:
        data: DataFrame with price data
        vwap: VWAP values
        upper_band: Upper VWAP band
        lower_band: Lower VWAP band
        threshold: Additional threshold beyond bands (percentage)
        duration: Minimum number of consecutive periods beyond threshold
        
    Returns:
        Series with breakout signals (1 for upward, -1 for downward, 0 for none)
    """
    # Check for price beyond bands plus threshold
    upper_threshold = upper_band * (1 + threshold)
    lower_threshold = lower_band * (1 - threshold)
    
    # Initialize signal series
    signals = pd.Series(0, index=data.index)
    
    # Calculate immediate breakouts first
    above_upper = data['tr_close'] > upper_threshold
    below_lower = data['tr_close'] < lower_threshold
    
    # Now check duration requirement
    if duration > 1:
        # For upward breakouts
        for i in range(duration-1, len(above_upper)):
            if all(above_upper.iloc[i-duration+1:i+1].values):
                signals.iloc[i] = 1
        
        # For downward breakouts
        for i in range(duration-1, len(below_lower)):
            if all(below_lower.iloc[i-duration+1:i+1].values):
                signals.iloc[i] = -1
    else:
        # If duration is 1 or less, use simple threshold crossing
        signals.loc[above_upper] = 1
        signals.loc[below_lower] = -1
    
    # Additional check: Consider VWAP slope
    if 'vwap' in data.columns:
        vwap_slope = vwap.diff().rolling(window=3).mean()
        
        # Strengthen signals that align with VWAP slope direction
        for i in range(len(signals)):
            if signals.iloc[i] == 1 and i > 0 and vwap_slope.iloc[i] > 0:
                signals.iloc[i] = 1.5  # Stronger upward signal when VWAP is rising
            elif signals.iloc[i] == -1 and i > 0 and vwap_slope.iloc[i] < 0:
                signals.iloc[i] = -1.5  # Stronger downward signal when VWAP is falling
    
    return signals

def detect_cvd_breakout(cvd: pd.Series, lookback_period: int = 5, threshold: float = 1.5) -> pd.Series:
    """
    Detect significant breakouts in CVD.
    
    Args:
        cvd: Series with CVD values
        lookback_period: Period for calculating CVD rate of change
        threshold: Number of standard deviations for breakout signal
        
    Returns:
        Series with breakout signals (1 for upward, -1 for downward, 0 for none)
    """
    # Calculate CVD change rate
    cvd_change = cvd.diff(lookback_period)
    
    # Calculate rolling mean and standard deviation of CVD change
    mean = cvd_change.rolling(window=lookback_period*5).mean()
    std = cvd_change.rolling(window=lookback_period*5).std()
    
    # Generate signals
    signals = pd.Series(0, index=cvd.index)
    signals.loc[cvd_change > (mean + threshold * std)] = 1
    signals.loc[cvd_change < (mean - threshold * std)] = -1
    
    return signals

def calculate_enhanced_cvd(data: pd.DataFrame, volume_col: str = 'tr_volume', lookback: int = 10,
                         use_dask: bool = False, chunk_size: int = 5000) -> pd.DataFrame:
    """
    Calculate an enhanced Cumulative Volume Delta (CVD) with stronger signals.
    
    Args:
        data: DataFrame with OHLCV data
        volume_col: Name of the volume column
        lookback: Lookback period for normalization
        use_dask: Whether to use Dask for parallel processing
        chunk_size: Size of chunks when using Dask
        
    Returns:
        DataFrame with CVD values and normalized signal strength
    """
    logger.info("Calculating Enhanced CVD")
    
    if use_dask:
        try:
            import dask.dataframe as dd
            logger.info("Using Dask for parallel processing")
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(data, npartitions=max(1, len(data) // chunk_size))
        except ImportError:
            logger.warning("Dask not installed. Falling back to pandas")
            use_dask = False
    
    if volume_col not in data.columns:
        logger.warning(f"Volume column '{volume_col}' not found, using proxy volume")
        # Create proxy volume from price movement and range
        data['proxy_volume'] = (data['tr_high'] - data['tr_low']) * (abs(data['tr_close'] - data['tr_open']))
        volume_col = 'proxy_volume'
    
    # Copy to avoid modifying original data
    df = data.copy()
    
    # Add timestamp and date columns for grouping
    if 'timestamp' not in df.columns:
        df['timestamp'] = df.index
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Calculate bullish/bearish classification with stronger bias
    df['bullish'] = (df['tr_close'] > df['tr_open']).astype(int)
    
    # Enhanced volume delta calculation
    df['body'] = abs(df['tr_close'] - df['tr_open'])
    df['range'] = df['tr_high'] - df['tr_low']
    df['body_to_range'] = df['body'] / df['range'].replace(0, 0.001)
    
    # Calculate volume delta with enhanced weighting
    df['vol_multiplier'] = 1 + df['body_to_range']
    
    # Calculate bullish and bearish volume
    df['bull_volume'] = df[volume_col] * df['vol_multiplier'] * df['bullish']
    df['bear_volume'] = df[volume_col] * df['vol_multiplier'] * (1 - df['bullish'])
    
    # Final delta
    df['enhanced_delta'] = df['bull_volume'] - df['bear_volume']
    
    if use_dask:
        # Compute CVD using Dask
        df['cvd'] = df.groupby('date')['enhanced_delta'].cumsum().compute()
    else:
        # Calculate cumulative delta by date using pandas
        df['cvd'] = df.groupby('date')['enhanced_delta'].cumsum()
    
    # Initialize signal column
    df['cvd_signal'] = 0
    
    # Calculate signals on rolling basis
    for i in range(lookback, len(df)):
        window = df['cvd'].iloc[i-lookback:i]
        mean = window.mean()
        std = window.std() if len(window) > 1 else 1.0
        
        if std > 0:
            z_score = (df['cvd'].iloc[i] - mean) / std
            
            if z_score > 1.5:  # Changed from 1.0 to 1.5
                df.loc[df.index[i], 'cvd_signal'] = 1
            elif z_score < -1.5:  # Changed from -1.0 to -1.5
                df.loc[df.index[i], 'cvd_signal'] = -1
            else:
                if 'vwap' in df.columns:
                    df.loc[df.index[i], 'cvd_signal'] = 1 if df['tr_close'].iloc[i] > df['vwap'].iloc[i] else -1
                else:
                    df.loc[df.index[i], 'cvd_signal'] = 0
    
    logger.info("Enhanced CVD calculation complete")
    return df[['cvd', 'cvd_signal']]

def calculate_all_technical_indicators(futures_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for futures data.
    """
    logger.info("Calculating all technical indicators")
    
    result = futures_data.copy()
    
    # Calculate enhanced CVD (replaces original CVD calculation)
    try:
        # Determine if we should use Dask based on data size
        use_dask = len(result) > 50000
        cvd_result = calculate_enhanced_cvd(
            result, 
            lookback=10, 
            use_dask=use_dask, 
            chunk_size=5000
        )
        result['cvd'] = cvd_result['cvd']
        result['cvd_signal'] = cvd_result['cvd_signal']
    except Exception as e:
        logger.error(f"Error calculating enhanced CVD: {e}")
    
    # Calculate VWAP and bands
    try:
        vwap_data = calculate_vwap(result)
        # Assign VWAP values to result DataFrame
        result['vwap'] = vwap_data['vwap']
        result['vwap_upper'] = vwap_data['vwap_upper']
        result['vwap_lower'] = vwap_data['vwap_lower']
        result['vwap_stdev'] = vwap_data['stdev']
        result['prev_day_vwap'] = vwap_data['prev_day_vwap']
        
        # Add VWAP bands if needed
        for i in range(1, 6):
            result[f'vwap_upper_{i}'] = vwap_data[f'upper_{i}']
            result[f'vwap_lower_{i}'] = vwap_data[f'lower_{i}']
            
    except Exception as e:
        logger.error(f"Error calculating VWAP: {e}")
    
    logger.info("Technical indicator calculations complete")
    return result