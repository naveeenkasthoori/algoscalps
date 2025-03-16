"""
Data loading and preprocessing module.
Handles loading data from CSV files and organizing it for the trading engine.
"""
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
import dask.dataframe as dd
import multiprocessing
import psutil
from src.utils.date_utils import validate_expiry_date, get_next_expiry_date
import numpy as np

from config.settings import (
    SPOT_DATA_DIR, FUTURES_DATA_DIR, OPTIONS_DATA_DIR,
    PROCESSED_DATA_DIR, RESAMPLE_INTERVAL, LOG_DIR
)

logger = logging.getLogger(__name__)

class EnhancedDataLoader:
    """
    Handles loading and preprocessing of market data with optimized memory usage
    and parallel processing for large datasets.
    """
    
    def __init__(self, start_date: Optional[str] = None, end_date: Optional[str] = None, 
                 num_workers: Optional[int] = None, chunksize: int = 500000,
                 use_dask: bool = False):
        """
        Initialize the enhanced data loader.
        
        Args:
            start_date: Start date for filtering data (format: YYYY-MM-DD)
            end_date: End date for filtering data (format: YYYY-MM-DD)
            num_workers: Number of worker processes for parallel processing (default: CPU count)
            chunksize: Size of chunks for reading large files
            use_dask: Whether to use Dask for processing very large files
        """
        self.spot_data = None
        self.futures_data = None
        self.options_data = {}  # Dictionary to store options data by expiry
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.num_workers = num_workers if num_workers else max(1, multiprocessing.cpu_count() - 1)
        self.chunksize = chunksize
        self.use_dask = use_dask
        
        logger.info(f"Enhanced Data Loader initialized with date range: {start_date} to {end_date}")
        logger.info(f"Using {self.num_workers} workers for parallel processing")
        logger.info(f"Using chunk size of {self.chunksize} rows")
        logger.info(f"Dask processing: {'enabled' if self.use_dask else 'disabled'}")
        
        # Create required directories if they don't exist
        for directory in [LOG_DIR, PROCESSED_DATA_DIR]:
            os.makedirs(directory, exist_ok=True)

        # Add column mapping dictionary
        self.column_mappings = {
            'volume': 'tr_volume',
            'option_type': 'otype',
            'datetime': 'tr_datetime',
            'date': 'tr_date',
            'time': 'tr_time',
            'open': 'tr_open',
            'high': 'tr_high',
            'low': 'tr_low',
            'close': 'tr_close'
        }
        
        # Critical columns that must be present
        self.critical_columns = {
            'spot': ['tr_open', 'tr_high', 'tr_low', 'tr_close'],
            'futures': ['tr_open', 'tr_high', 'tr_low', 'tr_close', 'open_interest'],
            'options': ['tr_open', 'tr_high', 'tr_low', 'tr_close', 'strike_price', 
                       'tr_volume', 'open_interest', 'otype']
        }

    def _cleanup_memory(self):
        """
        Clean up memory by forcing garbage collection and clearing pandas cache.
        """
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Clear pandas cache if available
        try:
            from pandas import _libs
            _libs.lib.clear_cache()
        except:
            pass
        
        # Log memory usage
        self._log_memory_usage("After memory cleanup")

    def _log_memory_usage(self, label: str = ""):
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage ({label}): {memory_usage_mb:.2f} MB")
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting to appropriate dtypes.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        start_mem = df.memory_usage().sum() / 1024**2
        
        # Numeric columns
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Object columns (strings)
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['tr_date', 'tr_time', 'datetime']:  # Skip date columns
                df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        
        logger.debug(f"Memory optimization: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% reduction)")
        
        return df

    def _standardize_columns(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Standardize column names and verify critical columns.
        
        Args:
            df: DataFrame to process
            data_type: Type of data ('spot', 'futures', or 'options')
            
        Returns:
            DataFrame with standardized column names
        """
        # Log original columns
        logger.debug(f"Original columns for {data_type} data: {list(df.columns)}")
        
        # Enhanced column mappings dictionary
        enhanced_mappings = {
            'volume': 'tr_volume',
            'option_type': 'otype',
            'otype': 'otype',  # Prevent duplicate mapping
            'expiry_date': 'expiry_date',
            'week_expiry_date': 'expiry_date',  # Standardize week_expiry_date
            'datetime': 'tr_datetime',
            'date': 'tr_date',
            'time': 'tr_time',
            'open': 'tr_open',
            'high': 'tr_high',
            'low': 'tr_low',
            'close': 'tr_close',
            # Add additional common variations
            'Volume': 'tr_volume',
            'VOLUME': 'tr_volume',
            'Open': 'tr_open',
            'High': 'tr_high',
            'Low': 'tr_low',
            'Close': 'tr_close',
            'DATE': 'tr_date',
            'Time': 'tr_time',
            'TIME': 'tr_time',
            'DateTime': 'tr_datetime',
            'DATETIME': 'tr_datetime'
        }
        
        # Update the instance column_mappings with enhanced mappings
        self.column_mappings.update(enhanced_mappings)
        
        # Map common column variations to standard names
        columns_mapped = set()
        for orig_col, std_col in self.column_mappings.items():
            if orig_col in df.columns and std_col not in df.columns:
                logger.info(f"Mapping column '{orig_col}' to '{std_col}'")
                df[std_col] = df[orig_col]
                columns_mapped.add(orig_col)
        
        # Special handling for option type columns
        if data_type == 'options':
            if 'otype' not in df.columns:
                if 'option_type' in df.columns:
                    logger.info("Converting 'option_type' to 'otype'")
                    df['otype'] = df['option_type']
                    # Standardize option type values
                    df['otype'] = df['otype'].str.upper()
                    df['otype'] = df['otype'].map({'CALL': 'CE', 'PUT': 'PE', 'CE': 'CE', 'PE': 'PE'})
                else:
                    logger.warning("No option type column found in options data, attempting to infer from strike price")
                    try:
                        # Try to infer option type from strike price
                        if 'strike_price' in df.columns:
                            # Get unique dates in the data
                            dates = df.index.unique()
                            df['otype'] = 'NA'  # Initialize with placeholder
                            
                            for date in dates:
                                date_data = df[df.index == date]
                                if len(date_data) > 0:
                                    # Find approximate ATM price for this date
                                    avg_strike = date_data['strike_price'].median()
                                    # Infer option type (CE for strikes above median, PE for below)
                                    df.loc[df.index == date, 'otype'] = np.where(
                                        df.loc[df.index == date, 'strike_price'] >= avg_strike,
                                        'CE', 'PE'
                                    )
                            
                            # Verify inference results
                            type_counts = df['otype'].value_counts()
                            logger.info(f"Inferred option types: CE={type_counts.get('CE', 0)}, PE={type_counts.get('PE', 0)}")
                            
                            if df['otype'].isin(['CE', 'PE']).all():
                                logger.info("Successfully inferred all option types")
                            else:
                                logger.warning(f"Some option types could not be inferred: {df['otype'].unique()}")
                        else:
                            raise ValueError("Cannot infer option type - strike_price column missing")
                    except Exception as e:
                        logger.error(f"Error inferring option types: {str(e)}")
                        raise ValueError(f"Failed to infer option types: {str(e)}")
        
        # Special handling for expiry date columns
        if data_type in ['options', 'futures']:
            if 'expiry_date' not in df.columns and 'week_expiry_date' in df.columns:
                logger.info("Standardizing 'week_expiry_date' to 'expiry_date'")
                df['expiry_date'] = pd.to_datetime(df['week_expiry_date'])
        
        # Verify critical columns
        missing_cols = [col for col in self.critical_columns[data_type] 
                       if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing critical columns for {data_type} data: {missing_cols}")
            
            # Try to handle missing columns
            for col in missing_cols:
                # Special handling for volume/tr_volume
                if col == 'tr_volume' and 'volume' in df.columns:
                    df['tr_volume'] = df['volume']
                    logger.info("Mapped 'volume' to 'tr_volume'")
                    missing_cols.remove('tr_volume')
                
                # Special handling for option type
                elif col == 'otype' and 'option_type' in df.columns:
                    df['otype'] = df['option_type']
                    logger.info("Mapped 'option_type' to 'otype'")
                    missing_cols.remove('otype')
        
        if missing_cols:
            raise ValueError(f"Critical columns still missing for {data_type} data: {missing_cols}")
        
        # Log final columns
        logger.info(f"Final columns for {data_type} data: {list(df.columns)}")
        
        return df

    def load_spot_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load spot market data from CSV file.
        
        Args:
            file_path: Optional path to the spot data CSV file.
                       If None, will look in the default directory.
                       
        Returns:
            DataFrame with spot market data.
        """
        if file_path is None:
            # Find the latest file in the spot data directory
            files = os.listdir(SPOT_DATA_DIR)
            if not files:
                raise FileNotFoundError(f"No spot data files found in {SPOT_DATA_DIR}")
            file_path = os.path.join(SPOT_DATA_DIR, sorted(files)[-1])
            
        logger.info(f"Loading spot data from {file_path}")
        
        try:
            # Load the spot data
            spot_data = pd.read_csv(file_path)
            
            # Standardize column names
            spot_data = self._standardize_columns(spot_data, 'spot')
            
            # Convert date and time columns to datetime
            if 'tr_date' in spot_data.columns and 'tr_time' in spot_data.columns:
                spot_data['datetime'] = pd.to_datetime(
                    spot_data['tr_date'] + ' ' + spot_data['tr_time']
                )
            elif 'datetime' not in spot_data.columns:
                raise ValueError("Spot data must have either 'datetime' column or 'tr_date' and 'tr_time' columns")
                
            # Set datetime as index
            spot_data.set_index('datetime', inplace=True)
            
            # Log available date range before filtering
            data_start = spot_data.index.min()
            data_end = spot_data.index.max()
            logger.info(f"Spot data available range: {data_start} to {data_end}")
            
            # Add date range validation
            if self.start_date and self.end_date:
                if self.start_date < data_start:
                    raise ValueError(f"Requested start date {self.start_date} is before available data start date {data_start}")
                if self.end_date > data_end:
                    raise ValueError(f"Requested end date {self.end_date} is after available data end date {data_end}")
                    
                before_filter = len(spot_data)
                spot_data = spot_data[
                    (spot_data.index >= self.start_date) & 
                    (spot_data.index <= self.end_date)
                ]
                after_filter = len(spot_data)
                
                if after_filter == 0:
                    raise ValueError(
                        f"No data found in requested date range {self.start_date} to {self.end_date}. "
                        f"Available data range is {data_start} to {data_end}"
                    )
                
                logger.info(f"Date filtering: {before_filter} -> {after_filter} rows")
            
            # Ensure OHLC columns exist
            required_columns = ['tr_open', 'tr_high', 'tr_low', 'tr_close']
            missing_columns = [col for col in required_columns if col not in spot_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in spot data: {missing_columns}")
                
            # Resample to the specified interval if needed
            spot_data = spot_data.resample(RESAMPLE_INTERVAL).agg({
                'tr_open': 'first',
                'tr_high': 'max',
                'tr_low': 'min',
                'tr_close': 'last'
            })
            
            self.spot_data = spot_data
            logger.info(f"Loaded spot data with {len(spot_data)} rows")
            return spot_data
            
        except Exception as e:
            logger.error(f"Error loading spot data: {e}")
            raise
    
    def load_futures_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load futures market data from CSV file.
        
        Args:
            file_path: Optional path to the futures data CSV file.
                       If None, will look in the default directory.
                       
        Returns:
            DataFrame with futures market data.
        """
        if file_path is None:
            # Find the latest file in the futures data directory
            files = os.listdir(FUTURES_DATA_DIR)
            if not files:
                raise FileNotFoundError(f"No futures data files found in {FUTURES_DATA_DIR}")
            file_path = os.path.join(FUTURES_DATA_DIR, sorted(files)[-1])
            
        logger.info(f"Loading futures data from {file_path}")
        
        try:
            # Load the futures data
            futures_data = pd.read_csv(file_path)
            
            # Standardize column names
            futures_data = self._standardize_columns(futures_data, 'futures')
            
            # Convert date and time columns to datetime
            if 'tr_date' in futures_data.columns and 'tr_time' in futures_data.columns:
                futures_data['datetime'] = pd.to_datetime(
                    futures_data['tr_date'] + ' ' + futures_data['tr_time']
                )
            elif 'tr_datetime' in futures_data.columns:
                futures_data['datetime'] = pd.to_datetime(futures_data['tr_datetime'])
            elif 'datetime' not in futures_data.columns:
                raise ValueError("Futures data must have either 'datetime', 'tr_datetime', or 'tr_date' and 'tr_time' columns")
                
            # Set datetime as index
            futures_data.set_index('datetime', inplace=True)
            
            # Log available date range before filtering
            data_start = futures_data.index.min()
            data_end = futures_data.index.max()
            logger.info(f"Futures data available range: {data_start} to {data_end}")
            
            # Add date range validation
            if self.start_date and self.end_date:
                if self.start_date < data_start:
                    raise ValueError(f"Requested start date {self.start_date} is before available data start date {data_start}")
                if self.end_date > data_end:
                    raise ValueError(f"Requested end date {self.end_date} is after available data end date {data_end}")
            
                before_filter = len(futures_data)
                futures_data = futures_data[
                    (futures_data.index >= self.start_date) & 
                    (futures_data.index <= self.end_date)
                ]
                after_filter = len(futures_data)
                
                if after_filter == 0:
                    raise ValueError(
                        f"No data found in requested date range {self.start_date} to {self.end_date}. "
                        f"Available data range is {data_start} to {data_end}"
                    )
                
                logger.info(f"Date filtering: {before_filter} -> {after_filter} rows")
            
            # Ensure required columns exist
            required_columns = ['tr_open', 'tr_high', 'tr_low', 'tr_close', 'open_interest']
            missing_columns = [col for col in required_columns if col not in futures_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in futures data: {missing_columns}")
                
            # Resample to the specified interval if needed
            futures_data = futures_data.resample(RESAMPLE_INTERVAL).agg({
                'tr_open': 'first',
                'tr_high': 'max',
                'tr_low': 'min',
                'tr_close': 'last',
                'open_interest': 'last',
                'tr_volume': 'sum' if 'tr_volume' in futures_data.columns else None
            }).dropna()
            
            self.futures_data = futures_data
            logger.info(f"Loaded futures data with {len(futures_data)} rows")
            return futures_data
            
        except Exception as e:
            logger.error(f"Error loading futures data: {e}")
            raise
    
    def load_options_data(self, directory: Optional[str] = None, chunksize: int = 100000) -> Dict[str, pd.DataFrame]:
        """
        Load options market data from CSV files with support for both single-expiry and consolidated files.
        
        Args:
            directory: Directory containing options data CSV files
            chunksize: Size of chunks for reading large files
            
        Returns:
            Dictionary of DataFrames with options data by expiry
        """
        if directory is None:
            directory = OPTIONS_DATA_DIR
        
        logger.info(f"Loading options data from {directory} using chunks of {chunksize} rows")
        
        options_data = {}
        files_processed = 0
        total_rows_processed = 0
        
        try:
            csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
            total_files = len(csv_files)
            
            for file_name in csv_files:
                file_path = os.path.join(directory, file_name)
                logger.info(f"Processing options file ({files_processed + 1}/{total_files}): {file_name}")
                
                try:
                    # Read file metadata and determine structure
                    sample = pd.read_csv(file_path, nrows=5)
                    file_columns = sample.columns.tolist()
                    
                    # Determine datetime columns
                    datetime_cols = []
                    if 'tr_datetime' in file_columns:
                        datetime_cols = ['tr_datetime']
                    elif all(col in file_columns for col in ['tr_date', 'tr_time']):
                        datetime_cols = ['tr_date', 'tr_time']
                    
                    if not datetime_cols:
                        logger.warning(f"No datetime columns found in {file_name}, skipping file")
                        continue
                    
                    # Determine expiry column
                    expiry_col = None
                    if 'week_expiry_date' in file_columns:
                        expiry_col = 'week_expiry_date'
                    elif 'expiry_date' in file_columns:
                        expiry_col = 'expiry_date'
                    
                    # Define essential columns to read
                    essential_cols = datetime_cols + [
                        'tr_open', 'tr_high', 'tr_low', 'tr_close',
                        'otype', 'strike_price', 'open_interest', 'tr_volume'
                    ]
                    if expiry_col:
                        essential_cols.append(expiry_col)
                    
                    # Define optimized dtypes
                    dtype_dict = {
                        'tr_open': 'float32',
                        'tr_high': 'float32',
                        'tr_low': 'float32',
                        'tr_close': 'float32',
                        'strike_price': 'float32',
                        'open_interest': 'int32',
                        'tr_volume': 'int32',
                        'otype': 'category'
                    }
                    
                    # Process file in chunks
                    chunk_data = {}
                    for chunk_num, chunk in enumerate(pd.read_csv(file_path, 
                                                                usecols=essential_cols,
                                                                dtype=dtype_dict,
                                                                chunksize=chunksize)):
                        
                        # Create datetime index
                        if 'tr_datetime' in chunk.columns:
                            chunk['datetime'] = pd.to_datetime(chunk['tr_datetime'])
                        else:
                            chunk['datetime'] = pd.to_datetime(chunk['tr_date'] + ' ' + chunk['tr_time'])
                        
                        chunk.set_index('datetime', inplace=True)
                        
                        # Filter by date range if specified
                        if self.start_date and self.end_date:
                            chunk = chunk[
                                (chunk.index >= self.start_date) & 
                                (chunk.index <= self.end_date)
                            ]
                        
                        if chunk.empty:
                            continue
                        
                        # Group by expiry if available
                        if expiry_col:
                            chunk[expiry_col] = pd.to_datetime(chunk[expiry_col])
                            for expiry, group in chunk.groupby(expiry_col):
                                expiry_key = str(expiry.date())
                                if expiry_key not in chunk_data:
                                    chunk_data[expiry_key] = []
                                chunk_data[expiry_key].append(group)
                        else:
                            # Use filename or data range for expiry
                            expiry_key = str(chunk.index.max().date())
                            if expiry_key not in chunk_data:
                                chunk_data[expiry_key] = []
                            chunk_data[expiry_key].append(chunk)
                        
                        total_rows_processed += len(chunk)
                        
                        if (chunk_num + 1) % 10 == 0:
                            logger.debug(f"Processed {chunk_num + 1} chunks of {file_name}")
                            self._cleanup_memory()
                    
                    # Combine chunks for each expiry
                    for expiry_key, chunks in chunk_data.items():
                        if chunks:
                            combined_data = pd.concat(chunks, axis=0)
                            # Remove duplicates and sort
                            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
                            combined_data.sort_index(inplace=True)
                            
                            # Standardize column names for each options DataFrame
                            combined_data = self._standardize_columns(combined_data, 'options')
                            
                            if expiry_key in options_data:
                                options_data[expiry_key] = pd.concat([options_data[expiry_key], combined_data])
                            else:
                                options_data[expiry_key] = combined_data
                    
                    files_processed += 1
                    logger.info(f"Completed processing {file_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {str(e)}")
                    continue
                
                # Cleanup after each file
                self._cleanup_memory()
            
            # Final cleanup and sorting of all expiry data
            for expiry_key in list(options_data.keys()):
                df = options_data[expiry_key]
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)
                options_data[expiry_key] = df
                
                logger.info(
                    f"Expiry {expiry_key}: {len(df):,} rows, "
                    f"from {df.index.min()} to {df.index.max()}"
                )
            
            logger.info(
                f"Options data loading complete: "
                f"{files_processed:,} files processed, "
                f"{len(options_data):,} expiry groups, "
                f"{total_rows_processed:,} total rows"
            )
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error loading options data: {e}")
            raise
    
    def load_options_data_with_dask(self, directory: Optional[str] = None) -> Dict[str, dd.DataFrame]:
        """
        Load options data using Dask for better memory management.
        """
        if directory is None:
            directory = OPTIONS_DATA_DIR
        
        logger.info(f"Loading options data from {directory} using Dask")
        
        try:
            # Find all options data files
            files = [f for f in os.listdir(directory) if f.endswith('.csv')]
            if not files:
                raise FileNotFoundError(f"No options data files found in {directory}")
            
            options_data = {}
            
            # Define optimal dtypes
            dtypes = {
                'tr_open': 'float32',
                'tr_high': 'float32',
                'tr_low': 'float32',
                'tr_close': 'float32',
                'strike_price': 'float32',
                'open_interest': 'int32',
                'otype': 'category'
            }
            
            # Process each file
            for file_name in files:
                file_path = os.path.join(directory, file_name)
                logger.info(f"Processing options file with Dask: {file_name}")
                
                # First get the columns from a small sample
                sample = pd.read_csv(file_path, nrows=10)
                essential_cols = self._get_essential_columns(sample.columns)
                
                # Use a larger block size for efficiency but not too large to cause memory issues
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                block_size_mb = max(25, min(100, file_size_mb / 50))
                
                try:
                    # Load with optimized settings
                    ddf = dd.read_csv(file_path, 
                                    usecols=essential_cols,
                                    dtype=dtypes,
                                    blocksize=f"{int(block_size_mb)}MB",
                                    assume_missing=True)
                    
                    # Efficient datetime conversion
                    if {'tr_date', 'tr_time'}.issubset(ddf.columns):
                        ddf['datetime'] = dd.to_datetime(
                            ddf['tr_date'].astype(str) + ' ' + ddf['tr_time'].astype(str)
                        )
                    elif 'tr_datetime' in ddf.columns:
                        ddf['datetime'] = dd.to_datetime(ddf['tr_datetime'])
                    
                    # Apply date filtering if specified - using Dask-optimized approach
                    if self.start_date and self.end_date:
                        start_str = self.start_date.strftime('%Y-%m-%d')
                        end_str = self.end_date.strftime('%Y-%m-%d')
                        ddf = ddf[(ddf['datetime'] >= start_str) & (ddf['datetime'] <= end_str)]
                    
                    # Determine expiry handling strategy
                    if 'week_expiry_date' in ddf.columns:
                        expiry_col = 'week_expiry_date'
                    elif 'expiry_date' in ddf.columns:
                        expiry_col = 'expiry_date'
                    else:
                        # Extract from filename
                        expiry_date = self._extract_expiry_from_filename(file_name)
                        if expiry_date:
                            options_data[str(expiry_date)] = ddf.compute()
                            logger.info(f"Processed {file_name} with expiry {expiry_date}")
                            continue
                        else:
                            logger.warning(f"Could not determine expiry for {file_name}")
                            continue
                    
                    # Group by expiry using Dask's efficient approach
                    # Convert to pandas for final grouping as it's more memory-efficient after filtering
                    df = ddf.compute(scheduler='processes', num_workers=self.num_workers)
                    df[expiry_col] = pd.to_datetime(df[expiry_col])
                    
                    # Group and add to dict by expiry
                    for expiry, group in df.groupby(expiry_col):
                        expiry_key = str(expiry.date())
                        options_data[expiry_key] = group
                        logger.info(f"Added data for expiry {expiry_key}: {len(group)} rows")
                    
                    logger.debug(f"Successfully processed {file_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {e}")
                    continue
            
            if not options_data:
                logger.warning("No options data was successfully loaded")
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error in Dask options data loading: {e}")
            raise
        
    def _get_essential_columns(self, available_columns):
        """Helper to identify essential columns for options data."""
        essential_sets = [
            # Core pricing columns
            ['tr_open', 'tr_high', 'tr_low', 'tr_close'],
            # Option-specific columns
            ['otype', 'strike_price', 'open_interest'],
            # Date/time columns (try multiple formats)
            ['tr_datetime'],
            ['tr_date', 'tr_time'],
            # Expiry columns (try multiple formats)
            ['week_expiry_date'],
            ['expiry_date']
        ]
        
        # Select columns available in the data
        selected_columns = []
        for col_set in essential_sets:
            selected_columns.extend([col for col in col_set if col in available_columns])
        
        return list(set(selected_columns))
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load all market data (spot, futures, options).
        
        Returns:
            Tuple of (spot_data, futures_data, options_data)
        """
        spot_data = self.load_spot_data()
        futures_data = self.load_futures_data()
        options_data = self.load_options_data()
        
        return spot_data, futures_data, options_data
    
    def save_processed_data(self, data: pd.DataFrame, name: str) -> str:
        """
        Save processed data to disk.
        
        Args:
            data: DataFrame to save
            name: Name for the saved file
            
        Returns:
            Path to the saved file
        """
        today = datetime.now().strftime("%Y%m%d")
        file_name = f"{name}_{today}.csv"
        file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
        
        data.to_csv(file_path)
        logger.info(f"Saved processed data to {file_path}")
        
        return file_path

    def load_single_option_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single options data file.
        
        Args:
            file_path: Path to the options data CSV file
            
        Returns:
            DataFrame containing the options data
        """
        logger.info(f"Loading options file: {file_path}")
        
        try:
            # Load the options data
            options_data = pd.read_csv(file_path)
            
            # Convert date and time columns to datetime
            if 'tr_date' in options_data.columns and 'tr_time' in options_data.columns:
                options_data['datetime'] = pd.to_datetime(
                    options_data['tr_date'] + ' ' + options_data['tr_time']
                )
            elif 'tr_datetime' in options_data.columns:
                options_data['datetime'] = pd.to_datetime(options_data['tr_datetime'])
            elif 'datetime' not in options_data.columns:
                raise ValueError("Options data must have either 'datetime', 'tr_datetime', or 'tr_date' and 'tr_time' columns")
            
            # Set datetime as index
            options_data.set_index('datetime', inplace=True)
            
            # Apply date filtering if specified
            if self.start_date and self.end_date:
                options_data = options_data[
                    (options_data.index >= self.start_date) & 
                    (options_data.index <= self.end_date)
                ]
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error loading options file {file_path}: {e}")
            raise

    def process_single_expiry(self, spot_df: pd.DataFrame, futures_df: pd.DataFrame, 
                             options_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Process data for a single options expiry.
        
        Args:
            spot_df: Spot market data
            futures_df: Futures market data
            options_df: Options data for one expiry
            
        Returns:
            Dictionary containing processed results for this expiry
        """
        # Get expiry date from the data
        if 'week_expiry_date' in options_df.columns:
            expiry = str(options_df['week_expiry_date'].iloc[0])
        elif 'expiry_date' in options_df.columns:
            expiry = str(options_df['expiry_date'].iloc[0])
        else:
            expiry = options_df.index.max().strftime('%Y-%m-%d')
        
        # Resample to the specified interval if needed
        options_df = options_df.resample(RESAMPLE_INTERVAL).agg({
            'tr_open': 'first',
            'tr_high': 'max',
            'tr_low': 'min',
            'tr_close': 'last',
            'open_interest': 'last',
            'tr_volume': 'sum' if 'tr_volume' in options_df.columns else None
        }).dropna()
        
        return {expiry: options_df}

    def process_by_expiry(self, spot_data: str, futures_data: str, 
                         options_data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Process options data one expiry at a time.
        
        Args:
            spot_data: Path to spot data file
            futures_data: Path to futures data file
            options_data_dir: Directory containing options data files
            
        Returns:
            Dictionary of processed results, keyed by expiry date
        """
        results = {}
        
        # First load spot and futures data (usually manageable in size)
        spot_df = self.load_spot_data(spot_data)
        futures_df = self.load_futures_data(futures_data)
        
        # Get list of option files
        option_files = [f for f in os.listdir(options_data_dir) if f.endswith('.csv')]
        
        # Process one file at a time
        for option_file in option_files:
            try:
                # Load just this file's data
                current_option_data = self.load_single_option_file(
                    os.path.join(options_data_dir, option_file)
                )
                
                # Skip if no data after filtering
                if current_option_data.empty:
                    logger.warning(f"No data in {option_file} after date filtering")
                    continue
                
                # Process this expiry
                processed_results = self.process_single_expiry(
                    spot_df, futures_df, current_option_data
                )
                
                # Store results
                results.update(processed_results)
                
                # Clear memory
                del current_option_data
                
            except Exception as e:
                logger.error(f"Error processing {option_file}: {e}")
                continue
        
        return results
    
    def load_all_data_efficient(self, spot_file: str, futures_file: str, options_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Efficiently load all market data using parallel processing.
        
        Args:
            spot_file: Path to spot market data CSV file
            futures_file: Path to futures market data CSV file
            options_dir: Directory containing options data files
            
        Returns:
            Tuple of (spot_data, futures_data, options_data) where options_data is a dict
            keyed by expiry dates
            
        Raises:
            ValueError: If data validation fails or required date ranges are not available
            FileNotFoundError: If input files are not found
        """
        logger.info("Loading market data efficiently")
        
        try:
            # Load spot data
            logger.info("Loading spot data")
            spot_data = self._load_csv_efficient(spot_file)
            if spot_data.empty:
                raise ValueError(f"No data found in spot file: {spot_file}")
            spot_range = f"{spot_data.index.min()} to {spot_data.index.max()}"
            logger.info(f"Spot data available range: {spot_range}")

            # Load futures data
            logger.info("Loading futures data")
            futures_data = self._load_csv_efficient(futures_file)
            if futures_data.empty:
                raise ValueError(f"No data found in futures file: {futures_file}")
            futures_range = f"{futures_data.index.min()} to {futures_data.index.max()}"
            logger.info(f"Futures data available range: {futures_range}")

            # Find common date range
            common_start = max(spot_data.index.min(), futures_data.index.min())
            common_end = min(spot_data.index.max(), futures_data.index.max())
            logger.info(f"Common data range across spot and futures: {common_start} to {common_end}")

            # Validate requested date range against common available range
            if self.start_date and self.end_date:
                # Convert start_date to start of day and end_date to end of day
                filter_start = pd.Timestamp(self.start_date).normalize()
                filter_end = pd.Timestamp(self.end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                if filter_start > common_end.normalize() or filter_end < common_start.normalize():
                    raise ValueError(
                        f"Requested date range ({self.start_date} to {self.end_date}) "
                        f"does not overlap with available common data range ({common_start} to {common_end})"
                    )

                # Filter data by date range
                spot_data = spot_data[
                    (spot_data.index >= filter_start) & 
                    (spot_data.index <= filter_end)
                ]
                futures_data = futures_data[
                    (futures_data.index >= filter_start) & 
                    (futures_data.index <= filter_end)
                ]

                # Check if we have data after filtering
                if len(spot_data) == 0 or len(futures_data) == 0:
                    raise ValueError(
                        f"No data available after date filtering. "
                        f"Spot rows: {len(spot_data)}, Futures rows: {len(futures_data)}. "
                        f"Check if your date range ({filter_start} to {filter_end}) "
                        f"is within the available data range ({common_start} to {common_end})."
                    )

            # Load options data
            logger.info("Loading options data")
            options_data = self._load_options_data_parallel(options_dir)
            
            # Filter options data with the same date range logic
            filtered_options = {}
            for expiry, data in options_data.items():
                if self.start_date and self.end_date:
                    data = data[
                        (data.index >= filter_start) & 
                        (data.index <= filter_end)
                    ]
                if len(data) > 0:
                    filtered_options[expiry] = data

            if not filtered_options:
                logger.warning("No options data available for the specified date range")

            # Verify data coverage
            coverage_stats = self.verify_data_coverage(filtered_options)
            if coverage_stats["status"] == "error":
                logger.error(f"Data coverage verification failed: {coverage_stats['message']}")
            else:
                logger.info(f"Data coverage: {coverage_stats.get('backtest_coverage', 'N/A')}")
                if coverage_stats.get('missing_dates'):
                    logger.warning(f"Missing data for {len(coverage_stats['missing_dates'])} dates")

            # Final data quality checks
            self._validate_data_quality(spot_data, futures_data, filtered_options)

            logger.info(
                f"Data loaded successfully: "
                f"spot={len(spot_data)} rows, "
                f"futures={len(futures_data)} rows, "
                f"options={len(filtered_options)} expiries"
            )
            return spot_data, futures_data, filtered_options
            
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            raise

    def verify_data_coverage(self, options_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Verify data coverage across dates to ensure backtest will have data for all days.
        
        Args:
            options_data: Dictionary of options data by expiry
            
        Returns:
            Dictionary with coverage statistics
        """
        if not options_data:
            return {"status": "error", "message": "No options data available"}
        
        # Find date range
        all_datetimes = []
        for df in options_data.values():
            all_datetimes.extend(df.index.tolist())
        
        if not all_datetimes:
            return {"status": "error", "message": "No datetime data found"}
        
        all_datetimes = sorted(all_datetimes)
        start_date = all_datetimes[0].date()
        end_date = all_datetimes[-1].date()
        
        # Count data points per day
        date_counts = {}
        for df in options_data.values():
            dates = df.index.date
            for date in dates:
                if date in date_counts:
                    date_counts[date] += len(df[df.index.date == date])
                else:
                    date_counts[date] = len(df[df.index.date == date])
        
        # Check if we have data for the backtest period
        if self.start_date and self.end_date:
            backtest_start = self.start_date.date()
            backtest_end = self.end_date.date()
            
            backtest_dates = []
            current_date = backtest_start
            while current_date <= backtest_end:
                if current_date.weekday() < 5:  # Only weekdays
                    backtest_dates.append(current_date)
                current_date += timedelta(days=1)
            
            missing_dates = [d for d in backtest_dates if d not in date_counts]
            covered_dates = [d for d in backtest_dates if d in date_counts]
            
            coverage_ratio = len(covered_dates) / len(backtest_dates) if backtest_dates else 0
            
            result = {
                "status": "success",
                "start_date": start_date,
                "end_date": end_date,
                "total_dates": len(date_counts),
                "backtest_coverage": f"{coverage_ratio:.2%}",
                "backtest_dates": len(backtest_dates),
                "covered_dates": len(covered_dates),
                "missing_dates": missing_dates
            }
            
            # Log the information
            logger.info(f"Data coverage: {coverage_ratio:.2%} of backtest period ({len(covered_dates)}/{len(backtest_dates)} days)")
            if missing_dates:
                logger.warning(f"Missing data for {len(missing_dates)} dates in backtest period")
                logger.warning(f"First 5 missing dates: {missing_dates[:5]}")
            
            return result
        
        return {
            "status": "success",
            "start_date": start_date,
            "end_date": end_date,
            "total_dates": len(date_counts),
            "date_counts": date_counts
        }

    def _load_csv_efficient(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file efficiently using chunks or Dask.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
            ValueError: If the data format is invalid
            Exception: For other unexpected errors
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            # Define optimal dtypes for memory efficiency
            dtypes = {
                'tr_open': 'float32',
                'tr_high': 'float32',
                'tr_low': 'float32',
                'tr_close': 'float32',
                'strike_price': 'float32',
                'open_interest': 'int32',
                'otype': 'category'
            }
            
            # Update dtypes with any provided in kwargs
            if 'dtype' in kwargs:
                dtypes.update(kwargs.pop('dtype'))
                
            if self.use_dask:
                logger.info(f"Loading {file_path} using Dask")
                
                # Set optimal Dask partition size based on file size
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                partition_size = max(25, min(128, file_size_mb / 100))
                
                # Read CSV with Dask
                df = dd.read_csv(file_path, 
                               blocksize=f"{int(partition_size)}MB",
                               dtype=dtypes,
                               assume_missing=True,
                               **kwargs)
                
                # Efficient datetime conversion
                if {'tr_date', 'tr_time'}.issubset(df.columns):
                    df['datetime'] = dd.to_datetime(
                        df['tr_date'].astype(str) + ' ' + df['tr_time'].astype(str)
                    )
                elif 'tr_datetime' in df.columns:
                    df['datetime'] = dd.to_datetime(df['tr_datetime'])
                else:
                    raise ValueError("Required datetime columns not found")
                
                # Compute with optimized workers
                num_workers = min(self.num_workers, multiprocessing.cpu_count() - 1)
                df = df.compute(scheduler='processes', num_workers=num_workers)
                
            else:
                logger.info(f"Loading {file_path} using pandas chunks")
                chunks = []
                total_rows = 0
                
                # Read file in chunks
                chunk_iterator = pd.read_csv(file_path, 
                                           chunksize=self.chunksize,
                                           dtype=dtypes,
                                           **kwargs)
                
                for chunk_num, chunk in enumerate(chunk_iterator, 1):
                    # Efficient datetime conversion
                    if {'tr_date', 'tr_time'}.issubset(chunk.columns):
                        chunk['datetime'] = pd.to_datetime(
                            chunk['tr_date'] + ' ' + chunk['tr_time'],
                            format='%Y-%m-%d %H:%M:%S',  # Specify format for better performance
                            cache=True
                        )
                    elif 'tr_datetime' in chunk.columns:
                        chunk['datetime'] = pd.to_datetime(
                            chunk['tr_datetime'],
                            cache=True
                        )
                    else:
                        raise ValueError("Required datetime columns not found")
                    
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    
                    # Log progress and cleanup memory periodically
                    if chunk_num % 10 == 0:
                        logger.debug(f"Processed {total_rows:,} rows from {file_path}")
                        self._cleanup_memory()
                
                if not chunks:
                    raise pd.errors.EmptyDataError(f"No data loaded from {file_path}")
                
                # Combine chunks efficiently
                df = pd.concat(chunks, ignore_index=True, copy=False)
                
            # Set datetime as index and sort
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Verify we have data
            if df.empty:
                raise ValueError(f"No data loaded from {file_path} after processing")
                
            # Log memory usage after loading
            self._log_memory_usage(f"After loading {os.path.basename(file_path)}")
            
            return df
            
        except pd.errors.EmptyDataError:
            logger.error(f"File is empty: {file_path}")
            raise
        except ValueError as ve:
            logger.error(f"Invalid data format in {file_path}: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {str(e)}")
            raise

    def _load_options_data_parallel(self, options_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load options data files in parallel with memory optimization.
        """
        options_files = [f for f in os.listdir(options_dir) if f.endswith('.csv')]
        
        # Process files in smaller batches to manage memory
        batch_size = max(1, len(options_files) // (self.num_workers * 2))
        options_data = {}
        
        for i in range(0, len(options_files), batch_size):
            batch_files = options_files[i:i + batch_size]
            
            with multiprocessing.Pool(self.num_workers) as pool:
                batch_results = pool.map(
                    self._load_single_options_file,
                    [os.path.join(options_dir, file_name) for file_name in batch_files]
                )
                
                # Add valid results to options_data
                for expiry, df in batch_results:
                    if expiry is not None and df is not None:
                        options_data[expiry] = df
                
                # Clean up after each batch
                self._cleanup_memory()
        
        return options_data

    def _load_single_options_file(self, file_path: str) -> Tuple[str, pd.DataFrame]:
        """
        Load a single options file and return its expiry date and data.
        
        Args:
            file_path: Path to the options CSV file
        
        Returns:
            Tuple of (expiry_str, DataFrame)
        """
        try:
            # Define columns based on file type
            is_futures = 'futures' in file_path.lower()
            expiry_col = 'expiry_date' if is_futures else 'week_expiry_date'
            
            # Enhanced essential columns list with tr_volume included
            essential_cols = ['tr_datetime', 'tr_date', 'tr_time', 'tr_open', 'tr_high', 
                             'tr_low', 'tr_close', 'otype', 'strike_price', 'open_interest',
                             'tr_volume', expiry_col, 'stock_name']
            
            # Additional potentially useful columns
            additional_cols = ['delta', 'gamma', 'vega', 'theta', 'rho', 'implied_volatility']
            
            # Read first chunk to get metadata
            first_chunk = pd.read_csv(file_path, nrows=1)
            
            # Check which columns are actually available in the file
            available_cols = [col for col in essential_cols if col in first_chunk.columns]
            
            # Add any additional columns that exist
            for col in additional_cols:
                if col in first_chunk.columns:
                    available_cols.append(col)
            
            # Define dtypes for optimization
            dtype_dict = {
                'tr_open': 'float32',
                'tr_high': 'float32',
                'tr_low': 'float32',
                'tr_close': 'float32',
                'strike_price': 'float32',
                'open_interest': 'int32',
                'tr_volume': 'int32',
                'otype': 'category',
                'stock_name': 'category'
            }
            
            # Process file in chunks
            chunks = []
            expiry = None
            
            for chunk in pd.read_csv(file_path, usecols=available_cols, 
                                   dtype=dtype_dict, chunksize=self.chunksize):
                
                # Convert dates
                if 'tr_datetime' in chunk.columns:
                    chunk.index = pd.to_datetime(chunk['tr_datetime'])
                else:
                    chunk.index = pd.to_datetime(chunk['tr_date'] + ' ' + chunk['tr_time'])
                
                # Get expiry if not already found
                if expiry is None:
                    data_date = chunk.index.min().date()
                    if expiry_col in chunk.columns:
                        parsed_expiry = pd.to_datetime(chunk[expiry_col].iloc[0]).date()
                        # Use the utility function to validate expiry
                        valid_expiry = validate_expiry_date(
                            parsed_expiry, 
                            data_date, 
                            f"File: {os.path.basename(file_path)}"
                        )
                        expiry = str(valid_expiry)
                    else:
                        # Infer expiry from data range using utility
                        expiry = str(get_next_expiry_date(chunk.index.max()))
                
                # Filter by date range if specified
                if self.start_date and chunk.index.max() < self.start_date:
                    continue
                if self.end_date and chunk.index.min() > self.end_date:
                    continue
                
                chunks.append(chunk)
                
                # Log memory usage periodically
                if len(chunks) % 10 == 0:
                    self._log_memory_usage(f"Processing {os.path.basename(file_path)} - chunk {len(chunks)}")
                    
                # Force garbage collection
                if len(chunks) % 20 == 0:
                    self._cleanup_memory()
            
            # If no valid chunks found
            if not chunks:
                return None, None
            
            # Combine chunks efficiently
            df = pd.concat(chunks, axis=0, copy=False)
            df = df.sort_index()
            
            # Add expiry date column with the correct name
            df[expiry_col] = pd.to_datetime(expiry)
            
            return expiry, df
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None, None

    def _validate_data_quality(self, spot_data: pd.DataFrame, futures_data: pd.DataFrame, 
                             options_data: Dict[str, pd.DataFrame]) -> None:
        """
        Validate the quality of loaded data.
        
        Args:
            spot_data: Spot market data
            futures_data: Futures market data
            options_data: Dictionary of options data by expiry
            
        Raises:
            ValueError: If data quality checks fail
        """
        # Check for required columns
        spot_cols = ['tr_open', 'tr_high', 'tr_low', 'tr_close']
        futures_cols = spot_cols + ['open_interest']
        options_cols = futures_cols + ['strike_price', 'otype']
        
        missing_spot = [col for col in spot_cols if col not in spot_data.columns]
        if missing_spot:
            raise ValueError(f"Missing required columns in spot data: {missing_spot}")
        
        missing_futures = [col for col in futures_cols if col not in futures_data.columns]
        if missing_futures:
            raise ValueError(f"Missing required columns in futures data: {missing_futures}")
        
        # Check for NaN values
        spot_nulls = spot_data[spot_cols].isnull().sum()
        if spot_nulls.any():
            logger.warning(f"Found NaN values in spot data:\n{spot_nulls[spot_nulls > 0]}")
        
        futures_nulls = futures_data[futures_cols].isnull().sum()
        if futures_nulls.any():
            logger.warning(f"Found NaN values in futures data:\n{futures_nulls[futures_nulls > 0]}")
        
        # Check options data quality
        for expiry, data in options_data.items():
            missing_options = [col for col in options_cols if col not in data.columns]
            if missing_options:
                raise ValueError(f"Missing required columns in options data for {expiry}: {missing_options}")
            
            options_nulls = data[options_cols].isnull().sum()
            if options_nulls.any():
                logger.warning(f"Found NaN values in options data for {expiry}:\n{options_nulls[options_nulls > 0]}")

    def load_options_data_sampled(self, directory: Optional[str] = None, 
                                sample_rate: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Load options data with intelligent sampling to reduce processing time while preserving
        important market periods.
        
        Args:
            directory: Directory containing options data CSV files. If None, uses OPTIONS_DATA_DIR
            sample_rate: Fraction of data to load (0.0-1.0). Default is 1.0 (all data)
            
        Returns:
            Dictionary of DataFrames with options data by expiry
            
        Raises:
            ValueError: If sample_rate is invalid or no data is found
            FileNotFoundError: If directory doesn't exist
        """
        if directory is None:
            directory = OPTIONS_DATA_DIR
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Options data directory not found: {directory}")
        
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"Sample rate must be between 0.0 and 1.0, got {sample_rate}")
        
        logger.info(f"Loading options data from {directory} using sampling rate {sample_rate}")
        
        options_data = {}
        files_processed = 0
        total_rows = 0
        
        try:
            csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {directory}")
            
            for file_name in csv_files:
                file_path = os.path.join(directory, file_name)
                logger.debug(f"Processing file: {file_name}")
                
                try:
                    # Load data in chunks for memory efficiency
                    chunks = []
                    for chunk in pd.read_csv(file_path, chunksize=self.chunksize):
                        if sample_rate < 1.0:
                            chunk = self._smart_sample_chunk(chunk, sample_rate)
                        chunks.append(chunk)
                        
                        # Periodic memory cleanup
                        if len(chunks) % 10 == 0:
                            self._cleanup_memory()
                    
                    if not chunks:
                        logger.warning(f"No data loaded from {file_name}")
                        continue
                        
                    # Combine chunks and process
                    df = pd.concat(chunks, axis=0)
                    
                    # Convert datetime and set index
                    if 'tr_datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['tr_datetime'])
                    elif 'tr_date' in df.columns and 'tr_time' in df.columns:
                        df['datetime'] = pd.to_datetime(df['tr_date'] + ' ' + df['tr_time'])
                    else:
                        raise ValueError(f"No datetime columns found in {file_name}")
                        
                    df.set_index('datetime', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Extract expiry date
                    if 'expiry_date' in df.columns:
                        expiry = str(pd.to_datetime(df['expiry_date'].iloc[0]).date())
                    else:
                        # Infer from filename or data range
                        expiry = str(df.index.max().date())
                    
                    # Store in dictionary
                    if expiry in options_data:
                        options_data[expiry] = pd.concat([options_data[expiry], df])
                    else:
                        options_data[expiry] = df
                    
                    files_processed += 1
                    total_rows += len(df)
                    
                    logger.debug(f"Processed {file_name}: {len(df)} rows for expiry {expiry}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {str(e)}")
                    continue
                
                # Cleanup after each file
                self._cleanup_memory()
            
            # Final processing of combined data
            for expiry in options_data:
                df = options_data[expiry]
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)
                options_data[expiry] = df
            
            logger.info(
                f"Completed loading sampled options data: "
                f"{files_processed} files processed, "
                f"{len(options_data)} expiries, "
                f"{total_rows:,} total rows"
            )
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error loading sampled options data: {str(e)}")
            raise

    def _smart_sample_chunk(self, chunk: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
        """
        Intelligently sample a chunk of options data preserving important market periods.
        
        Args:
            chunk: DataFrame chunk to sample
            sample_rate: Sampling rate (0.0-1.0)
            
        Returns:
            Sampled DataFrame
        """
        try:
            # Extract hour information
            if 'tr_datetime' in chunk.columns:
                chunk['hour'] = pd.to_datetime(chunk['tr_datetime']).dt.hour
            elif 'tr_time' in chunk.columns:
                chunk['hour'] = pd.to_datetime(chunk['tr_time']).dt.hour
            else:
                return chunk.sample(frac=sample_rate)  # Fallback to random sampling
            
            # Define important market periods
            important_mask = (
                ((chunk['hour'] >= 9) & (chunk['hour'] <= 10)) |   # Market open
                ((chunk['hour'] >= 15) & (chunk['hour'] <= 16)) |  # Market close
                ((chunk['hour'] >= 12) & (chunk['hour'] <= 13))    # Mid-day
            )
            
            # Split into important and other periods
            important_rows = chunk[important_mask]
            other_rows = chunk[~important_mask]
            
            # Sample other periods
            if len(other_rows) > 0:
                step = max(1, int(1/sample_rate))
                sampled_other = other_rows.iloc[::step]
            else:
                sampled_other = pd.DataFrame(columns=chunk.columns)
            
            # Combine and sort
            result = pd.concat([important_rows, sampled_other])
            result.sort_index(inplace=True)
            
            # Drop temporary column
            result.drop('hour', axis=1, inplace=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in smart sampling: {str(e)}")
            # Fallback to simple sampling
            return chunk.sample(frac=sample_rate)
