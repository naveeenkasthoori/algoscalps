"""
Parallel processing utilities for calculating technical indicators and options Greeks.
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import dask.dataframe as dd

from config.settings import LOG_DIR
from src.indicators.technical import calculate_all_technical_indicators
from src.indicators.greeks import GreeksCalculator
from src.indicators.sentiment import calculate_options_sentiment, calculate_pcr, analyze_oi_concentration, calculate_oi_change_velocity

# Configure file handler for indicator processor logs
indicator_logger = logging.getLogger("indicator_processor")
log_file = os.path.join(LOG_DIR, 'indicators.log')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
indicator_logger.addHandler(file_handler)
indicator_logger.setLevel(logging.INFO)

class ParallelIndicatorProcessor:
    """
    Process technical indicators and options Greeks in parallel.
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize the parallel indicator processor.
        
        Args:
            num_workers: Number of worker processes for parallel processing (default: CPU count)
        """
        self.num_workers = num_workers if num_workers else max(1, multiprocessing.cpu_count() - 1)
        self.greeks_calculator = GreeksCalculator()
        
        indicator_logger.info(f"Parallel Indicator Processor initialized with {self.num_workers} workers")
        
        # Create required directories if they don't exist
        os.makedirs(LOG_DIR, exist_ok=True)
    
    def _log_memory_usage(self, label: str = ""):
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        indicator_logger.info(f"Memory usage ({label}): {memory_usage_mb:.2f} MB")
    
    def _calculate_technical_indicators_chunk(self, futures_chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a chunk of futures data.
        
        Args:
            futures_chunk: DataFrame chunk to process
            
        Returns:
            DataFrame with calculated indicators
        """
        try:
            return calculate_all_technical_indicators(futures_chunk)
        except Exception as e:
            indicator_logger.error(f"Error calculating technical indicators for chunk: {e}")
            return futures_chunk
    
    def calculate_cvd_partition(self, df: Union[pd.DataFrame, dd.DataFrame]) -> pd.DataFrame:
        """
        Calculate Cumulative Volume Delta for a partition of data.
        
        Args:
            df: DataFrame partition to process
            
        Returns:
            DataFrame with CVD calculated for the partition
        """
        try:
            # Calculate volume delta
            df['volume_delta'] = df['volume'] * df['price_direction']
            
            # Calculate cumulative sum for this partition
            df['cvd'] = df['volume_delta'].cumsum()
            
            return df
        except Exception as e:
            indicator_logger.error(f"Error calculating CVD partition: {e}")
            return df
    
    def process_technical_indicators(self, futures_data: pd.DataFrame, use_dask: bool = False, chunk_size: int = 5000) -> pd.DataFrame:
        """
        Process technical indicators for futures data in parallel, optionally using Dask.
        
        Args:
            futures_data: DataFrame with futures market data
            use_dask: Whether to use Dask for processing large datasets
            chunk_size: Size of chunks for parallel processing when not using Dask
            
        Returns:
            DataFrame with calculated indicators
        """
        indicator_logger.info("Processing technical indicators")
        self._log_memory_usage("Before technical indicators")
        
        try:
            # Check if data is too small to benefit from Dask
            if len(futures_data) <= 10000 or not use_dask:
                indicator_logger.info("Using conventional parallel processing")
                return self._process_technical_indicators_parallel(futures_data, chunk_size)
            
            else:
                indicator_logger.info("Using Dask for technical indicator calculation")
                
                # Convert to Dask DataFrame
                dask_futures = dd.from_pandas(futures_data, npartitions=self.num_workers)
                
                # Calculate indicators that can be parallelized easily
                dask_futures = dask_futures.map_partitions(
                    lambda df: self.calculate_cvd_partition(df), meta=futures_data
                )
                
                # Compute results
                result = dask_futures.compute(scheduler='processes')
                
                # Apply final calculations that need the complete series
                result = calculate_all_technical_indicators(result)
                
                indicator_logger.info(f"Technical indicators calculation complete: {len(result)} rows")
                self._log_memory_usage("After technical indicators")
                
                return result
            
        except Exception as e:
            indicator_logger.error(f"Error in parallel technical indicators processing: {e}", exc_info=True)
            return futures_data
    
    def _process_technical_indicators_parallel(self, futures_data: pd.DataFrame, chunk_size: int = 5000) -> pd.DataFrame:
        """
        Process technical indicators using conventional parallel processing.
        
        Args:
            futures_data: DataFrame with futures market data
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            DataFrame with calculated indicators
        """
        # Check if data is too small to chunk
        if len(futures_data) <= chunk_size:
            indicator_logger.info("Data size is small, processing without chunking")
            result = calculate_all_technical_indicators(futures_data)
            indicator_logger.info("Technical indicators calculation complete")
            self._log_memory_usage("After technical indicators")
            return result
        
        # Split data into chunks based on index
        all_indices = futures_data.index.unique()
        chunks = []
        
        for i in range(0, len(all_indices), chunk_size):
            chunk_indices = all_indices[i:i+chunk_size]
            chunks.append(futures_data.loc[chunk_indices])
        
        indicator_logger.info(f"Split data into {len(chunks)} chunks for parallel processing")
        
        # Process chunks in parallel
        processed_chunks = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for chunk in chunks:
                futures.append(executor.submit(self._calculate_technical_indicators_chunk, chunk))
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    processed_chunk = future.result()
                    processed_chunks.append(processed_chunk)
                    if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                        indicator_logger.info(f"Processed {i + 1}/{len(futures)} chunks")
                except Exception as e:
                    indicator_logger.error(f"Error processing chunk: {e}")
        
        # Combine processed chunks
        if not processed_chunks:
            indicator_logger.warning("No chunks were successfully processed")
            return futures_data
        
        result = pd.concat(processed_chunks)
        
        # Sort by index
        result = result.sort_index()
        
        indicator_logger.info(f"Technical indicators calculation complete: {len(result)} rows")
        self._log_memory_usage("After technical indicators")
        
        return result
    
    def _determine_chunk_size(self, total_rows: int, default_size: int = 50000) -> int:
        """
        Determine optimal chunk size based on data size and available memory.
        
        Args:
            total_rows: Total number of rows to process
            default_size: Default chunk size
            
        Returns:
            Optimal chunk size
        """
        try:
            # Get available memory in MB
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            
            # Use a simple heuristic: try to use at most 20% of available memory per chunk
            # Assume each row needs about 1KB of memory (this is a rough estimate)
            suggested_chunk_size = int(available_memory_mb * 0.2 * 1024)
            
            # Ensure chunk size is reasonable
            chunk_size = min(max(5000, suggested_chunk_size), default_size)
            
            # Ensure we have at least 10 chunks for large datasets to prevent memory spikes
            if total_rows > 1000000:
                chunk_size = min(chunk_size, total_rows // 10)
            
            indicator_logger.info(f"Determined chunk size of {chunk_size} rows for {total_rows} total rows")
            return chunk_size
        except:
            # Fall back to default if we can't determine memory
            return default_size
    
    def _process_greeks_for_expiry(self, expiry: str, options_df: pd.DataFrame, spot_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process options Greeks for a specific expiry.
        Handle large datasets by processing in chunks.
        
        Args:
            expiry: Expiry date string
            options_df: DataFrame with options data for this expiry
            spot_data: DataFrame with spot market data
            
        Returns:
            DataFrame with calculated Greeks
        """
        try:
            indicator_logger.info(f"Processing Greeks for expiry {expiry}: {len(options_df)} rows")
            
            # Determine optimal chunk size based on available memory and data size
            chunk_size = self._determine_chunk_size(len(options_df), 50000)
            
            # Get latest spot price
            latest_spot = spot_data['tr_close'].iloc[-1]
            
            # Process options chain with chunking
            processed_df = self.greeks_calculator.process_options_chain(
                options_df, latest_spot, expiry_date=pd.to_datetime(expiry), chunk_size=chunk_size
            )
            
            indicator_logger.info(f"Completed Greeks calculation for expiry {expiry}")
            return processed_df
            
        except Exception as e:
            indicator_logger.error(f"Error processing Greeks for expiry {expiry}: {e}")
            return options_df
    
    def process_options_greeks(self, options_data: Dict[str, pd.DataFrame], spot_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Process options Greeks for all expiry dates in parallel.
        
        Args:
            options_data: Dictionary of DataFrames with options data by expiry
            spot_data: DataFrame with spot market data
            
        Returns:
            Dictionary of DataFrames with calculated Greeks
        """
        indicator_logger.info("Processing options Greeks in parallel")
        self._log_memory_usage("Before options Greeks")
        
        try:
            if not options_data:
                indicator_logger.warning("No options data to process")
                return {}
            
            # Validate and handle missing option type column
            for expiry, df in options_data.items():
                if 'option_type' not in df.columns and 'otype' not in df.columns:
                    indicator_logger.warning(f"Missing option type column for expiry {expiry}, attempting to infer")
                    
                    if 'strike_price' in df.columns:
                        # Infer based on strikes relative to spot price
                        latest_spot = spot_data['tr_close'].iloc[-1]
                        median_strike = df['strike_price'].median()
                        
                        # Use spot price and median strike as reference points
                        df['option_type'] = np.where(
                            df['strike_price'] >= median_strike,
                            'CE',  # Calls typically above median
                            'PE'   # Puts typically below median
                        )
                        indicator_logger.info(f"Created synthetic option types for {expiry} based on strikes")
                        options_data[expiry] = df
                    else:
                        indicator_logger.error(f"Cannot infer option types for {expiry} - missing strike_price column")
                elif 'otype' in df.columns and 'option_type' not in df.columns:
                    # Standardize column name if needed
                    df['option_type'] = df['otype']
                    options_data[expiry] = df
            
            processed_data = {}
            
            # Process each expiry in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}
                
                for expiry, options_df in options_data.items():
                    futures[expiry] = executor.submit(
                        self._process_greeks_for_expiry, expiry, options_df, spot_data
                    )
                
                for expiry, future in futures.items():
                    try:
                        processed_data[expiry] = future.result()
                        indicator_logger.info(f"Completed processing for expiry {expiry}")
                    except Exception as e:
                        indicator_logger.error(f"Error getting result for expiry {expiry}: {e}")
                        processed_data[expiry] = options_data[expiry]  # Use original data as fallback
            
            indicator_logger.info(f"Options Greeks calculation complete for {len(processed_data)} expiry dates")
            self._log_memory_usage("After options Greeks")
            
            return processed_data
            
        except Exception as e:
            indicator_logger.error(f"Error in parallel options Greeks processing: {e}", exc_info=True)
            return options_data
    
    def process_options_sentiment(self, options_data_with_greeks: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Union[pd.DataFrame, dict]]]:
        """
        Process options sentiment indicators using parallel processing.
        Handle large datasets by processing in chunks.
        Enhanced with column verification and preservation.
        
        Args:
            options_data_with_greeks: Dictionary of DataFrames with options data and Greeks
            
        Returns:
            Dictionary with sentiment indicators containing:
                - pcr: DataFrame with put-call ratios
                - oi_concentration: Dictionary with open interest concentration metrics
                - oi_velocity: Dictionary with OI velocity metrics
        """
        indicator_logger.info("Processing options sentiment in parallel")
        self._log_memory_usage("Before options sentiment")
        
        try:
            if not options_data_with_greeks:
                indicator_logger.warning("No options data with Greeks to process")
                return {}
            
            # Initialize results dictionary
            sentiment: Dict[str, Union[pd.DataFrame, dict]] = {}
            
            # Verify required columns are present in all dataframes
            required_columns = ['tr_volume', 'option_type', 'open_interest']
            missing_columns = {}
            for expiry, df in options_data_with_greeks.items():
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    missing_columns[expiry] = missing
                    indicator_logger.warning(f"Missing columns {missing} in expiry {expiry}")
            
            if missing_columns:
                indicator_logger.error(f"Required columns missing in some expiries: {missing_columns}")
            
            # Calculate PCR directly in the main process
            try:
                indicator_logger.info("Starting PCR calculation in main process")
                pcr_data = calculate_pcr(options_data_with_greeks)
                sentiment['pcr'] = pcr_data
                indicator_logger.info(f"PCR calculation complete - generated {len(pcr_data)} records")
            except Exception as e:
                indicator_logger.error(f"Error calculating PCR: {e}", exc_info=True)
                sentiment['pcr'] = pd.DataFrame()
            
            # Rest of the code remains unchanged
            total_rows = sum(len(df) for df in options_data_with_greeks.values())
            chunk_size = self._determine_chunk_size(total_rows, 50000)
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                oi_concentration_future = executor.submit(
                    analyze_oi_concentration, options_data_with_greeks, chunk_size
                )
                
                oi_velocity_future = executor.submit(
                    calculate_oi_change_velocity, options_data_with_greeks, 5, chunk_size
                )
                
                try:
                    sentiment['oi_concentration'] = oi_concentration_future.result()
                    indicator_logger.info("OI concentration analysis complete")
                except Exception as e:
                    indicator_logger.error(f"Error analyzing OI concentration: {e}")
                    sentiment['oi_concentration'] = {}
                
                try:
                    sentiment['oi_velocity'] = oi_velocity_future.result()
                    indicator_logger.info("OI change velocity calculation complete")
                except Exception as e:
                    indicator_logger.error(f"Error calculating OI velocity: {e}")
                    sentiment['oi_velocity'] = {}
            
            indicator_logger.info("Options sentiment calculations complete")
            self._log_memory_usage("After options sentiment")
            
            return sentiment
            
        except Exception as e:
            indicator_logger.error(f"Error in parallel options sentiment processing: {e}", exc_info=True)
            return {}
    
    def process_all_indicators(self, spot_data: pd.DataFrame, futures_data: pd.DataFrame, 
                             options_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict[str, Union[pd.DataFrame, dict]]]]:
        """
        Process all indicators in a coordinated parallel workflow.
        
        Args:
            spot_data: DataFrame with spot market data
            futures_data: DataFrame with futures market data
            options_data: Dictionary of DataFrames with options data by expiry
            
        Returns:
            Tuple of (futures_data_with_indicators, options_data_with_greeks, options_sentiment)
        """
        indicator_logger.info("Processing all indicators in efficient parallel workflow")
        
        # Step 1: Process technical indicators for futures data
        futures_data_with_indicators = self.process_technical_indicators(futures_data)
        
        # Step 2: Process options Greeks
        options_data_with_greeks = self.process_options_greeks(options_data, spot_data)
        
        # Step 3: Process options sentiment
        options_sentiment = self.process_options_sentiment(options_data_with_greeks)
        
        indicator_logger.info("All indicators processed successfully")
        
        return futures_data_with_indicators, options_data_with_greeks, options_sentiment