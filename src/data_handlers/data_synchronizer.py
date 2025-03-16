"""
Data synchronization module for aligning options, futures, and spot data.
This ensures that data used in backtests has consistent date ranges and valid expiry dates.
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from src.utils.date_utils import validate_expiry_date

logger = logging.getLogger(__name__)

class DataSynchronizer:
    """
    Synchronizes different market data sources to ensure consistent date ranges and valid options data.
    """
    
    def __init__(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Initialize the data synchronizer.
        
        Args:
            start_date: Start date for filtering data (format: YYYY-MM-DD)
            end_date: End date for filtering data (format: YYYY-MM-DD)
        """
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        logger.info(f"Data Synchronizer initialized with date range: {start_date} to {end_date}")
    
    def validate_data_sources(self, 
                             spot_data: pd.DataFrame, 
                             futures_data: pd.DataFrame,
                             options_data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
        """
        Validate data sources for date range consistency and validity.
        
        Args:
            spot_data: Spot market data
            futures_data: Futures market data
            options_data: Dictionary of options data by expiry
            
        Returns:
            Tuple of (valid: bool, message: str)
        """
        if spot_data.empty or futures_data.empty or not options_data:
            return False, "One or more data sources are empty"
        
        # Find common date range
        common_start = max(spot_data.index.min(), futures_data.index.min())
        common_end = min(spot_data.index.max(), futures_data.index.max())
        
        logger.info(f"Common date range across spot and futures: {common_start} to {common_end}")
        
        # Check if there's enough overlap
        if common_start >= common_end:
            return False, f"No overlap between spot ({spot_data.index.min()} to {spot_data.index.max()}) and futures ({futures_data.index.min()} to {futures_data.index.max()}) data"
        
        # Validate options expiry dates
        valid_expirations = []
        for expiry, expiry_data in options_data.items():
            try:
                expiry_date = pd.to_datetime(expiry)
                
                # Check if expiry is in the future relative to the data we're using
                if expiry_date >= common_start:
                    valid_expirations.append(expiry)
                else:
                    logger.warning(f"Expiry {expiry} is in the past relative to data start date {common_start}")
            except:
                logger.warning(f"Could not parse expiry date from {expiry}")
        
        if not valid_expirations:
            return False, f"No valid option expiry dates found for time period {common_start} to {common_end}"
        
        logger.info(f"Found {len(valid_expirations)} valid option expiry dates: {valid_expirations}")
        
        return True, f"Data is valid with common range {common_start} to {common_end}"
    
    def synchronize_data(self, 
                        spot_data: pd.DataFrame, 
                        futures_data: pd.DataFrame,
                        options_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Synchronize data sources to ensure consistent date ranges and valid options.
        
        Args:
            spot_data: Spot market data
            futures_data: Futures market data
            options_data: Dictionary of options data by expiry
            
        Returns:
            Tuple of (aligned_spot, aligned_futures, filtered_options)
        """
        logger.info("Synchronizing data sources")
        
        # Find common date range
        common_start = max(spot_data.index.min(), futures_data.index.min())
        common_end = min(spot_data.index.max(), futures_data.index.max())
        
        # Adjust range based on user-specified dates
        if self.start_date and self.start_date > common_start:
            common_start = self.start_date
        if self.end_date and self.end_date < common_end:
            common_end = self.end_date
        
        logger.info(f"Using date range: {common_start} to {common_end}")
        
        # Align spot and futures data to common range
        aligned_spot = spot_data.loc[(spot_data.index >= common_start) & (spot_data.index <= common_end)]
        aligned_futures = futures_data.loc[(futures_data.index >= common_start) & (futures_data.index <= common_end)]
        
        # Filter options data by expiry date and time range
        filtered_options = {}
        for expiry, expiry_data in options_data.items():
            try:
                expiry_date = pd.to_datetime(expiry)
                
                # Only include options expiring after the start of our data range
                if expiry_date >= common_start:
                    # Filter by date range
                    filtered_data = expiry_data.loc[(expiry_data.index >= common_start) & (expiry_data.index <= common_end)]
                    
                    if not filtered_data.empty:
                        filtered_options[expiry] = filtered_data
                        logger.info(f"Including options for expiry {expiry} with {len(filtered_data)} rows")
                else:
                    logger.warning(f"Excluding options with expiry {expiry} (before data start date)")
            except:
                logger.warning(f"Skipping options with unparseable expiry {expiry}")
        
        if not filtered_options:
            logger.warning("No valid options data after synchronization!")
        
        logger.info(f"Data synchronization complete: spot={len(aligned_spot)}, futures={len(aligned_futures)}, options={len(filtered_options)} expiries")
        
        return aligned_spot, aligned_futures, filtered_options
    
    def find_matching_data_files(self, 
                                spot_dir: str, 
                                futures_dir: str, 
                                options_dir: str,
                                target_date_range: Optional[Tuple[str, str]] = None) -> Tuple[str, str, List[str]]:
        """
        Find data files with matching date ranges.
        
        Args:
            spot_dir: Directory containing spot data CSV files
            futures_dir: Directory containing futures data CSV files
            options_dir: Directory containing options data CSV files
            target_date_range: Optional target date range (start_date, end_date)
            
        Returns:
            Tuple of (spot_file, futures_file, options_files)
        """
        logger.info("Finding matching data files")
        
        # Target date range from init if not specified
        if target_date_range is None and (self.start_date is not None or self.end_date is not None):
            start_str = self.start_date.strftime('%Y-%m-%d') if self.start_date else None
            end_str = self.end_date.strftime('%Y-%m-%d') if self.end_date else None
            target_date_range = (start_str, end_str)
        
        # Scan directories for files
        spot_files = [f for f in os.listdir(spot_dir) if f.endswith('.csv')]
        futures_files = [f for f in os.listdir(futures_dir) if f.endswith('.csv')]
        options_files = [f for f in os.listdir(options_dir) if f.endswith('.csv')]
        
        if not spot_files or not futures_files or not options_files:
            raise ValueError("One or more data directories are empty")
        
        # Determine suitable files based on date range
        selected_spot = spot_files[0]  # Default to first file
        selected_futures = futures_files[0]  # Default to first file
        selected_options = options_files  # Default to all files
        
        # Advanced selection based on target date range (if provided)
        if target_date_range and target_date_range[0]:
            # Here you would implement logic to select files matching the target date range
            # This requires reading file headers or metadata to determine contained dates
            # For this example, we'll stay with the default selection
            logger.info(f"Would filter files by date range {target_date_range}")
        
        # Build full paths
        spot_path = os.path.join(spot_dir, selected_spot)
        futures_path = os.path.join(futures_dir, selected_futures)
        options_paths = [os.path.join(options_dir, f) for f in selected_options]
        
        logger.info(f"Selected data files: spot={selected_spot}, futures={selected_futures}, options={len(selected_options)} files")
        
        return spot_path, futures_path, options_paths
    
    def detect_data_errors(self, 
                          spot_data: pd.DataFrame, 
                          futures_data: pd.DataFrame,
                          options_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Detect common data errors that could impact backtest results.
        
        Args:
            spot_data: Spot market data
            futures_data: Futures market data
            options_data: Dictionary of options data by expiry
            
        Returns:
            Dictionary of errors by category
        """
        errors = {
            "date_range": [],
            "missing_data": [],
            "expired_options": [],
            "inconsistent_data": []
        }
        
        # Check date mismatches
        if self.start_date:
            if spot_data.index.min() > self.start_date:
                errors["date_range"].append(f"Spot data starts at {spot_data.index.min()}, later than requested start {self.start_date}")
            if futures_data.index.min() > self.start_date:
                errors["date_range"].append(f"Futures data starts at {futures_data.index.min()}, later than requested start {self.start_date}")
        
        if self.end_date:
            if spot_data.index.max() < self.end_date:
                errors["date_range"].append(f"Spot data ends at {spot_data.index.max()}, earlier than requested end {self.end_date}")
            if futures_data.index.max() < self.end_date:
                errors["date_range"].append(f"Futures data ends at {futures_data.index.max()}, earlier than requested end {self.end_date}")
        
        # Check for missing data fields
        required_spot_cols = ['tr_open', 'tr_high', 'tr_low', 'tr_close']
        missing_spot_cols = [col for col in required_spot_cols if col not in spot_data.columns]
        if missing_spot_cols:
            errors["missing_data"].append(f"Spot data missing columns: {missing_spot_cols}")
        
        required_futures_cols = ['tr_open', 'tr_high', 'tr_low', 'tr_close', 'open_interest']
        missing_futures_cols = [col for col in required_futures_cols if col not in futures_data.columns]
        if missing_futures_cols:
            errors["missing_data"].append(f"Futures data missing columns: {missing_futures_cols}")
        
        # Check for expired options
        for expiry, expiry_data in options_data.items():
            try:
                expiry_date = pd.to_datetime(expiry)
                data_start = min(spot_data.index.min(), futures_data.index.min())
                
                if expiry_date < data_start:
                    errors["expired_options"].append(f"Expiry {expiry} is before data start date {data_start}")
            except:
                errors["inconsistent_data"].append(f"Could not parse expiry date from {expiry}")
        
        # Check for specific data inconsistencies
        if not options_data:
            errors["missing_data"].append("No options data available")
        else:
            for expiry, expiry_data in options_data.items():
                required_option_cols = ['tr_open', 'tr_high', 'tr_low', 'tr_close', 'otype', 'strike_price', 'open_interest']
                missing_option_cols = [col for col in required_option_cols if col not in expiry_data.columns]
                if missing_option_cols:
                    errors["missing_data"].append(f"Options data for {expiry} missing columns: {missing_option_cols}")
        
        # Remove empty error categories
        errors = {k: v for k, v in errors.items() if v}
        
        if errors:
            logger.warning(f"Found {sum(len(v) for v in errors.values())} potential data errors")
        else:
            logger.info("No data errors detected")
        
        return errors
    
    def align_option_expirations(self, options_data: Dict[str, pd.DataFrame], target_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Filter options to only include those with valid expiration dates relative to the target date.
        
        Args:
            options_data: Dictionary of options data by expiry
            target_date: Target date for backtesting
            
        Returns:
            Dictionary with filtered options data
        """
        logger.info(f"Aligning option expirations to target date {target_date}")
        
        valid_options = {}
        target_date_normalized = pd.Timestamp(target_date).normalize()
        
        for expiry_key, expiry_data in options_data.items():
            try:
                # Try multiple methods to get a reliable expiry date
                if 'week_expiry_date' in expiry_data.columns:
                    expiry_date = pd.to_datetime(expiry_data['week_expiry_date'].iloc[0]).normalize()
                elif 'expiry_date' in expiry_data.columns:
                    expiry_date = pd.to_datetime(expiry_data['expiry_date'].iloc[0]).normalize()
                else:
                    # Try to parse from the key
                    try:
                        expiry_date = pd.to_datetime(expiry_key).normalize()
                    except:
                        # Skip if we can't determine expiry
                        logger.warning(f"Cannot determine expiry date for {expiry_key}, skipping")
                        continue
                
                # Validate that the expiry date is logical
                data_start_date = expiry_data.index.min().date()
                expiry_date = pd.Timestamp(validate_expiry_date(expiry_date, data_start_date, f"Expiry key: {expiry_key}"))
                
                # Option must expire after the target date
                if expiry_date > target_date_normalized:
                    valid_options[expiry_key] = expiry_data
                    logger.info(f"Including options with expiry {expiry_date} (expires after target date)")
                else:
                    logger.warning(f"Excluding options with expiry {expiry_date} (already expired at target date)")
            except Exception as e:
                logger.warning(f"Error processing expiry {expiry_key}: {e} - skipping")
                continue
        
        logger.info(f"Aligned {len(valid_options)} option expirations")
        return valid_options
    
    def extract_date_from_filename(self, filename: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Attempt to extract date range from a filename.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Tuple of (start_date, end_date) if successful, None otherwise
        """
        # Various patterns to try
        import re
        
        # Pattern for "data_20200101_20200131.csv" format
        pattern1 = r'.*?(\d{8})_(\d{8})\.csv'
        match = re.match(pattern1, filename)
        if match:
            try:
                start = datetime.strptime(match.group(1), '%Y%m%d')
                end = datetime.strptime(match.group(2), '%Y%m%d')
                return (start, end)
            except:
                pass
                
        # Pattern for "data_2020-01.csv" (month data) format
        pattern2 = r'.*?(\d{4}-\d{2})\.csv'
        match = re.match(pattern2, filename)
        if match:
            try:
                month_str = match.group(1)
                start = datetime.strptime(f"{month_str}-01", '%Y-%m-%d')
                # Last day of month
                if start.month == 12:
                    end = datetime(start.year + 1, 1, 1) - timedelta(days=1)
                else:
                    end = datetime(start.year, start.month + 1, 1) - timedelta(days=1)
                return (start, end)
            except:
                pass
        
        # Pattern for "data_2020.csv" (year data) format
        pattern3 = r'.*?(\d{4})\.csv'
        match = re.match(pattern3, filename)
        if match:
            try:
                year = int(match.group(1))
                start = datetime(year, 1, 1)
                end = datetime(year, 12, 31)
                return (start, end)
            except:
                pass
        
        return None