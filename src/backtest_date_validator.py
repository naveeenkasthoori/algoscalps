"""
Backtest Date Range Validator and Fixer.
Ensures compatibility between backtest date range and options data.

This module addresses the critical issue of date mismatches between options expiry dates
and backtest periods, which can lead to no trades being executed in the backtest.
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from src.utils.date_utils import validate_expiry_date

logger = logging.getLogger(__name__)

class BacktestDateValidator:
    """
    Validates and fixes backtest date range to ensure compatibility with options data.
    """
    
    def __init__(self):
        """Initialize the backtest date validator."""
        logger.info("Backtest Date Validator initialized")
    
    def validate_dates(self, 
                      start_date: Union[str, datetime, None],
                      end_date: Union[str, datetime, None],
                      options_data: Dict[str, pd.DataFrame]) -> Tuple[bool, str, Optional[datetime], Optional[datetime]]:
        """
        Validate the requested date range against available data.
        
        Args:
            start_date: Requested start date
            end_date: Requested end date
            options_data: Dictionary of options data by expiry
            
        Returns:
            Tuple of (is_valid, message, validated_start_date, validated_end_date)
        """
        logger.info(f"Validating backtest dates: {start_date} to {end_date}")
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date).normalize()
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Get all available expiry dates
        expiry_dates = []
        logger.info(f"Processing {len(options_data)} options files")
        
        # Debug: Print all available expiries and their data ranges
        logger.info("Available options data:")
        for expiry_key, data in options_data.items():
            try:
                data_start = data.index.min()
                data_end = data.index.max()
                data_date = data_start.date()
                
                # Try to get expiry from the data first
                if 'week_expiry_date' in data.columns:
                    expiry_date = pd.to_datetime(data['week_expiry_date'].iloc[0]).normalize()
                elif 'expiry_date' in data.columns:  # For futures data compatibility
                    expiry_date = pd.to_datetime(data['expiry_date'].iloc[0]).normalize()
                else:
                    # Try to parse expiry date from the key
                    try:
                        expiry_date = pd.to_datetime(expiry_key).normalize()
                    except:
                        # If can't parse from key, use the one already set in the dataframe
                        expiry_date = pd.to_datetime(expiry_key).normalize()
                
                # Validate the expiry date against the data date
                expiry_date = pd.Timestamp(validate_expiry_date(expiry_date, data_date, f"Expiry key: {expiry_key}"))
                
                logger.info(f"Expiry {expiry_date}: Data range {data_start} to {data_end}")
                
                # Check if this expiry's data overlaps with our date range
                if (end_date is None or data_start <= end_date) and \
                   (start_date is None or data_end >= start_date):
                    expiry_dates.append(expiry_date)
                    logger.info(f"Valid expiry found: {expiry_date}")
            
            except Exception as e:
                logger.warning(f"Error processing expiry {expiry_key}: {e}")
                continue
        
        if not expiry_dates:
            logger.warning("No valid expiry dates found")
            return False, "No valid options data found for the specified date range", None, None
        
        # Sort expiry dates
        expiry_dates.sort()
        logger.info(f"Found valid expiry dates: {expiry_dates}")
        
        # Find the earliest and latest valid dates
        earliest_data = min(data.index.min() for data in options_data.values())
        latest_data = max(data.index.max() for data in options_data.values())
        
        logger.info(f"Data range: {earliest_data} to {latest_data}")
        
        # Adjust start and end dates to be within data bounds
        validated_start = max(start_date, earliest_data)
        validated_end = min(end_date, latest_data)
        
        # Check if we have any valid expiries after the start date
        valid_expiries = [d for d in expiry_dates if d >= validated_start]
        if not valid_expiries:
            msg = f"No options expire after start date {start_date}"
            logger.warning(msg)
            return False, msg, None, None
        
        # Ensure the date range is valid
        if validated_end < validated_start:
            msg = f"Invalid date range: end date {validated_end} is before start date {validated_start}"
            logger.warning(msg)
            return False, msg, None, None
        
        msg = f"Valid date range found: {validated_start} to {validated_end}"
        logger.info(msg)
        return True, msg, validated_start, validated_end
    
    def suggest_valid_date_range(self, options_data: Dict[str, pd.DataFrame]) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Suggest a valid date range for backtesting based on available options data.
        
        Args:
            options_data: Dictionary of options data by expiry
            
        Returns:
            Tuple of (suggested_start: datetime, suggested_end: datetime)
        """
        if not options_data:
            return None, None
        
        # Extract option expiry dates
        option_expiry_dates = []
        for expiry_str, _ in options_data.items():
            try:
                expiry_date = pd.to_datetime(expiry_str)
                option_expiry_dates.append(expiry_date)
            except:
                continue
        
        if not option_expiry_dates:
            return None, None
        
        # Sort expiry dates
        option_expiry_dates.sort()
        
        # Find earliest and latest data timestamps across all options
        earliest_data = None
        latest_data = None
        
        for _, expiry_data in options_data.items():
            if not expiry_data.empty:
                if earliest_data is None or expiry_data.index.min() < earliest_data:
                    earliest_data = expiry_data.index.min()
                if latest_data is None or expiry_data.index.max() > latest_data:
                    latest_data = expiry_data.index.max()
        
        if earliest_data is None or latest_data is None:
            return None, None
        
        # For a valid backtest, we need options that expire after our start date
        # Find earliest expiry date
        earliest_expiry = min(option_expiry_dates)
        
        # Suggest a start date a few days before the earliest expiry
        # This ensures we have at least one valid option to trade
        suggested_start = earliest_data
        
        # Suggest an end date before the latest data date
        suggested_end = latest_data
        
        logger.info(f"Suggested date range: {suggested_start} to {suggested_end}")
        
        return suggested_start, suggested_end
    
    def filter_valid_options(self,
                            options_data: Dict[str, pd.DataFrame],
                            start_date: Union[str, datetime],
                            end_date: Optional[Union[str, datetime]] = None) -> Dict[str, pd.DataFrame]:
        """
        Filter options data to only include options with expiry dates valid for the backtest period.
        
        Args:
            options_data: Dictionary of options data by expiry
            start_date: Backtest start date
            end_date: Optional backtest end date
            
        Returns:
            Filtered options data dictionary
        """
        # Convert dates to datetime objects if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str) and end_date is not None:
            end_date = pd.to_datetime(end_date)
        
        logger.info(f"Filtering options for date range: {start_date} to {end_date}")
        
        filtered_options = {}
        
        for expiry_str, expiry_data in options_data.items():
            try:
                expiry_date = pd.to_datetime(expiry_str)
                
                # Option must expire after the start date
                if expiry_date >= start_date:
                    # If end date is specified, option must expire before or on the end date
                    if end_date is None or expiry_date <= end_date:
                        filtered_options[expiry_str] = expiry_data
                        logger.info(f"Including options with expiry {expiry_str}")
                    else:
                        logger.debug(f"Excluding options with expiry {expiry_str} (expires after end date)")
                else:
                    logger.debug(f"Excluding options with expiry {expiry_str} (expires before start date)")
            except:
                logger.warning(f"Excluding options with unparseable expiry {expiry_str}")
        
        if not filtered_options:
            logger.warning("No valid options found for the specified date range!")
        else:
            logger.info(f"Filtered to {len(filtered_options)} valid option expiry dates")
        
        return filtered_options
    
    def find_compatible_data_period(self,
                                  spot_data: pd.DataFrame,
                                  futures_data: pd.DataFrame,
                                  options_data: Dict[str, pd.DataFrame]) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Find a period where all data sources have compatible data.
        """
        if spot_data.empty or futures_data.empty or not options_data:
            return None, None
        
        # Find common range between spot and futures
        common_start = max(spot_data.index.min(), futures_data.index.min())
        common_end = min(spot_data.index.max(), futures_data.index.max())
        
        # Get valid options expiries
        valid_expiries = []
        for expiry, data in options_data.items():
            try:
                expiry_date = pd.Timestamp(expiry).normalize()
                data_start = data.index.min()
                data_end = data.index.max()
                
                # Check if this expiry's data overlaps with our common range
                if data_end >= common_start and data_start <= common_end:
                    valid_expiries.append(expiry_date)
            except:
                continue
        
        if not valid_expiries:
            return None, None
        
        # Sort expiries
        valid_expiries.sort()
        
        # Find the earliest and latest valid trading times
        earliest_time = pd.Timestamp('09:15:00').time()
        latest_time = pd.Timestamp('15:30:00').time()
        
        # Adjust common range to trading hours
        final_start = pd.Timestamp.combine(common_start.date(), earliest_time)
        final_end = pd.Timestamp.combine(common_end.date(), latest_time)
        
        return final_start, final_end

    def get_next_expiry_date(self, 
                            from_date: Union[str, datetime], 
                            options_data: Dict[str, pd.DataFrame]) -> Optional[datetime]:
        """
        Get the next available expiry date after the given date.
        
        Args:
            from_date: Date to find the next expiry from
            options_data: Dictionary of options data by expiry
            
        Returns:
            Next expiry date, or None if not found
        """
        # Convert to datetime if string
        if isinstance(from_date, str):
            from_date = pd.to_datetime(from_date)
            
        # Extract all expiry dates
        expiry_dates = []
        for expiry_key, data in options_data.items():
            try:
                # Try to get expiry from the data first
                if 'week_expiry_date' in data.columns:
                    expiry_date = pd.to_datetime(data['week_expiry_date'].iloc[0]).normalize()
                elif 'expiry_date' in data.columns:
                    expiry_date = pd.to_datetime(data['expiry_date'].iloc[0]).normalize()
                else:
                    # Try to parse from key
                    expiry_date = pd.to_datetime(expiry_key).normalize()
                
                expiry_dates.append(expiry_date)
            except Exception as e:
                logger.warning(f"Error parsing expiry date from {expiry_key}: {e}")
                continue
        
        if not expiry_dates:
            return None
            
        # Sort expiry dates
        expiry_dates.sort()
        
        # Find first expiry after from_date
        for expiry in expiry_dates:
            if expiry > from_date:
                return expiry
                
        return None