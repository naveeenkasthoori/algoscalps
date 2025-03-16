"""
Utility functions for date handling and validation in options trading.
"""
import pandas as pd
import logging
from datetime import datetime, date, timedelta
import os

logger = logging.getLogger(__name__)

def validate_expiry_date(expiry_date, data_date, source_info=""):
    """
    Validate that an options expiry date makes logical sense relative to its data date.
    Preserves weekly sequencing for options data files.
    
    Args:
        expiry_date: The expiry date to validate (date or datetime)
        data_date: The date of the actual data (date or datetime)
        source_info: Optional source information for logging
        
    Returns:
        A valid expiry date (always a date object, not datetime)
    """
    # Convert to date objects if they're datetimes
    if isinstance(expiry_date, datetime):
        expiry_date = expiry_date.date()
    if isinstance(data_date, datetime):
        data_date = data_date.date()
        
    # Convert to date if it's a pandas Timestamp
    if isinstance(expiry_date, pd.Timestamp):
        expiry_date = expiry_date.date()
    if isinstance(data_date, pd.Timestamp):
        data_date = data_date.date()
        
    # Check if data_date is in 2022-2025 range but expiry_date is from 2020
    # This suggests a mismatch in the data that needs correction
    if (2022 <= data_date.year <= 2025) and expiry_date.year == 2020:
        logger.warning(f"Detected expiry year mismatch: {expiry_date} vs data date {data_date}. {source_info}")
        
        # Find the upcoming Thursday that would match the same month and approximate day
        # First find what Thursday in the month this would have been in 2020
        days_to_thursday = (3 - expiry_date.weekday()) % 7
        if days_to_thursday == 0:  # If it's already Thursday, use same date
            days_to_thursday = 7
            
        matching_thursday = expiry_date + timedelta(days=days_to_thursday)
        
        # Now create a date with same month/day pattern but in the data_date year
        matching_date_in_current_year = date(data_date.year, 
                                           expiry_date.month, 
                                           min(expiry_date.day, 28))  # Avoid Feb 29 issues
        
        # Find the next Thursday from this date
        days_to_thursday = (3 - matching_date_in_current_year.weekday()) % 7
        corrected_expiry = matching_date_in_current_year + timedelta(days=days_to_thursday)
        
        # Make sure it's not in the past relative to data_date
        while corrected_expiry < data_date:
            corrected_expiry += timedelta(days=7)  # Move to next Thursday
            
        logger.info(f"Corrected expiry date from {expiry_date} to {corrected_expiry}")
        return corrected_expiry
        
    # If expiry date is before data date, but doesn't match the year pattern above,
    # use a different correction strategy
    if expiry_date < data_date:
        logger.warning(f"Invalid expiry date {expiry_date} found (before data date {data_date}). {source_info}")
        
        # Find the next Thursday after the data date
        days_to_thursday = (3 - data_date.weekday()) % 7
        first_thursday = data_date + timedelta(days=days_to_thursday)
        
        # If first_thursday is the same day as data_date, move to next week
        if first_thursday == data_date:
            first_thursday += timedelta(days=7)
            
        logger.info(f"Corrected expiry date to {first_thursday}")
        return first_thursday
    
    # Check if expiry date is too far in the future (more than 3 months)
    # This could indicate a year error in the other direction
    if (expiry_date - data_date).days > 90:
        logger.warning(f"Expiry date {expiry_date} is too far in the future from data date {data_date}. {source_info}")
        
        # Try to correct by finding a closer Thursday
        corrected_expiry = find_closest_thursday_to_date(data_date)
        logger.info(f"Corrected far-future expiry date from {expiry_date} to {corrected_expiry}")
        return corrected_expiry
    
    return expiry_date

def get_next_expiry_date(from_date):
    """
    Get the next options expiry date (typically a Thursday) from a given date.
    
    Args:
        from_date: The starting date
        
    Returns:
        The next expiry date
    """
    if isinstance(from_date, pd.Timestamp):
        from_date = from_date.date()
    elif isinstance(from_date, datetime):
        from_date = from_date.date()
        
    # Calculate next Thursday
    days_to_thursday = (3 - from_date.weekday()) % 7
    if days_to_thursday == 0:  # If it's already Thursday, go to next week
        days_to_thursday = 7
        
    next_expiry = from_date + timedelta(days=days_to_thursday)
    
    # Try to load actual expiry dates from file if available
    expiry_file = 'data/processed/weekly_expiries.txt'
    if os.path.exists(expiry_file):
        try:
            with open(expiry_file, 'r') as f:
                expiry_dates = [pd.to_datetime(line.strip()).date() for line in f if line.strip()]
            
            # Find the next expiry date after from_date
            for expiry in sorted(expiry_dates):
                if expiry >= from_date:
                    return expiry
        except Exception as e:
            logger.warning(f"Error loading expiry dates from file: {e}")
    
    return next_expiry

def find_closest_thursday_to_date(target_date):
    """
    Find the closest Thursday to a given date.
    
    Args:
        target_date: The target date
        
    Returns:
        The closest Thursday (date object)
    """
    if isinstance(target_date, pd.Timestamp):
        target_date = target_date.date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    
    # Calculate days to previous and next Thursday
    days_to_prev_thursday = target_date.weekday() - 3 if target_date.weekday() >= 3 else target_date.weekday() + 4
    days_to_next_thursday = (3 - target_date.weekday()) % 7
    
    # Return the closest one
    if days_to_prev_thursday <= days_to_next_thursday:
        return target_date - timedelta(days=days_to_prev_thursday)
    else:
        return target_date + timedelta(days=days_to_next_thursday)

def is_valid_backtest_date_range(start_date, end_date, options_data=None):
    """
    Check if a date range is valid for backtesting with the available options data.
    
    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        options_data: Dictionary of options data by expiry (optional)
        
    Returns:
        Boolean indicating if the date range is valid
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).normalize()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).normalize()
    
    # If no options data provided, use a default valid range
    if options_data is None:
        # Default valid range is 2022-01-01 to 2022-12-31
        default_start = pd.to_datetime('2022-01-01').normalize()
        default_end = pd.to_datetime('2022-12-31').normalize()
        
        # Check if the requested date range is within the default valid range
        if start_date < default_start or end_date > default_end:
            logger.warning(f"Date range {start_date.date()} to {end_date.date()} is outside the recommended range")
            logger.warning(f"Recommended date range is {default_start.date()} to {default_end.date()}")
            return False
        return True
    
    # If options data is provided, check against it
    if not options_data:
        logger.warning("No options data available to validate date range")
        return False
    
    # Find the earliest and latest dates in the options data
    earliest_date = None
    latest_date = None
    
    for expiry, data in options_data.items():
        if data.empty:
            continue
        
        expiry_earliest = data.index.min()
        expiry_latest = data.index.max()
        
        if earliest_date is None or expiry_earliest < earliest_date:
            earliest_date = expiry_earliest
        
        if latest_date is None or expiry_latest > latest_date:
            latest_date = expiry_latest
    
    if earliest_date is None or latest_date is None:
        logger.warning("No valid data found in options dataset")
        return False
    
    # Check if the requested date range is within the available data
    if start_date < earliest_date:
        logger.warning(f"Start date {start_date.date()} is before earliest available data {earliest_date.date()}")
        return False
        
    if end_date > latest_date:
        logger.warning(f"End date {end_date.date()} is after latest available data {latest_date.date()}")
        return False
    
    return True 