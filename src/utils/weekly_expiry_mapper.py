"""
Utility for mapping weekly option expiry dates to enable correct historical data processing.
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class WeeklyExpiryMapper:
    """
    Utility for managing and mapping weekly option expiry dates.
    """
    
    def __init__(self, expiry_dates: List[str] = None):
        """
        Initialize the weekly expiry mapper.
        
        Args:
            expiry_dates: Optional list of expiry dates as strings (YYYY-MM-DD)
        """
        self.expiry_dates = []
        self.expiry_map = {}  # Maps date ranges to expiry dates
        
        if expiry_dates:
            self.load_expiry_dates(expiry_dates)
            
        logger.info(f"Weekly Expiry Mapper initialized with {len(self.expiry_dates)} expiry dates")
    
    def load_expiry_dates(self, expiry_dates: List[str]) -> bool:
        """
        Load expiry dates from a list of strings.
        
        Args:
            expiry_dates: List of expiry dates as strings (YYYY-MM-DD)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.expiry_dates = [pd.to_datetime(date.strip()) for date in expiry_dates if date.strip()]
            self.expiry_dates.sort()
            self._build_expiry_map()
            logger.info(f"Loaded {len(self.expiry_dates)} expiry dates")
            return True
        except Exception as e:
            logger.error(f"Error loading expiry dates: {e}")
            return False
    
    def load_expiry_dates_from_file(self, file_path: str) -> bool:
        """
        Load expiry dates from a file (one date per line).
        
        Args:
            file_path: Path to the file containing expiry dates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                expiry_dates = [line.strip() for line in f if line.strip()]
            
            return self.load_expiry_dates(expiry_dates)
        except Exception as e:
            logger.error(f"Error loading expiry dates from file: {e}")
            return False
    
    def save_expiry_dates_to_file(self, file_path: str) -> bool:
        """
        Save the current expiry dates to a file.
        
        Args:
            file_path: Path to save the expiry dates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                for date in self.expiry_dates:
                    f.write(f"{date.strftime('%Y-%m-%d')}\n")
            
            logger.info(f"Saved {len(self.expiry_dates)} expiry dates to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving expiry dates to file: {e}")
            return False
    
    def _build_expiry_map(self):
        """Build a mapping between date ranges and corresponding expiry dates."""
        if not self.expiry_dates:
            return
        
        self.expiry_map = {}
        
        # For each expiry date, map the preceding week (or partial week) to it
        for i, expiry_date in enumerate(self.expiry_dates):
            # Start from day after previous expiry, or beginning of time if first expiry
            if i == 0:
                start_date = pd.Timestamp.min
            else:
                start_date = self.expiry_dates[i-1] + pd.Timedelta(days=1)
            
            # End at this expiry date
            end_date = expiry_date
            
            # Map this range to the expiry date
            self.expiry_map[(start_date, end_date)] = expiry_date
            
        logger.debug(f"Built expiry map with {len(self.expiry_map)} date ranges")
    
    def get_expiry_for_date(self, date: Union[str, datetime, pd.Timestamp]) -> Optional[pd.Timestamp]:
        """
        Get the next expiry date for a given date.
        
        Args:
            date: The date to find the expiry for
            
        Returns:
            Next expiry date, or None if not found
        """
        if not self.expiry_dates:
            logger.warning("No expiry dates loaded")
            return None
        
        # Convert to pandas Timestamp for consistent comparison
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Direct lookup from map
        for (start, end), expiry in self.expiry_map.items():
            if start <= date <= end:
                return expiry
        
        # Fallback: find the next expiry after the date
        for expiry in self.expiry_dates:
            if expiry >= date:
                return expiry
        
        # If no expiry is found (date is after all known expiries)
        logger.warning(f"No valid expiry found for date {date}")
        return None
    
    def map_historical_data_expiries(self, 
                                    options_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Correct the expiry dates in historical options data.
        
        Args:
            options_data: Dictionary of options data by expiry
            
        Returns:
            Dictionary with corrected expiry dates
        """
        if not self.expiry_dates:
            logger.warning("No expiry dates loaded, cannot map historical data")
            return options_data
        
        logger.info(f"Mapping historical options data expiries ({len(options_data)} datasets)")
        
        corrected_data = {}
        
        for expiry_key, data_df in options_data.items():
            try:
                # Get the actual data date range
                if data_df.empty:
                    logger.warning(f"Empty DataFrame for expiry {expiry_key}, skipping")
                    continue
                    
                data_date = data_df.index.min()
                
                # Try to determine historical expiry date
                try:
                    # First try direct parsing from key
                    historical_expiry = pd.to_datetime(expiry_key)
                except:
                    # Check if it's in the DataFrame
                    if 'week_expiry_date' in data_df.columns:
                        historical_expiry = pd.to_datetime(data_df['week_expiry_date'].iloc[0])
                    elif 'expiry_date' in data_df.columns:
                        historical_expiry = pd.to_datetime(data_df['expiry_date'].iloc[0])
                    else:
                        logger.warning(f"Could not determine expiry date for {expiry_key}, skipping")
                        continue
                
                # Map to corrected expiry
                corrected_expiry = self.get_expiry_for_date(data_date)
                
                if corrected_expiry:
                    # Create a copy of the DataFrame with updated expiry
                    df_copy = data_df.copy()
                    
                    # Update expiry date columns
                    for col in ['week_expiry_date', 'expiry_date']:
                        if col in df_copy.columns:
                            df_copy[col] = corrected_expiry
                    
                    # Store with corrected key
                    corrected_key = corrected_expiry.strftime('%Y-%m-%d')
                    corrected_data[corrected_key] = df_copy
                    
                    logger.info(f"Mapped {expiry_key} ({historical_expiry}) to {corrected_key} for data date {data_date}")
                else:
                    logger.warning(f"Could not find a valid expiry for {expiry_key} (data date: {data_date})")
                    # Include the original data as fallback
                    corrected_data[expiry_key] = data_df
            
            except Exception as e:
                logger.error(f"Error processing expiry {expiry_key}: {e}")
                # Include the original data as fallback
                corrected_data[expiry_key] = data_df
        
        logger.info(f"Mapping complete: {len(corrected_data)} datasets")
        return corrected_data
    
    def filter_by_date_range(self, 
                           start_date: Union[str, datetime, pd.Timestamp],
                           end_date: Optional[Union[str, datetime, pd.Timestamp]] = None) -> List[pd.Timestamp]:
        """
        Get expiry dates that fall within the specified date range.
        
        Args:
            start_date: Start date
            end_date: Optional end date
            
        Returns:
            List of expiry dates within the range
        """
        if not self.expiry_dates:
            return []
        
        # Convert to pandas Timestamp for consistent comparison
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        if isinstance(end_date, str) and end_date is not None:
            end_date = pd.to_datetime(end_date)
        
        if end_date is None:
            end_date = pd.Timestamp.max
        
        filtered_dates = [date for date in self.expiry_dates if start_date <= date <= end_date]
        
        logger.info(f"Filtered {len(filtered_dates)} expiry dates in range {start_date} to {end_date}")
        return filtered_dates