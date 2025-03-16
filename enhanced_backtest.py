#!/usr/bin/env python
"""
Enhanced Efficient Options Trading System Backtest
With optimized parameters and improved signal generation
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import psutil
from typing import Dict, Optional, Tuple, List, Union, Any
import gc
import warnings
import types
import multiprocessing
import re
from scipy.stats import norm

# Set up root logger
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_backtest")

# Import components
from config.settings import PROCESSED_DATA_DIR, OUTPUT_DATA_DIR, MARKET_OPEN_TIME, MARKET_CLOSE_TIME
from config.strategy_params import TOTAL_CAPITAL

from src.data_handlers.data_loader import EnhancedDataLoader
from src.data_handlers.data_synchronizer import DataSynchronizer
from src.indicators.parallel_processor import ParallelIndicatorProcessor
from src.strategy.signal_generator import SignalGenerator
from src.engine.position_manager import PositionManager
from src.engine.risk_manager import RiskManager
from src.engine.trading_engine import TradingEngine
from src.reporting.report_generator import ReportGenerator
from src.utils.date_utils import validate_expiry_date, is_valid_backtest_date_range, find_closest_thursday_to_date

# Import get_next_expiry_date function
try:
    # Try to import from strategy.rules first
    from src.strategy.rules import find_next_expiry_options, get_next_expiry_date
except ImportError:
    try:
        # Try to import from backtest_date_validator as fallback
        from src.backtest_date_validator import get_next_expiry_date
    except ImportError:
        # Define a fallback implementation if import fails
        logger.warning("Could not import get_next_expiry_date, using fallback implementation")
        
        def get_next_expiry_date(from_date: datetime) -> Optional[datetime]:
            """
            Fallback implementation to get the next expiry date.
            Uses standard weekly expiry logic (Thursday of current or next week).
            
            Args:
                from_date: Date to find the next expiry from
                
            Returns:
                Next expiry date, or None if not found
            """
            if not isinstance(from_date, datetime):
                try:
                    from_date = pd.to_datetime(from_date)
                except:
                    return None
            
            # Get the day of week (0=Monday, 6=Sunday)
            day_of_week = from_date.weekday()
            
            # Calculate days until next Thursday (3 is Thursday)
            days_to_thursday = (3 - day_of_week) % 7
            
            # If today is Thursday and it's before market close, use today
            if day_of_week == 3 and from_date.time() < datetime.strptime(MARKET_CLOSE_TIME, "%H:%M:%S").time():
                next_expiry = from_date.date()
            else:
                # Otherwise use next Thursday
                if days_to_thursday == 0:  # It's Thursday but after market close
                    days_to_thursday = 7  # Go to next Thursday
                
                next_expiry = (from_date + timedelta(days=days_to_thursday)).date()
            
            return pd.Timestamp(next_expiry)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Enhanced Options Trading System Backtest')
    
    parser.add_argument('--spot-data', type=str, default='data/input/spot/Nifty_spot_2020_25.csv',
                        help='Path to spot data CSV file')
    
    parser.add_argument('--futures-data', type=str, default='data/input/futures/nifty_fut_1min_month_1.csv',
                        help='Path to futures data CSV file')
    
    parser.add_argument('--options-data-dir', type=str, default='data/input/options/',
                        help='Directory containing options data CSV files')
    
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DATA_DIR,
                        help='Directory for output files')
    
    parser.add_argument('--start-date', type=str, default='2022-01-03',
                        help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default='2022-01-28',
                        help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--capital', type=float, default=2000000,
                        help='Total capital for trading')
    
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of worker processes for parallel processing')
    
    parser.add_argument('--chunk-size', type=int, default=500,
                        help='Size of chunks for timestamp processing (default: 500)')
    
    parser.add_argument('--skip-greeks', action='store_true',
                        help='Skip calculation of options Greeks (saves memory)')
    
    parser.add_argument('--max-rows-per-expiry', type=int, default=1000000,
                        help='Maximum number of rows to process per options expiry')
    
    parser.add_argument('--debug-report', action='store_true',
                        help='Generate detailed debug report for data analysis')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    parser.add_argument('--require-conditions', type=int, default=5,
                        help='Minimum number of conditions required for signal generation (default: 5)')
    
    parser.add_argument('--mandatory-conditions', type=str, default='cvd,vwap,gamma',
                        help='Comma-separated list of mandatory conditions (default: cvd,vwap,gamma)')
    
    parser.add_argument('--strikes-range', type=int, default=6,
                        help='Number of strikes above/below ATM to consider (default: 6)')
    
    parser.add_argument('--max-signals-per-timestamp', type=int, default=3,
                        help='Maximum number of signals to generate per timestamp (default: 3)')
    
    # Add new Dask-related arguments
    parser.add_argument('--use-dask', action='store_true',
                        help='Use Dask for processing very large files')
    
    parser.add_argument('--dask-partition-size', type=int, default=100,
                        help='Dask partition size in MB (default: 100MB)')
    
    parser.add_argument('--dask-memory-limit', type=str, default='8GB',
                        help='Memory limit for Dask operations (default: 8GB)')
    
    parser.add_argument('--dask-temp-dir', type=str, default=None,
                        help='Directory for Dask temporary files')
    
    parser.add_argument('--sample-rate', type=float, default=1.0,
                       help='Sample fraction of data for faster processing (0.0-1.0)')
    
    # New arguments for win rate optimization features
    parser.add_argument('--adaptive-stop-loss', action='store_true', default=True,
                        help='Use adaptive stop loss based on volatility (default: True)')
    
    parser.add_argument('--trailing-stop', action='store_true', default=True,
                        help='Use trailing stop loss to lock in profits (default: True)')
    
    parser.add_argument('--weighted-scoring', action='store_true', default=True,
                        help='Use weighted condition scoring instead of binary checks (default: True)')
    
    parser.add_argument('--market-regime-adaptation', action='store_true', default=True,
                        help='Adapt strategy to different market regimes (default: True)')
    
    parser.add_argument('--time-based-targets', action='store_true', default=True,
                        help='Adjust profit targets based on time to expiry (default: True)')
    
    parser.add_argument('--optimal-strikes', action='store_true', default=True,
                        help='Use optimized strike selection (default: True)')
    
    parser.add_argument('--max-trades-per-day', type=int, default=5,
                        help='Maximum number of trades allowed per day (default: 5)')
    
    # Add new log level argument
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Set the logging level')
    
    parser.add_argument('--auto-adjust-dates', action='store_true',
                        help='Automatically adjust date range if invalid')
    
    parser.add_argument('--skip-options', action='store_true',
                        help='Skip loading options data (for testing only)')
    
    return parser.parse_args()

def log_system_info():
    """Log system information for diagnostics."""
    logger.info("System Information:")
    logger.info(f"CPU Count: {os.cpu_count()}")
    
    memory = psutil.virtual_memory()
    logger.info(f"Total Memory: {memory.total / (1024**3):.2f} GB")
    logger.info(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    
    disk = psutil.disk_usage('/')
    logger.info(f"Disk Space: Total={disk.total / (1024**3):.2f} GB, Free={disk.free / (1024**3):.2f} GB")

def log_memory_usage(label=""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    logger.info(f"Memory usage ({label}): {memory_usage_mb:.2f} MB")

def filter_strikes_near_atm(options_df: pd.DataFrame, futures_price: float, num_strikes: int = 6) -> pd.DataFrame:
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
    
    if not unique_strikes:
        return options_df
    
    # Find ATM strike (closest to futures price)
    atm_strike = min(unique_strikes, key=lambda x: abs(x - futures_price))
    
    # Find ATM strike index
    try:
        atm_idx = unique_strikes.index(atm_strike)
    except ValueError:
        # If ATM strike not found, return original DataFrame
        return options_df
    
    # Get strikes above and below ATM
    start_idx = max(0, atm_idx - num_strikes)
    end_idx = min(len(unique_strikes) - 1, atm_idx + num_strikes)
    
    selected_strikes = unique_strikes[start_idx:end_idx + 1]
    
    # Filter options dataframe
    return options_df[options_df['strike_price'].isin(selected_strikes)]

def filter_top_signals(signals: dict, max_signals: int = 3) -> dict:
    """
    Filter to keep only the top N signals with the highest quality score.
    
    Args:
        signals: Dictionary with signal information
        max_signals: Maximum number of signals to keep
        
    Returns:
        Filtered signals dictionary
    """
    if 'entry_signals' not in signals or not signals['entry_signals']:
        return signals
    
    # Get entry signals
    entry_signals = signals['entry_signals']
    
    # Calculate a quality score for each signal
    for signal in entry_signals:
        option = signal.get('option', {})
        conditions = signal.get('conditions', {})
        conditions_met = signal.get('conditions_met', 0)
        
        # Quality factors
        delta_score = abs(option.get('delta', 0)) * 10  # 0-10 points for delta strength
        gamma_score = min(option.get('gamma', 0) * 100, 5)  # 0-5 points for gamma
        vega_score = min(option.get('vega', 0) if 'vega' in option else 0, 3)  # 0-3 points for vega
        
        # Condition-based score
        condition_score = conditions_met * 2  # 2 points per condition
        
        # CVD confirmation adds value
        cvd_bonus = 3 if conditions.get('cvd', False) else 0
        
        # Calculate total quality score
        signal['quality_score'] = delta_score + gamma_score + vega_score + condition_score + cvd_bonus
    
    # Sort signals by quality score (descending)
    sorted_signals = sorted(entry_signals, key=lambda x: x.get('quality_score', 0), reverse=True)
    
    # Keep only top N signals
    signals['entry_signals'] = sorted_signals[:max_signals]
    
    return signals

def fix_options_expiry_dates(options_data: Dict[str, pd.DataFrame], start_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """
    Fix options expiry dates with enhanced validation and error handling.
    
    Args:
        options_data: Dictionary of options data by expiry
        start_date: Start date of the backtest
        
    Returns:
        Dictionary of fixed options data with validated expiry dates
    """
    logger.info(f"Verifying options expiry dates for backtest period starting {start_date}")
    
    fixed_options = {}
    weekly_expiries = set()
    
    # Step 1: Load known weekly expiry dates from file
    try:
        # Check multiple possible locations for the expiry file
        expiry_file_paths = [
            "data/processed/weekly_expiries.txt",  # Standard location
            "weekly_expiries.txt",                 # Root directory
            "config/weekly_expiries.txt"           # Config directory
        ]
        
        expiry_file = None
        for path in expiry_file_paths:
            if os.path.exists(path):
                expiry_file = path
                break
        
        if expiry_file:
            with open(expiry_file, 'r') as f:
                for line in f:
                    try:
                        expiry = pd.to_datetime(line.strip()).normalize()
                        if expiry.date() >= start_date.date():
                            weekly_expiries.add(str(expiry.date()))
                    except Exception as e:
                        logger.debug(f"Skipping invalid expiry date in file: {line.strip()}, error: {e}")
                        continue
            logger.info(f"Loaded {len(weekly_expiries)} weekly expiry dates from file: {expiry_file}")
            logger.debug(f"Weekly expiries: {sorted(weekly_expiries)}")
        else:
            # If no expiry file found, try to create one from paste.txt
            if os.path.exists("paste.txt"):
                logger.info("No expiry file found, but paste.txt exists. Attempting to create expiry file.")
                try:
                    # Run the extract_weekly_expiries.py script
                    import subprocess
                    result = subprocess.run(["python", "extract_weekly_expiries.py"], 
                                           capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info("Successfully created weekly expiry file from paste.txt")
                        # Try loading the newly created file
                        if os.path.exists("data/processed/weekly_expiries.txt"):
                            with open("data/processed/weekly_expiries.txt", 'r') as f:
                                for line in f:
                                    try:
                                        expiry = pd.to_datetime(line.strip()).normalize()
                                        if expiry.date() >= start_date.date():
                                            weekly_expiries.add(str(expiry.date()))
                                    except Exception as e:
                                        continue
                            logger.info(f"Loaded {len(weekly_expiries)} weekly expiry dates from newly created file")
                    else:
                        logger.warning(f"Failed to create expiry file: {result.stderr}")
                except Exception as e:
                    logger.warning(f"Error creating expiry file from paste.txt: {e}")
    except Exception as e:
        logger.warning(f"Could not load expiry dates from file: {e}")
    
    # Step 2: Process and validate each expiry dataset
    for expiry_key, expiry_df in options_data.items():
        try:
            if expiry_df.empty:
                logger.warning(f"Skipping empty DataFrame for expiry {expiry_key}")
                continue
                
            # Get data date range for validation
            data_start = expiry_df.index.min()
            data_end = expiry_df.index.max()
            
            # Try multiple methods to determine expiry date
            expiry_date = None
            expiry_sources = [
                ('key', expiry_key),
                ('week_expiry_date', expiry_df['week_expiry_date'].iloc[0] if 'week_expiry_date' in expiry_df.columns else None),
                ('expiry_date', expiry_df['expiry_date'].iloc[0] if 'expiry_date' in expiry_df.columns else None)
            ]
            
            for source_name, source_value in expiry_sources:
                if source_value is not None:
                    try:
                        candidate_date = pd.to_datetime(source_value).normalize()
                        if expiry_date is None or candidate_date > expiry_date:
                            expiry_date = candidate_date
                            logger.debug(f"Found expiry date {expiry_date} from {source_name}")
                    except Exception as e:
                        logger.debug(f"Could not parse expiry from {source_name}: {e}")
                        continue
            
            # Check if the expiry date is in our known weekly expiries
            if expiry_date is not None and str(expiry_date.date()) in weekly_expiries:
                logger.debug(f"Expiry date {expiry_date.date()} found in weekly expiries list")
            elif weekly_expiries:
                # Try to find the closest weekly expiry date
                closest_expiry = None
                min_days_diff = float('inf')
                
                for expiry_str in weekly_expiries:
                    expiry = pd.to_datetime(expiry_str).date()
                    if expiry > data_start.date():
                        days_diff = (expiry - data_start.date()).days
                        if days_diff < min_days_diff:
                            min_days_diff = days_diff
                            closest_expiry = expiry
                
                if closest_expiry and (expiry_date is None or abs((closest_expiry - expiry_date.date()).days) < 7):
                    logger.info(f"Using closest weekly expiry {closest_expiry} instead of {expiry_date.date() if expiry_date else 'None'}")
                    expiry_date = pd.Timestamp(closest_expiry)
            
            if expiry_date is None:
                logger.warning(f"Could not determine expiry date for {expiry_key}, attempting to infer")
                # Try to infer from data dates
                try:
                    expiry_date = get_next_expiry_date(data_start)
                    if expiry_date is None:
                        # If get_next_expiry_date returns None, try a simple heuristic
                        # Use the next Thursday after data_start
                        day_of_week = data_start.weekday()
                        days_to_thursday = (3 - day_of_week) % 7
                        if days_to_thursday == 0:  # It's Thursday
                            days_to_thursday = 7  # Use next Thursday
                        expiry_date = data_start + timedelta(days=days_to_thursday)
                        logger.info(f"Using heuristic to set expiry date to next Thursday: {expiry_date.date()}")
                except Exception as e:
                    logger.error(f"Error inferring expiry date: {e}")
                    # Last resort: use data_start + 7 days (typical weekly expiry)
                    expiry_date = data_start + timedelta(days=7)
                    logger.warning(f"Using fallback expiry date (data_start + 7 days): {expiry_date.date()}")
                
                if expiry_date is None:
                    logger.error(f"Could not infer expiry date for {expiry_key}, skipping")
                    continue
            
            # Validate expiry date
            try:
                # Use the validation utility
                validated_expiry = validate_expiry_date(expiry_date, data_start.date(), f"Expiry key: {expiry_key}")
                
                # Additional validation checks
                if validated_expiry <= start_date.date():
                    logger.warning(f"Expiry {validated_expiry} is before backtest start date {start_date.date()}, skipping")
                    continue
                
                if validated_expiry <= data_start.date():
                    logger.warning(f"Expiry {validated_expiry} is before data start date {data_start.date()}, skipping")
                    continue
                
                # Verify it's a Thursday (typical expiry day)
                if validated_expiry.weekday() != 3:  # 3 is Thursday
                    logger.warning(f"Expiry {validated_expiry} is not a Thursday, adjusting")
                    # Find the closest Thursday
                    closest_thursday = find_closest_thursday_to_date(validated_expiry)
                    logger.info(f"Adjusted expiry from {validated_expiry} to closest Thursday: {closest_thursday}")
                    validated_expiry = closest_thursday
                
                # Create standardized key
                std_key = validated_expiry.strftime("%Y-%m-%d")
                
                # Verify data quality
                if len(expiry_df) < 10:  # Arbitrary minimum data points
                    logger.warning(f"Very few data points ({len(expiry_df)}) for {std_key}, verify data quality")
                
                # Check for price continuity
                if 'tr_close' in expiry_df.columns:
                    price_changes = expiry_df['tr_close'].pct_change().abs()
                    large_changes = price_changes[price_changes > 0.1].index  # 10% changes
                    if not large_changes.empty:
                        logger.warning(f"Large price changes detected for {std_key} at: {large_changes}")
                
                # Store validated data
                fixed_options[std_key] = expiry_df
                logger.info(f"Validated expiry {std_key} with {len(expiry_df)} rows, data range: {data_start} to {data_end}")
            
            except Exception as e:
                logger.error(f"Error validating expiry date for {expiry_key}: {e}")
                continue
            
        except Exception as e:
            logger.error(f"Error processing expiry dataset {expiry_key}: {e}")
            continue
    
    # Step 3: Final validation and reporting
    if not fixed_options:
        logger.error("No valid options data after fixing expiry dates!")
        return {}
    
    # Report on the fixed dataset
    logger.info(f"Successfully processed {len(fixed_options)} expiry dates")
    logger.info("Expiry date summary:")
    
    for expiry in sorted(fixed_options.keys()):
        df = fixed_options[expiry]
        date_range = f"{df.index.min()} to {df.index.max()}"
        avg_volume = df['tr_volume'].mean() if 'tr_volume' in df.columns else 'N/A'
        logger.info(f"  {expiry}: {len(df)} rows, {date_range}, Avg Volume: {avg_volume}")
    
    return fixed_options

def apply_custom_signal_filters(signal_generator: SignalGenerator, 
                               args: argparse.Namespace) -> SignalGenerator:
    """
    Apply custom signal filters to the signal generator.
    
    Args:
        signal_generator: Signal generator instance
        args: Command-line arguments
        
    Returns:
        Modified signal generator
    """
    # Parse mandatory conditions from command-line argument
    if args.mandatory_conditions:
        mandatory_conditions = args.mandatory_conditions.split(',')
        signal_generator.set_mandatory_conditions(mandatory_conditions)
        logger.info(f"Set mandatory conditions: {mandatory_conditions}")
    
    # Set the minimum number of required conditions
    if args.require_conditions:
        signal_generator.set_min_conditions_required(args.require_conditions)
        logger.info(f"Set minimum required conditions: {args.require_conditions}")
    
    # Store original generate_signals method to wrap
    original_generate_signals = signal_generator.generate_signals
    
    # Define wrapper function with modified behavior
    def enhanced_generate_signals(timestamp, spot_data, futures_data, options_data, open_positions=None, market_regime=None):
        # Call original method with all parameters including market_regime
        signals = original_generate_signals(
            timestamp=timestamp, 
            spot_data=spot_data, 
            futures_data=futures_data, 
            options_data=options_data, 
            open_positions=open_positions,
            market_regime=market_regime
        )
        
        # Apply custom filtering
        signals = filter_top_signals(signals, max_signals=args.max_signals_per_timestamp)
        
        return signals
    
    # Replace the method
    signal_generator.generate_signals = enhanced_generate_signals
    
    # Enable debug mode if verbose
    if args.verbose:
        signal_generator.enable_debug_mode()
    
    logger.info("Applied custom signal filters to signal generator")
    return signal_generator

def setup_dask_client(args):
    """
    Set up and configure Dask client with specified parameters.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dask client instance
    """
    try:
        from dask.distributed import Client, LocalCluster
        import dask
        
        # Configure Dask settings
        dask.config.set({
            'temporary-directory': args.dask_temp_dir or '/tmp/dask-temp',
            'distributed.worker.memory.target': 0.6,  # Target 60% memory usage
            'distributed.worker.memory.spill': 0.7,   # Spill to disk at 70%
            'distributed.worker.memory.pause': 0.8,   # Pause work at 80%
            'distributed.worker.memory.terminate': 0.95  # Emergency terminate at 95%
        })
        
        # Set up cluster with memory limits
        cluster = LocalCluster(
            n_workers=args.num_workers,
            threads_per_worker=1,  # Use processes for CPU-bound tasks
            memory_limit=args.dask_memory_limit,
            local_directory=args.dask_temp_dir
        )
        
        client = Client(cluster)
        logger.info(f"Dask client started with {args.num_workers} workers")
        logger.info(f"Dask dashboard available at: {client.dashboard_link}")
        
        return client
        
    except ImportError:
        logger.error("Dask not installed. Please install dask[distributed] to use this feature.")
        raise

def run_enhanced_backtest(args):
    """
    Run the enhanced backtest with optimized parameters.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Backtest results or None if error
    """
    start_time = datetime.now()
    logger.info("Starting enhanced backtest with improved win rate features")
    log_system_info()
    
    def monitor_memory(label):
        """Monitor memory usage and trigger GC if needed."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        logger.info(f"Memory usage ({label}): {memory_usage_mb:.2f} MB (Available: {available_mb:.2f} MB)")
        
        # Trigger garbage collection if memory usage is high
        if memory_usage_mb > 3000:  # 3GB threshold
            logger.info("High memory usage detected, triggering garbage collection")
            gc.collect()
            # Log memory after GC
            memory_after = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory after GC: {memory_after:.2f} MB (Freed: {memory_usage_mb - memory_after:.2f} MB)")
    
    monitor_memory("Start of backtest")
    
    # Initialize Dask if enabled
    dask_client = None
    if args.use_dask:
        try:
            dask_client = setup_dask_client(args)
            logger.info("Dask initialization successful")
            monitor_memory("After Dask init")
        except Exception as e:
            logger.error(f"Failed to initialize Dask: {e}")
            logger.warning("Falling back to standard processing")
            args.use_dask = False
    
    # Adjust chunk size when using Dask
    chunk_size = 25000 if args.use_dask else args.chunk_size
    
    # Initialize data loader with Dask settings
    data_loader = EnhancedDataLoader(
        start_date=args.start_date,
        end_date=args.end_date,
        num_workers=args.num_workers,
        chunksize=chunk_size,
        use_dask=args.use_dask
    )
    
    # Track performance metrics
    timings = {}
    
    # Step 1: Load market data
    data_load_start = time.time()
    
    logger.info("Loading market data")
    
    # Load spot and futures data normally
    spot_data = data_loader.load_spot_data(args.spot_data)
    futures_data = data_loader.load_futures_data(args.futures_data)
    
    # Check if we have the necessary base data
    if spot_data.empty or futures_data.empty:
        logger.error("Failed to load necessary spot or futures data for the specified date range")
        print("\nBacktest failed. Check the logs for errors.")
        print("\nRecommendations:")
        print("1. Check spot and futures data availability for the selected period")
        print("2. Verify the file paths are correct")
        print("3. Use a different date range if data is not available")
        return None
    
    # Load options data with improved memory management
    logger.info("Loading options data with optimized memory management")
    try:
        # Dynamically determine optimal chunk size based on available memory
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # Available memory in MB
        # Use 10% of available memory for each chunk, but cap at 100MB
        optimal_chunk_size = min(int(available_memory * 0.1), 100) * 1000
        # Ensure chunk size is at least 10,000 rows but not more than 100,000
        options_chunk_size = max(10000, min(optimal_chunk_size, 100000))
        
        logger.info(f"Using dynamic chunk size of {options_chunk_size} rows (based on {available_memory:.2f}MB available memory)")
        print(f"Using dynamic chunk size of {options_chunk_size} rows for efficient processing")
        
        # Load options data with progress reporting
        options_data = {}
        options_files = []
        
        # Get list of options files
        if os.path.isdir(args.options_data_dir):
            options_files = [os.path.join(args.options_data_dir, f) for f in os.listdir(args.options_data_dir) 
                           if f.endswith('.csv')]
        else:
            logger.error(f"Options data directory not found: {args.options_data_dir}")
            print(f"\nOptions data directory not found: {args.options_data_dir}")
            print("Please check the path and try again.")
            return None
        
        if not options_files:
            logger.error(f"No options data files found in {args.options_data_dir}")
            print(f"\nNo options data files found in {args.options_data_dir}")
            print("Please check the directory and try again.")
            return None
        
        logger.info(f"Found {len(options_files)} options data files")
        print(f"Found {len(options_files)} options data files to process")
        
        # Process each options file
        for i, file_path in enumerate(options_files):
            file_name = os.path.basename(file_path)
            logger.info(f"Processing options file ({i+1}/{len(options_files)}): {file_name}")
            print(f"\nLoading options file {i+1}/{len(options_files)}: {file_name}")
            
            try:
                # Get file size for progress reporting
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                logger.info(f"File size: {file_size:.2f} MB")
                print(f"File size: {file_size:.2f} MB")
                
                # Estimate number of chunks
                estimated_chunks = max(1, int(file_size / (options_chunk_size * 0.0001)))  # More accurate estimation
                
                # First pass: Quick scan to determine date range in file (read only datetime column)
                # This avoids loading chunks that don't contain data in our date range
                has_relevant_data = False
                date_sample_size = min(1000000, int(file_size * 10000))  # Sample size proportional to file size
                
                try:
                    # Read only datetime column with larger chunks for quick filtering
                    for date_chunk in pd.read_csv(file_path, 
                                                 usecols=['tr_datetime'], 
                                                 chunksize=date_sample_size,
                                                 parse_dates=['tr_datetime']):
                        
                        # Check if any data falls within our date range
                        if 'tr_datetime' in date_chunk.columns:
                            filtered = date_chunk[(date_chunk['tr_datetime'] >= args.start_date) & 
                                                (date_chunk['tr_datetime'] <= args.end_date)]
                            if not filtered.empty:
                                has_relevant_data = True
                                break
                    
                    if not has_relevant_data:
                        logger.info(f"File {file_name} contains no data in the specified date range, skipping")
                        print(f"File contains no data in the specified date range, skipping")
                        continue
                    
                except Exception as e:
                    # If pre-scan fails, proceed with normal loading
                    logger.warning(f"Error during date pre-scan: {e}, proceeding with full processing")
                    has_relevant_data = True
                
                # Process file in chunks to reduce memory usage
                chunk_count = 0
                total_rows = 0
                
                # Create a temporary DataFrame to hold the data for this file
                file_data = []
                
                # Define essential columns to read (optimize I/O)
                essential_cols = ['tr_datetime', 'tr_open', 'tr_high', 'tr_low', 'tr_close', 
                                 'otype', 'strike_price', 'open_interest', 'tr_volume']
                if 'expiry_date' in pd.read_csv(file_path, nrows=1).columns:
                    essential_cols.append('expiry_date')
                
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
                
                # Read the file in chunks
                for chunk in pd.read_csv(file_path, 
                                        usecols=essential_cols,
                                        dtype=dtype_dict,
                                        chunksize=options_chunk_size, 
                                        parse_dates=['tr_datetime']):
                    chunk_count += 1
                    
                    # Filter by date range
                    if 'tr_datetime' in chunk.columns:
                        chunk = chunk[(chunk['tr_datetime'] >= args.start_date) & 
                                     (chunk['tr_datetime'] <= args.end_date)]
                    
                    if not chunk.empty:
                        # Set index to tr_datetime
                        chunk.set_index('tr_datetime', inplace=True)
                        
                        # Optimize memory usage
                        chunk = optimize_dataframe_memory(chunk)
                        
                        # Append to file data
                        file_data.append(chunk)
                    
                    total_rows += len(chunk)
                    
                    # Report progress with a visual progress bar
                    progress_pct = min(100, int(chunk_count * 100 / estimated_chunks))
                    bar_length = 30
                    filled_length = int(bar_length * progress_pct / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    if chunk_count % 5 == 0 or chunk_count == estimated_chunks:
                        logger.info(f"Processed {chunk_count} chunks of {file_name}, {total_rows} rows so far")
                        print(f"\r  Progress: |{bar}| {progress_pct}% - {total_rows} rows", end='')
                        
                        # Monitor memory only occasionally to reduce overhead
                        if chunk_count % 20 == 0:
                            monitor_memory(f"After processing {chunk_count} chunks of {file_name}")
                
                # Print newline after progress bar
                print()
                
                # Combine all chunks for this file
                if file_data:
                    # Use a more memory-efficient concat
                    combined_data = pd.concat(file_data, copy=False)
                    logger.info(f"Combined {len(file_data)} chunks into DataFrame with {len(combined_data)} rows")
                    print(f"  Processed {len(combined_data)} rows of options data")
                    
                    # Clear file_data immediately to free memory
                    file_data.clear()
                    del file_data
                    gc.collect()
                    
                    # Extract expiry date from filename or data
                    expiry_date = None
                    
                    # Try to extract from filename (e.g., nifty_options_2022-01-06.csv)
                    filename_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_name)
                    if filename_match:
                        try:
                            expiry_date = pd.to_datetime(filename_match.group(1)).date()
                            logger.info(f"Extracted expiry date from filename: {expiry_date}")
                        except:
                            pass
                    
                    # If not found in filename, try to extract from data
                    if expiry_date is None and 'expiry_date' in combined_data.columns:
                        try:
                            # Get the most common expiry date
                            expiry_date = combined_data['expiry_date'].mode()[0]
                            if isinstance(expiry_date, str):
                                expiry_date = pd.to_datetime(expiry_date).date()
                            logger.info(f"Extracted expiry date from data: {expiry_date}")
                        except:
                            pass
                    
                    # If still not found, use the filename as the key
                    if expiry_date is None:
                        key = os.path.splitext(file_name)[0]
                        logger.warning(f"Could not extract expiry date, using filename as key: {key}")
                    else:
                        key = str(expiry_date)
                    
                    # Store in options data dictionary
                    options_data[key] = combined_data
                    logger.info(f"Added options data for {key} with {len(combined_data)} rows")
                    print(f"  Added options data for expiry: {key}")
                    
                    # Force memory cleanup after processing each file
                    monitor_memory(f"After adding {key} to options_data")
                else:
                    logger.warning(f"No valid data found in {file_name} for the specified date range")
                    print(f"  No valid data found in file for the specified date range")
            
            except Exception as e:
                logger.error(f"Error processing options file {file_name}: {e}")
                print(f"Error processing options file {file_name}: {e}")
                # Continue with other files
        
        # Check if we have any options data
        if not options_data:
            logger.error("No valid options data found for the specified date range")
            print("\nNo valid options data found for the specified date range.")
            print("Please check the options data files and date range.")
            return None
        
        # Check if we have enough data for a meaningful backtest
        total_rows = sum(len(df) for df in options_data.values())
        if total_rows < 1000:  # Arbitrary threshold for a meaningful backtest
            logger.warning(f"Only {total_rows} rows of options data found for the specified date range")
            print(f"\nWarning: Only {total_rows} rows of options data found for the specified date range.")
            print("This may not be enough for a meaningful backtest.")
            print("Consider using a different date range with more data.")
            
            # Ask for confirmation to continue
            if not args.auto_adjust_dates:
                response = input("Do you want to continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Backtest cancelled.")
                    return None
        
        logger.info(f"Successfully loaded options data for {len(options_data)} expiries with {total_rows} total rows")
        print(f"\nSuccessfully loaded options data for {len(options_data)} expiries with {total_rows:,} total rows")
        
        # Step 2: Fix options expiry dates
        logger.info("Fixing options expiry dates")
        print("Fixing options expiry dates...")
        fixed_options_data = fix_options_expiry_dates(options_data, pd.to_datetime(args.start_date))
        
        if not fixed_options_data:
            logger.error("No valid options data after fixing expiry dates")
            print("\nNo valid options data after fixing expiry dates.")
            print("Please check the options data files and expiry dates.")
            return None
                
        logger.info(f"Fixed options data: {len(fixed_options_data)} expiries")
        print(f"Fixed options data: {len(fixed_options_data)} expiries")

        # Downsample large options datasets
        print("\nOptimizing options data size for efficient processing...")
        total_expiries = len(fixed_options_data)
        for idx, (expiry, data) in enumerate(fixed_options_data.items()):
            if len(data) > args.max_rows_per_expiry:
                logger.info(f"Downsampling options data for expiry {expiry} from {len(data):,} to {args.max_rows_per_expiry:,} rows")
                print(f"\r  Optimizing expiry {idx+1}/{total_expiries}: {expiry} ({len(data):,} → {args.max_rows_per_expiry:,} rows)", end='')
                
                # IMPROVED DOWNSAMPLING: Use intelligent time-based stratified sampling
                try:
                    # Step 1: Ensure the data is sorted by timestamp
                    data = data.sort_index()
                    original_len = len(data)
                    
                    # Step 2: Always keep market open/close data points
                    market_open_time = datetime.strptime(MARKET_OPEN_TIME, "%H:%M:%S").time()
                    market_close_time = datetime.strptime(MARKET_CLOSE_TIME, "%H:%M:%S").time()
                    
                    # Find rows near market open and close
                    open_rows = data.index.indexer_between_time(
                        (market_open_time.hour, market_open_time.minute), 
                        (market_open_time.hour, market_open_time.minute + 5)
                    )
                    close_rows = data.index.indexer_between_time(
                        (market_close_time.hour - 1, market_close_time.minute), 
                        (market_close_time.hour, market_close_time.minute)
                    )
                    
                    # Step 3: Identify rows with significant price movements
                    significant_moves = []
                    if 'tr_close' in data.columns:
                        # Calculate price changes more efficiently
                        price_changes = data['tr_close'].pct_change().abs()
                        
                        # Find significant price movements (top 10% of changes)
                        significant_threshold = np.percentile(price_changes.dropna(), 90)
                        significant_moves = price_changes[price_changes > significant_threshold].index
                        
                        logger.debug(f"Found {len(significant_moves)} significant price movements for {expiry}")
                    
                    # Step 4: Stratify by time of day to ensure even coverage
                    # Divide trading day into segments and sample from each
                    trading_segments = 8  # Divide day into 8 segments
                    
                    # Create a mask of rows to keep
                    rows_to_keep = set()
                    
                    # Always include market open/close and significant moves
                    rows_to_keep.update([data.index[i] for i in open_rows if i < len(data)])
                    rows_to_keep.update([data.index[i] for i in close_rows if i < len(data)])
                    rows_to_keep.update(significant_moves)
                    
                    # Calculate how many more rows we need after keeping important points
                    remaining_budget = args.max_rows_per_expiry - len(rows_to_keep)
                    
                    if remaining_budget > 0:
                        # Group by date and time segment more efficiently
                        data['hour'] = data.index.hour
                        segments = pd.cut(data['hour'], bins=trading_segments)
                        grouped = data.groupby([data.index.date, segments])
                        
                        # Calculate samples per group, ensuring at least 1 sample per group if possible
                        groups_count = len(grouped)
                        if groups_count > 0:
                            samples_per_group = max(1, remaining_budget // groups_count)
                            
                            # Sample from each group
                            for _, group in grouped:
                                if len(group) <= samples_per_group:
                                    # Keep all rows if group is smaller than samples_per_group
                                    rows_to_keep.update(group.index)
                                else:
                                    # Systematic sampling within the group
                                    step = max(1, len(group) // samples_per_group)
                                    sampled_indices = group.index[::step][:samples_per_group]
                                    rows_to_keep.update(sampled_indices)
                    
                    # Final sampling: if we still have too many rows, do a final systematic sampling
                    rows_to_keep = sorted(rows_to_keep)
                    if len(rows_to_keep) > args.max_rows_per_expiry:
                        final_step = len(rows_to_keep) // args.max_rows_per_expiry + 1
                        rows_to_keep = rows_to_keep[::final_step][:args.max_rows_per_expiry]
                    
                    # Create the downsampled dataframe
                    downsampled_data = data.loc[rows_to_keep].sort_index()
                    
                    # Update the options data with the downsampled version
                    fixed_options_data[expiry] = downsampled_data
                    
                    # Log the reduction
                    reduction_pct = (1 - len(downsampled_data) / original_len) * 100
                    logger.info(f"Reduced options data for {expiry} by {reduction_pct:.1f}% ({original_len:,} → {len(downsampled_data):,} rows)")
                    
                except Exception as e:
                    logger.error(f"Error downsampling options data for {expiry}: {e}")
                    # Keep the original data if downsampling fails
            else:
                # Just print progress for expiries that don't need downsampling
                print(f"\r  Optimizing expiry {idx+1}/{total_expiries}: {expiry} (already optimal: {len(data):,} rows)", end='')
        
        # Print newline after progress
        print()
        
        # Calculate total rows after downsampling
        total_rows_after = sum(len(df) for df in fixed_options_data.values())
        logger.info(f"Total options data after downsampling: {total_rows_after:,} rows")
        print(f"Total options data size after optimization: {total_rows_after:,} rows")
        
        # Step 3: Calculate Greeks if needed
        if not args.skip_greeks:
            logger.info("Calculating Greeks for options data")
            print("\nCalculating Greeks for options data...")
            
            # Import the GreeksCalculator from the proper module
            from src.indicators.greeks import GreeksCalculator
            
            # Initialize the GreeksCalculator
            greeks_calculator = GreeksCalculator()
            
            # Patch the calculate_greeks method to handle Series inputs
            original_calculate_greeks = greeks_calculator.calculate_greeks
            
            def patched_calculate_greeks(self, spot_price, strike_price, time_to_expiry, implied_vol, option_type):
                """Patched version of calculate_greeks that handles Series inputs."""
                # Ensure all inputs are scalar values
                if isinstance(spot_price, pd.Series):
                    spot_price = float(spot_price.iloc[0])
                if isinstance(strike_price, pd.Series):
                    strike_price = float(strike_price.iloc[0])
                if isinstance(time_to_expiry, pd.Series):
                    time_to_expiry = float(time_to_expiry.iloc[0])
                if isinstance(implied_vol, pd.Series):
                    implied_vol = float(implied_vol.iloc[0])
                if isinstance(option_type, pd.Series):
                    option_type = str(option_type.iloc[0])
                    
                # Convert to proper types
                spot_price = float(spot_price)
                strike_price = float(strike_price)
                time_to_expiry = float(time_to_expiry)
                implied_vol = float(implied_vol)
                option_type = str(option_type)
                
                # Call the original method with sanitized inputs
                try:
                    return original_calculate_greeks(self, spot_price, strike_price, time_to_expiry, implied_vol, option_type)
                except Exception as e:
                    logger.error(f"Error in patched calculate_greeks: {e}")
                    # Return default values if calculation fails
                    return {
                        'delta': 0.0,
                        'gamma': 0.0,
                        'theta': 0.0,
                        'vega': 0.0,
                        'rho': 0.0,
                        'implied_volatility': implied_vol
                    }
            
            # Apply the patch
            greeks_calculator.calculate_greeks = types.MethodType(patched_calculate_greeks, greeks_calculator)
            
            # Also patch the estimate_implied_volatility method
            original_estimate_iv = greeks_calculator.estimate_implied_volatility
            
            def patched_estimate_iv(self, option_price, spot_price, strike_price, time_to_expiry, option_type):
                """Patched version of estimate_implied_volatility that handles Series inputs."""
                # Ensure all inputs are scalar values
                if isinstance(option_price, pd.Series):
                    option_price = float(option_price.iloc[0])
                if isinstance(spot_price, pd.Series):
                    spot_price = float(spot_price.iloc[0])
                if isinstance(strike_price, pd.Series):
                    strike_price = float(strike_price.iloc[0])
                if isinstance(time_to_expiry, pd.Series):
                    time_to_expiry = float(time_to_expiry.iloc[0])
                if isinstance(option_type, pd.Series):
                    option_type = str(option_type.iloc[0])
                    
                # Convert to proper types
                option_price = float(option_price)
                spot_price = float(spot_price)
                strike_price = float(strike_price)
                time_to_expiry = float(time_to_expiry)
                option_type = str(option_type)
                
                # Call the original method with sanitized inputs
                try:
                    return original_estimate_iv(self, option_price, spot_price, strike_price, time_to_expiry, option_type)
                except Exception as e:
                    logger.warning(f"Error in patched estimate_implied_volatility: {e}, using default of 0.3")
                    return 0.3
            
            # Apply the patch
            greeks_calculator.estimate_implied_volatility = types.MethodType(patched_estimate_iv, greeks_calculator)
            
            # Preprocess options data for Greeks calculation
            logger.info("Preprocessing options data for Greeks calculation")
            preprocessed_options = preprocess_options_for_greeks(fixed_options_data)
            
            # Calculate Greeks with progress reporting
            options_with_greeks = {}
            total_expiries = len(preprocessed_options)
            
            for idx, (expiry, data) in enumerate(preprocessed_options.items()):
                try:
                    print(f"\r  Calculating Greeks for expiry {idx+1}/{total_expiries}: {expiry}", end='')
                    
                    # Get the spot price for this expiry's data
                    current_datetime = data.index.min()
                    spot_price = None
                    
                    # Find the closest spot price to the current datetime
                    if not spot_data.empty:
                        closest_idx = spot_data.index.get_indexer([current_datetime], method='nearest')[0]
                        if closest_idx >= 0 and closest_idx < len(spot_data):
                            spot_price = spot_data['tr_close'].iloc[closest_idx]
                    
                    # If spot price is still None, use the last available price
                    if spot_price is None and not spot_data.empty:
                        spot_price = spot_data['tr_close'].iloc[-1]
                    
                    # If we still don't have a spot price, log an error and skip this expiry
                    if spot_price is None:
                        logger.error(f"Could not determine spot price for expiry {expiry}, skipping Greeks calculation")
                        options_with_greeks[expiry] = data
                        continue
                    
                    # Ensure spot_price is a scalar, not a Series
                    if isinstance(spot_price, pd.Series):
                        spot_price = float(spot_price.iloc[0])
                    
                    # Use the GreeksCalculator to process the options data
                    with_greeks = greeks_calculator.process_options_chain(
                        options_data=data,
                        spot_price=spot_price,
                        current_datetime=current_datetime,
                        expiry_date=pd.to_datetime(expiry)
                    )
                    options_with_greeks[expiry] = with_greeks
                    
                    logger.info(f"Calculated Greeks for {expiry} with {len(with_greeks)} rows")
                except Exception as e:
                    logger.error(f"Error calculating Greeks for {expiry}: {e}", exc_info=True)
                    # Use original data if Greeks calculation fails
                    options_with_greeks[expiry] = data
            
            # Print newline after progress
            print()
            
            logger.info(f"Completed Greeks calculation for {len(options_with_greeks)} expiries")
            print(f"Completed Greeks calculation for {len(options_with_greeks)} expiries")
            
            # Use the data with Greeks
            options_data_for_backtest = options_with_greeks
        else:
            logger.info("Skipping Greeks calculation as requested")
            print("Skipping Greeks calculation as requested")
            
            # Use the downsampled data without Greeks
            options_data_for_backtest = fixed_options_data
        
        # Step 4: Initialize strategy components
        strategy_start = time.time()
        
        # Calculate options sentiment for market regime adaptation
        if args.market_regime_adaptation:
            logger.info("Calculating options sentiment for market regime adaptation")
            print("\nCalculating options sentiment for market regime adaptation...")
            
            try:
                # Import the sentiment calculation functions from the proper module
                from src.indicators.sentiment import calculate_options_sentiment
                
                options_sentiment = calculate_options_sentiment(options_data_for_backtest)
                logger.info(f"Calculated options sentiment with {len(options_sentiment)} data points")
                print(f"Calculated options sentiment with {len(options_sentiment)} data points")
            except Exception as e:
                logger.error(f"Error calculating options sentiment: {e}")
                options_sentiment = None
        else:
            options_sentiment = None

    except Exception as e:
        logger.error(f"Error loading options data: {e}", exc_info=True)
        print(f"\nError loading options data: {str(e)}")
        print("Please check the options data files and try again.")
        return None
    
    # Step 3: Process indicators
    indicators_start = time.time()
    
    try:
        logger.info("Processing technical indicators")
        indicator_processor = ParallelIndicatorProcessor(num_workers=args.num_workers)
        
        # Process futures data indicators
        futures_data_with_indicators = indicator_processor.process_technical_indicators(futures_data)
        
        # Process options data
        if args.skip_greeks:
            logger.info("Skipping options Greeks calculation to save memory")
            options_data_with_greeks = fixed_options_data
            
            # Calculate basic options sentiment without Greeks
            options_sentiment = {
                'pcr': pd.DataFrame(),
                'oi_concentration': {},
                'oi_velocity': {}
            }
        else:
            logger.info("Calculating options Greeks")
            options_data_with_greeks = indicator_processor.process_options_greeks(fixed_options_data, spot_data)
            
            logger.info("Calculating options sentiment")
            options_sentiment = indicator_processor.process_options_sentiment(options_data_with_greeks)
        
        timings['indicator_processing'] = time.time() - indicators_start
        
        monitor_memory("After indicator processing")
        
    except Exception as e:
        logger.error(f"Error processing indicators: {e}", exc_info=True)
        # Continue with raw data if indicators fail
        futures_data_with_indicators = futures_data
        options_data_with_greeks = fixed_options_data
        options_sentiment = {}
    
    # Step 4: Initialize strategy components with improvements
    strategy_start = time.time()
    
    try:
        logger.info("Initializing strategy components with adaptive features")
        
        # Initialize signal generator with enhanced features
        signal_generator = SignalGenerator(
            options_sentiment=options_sentiment,
            mandatory_conditions=args.mandatory_conditions.split(',') if args.mandatory_conditions else None,
            min_conditions_required=args.require_conditions,
            weighted_scoring=args.weighted_scoring,
            market_regime_adaptation=args.market_regime_adaptation,
            optimal_strikes=args.optimal_strikes
        )
        
        # Apply custom signal filters
        signal_generator = apply_custom_signal_filters(signal_generator, args)
        
        # Initialize position manager with enhanced features
        position_manager = PositionManager(
            adaptive_stop_loss=args.adaptive_stop_loss,
            trailing_stop=args.trailing_stop,
            time_based_targets=args.time_based_targets
        )
        
        risk_manager = RiskManager(args.capital)
        
        # Initialize trading engine with optimized settings
        trading_engine = TradingEngine(
            signal_generator, position_manager, risk_manager, 
            backtest_mode=True,
            max_trades_per_day=args.max_trades_per_day
        )
        
        # Set trading engine reference in signal generator for market regime adaptation
        signal_generator.trading_engine = trading_engine
        
        # Set chunk size for efficient processing
        trading_engine.chunk_size = chunk_size
        
        # Apply patches to fix encoding issues
        patch_trading_engine_methods(trading_engine)
        
        timings['strategy_initialization'] = time.time() - strategy_start
        
        monitor_memory("After strategy initialization")
        
    except Exception as e:
        logger.error(f"Error initializing strategy components: {e}", exc_info=True)
        return None
    
    # Step 5: Run backtest
    backtest_start = time.time()
    monitor_memory("Before backtest execution")
    
    try:
        logger.info("Running backtest")
        
        # Free memory before running the backtest
        del options_data
        del fixed_options_data
        gc.collect()
        monitor_memory("After memory cleanup")
        
        # Enhance the trading engine process_timestamp_chunk method to apply strike filtering
        original_process_snapshot = trading_engine.process_data_snapshot
        
        def enhanced_process_snapshot(self, timestamp, spot_data, futures_data, options_data):
            """
            Enhanced data snapshot processor with improved memory management and error handling.
            
            Args:
                timestamp: Current timestamp
                spot_data: Spot market data
                futures_data: Futures market data
                options_data: Options market data by expiry
                
            Returns:
                Result from original process_snapshot with filtered options
            """
            # Check if spot data is empty and try to fill it from futures data if possible
            if spot_data.empty and not futures_data.empty:
                logger.debug(f"Filling empty spot data at {timestamp} with futures data")
                
                # Determine the column names in futures_data
                futures_columns = futures_data.columns.tolist()
                
                # Map standard column names to possible alternatives
                column_mapping = {
                    'tr_open': ['tr_open', 'open', 'Open', 'OPEN'],
                    'tr_high': ['tr_high', 'high', 'High', 'HIGH'],
                    'tr_low': ['tr_low', 'low', 'Low', 'LOW'],
                    'tr_close': ['tr_close', 'close', 'Close', 'CLOSE']
                }
                
                # Find the actual column names in the futures data
                actual_columns = {}
                for standard_col, alternatives in column_mapping.items():
                    for alt_col in alternatives:
                        if alt_col in futures_columns:
                            actual_columns[standard_col] = alt_col
                            break
                    if standard_col not in actual_columns:
                        logger.warning(f"Could not find a suitable column for {standard_col} in futures data")
                        # Default to the last column as a fallback
                        if futures_columns:
                            actual_columns[standard_col] = futures_columns[-1]
                
                # Create a copy of spot data with futures price
                try:
                    spot_data = pd.DataFrame({
                        'tr_open': [futures_data[actual_columns.get('tr_open', futures_columns[-1])].iloc[-1]],
                        'tr_high': [futures_data[actual_columns.get('tr_high', futures_columns[-1])].iloc[-1]],
                        'tr_low': [futures_data[actual_columns.get('tr_low', futures_columns[-1])].iloc[-1]],
                        'tr_close': [futures_data[actual_columns.get('tr_close', futures_columns[-1])].iloc[-1]]
                    }, index=[timestamp])
                except Exception as e:
                    logger.error(f"Error creating spot data from futures: {e}")
                    # Create a simple DataFrame with a single value as a last resort
                    if len(futures_data.columns) > 0:
                        value = futures_data.iloc[-1, 0]  # Use the first column's last value
                        spot_data = pd.DataFrame({
                            'tr_open': [value],
                            'tr_high': [value],
                            'tr_low': [value],
                            'tr_close': [value]
                        }, index=[timestamp])
            
            # Continue with the original enhanced processing
            if args.strikes_range > 0 and not futures_data.empty:
                try:
                    # Get current price from futures data
                    futures_columns = futures_data.columns.tolist()
                    price_column = None
                    
                    # Try to find a suitable price column
                    for col in ['tr_close', 'close', 'Close', 'CLOSE', 'price', 'Price']:
                        if col in futures_columns:
                            price_column = col
                            break
                    
                    if price_column is None and futures_columns:
                        # Use the last column as a fallback
                        price_column = futures_columns[-1]
                        
                    current_price = futures_data[price_column].iloc[-1]
                    
                    # Filter options to include only strikes within range
                    filtered_options = {}
                    
                    for expiry, expiry_data in options_data.items():
                        if expiry_data.empty:
                            filtered_options[expiry] = expiry_data
                            continue
                            
                        # Find the strike column
                        strike_column = None
                        for col in ['strike', 'strike_price', 'Strike', 'STRIKE']:
                            if col in expiry_data.columns:
                                strike_column = col
                                break
                        
                        if strike_column is None:
                            logger.warning(f"Could not find strike column in options data for expiry {expiry}")
                            filtered_options[expiry] = expiry_data
                            continue
                            
                        # Get unique strikes
                        try:
                            strikes = expiry_data[strike_column].unique()
                            
                            # Find closest strikes to current price
                            closest_strikes = sorted(strikes, key=lambda x: abs(float(x) - current_price))
                            
                            # Take N strikes above and below current price
                            selected_strikes = closest_strikes[:args.strikes_range * 2]
                            
                            # Filter data to include only selected strikes
                            filtered_data = expiry_data[expiry_data[strike_column].isin(selected_strikes)]
                            
                            filtered_options[expiry] = filtered_data
                        except Exception as e:
                            logger.error(f"Error processing strikes for expiry {expiry}: {e}")
                            filtered_options[expiry] = expiry_data
                    
                    # Process with filtered options
                    return original_process_snapshot(timestamp, spot_data, futures_data, filtered_options)
                    
                except Exception as e:
                    logger.error(f"Error in enhanced_process_snapshot: {e}")
                    # Fall back to original method with unfiltered data
                    return original_process_snapshot(timestamp, spot_data, futures_data, options_data)
            else:
                return original_process_snapshot(timestamp, spot_data, futures_data, options_data)
        
        # Replace the method
        trading_engine.process_data_snapshot = types.MethodType(enhanced_process_snapshot, trading_engine)
        
        # Apply patches to fix encoding issues
        patch_trading_engine_methods(trading_engine)
        
        backtest_results = trading_engine.run_backtest(
            spot_data, futures_data_with_indicators, options_data_with_greeks
        )
        
        timings['backtest_execution'] = time.time() - backtest_start
        
        monitor_memory("After backtest execution")
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        return None
    
    # Step 6: Generate reports
    reporting_start = time.time()
    
    try:
        logger.info("Generating backtest reports")
        report_generator = ReportGenerator(args.output_dir)
        
        # Add backtest configuration to results
        backtest_results['configuration'] = {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'capital': args.capital,
            'require_conditions': args.require_conditions,
            'mandatory_conditions': args.mandatory_conditions,
            'strikes_range': args.strikes_range,
            'max_signals_per_timestamp': args.max_signals_per_timestamp
        }
        
        # Generate Excel report
        report_path = report_generator.generate_backtest_report(backtest_results)
        
        # Generate trade log CSV
        trades = backtest_results.get('trades', [])
        csv_path = report_generator.generate_trade_log_csv(trades)
        
        # Generate debug report if requested
        if args.debug_report:
            debug_report = report_generator.generate_debug_report(
                spot_data, futures_data, options_data_with_greeks, vars(args)
            )
            logger.info(f"Generated debug report: {debug_report}")
        
        timings['report_generation'] = time.time() - reporting_start
        
        monitor_memory("After report generation")
        
    except Exception as e:
        logger.error(f"Error generating reports: {e}", exc_info=True)
        report_path = None
    
    # Calculate total execution time
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    logger.info(f"Backtest completed in {execution_time:.2f} seconds")
    
    # Log performance breakdown
    logger.info("Performance breakdown:")
    for stage, duration in timings.items():
        percentage = (duration / execution_time) * 100
        logger.info(f"  {stage}: {duration:.2f} seconds ({percentage:.1f}%)")
    
    # Clean up Dask client if used
    if dask_client is not None:
        try:
            dask_client.close()
            logger.info("Dask client closed successfully")
        except Exception as e:
            logger.warning(f"Error closing Dask client: {e}")

    # Return key results
    return {
        'status': 'completed',
        'num_trades': backtest_results.get('num_trades', 0),
        'win_rate': backtest_results.get('win_rate', 0),
        'total_pnl': backtest_results.get('total_pnl', 0),
        'max_drawdown': backtest_results.get('max_drawdown', 0),
        'report_path': report_path,
        'execution_time': execution_time,
        'performance_metrics': timings
    }

def main():
    """Main entry point for the enhanced backtest."""
    args = parse_args()
    
    try:
        # Run the backtest with the parsed arguments
        results = run_enhanced_backtest(args)
        
        # Print execution time
        elapsed_time = time.time() - start_time
        print(f"\nBacktest completed in {elapsed_time:.2f} seconds")
        
        # Print report location
        if 'report_path' in results:
            print(f"\nDetailed report saved to: {results['report_path']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError running backtest: {e}")
        logger.error(f"Error running backtest: {e}", exc_info=True)
        return 1

def optimize_dataframe_memory(df):
    """
    Optimize memory usage of a DataFrame by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Memory-optimized DataFrame
    """
    if df is None or df.empty:
        return df
        
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Downcast numeric columns
    for col in result.select_dtypes(include=['int']).columns:
        result[col] = pd.to_numeric(result[col], downcast='integer')
        
    for col in result.select_dtypes(include=['float']).columns:
        result[col] = pd.to_numeric(result[col], downcast='float')
    
    # Convert object columns to categories if they have few unique values
    for col in result.select_dtypes(include=['object']).columns:
        if result[col].nunique() < len(result) * 0.5:  # If less than 50% unique values
            result[col] = result[col].astype('category')
    
    return result

def patch_trading_engine_methods(trading_engine):
    """
    Patch trading engine methods to fix encoding issues and improve memory management.
    
    Args:
        trading_engine: TradingEngine instance to patch
    """
    # Store original methods
    original_handle_day_transition = trading_engine._handle_day_transition
    original_process_data_snapshot = trading_engine.process_data_snapshot
    
    # Define patched method for day transition
    def patched_handle_day_transition(self, new_timestamp, previous_timestamp):
        """Patched method to handle day transition without Unicode issues."""
        prev_date = previous_timestamp.date() if previous_timestamp else "None"
        new_date = new_timestamp.date()
        
        # Use ASCII arrow instead of Unicode
        logger.info(f"=== DAY TRANSITION: {prev_date} -> {new_date} ===")
        
        # Call the rest of the original method
        result = original_handle_day_transition(new_timestamp, previous_timestamp)
        return result
    
    # Define patched method for data snapshot processing
    def patched_process_data_snapshot(self, timestamp, spot_data, futures_data, options_data):
        """
        Patched method to handle empty spot data by using futures data as a fallback.
        
        Args:
            timestamp: Current timestamp
            spot_data: Spot market data
            futures_data: Futures market data
            options_data: Options market data by expiry
            
        Returns:
            Result from original process_data_snapshot with proper data handling
        """
        # Check if spot data is empty and try to fill it from futures data if possible
        if spot_data.empty and not futures_data.empty:
            logger.debug(f"Filling empty spot data at {timestamp} with futures data")
            
            # Determine the column names in futures_data
            futures_columns = futures_data.columns.tolist()
            
            # Map standard column names to possible alternatives
            column_mapping = {
                'tr_open': ['tr_open', 'open', 'Open', 'OPEN'],
                'tr_high': ['tr_high', 'high', 'High', 'HIGH'],
                'tr_low': ['tr_low', 'low', 'Low', 'LOW'],
                'tr_close': ['tr_close', 'close', 'Close', 'CLOSE']
            }
            
            # Find the actual column names in the futures data
            actual_columns = {}
            for standard_col, alternatives in column_mapping.items():
                for alt_col in alternatives:
                    if alt_col in futures_columns:
                        actual_columns[standard_col] = alt_col
                        break
                if standard_col not in actual_columns:
                    logger.warning(f"Could not find a suitable column for {standard_col} in futures data")
                    # Default to the last column as a fallback
                    if futures_columns:
                        actual_columns[standard_col] = futures_columns[-1]
            
            # Create a copy of spot data with futures price
            try:
                spot_data = pd.DataFrame({
                    'tr_open': [futures_data[actual_columns.get('tr_open', futures_columns[-1])].iloc[-1]],
                    'tr_high': [futures_data[actual_columns.get('tr_high', futures_columns[-1])].iloc[-1]],
                    'tr_low': [futures_data[actual_columns.get('tr_low', futures_columns[-1])].iloc[-1]],
                    'tr_close': [futures_data[actual_columns.get('tr_close', futures_columns[-1])].iloc[-1]]
                }, index=[timestamp])
            except Exception as e:
                logger.error(f"Error creating spot data from futures: {e}")
                # Create a simple DataFrame with a single value as a last resort
                if len(futures_data.columns) > 0:
                    value = futures_data.iloc[-1, 0]  # Use the first column's last value
                    spot_data = pd.DataFrame({
                        'tr_open': [value],
                        'tr_high': [value],
                        'tr_low': [value],
                        'tr_close': [value]
                    }, index=[timestamp])
            
            # Continue with the original enhanced processing
            if args.strikes_range > 0 and not futures_data.empty:
                try:
                    # Get current price from futures data
                    futures_columns = futures_data.columns.tolist()
                    price_column = None
                    
                    # Try to find a suitable price column
                    for col in ['tr_close', 'close', 'Close', 'CLOSE', 'price', 'Price']:
                        if col in futures_columns:
                            price_column = col
                            break
                    
                    if price_column is None and futures_columns:
                        # Use the last column as a fallback
                        price_column = futures_columns[-1]
                        
                    current_price = futures_data[price_column].iloc[-1]
                    
                    # Filter options to include only strikes within range
                    filtered_options = {}
                    
                    for expiry, expiry_data in options_data.items():
                        if expiry_data.empty:
                            filtered_options[expiry] = expiry_data
                            continue
                            
                        # Find the strike column
                        strike_column = None
                        for col in ['strike', 'strike_price', 'Strike', 'STRIKE']:
                            if col in expiry_data.columns:
                                strike_column = col
                                break
                        
                        if strike_column is None:
                            logger.warning(f"Could not find strike column in options data for expiry {expiry}")
                            filtered_options[expiry] = expiry_data
                            continue
                            
                        # Get unique strikes
                        try:
                            strikes = expiry_data[strike_column].unique()
                            
                            # Find closest strikes to current price
                            closest_strikes = sorted(strikes, key=lambda x: abs(float(x) - current_price))
                            
                            # Take N strikes above and below current price
                            selected_strikes = closest_strikes[:args.strikes_range * 2]
                            
                            # Filter data to include only selected strikes
                            filtered_data = expiry_data[expiry_data[strike_column].isin(selected_strikes)]
                            
                            filtered_options[expiry] = filtered_data
                        except Exception as e:
                            logger.error(f"Error processing strikes for expiry {expiry}: {e}")
                            filtered_options[expiry] = expiry_data
                    
                    # Process with filtered options
                    return original_process_data_snapshot(timestamp, spot_data, futures_data, filtered_options)
                    
                except Exception as e:
                    logger.error(f"Error in enhanced_process_snapshot: {e}")
                    # Fall back to original method with unfiltered data
                    return original_process_data_snapshot(timestamp, spot_data, futures_data, options_data)
            else:
                return original_process_data_snapshot(timestamp, spot_data, futures_data, options_data)
        
        # Replace the method
        trading_engine.process_data_snapshot = types.MethodType(patched_process_data_snapshot, trading_engine)
        
        # Apply patches to fix encoding issues
        patch_trading_engine_methods(trading_engine)
        
        backtest_results = trading_engine.run_backtest(
            spot_data, futures_data_with_indicators, options_data_with_greeks
        )
        
        timings['backtest_execution'] = time.time() - backtest_start
        
        monitor_memory("After backtest execution")
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        return None
    
    # Step 6: Generate reports
    reporting_start = time.time()
    
    try:
        logger.info("Generating backtest reports")
        report_generator = ReportGenerator(args.output_dir)
        
        # Add backtest configuration to results
        backtest_results['configuration'] = {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'capital': args.capital,
            'require_conditions': args.require_conditions,
            'mandatory_conditions': args.mandatory_conditions,
            'strikes_range': args.strikes_range,
            'max_signals_per_timestamp': args.max_signals_per_timestamp
        }
        
        # Generate Excel report
        report_path = report_generator.generate_backtest_report(backtest_results)
        
        # Generate trade log CSV
        trades = backtest_results.get('trades', [])
        csv_path = report_generator.generate_trade_log_csv(trades)
        
        # Generate debug report if requested
        if args.debug_report:
            debug_report = report_generator.generate_debug_report(
                spot_data, futures_data, options_data_with_greeks, vars(args)
            )
            logger.info(f"Generated debug report: {debug_report}")
        
        timings['report_generation'] = time.time() - reporting_start
        
        monitor_memory("After report generation")
        
    except Exception as e:
        logger.error(f"Error generating reports: {e}", exc_info=True)
        report_path = None
    
    # Calculate total execution time
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    logger.info(f"Backtest completed in {execution_time:.2f} seconds")
    
    # Log performance breakdown
    logger.info("Performance breakdown:")
    for stage, duration in timings.items():
        percentage = (duration / execution_time) * 100
        logger.info(f"  {stage}: {duration:.2f} seconds ({percentage:.1f}%)")
    
    # Clean up Dask client if used
    if dask_client is not None:
        try:
            dask_client.close()
            logger.info("Dask client closed successfully")
        except Exception as e:
            logger.warning(f"Error closing Dask client: {e}")

    # Return key results
    return {
        'status': 'completed',
        'num_trades': backtest_results.get('num_trades', 0),
        'win_rate': backtest_results.get('win_rate', 0),
        'total_pnl': backtest_results.get('total_pnl', 0),
        'max_drawdown': backtest_results.get('max_drawdown', 0),
        'report_path': report_path,
        'execution_time': execution_time,
        'performance_metrics': timings
    }

def preprocess_options_for_greeks(options_data):
    """
    Preprocess options data to ensure it's in the right format for Greeks calculation.
    
    Args:
        options_data: Dictionary of options data by expiry
        
    Returns:
        Preprocessed options data ready for Greeks calculation
    """
    processed_data = {}
    for expiry, data in options_data.items():
        if data.empty:
            logger.warning(f"Skipping empty DataFrame for expiry {expiry}")
            continue
            
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.warning(f"Could not convert index to datetime for expiry {expiry}: {e}")
        
        # Ensure all required columns are present
        required_cols = ['tr_close', 'strike_price', 'otype']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns {missing_cols} for expiry {expiry}, skipping")
            continue
        
        # Fix all columns that might contain Series instead of scalar values
        for col in required_cols:
            if col in df.columns and len(df) > 0 and isinstance(df[col].iloc[0], pd.Series):
                logger.warning(f"{col} column contains Series instead of scalar values for expiry {expiry}, fixing")
                df[col] = df[col].apply(lambda x: x.iloc[0] if isinstance(x, pd.Series) and not x.empty else x)
        
        # Convert numeric columns to float
        numeric_cols = ['tr_close', 'strike_price']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN values with a reasonable default
                    if df[col].isna().any():
                        logger.warning(f"Found NaN values in {col} for expiry {expiry}, filling with defaults")
                        if col == 'tr_close':
                            df[col] = df[col].fillna(df[col].mean())
                        elif col == 'strike_price':
                            df[col] = df[col].fillna(df[col].mean())
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric for expiry {expiry}: {e}")
        
        # Ensure otype is a string
        if 'otype' in df.columns:
            df['otype'] = df['otype'].astype(str)
            # Standardize option type values
            df['otype'] = df['otype'].apply(lambda x: 'CE' if x.upper() in ['CE', 'C', 'CALL'] else 'PE' if x.upper() in ['PE', 'P', 'PUT'] else x)
        
        processed_data[expiry] = df
    
    return processed_data

if __name__ == "__main__":
    main()