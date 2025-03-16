"""
Global settings for the options trading system.
"""
import os
import logging
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, 'data')
INPUT_DATA_DIR = os.path.join(DATA_DIR, 'input')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DATA_DIR = os.path.join(DATA_DIR, 'output')

# Ensure directories exist
for directory in [DATA_DIR, INPUT_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data input paths
SPOT_DATA_DIR = os.path.join(INPUT_DATA_DIR, 'spot')
FUTURES_DATA_DIR = os.path.join(INPUT_DATA_DIR, 'futures')
OPTIONS_DATA_DIR = os.path.join(INPUT_DATA_DIR, 'options')

# Logging configuration
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'trading_system.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Market hours
MARKET_OPEN_TIME = "09:15:00"
MARKET_CLOSE_TIME = "15:30:00"

# Calculation settings
VWAP_STD_DEV_PERIODS = 2  # Number of standard deviations for VWAP bands

# Data processing settings
RESAMPLE_INTERVAL = '1min'  # Instead of '1T'