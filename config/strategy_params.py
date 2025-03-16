"""
Strategy-specific parameters for the options trading system.
"""

# Capital allocation
TOTAL_CAPITAL = 2000000  # 20 Lakhs
TRADE_CAPITAL = 100000   # 1 Lakh per trade
MAX_CONCURRENT_TRADES = 3

# Contract specifications
NIFTY_LOT_SIZE = 75  # Standard Nifty lot size

# Risk management
INITIAL_STOP_LOSS_PERCENT = 0.20  # 20% of trade capital
DAILY_LOSS_LIMIT = 40000  # Maximum loss allowed per day

# Position management
PROFIT_TARGET_1 = 0.30  # First profit target at 30% gain
SIZE_REDUCTION_1 = 0.50  # Reduce position by 50% at first target
PROFIT_TARGET_2 = 0.70  # Second profit target at 70% gain
SIZE_REDUCTION_2 = 0.50  # Reduce remaining position by 50% at second target
EXIT_TIME = "15:00:59"  # Exit all positions by this time
EXPIRY_DAY_CUTOFF_TIME = "12:00:59"  # Cutoff time for trading on expiry day

# Kelly criterion parameters
KELLY_FRACTION = 0.5  # Half-Kelly for conservative sizing

# Delta parameters
CALL_DELTA_THRESHOLD = 0.35  # Minimum delta for calls
PUT_DELTA_THRESHOLD = -0.35  # Maximum delta for puts

# Gamma parameters
GAMMA_THRESHOLD = 0.03  # Reduced from 0.05, increased from 0.025

# Open Interest parameters
OI_CHANGE_THRESHOLD = 0.035  # 3.5% OI change - balanced threshold
OI_PERCENTILE_THRESHOLD = 75  # Focus on strikes with OI >= 75th percentile
OI_CONCENTRATION_THRESHOLD = 0.5  # Threshold for OI concentration detection (new parameter)
OI_PRICE_DIVERGENCE_LOOKBACK = 5  # Periods to look back for OI-price divergence (new parameter)

# VWAP parameters
VWAP_BREAKOUT_THRESHOLD = 0.005  # Increased from 0.002 to 0.5%
VWAP_DURATION_PERIODS = 2  # New parameter: minimum periods for confirmed breakout

# CVD parameters
CVD_LOOKBACK_PERIOD = 5  # 5-minute lookback for CVD analysis
CVD_THRESHOLD = 1.5  # Increased from 0.75 for stronger directional signals

# Time-based exit parameters
LOSING_TRADE_MAX_DURATION = 20  # Maximum time to hold losing trades (minutes)

MIN_CONDITIONS_MET = 2

# Transaction costs parameters
BROKERAGE_PER_LOT = 20.0  # Flat fee per lot (₹)
EXCHANGE_CHARGES_RATE = 5.5 / 10000000  # ₹5.5 per crore of turnover
GST_RATE = 0.18  # 18% GST on (brokerage + exchange charges)
STAMP_DUTY_RATE = 0.00003  # 0.003% of premium
SEBI_CHARGES_RATE = 10.0 / 10000000  # ₹10 per crore (negligible)
STT_RATE_SELL = 0.0005  # 0.05% STT on selling
SLIPPAGE_PERCENT = 0.001  # 0.1% slippage on premium

# Implied Volatility parameters
IV_PERCENTILE_THRESHOLD = 30  # Focus on options with IV below this percentile
IV_LOOKBACK_PERIODS = 20  # Number of periods for IV percentile calculation

# Put-Call Ratio parameters
PCR_SHORT_MA = 5  # Short-term PCR moving average periods
PCR_LONG_MA = 120  # Longer-term PCR moving average periods for broader sentiment