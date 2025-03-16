# Options Trading System

## Overview

The Options Trading System is a comprehensive backtesting and trading platform designed for NIFTY index options trading. The system uses a multi-factor approach combining technical indicators, options Greeks, open interest analysis, and market sentiment to generate high-probability trading signals.

## Features

- **Multi-factor Signal Generation**: Combines price action, volatility metrics, and market sentiment
- **Advanced Position Management**: Adaptive stop loss, trailing stops, and time-based profit targets
- **Risk Management**: Per-trade and daily risk controls with Kelly criterion position sizing
- **Market Regime Adaptation**: Adjusts strategy rules based on trending, range-bound, or volatile markets
- **Transaction Cost Analysis**: Accurate modeling of all trading costs (brokerage, STT, GST, exchange charges)
- **Performance Reporting**: Detailed backtest reports, trade logs, and visualization
- **Memory-efficient Processing**: Handles large datasets with parallel processing and memory optimization

## Project Structure

```
options_trading_system/
│
├── main.py                      # Main entry point for the trading system
├── quick_backtest.py            # Simplified backtest script
├── enhanced_backtest.py         # Optimized backtest with improved features
├── diagnostics_tool.py          # Tool for diagnosing system issues
├── extract_weekly_expiries.py   # Utility to extract weekly expiry dates
├── requirements.txt             # Project dependencies
│
├── config/                      # Configuration settings
│   ├── settings.py              # Global settings and configurations
│   └── strategy_params.py       # Strategy-specific parameters
│
├── data/                        # Data directory
│   ├── input/                   # Raw input data (CSVs)
│   │   ├── spot/                # Spot market data
│   │   ├── futures/             # Futures market data
│   │   └── options/             # Options market data
│   ├── processed/               # Processed/calculated indicators
│   └── output/                  # Strategy results, reports, etc.
│
├── logs/                        # Log files directory
│
├── notebooks/                   # Jupyter notebooks
│   └── data_exploration.ipynb   # For data analysis and exploration
│
├── src/                         # Source code
    ├── backtest_date_validator.py  # Validates date ranges for backtests
    │
    ├── data_handlers/           # Data loading and preprocessing
    │   ├── data_loader.py       # Loads data from various sources
    │   └── data_synchronizer.py # Aligns data from different sources
    │
    ├── engine/                  # Trading engine components
    │   ├── position_manager.py  # Manages trading positions
    │   ├── risk_manager.py      # Handles risk management
    │   └── trading_engine.py    # Core trading logic
    │
    ├── indicators/              # Market indicators
    │   ├── greeks.py            # Options Greeks calculations
    │   ├── parallel_processor.py # Parallel processing for indicators
    │   ├── sentiment.py         # Sentiment indicators (PCR, OI, etc.)
    │   └── technical.py         # Technical indicators (VWAP, CVD, etc.)
    │
    ├── reporting/               # Reporting and visualization
    │   ├── report_generator.py  # Generates performance reports
    │   └── visualizations.py    # Creates performance charts
    │
    ├── strategy/                # Trading strategy implementation
    │   ├── rules.py             # Entry/exit rule conditions
    │   └── signal_generator.py  # Generates trading signals
    │
    └── utils/                   # Utility functions
        ├── date_utils.py        # Date handling utilities
        └── weekly_expiry_mapper.py  # Maps expiry dates for options
```

## Workflow

The system follows this general workflow:

1. **Data Loading**: 
   - Loads spot, futures, and options data from CSV files
   - Synchronizes data to ensure consistent timestamps
   - Validates expiry dates for options data

2. **Indicator Calculation**:
   - Calculates technical indicators on futures data (VWAP, CVD)
   - Calculates options Greeks (Delta, Gamma, Theta, Vega)
   - Processes options sentiment indicators (PCR, OI analysis)

3. **Signal Generation**:
   - Applies entry rules based on multiple conditions
   - Filters signals based on quality scores
   - Adapts to different market regimes

4. **Position Management**:
   - Manages position entry and exit
   - Implements adaptive stop losses
   - Uses trailing stops for profit protection
   - Adjusts profit targets based on time to expiry

5. **Risk Management**:
   - Applies position sizing using Kelly criterion
   - Enforces per-trade and daily risk limits
   - Tracks drawdown and performance metrics

6. **Reporting**:
   - Generates detailed backtest reports
   - Creates performance visualizations
   - Provides trade logs for analysis

## Data Requirements

The system requires the following data files:

1. **Spot Data**: 1-minute NIFTY spot price data with OHLC
2. **Futures Data**: 1-minute NIFTY futures data with OHLC, volume, and OI
3. **Options Data**: 1-minute NIFTY options data with OHLC, volume, OI, strike price, and option type
4. **Weekly Expiries**: List of weekly expiry dates for accurate options mapping

### Sample Data Format

#### Spot Data
```
ticker tr_date tr_time tr_open tr_high tr_low tr_close tr_segment stock_name
NIFTY 50.NSE_IDX 2021-08-27 09:15:59 16642.55 16652.7 16630.45 16630.45 4 NIFTY
```

#### Futures Data
```
tr_datetime tr_date tr_time tr_open tr_high tr_low tr_close open_interest stock_name ticker expiry_date tr_volume
2020-01-01 09:15:59 2020-01-01 09:15:59 12253.95 12266.75 12252.7 12265.3 12263025 NIFTY NIFTY-I.NFO 2020-01-30 106350
```

#### Options Data
```
tr_datetime tr_date tr_time tr_open tr_high tr_low tr_close otype strike_price open_interest stock_name week_expiry_date tr_volume
2020-01-01 09:15:59 2020-01-01 09:15:59 0.5 0.65 0.5 0.55 CE 12550 86550 NIFTY 2020-01-02 2250
```

## Strategy Logic

The strategy uses a combination of conditions to generate entry signals:

1. **Delta Condition**: Ensures options have sufficient directional bias
2. **Gamma Condition**: Focuses on options with appropriate price sensitivity
3. **CVD Breakout**: Confirms directional momentum with Cumulative Volume Delta
4. **VWAP Condition**: Uses price relative to VWAP for trend confirmation
5. **PCR Analysis**: Analyzes Put-Call Ratio for sentiment confirmation
6. **OI Change**: Detects significant Open Interest changes for institutional activity
7. **IV Analysis**: Considers Implied Volatility percentile for optimal entry

Exit signals are generated based on:
- Stop loss (adaptive based on volatility)
- Trailing stops for profitable trades
- Time-based exits for non-performing trades
- Signal reversals
- Profit targets (adjusted based on time to expiry)

## Running the System

### Dependencies

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Backtesting

To run a backtest with enhanced features:

```bash
python enhanced_backtest.py \
  --spot-data data/input/spot/Nifty_spot_2020_25.csv \
  --futures-data data/input/futures/nifty_fut_1min_month_1.csv \
  --options-data-dir data/input/options/ \
  --start-date 2022-01-03 \
  --end-date 2022-01-28 \
  --capital 2000000 \
  --require-conditions 5 \
  --mandatory-conditions cvd,vwap,gamma \
  --strikes-range 6 \
  --adaptive-stop-loss \
  --trailing-stop \
  --market-regime-adaptation
```

### Parameter Reference

Key strategy parameters can be configured in `config/strategy_params.py`:

- `TRADE_CAPITAL`: Capital allocated per trade (default: ₹100,000)
- `MAX_CONCURRENT_TRADES`: Maximum simultaneous positions (default: 3)
- `INITIAL_STOP_LOSS_PERCENT`: Default stop loss percentage (default: 20%)
- `PROFIT_TARGET_1`: First profit target (default: 30%)
- `SIZE_REDUCTION_1`: Position size reduction at first target (default: 50%)
- `PROFIT_TARGET_2`: Second profit target (default: 70%)
- `SIZE_REDUCTION_2`: Position size reduction at second target (default: 50%)
- `CALL_DELTA_THRESHOLD`: Minimum delta for call options (default: 0.35)
- `PUT_DELTA_THRESHOLD`: Maximum delta for put options (default: -0.35)
- `GAMMA_THRESHOLD`: Threshold for gamma condition (default: 0.03)
- `OI_CHANGE_THRESHOLD`: Minimum OI change for signal (default: 3.5%)
- `VWAP_BREAKOUT_THRESHOLD`: VWAP breakout threshold (default: 0.5%)
- `CVD_THRESHOLD`: CVD signal threshold (default: 1.5)

## Performance Optimization

For large datasets, the system provides several optimization features:

- **Memory Efficiency**: Processes data in chunks to manage memory usage
- **Parallel Processing**: Uses multiprocessing for indicator calculations
- **Dask Integration**: Optional distributed computing for very large datasets
- **Data Sampling**: Selectively samples data to reduce processing time
- **Strike Filtering**: Focuses on strikes near ATM to reduce computation

## Interpretation of Results

The system generates comprehensive reports in both HTML and CSV formats, including:

- **Performance Summary**: Total P&L, win rate, profit factor, drawdown
- **Trade Log**: Detailed record of all trades with entry/exit times, prices, and reasons
- **P&L Distribution**: Analysis of trade profitability distribution
- **Drawdown Analysis**: Maximum drawdown and drawdown periods
- **Market Regime Analysis**: Performance across different market regimes

## Recommended Workflow for Developers

1. **Explore the Data**: Use notebooks in the `notebooks/` directory to understand data characteristics
2. **Adjust Parameters**: Modify strategy parameters in `config/strategy_params.py`
3. **Run Test Backtests**: Use `quick_backtest.py` for initial testing with smaller data samples
4. **Optimize Parameters**: Use multiple runs to find optimal parameter combinations
5. **Deep Analysis**: Run detailed backtests with `enhanced_backtest.py` and analyze reports
6. **Extend Functionality**: Add new indicators or rules to improve performance


## License

Proprietary - All rights reserved.

## Acknowledgements

This system was developed for institutional options trading and incorporates advanced quantitative techniques for market analysis and risk management.