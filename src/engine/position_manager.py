"""
Position management for trade execution.
"""
import pandas as pd
import numpy as np
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from config.strategy_params import (
    TRADE_CAPITAL, MAX_CONCURRENT_TRADES, INITIAL_STOP_LOSS_PERCENT,
    PROFIT_TARGET_1, SIZE_REDUCTION_1, PROFIT_TARGET_2, SIZE_REDUCTION_2,
    EXIT_TIME, LOSING_TRADE_MAX_DURATION
)

logger = logging.getLogger(__name__)

class PositionManager:
    """
    Manages trading positions, including entry, exit, and tracking.
    """
    
    def __init__(self, trade_capital: float = TRADE_CAPITAL,
                 adaptive_stop_loss: bool = False,
                 trailing_stop: bool = False,
                 time_based_targets: bool = False):
        """
        Initialize the position manager.
        
        Args:
            trade_capital: Capital allocated per trade
            adaptive_stop_loss: Whether to use adaptive stop loss based on volatility
            trailing_stop: Whether to use trailing stop loss to lock in profits
            time_based_targets: Whether to adjust profit targets based on time to expiry
        """
        self.trade_capital = trade_capital
        self.open_positions = {}  # Dictionary to store open positions by ID
        self.closed_positions = []  # List to store closed positions
        self.exit_time = datetime.strptime(EXIT_TIME, "%H:%M:%S").time()
        
        # Initialize a reference to the risk manager (will be set later)
        self.risk_manager = None
        
        # Track transaction costs in a more accessible way
        self.total_transaction_costs = 0.0
        
        # Store feature flags
        self.adaptive_stop_loss = adaptive_stop_loss
        self.trailing_stop = trailing_stop
        self.time_based_targets = time_based_targets
        
        logger.info(f"Position manager initialized with {trade_capital} capital per trade")
        if adaptive_stop_loss:
            logger.info("Adaptive stop loss enabled")
        if trailing_stop:
            logger.info("Trailing stop enabled")
        if time_based_targets:
            logger.info("Time-based targets enabled")
    
    def reset(self):
        """Reset the position manager state."""
        self.open_positions = {}
        self.closed_positions = []
        self.total_transaction_costs = 0.0
        logger.info("Position manager reset")
    
    def has_capacity(self) -> bool:
        """
        Check if we have capacity for new positions.
        
        Returns:
            True if we can open more positions, False otherwise
        """
        return len(self.open_positions) < MAX_CONCURRENT_TRADES
    
    def set_risk_manager(self, risk_manager):
        """Set the risk manager reference."""
        self.risk_manager = risk_manager
        logger.info("Risk manager reference set in position manager")
    
    def open_position(self, 
                     timestamp: datetime, 
                     signal: Dict[str, Any], 
                     position_size: float) -> Optional[Dict[str, Any]]:
        """
        Open a new position based on a signal.
        
        Args:
            timestamp: Current timestamp
            signal: Signal dictionary with trade details
            position_size: Position size (number of contracts)
            
        Returns:
            Position dictionary if opened successfully, None otherwise
        """
        # Check if we have capacity
        if not self.has_capacity():
            logger.warning("Cannot open position: maximum positions reached")
            return None
        
        if self.risk_manager is None:
            logger.warning("Risk manager not set in position manager")
            return None

        # Validate signal structure
        if not isinstance(signal, dict):
            logger.error(f"Invalid signal type: {type(signal)}, expected dict")
            return None
        
        if 'option' not in signal:
            logger.error("Missing 'option' field in signal")
            return None

        # Extract option details from signal with enhanced validation
        try:
            option_data = signal['option']
            
            # Required fields validation
            required_fields = ['otype', 'strike_price', 'tr_close']
            missing_fields = [field for field in required_fields if field not in option_data]
            
            if missing_fields:
                logger.error(f"Missing required fields in option data: {missing_fields}")
                logger.debug(f"Available fields: {list(option_data.keys())}")
                return None

            # Extract and validate option type
            option_type = option_data['otype']
            if option_type not in ['CE', 'PE']:
                logger.error(f"Invalid option type: {option_type}")
                return None

            # Extract and validate strike price
            strike_price = option_data['strike_price']
            if not isinstance(strike_price, (int, float)) or strike_price <= 0:
                logger.error(f"Invalid strike price: {strike_price}")
                return None

            # Extract and validate option price
            option_price = option_data['tr_close']
            if not isinstance(option_price, (int, float)) or option_price <= 0:
                logger.error(f"Invalid option price: {option_price}")
                return None

            # Enhanced expiry date handling
            expiry_date = None
            expiry_fields = ['expiry_date', 'week_expiry_date']
            
            for field in expiry_fields:
                if field in option_data:
                    try:
                        expiry_date = pd.to_datetime(option_data[field])
                        break
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse {field}: {e}")
                        continue

            if expiry_date is None:
                logger.error(f"No valid expiry date found. Available fields: {list(option_data.keys())}")
                return None

            if expiry_date.date() < timestamp.date():
                logger.error(f"Invalid expiry date: {expiry_date.date()} is before current date {timestamp.date()}")
                return None

            # Check for similar existing positions with enhanced logging
            if self.has_similar_position(option_type, strike_price, option_price):
                logger.warning(
                    f"Skipping similar position: {option_type} {strike_price} @ {option_price}"
                    f" (existing positions: {len(self.open_positions)})"
                )
                return None

            # Check for existing positions with same attributes
            for pos_id, pos in self.open_positions.items():
                if (pos['option_type'] == option_type and 
                    pos['strike_price'] == strike_price and 
                    pos['expiry_date'].date() == expiry_date.date()):
                    logger.warning(
                        f"Skipping duplicate position: {option_type} {strike_price}"
                        f" expiry {expiry_date.date()} (position_id: {pos_id})"
                    )
                    return None

            # Create unique position ID
            position_id = str(uuid.uuid4())
            
            # Extract signal details
            delta = option_data.get('delta', 0)
            gamma = option_data.get('gamma', 0)
            
            # Get IV if available, default to 30% if not
            implied_vol = option_data.get('implied_volatility', 0.3)
            
            # Calculate adaptive stop loss based on IV
            # Higher IV = wider stop, but with absolute maximum
            iv_factor = min(1.0, implied_vol)  # Cap IV factor at 100%
            
            # Base stop percentage between 8-15% depending on IV
            base_stop_pct = 0.08 + (0.07 * iv_factor)
            
            # Apply time-to-expiry adjustment (tighter stops closer to expiry)
            days_to_expiry = (expiry_date - timestamp).days
            time_factor = min(1.0, days_to_expiry / 5)  # Scale factor based on days to expiry
            
            # Calculate final stop loss percentage (max 15%)
            adaptive_stop_pct = min(0.15, base_stop_pct * time_factor)
            
            # Set stop loss price based on option type
            if option_type == 'CE':
                stop_loss = option_price * (1 - adaptive_stop_pct)
            else:  # put option
                stop_loss = option_price * (1 + adaptive_stop_pct)
            
            # Calculate transaction costs for entry
            entry_costs = self.risk_manager.calculate_transaction_costs(
                option_price, position_size, is_entry=True
            )
            
            # Calculate trade details with costs
            trade_value = option_price * position_size
            total_entry_cost = trade_value + entry_costs['total']
            
            # Record total costs for easier reference
            self.total_transaction_costs += entry_costs['total']
            
            # Create position record with details about conditions met
            position = {
                'id': position_id,
                'entry_time': timestamp,
                'option_type': option_type,
                'strike_price': strike_price,
                'expiry_date': expiry_date,
                'entry_price': option_price,
                'position_size': position_size,
                'original_size': position_size,
                'trade_value': trade_value,
                'current_price': option_price,
                'current_value': trade_value,
                'stop_loss': stop_loss,
                'adaptive_stop_pct': adaptive_stop_pct,
                'initial_stop_pct': adaptive_stop_pct,  # Store for reference
                'trailing_active': False,  # Flag for trailing stop
                'highest_price': option_price if option_type == 'CE' else float('-inf'),
                'lowest_price': option_price if option_type == 'PE' else float('inf'),
                'first_target_hit': False,
                'second_target_hit': False,
                'delta': delta,
                'gamma': gamma,
                'pnl': 0,
                'pnl_percent': 0,
                'signal': signal,
                'status': 'open',
                'partial_exits': [],
                'entry_costs': entry_costs,
                'total_entry_cost': total_entry_cost,
                'net_pnl': 0,  # P&L after all costs
                'conditions_met': signal.get('conditions_met', 0),
                'conditions': signal.get('conditions', {}),
            }
            
            # Log transaction costs details
            costs_breakdown = ", ".join([f"{k}: ₹{v:.2f}" for k, v in entry_costs.items() if k != 'total'])
            logger.info(f"Entry costs for {position_id}: ₹{entry_costs['total']:.2f} ({costs_breakdown})")
            
            # Add to open positions
            self.open_positions[position_id] = position
            
            logger.info(f"Opened position {position_id} with adaptive stop loss {adaptive_stop_pct:.1%}")
            
            # Validate the created position before returning
            required_position_fields = [
                'id', 'entry_time', 'option_type', 'strike_price', 
                'expiry_date', 'entry_price', 'position_size', 'stop_loss'
            ]
            
            missing_position_fields = [
                field for field in required_position_fields 
                if field not in position
            ]
            
            if missing_position_fields:
                logger.error(f"Created position missing required fields: {missing_position_fields}")
                return None

            # Validate position values
            if position['position_size'] <= 0:
                logger.error(f"Invalid position size: {position['position_size']}")
                return None
            
            if position['stop_loss'] <= 0:
                logger.error(f"Invalid stop loss: {position['stop_loss']}")
                return None

            # Log successful position creation with details
            logger.info(
                f"Successfully opened position {position['id']}: "
                f"{option_type} {strike_price} @ {option_price} "
                f"(size: {position_size}, stop: {position['stop_loss']:.2f})"
            )
            
            return position

        except Exception as e:
            logger.error(f"Error opening position: {str(e)}", exc_info=True)
            return None
    
    def update_position(self, 
                       position_id: str, 
                       current_price: float,
                       timestamp: datetime) -> Dict[str, Any]:
        """
        Update an open position with current market data.
        
        Args:
            position_id: Position ID to update
            current_price: Current option price
            timestamp: Current timestamp
            
        Returns:
            Updated position dictionary
        """
        if position_id not in self.open_positions:
            logger.warning(f"Cannot update position {position_id}: not found")
            return {}
        
        if self.risk_manager is None:
            logger.warning("Risk manager not set in position manager")
            return self.open_positions[position_id]
        
        position = self.open_positions[position_id]
        
        # Calculate time-based factors
        days_to_expiry = (position['expiry_date'].date() - timestamp.date()).days
        time_in_trade = (timestamp - position['entry_time']).total_seconds() / 60  # minutes
        
        # Get signal strength from conditions met (normalized to 0-1)
        signal_strength = min(1.0, position['conditions_met'] / 5)  # Assume max 5 conditions
        
        # Calculate volatility factor from option data
        implied_vol = position.get('signal', {}).get('option', {}).get('implied_volatility', 0.3)
        vol_factor = min(1.2, max(0.8, implied_vol / 0.3))  # Scale around typical 30% IV
        
        # Calculate dynamic profit targets
        if days_to_expiry <= 1:
            # More aggressive on expiry day
            base_target_1 = PROFIT_TARGET_1 * 0.7
            base_target_2 = PROFIT_TARGET_2 * 0.6
        elif days_to_expiry <= 3:
            base_target_1 = PROFIT_TARGET_1 * 0.85
            base_target_2 = PROFIT_TARGET_2 * 0.8
        else:
            base_target_1 = PROFIT_TARGET_1
            base_target_2 = PROFIT_TARGET_2
        
        # Adjust targets based on signal strength and volatility
        adjusted_target_1 = base_target_1 * (0.8 + 0.4 * signal_strength) * vol_factor
        adjusted_target_2 = base_target_2 * (0.8 + 0.4 * signal_strength) * vol_factor
        
        logger.debug(f"Position {position_id} targets adjusted: T1={adjusted_target_1:.1%}, T2={adjusted_target_2:.1%} "
                    f"(Signal strength={signal_strength:.2f}, Vol factor={vol_factor:.2f})")
        
        # Check for target hits with adjusted values
        if not position['first_target_hit'] and position['pnl_percent'] >= adjusted_target_1:
            position['first_target_hit'] = True
            
            # Calculate optimal size reduction based on volatility
            size_reduction = position['position_size'] * (SIZE_REDUCTION_1 * min(1.2, vol_factor))
            new_size = position['position_size'] - size_reduction
            
            # Calculate PnL for this partial exit
            if position['option_type'] == 'CE':
                pnl = (current_price - position['entry_price']) * size_reduction
            else:  # Put option
                pnl = (position['entry_price'] - current_price) * size_reduction
            
            # Calculate transaction costs for this partial exit
            exit_costs = self.risk_manager.calculate_transaction_costs(
                current_price, size_reduction, is_entry=False
            )
            
            # Calculate proportional entry costs for this partial exit
            entry_cost_ratio = size_reduction / position['original_size']
            partial_entry_costs = position['entry_costs']['total'] * entry_cost_ratio
            
            # Calculate net PnL for this partial exit after costs
            net_pnl = pnl - partial_entry_costs - exit_costs['total']
            
            # Record partial exit with net PnL
            exit_value = current_price * size_reduction
            partial_exit = {
                'time': timestamp,
                'price': current_price,
                'size': size_reduction,
                'value': exit_value,
                'pnl': pnl,
                'entry_costs': partial_entry_costs,
                'exit_costs': exit_costs['total'],
                'net_pnl': net_pnl,
                'reason': 'first_target'
            }
            
            position['partial_exits'].append(partial_exit)
            position['position_size'] = new_size
            
            # Set dynamic break-even stop with volatility buffer
            buffer = 0.03 * vol_factor
            if position['option_type'] == 'CE':
                position['stop_loss'] = position['entry_price'] * (1 - buffer)
            else:
                position['stop_loss'] = position['entry_price'] * (1 + buffer)
            
            logger.info(f"First target hit for {position_id}: Reduced size by {(size_reduction/position['position_size'])*100:.1f}%, "
                       f"New stop at {position['stop_loss']:.2f}")
            
        elif position['first_target_hit'] and not position['second_target_hit'] and position['pnl_percent'] >= adjusted_target_2:
            position['second_target_hit'] = True
            
            # Adjust second reduction based on performance
            performance_factor = min(1.2, max(0.8, position['pnl_percent'] / adjusted_target_2))
            size_reduction = position['position_size'] * (SIZE_REDUCTION_2 * performance_factor)
            new_size = position['position_size'] - size_reduction
            
            # Calculate PnL for this partial exit
            if position['option_type'] == 'CE':
                pnl = (current_price - position['entry_price']) * size_reduction
            else:  # Put option
                pnl = (position['entry_price'] - current_price) * size_reduction
            
            # Calculate transaction costs for this partial exit
            exit_costs = self.risk_manager.calculate_transaction_costs(
                current_price, size_reduction, is_entry=False
            )
            
            # Calculate proportional entry costs for this partial exit
            entry_cost_ratio = size_reduction / position['original_size']
            partial_entry_costs = position['entry_costs']['total'] * entry_cost_ratio
            
            # Calculate net PnL for this partial exit after costs
            net_pnl = pnl - partial_entry_costs - exit_costs['total']
            
            # Record partial exit with net PnL
            exit_value = current_price * size_reduction
            partial_exit = {
                'time': timestamp,
                'price': current_price,
                'size': size_reduction,
                'value': exit_value,
                'pnl': pnl,
                'entry_costs': partial_entry_costs,
                'exit_costs': exit_costs['total'],
                'net_pnl': net_pnl,
                'reason': 'second_target'
            }
            
            position['partial_exits'].append(partial_exit)
            position['position_size'] = new_size
            
            # Set aggressive trailing stop
            trail_percent = 0.10 * vol_factor  # 10% base, adjusted for volatility
            if position['option_type'] == 'CE':
                position['stop_loss'] = current_price * (1 - trail_percent)
            else:
                position['stop_loss'] = current_price * (1 + trail_percent)
            
            logger.info(f"Second target hit for {position_id}: Reduced size by {(size_reduction/position['position_size'])*100:.1f}%, "
                       f"New trailing stop at {trail_percent:.1%}")
        
        # Time-based stop loss adjustments
        if not position['first_target_hit']:
            # Tighten stop loss based on time in trade
            if time_in_trade > 30:
                time_factor = min(1.0, (time_in_trade - 30) / 60)  # Scale up over next hour
                if position['option_type'] == 'CE':
                    time_adjusted_stop = position['entry_price'] * (1 - 0.1 * time_factor)
                    position['stop_loss'] = max(position['stop_loss'], time_adjusted_stop)
                else:
                    time_adjusted_stop = position['entry_price'] * (1 + 0.1 * time_factor)
                    position['stop_loss'] = min(position['stop_loss'], time_adjusted_stop)
                
                if time_in_trade % 15 == 0:  # Log every 15 minutes
                    logger.debug(f"Time-based stop adjustment for {position_id}: New stop at {position['stop_loss']:.2f}")
        
        # Update price tracking for trailing stops
        if position['option_type'] == 'CE':
            position['highest_price'] = max(position['highest_price'], current_price)
        else:
            position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Update current price and value
        position['current_price'] = current_price
        position['current_value'] = current_price * position['position_size']
        
        # Calculate P&L
        entry_value = position['entry_price'] * position['position_size']
        current_value = position['current_value']
        
        # P&L calculation for option buying strategy
        position['pnl'] = current_value - entry_value
        
        position['pnl_percent'] = position['pnl'] / entry_value
        
        # Estimate potential exit costs
        exit_costs = self.risk_manager.calculate_transaction_costs(
            current_price, position['position_size'], is_entry=False
        )
        
        # Calculate P&L with costs
        position['exit_costs'] = exit_costs
        all_costs = position['entry_costs']['total'] + exit_costs['total']
        position['all_costs'] = all_costs
        position['net_pnl'] = position['pnl'] - all_costs
        
        # Implement trailing stop logic for profitable trades
        if position['pnl_percent'] > 0.1:  # If >10% profit
            position['trailing_active'] = True
            
            # Calculate trailing stop based on profit level
            trailing_factor = min(0.5, position['pnl_percent'])  # Max 50% of current price
            base_stop = position['initial_stop_pct'] * (1 - trailing_factor)
            
            if position['option_type'] == 'CE':
                # For calls, trail below highest price
                trailing_stop = position['highest_price'] * (1 - base_stop)
                position['stop_loss'] = max(position['stop_loss'], trailing_stop)
            else:
                # For puts, trail above lowest price
                trailing_stop = position['lowest_price'] * (1 + base_stop)
                position['stop_loss'] = min(position['stop_loss'], trailing_stop)
            
            if position['trailing_active']:
                logger.debug(f"Updated trailing stop for {position_id} to {position['stop_loss']:.2f}")
        
        return position
    
    def check_exit_conditions(self, 
                             position_id: str, 
                             timestamp: datetime,
                             signals: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a position should be exited based on the exit rules.
        
        Args:
            position_id: Position ID to check
            timestamp: Current timestamp
            signals: Current market signals
            
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        if position_id not in self.open_positions:
            return False, ""
        
        position = self.open_positions[position_id]
        
        # 1. Enhanced check for strict stop loss
        if position['option_type'] == 'CE':
            if position['current_price'] <= position['stop_loss']:
                logger.info(f"Stop loss triggered for {position_id}: "
                          f"Current price {position['current_price']} <= Stop loss {position['stop_loss']}")
                return True, "stop_loss"
        else:  # PE
            if position['current_price'] >= position['stop_loss']:
                logger.info(f"Stop loss triggered for {position_id}: "
                          f"Current price {position['current_price']} >= Stop loss {position['stop_loss']}")
                return True, "stop_loss"
        
        # 2. Check time-based exit for losing trades with improved logging
        if position['pnl'] < 0:
            duration = timestamp - position['entry_time']
            duration_minutes = duration.total_seconds() / 60
            if duration_minutes >= LOSING_TRADE_MAX_DURATION:
                logger.info(f"Time stop triggered for {position_id}: "
                          f"Duration {duration_minutes:.1f} min >= Limit {LOSING_TRADE_MAX_DURATION} min")
                return True, "time_stop"
        
        # 3. Check end-of-day conditions
        # First check if we've moved to a new trading day
        if position['entry_time'].date() < timestamp.date():
            return True, "new_trading_day"
        
        # Then check if we've hit the exit time on the current day
        if timestamp.time() >= self.exit_time:
            return True, "session_end"
        
        # 4. Check signal reversal
        if 'exit_signals' in signals:
            for exit_signal in signals['exit_signals']:
                if exit_signal.get('position_id') == position_id:
                    return True, "signal_reversal"
        
        # 5. Check delta reversal for directional trades
        if 'delta' in position and position['delta'] != 0:
            current_delta = position.get('current_delta', position['delta'])
            
            # Exit if delta reverses direction significantly
            if (position['option_type'] == 'CE' and current_delta < 0.4 and position['delta'] >= 0.4) or \
               (position['option_type'] == 'PE' and current_delta > -0.4 and position['delta'] <= -0.4):
                return True, "delta_reversal"
        
        return False, ""
    
    def close_position(self, 
                      position_id: str, 
                      timestamp: datetime, 
                      exit_reason: str) -> Dict[str, Any]:
        """
        Close an open position.
        
        Args:
            position_id: Position ID to close
            timestamp: Current timestamp
            exit_reason: Reason for closing the position
            
        Returns:
            Closed position dictionary
        """
        if position_id not in self.open_positions:
            logger.warning(f"Cannot close position {position_id}: not found")
            return {}
        
        position = self.open_positions[position_id]
        
        # Update position status
        position['exit_time'] = timestamp
        position['exit_price'] = position['current_price']
        position['exit_value'] = position['current_value']
        position['exit_reason'] = exit_reason
        position['status'] = 'closed'
        
        # Ensure exit costs are calculated
        if self.risk_manager is not None and 'exit_costs' not in position:
            position['exit_costs'] = self.risk_manager.calculate_transaction_costs(
                position['exit_price'], position['position_size'], is_entry=False
            )
        
        # Calculate remaining costs and net P&L
        remaining_ratio = position['position_size'] / position['original_size']
        remaining_entry_costs = position['entry_costs']['total'] * remaining_ratio
        final_exit_costs = position.get('exit_costs', {}).get('total', 0)
        
        position['net_pnl'] = position['pnl'] - remaining_entry_costs - final_exit_costs
        
        # Calculate total P&L including partial exits
        partial_net_pnl = sum([exit_info['net_pnl'] for exit_info in position['partial_exits']])
        position['total_net_pnl'] = position['net_pnl'] + partial_net_pnl
        
        # Calculate trade duration
        duration = timestamp - position['entry_time']
        position['duration_minutes'] = duration.total_seconds() / 60
        
        # Update total cost tracking
        if 'exit_costs' in position and 'total' in position['exit_costs']:
            self.total_transaction_costs += position['exit_costs']['total']
        
        # Move to closed positions list
        self.closed_positions.append(position)
        del self.open_positions[position_id]
        
        # Log comprehensive trade details
        costs_detail = f"Entry: ₹{remaining_entry_costs:.2f}, Exit: ₹{final_exit_costs:.2f}, Total: ₹{remaining_entry_costs + final_exit_costs:.2f}"
        
        logger.info(f"Closed position {position_id}: {exit_reason}")
        logger.info(f"  Trade details: {position['option_type']} {position['strike_price']} @ {position['entry_price']:.2f}")
        logger.info(f"  Entry: {position['entry_time']}, Exit: {position['exit_time']}")
        logger.info(f"  Duration: {position['duration_minutes']:.1f} minutes")
        logger.info(f"  Raw P&L: ₹{position['pnl']:.2f}, Net P&L: ₹{position['total_net_pnl']:.2f}")
        logger.info(f"  Transaction costs: {costs_detail}")
        
        return position
    
    def close_all_positions(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """
        Close all open positions.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            List of closed position dictionaries
        """
        closed = []
        position_ids = list(self.open_positions.keys())
        
        for position_id in position_ids:
            closed_position = self.close_position(position_id, timestamp, "forced_exit")
            closed.append(closed_position)
        
        return closed
    
    def manage_positions(self,
                        timestamp: datetime,
                        signals: Dict[str, Any],
                        spot_data: pd.DataFrame,
                        futures_data: pd.DataFrame,
                        options_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Manage all open positions based on current market data.
        With improved error handling for empty DataFrames and missing options data.
        Adds price continuity validation for safer position updates.
        
        Args:
            timestamp: Current timestamp
            signals: Current market signals
            spot_data: Current spot market data
            futures_data: Current futures market data
            options_data: Current options market data
            
        Returns:
            List of position updates
        """
        updates = []
        position_ids = list(self.open_positions.keys())
        
        for position_id in position_ids:
            if position_id not in self.open_positions:
                continue
                
            position = self.open_positions[position_id]
            try:
                if position['expiry_date'] is None:
                    logger.error(f"Invalid expiry date for position {position_id}")
                    continue
                
                expiry = str(position['expiry_date'].date())
                option_type = position['option_type']
                strike_price = position['strike_price']
                
                # Check if we have the required data
                if expiry not in options_data or options_data[expiry].empty:
                    logger.warning(f"No data available for expiry {expiry}, closing position {position_id}")
                    closed_position = self.close_position(position_id, timestamp, "Missing data for expiry")
                    updates.append(closed_position)
                    continue
                
                # Look for this specific option
                filtered_options = options_data[expiry][
                    (options_data[expiry]['otype'] == option_type) & 
                    (options_data[expiry]['strike_price'] == strike_price)
                ]
                
                # If not found in this expiry, close the position rather than looking in other expiries
                if filtered_options.empty:
                    logger.warning(f"Option {option_type} {strike_price} not found in expiry {expiry}, closing position")
                    closed_position = self.close_position(position_id, timestamp, "Option not found in expiry")
                    updates.append(closed_position)
                    continue
                
                # Filter options data up to current timestamp
                filtered_options = filtered_options[filtered_options.index <= timestamp]
                
                if filtered_options.empty:
                    logger.warning(f"No data up to timestamp {timestamp} for {option_type} {strike_price}, closing position")
                    closed_position = self.close_position(position_id, timestamp, "No current data for option")
                    updates.append(closed_position)
                    continue
                
                # Get current option data
                option_data = filtered_options.iloc[-1]
                
                if 'tr_close' not in option_data:
                    logger.error(f"Missing price data for option {option_type} {strike_price}, closing position")
                    closed_position = self.close_position(position_id, timestamp, "Missing price data")
                    updates.append(closed_position)
                    continue
                
                # Get current price and validate for continuity
                current_price = option_data['tr_close']
                previous_price = position['current_price']
                
                # Check for unrealistic price change (more than 30%)
                price_change_pct = abs((current_price - previous_price) / previous_price * 100)
                max_allowed_change = 30.0  # 30% limit
                
                if price_change_pct > max_allowed_change:
                    # Log the anomaly
                    logger.warning(
                        f"Unrealistic price change detected in position {position_id}: "
                        f"{previous_price:.2f} -> {current_price:.2f} ({price_change_pct:.2f}%). "
                        f"Limiting to {max_allowed_change}% change."
                    )
                    
                    # Limit the price change to the maximum allowed
                    change_direction = 1 if current_price > previous_price else -1
                    max_allowed_price_change = previous_price * (max_allowed_change / 100)
                    current_price = previous_price + (change_direction * max_allowed_price_change)
                    
                    # Record the adjustment
                    logger.info(f"Adjusted price from {option_data['tr_close']:.2f} to {current_price:.2f}")
                
                # Update delta if available
                if 'delta' in option_data:
                    position['current_delta'] = option_data['delta']
                
                # Update position with current price
                updated_position = self.update_position(position_id, current_price, timestamp)
                
                # Check exit conditions
                should_exit, exit_reason = self.check_exit_conditions(position_id, timestamp, signals)
                
                if should_exit:
                    closed_position = self.close_position(position_id, timestamp, exit_reason)
                    updates.append(closed_position)
                else:
                    updates.append(updated_position)
                
            except Exception as e:
                logger.error(f"Error managing position {position_id}: {e}")
                # Close problematic positions
                try:
                    closed_position = self.close_position(position_id, timestamp, f"Error: {str(e)}")
                    updates.append(closed_position)
                except Exception as close_error:
                    logger.error(f"Failed to close problematic position {position_id}: {close_error}")
        
        return updates
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            Dictionary of open positions by ID
        """
        return self.open_positions
    
    def get_closed_positions(self) -> List[Dict[str, Any]]:
        """
        Get all closed positions.
        
        Returns:
            List of closed position dictionaries
        """
        return self.closed_positions
    
    def calculate_current_pnl(self) -> float:
        """
        Calculate current P&L across all open positions.
        
        Returns:
            Total P&L including realized and unrealized
        """
        total_pnl = 0.0
        
        # Add P&L from open positions
        for position in self.open_positions.values():
            position_pnl = position.get('net_pnl', 0.0)  # Use net P&L that includes costs
            total_pnl += position_pnl
        
        # Add P&L from positions closed today
        today = datetime.now().date()
        for position in self.closed_positions:
            if position.get('exit_time', datetime.now()).date() == today:
                total_pnl += position.get('total_net_pnl', 0.0)  # Include costs
        
        return total_pnl
    
    def calculate_cumulative_pnl(self) -> float:
        """
        Calculate cumulative P&L including all closed positions.
        
        Returns:
            Total cumulative P&L
        """
        total_pnl = 0.0
        
        # Add P&L from all closed positions
        for position in self.closed_positions:
            total_pnl += position.get('total_net_pnl', 0.0)
        
        # Add P&L from current open positions
        for position in self.open_positions.values():
            total_pnl += position.get('net_pnl', 0.0)
        
        return total_pnl
    
    def get_total_transaction_costs(self) -> float:
        """
        Get the total transaction costs across all trades.
        
        Returns:
            Total transaction costs
        """
        return self.total_transaction_costs
    
    def has_similar_position(self, option_type: str, strike_price: float, entry_price: float, price_tolerance: float = 0.01) -> bool:
        """
        Check if we already have a similar position with nearly identical price.
        
        Args:
            option_type: Option type (CE/PE)
            strike_price: Strike price
            entry_price: Entry price
            price_tolerance: Maximum price difference to consider similar (proportion)
            
        Returns:
            True if similar position exists, False otherwise
        """
        for position in self.open_positions.values():
            if (position['option_type'] == option_type and 
                position['strike_price'] == strike_price):
                
                # Calculate price difference as percentage
                price_diff = abs(position['entry_price'] - entry_price) / position['entry_price']
                
                if price_diff <= price_tolerance:
                    logger.warning(f"Similar position found: {option_type} {strike_price} with price {position['entry_price']} vs {entry_price}")
                    return True
                
        return False