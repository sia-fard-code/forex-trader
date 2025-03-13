# trade_manager.py

import numpy as np
import logging
import threading
from numba_functions import update_positions_numba_parallel, calculate_pnl_numba, calculate_floating_pnl_numba
# Define the structured array dtype once here
position_dtype = np.dtype([
    ('id', 'i4'),             # Unique identifier (int32)
    ('direction', 'i1'),      # 1 for long, -1 for short (int8)
    ('open_id', 'i8'),      # Tick index or timestamp when opened (int64)
    ('open_price', 'f4'),     # Price at which the position was opened (float32)
    ('volume', 'f8'),         # Lot size (float32)
    ('commission', 'f4'),     # Commission applied at opening (float32)
    ('slippage', 'f4'),       # Slippage applied at opening (float32)
    ('leverage', 'f4'),       # Leverage used (float32)
    ('close_id', 'i8'),     # Tick index or timestamp when closed (-1 if open) (int64)
    ('close_price', 'f4'),    # Price at which the position was closed (float32)
    ('highest_price', 'f8'),    # highest Price at which the position (float32)
    ('lowest_price', 'f8'),    # lowest Price at which the position (float32)
    ('trailing_pct', 'f4'),    # trailing_pct (float32)
    ('profit_threshold', 'f4'),    # profit_threshold (float32)
    ('loss_threshold', 'f4'),    # loss_threshold (float32)
    ('closing_method', 'i4'),    # closing_method (str)
    ('pnl', 'f4')             # Profit and Loss (float32)
])
class TradeManager:
    def __init__(self, config, max_positions=1000000, data_manager=None):
        """
        Initialize TradeManager with broker configurations and pre-allocate positions array.

        Parameters:
        - config (dict): Configuration dictionary containing broker specs and initial settings.
        - max_positions (int): Maximum number of positions to handle.
        """
        self.config = config
        self.position_dtype = position_dtype
        self.direction_multiplier = 1#config.get("direction_multiplier", 1)
        # Initialize positions array
        self.positions_np = np.empty(max_positions, dtype=self.position_dtype)
        self.current_size = 0
        self.next_id = 0
        self.lock = threading.Lock()
        self.data_manager = data_manager
        # Initialize financial metrics
        self.balance = config["initial_capital"]  # Starting balance
        self.floating_pnl = 0.0  # Unrealized PnL from open positions
        self.equity = self.balance + self.floating_pnl  # Real-time equity
        self.margin_used = 0.0  # Total margin currently used
        self.free_margin = self.equity - self.margin_used  # Available margin for new positions

        # For drawdown tracking
        self.peak_equity = self.equity
        self.max_drawdown = 0.0
        

    def update_positions(
        self,
        current_bid,
        current_ask,
        tick_id,
        # profit_threshold,
        # loss_threshold,
        slippage,
        commission_per_lot,
        # closing_method="stopLoss"
    ):
        """
        Update positions, handle closures, and recalculate margins.

        Args:
            current_bid (float): Current bid price.
            current_ask (float): Current ask price.
            tick_id (int): Current tick ID.
            profit_threshold (float): PnL threshold to take profit.
            loss_threshold (float): PnL threshold to stop loss.
            slippage (float): Slippage value.
            commission_per_lot (float): Commission per lot.
            closing_method (str): Method of closing positions ('Trailing', 'StopLoss', etc.).
        """
        # with self.lock:
        # Access a subset of positions for updating
        positions_subset = self.positions_np[:self.current_size]

        # Update positions using Numba-accelerated function
        update_positions_numba_parallel(
            positions=positions_subset,
            current_bid=current_bid,
            current_ask=current_ask,
            tick_id=tick_id,
            # profit_threshold=profit_threshold,
            # loss_threshold=loss_threshold,
            slippage=slippage,
            commission_per_lot=commission_per_lot,
            # closing_method=closing_method
        )

        # Identify closed positions
        closed_mask = (positions_subset['close_id'] == tick_id) & (positions_subset['pnl'] != 0.0)
        closed_positions = positions_subset[closed_mask]

        # Update main positions array with changes
        self.positions_np[:self.current_size][closed_mask] = positions_subset[closed_mask]

        # Adjust balance, margin, and equity for closed positions
        for pos in closed_positions:
            pnl = pos['pnl']
            required_margin = (pos['open_price'] * pos['volume']) / pos['leverage']
            self.margin_used -= required_margin
            self.free_margin += required_margin
            self.balance += pnl

        # Recalculate floating PnL and update equity
        self.update_floating_pnl(current_bid, current_ask)
            

    def update_balance(self, pnl):
        """
        Update the account balance based on realized PnL.

        Parameters:
        - pnl (float): Realized Profit and Loss from a closed position.
        """
        with self.lock:
            self.balance += pnl
            self.update_equity()

    def update_floating_pnl(self, current_bid, current_ask):
        """
        Recalculate the floating (unrealized) PnL based on current market prices.

        Parameters:
        - current_bid (float): Current bid price.
        - current_ask (float): Current ask price.
        """
        # with self.lock:
        self.floating_pnl = calculate_floating_pnl_numba(
            positions=self.positions_np[:self.current_size],
            current_bid=current_bid,
            current_ask=current_ask,
            slippage=self.config.get("slippage_points", 0.0001),
            commission=self.config.get("commission_per_lot", 2.0)
        )
        self.update_equity()

    def update_equity(self):
        """
        Recalculate the equity based on balance and floating PnL.
        """
        with self.lock:
            self.equity = self.balance + self.floating_pnl
            self.free_margin = self.equity - self.margin_used
            self.data_manager.set("equity", self.equity)
            self.data_manager.set("balance", self.balance)

            # Update peak equity
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity

            # Calculate drawdown
            current_drawdown = self.peak_equity - self.equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

    def open_position(self, direction, bid_price, ask_price, volume, tick_id, profit_threshold, loss_threshold,closing_method=1):
        """
        Open a new position with thread safety.

        Parameters:
        - direction (int): 1 for long, -1 for short.
        - bid_price (float): Current bid price.
        - ask_price (float): Current ask price.
        - volume (float): Lot size.
        - t (int): Current tick index or timestamp.

        Returns:
        - int: ID of the newly opened position.
        """
        # self._check_volume_constraints(volume)
        # with self.lock:
        if self.current_size >= len(self.positions_np):
            raise MemoryError("Maximum number of positions reached in TradeManager.")

        leverage = self.config.get("leverage", 100)
        slippage = self.config.get("slippage_points", 0.0001)
        direction *= self.direction_multiplier
        if direction == 1:  # Long position
            open_price_slipped = ask_price + slippage
            required_margin = (open_price_slipped * volume) / leverage
        else:  # Short position
            open_price_slipped = bid_price - slippage
            required_margin = (open_price_slipped * volume) / leverage

        # Check if there is enough free margin
        if required_margin > self.free_margin:
            raise ValueError(f"Insufficient free margin to open position. Required: {required_margin}, Available: {self.free_margin}")

        commission = self.config.get("commission_per_lot", 2.0) * volume

        idx = self.current_size
        self.positions_np[idx]['id'] = self.next_id
        self.positions_np[idx]['direction'] = direction
        self.positions_np[idx]['open_id'] = tick_id
        self.positions_np[idx]['open_price'] = open_price_slipped
        self.positions_np[idx]['volume'] = volume
        self.positions_np[idx]['commission'] = commission
        self.positions_np[idx]['slippage'] = slippage
        self.positions_np[idx]['leverage'] = leverage
        self.positions_np[idx]['close_id'] = -1
        self.positions_np[idx]['close_price'] = np.nan
        self.positions_np[idx]['pnl'] = 0.0
        self.positions_np[idx]['profit_threshold'] = profit_threshold
        self.positions_np[idx]['loss_threshold'] = loss_threshold
        self.positions_np[idx]['closing_method'] = closing_method

        position_id = self.next_id
        self.next_id += 1
        self.current_size += 1

        # Update margin used and free margin
        self.margin_used += required_margin
        self.free_margin -= required_margin
        self.update_equity()
        
        logging.debug(f"Opened position {position_id}: Direction={direction}, Price={open_price_slipped}, Volume={volume}, Required Margin={required_margin}")
        return position_id
    
    def close_position_by_id(self, position_id, close_price, tick_id,pnl):
        """
        Close a specific position by ID with thread safety.

        Parameters:
        - position_id (int): ID of the position to close.
        - bid_price (float): Current bid price.
        - ask_price (float): Current ask price.
        - t (int): Current tick index or timestamp.
        """
        # with self.lock:
        idx = -1
        for i in range(self.current_size):
            if self.positions_np[i]['id'] == position_id and self.positions_np[i]['close_id'] == -1:
                idx = i
                break
        if idx == -1:
            logging.warning(f"No open position found with id {position_id}")
            return
        direction = int(self.positions_np[idx]['direction'])
        slippage = self.positions_np[idx]['slippage']
        leverage = self.positions_np[idx]['leverage']
        volume = self.positions_np[idx]['volume']
        direction *= self.direction_multiplier
        if direction == 1:  # Long position
            close_price_slipped = close_price - slippage
            required_margin = (self.positions_np[idx]['open_price'] * volume) / leverage
        else:  # Short position
            close_price_slipped = close_price + slippage
            required_margin = (self.positions_np[idx]['open_price'] * volume) / leverage

        pnl = (close_price_slipped - self.positions_np[idx]['open_price']) * direction * volume - (self.config.get("commission_per_lot", 2.0) * volume)

        self.positions_np[idx]['close_price'] = close_price_slipped
        self.positions_np[idx]['close_id'] = tick_id
        self.positions_np[idx]['pnl'] = pnl

        # Update margin used and free margin
        self.margin_used -= required_margin
        self.free_margin += required_margin
        self.balance += pnl
        self.update_equity()

        # Update balance with realized PnL

        logging.debug(f"Closed position {position_id}: Close Price={close_price_slipped}, PnL={pnl}, Released Margin={required_margin}")
    
    def close_positions_by_direction_and_pnl(self, direction, profit_threshold, loss_threshold, current_bid, current_ask, tick_id):
        """
        Close all open positions matching the specified direction and meeting PnL thresholds.

        Parameters:
        - direction (int): 1 for long positions or -1 for short positions.
        - profit_threshold (float): PnL threshold to take profit.
        - loss_threshold (float): PnL threshold to stop loss.
        - current_bid (float): Current bid price.
        - current_ask (float): Current ask price.
        - t (int): Current tick index or timestamp.
        """
        slippage = self.config.get("slippage_points", 0.0001)
        commission_per_lot = self.config.get("commission_per_lot", 2.0)
        leverage = self.config.get("leverage", 100)
        direction *= self.direction_multiplier

        # with self.lock:
        positions_subset = self.positions_np[:self.current_size].copy()
        # Filter positions by direction
        mask = (positions_subset['direction'] == direction) & (positions_subset['close_id'] == -1)

        if not np.any(mask):
            logging.debug(f"No open positions found with direction {direction} to close.")
            return

        # Update positions using Numba
        update_positions_numba_parallel(
            positions=positions_subset[mask],
            current_bid=current_bid,
            current_ask=current_ask,
            tick_id=tick_id,
            profit_threshold=profit_threshold,
            loss_threshold=loss_threshold,
            slippage=slippage,
            commission_per_lot=commission_per_lot
        )

        # Identify which positions have been closed
        closed_mask = (positions_subset['close_id'] == tick_id) & (positions_subset['pnl'] != 0.0)
        closed_positions = positions_subset[mask][closed_mask]

        # Update the main positions array
        self.positions_np[:self.current_size][mask] = positions_subset[mask]

        # Update margin used and free margin for each closed position
        for pos in closed_positions:
            required_margin = (pos['open_price'] * pos['volume']) / leverage
            self.margin_used -= required_margin
            self.free_margin += required_margin

            # Update balance with realized PnL
            self.balance += pos['pnl']

        self.update_equity()

        logging.debug(f"Closed {len(closed_positions)} positions by direction {direction} based on PnL thresholds at bid {current_bid} and ask {current_ask}")
    
    def _check_volume_constraints(self, volume):
        """
        Validate the volume against broker's constraints.

        Parameters:
        - volume (float): Volume to be validated.

        Raises:
        - ValueError: If volume is out of allowed range or step size.
        """
        min_vol = self.config.get("min_volume", 0.01)
        max_vol = self.config.get("max_volume", 100)
        step = self.config.get("volume_step", 0.01)
        if volume < min_vol or volume > max_vol:
            raise ValueError(f"Volume {volume} out of allowed range ({min_vol} - {max_vol}).")
        # Check step
        if abs((volume / step) - round(volume / step)) > 1e-9:
            raise ValueError(f"Volume {volume} does not conform to step size {step}.")
    
    def calculate_pnl_vectorized_numba(self):
        """
        Recalculate PnL for all closed positions using Numba.
        """
        with self.lock:
            closed_mask = (self.positions_np[:self.current_size]['close_id'] != -1)
            ixs = np.where(closed_mask)[0]
            if len(ixs) == 0:
                return
            open_prices = self.positions_np[ixs]['open_price']
            close_prices = self.positions_np[ixs]['close_price']
            directions = self.positions_np[ixs]['direction']
            volumes = self.positions_np[ixs]['volume']
            commissions = self.positions_np[ixs]['commission']
            
            # Calculate PnL using Numba
            pnls = calculate_pnl_numba(open_prices, close_prices, directions, volumes, commissions)
            self.positions_np[ixs]['pnl'] = pnls
            logging.debug(f"Updated PnL for {len(ixs)} closed positions using Numba.")
    
    def get_all_positions(self):
        """
        Retrieve all positions managed by the TradeManager.

        Returns:
        - np.ndarray: Structured array of all positions.
        """
        with self.lock:
            return self.positions_np[:self.current_size].copy()
    
    def get_open_positions(self):
        """
        Retrieve all currently open positions.

        Returns:
        - np.ndarray: Structured array of open positions.
        """
        with self.lock:
            mask = (self.positions_np[:self.current_size]['close_id'] == -1)
            open_positions = self.positions_np[:self.current_size][mask]
            return open_positions
        
    def identify_positions_to_close(self, profit_threshold, loss_threshold):
        """
        Identify positions that meet the profit or loss thresholds for closure.

        Parameters:
        - profit_threshold (float): PnL threshold to take profit.
        - loss_threshold (float): PnL threshold to stop loss.

        Returns:
        - list: List of position dictionaries to close.
        """
        with self.lock:
            positions_to_close = []
            for i in range(self.current_size):
                pos = self.positions_np[i]
                if pos['close_id'] == -1:
                    if pos['pnl'] >= profit_threshold or pos['pnl'] <= loss_threshold:
                        positions_to_close.append(pos.copy())
            return positions_to_close

    def get_current_drawdown(self):
        """
        Get the current drawdown from the peak equity.

        Returns:
        - float: Current drawdown value.
        """
        with self.lock:
            return self.peak_equity - self.equity

    def get_max_drawdown(self):
        """
        Get the maximum drawdown observed.

        Returns:
        - float: Maximum drawdown value.
        """
        with self.lock:
            return self.max_drawdown
        
    def get_results(self):
        """
        Retrieve results for analysis.

        Returns:
        - dict: Dictionary containing financial metrics and positions.
        """
        return {
            "balance": self.balance,
            "equity": self.equity,
            "floating_pnl": self.floating_pnl,
            "margin_used": self.margin_used,
            "free_margin": self.free_margin,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.get_current_drawdown(),
            "max_drawdown": self.get_max_drawdown(),
            "positions": self.get_all_positions()
        }