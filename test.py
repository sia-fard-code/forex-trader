import pandas as pd
import numpy as np
import logging
from numba import njit, prange
import threading
from queue import Queue
import time
import random
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# ============================
# Numba-Accelerated Functions
# ============================

@njit(parallel=True)
def update_positions_numba_parallel(
    positions,
    current_bid,
    current_ask,
    t,
    profit_threshold,
    loss_threshold,
    slippage,
    commission_per_lot
):
    """
    Parallelized function to update positions based on current bid and ask prices and PnL thresholds.
    """
    for i in prange(len(positions)):
        if positions[i]['close_id'] == -1:
            direction = positions[i]['direction']
            open_price = positions[i]['open_price']
            volume = positions[i]['volume']
            
            # Determine the appropriate price based on position direction
            if direction == 1:  # Long position
                adjusted_price = current_bid - slippage
            else:  # Short position
                adjusted_price = current_ask + slippage
            
            pnl = (adjusted_price - open_price) * direction * volume - (commission_per_lot * volume)
            
            # Update PnL
            positions[i]['pnl'] = pnl
            
            # Check thresholds
            if pnl >= profit_threshold or pnl <= -loss_threshold:
                positions[i]['close_price'] = adjusted_price
                positions[i]['close_id'] = t
                positions[i]['pnl'] = pnl

@njit
def calculate_pnl_numba(open_prices, close_prices, directions, volumes, commissions):
    """
    Calculate PnL for multiple positions using Numba.

    Parameters:
    - open_prices (np.ndarray): Array of open prices.
    - close_prices (np.ndarray): Array of close prices.
    - directions (np.ndarray): Array of directions (1 or -1).
    - volumes (np.ndarray): Array of volumes.
    - commissions (np.ndarray): Array of commissions.

    Returns:
    - np.ndarray: Array of PnL values.
    """
    pnls = np.empty(len(open_prices), dtype=np.float32)
    for i in range(len(open_prices)):
        pnl = (close_prices[i] - open_prices[i]) * directions[i] * volumes[i] - commissions[i]
        pnls[i] = pnl
    return pnls

# ============================
# TradeManager Class
# ============================

class TradeManager:
    def __init__(self, config, max_positions=1000000):
        """
        Initialize TradeManager with broker configurations and pre-allocate positions array.

        Parameters:
        - config (dict): Configuration dictionary containing broker specs and initial settings.
        - max_positions (int): Maximum number of positions to handle.
        """
        self.config = config
        self.position_dtype = np.dtype([
            ('id', 'i4'),             # Unique identifier (int32)
            ('direction', 'i1'),      # 1 for long, -1 for short (int8)
            ('open_id', 'i8'),      # Tick index or timestamp when opened (int64)
            ('open_price', 'f4'),     # Price at which the position was opened (float32)
            ('volume', 'f4'),         # Lot size (float32)
            ('commission', 'f4'),     # Commission applied at opening (float32)
            ('slippage', 'f4'),       # Slippage applied at opening (float32)
            ('leverage', 'f4'),       # Leverage used (float32)
            ('close_id', 'i8'),     # Tick index or timestamp when closed (-1 if open) (int64)
            ('close_price', 'f4'),    # Price at which the position was closed (float32)
            ('pnl', 'f4')             # Profit and Loss (float32)
        ])
        # Initialize positions array
        self.positions_np = np.empty(max_positions, dtype=self.position_dtype)
        self.current_size = 0
        self.next_id = 0
        self.lock = threading.Lock()
    
    def open_position(self, direction, bid_price, ask_price, volume, t):
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
        self._check_volume_constraints(volume)
        with self.lock:
            if self.current_size >= len(self.positions_np):
                raise MemoryError("Maximum number of positions reached in TradeManager.")
            
            slippage = self.config.get("slippage_points", 0.0001)
            if direction == 1:  # Long position
                open_price_slipped = ask_price + slippage
            else:  # Short position
                open_price_slipped = bid_price - slippage
            
            commission = self.config.get("commission_per_lot", 2.0) * volume
            leverage = self.config.get("leverage", 100)
            
            idx = self.current_size
            self.positions_np[idx]['id'] = self.next_id
            self.positions_np[idx]['direction'] = direction
            self.positions_np[idx]['open_id'] = t
            self.positions_np[idx]['open_price'] = open_price_slipped
            self.positions_np[idx]['volume'] = volume
            self.positions_np[idx]['commission'] = commission
            self.positions_np[idx]['slippage'] = slippage
            self.positions_np[idx]['leverage'] = leverage
            self.positions_np[idx]['close_id'] = -1
            self.positions_np[idx]['close_price'] = np.nan
            self.positions_np[idx]['pnl'] = 0.0
            
            position_id = self.next_id
            self.next_id += 1
            self.current_size += 1
            
            logging.debug(f"Opened position {position_id}: Direction={direction}, Price={open_price_slipped}, Volume={volume}")
            return position_id
    
    def close_position_by_id(self, position_id, bid_price, ask_price, t):
        """
        Close a specific position by ID with thread safety.

        Parameters:
        - position_id (int): ID of the position to close.
        - bid_price (float): Current bid price.
        - ask_price (float): Current ask price.
        - t (int): Current tick index or timestamp.
        """
        with self.lock:
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
            if direction == 1:  # Long position
                close_price_slipped = bid_price - slippage
            else:  # Short position
                close_price_slipped = ask_price + slippage
            
            pnl = (close_price_slipped - self.positions_np[idx]['open_price']) * direction * self.positions_np[idx]['volume'] - (self.config.get("commission_per_lot", 2.0) * self.positions_np[idx]['volume'])
            
            self.positions_np[idx]['close_price'] = close_price_slipped
            self.positions_np[idx]['close_id'] = t
            self.positions_np[idx]['pnl'] = pnl
            
            logging.debug(f"Closed position {position_id}: Close Price={close_price_slipped}, PnL={pnl}")
    
    def close_positions_by_direction_and_pnl(self, direction, profit_threshold, loss_threshold, current_bid, current_ask, t):
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
        
        with self.lock:
            positions_subset = self.positions_np[:self.current_size].copy()
            # Filter positions by direction
            mask = (positions_subset['direction'] == direction) & (positions_subset['close_id'] == -1)
            filtered_positions = positions_subset[mask]
            
            if len(filtered_positions) == 0:
                logging.debug(f"No open positions found with direction {direction} to close.")
                return
            
            # Update positions using Numba
            update_positions_numba_parallel(
                positions=filtered_positions,
                current_bid=current_bid,
                current_ask=current_ask,
                t=t,
                profit_threshold=profit_threshold,
                loss_threshold=loss_threshold,
                slippage=slippage,
                commission_per_lot=commission_per_lot
            )
            
            # Apply updates back to the main positions array
            self.positions_np[:self.current_size][mask] = filtered_positions
            
            logging.debug(f"Closed positions by direction {direction} based on PnL thresholds at bid {current_bid} and ask {current_ask}")
    
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

# ============================
# PositionManager Class (Thread-Based)
# ============================

class PositionManager:
    def __init__(self, config, trade_manager, max_positions=1000000):
        """
        Initialize the PositionManager with configurations and dependencies.

        Parameters:
        - config (dict): Configuration dictionary containing strategy parameters.
        - trade_manager (TradeManager): Instance of TradeManager to execute trades.
        - max_positions (int): Maximum number of positions to handle.
        """
        self.config = config
        self.trade_manager = trade_manager
        self.position_dtype = trade_manager.position_dtype
        self.positions_np = trade_manager.positions_np  # Shared array
        self.current_size = trade_manager.current_size
        self.next_id = trade_manager.next_id
        self.lock = trade_manager.lock  # Shared lock
        
        # Strategy parameters
        self.profit_threshold = self.config.get("profit_threshold", 50.0)
        self.loss_threshold = self.config.get("loss_threshold", 20.0)
        
        # Queue for incoming ticks
        self.tick_queue = Queue()
        
        # Thread for processing ticks
        self.updater_thread = threading.Thread(target=self.updater, daemon=True)
        self.updater_thread.start()
    
    def updater(self):
        """
        Thread that continuously updates positions based on incoming ticks.
        """
        while True:
            tick = self.tick_queue.get()  # Blocking call
            if tick == "STOP":
                break
            current_bid = tick.get('bid')
            current_ask = tick.get('ask')
            t = tick.get('t')
            self.update_positions(current_bid, current_ask, t)
    
    def update_positions(self, current_bid, current_ask, t):
        """
        Recalculate PnL for all open positions and close those meeting strategy criteria.

        Parameters:
        - current_bid (float): Current bid price.
        - current_ask (float): Current ask price.
        - t (int): Current tick index or timestamp.
        """
        profit_threshold = self.profit_threshold
        loss_threshold = self.loss_threshold
        slippage = self.config.get("slippage_points", 0.0001)
        commission_per_lot = self.config.get("commission_per_lot", 2.0)
        
        # Access shared positions array
        with self.lock:
            positions_subset = self.positions_np[:self.current_size]
            # Apply Numba-accelerated update
            update_positions_numba_parallel(
                positions=positions_subset,
                current_bid=current_bid,
                current_ask=current_ask,
                t=t,
                profit_threshold=profit_threshold,
                loss_threshold=loss_threshold,
                slippage=slippage,
                commission_per_lot=commission_per_lot
            )
            # Update the shared positions array based on the modified subset
            self.positions_np[:self.current_size] = positions_subset
            logging.debug(f"Updated and evaluated positions based on tick {t} with bid {current_bid} and ask {current_ask}")
    
    def process_tick(self, tick):
        """
        Add a new tick to the processing queue.

        Parameters:
        - tick (dict): Dictionary containing tick data, e.g., {'bid': 1.1995, 'ask': 1.2005, 't': 1001}
        """
        self.tick_queue.put(tick)
    
    def open_position(self, direction, bid_price, ask_price, volume, t):
        """
        Open a new position.

        Parameters:
        - direction (int): 1 for long, -1 for short.
        - bid_price (float): Current bid price.
        - ask_price (float): Current ask price.
        - volume (float): Lot size.
        - t (int): Current tick index or timestamp.

        Returns:
        - int: ID of the newly opened position.
        """
        position_id = self.trade_manager.open_position(direction, bid_price, ask_price, volume, t)
        logging.debug(f"PositionManager opened position {position_id}")
        return position_id
    
    def close_positions_by_direction_and_pnl(
        self, 
        direction, 
        profit_threshold, 
        loss_threshold, 
        current_bid, 
        current_ask, 
        t
    ):
        """
        Close all positions matching a specific direction and meeting PnL thresholds.

        Parameters:
        - direction (int): 1 for long positions or -1 for short positions.
        - profit_threshold (float): PnL threshold to take profit.
        - loss_threshold (float): PnL threshold to stop loss.
        - current_bid (float): Current bid price.
        - current_ask (float): Current ask price.
        - t (int): Current tick index or timestamp.
        """
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
        self.trade_manager.close_positions_by_direction_and_pnl(
            direction=direction, 
            profit_threshold=profit_threshold, 
            loss_threshold=loss_threshold, 
            current_bid=current_bid, 
            current_ask=current_ask, 
            t=t
        )
        logging.debug(
            f"PositionManager initiated closure of positions with direction {direction} based on PnL thresholds."
        )
    
    def shutdown(self):
        """
        Gracefully shutdown the updater thread.
        """
        self.tick_queue.put("STOP")
        self.updater_thread.join()
        logging.debug("PositionManager updater thread has been stopped.")
    
    def get_all_positions(self):
        """
        Retrieve all positions managed by the PositionManager.

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

# ============================
# Helper Functions
# ============================

def get_current_bid_ask(tick):
    """
    Simulate fetching the current bid and ask prices based on the tick.
    Replace this with actual price retrieval logic.
    
    Parameters:
    - tick (int): Current tick index.
    
    Returns:
    - tuple: (current_bid, current_ask)
    """
    # Simulate base price
    base_price = 1.2000
    price_variation = (tick % 1000) * 0.0001
    random_spread = random.uniform(0.0001, 0.0003)  # Simulated spread between bid and ask
    
    bid_price = base_price + price_variation + random.uniform(-0.0005, 0.0005)
    ask_price = bid_price + random_spread
    return bid_price, ask_price

def should_open_long(tick):
    """
    Determine whether to open a long position.
    Replace this with your actual condition.

    Parameters:
    - tick (int): Current tick index.

    Returns:
    - bool: True if a long position should be opened, False otherwise.
    """
    return tick % 100 == 0  # Example condition: every 100 ticks

def should_open_short(tick):
    """
    Determine whether to open a short position.
    Replace this with your actual condition.

    Parameters:
    - tick (int): Current tick index.

    Returns:
    - bool: True if a short position should be opened, False otherwise.
    """
    return tick % 150 == 0  # Example condition: every 150 ticks

def determine_volume(tick):
    """
    Determine the volume size for a new position.
    Replace this with your actual volume determination logic.

    Parameters:
    - tick (int): Current tick index.

    Returns:
    - float: Volume size.
    """
    return 0.1  # Example fixed volume

def some_profit_condition_long(tick):
    """
    Define the condition to take profit on long positions.
    Replace this with your actual condition.

    Parameters:
    - tick (int): Current tick index.

    Returns:
    - bool: True if profit condition is met, False otherwise.
    """
    return tick % 500 == 0  # Example: every 500 ticks

def some_loss_condition_short(tick):
    """
    Define the condition to stop loss on short positions.
    Replace this with your actual condition.

    Parameters:
    - tick (int): Current tick index.

    Returns:
    - bool: True if loss condition is met, False otherwise.
    """
    return tick % 750 == 0  # Example: every 750 ticks

# ============================
# Precompile Numba Functions
# ============================

def precompile_numba_functions():
    """
    Precompile Numba-accelerated functions with dummy data to avoid runtime compilation in threads.
    """
    dummy_positions = np.empty(1, dtype=[
        ('id', 'i4'),
        ('direction', 'i1'),
        ('open_id', 'i8'),
        ('open_price', 'f4'),
        ('volume', 'f4'),
        ('commission', 'f4'),
        ('slippage', 'f4'),
        ('leverage', 'f4'),
        ('close_id', 'i8'),
        ('close_price', 'f4'),
        ('pnl', 'f4')
    ])
    dummy_positions['close_id'] = -1
    dummy_positions['direction'] = 1
    dummy_positions['open_price'] = 1.2000
    dummy_positions['volume'] = 0.1

    update_positions_numba_parallel(
        positions=dummy_positions,
        current_bid=1.1995,
        current_ask=1.2005,
        t=0,
        profit_threshold=0.005,
        loss_threshold=0.002,
        slippage=0.0001,
        commission_per_lot=2.0
    )

    calculate_pnl_numba(
        open_prices=np.array([1.2000], dtype=np.float32),
        close_prices=np.array([1.2005], dtype=np.float32),
        directions=np.array([1], dtype=np.int8),
        volumes=np.array([0.1], dtype=np.float32),
        commissions=np.array([2.0], dtype=np.float32)
    )

# ============================
# Main Trading Loop
# ============================

def main_trading_loop():
    # Sample configuration
    config = {
        "commission_per_lot": 2.0,
        "slippage_points": 0.0001,
        "leverage": 100,
        "min_volume": 0.01,
        "max_volume": 100,
        "volume_step": 0.01,
        "profit_threshold": 0.005,  # $0.005 profit
        "loss_threshold": 0.002,    # $0.002 loss
        "broker_config": {          # Added broker_config for TradeManager
            "commission_per_lot": 2.0,
            "slippage_points": 0.0001,
            "leverage": 100,
            "min_volume": 0.01,
            "max_volume": 100,
            "volume_step": 0.01,
            "profit_threshold": 0.005,
            "loss_threshold": 0.002
        }
    }
    
    # Precompile Numba functions
    precompile_numba_functions()
    
    # Initialize TradeManager
    trade_manager = TradeManager(config['broker_config'], max_positions=1000000)
    
    # Initialize PositionManager
    position_manager = PositionManager(config, trade_manager, max_positions=1000000)
    
    try:
        # Simulate trading ticks
        for tick in range(1, 1000001):  # Simulating 1,000,000 ticks
            # Fetch current bid and ask prices
            current_bid, current_ask = get_current_bid_ask(tick)
            
            # Create a tick dictionary
            tick_data = {'bid': current_bid, 'ask': current_ask, 't': tick}
            
            # Process the tick
            position_manager.process_tick(tick_data)
            
            # Example: Open a long position based on some condition
            if should_open_long(tick):
                volume = determine_volume(tick)
                position_manager.open_position(direction=1, bid_price=current_bid, ask_price=current_ask, volume=volume, t=tick)
            
            # Example: Open a short position based on another condition
            if should_open_short(tick):
                volume = determine_volume(tick)
                position_manager.open_position(direction=-1, bid_price=current_bid, ask_price=current_ask, volume=volume, t=tick)
            
            # Example: Close positions by direction and PnL thresholds based on strategy signals
            if some_profit_condition_long(tick):
                position_manager.close_positions_by_direction_and_pnl(
                    direction=1,
                    profit_threshold=config["profit_threshold"],
                    loss_threshold=config["loss_threshold"],
                    current_bid=current_bid,
                    current_ask=current_ask,
                    t=tick
                )
            
            if some_loss_condition_short(tick):
                position_manager.close_positions_by_direction_and_pnl(
                    direction=-1,
                    profit_threshold=config["profit_threshold"],
                    loss_threshold=config["loss_threshold"],
                    current_bid=current_bid,
                    current_ask=current_ask,
                    t=tick
                )
            
            # Periodically update PnL for all closed positions
            if tick % 1000 == 0:
                trade_manager.calculate_pnl_vectorized_numba()
                logging.info(f"Processed tick {tick}")
            
            # Simulate waiting for EGARCH model retraining (placeholder)
            # In a real scenario, this would be replaced with actual model retraining logic
            if tick % 250000 == 0:
                logging.info(f"Tick {tick}: Retraining EGARCH model...")
                time.sleep(2)  # Simulate time delay for retraining
                
    except KeyboardInterrupt:
        logging.info("Trading loop terminated by user.")
    finally:
        # Gracefully shutdown the PositionManager
        position_manager.shutdown()
        logging.info("Trading system has been shut down.")
        
        # Optionally, retrieve and log final positions
        all_positions = trade_manager.get_all_positions()
        logging.info(f"Total positions opened: {all_positions.shape[0]}")
        open_positions = trade_manager.get_open_positions()
        logging.info(f"Total open positions: {open_positions.shape[0]}")
        closed_positions = all_positions[all_positions['close_id'] != -1]
        logging.info(f"Total closed positions: {closed_positions.shape[0]}")
        
        # Example: Print first 5 closed positions
        if closed_positions.shape[0] > 0:
            logging.info("Sample closed positions:")
            for pos in closed_positions[:5]:
                logging.info(f"ID: {int(pos['id'])}, Direction: {int(pos['direction'])}, Open Price: {pos['open_price']}, Close Price: {pos['close_price']}, PnL: {pos['pnl']}")

# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    main_trading_loop()