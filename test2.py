import numpy as np
import logging
from numba import njit, prange
import multiprocessing
from multiprocessing import Process, Queue, Lock, Value
import threading
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
    current_price,
    t,
    profit_threshold,
    loss_threshold,
    slippage,
    commission_per_lot
):
    """
    Parallelized function to update positions based on current price and PnL thresholds.
    """
    for i in prange(len(positions)):
        if positions['close_id'][i] == -1:
            direction = positions['direction'][i]
            open_price = positions['open_price'][i]
            volume = positions['volume'][i]
            adjusted_price = current_price - slippage * direction
            pnl = (adjusted_price - open_price) * direction * volume - (commission_per_lot * volume)
            
            # Update PnL
            positions['pnl'][i] = pnl
            
            # Check thresholds
            if pnl >= profit_threshold or pnl <= -loss_threshold:
                positions['close_price'][i] = adjusted_price
                positions['close_id'][i] = t
                positions['pnl'][i] = pnl

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
        self.positions = multiprocessing.Array('d', self.position_dtype.itemsize * max_positions, lock=False)
        self.positions_np = np.frombuffer(self.positions.get_obj(), dtype=np.float64).reshape(max_positions, len(self.position_dtype.names))
        self.current_size = Value('i', 0)
        self.next_id = Value('i', 0)
        self.lock = Lock()
    
    def open_position(self, direction, open_price, volume, t):
        """
        Open a new position with thread safety.

        Parameters:
        - direction (int): 1 for long, -1 for short.
        - open_price (float): Price at which the position is opened.
        - volume (float): Lot size.
        - t (int): Current tick index or timestamp.

        Returns:
        - int: ID of the newly opened position.
        """
        self._check_volume_constraints(volume)
        with self.lock:
            if self.current_size.value >= len(self.positions_np):
                raise MemoryError("Maximum number of positions reached in TradeManager.")
            
            slippage = self.config.get("slippage_points", 0.0001)
            open_price_slipped = open_price + slippage * direction
            commission = self.config.get("commission_per_lot", 2.0) * volume
            leverage = self.config.get("leverage", 100)
            
            idx = self.current_size.value
            self.positions_np[idx][0] = self.next_id.value      # id
            self.positions_np[idx][1] = direction               # direction
            self.positions_np[idx][2] = t                       # open_id
            self.positions_np[idx][3] = open_price_slipped      # open_price
            self.positions_np[idx][4] = volume                  # volume
            self.positions_np[idx][5] = commission              # commission
            self.positions_np[idx][6] = slippage                # slippage
            self.positions_np[idx][7] = leverage                # leverage
            self.positions_np[idx][8] = -1                      # close_id
            self.positions_np[idx][9] = np.nan                   # close_price
            self.positions_np[idx][10] = 0.0                     # pnl
            
            position_id = self.next_id.value
            self.next_id.value += 1
            self.current_size.value += 1
            
            logging.debug(f"Opened position {position_id}: Direction={direction}, Price={open_price_slipped}, Volume={volume}")
            return position_id
    
    def close_position_by_id(self, position_id, close_price, t):
        """
        Close a specific position by ID with thread safety.

        Parameters:
        - position_id (int): ID of the position to close.
        - close_price (float): Current closing price.
        - t (int): Current tick index or timestamp.
        """
        with self.lock:
            idx = -1
            for i in range(self.current_size.value):
                if self.positions_np[i][0] == position_id and self.positions_np[i][8] == -1:
                    idx = i
                    break
            if idx == -1:
                logging.warning(f"No open position found with id {position_id}")
                return
            direction = int(self.positions_np[idx][1])
            slippage = self.positions_np[idx][6]
            close_price_slipped = close_price - slippage * direction
            
            pnl = (close_price_slipped - self.positions_np[idx][3]) * direction * self.positions_np[idx][4] - (self.config.get("commission_per_lot", 2.0) * self.positions_np[idx][4])
            
            self.positions_np[idx][9] = close_price_slipped    # close_price
            self.positions_np[idx][8] = t                      # close_id
            self.positions_np[idx][10] = pnl                    # pnl
            
            logging.debug(f"Closed position {position_id}: Close Price={close_price_slipped}, PnL={pnl}")
    
    def close_positions_by_direction_and_pnl(self, direction, profit_threshold, loss_threshold, current_price, t):
        """
        Close all open positions matching the specified direction and meeting PnL thresholds.

        Parameters:
        - direction (int): 1 for long positions or -1 for short positions.
        - profit_threshold (float): PnL threshold to take profit.
        - loss_threshold (float): PnL threshold to stop loss.
        - current_price (float): Current market price.
        - t (int): Current tick index or timestamp.
        """
        slippage = self.config.get("slippage_points", 0.0001)
        commission_per_lot = self.config.get("commission_per_lot", 2.0)
        
        with self.lock:
            positions_subset = self.positions_np[:self.current_size.value].copy()
            update_positions_numba_parallel(
                positions_subset,
                current_price,
                t,
                profit_threshold,
                loss_threshold,
                slippage,
                commission_per_lot
            )
            # Update the shared positions array based on the modified subset
            self.positions_np[:self.current_size.value] = positions_subset
            logging.debug(f"Closed positions by direction {direction} based on PnL thresholds at price {current_price}")
    
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
            closed_mask = (self.positions_np[:self.current_size.value][:,8] != -1)
            ixs = np.where(closed_mask)[0]
            if len(ixs) == 0:
                return
            open_prices = self.positions_np[ixs, 3]
            close_prices = self.positions_np[ixs, 9]
            directions = self.positions_np[ixs, 1]
            volumes = self.positions_np[ixs, 4]
            commissions = self.positions_np[ixs, 5]
            
            # Calculate PnL using Numba
            pnls = calculate_pnl_numba(open_prices, close_prices, directions, volumes, commissions)
            self.positions_np[ixs, 10] = pnls
            logging.debug(f"Updated PnL for {len(ixs)} closed positions using Numba.")
    
    def _calculate_pnl(self, idx):
        """
        Calculate PnL for a single position at index idx.

        Parameters:
        - idx (int): Index of the position.

        Returns:
        - float: Calculated PnL.
        """
        direction = int(self.positions_np[idx][1])
        open_price = self.positions_np[idx][3]
        close_price = self.positions_np[idx][9]
        volume = self.positions_np[idx][4]
        commission = self.positions_np[idx][5]
        
        pnl = (close_price - open_price) * direction * volume - commission
        return pnl
    
    def get_all_positions(self):
        """
        Retrieve all positions managed by the TradeManager.

        Returns:
        - np.ndarray: Structured array of all positions.
        """
        with self.lock:
            return self.positions_np[:self.current_size.value].copy()
    
    def get_open_positions(self):
        """
        Retrieve all currently open positions.

        Returns:
        - np.ndarray: Structured array of open positions.
        """
        with self.lock:
            mask = (self.positions_np[:self.current_size.value][:,8] == -1)
            open_positions = self.positions_np[:self.current_size.value][mask]
            return open_positions

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
# PositionManager Class
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
        self.positions_np = trade_manager.positions_np  # Shared memory
        self.current_size = trade_manager.current_size
        self.next_id = trade_manager.next_id
        self.lock = trade_manager.lock  # Shared lock
        
        # Strategy parameters
        self.profit_threshold = self.config.get("profit_threshold", 50.0)
        self.loss_threshold = self.config.get("loss_threshold", 20.0)
        
        # Multiprocessing queue for incoming ticks
        self.tick_queue = Queue()
        
        # Start the position updater process
        self.updater_process = Process(target=self.updater, args=())
        self.updater_process.start()
    
    def updater(self):
        """
        Process that continuously updates positions based on incoming ticks.
        """
        while True:
            tick = self.tick_queue.get()  # Blocking call
            if tick == "STOP":
                break
            current_price = tick.get('price')
            t = tick.get('t')
            self.update_positions(current_price, t)
    
    def update_positions(self, current_price, t):
        """
        Recalculate PnL for all open positions and close those meeting strategy criteria.

        Parameters:
        - current_price (float): Current market price.
        - t (int): Current tick index or timestamp.
        """
        profit_threshold = self.profit_threshold
        loss_threshold = self.loss_threshold
        slippage = self.config.get("slippage_points", 0.0001)
        commission_per_lot = self.config.get("commission_per_lot", 2.0)
        
        # Access shared positions array
        with self.lock:
            positions_subset = self.positions_np[:self.current_size.value]
            update_positions_numba_parallel(
                positions_subset,
                current_price,
                t,
                profit_threshold,
                loss_threshold,
                slippage,
                commission_per_lot
            )
            # Update the shared positions array based on the modified subset
            self.positions_np[:self.current_size.value] = positions_subset
            logging.debug(f"Updated and evaluated positions based on tick {t} with price {current_price}")
    
    def process_tick(self, tick):
        """
        Add a new tick to the processing queue.

        Parameters:
        - tick (dict): Dictionary containing tick data, e.g., {'price': 1.2100, 't': 1001}
        """
        self.tick_queue.put(tick)
    
    def open_position(self, direction, open_price, volume, t):
        """
        Open a new position.

        Parameters:
        - direction (int): 1 for long, -1 for short.
        - open_price (float): Price at which the position is opened.
        - volume (float): Lot size.
        - t (int): Current tick index or timestamp.

        Returns:
        - int: ID of the newly opened position.
        """
        position_id = self.trade_manager.open_position(direction, open_price, volume, t)
        logging.debug(f"PositionManager opened position {position_id}")
        return position_id
    
    def close_positions_by_direction_and_pnl(
        self, 
        direction, 
        profit_threshold, 
        loss_threshold, 
        current_price, 
        t
    ):
        """
        Close all positions matching a specific direction and meeting PnL thresholds.

        Parameters:
        - direction (int): 1 for long positions or -1 for short positions.
        - profit_threshold (float): PnL threshold to take profit.
        - loss_threshold (float): PnL threshold to stop loss.
        - current_price (float): Current market price.
        - t (int): Current tick index or timestamp.
        """
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
        self.trade_manager.close_positions_by_direction_and_pnl(
            direction, 
            profit_threshold, 
            loss_threshold, 
            current_price, 
            t
        )
        logging.debug(
            f"PositionManager initiated closure of positions with direction {direction} based on PnL thresholds."
        )
    
    def shutdown(self):
        """
        Gracefully shutdown the updater process.
        """
        self.tick_queue.put("STOP")
        self.updater_process.join()
        logging.debug("PositionManager updater process has been stopped.")
    
    def get_all_positions(self):
        """
        Retrieve all positions managed by the PositionManager.

        Returns:
        - np.ndarray: Structured array of all positions.
        """
        with self.lock:
            return self.positions_np[:self.current_size.value].copy()
    
    def get_open_positions(self):
        """
        Retrieve all currently open positions.

        Returns:
        - np.ndarray: Structured array of open positions.
        """
        with self.lock:
            mask = (self.positions_np[:self.current_size.value][:,8] == -1)
            open_positions = self.positions_np[:self.current_size.value][mask]
            return open_positions

# ============================
# Helper Functions
# ============================

def get_current_price(tick):
    """
    Simulate fetching the current market price based on the tick.
    Replace this with actual price retrieval logic.

    Parameters:
    - tick (int): Current tick index.

    Returns:
    - float: Current market price.
    """
    # Simulate price with some randomness
    base_price = 1.2000
    price_variation = (tick % 1000) * 0.0001
    random_noise = random.uniform(-0.0005, 0.0005)
    return base_price + price_variation + random_noise

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
        "loss_threshold": 0.002     # $0.002 loss
    }
    
    # Initialize TradeManager
    trade_manager = TradeManager(config, max_positions=1000000)
    
    # Initialize PositionManager
    position_manager = PositionManager(config, trade_manager, max_positions=1000000)
    
    try:
        # Simulate trading ticks
        for tick in range(1, 1000001):  # Simulating 1,000,000 ticks
            # Fetch current price from DataManager (placeholder)
            current_price = get_current_price(tick)
            
            # Create a tick dictionary
            tick_data = {'price': current_price, 't': tick}
            
            # Process the tick
            position_manager.process_tick(tick_data)
            
            # Example: Open a long position based on some condition
            if should_open_long(tick):
                volume = determine_volume(tick)
                position_manager.open_position(direction=1, open_price=current_price, volume=volume, t=tick)
            
            # Example: Open a short position based on another condition
            if should_open_short(tick):
                volume = determine_volume(tick)
                position_manager.open_position(direction=-1, open_price=current_price, volume=volume, t=tick)
            
            # Example: Close positions by direction and PnL thresholds based on strategy signals
            if some_profit_condition_long(tick):
                position_manager.close_positions_by_direction_and_pnl(
                    direction=1,
                    profit_threshold=config["profit_threshold"],
                    loss_threshold=config["loss_threshold"],
                    current_price=current_price,
                    t=tick
                )
            
            if some_loss_condition_short(tick):
                position_manager.close_positions_by_direction_and_pnl(
                    direction=-1,
                    profit_threshold=config["profit_threshold"],
                    loss_threshold=config["loss_threshold"],
                    current_price=current_price,
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
        closed_positions = all_positions[all_positions[:,8] != -1]
        logging.info(f"Total closed positions: {closed_positions.shape[0]}")
        
        # Example: Print first 5 closed positions
        if closed_positions.shape[0] > 0:
            logging.info("Sample closed positions:")
            for pos in closed_positions[:5]:
                logging.info(f"ID: {int(pos[0])}, Direction: {int(pos[1])}, Open Price: {pos[3]}, Close Price: {pos[9]}, PnL: {pos[10]}")

# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.set_start_method("fork")
    main_trading_loop()
