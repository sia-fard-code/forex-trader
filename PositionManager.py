import logging
import threading
from queue import Queue
# from TradeManager import TradeManager

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
        # self.updater_thread.start()

    def process_all_ticks(self, tick):
        """
        Process all ticks sequentially for debugging purposes.
        """
        bid = tick['bid']
        ask = tick['ask']
        t = tick['t']
        tick_id = tick['tick_id']
        self.update_positions(bid, ask, tick_id)
    
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
            tick_id = tick.get('tick_id')
            self.update_positions(current_bid, current_ask, tick_id)
    
    def update_positions(self, current_bid, current_ask, tick_id):
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

        # Delegate to TradeManager
        self.trade_manager.update_positions(
            current_bid=current_bid,
            current_ask=current_ask,
            tick_id=tick_id,
            # profit_threshold=profit_threshold,
            # loss_threshold=loss_threshold,
            slippage=slippage,
            commission_per_lot=commission_per_lot,
            # closing_method="stopLoss"
        )
            
        logging.debug(f"Updated and evaluated positions based on tick {tick_id} with bid {current_bid} and ask {current_ask}")
    
    def process_tick(self, tick):
        """
        Add a new tick to the processing queue.

        Parameters:
        - tick (dict): Dictionary containing tick data, e.g., {'bid': 1.1995, 'ask': 1.2005, 't': 1001}
        """
        self.process_all_ticks(tick)
        # self.tick_queue.put(tick)
    
    def open_position(self, direction, bid_price, ask_price, volume, tick_id):
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
        position_id = self.trade_manager.open_position(direction, bid_price, ask_price, volume, tick_id)
        logging.debug(f"PositionManager opened position {position_id}")
        return position_id
    
    def close_positions_by_direction_and_pnl(
        self, 
        direction, 
        profit_threshold, 
        loss_threshold, 
        current_bid, 
        current_ask, 
        tick_id
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
            tick_id=tick_id
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