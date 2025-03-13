import pandas as pd
import numpy as np
from PositionManager import PositionManager
from TradeManager import TradeManager
from MarketProcessing import MarketProcessing
from StateIdentifier import StateIdentifier
from PositionSizing import PositionSizing
from DataManager import DataManager
from PositionClosureHandler import PositionClosureHandler
from PositionOpeningHandler import PositionOpeningHandler
from ProfitabilityFilter import ProfitabilityFilter
import logging
import time
import threading
import queue
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tick_times = []

class TradingStrategyWithFilter:
    def __init__(self, config):
        """
        Initialize the TradingStrategy class with integrated components and profitability filter.
        :param config: Dictionary of configuration parameters.
        """
        self.data_manager = DataManager(config)

        # Initialize components for bid and ask
        self.market_processing_bid = MarketProcessing(self.data_manager, "bid")
        self.market_processing_ask = MarketProcessing(self.data_manager, "ask")
        self.state_identifier = StateIdentifier(self.data_manager)
        self.position_sizing_bid = PositionSizing(self.data_manager, "bid")
        self.position_sizing_ask = PositionSizing(self.data_manager, "ask")
        trade_manager = TradeManager(config['broker_config'], max_positions=100000, data_manager=self.data_manager)
        position_manager = PositionManager(config['broker_config'], trade_manager, max_positions=100000)

        # Initialize the profitability filter
        self.profitability_filter = ProfitabilityFilter(self.data_manager, config)

        # Initialize state variables
        self.config = config
        self.equity = config['broker_config']["initial_capital"]
        self.free_margin = self.equity
        self.margin_used = 0.0
        self.position_manager = position_manager
        self.trade_manager = trade_manager
        self.drawdown_threshold = self.config.get("drawdown_threshold", 0.10)

        # Initialize Closure and Opening Handlers
        self.position_closure_handler = PositionClosureHandler(self.trade_manager, self.config.get("broker_config", {}))
        self.position_opening_handler = PositionOpeningHandler(self.trade_manager, self.config.get("broker_config", {}))

        # Initialize the queue and worker threads
        self.tick_queue = queue.Queue()
        self.num_workers = 1
        self.workers = []
        self.shutdown_event = threading.Event()
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self.worker_process, daemon=True)
            worker.start()
            self.workers.append(worker)

        # Event to signal shutdown
        self.shutdown_event = threading.Event()

    def process_tick(self, t, bid, ask, tick_id):
        """
        Process a single tick and execute the trading strategy with profitability filtering.
        :param t: Current tick index.
        :param bid: Bid price at the current tick.
        :param ask: Ask price at the current tick.
        :param tick_id: Unique identifier for the tick.
        """
        
        # Create a tick dictionary
        tick_data = {'tick_id': tick_id, 'bid': bid, 'ask': ask, 't': t}
        # Update DataManager with the new tick
        self.data_manager.update(t, bid=bid, ask=ask, tick_id=tick_id)
        
        # Process the tick
        self.position_manager.process_tick(tick_data)

        # Process market data for bid and ask
        self.market_processing_bid.process_tick(t)
        self.market_processing_ask.process_tick(t)

        # Check if training is complete using is_trained from DataManager
        if self.data_manager.get_config("is_trained"):
            self.state_identifier.process(t, bid, ask)

            # Get the refined state from DataManager
            refined_state_series = self.data_manager.get("refined_state")
            refined_state = refined_state_series[-1]
            span = 50  # Or a configurable span from the strategy's configuration
            
            # Get volatility data for filtering
            forecasted_vol_bid = self.data_manager.get_latest_value("forecasted_vol_bid")
            forecasted_vol_ask = self.data_manager.get_latest_value("forecasted_vol_ask")
            
            # Apply profitability filter to determine if we should trade
            should_trade = self.profitability_filter.process(t, refined_state, forecasted_vol_bid, forecasted_vol_ask)
            
            # Only proceed with position sizing and opening if filter approves
            if should_trade:
                self.position_sizing_bid.process(t, refined_state)
                self.position_sizing_ask.process(t, refined_state)
                bid_position_sizes = np.array(self.data_manager.get("position_size_bid", span))
                ask_position_sizes = np.array(self.data_manager.get("position_size_ask", span))
                f = 1
                
                if refined_state == 1:
                    # Calculate position sizes
                    bid_input = bid_position_sizes + (-ask_position_sizes)
                    bid_ewma = self.calculate_ewma(bid_input, span)
                    bid_ewma_filtered = np.maximum(bid_ewma, 0)
                    smoothed_bid_positions = bid_ewma_filtered * bid_position_sizes
                    self.data_manager.set("adjusted_position_size_bid", smoothed_bid_positions[-1])
                    if not np.isnan(smoothed_bid_positions[-1]):
                        self.position_opening_handler.handle_openings(refined_state*1, bid, ask, tick_id, f*smoothed_bid_positions[-1])
                
                elif refined_state == -1:
                    ask_input = ask_position_sizes + (-bid_position_sizes)
                    ask_ewma = self.calculate_ewma(ask_input, span)
                    ask_ewma_filtered = np.maximum(ask_ewma, 0)
                    smoothed_ask_positions = ask_ewma_filtered * ask_position_sizes
                    self.data_manager.set("adjusted_position_size_ask", smoothed_ask_positions[-1])
                    if not np.isnan(smoothed_ask_positions[-1]):
                        self.position_opening_handler.handle_openings(refined_state*1, bid, ask, tick_id, f*smoothed_ask_positions[-1])
            else:
                # Log that the trade was filtered out
                logging.debug(f"t={t}: Trade filtered out by profitability filter.")
                # Set position sizes to 0 when filtered out
                self.data_manager.set("adjusted_position_size_bid", 0)
                self.data_manager.set("adjusted_position_size_ask", 0)

            if t % 1000 == 0 and t != 0:
                # Log filter statistics periodically
                filter_stats = self.profitability_filter.get_statistics()
                logging.info(f"t={t}: Filter stats - Approved: {filter_stats['trades_approved']}, "
                           f"Rejected: {filter_stats['trades_rejected']}, "
                           f"Approval rate: {filter_stats['approval_rate']:.2f}")
                logging.info(f"t={t}: jj={self.position_sizing_ask.jj}, Margin Used={'margin_used'}, Free Margin={'free_margin'}")
        else:
            logging.debug(f"t={t}: Training in progress. Skipping state determination and position sizing.")

    # Assuming combined_positions is a NumPy array or pandas Series
    def calculate_ewma(self, position_sizes, span):
        # Convert to pandas Series for simplicity
        position_series = pd.Series(position_sizes)
        # Calculate EWMA
        ewma_positions = position_series.ewm(span=span, adjust=False).mean()
        return ewma_positions.values  # Convert back to NumPy array if needed

    def halt_trading(self):
        """
        Implement actions to take when drawdown threshold is reached.
        """
        logging.info("Halting trading due to drawdown threshold.")

    def worker_process(self):
        while not self.shutdown_event.is_set():
            try:
                # Wait for a tick to process
                update_interval = 100
                tick_data = self.tick_queue.get(timeout=1)  # Timeout to check for shutdown
                tick_start = time.time()
                self.process_tick(tick_data['t'], tick_data['bid'], tick_data['ask'], tick_data['tick_id'])
                tick_end = time.time()
                tick_times.append(tick_end - tick_start)
                self.tick_queue.task_done()

                equity = self.data_manager.get_latest_value('equity')
                balance_val = self.data_manager.get_latest_value('balance')
                pnl_val = self.data_manager.get_latest_value('pnl')
                position_size_bid = self.data_manager.get_latest_value('adjusted_position_size_bid')
                position_size_ask = self.data_manager.get_latest_value('adjusted_position_size_ask')
                positions_df = pd.DataFrame(self.trade_manager.get_all_positions())
                        
            except queue.Empty:
                continue  # Check for shutdown_event

    def run_strategy(self, data):
        """
        Run the strategy on a dataset using a queue-based approach.
        :param data: DataFrame with 'tick_id', 'bid', and 'ask' columns.
        """
        logging.info("Starting strategy execution with profitability filtering...")
        start_time = time.time()

        for idx, row in data.iterrows():
            tick_id = row["tick_id"]
            bid = row["bid"]
            ask = row["ask"]

            # Enqueue the tick for processing
            self.tick_queue.put({'t': idx, 'bid': bid, 'ask': ask, 'tick_id': tick_id})

        # Wait until all ticks are processed
        self.tick_queue.join()

        end_time = time.time()
        total_execution_time = end_time - start_time
        average_tick_time = np.mean(tick_times)
        plt.ioff()
        plt.show()
        logging.info(f"Total Execution Time: {total_execution_time:.4f} seconds")
        logging.info(f"Average Time per Tick: {average_tick_time*1000:.6f} milliseconds")

        # Log final filter statistics
        filter_stats = self.profitability_filter.get_statistics()
        logging.info(f"Final Filter Statistics:")
        logging.info(f"  Total trades evaluated: {filter_stats['total_trades_evaluated']}")
        logging.info(f"  Trades approved: {filter_stats['trades_approved']}")
        logging.info(f"  Trades rejected: {filter_stats['trades_rejected']}")
        logging.info(f"  Approval rate: {filter_stats['approval_rate']:.2f}")
        logging.info(f"  State probabilities: Bullish={filter_stats['state_probabilities'][1]:.2f}, "
                   f"Bearish={filter_stats['state_probabilities'][-1]:.2f}, "
                   f"Neutral={filter_stats['state_probabilities'][0]:.2f}")

    def shutdown(self):
        """
        Shutdown the worker threads and perform any necessary cleanup.
        """
        logging.info("Shutting down TradingStrategy.")

        # Signal all workers to shutdown
        self.shutdown_event.set()

        # Wait for all workers to finish
        for worker in self.workers:
            worker.join()

        logging.info("TradingStrategy shutdown complete.")

    def get_results(self):
        """
        Retrieve results for analysis.
        :return: Dictionary containing equity curve, positions, and other metrics.
        """
        # Get filter statistics
        filter_stats = self.profitability_filter.get_statistics()
        
        return {
            "refined_state": self.data_manager.get("refined_state"),
            "positions": self.trade_manager.get_all_positions(),
            "equity_curve": self.data_manager.get("equity"),
            "balance": self.data_manager.get("balance"),
            "positions_bid": self.data_manager.get("position_size_bid"),
            "positions_ask": self.data_manager.get("position_size_ask"),
            "adjusted_position_size_bid": self.data_manager.get("adjusted_position_size_bid"),
            "adjusted_position_size_ask": self.data_manager.get("adjusted_position_size_ask"),
            "pnl": self.data_manager.get("pnl"),
            "margin_used": self.data_manager.get("margin_used"),
            "free_margin": self.data_manager.get("free_margin"),
            "filter_statistics": filter_stats
        }
