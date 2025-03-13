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
from LivePlot import LivePlotWithSlider

import logging
import time
import threading
import queue
import matplotlib.pyplot as plt
# from precompile_numba import precompile_numba_functions
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tick_times = []

class TradingStrategy:
    def __init__(self, config):
        """
        Initialize the TradingStrategy class with integrated components.
        :param config: Dictionary of configuration parameters.
        """
        self.data_manager = DataManager(config)
        # self.live_plot = LivePlotWithSlider(window_size=500)

        # Initialize components for bid and ask
        self.market_processing_bid = MarketProcessing(self.data_manager, "bid")
        self.market_processing_ask = MarketProcessing(self.data_manager, "ask")
        self.state_identifier = StateIdentifier(self.data_manager)
        self.position_sizing_bid = PositionSizing(self.data_manager, "bid")  # Process bid 
        self.position_sizing_ask = PositionSizing(self.data_manager, "ask")  # Process ask 
        trade_manager = TradeManager(config['broker_config'], max_positions=100000, data_manager=self.data_manager)
        position_manager = PositionManager(config['broker_config'], trade_manager, max_positions=100000)


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
        # broker_config = {
        #     "commission_per_lot": 2.0,
        #     "slippage_points": 0.0001,
        #     "leverage": 100,
        #     "min_volume": 0.01,
        #     "max_volume": 100,
        #     "volume_step": 0.01,
        #     "profit_threshold": 0.005,  # $0.005 profit
        #     "loss_threshold": 0.002     # $0.002 loss
        # }

    def process_tick(self, t, bid, ask,tick_id):
        """
        Process a single tick and execute the trading strategy.
        :param t: Current tick index.
        :param bid: Bid price at the current tick.
        :param ask: Ask price at the current tick.
        """
        
        # Create a tick dictionary
        tick_data = {'tick_id':tick_id, 'bid': bid,'ask': ask, 't': t}
        # Update DataManager with the new tick
        self.data_manager.update(t, bid=bid, ask=ask, tick_id=tick_id)
        
        # Process the tick
        self.position_manager.process_tick(tick_data)
        # self.position_manager.update_positions(tick_data)


        # Process market data for bid and ask
        self.market_processing_bid.process_tick(t)
        self.market_processing_ask.process_tick(t)

        # Check if training is complete using is_trained from DataManager
        if self.data_manager.get_config("is_trained"):
            self.state_identifier.process(t, bid, ask)

            # Get the refined state from DataManager
            refined_state_series = self.data_manager.get("refined_state")
            refined_state = refined_state_series[-1]# if not refined_state_series.empty else 0
            span = 50  # Or a configurable span from the strategy's configuration
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
                    self.position_opening_handler.handle_openings(refined_state*1, bid, ask, tick_id,f*smoothed_bid_positions[-1])
            elif refined_state == -1:
                ask_input = ask_position_sizes + (-bid_position_sizes)
                ask_ewma = self.calculate_ewma(ask_input, span)
                ask_ewma_filtered = np.maximum(ask_ewma, 0)
                smoothed_ask_positions = ask_ewma_filtered * ask_position_sizes
                self.data_manager.set("adjusted_position_size_ask", smoothed_ask_positions[-1])
                if not np.isnan(smoothed_ask_positions[-1]):
                    self.position_opening_handler.handle_openings(refined_state*1, bid, ask, tick_id,f*smoothed_ask_positions[-1])

            # self.position_closure_handler.handle_closures(refined_state, tick_id, bid, ask)

            # # Log the results for the tick
            # equity = self.data_manager.get("equity").iloc[-1] if not self.data_manager.get("equity").empty else self.equity
            # margin_used = self.data_manager.get("margin_used").iloc[-1] if not self.data_manager.get("margin_used").empty else 0.0
            # free_margin = self.data_manager.get("free_margin").iloc[-1] if not self.data_manager.get("free_margin").empty else self.free_margin
            # current_drawdown = self.trade_manager.get_current_drawdown()
            # if current_drawdown / self.trade_manager.peak_equity >= self.drawdown_threshold:
            #     logging.warning(f"Drawdown threshold reached: {current_drawdown} ({(current_drawdown / self.trade_manager.peak_equity) * 100:.2f}%)")
            #     # Implement risk management actions, e.g., halt trading, reduce position sizes, etc.
            #     self.halt_trading()

            if t % 1000 == 0 and t != 0:
                # logging.info(f"t={t}: Equity={equity}, Margin Used={margin_used}, Free Margin={free_margin}")
                logging.info(f"t={t}: jj={self.position_sizing_ask.jj}, Margin Used={'margin_used'}, Free Margin={'free_margin'}")
        else:
            logging.debug(f"t={t}: Training in progress. Skipping state determination and position sizing.")


    # Assuming combined_positions is a NumPy array or pandas Series
    def calculate_ewma(self,position_sizes, span):
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
                

                # Append new data to the plotting class's deques
                # self.live_plot.timestamps.append(tick_data['t'])
                # self.live_plot.bid.append(tick_data['bid'])
                # self.live_plot.ask.append(tick_data['ask'])
                # self.live_plot.equity_curve.append(equity)
                # self.live_plot.balance.append(balance_val)
                # self.live_plot.pnl.append(pnl_val)
                # self.live_plot.position_size_bid.append(position_size_bid)
                # self.live_plot.position_size_ask.append(position_size_ask)
                
                # # Update positions DataFrame if there are new positions
                # if not positions_df.empty:
                #     self.live_plot.positions = positions_df.copy()
                
                # # Update slider maximum based on current data length
                # self.live_plot.slider.valmax = len(self.live_plot.timestamps) - 1
                # self.live_plot.slider.ax.set_xlim(self.live_plot.slider.valmin, self.live_plot.slider.valmax)
                
                # # Update the plot at specified intervals
                # if tick_data['tick_id'] % update_interval == 0:
                #     self.live_plot.update_plot(tick_data['tick_id'])
                #     logging.info(f"Processed tick {tick_data['tick_id']}")
                        
            except queue.Empty:
                continue  # Check for shutdown_event

    def run_strategy(self, data):
        """
        Run the strategy on a dataset using a queue-based approach.
        :param data: DataFrame with 'tick_id', 'bid', and 'ask' columns.
        """
        logging.info("Starting strategy execution...")
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

    def shutdown(self):
        """
        Shutdown the worker threads and perform any necessary cleanup.
        """
        logging.info("Shutting down TradingStrategy.")
        self.thread_manager.shutdown()

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
            
        }
    
