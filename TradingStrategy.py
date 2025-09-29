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
from RealtimeTickSimulator import RealtimeTickSimulator

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
        self.approved = 0
        self.not_approved = 0
        self.adaptive_confidence_threshold = 0.95 # Initial value
        self.adaptive_profitability_factor = config.get("profitability_factor", 1.5) # Initial value
        trade_manager = TradeManager(config['broker_config'], max_positions=100000, data_manager=self.data_manager)
        position_manager = PositionManager(config['broker_config'], trade_manager, max_positions=100000)
        self.processing_times = []
        self.last_tick_time = 0
        self.tick_count = 0
        self.tick_simulator = None
        self.training_tick_count = 0
        self.post_training_tick_count = 0
        self.dropped_tick_count = 0
        self.last_processed_timestamp = None

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

    def process_tick(self, t, bid, ask, tick_id, timestamp):
        """
        Process a single tick with realistic processing delays.
        """
        import random
        import time
        
        # Simulate realistic processing delays based on market conditions
        spread = ask - bid
        
        # Higher spread = more complex processing
        if spread > 0.0002:  # Wide spread
            base_delay = random.uniform(0.020, 0.100)  # 20-100ms
        elif spread > 0.0001:  # Normal spread
            base_delay = random.uniform(0.005, 0.030)  # 5-30ms  
        else:  # Tight spread
            base_delay = random.uniform(0.001, 0.015)  # 1-15ms
        
        # Add occasional "system hiccups"
        if random.random() < 0.02:  # 2% chance of system delay
            hiccup_delay = random.uniform(0.200, 0.50)  # 100-500ms hiccup
            base_delay += hiccup_delay
            # logging.warning(f"System hiccup: added {hiccup_delay*1000:.1f}ms delay to tick {tick_id}")
        
        """
        Process a single tick and execute the trading strategy.
        """
        start_time = time.perf_counter()
        self.tick_count += 1
        
        # Determine phase BEFORE processing
        current_size = self.data_manager.get_size() or 0
        training_window = self.config.get("training_window_size", 6000)
        is_trained = self.data_manager.get_config("is_trained")
        
        # Check if this tick should be processed (realistic filtering)
        should_process = True
        if current_size >= training_window and is_trained:
            # time.sleep(base_delay)
            # Post-training phase - apply realistic filtering
            if self.last_processed_timestamp is not None:
                current_timestamp = pd.to_datetime(timestamp)
                time_between_ticks = (current_timestamp - self.last_processed_timestamp).total_seconds()
                estimated_processing_time = self._estimate_processing_time()
                
                # Add debug logging to see what's happening
                if tick_id % 1000 == 0:  # Log every 1000th tick
                    logging.info(f"Filtering check for tick {tick_id}: "
                            f"estimated_time={estimated_processing_time*1000:.2f}ms, "
                            f"time_between_ticks={time_between_ticks*1000:.2f}ms, "
                            f"threshold={time_between_ticks*0.8*1000:.2f}ms")
                
                if estimated_processing_time > time_between_ticks * 0.3:
                    # This tick would be too slow - drop it
                    self.dropped_tick_count += 1
                    should_process = False
                    logging.warning(f"DROPPED tick {tick_id} - estimated {estimated_processing_time*1000:.1f}ms > threshold {time_between_ticks*0.8*1000:.1f}ms")

        # Update DataManager with all tick
        self.data_manager.add_tick(t, bid=bid, ask=ask, tick_id=tick_id, timestamp=timestamp)
        
        if should_process:
            # Update timestamp for next comparison
            self.last_processed_timestamp = pd.to_datetime(timestamp)
            
            # Count this tick based on current phase
            if current_size < training_window:
                self.training_tick_count += 1
                phase = "training"
            else:
                self.post_training_tick_count += 1
                phase = "post-training"
            
            # Create a tick dictionary
            tick_data = {'tick_id':tick_id, 'bid': bid,'ask': ask, 't': t, 'timestamp': timestamp}
                        
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
                
                confidence_threshold = 0.98
                trade_approved = False
                prob = 0.0
                
                if refined_state == 1:  # Bullish
                    prob = self._calculate_profit_probability(price_type='ask')
                    self.data_manager.set("prob_profit_ask", prob)
                    if prob >= confidence_threshold:
                        trade_approved = True
                elif refined_state == -1:  # Bearish
                    prob = self._calculate_profit_probability(price_type='bid')
                    self.data_manager.set("prob_profit_bid", prob)
                    if prob >= confidence_threshold:
                        trade_approved = True

                # # DEBUG: Log approval decisions
                # if refined_state != 0:  # Only log for non-neutral states
                #     logging.info(f"Tick {tick_id}: state={refined_state}, prob={prob:.6f}, "
                #                 f"threshold={confidence_threshold}, approved={trade_approved}")

                self.data_manager.set("trade_approved", 1 if trade_approved else 0)
                
                if trade_approved:
                    self.approved = self.approved + 1
                    span = 50
                    self.position_sizing_bid.process(t, refined_state)
                    self.position_sizing_ask.process(t, refined_state)
                    bid_position_sizes = np.array(self.data_manager.get("position_size_bid", span))
                    ask_position_sizes = np.array(self.data_manager.get("position_size_ask", span))
                    f = 1
                    
                    if refined_state == 1:  # Bullish state
                        bid_input = bid_position_sizes + (-ask_position_sizes)
                        bid_ewma = self.calculate_ewma(bid_input, span)
                        bid_ewma_filtered = np.maximum(bid_ewma, 0)
                        smoothed_bid_positions = bid_ewma_filtered * bid_position_sizes
                        self.data_manager.set("adjusted_position_size_bid", smoothed_bid_positions[-1])
                        if not np.isnan(smoothed_bid_positions[-1]) and smoothed_bid_positions[-1] > 0:
                            self.position_opening_handler.handle_openings(refined_state * 1, bid, ask, tick_id, f * smoothed_bid_positions[-1])
                    elif refined_state == -1:  # Bearish state
                        ask_input = ask_position_sizes + (-bid_position_sizes)
                        ask_ewma = self.calculate_ewma(ask_input, span)
                        ask_ewma_filtered = np.maximum(ask_ewma, 0)
                        smoothed_ask_positions = ask_ewma_filtered * ask_position_sizes
                        self.data_manager.set("adjusted_position_size_ask", smoothed_ask_positions[-1])
                        if not np.isnan(smoothed_ask_positions[-1]) and smoothed_ask_positions[-1] > 0:
                            self.position_opening_handler.handle_openings(refined_state * 1, bid, ask, tick_id, f * smoothed_ask_positions[-1])
                elif refined_state != 0:
                    self.not_approved = self.not_approved + 1
                
                self.last_tick_time = t
                
                # Debug logging
                if t % 1000 == 0 and t != 0:
                    logging.info(f"t={t}: phase={phase}, training_count={self.training_tick_count}, "
                                f"post_training_count={self.post_training_tick_count}, dropped={self.dropped_tick_count}")        
        else:
            # Tick was dropped - set special state
            self.data_manager.set("refined_state", 10)  # Special state for dropped ticks
            self.data_manager.set("initial_state", 10)
            
            # Set other fields to neutral/zero values for dropped ticks
            self.data_manager.set("position_size_bid", 0)
            self.data_manager.set("position_size_ask", 0)
            self.data_manager.set("adjusted_position_size_bid", 0)
            self.data_manager.set("adjusted_position_size_ask", 0)
            self.data_manager.set("pnl", 0)
            self.data_manager.set("trade_approved", 0)
                        
            logging.debug(f"Processed dropped tick {tick_id} with state 10")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.processing_times.append(elapsed_time)

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
                tick_data = self.tick_queue.get(timeout=1)
                
                # Simply call process_tick - all counting happens there
                self.process_tick(
                    tick_data['t'], tick_data['bid'], tick_data['ask'], 
                    tick_data['tick_id'], tick_data['timestamp']
                )
                
                self.tick_queue.task_done()
                        
            except queue.Empty:
                continue

    def run_strategy(self, data):
        """Run strategy - just queue all ticks, filtering happens in worker"""
        logging.info("Starting strategy execution...")
        start_time = time.time()
        
        submitted_ticks = 0
        
        # Don't create simulator here - create it in worker where processing happens
        
        for idx, row in data.iterrows():
            tick_id = row["tick_id"]
            bid = row["bid"]
            ask = row["ask"]
            tick_timestamp = pd.to_datetime(row["timestamp"])
            
            submitted_ticks += 1
            
            # Queue ALL ticks - filtering decision made in worker
            self.tick_queue.put({
                't': idx, 
                'bid': bid, 
                'ask': ask, 
                'tick_id': tick_id, 
                'timestamp': tick_timestamp
            })
                
        self.tick_queue.join()
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        final_stats = self.get_processing_time_stats()
        
        logging.info("=== Strategy Execution Results ===")
        logging.info(f"Total execution time: {total_execution_time:.2f} seconds")
        logging.info(f"Total ticks in data: {len(data)}")
        logging.info(f"Training phase ticks: {self.training_tick_count}")
        logging.info(f"Post-training processed: {self.post_training_tick_count}")
        logging.info(f"Post-training dropped: {self.dropped_tick_count}")
        
        total_post_training = self.post_training_tick_count + self.dropped_tick_count
        if total_post_training > 0:
            efficiency = self.post_training_tick_count / total_post_training * 100
            logging.info(f"Post-training efficiency: {efficiency:.1f}%")
        
        logging.info(f"Processing time stats: {final_stats}")

    def _estimate_processing_time(self):
        """Estimate processing time based on recent performance"""
        if len(self.processing_times) < 10:
            return 0.005  # 5ms default
            
        # Use recent times for estimation
        recent_times = self.processing_times[-50:]
        return min(np.percentile(recent_times, 90), 0.050)  # 75th percentile, capped at 50ms

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
    
    def _get_current_processing_stats(self):
        """Use your existing processing_stats method but with current data"""
        if len(self.processing_times) < 5:
            return {
                "average": 0.005,  # 5ms default
                "p95": 0.010,      # 10ms default
                "recent_trend": 1.0
            }
            
        recent_times = np.array(self.processing_times[-100:])  # Last 100 ticks
        older_times = np.array(self.processing_times[-200:-100]) if len(self.processing_times) >= 200 else recent_times
        
        return {
            "average": np.mean(recent_times),
            "p95": np.percentile(recent_times, 95),
            "recent_trend": np.mean(recent_times) / np.mean(older_times) if len(older_times) > 0 else 1.0
        }
    

    def _calculate_profit_probability(self, price_type):
        try:
            latest_data = self.data_manager.get_latest_data()
            initial_price = latest_data[f"ema_{price_type}"]
            spread = latest_data["spread"]
            required_profit_margin = spread * self.config.get("profitability_factor", 1.5)

            # --- NEW: Calculate a drift term ---
            # Use the arithmetic return over a larger window (e.g., 200 ticks) as an estimate of the drift
            recent_returns_buffer = self.data_manager.get_window_data(200)[f"arithmetic_return_{price_type}"]
            # Take the last N non-NaN returns for drift calculation, ensuring enough data points
            recent_returns = recent_returns_buffer[~np.isnan(recent_returns_buffer)][-self.config.get("simulation_steps", 50):] if len(recent_returns_buffer[~np.isnan(recent_returns_buffer)]) > 0 else np.array([])
            
            drift_per_tick = np.mean(recent_returns) if len(recent_returns) > 0 else 0
            if np.isnan(drift_per_tick):
                drift_per_tick = 0
            # --- END NEW ---

            model = self.market_processing_ask.model if price_type == 'ask' else self.market_processing_bid.model
            scaling_factor = self.market_processing_ask.scaling_factor if price_type == 'ask' else self.market_processing_bid.scaling_factor

            # vol_paths = model.simulate_volatility_paths(
            #     steps=self.config.get("simulation_steps", 50),
            #     num_simulations=self.config.get("num_simulations", 5000)
            # )
            vol_paths = self.market_processing_bid.simulated_vol_paths if price_type == 'bid' else self.market_processing_ask.simulated_vol_paths

            price_paths = np.zeros((self.config.get("num_simulations", 5000), self.config.get("simulation_steps", 50)))
            current_prices = np.full(self.config.get("num_simulations", 5000), initial_price)

            for i in range(self.config.get("simulation_steps", 50)):
                de_scaled_vols = vol_paths[:, i] / scaling_factor
                log_return_shocks = np.random.normal(0, 1, size=self.config.get("num_simulations", 5000))
                
                # MODIFICATION: Add drift to the log returns
                simulated_log_returns = drift_per_tick + (de_scaled_vols * log_return_shocks)

                current_prices = current_prices * np.exp(simulated_log_returns)
                price_paths[:, i] = current_prices

            final_price_distribution = price_paths[:, -1]

            successful_simulations = 0
            if price_type == 'ask':
                target_price = initial_price + required_profit_margin
                successful_simulations = np.sum(final_price_distribution > target_price)
            elif price_type == 'bid':
                target_price = initial_price - required_profit_margin
                successful_simulations = np.sum(final_price_distribution < target_price)

            probability = successful_simulations / self.config.get("num_simulations", 5000)
            return probability

        except Exception as e:
            logging.error(f"Failed during profit probability calculation for {price_type}: {e}")
            return 0.0 # Return a neutral probability on error

    def _update_confidence_threshold(self):
        """
        Adaptively adjusts the confidence threshold based on recent trading performance.
        For simplicity, we'll use a basic win rate over the last N trades.
        """
        # Retrieve recent trade outcomes from TradeManager
        all_positions = self.trade_manager.get_all_positions()
        
        # Filter for closed positions that have a PnL recorded
        closed_positions = all_positions[all_positions["close_id"] != -1]

        if len(closed_positions) < 50: # Need a minimum number of trades to assess performance
            return

        # Consider the last 50 closed trades for performance assessment
        recent_trades = closed_positions[-50:]
        winning_trades = np.sum(recent_trades["pnl"] > 0)
        losing_trades = np.sum(recent_trades["pnl"] < 0)
        total_evaluated_trades = winning_trades + losing_trades

        if total_evaluated_trades == 0:
            return

        win_rate = winning_trades / total_evaluated_trades

        # Define adjustment parameters
        target_win_rate = 0.60 # Example target win rate
        adjustment_step = 0.01 # How much to adjust the threshold by
        min_threshold = 0.70   # Minimum allowed confidence threshold
        max_threshold = 0.99   # Maximum allowed confidence threshold

        if win_rate > target_win_rate: # Performance is good, can be less strict
            self.adaptive_confidence_threshold = max(min_threshold, self.adaptive_confidence_threshold - adjustment_step)
        elif win_rate < target_win_rate: # Performance is poor, be more strict
            self.adaptive_confidence_threshold = min(max_threshold, self.adaptive_confidence_threshold + adjustment_step)
        
        logging.info(f"Adaptive Confidence Threshold updated to: {self.adaptive_confidence_threshold:.2f} (Win Rate: {win_rate:.2f})")

    def _update_profitability_factor(self):
        """
        Adaptively adjusts the profitability factor based on recent trading performance (e.g., Sharpe ratio).
        """
        all_positions = self.trade_manager.get_all_positions()
        closed_positions = all_positions[all_positions["close_id"] != -1]

        if len(closed_positions) < 50: # Need a minimum number of trades to assess performance
            return

        # Consider the last 50 closed trades for performance assessment
        recent_trades = closed_positions[-50:]
        pnl_values = recent_trades["pnl"]

        if len(pnl_values) < 2 or np.std(pnl_values) == 0: # Need at least 2 trades and some variance for Sharpe
            return

        # Calculate a simple Sharpe ratio (assuming risk-free rate is 0 for simplicity)
        # This is a simplified Sharpe ratio for PnL per trade, not time-series returns
        mean_pnl = np.mean(pnl_values)
        std_pnl = np.std(pnl_values)
        sharpe_ratio = mean_pnl / std_pnl if std_pnl != 0 else 0

        target_sharpe_ratio = 0.5 # Example target Sharpe ratio
        adjustment_step = 0.01    # How much to adjust the factor by
        min_factor = 1.000001     # Minimum allowed profitability factor (must be > 1 to cover spread)
        max_factor = 2.0          # Maximum allowed profitability factor

        if sharpe_ratio > target_sharpe_ratio: # Performance is good, can be less strict with profit target
            self.adaptive_profitability_factor = max(min_factor, self.adaptive_profitability_factor - adjustment_step)
        elif sharpe_ratio < target_sharpe_ratio: # Performance is poor, be more strict with profit target
            self.adaptive_profitability_factor = min(max_factor, self.adaptive_profitability_factor + adjustment_step)
        
        logging.info(f"Adaptive Profitability Factor updated to: {self.adaptive_profitability_factor:.6f} (Sharpe Ratio: {sharpe_ratio:.2f})")

    def get_processing_time_stats(self):
        if not self.processing_times:
            return {"count": 0, "total_time": 0, "average": 0, "min": 0, "max": 0, "std_dev": 0}

        times = np.array(self.processing_times)
        return {
            "count": len(times),
            "total_time": np.sum(times),
            "average": np.mean(times),
            "min": np.min(times),
            "max": np.max(times),
            "std_dev": np.std(times)
        }