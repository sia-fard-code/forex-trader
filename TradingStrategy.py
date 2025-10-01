# from LivePlotManager import PlotControlObserver

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
# from RealtimeTickSimulator import RealtimeTickSimulator
import logging
import time
import threading
import queue
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tick_times = []

class TradingStrategy:
    def __init__(self, config, enable_live_plot=False):
        """Initialize TradingStrategy with fail-safe queue creation"""
        
        # 🚨 CRITICAL: Initialize queues FIRST (before any exceptions)
        self.tick_queue = queue.Queue()
        self.plot_command_queue = queue.Queue()
        self.plot_response_queue = queue.Queue() 
        self.plot_data_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        # 🎯 INITIALIZE THREADING EVENTS FIRST
        self.processing_event = threading.Event()
        self.step_event = threading.Event()
        # self.processing_event.set()  # Start unpaused
        # Plot control flags
        self.plot_initialized = False
        self.plot_init_requested = False
        self.training_completed = False        
        # Initialize safe attributes
        self.config = config
        self.enable_live_plot = enable_live_plot
        self.live_plotter = None
        self.workers = []
        self.num_workers = 1
        
        # Initialize counters and metrics (safe)
        self.approved = 0
        self.not_approved = 0
        self.processing_times = []
        self.tick_count = 0
        self.training_tick_count = 0
        self.post_training_tick_count = 0
        self.dropped_tick_count = 0
        self.last_processed_timestamp = None
        self._training_speed_set = False
        self._trading_speed_set = False
        
        try:
            # Now initialize components that might fail
            self.data_manager = DataManager(config)
            self.market_processing_bid = MarketProcessing(self.data_manager, "bid")
            self.market_processing_ask = MarketProcessing(self.data_manager, "ask")
            self.state_identifier = StateIdentifier(self.data_manager)
            self.position_sizing_bid = PositionSizing(self.data_manager, "bid")
            self.position_sizing_ask = PositionSizing(self.data_manager, "ask")
            
            # Trading components
            self.trade_manager = TradeManager(config['broker_config'], max_positions=100000, data_manager=self.data_manager)
            self.position_manager = PositionManager(config['broker_config'], self.trade_manager, max_positions=100000)
            
            # Account management
            self.equity = config['broker_config']["initial_capital"]
            self.free_margin = self.equity
            self.margin_used = 0.0
            self.drawdown_threshold = config.get("drawdown_threshold", 0.10)
            self.adaptive_confidence_threshold = 0.95
            self.adaptive_profitability_factor = config.get("profitability_factor", 1.5)
            
            # Position handlers
            self.position_closure_handler = PositionClosureHandler(self.trade_manager, config.get("broker_config", {}))
            self.position_opening_handler = PositionOpeningHandler(self.trade_manager, config.get("broker_config", {}))
            
            # Live plotting (might fail - but queues already created)
            if self.enable_live_plot:
                from LivePlotManager import LivePlotManager
                self.live_plotter = LivePlotManager()
                logging.info("Live plotting enabled successfully")
                
        except Exception as e:
            logging.error(f"Component initialization failed: {e}")
            # Queues are still available, so strategy can run without plotting
            self.enable_live_plot = False
            
        # Start worker threads last
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self.worker_process, daemon=True)
            worker.start()
            self.workers.append(worker)
            
        logging.info("TradingStrategy initialized successfully")

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
        """Worker with shutdown safety for live debugging phase"""
        
        while not self.shutdown_event.is_set():
            try:
                tick_data = self.tick_queue.get(timeout=1)
                
                # 🎯 SAFETY: Check if we should shutdown
                if self.shutdown_event.is_set():
                    logging.info("🛑 Worker received shutdown signal during processing")
                    self.tick_queue.task_done()
                    break
                
                # Extract tick data
                t = tick_data['t']
                bid = tick_data['bid']
                ask = tick_data['ask']
                tick_id = tick_data['tick_id']
                timestamp = tick_data['timestamp']
                
                # Process the tick (training phase only)
                logging.debug(f"🏃 Worker processing training tick {tick_id}")
                self.process_tick(t, bid, ask, tick_id, timestamp)
                
                self.tick_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker error: {e}")
                if not self.tick_queue.empty():
                    self.tick_queue.task_done()
        
        logging.info("🛑 Worker thread shutdown complete")

    # 🎯 IMPLEMENT OBSERVER INTERFACE METHODS
    def on_plot_pause(self):
        """Called by LivePlotManager when pause button clicked"""
        self.processing_event.clear()  # Block processing
        logging.info("🛑 STRATEGY PAUSED by plot control")
    
    def on_plot_resume(self):
        """Called by LivePlotManager when resume button clicked"""
        self.processing_event.set()  # Unblock processing
        logging.info("▶️ STRATEGY RESUMED by plot control")

    def on_plot_step(self):
        """Simple step fix - unblock processing temporarily"""
        self.step_event.set()          # Mark as step mode
        self.processing_event.set()    # Allow processing to continue
        logging.info("⏭️ Step requested - processing unblocked for one tick")
                
    def run_strategy(self, data, live_plot_speed=1.0, live_plot_delay=0.01):
        """Strategy with proper matplotlib threading"""
        
        logging.info("🎬 Starting strategy with proper threading...")
        
        # Store settings
        self.strategy_data = data
        self.live_plot_speed = live_plot_speed
        self.live_plot_delay = live_plot_delay
        self.training_window = self.config.get("training_window_size", 6000)
        
        # 🎯 INITIALIZE PLOT IN MAIN THREAD
        if self.enable_live_plot:
            success = self._initialize_plot_properly()
            if not success:
                logging.error("❌ Plot failed - running headless")
                self._run_headless_mode(data)
                return
        
        # 🎯 START PROCESSING IN SEPARATE THREAD
        import threading
        
        def processing_thread():
            """Run processing in separate thread"""
            try:
                self._run_controlled_processing(data)
            except Exception as e:
                logging.error(f"Processing thread error: {e}")
        
        # Start processing thread
        process_thread = threading.Thread(target=processing_thread, daemon=True)
        process_thread.start()
        
        # 🎯 KEEP MAIN THREAD FOR MATPLOTLIB
        if self.enable_live_plot:
            print("\n" + "="*60)
            print("🎬 FOREX TRADING STRATEGY INITIALIZED")
            print("⏸️ Strategy is PAUSED and ready")
            print("🎨 Live plot window is open")
            print("👆 Click the ▶️ RESUME button to start processing")
            print("="*60)
            
            # Keep main thread alive for matplotlib
            try:
                import matplotlib.pyplot as plt
                plt.show(block=True)  # This keeps main thread for matplotlib
            except KeyboardInterrupt:
                logging.info("🛑 User interrupted with Ctrl+C")
            except Exception as e:
                logging.error(f"Matplotlib error: {e}")
        
        # Wait for processing to complete
        process_thread.join()
        logging.info("🎉 Strategy completed")

    def _initialize_plot_properly(self):
        """Proper plot initialization for main thread"""
        try:
            logging.info("🎨 Initializing plot in main thread...")
            
            if not hasattr(self, 'live_plotter') or self.live_plotter is None:
                from LivePlotManager import LivePlotManager
                self.live_plotter = LivePlotManager(max_points=1000)
            
            # Setup observer relationship
            self.live_plotter.add_control_observer(self)
            
            # The plot will be shown when plt.show(block=True) is called
            self.plot_initialized = True
            logging.info("✅ Plot ready for display")
            return True
            
        except Exception as e:
            logging.error(f"❌ Plot setup failed: {e}")
            return False

    def _run_controlled_processing(self, data):
        """Process all data with pause/resume/step control from start"""
        
        logging.info("🎯 Starting controlled processing (starts paused)...")
        
        for idx, row in data.iterrows():
            # 🎯 BLOCK HERE IF PAUSED (including at the very start)
            self.processing_event.wait()  # This blocks until Resume is clicked
            
            # Extract and process tick
            tick_id = row["tick_id"]
            bid = row["bid"] 
            ask = row["ask"]
            tick_timestamp = pd.to_datetime(row["timestamp"])
            
            # Determine phase
            phase = "training" if idx < self.training_window else "live"
            
            # Process tick
            self.process_tick(idx, bid, ask, tick_id, tick_timestamp)
            
            # Update plot
            if self.plot_initialized:
                self._update_plot_direct(bid, ask, tick_timestamp, tick_id)
                self._update_plot_stats(phase)
            
            # Handle step mode (pause after processing one tick)
            if self.step_event.is_set():
                self.step_event.clear()
                self.processing_event.clear()  # Pause after this tick
                
                if self.plot_initialized:
                    print(f"📍 STEP COMPLETED: Tick {tick_id} ({phase})")
                    print("   👆 Click Step again or Resume to continue")
                
                logging.info(f"⏸️ Auto-paused after step - tick {tick_id}")
            
            # Progress logging
            if idx % 1000 == 0:
                progress = idx / len(data) * 100
                logging.info(f"🎯 Progress: {progress:.1f}% - Tick {tick_id} ({phase})")
            
            # Small delay for responsiveness
            time.sleep(self.live_plot_delay)
        
        logging.info("✅ Controlled processing completed")

    def _update_plot_stats(self, phase):
        """Update plot statistics display"""
        try:
            training_count = self.training_tick_count
            live_count = self.post_training_tick_count
            total_count = training_count + live_count
            
            self.live_plotter.stats_text.set_text(
                f'Training: {training_count} | Live: {live_count} | Total: {total_count} | Phase: {phase.upper()}'
            )
        except Exception as e:
            logging.debug(f"Error updating plot stats: {e}")
    
    # 🎯 ADD PLOT UPDATE METHOD
    def _update_plot_direct(self, bid, ask, timestamp, tick_id):
        """Update plot with current tick data"""
        if not self.plot_initialized or not self.live_plotter:
            return
            
        try:
            # Get current strategy state
            plot_data = {
                'timestamps': timestamp,
                'bid_prices': bid,
                'ask_prices': ask,
                'equity': self.config['broker_config']["initial_capital"],
                'balance': self.config['broker_config']["initial_capital"],
                'refined_states': 0,
                'position_sizes_bid': 0,
                'position_sizes_ask': 0,
                'trades': None,
                'pnl': 0
            }
            
            # Get processed data if available
            if self.data_manager.get_size() > 0:
                try:
                    plot_data.update({
                        'equity': self.data_manager.get_latest_value("equity"),
                        'balance': self.data_manager.get_latest_value("balance"),
                        'refined_states': self.data_manager.get_latest_value("refined_state"),
                        'position_sizes_bid': self.data_manager.get_latest_value("position_size_bid"),
                        'position_sizes_ask': self.data_manager.get_latest_value("position_size_ask"),
                        'pnl': self.data_manager.get_latest_value("pnl")
                    })
                    
                    # Trade signals
                    trade_approved = self.data_manager.get_latest_value("trade_approved")
                    if trade_approved > 0:
                        state = plot_data['refined_states']
                        plot_data['trades'] = 'buy' if state == 1 else 'sell' if state == -1 else None
                        
                except Exception as e:
                    logging.debug(f"Error getting latest data: {e}")
            
            # Update plot
            self.live_plotter.add_data_point(**plot_data)
            
            # Check for trade arrows
            self._check_for_trade_arrows(timestamp, tick_id)
            
        except Exception as e:
            logging.debug(f"Plot update failed: {e}")
    
    def _check_for_trade_arrows(self, timestamp, tick_id):
        """Check for newly closed positions and add arrows"""
        try:
            current_positions = self.trade_manager.get_all_positions()
            if len(current_positions) == 0:
                return
                
            newly_closed = current_positions[current_positions['close_id'] == tick_id]
            
            for _, closed_pos in newly_closed.iterrows():
                open_timestamp = self._estimate_timestamp_for_tick(closed_pos['open_id'], timestamp)
                
                self.live_plotter.add_trade_arrow(
                    open_time=open_timestamp,
                    close_time=timestamp,
                    open_price=closed_pos['open_price'],
                    close_price=closed_pos['close_price'],
                    direction=closed_pos['direction'],
                    trade_id=closed_pos.name,
                    pnl=closed_pos['pnl']
                )
                
                logging.info(f"🏹 Trade arrow added: {closed_pos['direction']} position closed with PnL: {closed_pos['pnl']:.5f}")
                
        except Exception as e:
            logging.debug(f"Trade arrow check failed: {e}")

    # def _update_plot_direct(self, bid, ask, timestamp, tick_id):
    #     """Enhanced plot update with better error handling"""
    #     if not self.plot_initialized or not self.live_plotter:
    #         return
            
    #     try:
    #         # Get latest processed data
    #         plot_data = self._get_current_plot_data(bid, ask, timestamp)
            
    #         # Update plot
    #         self.live_plotter.add_data_point(**plot_data)
            
    #         # Check for trade arrows
    #         self._check_for_trade_arrows(timestamp, tick_id)
            
    #         # Debug output every 100 ticks
    #         if tick_id % 100 == 0:
    #             logging.debug(f"📊 Plot updated: tick {tick_id}, state={plot_data.get('refined_states', 0)}")
            
    #     except Exception as e:
    #         logging.debug(f"Plot update failed for tick {tick_id}: {e}")

    # def _get_current_plot_data(self, bid, ask, timestamp):
    #     """Get current plot data from DataManager"""
    #     plot_data = {
    #         'timestamps': timestamp,
    #         'bid_prices': bid,
    #         'ask_prices': ask,
    #         'equity': self.config['broker_config']["initial_capital"],
    #         'balance': self.config['broker_config']["initial_capital"],
    #         'refined_states': 0,
    #         'position_sizes_bid': 0,
    #         'position_sizes_ask': 0,
    #         'trades': None,
    #         'pnl': 0
    #     }
        
    #     # Get processed data if available
    #     if self.data_manager.get_size() > 0:
    #         try:
    #             plot_data.update({
    #                 'equity': self.data_manager.get_latest_value("equity"),
    #                 'balance': self.data_manager.get_latest_value("balance"), 
    #                 'refined_states': self.data_manager.get_latest_value("refined_state"),
    #                 'position_sizes_bid': self.data_manager.get_latest_value("position_size_bid"),
    #                 'position_sizes_ask': self.data_manager.get_latest_value("position_size_ask"),
    #                 'pnl': self.data_manager.get_latest_value("pnl")
    #             })
                
    #             # Trade signals
    #             trade_approved = self.data_manager.get_latest_value("trade_approved")
    #             if trade_approved > 0:
    #                 state = plot_data['refined_states']
    #                 plot_data['trades'] = 'buy' if state == 1 else 'sell' if state == -1 else None
                    
    #         except Exception as e:
    #             logging.debug(f"Error getting latest data: {e}")
        
    #     return plot_data

    def process_tick(self, t, bid, ask, tick_id, timestamp):
        """Enhanced process_tick with improved live plotting"""
        
        # Add debug counter
        if not hasattr(self, 'debug_tick_count'):
            self.debug_tick_count = 0
        self.debug_tick_count += 1
        
        import random
        import time
        
        # Your existing realistic processing delay simulation
        spread = ask - bid
        
        if spread > 0.0002:
            base_delay = random.uniform(0.020, 0.100)
        elif spread > 0.0001:
            base_delay = random.uniform(0.005, 0.030)
        else:
            base_delay = random.uniform(0.001, 0.015)
        
        if random.random() < 0.02:
            hiccup_delay = random.uniform(0.200, 0.50)
            base_delay += hiccup_delay

        start_time = time.perf_counter()
        self.tick_count += 1
        
        # Your existing phase determination
        current_size = self.data_manager.get_size() or 0
        training_window = self.config.get("training_window_size", 6000)
        is_trained = self.data_manager.get_config("is_trained")
        
        # Enhanced speed control with one-time logging
        if is_trained and not self._trading_speed_set:
            self._trading_speed_set = True
            self._training_speed_set = False  # Reset for next run
            logging.info(f"🎓 Training completed at tick {self.debug_tick_count}. Switching to trading mode.")
        elif not is_trained and not self._training_speed_set:
            self._training_speed_set = True
            logging.info(f"🏃 Training mode active. Processing at high speed.")
        
        # Your existing filtering logic
        should_process = True
        if current_size >= training_window and is_trained:
            if self.last_processed_timestamp is not None:
                current_timestamp = pd.to_datetime(timestamp)
                time_between_ticks = (current_timestamp - self.last_processed_timestamp).total_seconds()
                estimated_processing_time = self._estimate_processing_time()
                
                if estimated_processing_time > time_between_ticks * 0.3:
                    self.dropped_tick_count += 1
                    should_process = False

        # Always update DataManager
        self.data_manager.add_tick(t, bid=bid, ask=ask, tick_id=tick_id, timestamp=timestamp)
        
        # Continue with your existing processing logic only if should_process
        if should_process:
            self.last_processed_timestamp = pd.to_datetime(timestamp)
            
            if current_size < training_window:
                self.training_tick_count += 1
                phase = "training"
            else:
                self.post_training_tick_count += 1
                phase = "post-training"
            
            tick_data = {'tick_id': tick_id, 'bid': bid, 'ask': ask, 't': t, 'timestamp': timestamp}
            self.position_manager.process_tick(tick_data)
            
            self.market_processing_bid.process_tick(t)
            self.market_processing_ask.process_tick(t)
            
            # Your existing trained logic
            if self.data_manager.get_config("is_trained"):
                self.state_identifier.process(t, bid, ask)
                
                refined_state_series = self.data_manager.get("refined_state")
                refined_state = refined_state_series[-1]
                
                confidence_threshold = 0.98
                trade_approved = False
                prob = 0.0
                
                if refined_state == 1:
                    prob = self._calculate_profit_probability(price_type='ask')
                    self.data_manager.set("prob_profit_ask", prob)
                    if prob >= confidence_threshold:
                        trade_approved = True
                elif refined_state == -1:
                    prob = self._calculate_profit_probability(price_type='bid')
                    self.data_manager.set("prob_profit_bid", prob)
                    if prob >= confidence_threshold:
                        trade_approved = True

                self.data_manager.set("trade_approved", 1 if trade_approved else 0)
                
                if trade_approved:
                    self.approved = self.approved + 1
                    span = 50
                    self.position_sizing_bid.process(t, refined_state)
                    self.position_sizing_ask.process(t, refined_state)
                    
                    # Your existing position sizing logic...
                    bid_position_sizes = np.array(self.data_manager.get("position_size_bid", span))
                    ask_position_sizes = np.array(self.data_manager.get("position_size_ask", span))
                    f = 1
                    
                    if refined_state == 1:
                        bid_input = bid_position_sizes + (-ask_position_sizes)
                        bid_ewma = self.calculate_ewma(bid_input, span)
                        bid_ewma_filtered = np.maximum(bid_ewma, 0)
                        smoothed_bid_positions = bid_ewma_filtered * bid_position_sizes
                        self.data_manager.set("adjusted_position_size_bid", smoothed_bid_positions[-1])
                        if not np.isnan(smoothed_bid_positions[-1]) and smoothed_bid_positions[-1] > 0:
                            self.position_opening_handler.handle_openings(refined_state * 1, bid, ask, tick_id, f * smoothed_bid_positions[-1])
                    elif refined_state == -1:
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
                
                if t % 1000 == 0 and t != 0:
                    logging.info(f"t={t}: phase={phase}, training_count={self.training_tick_count}, "
                                f"post_training_count={self.post_training_tick_count}, dropped={self.dropped_tick_count}")        
        else:
            # Tick was dropped - your existing logic
            self.data_manager.set("refined_state", 10)
            self.data_manager.set("initial_state", 10)
            self.data_manager.set("position_size_bid", 0)
            self.data_manager.set("position_size_ask", 0)
            self.data_manager.set("adjusted_position_size_bid", 0)
            self.data_manager.set("adjusted_position_size_ask", 0)
            self.data_manager.set("pnl", 0)
            self.data_manager.set("trade_approved", 0)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.processing_times.append(elapsed_time)

    def _estimate_timestamp_for_tick(self, target_tick_id, current_timestamp):
        """Estimate timestamp for a given tick_id"""
        try:
            # Simple estimation: assume 1 second per tick difference
            tick_diff = target_tick_id - self.debug_tick_count
            estimated_timestamp = current_timestamp + pd.Timedelta(seconds=tick_diff)
            return estimated_timestamp
        except:
            return current_timestamp - pd.Timedelta(seconds=30)  # Default fallback

    def set_live_plot_speed(self, speed):
        """Enhanced speed control with verification"""
        if self.enable_live_plot and self.live_plotter:
            try:
                old_speed = getattr(self.live_plotter, 'speed_multiplier', 1.0)
                self.live_plotter.set_speed(speed)
                new_speed = getattr(self.live_plotter, 'speed_multiplier', speed)
                
                if abs(new_speed - speed) > 0.1:
                    logging.warning(f"Speed change verification failed: requested {speed}x, got {new_speed}x")
                else:
                    logging.debug(f"📈 Live plot speed: {old_speed:.1f}x → {new_speed:.1f}x")
                    
            except Exception as e:
                logging.error(f"Failed to set live plot speed: {e}")

    def close_live_plot(self):
        """Close live plot"""
        if self.enable_live_plot and self.live_plotter:
            self.live_plotter.close()

    def _estimate_processing_time(self):
        """Estimate processing time based on recent performance"""
        if len(self.processing_times) < 10:
            return 0.005  # 5ms default
            
        # Use recent times for estimation
        recent_times = self.processing_times[-50:]
        return min(np.percentile(recent_times, 90), 0.050)  # 90th percentile, capped at 50ms

    def shutdown(self):
        """
        Shutdown the worker threads and perform any necessary cleanup.
        """
        logging.info("Shutting down TradingStrategy.")
        
        # Close live plot first
        if self.enable_live_plot:
            self.close_live_plot()
        
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
    
    def _calculate_profit_probability(self, price_type):
        try:
            latest_data = self.data_manager.get_latest_data()
            initial_price = latest_data[f"ema_{price_type}"]
            spread = latest_data["spread"]
            required_profit_margin = spread * self.config.get("profitability_factor", 1.5)
            
            # Calculate drift term
            recent_returns_buffer = self.data_manager.get_window_data(200)[f"arithmetic_return_{price_type}"]
            recent_returns = recent_returns_buffer[~np.isnan(recent_returns_buffer)][-self.config.get("simulation_steps", 50):] if len(recent_returns_buffer[~np.isnan(recent_returns_buffer)]) > 0 else np.array([])
            
            drift_per_tick = np.mean(recent_returns) if len(recent_returns) > 0 else 0
            if np.isnan(drift_per_tick):
                drift_per_tick = 0

            model = self.market_processing_ask.model if price_type == 'ask' else self.market_processing_bid.model
            scaling_factor = self.market_processing_ask.scaling_factor if price_type == 'ask' else self.market_processing_bid.scaling_factor
            
            vol_paths = self.market_processing_bid.simulated_vol_paths if price_type == 'bid' else self.market_processing_ask.simulated_vol_paths
            price_paths = np.zeros((self.config.get("num_simulations", 5000), self.config.get("simulation_steps", 50)))
            current_prices = np.full(self.config.get("num_simulations", 5000), initial_price)
            
            for i in range(self.config.get("simulation_steps", 50)):
                de_scaled_vols = vol_paths[:, i] / scaling_factor
                log_return_shocks = np.random.normal(0, 1, size=self.config.get("num_simulations", 5000))
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
            return 0.0

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