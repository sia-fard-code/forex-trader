from abc import ABC, abstractmethod

class PlotControlObserver(ABC):
    """Interface for objects that want to observe plot control events"""
    @abstractmethod
    def on_plot_pause(self):
        """Called when plot is paused"""
        pass
    
    @abstractmethod  
    def on_plot_resume(self):
        """Called when plot is resumed"""
        pass
    
    @abstractmethod
    def on_plot_step(self):
        """Called when step button is clicked"""
        pass
    @abstractmethod
    def on_plot_close(self):  # 🎯 NEW: Handle plot close
        """Called when plot window is closed"""
        pass

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import FancyArrowPatch
import numpy as np
from collections import deque
import logging
from queue import Queue
import time
import pandas as pd

class LivePlotManager:
    def __init__(self, max_points=500, update_interval=100):
        """
        Enhanced Live plot manager with trade arrows and proper data alignment
        
        Args:
            max_points: Maximum data points to display
            update_interval: Update interval in milliseconds
        """
        self.max_points = max_points
        self.update_interval = update_interval
        self.base_update_interval = update_interval
        self.current_interval = update_interval
        self.speed_multiplier = 1.0
        self.control_observers = []

        # Control states
        self.is_paused = True
        self.step_mode = False
        self.step_requested = False
        self.last_speed_change = time.time()
        
        # Data storage with fixed-size deques
        self.plot_data = {
            'timestamps': deque(maxlen=max_points),
            'bid_prices': deque(maxlen=max_points),
            'ask_prices': deque(maxlen=max_points),
            'equity': deque(maxlen=max_points),
            'balance': deque(maxlen=max_points),
            'refined_states': deque(maxlen=max_points),
            'position_sizes_bid': deque(maxlen=max_points),
            'position_sizes_ask': deque(maxlen=max_points),
            'trades': deque(maxlen=max_points),
            'pnl': deque(maxlen=max_points)
        }
        
        # Trade arrows storage
        self.trade_arrows = []  # Store FancyArrowPatch objects
        self.trade_arrows_data = deque(maxlen=max_points)  # Store arrow data
        
        # Thread-safe queues
        self.data_queue = Queue()
        self.arrows_queue = Queue()
        # 🎯 ADD SHUTDOWN HANDLING
        self.is_closing = False
        self.close_callbacks = []        
        # Plot elements
        self.fig = None
        self.axes = None
        self.animation = None
        
        # Control widgets
        self.play_pause_button = None
        self.step_button = None
        self.reset_button = None
        self.speed_slider = None
        self.points_slider = None
        
        # Status text elements
        self.status_text = None
        self.stats_text = None
        self.speed_text = None
        
        # Debug counter
        self.update_count = 0
        
        self._setup_plot()
    
    def _setup_plot(self):
        """Setup the matplotlib figure with controls and arrow support"""
        try:
            # Create figure with extra space for controls
            self.fig = plt.figure(figsize=(17, 13))
            self.fig.patch.set_facecolor('white')
            
            # Create main subplot area (leave room for controls at bottom)
            self.axes = []
            gs = self.fig.add_gridspec(4, 1, height_ratios=[2.5, 1, 1, 1], 
                                      top=0.92, bottom=0.25, left=0.08, right=0.95,
                                      hspace=0.3)
            
            for i in range(4):
                self.axes.append(self.fig.add_subplot(gs[i, 0]))
            
            # Price Chart (Bid/Ask + Trades + Arrows)
            self.bid_line, = self.axes[0].plot([], [], 'b-', linewidth=1.5, label='Bid', alpha=0.9)
            self.ask_line, = self.axes[0].plot([], [], 'r-', linewidth=1.5, label='Ask', alpha=0.9)
            self.buy_markers = self.axes[0].scatter([], [], c='green', marker='^', s=80, 
                                                   label='Buy Signal', zorder=5, alpha=0.8)
            self.sell_markers = self.axes[0].scatter([], [], c='red', marker='v', s=80, 
                                                    label='Sell Signal', zorder=5, alpha=0.8)
            
            # Equity & Balance Chart
            self.equity_line, = self.axes[1].plot([], [], 'g-', linewidth=2.5, label='Equity')
            self.balance_line, = self.axes[1].plot([], [], 'orange', linewidth=2, label='Balance')
            
            # Market State Chart
            self.state_line, = self.axes[2].plot([], [], 'purple', linewidth=2.5, label='Market State')
            self.axes[2].axhline(y=1, color='g', linestyle='--', alpha=0.6, label='Bullish')
            self.axes[2].axhline(y=-1, color='r', linestyle='--', alpha=0.6, label='Bearish')
            self.axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.4, label='Neutral')
            
            # Position Sizes Chart
            self.pos_bid_line, = self.axes[3].plot([], [], 'blue', linewidth=2, 
                                                   label='Position Size Bid', alpha=0.8)
            self.pos_ask_line, = self.axes[3].plot([], [], 'red', linewidth=2, 
                                                   label='Position Size Ask', alpha=0.8)
            
            # Styling
            titles = ['🎯 Price & Trades with Arrows', '💰 Account Performance', 
                     '📊 Market State', '⚖️ Position Sizes']
            ylabels = ['Price', 'Value ($)', 'State', 'Size']
            
            for i, (title, ylabel) in enumerate(zip(titles, ylabels)):
                self.axes[i].set_title(title, fontsize=13, fontweight='bold', pad=10)
                self.axes[i].set_ylabel(ylabel, fontsize=11)
                self.axes[i].legend(loc='upper left', fontsize=10, framealpha=0.9)
                self.axes[i].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                
                # Enhanced styling
                self.axes[i].spines['top'].set_visible(False)
                self.axes[i].spines['right'].set_visible(False)
                self.axes[i].tick_params(labelsize=10)
            
            self.axes[3].set_xlabel('Time', fontsize=11)
            
            # Add overall title
            self.fig.suptitle('🚀 Live Forex Trading Dashboard', 
                             fontsize=18, fontweight='bold', y=0.98)
            
            # Setup control panel
            self._setup_controls()
            
            # Initialize animation
            try:
                self.animation = animation.FuncAnimation(
                    self.fig, self._update_plot, interval=self.update_interval, 
                    blit=False, cache_frame_data=False, repeat=True
                )
                logging.info("✅ Animation initialized successfully")
            except Exception as e:
                logging.warning(f"⚠️ Animation initialization failed: {e}")
                self.animation = None  # Set to None if failed
            
            # Event connections
            self.fig.canvas.mpl_connect('resize_event', self._on_resize)
            self.fig.canvas.mpl_connect('close_event', self._on_close)
            
            logging.info("Enhanced LivePlotManager with trade arrows initialized successfully")
            
        except Exception as e:
            logging.error(f"Error setting up enhanced live plot: {e}")
            raise
    
    def _setup_controls(self):
        """Setup interactive control widgets"""
        try:
            # Control panel background
            control_bg = plt.axes([0.08, 0.02, 0.87, 0.21])
            control_bg.set_facecolor('#f8f9fa')
            control_bg.set_title('🎛️ Interactive Control Panel', fontweight='bold', 
                                fontsize=14, pad=15, color='#2c3e50')
            control_bg.set_xticks([])
            control_bg.set_yticks([])
            for spine in control_bg.spines.values():
                spine.set_edgecolor('#dee2e6')
                spine.set_linewidth(2)
            
            # Play/Pause Button
            play_pause_ax = plt.axes([0.12, 0.16, 0.09, 0.05])
            self.play_pause_button = Button(play_pause_ax, '▶️ Resume',  # Start as Resume
                                          color='#45b7d1', hovercolor='#039be5')  # Blue color
            self.play_pause_button.label.set_fontweight('bold')
            self.play_pause_button.on_clicked(self._toggle_pause)
            
            # Step Button
            step_ax = plt.axes([0.23, 0.16, 0.09, 0.05])
            self.step_button = Button(step_ax, '⏭️ Step', 
                                    color='#4ecdc4', hovercolor='#26a69a')
            self.step_button.label.set_fontweight('bold')
            self.step_button.on_clicked(self._step_forward)
            
            # Reset/Clear Button
            reset_ax = plt.axes([0.34, 0.16, 0.09, 0.05])
            self.reset_button = Button(reset_ax, '🔄 Reset', 
                                     color='#45b7d1', hovercolor='#039be5')
            self.reset_button.label.set_fontweight('bold')
            self.reset_button.on_clicked(self._reset_data)
            
            # Auto-scale state
            self.auto_scale_enabled = True


            # Speed Control Slider
            speed_ax = plt.axes([0.47, 0.17, 0.25, 0.03])
            self.speed_slider = Slider(speed_ax, 'Speed', 0.1, 5.0, 
                                     valinit=1.0, valfmt='%.1fx', 
                                     facecolor='#96ceb4', alpha=0.8)
            self.speed_slider.on_changed(self._update_speed)
            
            # Max Points Slider
            points_ax = plt.axes([0.47, 0.12, 0.25, 0.03])
            self.points_slider = Slider(points_ax, 'Buffer', 100, 2000, 
                                      valinit=self.max_points, valfmt='%d pts',
                                      facecolor='#f7dc6f', alpha=0.8)
            self.points_slider.on_changed(self._update_max_points)
            
            # Status Display
            self.status_text = self.fig.text(0.76, 0.17, '⏸️ Paused - Click Resume to start', 
                                           fontsize=12, fontweight='bold', color='#e67e22')
            self.stats_text = self.fig.text(0.76, 0.14, 'Data: 0 | Arrows: 0', 
                                          fontsize=11, color='#2c3e50')
            self.speed_text = self.fig.text(0.76, 0.11, 'Speed: 1.0x', 
                                          fontsize=11, color='#2c3e50')
            
            # Instructions
            instructions = [
                "🎛️ Controls Guide:",
                "• ⏸️▶️ Pause/Resume: Toggle real-time updates", 
                "• ⏭️ Step: Process one data point when paused",
                "• 🔄 Reset: Clear all data and restart",
                "• 🎚️ Speed: 0.1x (slow) to 5.0x (fast)",
                "• 📊 Buffer: Data points to keep in memory"
            ]
            
            for i, instruction in enumerate(instructions):
                color = '#2c3e50' if i == 0 else '#6c757d'
                weight = 'bold' if i == 0 else 'normal'
                size = 11 if i == 0 else 10
                self.fig.text(0.12, 0.105 - i*0.012, instruction, 
                            fontsize=size, alpha=0.9, color=color, fontweight=weight)
            
        except Exception as e:
            logging.error(f"Error setting up controls: {e}")
    
    def _toggle_pause(self, event):
        """Toggle pause/resume (starts paused by default)"""
        try:
            self.is_paused = not self.is_paused
            
            if self.is_paused:
                # Switching TO paused
                self._notify_pause()
                self.play_pause_button.label.set_text('▶️ Resume')
                self.play_pause_button.color = '#45b7d1'  # Blue
                self.status_text.set_text('Status: ⏸️ Paused')
                self.status_text.set_color('#e67e22')
                if self.animation:
                    self.animation.pause()
                logging.info("📊 Paused")
            else:
                # Switching TO resumed
                self._notify_resume()
                self.play_pause_button.label.set_text('⏸️ Pause')
                self.play_pause_button.color = '#ff6b6b'  # Red
                self.status_text.set_text('Status: ▶️ Running')
                self.status_text.set_color('#27ae60')
                if self.animation:
                    self.animation.resume()
                logging.info("📊 Resumed")
            
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            logging.error(f"Error toggling pause: {e}")
    
    def _step_forward(self, event):
        """Enhanced step forward with detailed debugging"""
        try:
            print("🔍 DEBUG: Step button clicked!")
            logging.info("🔍 Step button clicked - starting debug")
            
            # Auto-pause if not already paused
            if not self.is_paused:
                print("🔍 DEBUG: Not paused, auto-pausing first...")
                self._toggle_pause(event)
            
            self.step_requested = True
            self.step_mode = True
            self.status_text.set_text('Status: ⏭️ Step Mode')
            self.status_text.set_color('#8e44ad')
            
            print(f"🔍 DEBUG: Step flags set - step_requested: {self.step_requested}")
            
            # 🎯 Notify observers of step request
            print("🔍 DEBUG: Notifying observers...")
            self._notify_step()
            
            # Process one update for the plot itself
            self._update_plot(None)
            self.fig.canvas.draw_idle()
            
            print("🔍 DEBUG: Step forward completed")
            logging.info("📊 Step forward - observers notified")
            
        except Exception as e:
            logging.error(f"Error stepping forward: {e}")
            print(f"🔍 DEBUG: Step error: {e}")

    def _notify_step(self):
        """Enhanced step notification with debug"""
        print(f"🔍 DEBUG: Notifying {len(self.control_observers)} observers of step")
        for i, observer in enumerate(self.control_observers):
            try:
                print(f"🔍 DEBUG: Calling observer {i+1}: {type(observer).__name__}")
                observer.on_plot_step()
                print(f"🔍 DEBUG: Observer {i+1} step notification successful")
            except Exception as e:
                logging.error(f"Observer step notification failed: {e}")
                print(f"🔍 DEBUG: Observer {i+1} failed: {e}")
    
    def _reset_data(self, event):
        """Clear all plot data and arrows"""
        try:
            # Clear all data deques
            for key in self.plot_data:
                self.plot_data[key].clear()
            
            # Clear queues
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except:
                    break
            
            while not self.arrows_queue.empty():
                try:
                    self.arrows_queue.get_nowait()
                except:
                    break
            
            # Clear arrow patches from plot
            for arrow in self.trade_arrows:
                try:
                    arrow.remove()
                except:
                    pass
            
            self.trade_arrows.clear()
            self.trade_arrows_data.clear()
            
            # Clear all plot lines
            self.bid_line.set_data([], [])
            self.ask_line.set_data([], [])
            self.equity_line.set_data([], [])
            self.balance_line.set_data([], [])
            self.state_line.set_data([], [])
            self.pos_bid_line.set_data([], [])
            self.pos_ask_line.set_data([], [])
            self.buy_markers.set_offsets(np.empty((0, 2)))
            self.sell_markers.set_offsets(np.empty((0, 2)))
            
            # Reset axes
            for ax in self.axes:
                ax.relim()
                ax.autoscale()
            
            # Update status
            self.stats_text.set_text('Data: 0 | Arrows: 0')
            self.update_count = 0
            
            self.fig.canvas.draw_idle()
            
            logging.info("🔄 Plot data and arrows reset")
            
        except Exception as e:
            logging.error(f"Error resetting data: {e}")
    
    def _update_speed(self, val):
        """Update animation speed with robust method"""
        try:
            self.speed_multiplier = val
            self.current_interval = max(10, min(2000, int(self.base_update_interval / val)))
            
            # Stop current animation
            if hasattr(self, 'animation') and self.animation:
                self.animation.event_source.stop()
            
            # Create new animation with updated interval
            self.animation = animation.FuncAnimation(
                self.fig, self._update_plot, interval=self.current_interval, 
                blit=False, cache_frame_data=False, repeat=True
            )
            
            # Restore pause state
            if self.is_paused:
                self.animation.pause()
            
            # Update display
            self.speed_text.set_text(f'Speed: {val:.1f}x')
            self.last_speed_change = time.time()
            
            if self.fig and self.fig.canvas:
                self.fig.canvas.draw_idle()
            
            logging.info(f"📈 Speed updated to {val:.1f}x (interval: {self.current_interval}ms)")
            
        except Exception as e:
            logging.error(f"Error updating speed: {e}")
    
    def _update_max_points(self, val):
        """Update maximum data points"""
        try:
            new_max = int(val)
            if new_max != self.max_points:
                self.max_points = new_max
                
                # Create new deques with new max length
                for key in self.plot_data:
                    old_data = list(self.plot_data[key])
                    # Keep the most recent data
                    self.plot_data[key] = deque(old_data[-new_max:], maxlen=new_max)
                
                # Update arrows deque
                old_arrows = list(self.trade_arrows_data)
                self.trade_arrows_data = deque(old_arrows[-new_max:], maxlen=new_max)
                
                logging.info(f"📊 Buffer size updated to {new_max}")
                
        except Exception as e:
            logging.error(f"Error updating max points: {e}")
    
    def _on_resize(self, event):
        """Handle window resize"""
        try:
            self.fig.tight_layout()
        except:
            pass
    
    def _cleanup_animation(self):
        """Safely cleanup animation resources"""
        try:
            if hasattr(self, 'animation'):
                if self.animation is not None:
                    if hasattr(self.animation, 'event_source'):
                        if self.animation.event_source is not None:
                            self.animation.event_source.stop()
                            logging.info("✅ Animation event source stopped")
                        else:
                            logging.info("ℹ️ Animation event source was None")
                    else:
                        logging.info("ℹ️ Animation has no event_source")
                    self.animation = None
                else:
                    logging.info("ℹ️ Animation was already None")
            else:
                logging.info("ℹ️ No animation attribute found")
        except Exception as e:
            logging.warning(f"⚠️ Error during animation cleanup: {e}")

    def _on_close(self, event):
        """Handle plot window close event with robust cleanup"""
        try:
            logging.info("🚪 Plot window closing - starting cleanup")
            self.is_closing = True
            
            # Notify observers
            for observer in self.control_observers:
                try:
                    if hasattr(observer, 'on_plot_close'):
                        observer.on_plot_close()
                except Exception as e:
                    logging.error(f"Observer close notification failed: {e}")
            
            # 🎯 SAFE ANIMATION CLEANUP
            self._cleanup_animation()
            
            logging.info("✅ Plot cleanup completed successfully")
            
        except Exception as e:
            logging.error(f"Error during plot close: {e}")

    def add_trade_arrow(self, open_time, close_time, open_price, close_price, 
                       direction, trade_id=None, pnl=None):
        """
        Add a trade arrow connecting open and close positions
        
        Args:
            open_time: Timestamp when trade was opened
            close_time: Timestamp when trade was closed
            open_price: Price at which trade was opened
            close_price: Price at which trade was closed
            direction: 1 for long/buy, -1 for short/sell
            trade_id: Optional trade identifier
            pnl: Optional profit/loss value
        """
        try:
            arrow_data = {
                'open_time': open_time,
                'close_time': close_time,
                'open_price': open_price,
                'close_price': close_price,
                'direction': direction,
                'trade_id': trade_id,
                'pnl': pnl
            }
            
            self.arrows_queue.put(arrow_data)
            logging.info(f"🏹 Arrow queued for trade #{trade_id}:")
            logging.info(f"   Start: {open_time} at {open_price:.5f}")
            logging.info(f"   End: {close_time} at {close_price:.5f}")
            logging.info(f"   Direction: {'Long' if direction == 1 else 'Short'}")
            
            logging.info(f"🏹 Arrow queued: {'Long' if direction == 1 else 'Short'} "
                        f"trade #{trade_id} from {open_price:.5f} to {close_price:.5f} "
                        f"PnL: {pnl:.5f}")
            
        except Exception as e:
            logging.error(f"Error adding trade arrow: {e}")
    
    def _create_arrow_patch(self, arrow_data):
        """Enhanced arrow with perfect triangle alignment"""
        try:
            from matplotlib.patches import FancyArrowPatch
            import matplotlib.dates as mdates
            
            direction = arrow_data['direction']
            pnl = arrow_data.get('pnl', 0)
            
            # Enhanced color scheme
            if direction == 1:  # Long position
                color = '#0066cc' if pnl > 0 else '#6bb6ff'
                alpha = 0.8 if pnl > 0 else 0.6
            else:  # Short position
                color = '#cc0000' if pnl > 0 else '#ff6b6b'
                alpha = 0.8 if pnl > 0 else 0.6
            
            # Arrow thickness
            linewidth = max(1.5, min(4, abs(pnl) * 500 + 2))
            
            # Convert timestamps to matplotlib format if needed
            open_time = arrow_data['open_time']
            close_time = arrow_data['close_time']
            
            # Ensure proper coordinate system
            if hasattr(open_time, 'timestamp'):
                open_x = mdates.date2num(open_time)
                close_x = mdates.date2num(close_time)
            else:
                open_x = open_time
                close_x = close_time
            
            # Create arrow with slight offset to avoid overlap with triangle
            arrow = FancyArrowPatch(
                (open_x, arrow_data['open_price']),
                (close_x, arrow_data['close_price']),
                arrowstyle='->',
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                mutation_scale=20,
                zorder=3,  # Just below triangles (zorder=5)
                linestyle='-' if pnl > 0 else '--',
                # Add small offset to start slightly away from triangle
                shrinkA=5,  # Shrink 5 points from start
                shrinkB=5   # Shrink 5 points from end
            )
            
            return arrow
            
        except Exception as e:
            logging.error(f"Error creating enhanced arrow: {e}")
            return None
        
    def _update_plot(self, frame):
        """Enhanced update plot with manual interaction support"""
        try:
            # Update counter for debugging
            self.update_count += 1
            
            # Control logic
            if self.is_paused and not self.step_requested:
                return
            
            if self.step_requested:
                self.step_requested = False
            
            # Process regular data
            data_updated = False
            processed_items = 0
            max_items = 1 if self.step_mode else float('inf')
            
            while not self.data_queue.empty() and processed_items < max_items:
                try:
                    data = self.data_queue.get_nowait()
                    processed_items += 1
                    
                    for key, value in data.items():
                        if key in self.plot_data and value is not None:
                            self.plot_data[key].append(value)
                    
                    data_updated = True
                    
                except Exception as e:
                    logging.warning(f"Error processing queued data: {e}")
                    break
            
            # Process arrow data
            arrows_added = 0
            while not self.arrows_queue.empty():
                try:
                    arrow_data = self.arrows_queue.get_nowait()
                    
                    # Create and add arrow patch
                    arrow_patch = self._create_arrow_patch(arrow_data)
                    if arrow_patch:
                        self.axes[0].add_patch(arrow_patch)
                        self.trade_arrows.append(arrow_patch)
                        self.trade_arrows_data.append(arrow_data)
                        arrows_added += 1
                        
                        logging.debug(f"🏹 Arrow #{len(self.trade_arrows)} added: "
                                    f"{'Long' if arrow_data['direction'] == 1 else 'Short'} "
                                    f"PnL: {arrow_data.get('pnl', 0):.5f}")
                    
                except Exception as e:
                    logging.warning(f"Error processing arrow: {e}")
                    break
            
            # Reset step mode
            if self.step_mode and processed_items > 0:
                self.step_mode = False
            
            # Update statistics
            data_points = len(self.plot_data['timestamps'])
            arrow_count = len(self.trade_arrows)
            self.stats_text.set_text(f'Data: {data_points:,} | Arrows: {arrow_count}')
            
            # Only update if we have data
            if not data_updated or data_points < 1:
                return
            
            # Convert deques to lists for plotting
            timestamps = list(self.plot_data['timestamps'])
            bid_prices = list(self.plot_data['bid_prices'])
            ask_prices = list(self.plot_data['ask_prices'])
            equity = list(self.plot_data['equity'])
            balance = list(self.plot_data['balance'])
            refined_states = list(self.plot_data['refined_states'])
            position_sizes_bid = list(self.plot_data['position_sizes_bid'])
            position_sizes_ask = list(self.plot_data['position_sizes_ask'])
            trades = list(self.plot_data['trades'])
            
            # Update price chart
            if len(timestamps) > 0 and len(bid_prices) > 0 and len(ask_prices) > 0:
                self.bid_line.set_data(timestamps, bid_prices)
                self.ask_line.set_data(timestamps, ask_prices)
            
            # 🎯 Position size plotting logic (same as before)
            scaling_factor = 200
            
            bullish_times, bullish_prices, bullish_sizes = [], [], []
            bearish_times, bearish_prices, bearish_sizes = [], [], []
            dropped_times_bid, dropped_prices_bid = [], []
            dropped_times_ask, dropped_prices_ask = [], []
            
            for i in range(len(timestamps)):
                if (i < len(refined_states) and i < len(position_sizes_bid) and 
                    i < len(position_sizes_ask) and i < len(bid_prices) and i < len(ask_prices)):
                    
                    state = refined_states[i]
                    bid_pos = position_sizes_bid[i]
                    ask_pos = position_sizes_ask[i]
                    
                    if state == 1 and bid_pos > 0:
                        bullish_times.append(timestamps[i])
                        bullish_prices.append(bid_prices[i])
                        bullish_sizes.append(bid_pos * scaling_factor)
                    
                    elif state == -1 and ask_pos > 0:
                        bearish_times.append(timestamps[i])
                        bearish_prices.append(ask_prices[i])
                        bearish_sizes.append(ask_pos * scaling_factor)
                    
                    elif state == 10:
                        dropped_times_bid.append(timestamps[i])
                        dropped_prices_bid.append(bid_prices[i])
                        dropped_times_ask.append(timestamps[i])
                        dropped_prices_ask.append(ask_prices[i])
            
            # Update position size scatter plots (same logic as before)
            if hasattr(self, 'bullish_positions') and hasattr(self, 'bearish_positions'):
                if bullish_times:
                    bullish_offsets = np.column_stack([bullish_times, bullish_prices])
                    self.bullish_positions.set_offsets(bullish_offsets)
                    self.bullish_positions.set_sizes(bullish_sizes)
                else:
                    self.bullish_positions.set_offsets(np.empty((0, 2)))
                
                if bearish_times:
                    bearish_offsets = np.column_stack([bearish_times, bearish_prices])
                    self.bearish_positions.set_offsets(bearish_offsets)
                    self.bearish_positions.set_sizes(bearish_sizes)
                else:
                    self.bearish_positions.set_offsets(np.empty((0, 2)))
            else:
                # Fallback to existing markers
                if bullish_times:
                    bullish_offsets = np.column_stack([bullish_times, bullish_prices])
                    self.buy_markers.set_offsets(bullish_offsets)
                    if hasattr(self.buy_markers, 'set_sizes'):
                        self.buy_markers.set_sizes(bullish_sizes)
                    self.buy_markers.set_color('blue')
                    self.buy_markers.set_alpha(0.6)
                else:
                    self.buy_markers.set_offsets(np.empty((0, 2)))
                
                if bearish_times:
                    bearish_offsets = np.column_stack([bearish_times, bearish_prices])
                    self.sell_markers.set_offsets(bearish_offsets)
                    if hasattr(self.sell_markers, 'set_sizes'):
                        self.sell_markers.set_sizes(bearish_sizes)
                    self.sell_markers.set_color('red')
                    self.sell_markers.set_alpha(0.6)
                else:
                    self.sell_markers.set_offsets(np.empty((0, 2)))
            
            # Handle trade signals (same as before)
            trade_buy_times, trade_buy_prices = [], []
            trade_sell_times, trade_sell_prices = [], []
            
            for i, (t, trade) in enumerate(zip(timestamps, trades)):
                if trade == 'buy' and i < len(ask_prices):
                    trade_buy_times.append(t)
                    trade_buy_prices.append(ask_prices[i])
                elif trade == 'sell' and i < len(bid_prices):
                    trade_sell_times.append(t)
                    trade_sell_prices.append(bid_prices[i])
            
            # Update other charts
            if len(timestamps) == len(equity):
                self.equity_line.set_data(timestamps, equity)
            if len(timestamps) == len(balance):
                self.balance_line.set_data(timestamps, balance)
            if len(timestamps) == len(refined_states):
                self.state_line.set_data(timestamps, refined_states)
            if len(timestamps) == len(position_sizes_bid):
                self.pos_bid_line.set_data(timestamps, position_sizes_bid)
            if len(timestamps) == len(position_sizes_ask):
                self.pos_ask_line.set_data(timestamps, position_sizes_ask)
            
            # 🎯 FIXED: Smart auto-scaling that respects user interaction
            for i, ax in enumerate(self.axes):
                try:
                    # Check if user has manually interacted with this axis
                    if not hasattr(self, '_user_xlim_overrides'):
                        self._user_xlim_overrides = {}
                    
                    # Get current axis limits
                    current_xlim = ax.get_xlim()
                    
                    # Check if limits were changed by user interaction
                    if i in self._user_xlim_overrides:
                        # Compare with previous limits to detect user changes
                        prev_xlim = self._user_xlim_overrides[i]
                        if abs(current_xlim[0] - prev_xlim[0]) > 1e-6 or abs(current_xlim[1] - prev_xlim[1]) > 1e-6:
                            # User has changed the view - don't auto-scale X axis
                            user_modified_view = True
                        else:
                            user_modified_view = False
                    else:
                        user_modified_view = False
                    
                    # Always update data ranges
                    ax.relim()
                    
                    if not user_modified_view and len(timestamps) > 10:
                        # Auto-scale only if user hasn't manually panned/zoomed
                        ax.autoscale_view()
                        
                        # Force latest data to appear on the right side
                        if hasattr(timestamps[0], 'timestamp'):  # pandas Timestamp
                            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                            margin_seconds = time_span * 0.02  # 2% margin
                            new_xlim = (
                                timestamps[0] - pd.Timedelta(seconds=margin_seconds), 
                                timestamps[-1] + pd.Timedelta(seconds=margin_seconds)
                            )
                            ax.set_xlim(new_xlim)
                        else:
                            # Numeric timestamps
                            time_span = timestamps[-1] - timestamps[0]
                            margin = time_span * 0.02
                            new_xlim = (timestamps[0] - margin, timestamps[-1] + margin)
                            ax.set_xlim(new_xlim)
                        
                        # Store the auto-set limits
                        self._user_xlim_overrides[i] = ax.get_xlim()
                    else:
                        # User has manually set view - only auto-scale Y axis and update data
                        ax.autoscale_view(scalex=False, scaley=True)
                        # Update stored limits to current user-set limits
                        self._user_xlim_overrides[i] = current_xlim
                    
                    # Always do Y-axis scaling for price chart
                    if i == 0 and bid_prices and ask_prices and not user_modified_view:
                        y_min = min(min(bid_prices), min(ask_prices))
                        y_max = max(max(bid_prices), max(ask_prices))
                        y_range = y_max - y_min
                        if y_range > 0:
                            margin_y = y_range * 0.05
                            ax.set_ylim(y_min - margin_y, y_max + margin_y)
                    
                except Exception as e:
                    logging.debug(f"Autoscale warning for axis {i}: {e}")
            
            # Debug logging
            if self.update_count % 100 == 0:
                bullish_count = len(bullish_times)
                bearish_count = len(bearish_times)
                dropped_count = len(dropped_times_bid)
                
                logging.info(f"🎨 Plot update #{self.update_count}: {data_points} points, "
                        f"{arrow_count} arrows, speed: {self.speed_multiplier:.1f}x")
                logging.debug(f"   Position dots: {bullish_count} bullish (blue), "
                            f"{bearish_count} bearish (red), {dropped_count} dropped (gray)")
                        
        except Exception as e:
            logging.error(f"❌ Error updating enhanced live plot: {e}")
            import traceback
            traceback.print_exc()
    
    def add_data_point(self, **kwargs):
        """Thread-safe method to add new data point"""
        try:
            # Default data structure
            default_data = {
                'timestamps': None,
                'bid_prices': None,
                'ask_prices': None,
                'equity': 0,
                'balance': 0,
                'refined_states': 0,
                'position_sizes_bid': 0,
                'position_sizes_ask': 0,
                'trades': None,
                'pnl': 0
            }
            
            # Update with provided data
            default_data.update(kwargs)
            
            # Only queue if we have essential data
            if default_data['timestamps'] is not None:
                self.data_queue.put(default_data)
                
        except Exception as e:
            logging.warning(f"Error adding data point: {e}")
    
    def set_speed(self, speed_multiplier):
        """Set speed programmatically"""
        try:
            if hasattr(self, 'speed_slider') and self.speed_slider:
                self.speed_slider.set_val(speed_multiplier)
            else:
                self._update_speed(speed_multiplier)
        except Exception as e:
            logging.error(f"Error setting speed: {e}")
    
    def show(self):
        """Display the enhanced live plot"""
        try:
            if self.fig:
                plt.show(block=False)
                logging.info("🎨 Enhanced live plot displayed with full controls")
            else:
                logging.error("No figure to show")
        except Exception as e:
            logging.error(f"Error showing live plot: {e}")
    
    def close(self):
        """Close the live plot"""
        try:
            if hasattr(self, 'animation') and self.animation:
                self.animation.event_source.stop()
            if hasattr(self, 'fig') and self.fig:
                plt.close(self.fig)
            logging.info("Enhanced live plot closed")
        except Exception as e:
            logging.error(f"Error closing live plot: {e}")
    
    def pause(self):
        """Pause the animation programmatically"""
        try:
            if not self.is_paused:
                self._toggle_pause(None)
        except Exception as e:
            logging.error(f"Error pausing: {e}")
    
    def resume(self):
        """Resume the animation programmatically"""
        try:
            if self.is_paused:
                self._toggle_pause(None)
        except Exception as e:
            logging.error(f"Error resuming: {e}")
    
    def clear_data(self):
        """Clear all plot data programmatically"""
        try:
            self._reset_data(None)
        except Exception as e:
            logging.error(f"Error clearing data: {e}")
    
    def get_plot_stats(self):
        """Get comprehensive plot statistics"""
        try:
            return {
                'data_points': len(self.plot_data['timestamps']),
                'queue_size': self.data_queue.qsize(),
                'arrows_count': len(self.trade_arrows),
                'arrows_queue_size': self.arrows_queue.qsize(),
                'speed_multiplier': self.speed_multiplier,
                'update_interval': self.current_interval,
                'max_points': self.max_points,
                'is_paused': self.is_paused,
                'step_mode': self.step_mode,
                'update_count': self.update_count
            }
        except Exception as e:
            logging.error(f"Error getting plot stats: {e}")
            return {}

    def debug_alignment(self):
        """Debug method to check triangle-arrow alignment"""
        print("\n🔍 ALIGNMENT DEBUG:")
        
        # Get current triangle positions
        buy_offsets = self.buy_markers.get_offsets()
        sell_offsets = self.sell_markers.get_offsets()
        
        print(f"📍 Current triangle positions:")
        print(f"   Buy triangles (green ▲): {len(buy_offsets)} points")
        for i, offset in enumerate(buy_offsets):
            if i < 3:  # Show first 3
                print(f"     {i+1}: ({offset[0]}, {offset[1]:.5f})")
        
        print(f"   Sell triangles (red ▼): {len(sell_offsets)} points") 
        for i, offset in enumerate(sell_offsets):
            if i < 3:  # Show first 3
                print(f"     {i+1}: ({offset[0]}, {offset[1]:.5f})")
        
        print(f"📍 Current arrow positions:")
        print(f"   Total arrows: {len(self.trade_arrows_data)}")
        for i, arrow_data in enumerate(list(self.trade_arrows_data)[-3:]):  # Show last 3
            print(f"     {i+1}: Start({arrow_data['open_time']}, {arrow_data['open_price']:.5f})")
            print(f"        End({arrow_data['close_time']}, {arrow_data['close_price']:.5f})")

    def add_control_observer(self, observer: PlotControlObserver):
        """Register an observer for plot control events"""
        self.control_observers.append(observer)
        logging.info(f"🎛️ Plot control observer registered: {type(observer).__name__}")
    
    def remove_control_observer(self, observer):
        """Unregister an observer"""
        if observer in self.control_observers:
            self.control_observers.remove(observer)
            logging.info(f"🎛️ Plot control observer removed: {type(observer).__name__}")
    
    def _notify_pause(self):
        """Notify all observers of pause event"""
        for observer in self.control_observers:
            try:
                observer.on_plot_pause()
            except Exception as e:
                logging.error(f"Observer pause notification failed: {e}")
    
    def _notify_resume(self):
        """Notify all observers of resume event"""
        for observer in self.control_observers:
            try:
                observer.on_plot_resume()
            except Exception as e:
                logging.error(f"Observer resume notification failed: {e}")
    
    def _notify_step(self):
        """Notify all observers of step event"""
        for observer in self.control_observers:
            try:
                observer.on_plot_step()
            except Exception as e:
                logging.error(f"Observer step notification failed: {e}")
    
    # 🎯 UPDATE EXISTING METHODS TO NOTIFY OBSERVERS
    def _toggle_pause(self, event):
        """Enhanced pause toggle with observer notification"""
        try:
            self.is_paused = not self.is_paused
            
            if self.is_paused:
                self._notify_pause()  # 🎯 Notify observers first
                self.play_pause_button.label.set_text('▶️ Resume')
                self.play_pause_button.color = '#45b7d1'
                self.status_text.set_text('Status: ⏸️ Paused')
                self.status_text.set_color('#e67e22')
                if self.animation:
                    self.animation.pause()
                logging.info("📊 Plot paused - observers notified")
            else:
                self._notify_resume()  # 🎯 Notify observers first
                self.play_pause_button.label.set_text('⏸️ Pause')
                self.play_pause_button.color = '#ff6b6b'
                self.status_text.set_text('Status: ▶️ Running')
                self.status_text.set_color('#27ae60')
                if self.animation:
                    self.animation.resume()
                logging.info("📊 Plot resumed - observers notified")
            
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            logging.error(f"Error toggling pause: {e}")
    
    def _step_forward(self, event):
        """Enhanced step forward with observer notification"""
        try:
            # Auto-pause if not already paused
            if not self.is_paused:
                self._toggle_pause(event)
            
            self.step_requested = True
            self.step_mode = True
            self.status_text.set_text('Status: ⏭️ Step Mode')
            self.status_text.set_color('#8e44ad')
            
            # 🎯 Notify observers of step request
            self._notify_step()
            
            # Process one update for the plot itself
            self._update_plot(None)
            self.fig.canvas.draw_idle()
            
            logging.info("📊 Step forward - observers notified")
            
        except Exception as e:
            logging.error(f"Error stepping forward: {e}")

# Quick test functionality
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the LivePlotManager
    print("🧪 Testing LivePlotManager...")
    
    plotter = LivePlotManager(max_points=100)
    print("✅ LivePlotManager created successfully")
    
    # Check critical methods
    methods = ['add_data_point', 'add_trade_arrow', 'show', 'close']
    for method in methods:
        if hasattr(plotter, method):
            print(f"✅ {method} method exists")
        else:
            print(f"❌ {method} method missing")
    
    # Check critical attributes
    attributes = ['trade_arrows', 'arrows_queue', 'plot_data']
    for attr in attributes:
        if hasattr(plotter, attr):
            print(f"✅ {attr} attribute exists")
        else:
            print(f"❌ {attr} attribute missing")
    
    print("\n🎯 LivePlotManager ready for integration!")
    print("Features included:")
    print("  ✅ Interactive controls (pause/resume/step)")
    print("  ✅ Speed control slider (0.1x to 5.0x)")
    print("  ✅ Buffer size control")
    print("  ✅ Trade arrows (blue=long, red=short)")
    print("  ✅ Latest data appears on right side")
    print("  ✅ Enhanced styling and error handling")

