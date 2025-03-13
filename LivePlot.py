# LivePlot.py
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.widgets as widgets
import logging
from matplotlib.animation import FuncAnimation

class LivePlotWithSlider:
    def __init__(self, window_size=1000):
        """
        Initialize the live plot with required subplots and plot elements.
        
        Args:
            window_size (int): Number of recent ticks to display.
        """
        # Initialize deques for storing data with fixed maximum length
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.bid = deque(maxlen=window_size)
        self.ask = deque(maxlen=window_size)
        self.equity_curve = deque(maxlen=window_size)
        self.balance = deque(maxlen=window_size)
        self.pnl = deque(maxlen=window_size)
        self.position_size_bid = deque(maxlen=window_size)
        self.position_size_ask = deque(maxlen=window_size)
        self.positions = deque(maxlen=window_size)
        
        # Initialize Matplotlib in interactive mode
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(20, 12), 
                                         gridspec_kw={'height_ratios': [2, 2, 1]},
                                         constrained_layout=True)
        
        # Subplot 1: Bid and Ask Prices with Position Markers and Arrows
        self.ax1 = self.axs[0]
        self.line_bid, = self.ax1.plot([], [], color='black', linewidth=0.5, label='Bid')
        self.line_ask, = self.ax1.plot([], [], color='gray', linestyle='--', linewidth=0.5, label='Ask')
        self.scatter_bid = self.ax1.scatter([], [], c='blue', s=[], alpha=0.6, label='Bullish Positions (Bid)')
        self.scatter_ask = self.ax1.scatter([], [], c='red', s=[], alpha=0.6, label='Bearish Positions (Ask)')
        self.ax1.set_title('Bid and Ask Prices with Positions')
        self.ax1.set_ylabel('Price')
        self.ax1.legend(loc='upper left')
        self.ax1.grid(True)
        
        # Subplot 2: Equity Curve, Balance, and PnL
        self.ax2 = self.axs[1]
        self.line_equity, = self.ax2.plot([], [], color='green', linewidth=1.5, label='Equity Curve')
        self.line_balance, = self.ax2.plot([], [], color='orange', linewidth=1.0, label='Balance Curve')
        self.ax2_twin = self.ax2.twinx()
        self.line_pnl, = self.ax2_twin.plot([], [], color='blue', linewidth=1.5, linestyle='--', label='PnL')
        self.ax2_twin.set_ylabel('PnL', color='blue')
        self.ax2_twin.tick_params(axis='y', colors='blue')
        self.ax2.set_title('Equity Curve and PnL')
        self.ax2.set_ylabel('Equity', color='green')
        self.ax2.tick_params(axis='y', colors='green')
        self.ax2.legend(loc='upper left')
        self.ax2_twin.legend(loc='upper right')
        self.ax2.grid(True)
        
        # Subplot 3: Slider and Controls
        self.ax_slider = self.axs[2]
        self.ax_slider.axis('off')  # Hide the axes
        
        # Initialize Slider
        self.slider_ax = self.fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
        self.slider = widgets.Slider(
            ax=self.slider_ax,
            label='Tick',
            valmin=0,
            valmax=window_size - 1,
            valinit=0,
            valstep=1,
            color='lightblue'
        )
        self.slider.on_changed(self.update_plot_from_slider)
        
        # Add Play and Pause buttons
        self.play_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])  # [left, bottom, width, height]
        self.pause_ax = self.fig.add_axes([0.7, 0.025, 0.1, 0.04])
        self.play_button = widgets.Button(self.play_ax, 'Play', color='lightgreen', hovercolor='green')
        self.pause_button = widgets.Button(self.pause_ax, 'Pause', color='lightcoral', hovercolor='red')
        
        self.play_button.on_clicked(self.start_animation)
        self.pause_button.on_clicked(self.stop_animation)
        
        # Animation control
        self.anim = None
        self.is_playing = False
        self.current_tick = 0
        
        # Add text box for current tick info
        self.info_ax = self.fig.add_axes([0.1, 0.95, 0.8, 0.04])
        self.info_text = self.info_ax.text(0.5, 0.5, '', transform=self.info_ax.transAxes, 
                                          ha='center', va='center', fontsize=12)
        self.info_ax.axis('off')
        
        # Formatting X-axis for bid/ask plots
        self.axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())
        self.axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Initialize plot limits
        self.ax1.set_xlim(pd.Timestamp.now(), pd.Timestamp.now() + pd.Timedelta(seconds=10))
        self.ax1.set_ylim(0, 1)  # Adjust based on expected price range
        self.ax2.set_ylim(0, 1)  # Adjust based on expected equity range
        self.ax2_twin.set_ylim(-1, 1)  # Adjust based on expected PnL range
        
        # Display the plot in non-blocking mode
        plt.show(block=False)
        
        # Initialize plotted positions
        self.plotted_positions = set()
    
    def update_plot_from_slider(self, val):
        """
        Update the plot based on the slider's current value.
        
        Args:
            val (float): Current value of the slider (tick index).
        """
        tick = int(val)
        self.current_tick = tick
        self.update_plot(tick)
    
    def update_plot(self, tick):
        """
        Update the plot up to the given tick index.
        
        Args:
            tick (int): Current tick index.
        """
        if tick < 0 or tick >= len(self.timestamps):
            return
        
        # Define the window of data to display
        start = max(tick - self.window_size, 0)
        end = tick + 1  # Include the current tick
        
        # Slice the data from deques
        timestamps = list(self.timestamps)[start:end]
        bid = list(self.bid)[start:end]
        ask = list(self.ask)[start:end]
        equity_curve = list(self.equity_curve)[start:end]
        balance = list(self.balance)[start:end]
        pnl = list(self.pnl)[start:end]
        position_size_bid = list(self.position_size_bid)[start:end]
        position_size_ask = list(self.position_size_ask)[start:end]
        # Ensure self.positions is converted to a DataFrame before querying
        if isinstance(self.positions, deque):
            positions_df = pd.DataFrame(list(self.positions))  # Convert deque to DataFrame
        else:
            positions_df = self.positions  # Assume it's already a DataFrame
        positions_subset = positions_df[positions_df['open_id'] <= tick]
        # Update Subplot 1: Bid and Ask Prices
        self.line_bid.set_data(timestamps, bid)
        self.line_ask.set_data(timestamps, ask)
        
        # Update scatter plots
        scaling_factor_plot = 100  # Adjust as needed
        marker_size_bid = np.array(position_size_bid) * scaling_factor_plot
        marker_size_ask = np.array(position_size_ask) * scaling_factor_plot
        
        if len(timestamps) > 0:
            # Convert timestamps to Matplotlib's numeric format
            num_timestamps = mdates.date2num(timestamps)
            self.scatter_bid.set_offsets(np.c_[num_timestamps, ask])
            self.scatter_bid.set_sizes(marker_size_bid)
            self.scatter_ask.set_offsets(np.c_[num_timestamps, bid])
            self.scatter_ask.set_sizes(marker_size_ask)
        else:
            self.scatter_bid.set_offsets(np.empty((0, 2)))
            self.scatter_ask.set_offsets(np.empty((0, 2)))
        
        # Remove previous arrows
        for artist in self.ax1.artists:
            artist.remove()
        if len(self.ax1.lines) > 2:
            # Remove any additional lines beyond the first two (Bid and Ask)
            for line in self.ax1.lines[2:]:
                line.remove()
        
        # Plot arrows for closed positions
        num_long = 0
        num_short = 0
        
        for idx, row in positions_subset.iterrows():
            if row['close_id'] > 0 and row['close_id'] <= tick and idx not in self.plotted_positions:
                open_tick = int(row['open_id']) - 1  # Zero-based index
                close_tick = int(row['close_id']) - 1
                
                if open_tick < start or open_tick >= end or close_tick < start or close_tick >= end:
                    continue  # Skip if outside the current window
                
                aligned_open_price = row['open_price']
                aligned_close_price = row['close_price']
                
                if row['direction'] > 0:
                    color = 'blue'
                    num_long += 1
                else:
                    color = 'red'
                    num_short += 1
                
                # Convert timestamps to numeric format for plotting
                open_time = mdates.date2num(timestamps[open_tick - start])
                close_time = mdates.date2num(timestamps[close_tick - start])
                
                arrow = self.ax1.annotate(
                    '',
                    xy=(close_time, aligned_close_price),
                    xytext=(open_time, aligned_open_price),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5)
                )
                self.ax1.artists.append(arrow)
                
                # Mark this position as plotted
                self.plotted_positions.add(idx)
        
        logging.info(f"Tick: {tick} | Number of Long: {num_long} | Number of Short: {num_short}")
        
        # Update Subplot 2: Equity, Balance, PnL
        self.line_equity.set_data(timestamps, equity_curve)
        self.line_balance.set_data(timestamps, balance)
        self.line_pnl.set_data(timestamps, pnl)
        
        # Update X-axis limits only if there are enough data points
        if len(timestamps) >= 2:
            self.ax1.set_xlim(mdates.date2num(timestamps[0]), mdates.date2num(timestamps[-1]))
            self.ax2.set_xlim(mdates.date2num(timestamps[0]), mdates.date2num(timestamps[-1]))
            self.ax2_twin.set_xlim(mdates.date2num(timestamps[0]), mdates.date2num(timestamps[-1]))
        elif len(timestamps) == 1:
            current_time = mdates.date2num(timestamps[0])
            self.ax1.set_xlim(current_time, current_time + 1e-6)  # Small delta to prevent identical xlims
            self.ax2.set_xlim(current_time, current_time + 1e-6)
            self.ax2_twin.set_xlim(current_time, current_time + 1e-6)
        else:
            # No data; set default xlim
            default_time = mdates.date2num(pd.Timestamp.now())
            self.ax1.set_xlim(default_time, default_time + 1e-6)
            self.ax2.set_xlim(default_time, default_time + 1e-6)
            self.ax2_twin.set_xlim(default_time, default_time + 1e-6)
        
        # Rescale Y-axis limits
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2_twin.relim()
        self.ax2_twin.autoscale_view()
        
        # Update tick information
        self.update_plot_with_info(tick)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
    
    def update_plot_with_info(self, tick):
        """
        Update the plot and display current tick information.
        
        Args:
            tick (int): Current tick index.
        """
        if tick < len(self.timestamps):
            info = (
                f"Tick: {tick} | Timestamp: {self.timestamps[tick]} | "
                f"Bid: {self.bid[tick]:.5f} | Ask: {self.ask[tick]:.5f} | "
                f"Equity: {self.equity_curve[tick]:.2f} | Balance: {self.balance[tick]:.2f} | "
                f"PnL: {self.pnl[tick]:.2f}"
            )
            self.info_text.set_text(info)
    
    def animate(self, frame):
        """
        Animation function called by FuncAnimation.
        """
        if self.current_tick < len(self.timestamps) - 1:
            self.current_tick += 1
            self.slider.set_val(self.current_tick)
    
    def start_animation(self, event):
        if not self.is_playing:
            self.anim = FuncAnimation(self.fig, self.animate, frames=None, interval=50, blit=False)
            self.is_playing = True
            logging.info("Animation started.")
    
    def stop_animation(self, event):
        if self.is_playing:
            self.anim.event_source.stop()
            self.is_playing = False
            logging.info("Animation paused.")