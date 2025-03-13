import pandas as pd
import numpy as np
import logging
from TradingStrategy import TradingStrategy  # Import the TradingStrategy class
import argparse
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# import multiprocessing
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def precompile_numba_functions():
    """
    Precompile Numba-accelerated functions with dummy data to avoid runtime compilation in threads.
    """
    from numba_functions import update_positions_numba_parallel, calculate_pnl_numba, calculate_floating_pnl_numba
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
        ('profit_threshold', 'f4'),
        ('loss_threshold', 'f4'),
        ('closing_method', 'i4'),    # closing_method (str)
        ('pnl', 'f4')
    ])
    dummy_positions['close_id'] = -1
    dummy_positions['closing_method'] = 1 
    dummy_positions['direction'] = 1
    dummy_positions['open_price'] = 1.2000
    dummy_positions['volume'] = 0.1

    update_positions_numba_parallel(
        positions=dummy_positions,
        current_bid=1.1995,
        current_ask=1.2005,
        tick_id=0,
        # profit_threshold=0.005,
        # loss_threshold=0.002,
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
    # Precompile calculate_floating_pnl_numba
    calculate_floating_pnl_numba(
        positions=dummy_positions,
        current_bid=1.1995,
        current_ask=1.2005,
        slippage=0.0001
    )
def parse_arguments():
    """
    Parse command-line arguments.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run the Trading Strategy with specified parameters.')
    
    # Define total_points argument
    parser.add_argument(
        '--total_points',
        type=int,
        default=10000,  # Set a sensible default
        help='Total number of data points (ticks) to process.'
    )
    
    # Define start_points argument
    parser.add_argument(
        '--start_points',
        type=int,
        default=60000,  # Set a sensible default
        help='Number of initial data points to start processing from.'
    )
    
    return parser.parse_args()

def main():
    # ---------------------- Step 1: Load and Preprocess Data ----------------------
    # Initialize DataManager

    # Parse command-line arguments
    args = parse_arguments()
    total_points = args.total_points
    start_points = args.start_points
    filename = 'EURUSD_mt5_ticks-m.csv'

    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        logging.error(f"File {filename} not found. Please check the file path.")
        return
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        return

    # Limit data for performance
    # total_points = 6000
    # start_points = 60000
    data = data.iloc[start_points:start_points+total_points].reset_index(drop=True)
    data['tick_id'] = data.index

    # Extract relevant variables
    try:
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d %H:%M:%S.%f')
    except Exception as e:
        logging.error(f"Error parsing timestamps: {e}")
        return

    # Rename columns for clarity
    data = data.rename(columns={"timestamp": "timestamp", "bid": "bid", "ask": "ask"})

    # ---------------------- Step 2: Initialize Parameters ----------------------
    config = {
        "buffer_size": total_points,
        "ema_window": 20,
        "training_window_size": 3000,
        "rolling_window_size": 2000,
        "forecast_steps": 2,
        "num_simulations": 2,
        "max_window_size": 10,
        "k": 8e-7,  # Reduced scaling factor
        "margin_requirement": 0.01,  # 1% margin requirement
        "max_spread": 2e-4,
        "direction_multiplier": -1,
        "pip_value": 0.0001,  # Value of one pip
        'adaptive_min_sigma_bid': np.nan,
        'adaptive_min_sigma_ask': np.nan,
        'is_trained': False,
        'forecasted_ask_change': 0,
        'forecasted_bid_change': 0,
        'max_position_size': .5,
        "drawdown_threshold": 0.10,  # 10% drawdown threshold
        "broker_config": {          # Added broker_config for TradeManager
            "initial_capital": 10000,  # Example: $100,000
            "commission_per_lot": 0.02,
            "slippage_points": 0.00002,
            "leverage": 500,
            "min_volume": 0.01,
            "max_volume": 100,
            "volume_step": 0.01,
            "profit_threshold": 0.02,
            "loss_threshold": 0.01,
            "closing_method": 1
        }
    }
    # precompile_numba_functions()

    # Initialize the trading strategy
    strategy = TradingStrategy(config)

    # ---------------------- Step 3: Run the Strategy ----------------------
    logging.info("Starting backtest...")

    strategy.run_strategy(data)


    # ---------------------- Step 4: Analyze Results ----------------------
    results = strategy.get_results()
    equity_curve = results["equity_curve"]
    balance = results["balance"]
    # positions_bid = results["positions_bid"]
    # positions_ask = results["positions_ask"]
    positions_bid = results["adjusted_position_size_bid"]
    positions_ask = results["adjusted_position_size_ask"]
    refined_state = results["refined_state"]
    positions = results["positions"]
    pnl = results["pnl"]

    # Count and print the states
    num_bullish = np.sum(refined_state == 1)
    num_bearish = np.sum(refined_state == -1)
    num_neutral = np.sum(refined_state == 0)
    # num_long = np.sum(positions['direction'] == 1)
    # num_short = np.sum(positions['direction'] == -1)

    logging.info(f"Number of Bullish states: {num_bullish}")
    logging.info(f"Number of Bearish states: {num_bearish}")
    logging.info(f"Number of Neutral states: {num_neutral}")
    # logging.info(f"Number of Long: {num_long}")
    # logging.info(f"Number of Short: {num_short}")

    # num_bullish = np.sum(initial_state == 1)
    # num_bearish = np.sum(initial_state == -1)
    # num_neutral = np.sum(initial_state == 0)

    # logging.info(f"Number of Bullish initial states: {num_bullish}")
    # logging.info(f"Number of Bearish initial states: {num_bearish}")
    # logging.info(f"Number of Neutral initial states: {num_neutral}")


    # logging.info(f"Final Equity: {equity_curve.iloc[-1]}")
    # logging.info(f"Total PnL: {pnl.sum()}")

    # ---------------------- Step 5: Plot Results (Optional) ----------------------
    plot_results(data, equity_curve, positions_bid, positions_ask, refined_state)


    # plot_results(data, equity_curve, positions_bid, positions_ask, refined_state,pd.DataFrame(positions),pnl,balance)

def plot_results(data, equity_curve, positions_bid, positions_ask, refined_state):
    """
    Plot the results of the backtest.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    import matplotlib.ticker as ticker

    timestamps = data['timestamp'].values
    bid = data['bid'].values
    ask = data['ask'].values
    total_points = len(data)
    scaling_factor_plot = 100  # Scaling factor for visibility (adjust as needed)
    marker_size_bid = (positions_bid * scaling_factor_plot)  # Marker sizes for f_bid
    marker_size_ask = (positions_ask * scaling_factor_plot)  # Marker sizes for f_ask

    # Set default gray color
    state_colors_bid = np.full((total_points, 3), [0.5, 0.5, 0.5])  # Gray for neutral state
    state_colors_ask = np.full((total_points, 3), [0.5, 0.5, 0.5])  # Gray for neutral state

    state_colors_bid[refined_state == 1] = [0, 0, 1]  # Blue for Bullish on bid
    state_colors_ask[refined_state == -1] = [1, 0, 0]  # Red for Bearish on ask
    # Define pip increment for grid
    pip_value = 0.0001  # Set this based on the currency pair
    pip_increment = 2  # Number of pips per gridline
    pip_step = pip_value * pip_increment

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot bid and ask prices on the primary y-axis
    ax1.plot(timestamps, bid, color='black', linewidth=0.5, label='Bid')
    ax1.plot(timestamps, ask, color='gray', linestyle='--', linewidth=0.5, label='Ask')
    # Add pip-based gridlines
    min_price = min(min(bid), min(ask))
    max_price = max(max(bid), max(ask))

    # Set y-ticks based on pip increments
    yticks = np.arange(min_price, max_price, pip_step)
    ax1.set_yticks(yticks)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.5f}"))  # Format as a 5-decimal price
    ax1.grid(axis='y', which='major', linestyle='--', linewidth=0.5)

    # Add titles and labels
    # ax1.set_title('Bid and Ask Prices with Pip-Based Grid')
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Price (Pips)')
    # Ensure marker_size_bid and marker_size_ask have the same length as timestamps
    if len(marker_size_bid) != len(timestamps):
        marker_size_bid = np.zeros(len(timestamps))

    if len(marker_size_ask) != len(timestamps):
        marker_size_ask = np.zeros(len(timestamps))

    # Plot Bullish and Bearish positions with marker sizes based on f_bid and f_ask
    ax1.scatter(
        timestamps,
        bid,
        c='blue',
        s=marker_size_bid,
        alpha=0.6,
        label='Bullish Positions (f_bid)'
    )
    ax1.scatter(
        timestamps,
        ask,
        c='red',
        s=marker_size_ask,
        alpha=0.6,
        label='Bearish Positions (f_ask)'
    )

    # Configure the primary y-axis
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price (Pips)')
    ax1.set_title('Bid and Ask Prices with Market States and Position Sizes')
    # ax1.grid(True)

    # Format the x-axis for timestamps
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Create a secondary y-axis for position sizes
    ax2 = ax1.twinx()

    # Plot position sizes on the secondary y-axis
    ax2.plot(timestamps, positions_bid, color='blue', alpha=0.3, label='Position Size Bid')
    ax2.plot(timestamps, positions_ask, color='red', alpha=0.3, label='Position Size Ask')
    # ax2.plot(timestamps, position_size_ewma_combined, color='gray', alpha=0.3, label='Position Size EWMA')

    # Configure the secondary y-axis
    ax2.set_ylabel('Position Size (Fraction of Equity)')
    # Adjust ylim for visibility based on actual data
    max_f_bid = np.nanmax(positions_bid)# if np.max(positions_bid) > 0 else 1
    max_f_ask = np.nanmax(positions_ask)# if np.max(positions_ask) > 0 else 1
    ax2.set_ylim(0, max(max_f_bid, max_f_ask) * 1.1)  # 10% buffer
    ax2.legend(loc='upper right')

    # Create custom legend handles for marker sizes
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Bullish Positions (f_bid)',
               markerfacecolor='blue', markersize=10, alpha=0.6),
        Line2D([0], [0], marker='o', color='w', label='Bearish Positions (f_ask)',
               markerfacecolor='red', markersize=10, alpha=0.6)
    ]

    # Optional: Define a reference position size for the legend
    reference_f = 0.1  # Example reference position size
    reference_marker_size = reference_f * scaling_factor_plot
    reference_handle = Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=f'Reference Position Size (f={reference_f})',
        markerfacecolor='gray',
        markersize=np.sqrt(reference_marker_size),
        alpha=0.6
    )

    # Add the custom legend to the plot
    ax1.legend(handles=legend_elements + [reference_handle], loc='upper left')

    plt.show()

# def plot_results(data, equity_curve, positions_bid, positions_ask, refined_state, positions):
#     """
#     Plot the results of the backtest, including:
#     - Bid and Ask Prices with Position Markers
#     - Equity Curve
#     - Open/Close Position Arrows
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.dates as mdates
#     from matplotlib.lines import Line2D
#     import matplotlib.ticker as ticker
#     import pandas as pd
#     import numpy as np

#     # Extract data for plotting
#     timestamps = data['timestamp'].values
#     bid = data['bid'].values
#     ask = data['ask'].values
#     scaling_factor_plot = 100  # Scaling factor for position size markers
#     marker_size_bid = (positions_bid * scaling_factor_plot)
#     marker_size_ask = (positions_ask * scaling_factor_plot)

#     # Ensure positions is a DataFrame
#     if not isinstance(positions, pd.DataFrame):
#         raise TypeError("The 'positions' argument must be a pandas DataFrame.")

#     # Create a mapping from tick_id to timestamp for plotting arrows
#     tick_id_to_timestamp = pd.Series(data['timestamp'].values, index=data['tick_id']).to_dict()

#     fig, ax1 = plt.subplots(figsize=(14, 8))

#     # Plot bid and ask prices
#     ax1.plot(timestamps, bid, color='black', linewidth=0.5, label='Bid')
#     ax1.plot(timestamps, ask, color='gray', linestyle='--', linewidth=0.5, label='Ask')

#     # Add pip-based gridlines
#     pip_value = 0.0001  # Adjust based on the currency pair
#     pip_increment = 2  # Number of pips per gridline
#     pip_step = pip_value * pip_increment
#     min_price = min(min(bid), min(ask))
#     max_price = max(max(bid), max(ask))
#     yticks = np.arange(min_price, max_price, pip_step)
#     ax1.set_yticks(yticks)
#     ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.10f}"))  # Format as a 5-decimal price
#     ax1.grid(axis='y', which='major', linestyle='--', linewidth=0.5)

#     # Plot position sizes as scatter points
#     ax1.scatter(
#         timestamps,
#         bid,
#         c='blue',
#         s=marker_size_bid,
#         alpha=0.6,
#         label='Bullish Positions (Bid)'
#     )
#     ax1.scatter(
#         timestamps,
#         ask,
#         c='red',
#         s=marker_size_ask,
#         alpha=0.6,
#         label='Bearish Positions (Ask)'
#     )

#     for _, row in positions.iterrows():
#         if row['close_id'] > 0:  # Only plot closed positions
#             open_time = tick_id_to_timestamp.get(row['open_id'])
#             close_time = tick_id_to_timestamp.get(row['close_id'])

#             # Ensure open_time and close_time are compatible with timestamps
#             if open_time and close_time:
#                 open_time = np.datetime64(open_time)
#                 close_time = np.datetime64(close_time)

#                 open_idx = np.argmin(np.abs(timestamps - open_time))
#                 close_idx = np.argmin(np.abs(timestamps - close_time))

#                 # Align prices with bid/ask
#                 aligned_open_price = bid[open_idx] if row['direction'] > 0 else ask[open_idx]
#                 aligned_close_price = ask[close_idx] if row['direction'] > 0 else bid[close_idx]

#                 # Choose arrow color based on direction
#                 color = 'red' if row['direction']*1 > 0 else 'blue'

#                 # Plot the arrow
#                 ax1.annotate(
#                     '',
#                     xy=(timestamps[close_idx], aligned_close_price),
#                     xytext=(timestamps[open_idx], aligned_open_price),
#                     arrowprops=dict(arrowstyle='->', color=color, lw=1.0)
#                 )

#     # Secondary y-axis for equity curve
#     ax2 = ax1.twinx()
#     ax2.plot(timestamps, equity_curve, color='green', linewidth=1.5, label='Equity Curve')
#     ax2.set_ylabel('Equity')
#     ax2.legend(loc='upper right')

#     # Configure the legend
#     custom_legend = [
#         Line2D([0], [0], color='green', lw=1.5, label='Equity Curve'),
#         Line2D([0], [0], color='green', lw=1.5, label='Long Position (Arrow)'),
#         Line2D([0], [0], color='red', lw=1.5, label='Short Position (Arrow)'),
#         Line2D([0], [0], marker='o', color='w', label='Bullish Positions (Bid)',
#                markerfacecolor='blue', markersize=10, alpha=0.6),
#         Line2D([0], [0], marker='o', color='w', label='Bearish Positions (Ask)',
#                markerfacecolor='red', markersize=10, alpha=0.6)
#     ]
#     ax1.legend(handles=custom_legend, loc='upper left')

#     plt.show()
    
# def plot_results(data, equity_curve, positions_bid, positions_ask, refined_state, positions, pnl,balance):
#     """
#     Plot the results of the backtest, including:
#     - Bid and Ask Prices with Position Markers
#     - Equity Curve
#     - Open/Close Position Arrows
#     - PnL Plot
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.dates as mdates
#     from matplotlib.lines import Line2D
#     import matplotlib.ticker as ticker
#     import pandas as pd
#     import numpy as np

#     # Extract data for plotting
#     timestamps = data['timestamp'].values
#     bid = data['bid'].values
#     ask = data['ask'].values
#     scaling_factor_plot = 100  # Scaling factor for position size markers
#     marker_size_bid = (positions_bid * scaling_factor_plot)
#     marker_size_ask = (positions_ask * scaling_factor_plot)

#     # Ensure positions is a DataFrame
#     # if not isinstance(positions, pd.DataFrame):
#     #     raise TypeError("The 'positions' argument must be a pandas DataFrame.")

#     fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True, constrained_layout=True)

#     # ---------------------- 1. Bid and Ask Prices with Position Markers ----------------------
#     ax1 = axs[0]
#     ax1.plot(timestamps, bid, color='black', linewidth=0.5, label='Bid')
#     ax1.plot(timestamps, ask, color='gray', linestyle='--', linewidth=0.5, label='Ask')

#     # Plot position sizes as scatter points
#     ax1.scatter(
#         timestamps,
#         ask,
#         c='blue',
#         s=marker_size_bid,
#         alpha=0.6,
#         label='Bullish Positions (Ask)'
#     )
#     ax1.scatter(
#         timestamps,
#         bid,
#         c='red',
#         s=marker_size_ask,
#         alpha=0.6,
#         label='Bearish Positions (Bid)'
#     )
#     num_long = 0
#     num_short = 0
#     # Plot arrows for open/close positions
#     for _, row in positions.iterrows():
#         if row['close_id'] > 0:  # Only plot closed positions
#             open_idx = int(row['open_id'])  # Use index directly
#             close_idx = int(row['close_id'])  # Use index directly

#             aligned_open_price = ask[open_idx] if row['direction'] > 0 else bid[open_idx]
#             aligned_close_price = bid[close_idx] if row['direction'] > 0 else ask[close_idx]
#             aligned_open_price = row['open_price']
#             aligned_close_price = row['close_price']

#             if row['direction'] > 0:
#                 color = 'blue'
#                 num_long+=1
#             else:
#                 color = 'red'
#                 num_short+=1

#             ax1.annotate(
#                 '',
#                 xy=(timestamps[close_idx], aligned_close_price),
#                 xytext=(timestamps[open_idx], aligned_open_price),
#                 arrowprops=dict(arrowstyle='->', color=color, lw=1.5)
#             )

#     logging.info(f"Number of Long: {num_long}")
#     logging.info(f"Number of Short: {num_short}")
#     ax1.set_title('Bid and Ask Prices with Positions')
#     ax1.set_ylabel('Price')
#     ax1.legend(loc='upper left')

#     # ---------------------- 2. Combined Equity and PnL Plot ----------------------
#     ax2 = axs[1]
#     ax2.plot(timestamps, equity_curve, color='green', linewidth=1.5, label='Equity Curve')
#     ax2.plot(timestamps, balance, color='orange', linewidth=1.0, label='balance Curve')

#     # Add a secondary y-axis for PnL
#     ax2_twin = ax2.twinx()
#     ax2_twin.plot(timestamps, pnl, color='blue', linewidth=1.5, linestyle='--', label='PnL')

#     # Format the secondary y-axis
#     ax2_twin.set_ylabel('PnL', color='blue')
#     ax2_twin.tick_params(axis='y', colors='blue')

#     # Add labels and legends
#     ax2.set_title('Equity Curve and PnL')
#     ax2.set_ylabel('Equity', color='green')
#     ax2.tick_params(axis='y', colors='green')
#     ax2.legend(loc='upper left')

#     ax2_twin.legend(loc='upper right')

#     # ---------------------- Formatting ----------------------
#     for ax in axs:
#         ax.grid(True)
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#         ax.xaxis.set_major_locator(mdates.AutoDateLocator())


#     # Show the plot
#     plt.show()

if __name__ == "__main__":
    # multiprocessing.set_start_method("fork")
    main()