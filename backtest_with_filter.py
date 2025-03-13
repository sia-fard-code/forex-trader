import pandas as pd
import numpy as np
import logging
from TradingStrategyWithFilter import TradingStrategyWithFilter  # Import the filtered strategy
import argparse
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # Define filter parameters
    parser.add_argument(
        '--profit_threshold',
        type=float,
        default=0.6,
        help='Minimum probability threshold for trade execution (0.0-1.0).'
    )
    
    parser.add_argument(
        '--min_history_required',
        type=int,
        default=100,
        help='Minimum number of trades required before filtering is applied.'
    )
    
    parser.add_argument(
        '--volatility_weight',
        type=float,
        default=0.5,
        help='Weight given to volatility in the probability calculation (0.0-1.0).'
    )
    
    return parser.parse_args()

def main():
    # ---------------------- Step 1: Load and Preprocess Data ----------------------
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
        
        # Profitability filter configuration
        "filter_lookback_window": 500,
        "min_history_required": args.min_history_required,
        "profit_threshold": args.profit_threshold,
        "volatility_weight": args.volatility_weight,
        "state_weight": 1.0 - args.volatility_weight,
        
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

    # Initialize the trading strategy with filter
    strategy = TradingStrategyWithFilter(config)

    # ---------------------- Step 3: Run the Strategy ----------------------
    logging.info("Starting backtest with profitability filtering...")
    logging.info(f"Filter settings: Profit threshold={args.profit_threshold}, "
                f"Min history={args.min_history_required}, "
                f"Volatility weight={args.volatility_weight}")

    strategy.run_strategy(data)

    # ---------------------- Step 4: Analyze Results ----------------------
    results = strategy.get_results()
    equity_curve = results["equity_curve"]
    balance = results["balance"]
    positions_bid = results["adjusted_position_size_bid"]
    positions_ask = results["adjusted_position_size_ask"]
    refined_state = results["refined_state"]
    positions = results["positions"]
    pnl = results["pnl"]
    filter_stats = results["filter_statistics"]

    # Count and print the states
    num_bullish = np.sum(refined_state == 1)
    num_bearish = np.sum(refined_state == -1)
    num_neutral = np.sum(refined_state == 0)

    logging.info(f"Number of Bullish states: {num_bullish}")
    logging.info(f"Number of Bearish states: {num_bearish}")
    logging.info(f"Number of Neutral states: {num_neutral}")
    
    # Print filter statistics
    logging.info(f"Filter Statistics:")
    logging.info(f"  Total trades evaluated: {filter_stats['total_trades_evaluated']}")
    logging.info(f"  Trades approved: {filter_stats['trades_approved']}")
    logging.info(f"  Trades rejected: {filter_stats['trades_rejected']}")
    logging.info(f"  Approval rate: {filter_stats['approval_rate']:.2f}")

    # ---------------------- Step 5: Plot Results (Optional) ----------------------
    plot_results(data, equity_curve, positions_bid, positions_ask, refined_state)

def plot_results(data, equity_curve, positions_bid, positions_ask, refined_state):
    """
    Plot the results of the backtest.
    """
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})

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
    ax1.set_title('Forex Trading with Profitability Filtering', fontsize=16)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')

    # Format x-axis with dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)

    # Plot scatter points for bid and ask positions
    for i in range(total_points):
        if marker_size_bid[i] > 0:
            ax1.scatter(timestamps[i], bid[i], s=marker_size_bid[i], color=state_colors_bid[i], alpha=0.7)
        if marker_size_ask[i] > 0:
            ax1.scatter(timestamps[i], ask[i], s=marker_size_ask[i], color=state_colors_ask[i], alpha=0.7)

    # Plot equity curve on the second subplot
    ax2.plot(timestamps, equity_curve, color='green', linewidth=1.5, label='Equity')
    ax2.set_ylabel('Equity', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left')

    # Add custom legend for state colors
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0, 0, 1), markersize=10, label='Bullish'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(1, 0, 0), markersize=10, label='Bearish'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.5, 0.5, 0.5), markersize=10, label='Neutral')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('backtest_results_with_filter.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
