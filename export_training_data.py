import pandas as pd
import numpy as np
import logging
from TradingStrategy import TradingStrategy
import argparse
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """
    Parse command-line arguments.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Export training data for faster debugging.')
    
    # Define total_points argument
    parser.add_argument(
        '--total_points',
        type=int,
        default=10000,
        help='Total number of data points (ticks) to process.'
    )
    
    # Define start_points argument
    parser.add_argument(
        '--start_points',
        type=int,
        default=50000,
        help='Number of initial data points to start processing from.'
    )
    
    # Define output file
    parser.add_argument(
        '--output_file',
        type=str,
        default='training_data_export.csv',
        help='Output CSV file to save the exported data.'
    )
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    total_points = args.total_points
    start_points = args.start_points
    output_file = args.output_file
    filename = 'EURUSD_mt5_ticks-m.csv'

    logging.info(f"Starting data export process with {total_points} points starting from index {start_points}")
    
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

    # Initialize Parameters
    config = {
        "buffer_size": total_points,
        "ema_window": 20,
        "training_window_size": 3000,
        "rolling_window_size": 2000,
        "forecast_steps": 2,
        "num_simulations": 2,
        "max_window_size": 10,
        "k": 8e-7,
        "margin_requirement": 0.01,
        "max_spread": 2e-4,
        "direction_multiplier": -1,
        "pip_value": 0.0001,
        'adaptive_min_sigma_bid': np.nan,
        'adaptive_min_sigma_ask': np.nan,
        'is_trained': False,
        'forecasted_ask_change': 0,
        'forecasted_bid_change': 0,
        'max_position_size': .5,
        "drawdown_threshold": 0.10,
        "broker_config": {
            "initial_capital": 10000,
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

    # Initialize the trading strategy
    logging.info("Initializing trading strategy...")
    strategy = TradingStrategy(config)

    # Run the strategy to generate the data
    logging.info("Running strategy to generate training data...")
    strategy.run_strategy(data)

    # Get results
    logging.info("Extracting results...")
    results = strategy.get_results()
    
    # Extract the data structures we want to export
    refined_state = results["refined_state"]
    forecasted_vol_bid = strategy.data_manager.get("forecasted_vol_bid")
    forecasted_vol_ask = strategy.data_manager.get("forecasted_vol_ask")
    positions_bid = results["positions_bid"]
    positions_ask = results["positions_ask"]
    adjusted_position_size_bid = results["adjusted_position_size_bid"]
    adjusted_position_size_ask = results["adjusted_position_size_ask"]
    
    # Create a DataFrame with the exported data
    export_df = pd.DataFrame({
        'timestamp': data['timestamp'],
        'bid': data['bid'],
        'ask': data['ask'],
        'refined_state': refined_state,
        'forecasted_vol_bid': forecasted_vol_bid,
        'forecasted_vol_ask': forecasted_vol_ask,
        'positions_bid': positions_bid,
        'positions_ask': positions_ask,
        'adjusted_position_size_bid': adjusted_position_size_bid,
        'adjusted_position_size_ask': adjusted_position_size_ask
    })
    
    # Save to CSV
    logging.info(f"Saving exported data to {output_file}...")
    export_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    num_bullish = np.sum(refined_state == 1)
    num_bearish = np.sum(refined_state == -1)
    num_neutral = np.sum(refined_state == 0)
    
    logging.info(f"Data export complete. Summary statistics:")
    logging.info(f"Total data points: {len(export_df)}")
    logging.info(f"Number of Bullish states: {num_bullish} ({num_bullish/len(export_df)*100:.2f}%)")
    logging.info(f"Number of Bearish states: {num_bearish} ({num_bearish/len(export_df)*100:.2f}%)")
    logging.info(f"Number of Neutral states: {num_neutral} ({num_neutral/len(export_df)*100:.2f}%)")
    logging.info(f"Average volatility (bid): {np.nanmean(forecasted_vol_bid):.6f}")
    logging.info(f"Average volatility (ask): {np.nanmean(forecasted_vol_ask):.6f}")
    
    # Create a simple visualization
    plt.figure(figsize=(12, 10))
    
    # Plot states
    plt.subplot(3, 1, 1)
    plt.plot(refined_state, label='Market State')
    plt.title('Market State (1: Bullish, -1: Bearish, 0: Neutral)')
    plt.grid(True)
    plt.legend()
    
    # Plot volatility
    plt.subplot(3, 1, 2)
    plt.plot(forecasted_vol_bid, 'b-', label='Volatility Bid')
    plt.plot(forecasted_vol_ask, 'r-', label='Volatility Ask')
    plt.title('Forecasted Volatility')
    plt.grid(True)
    plt.legend()
    
    # Plot position sizes
    plt.subplot(3, 1, 3)
    plt.plot(adjusted_position_size_bid, 'b-', label='Position Size Bid')
    plt.plot(adjusted_position_size_ask, 'r-', label='Position Size Ask')
    plt.title('Adjusted Position Sizes')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the visualization
    viz_filename = os.path.splitext(output_file)[0] + '_visualization.png'
    plt.savefig(viz_filename)
    logging.info(f"Visualization saved to {viz_filename}")
    
    logging.info(f"Data export process completed successfully.")

if __name__ == "__main__":
    main()
