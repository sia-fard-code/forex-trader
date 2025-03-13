import pandas as pd
import numpy as np
import logging
from TradingStrategy import TradingStrategy
import argparse
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """
    Parse command-line arguments.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Debug data structures in the Trading Strategy.')
    
    # Define total_points argument
    parser.add_argument(
        '--total_points',
        type=int,
        default=5000,
        help='Total number of data points (ticks) to process.'
    )
    
    # Define start_points argument
    parser.add_argument(
        '--start_points',
        type=int,
        default=50000,
        help='Number of initial data points to start processing from.'
    )
    
    return parser.parse_args()

def main():
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
    strategy = TradingStrategy(config)

    # Run the strategy
    logging.info("Starting debug run...")
    strategy.run_strategy(data)

    # Get results
    results = strategy.get_results()
    
    # Extract the data structures we want to debug
    positions_bid = results["positions_bid"]
    positions_ask = results["positions_ask"]
    refined_state = results["refined_state"]
    adjusted_position_size_bid = results["adjusted_position_size_bid"]
    adjusted_position_size_ask = results["adjusted_position_size_ask"]
    
    # Debug information
    logging.info(f"Data structures debug information:")
    logging.info(f"refined_state shape: {refined_state.shape if hasattr(refined_state, 'shape') else len(refined_state)}")
    logging.info(f"positions_bid shape: {positions_bid.shape if hasattr(positions_bid, 'shape') else len(positions_bid)}")
    logging.info(f"positions_ask shape: {positions_ask.shape if hasattr(positions_ask, 'shape') else len(positions_ask)}")
    logging.info(f"adjusted_position_size_bid shape: {adjusted_position_size_bid.shape if hasattr(adjusted_position_size_bid, 'shape') else len(adjusted_position_size_bid)}")
    logging.info(f"adjusted_position_size_ask shape: {adjusted_position_size_ask.shape if hasattr(adjusted_position_size_ask, 'shape') else len(adjusted_position_size_ask)}")
    
    # Count states
    num_bullish = np.sum(refined_state == 1)
    num_bearish = np.sum(refined_state == -1)
    num_neutral = np.sum(refined_state == 0)
    
    logging.info(f"Number of Bullish states: {num_bullish}")
    logging.info(f"Number of Bearish states: {num_bearish}")
    logging.info(f"Number of Neutral states: {num_neutral}")
    
    # Check for non-zero position sizes
    non_zero_bid = np.sum(positions_bid > 0)
    non_zero_ask = np.sum(positions_ask > 0)
    non_zero_adj_bid = np.sum(adjusted_position_size_bid > 0)
    non_zero_adj_ask = np.sum(adjusted_position_size_ask > 0)
    
    logging.info(f"Number of non-zero positions_bid: {non_zero_bid}")
    logging.info(f"Number of non-zero positions_ask: {non_zero_ask}")
    logging.info(f"Number of non-zero adjusted_position_size_bid: {non_zero_adj_bid}")
    logging.info(f"Number of non-zero adjusted_position_size_ask: {non_zero_adj_ask}")
    
    # Print some sample values
    logging.info("\nSample values:")
    sample_indices = [1000, 2000, 3000, 4000] if total_points >= 5000 else [int(total_points/5), int(2*total_points/5), int(3*total_points/5), int(4*total_points/5)]
    
    for idx in sample_indices:
        if idx < len(refined_state):
            logging.info(f"Index {idx}:")
            logging.info(f"  refined_state: {refined_state[idx]}")
            if idx < len(positions_bid):
                logging.info(f"  positions_bid: {positions_bid[idx]}")
            if idx < len(positions_ask):
                logging.info(f"  positions_ask: {positions_ask[idx]}")
            if idx < len(adjusted_position_size_bid):
                logging.info(f"  adjusted_position_size_bid: {adjusted_position_size_bid[idx]}")
            if idx < len(adjusted_position_size_ask):
                logging.info(f"  adjusted_position_size_ask: {adjusted_position_size_ask[idx]}")
    
    # Save the data to CSV for further analysis
    debug_df = pd.DataFrame({
        'refined_state': refined_state[:min(len(refined_state), total_points)],
        'positions_bid': positions_bid[:min(len(positions_bid), total_points)] if len(positions_bid) > 0 else [np.nan] * min(len(refined_state), total_points),
        'positions_ask': positions_ask[:min(len(positions_ask), total_points)] if len(positions_ask) > 0 else [np.nan] * min(len(refined_state), total_points),
        'adjusted_position_size_bid': adjusted_position_size_bid[:min(len(adjusted_position_size_bid), total_points)] if len(adjusted_position_size_bid) > 0 else [np.nan] * min(len(refined_state), total_points),
        'adjusted_position_size_ask': adjusted_position_size_ask[:min(len(adjusted_position_size_ask), total_points)] if len(adjusted_position_size_ask) > 0 else [np.nan] * min(len(refined_state), total_points)
    })
    
    debug_df.to_csv('debug_data_structures.csv', index=True)
    logging.info("Debug data saved to debug_data_structures.csv")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot refined state
    plt.subplot(3, 1, 1)
    plt.plot(refined_state[:min(len(refined_state), total_points)], label='Refined State')
    plt.title('Refined State (1: Bullish, -1: Bearish, 0: Neutral)')
    plt.grid(True)
    plt.legend()
    
    # Plot position sizes
    plt.subplot(3, 1, 2)
    if len(positions_bid) > 0:
        plt.plot(positions_bid[:min(len(positions_bid), total_points)], 'b-', label='Positions Bid')
    if len(positions_ask) > 0:
        plt.plot(positions_ask[:min(len(positions_ask), total_points)], 'r-', label='Positions Ask')
    plt.title('Position Sizes')
    plt.grid(True)
    plt.legend()
    
    # Plot adjusted position sizes
    plt.subplot(3, 1, 3)
    if len(adjusted_position_size_bid) > 0:
        plt.plot(adjusted_position_size_bid[:min(len(adjusted_position_size_bid), total_points)], 'b-', label='Adjusted Positions Bid')
    if len(adjusted_position_size_ask) > 0:
        plt.plot(adjusted_position_size_ask[:min(len(adjusted_position_size_ask), total_points)], 'r-', label='Adjusted Positions Ask')
    plt.title('Adjusted Position Sizes')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('debug_visualization.png')
    logging.info("Debug visualization saved to debug_visualization.png")

if __name__ == "__main__":
    main()
