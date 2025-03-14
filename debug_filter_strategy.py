import pandas as pd
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from ProfitabilityFilter import ProfitabilityFilter
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MockDataManager:
    """
    A simplified DataManager for debugging that works with pre-exported data.
    """
    def __init__(self, data_df):
        self.data = data_df
        self.config = {}
        self.trade_approved_values = []
        
    def get(self, key, window_size=None):
        if key in self.data.columns:
            if window_size is not None:
                return self.data[key].iloc[-window_size:].values
            return self.data[key].values
        return np.array([])
    
    def get_latest_value(self, key):
        if key in self.data.columns:
            return self.data[key].iloc[-1]
        if key == "pnl":
            # Mock PnL for testing
            return 0.0
        return 0.0
    
    def set(self, key, value):
        if key == "trade_approved":
            self.trade_approved_values.append(value)
        # Other sets are ignored for simplicity
        
    def get_config(self, param_name):
        return self.config.get(param_name, None)
    
    def set_config(self, param_name, value):
        self.config[param_name] = value

def parse_arguments():
    """
    Parse command-line arguments.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Debug filtering strategies with pre-exported data.')
    
    # Define input file
    parser.add_argument(
        '--input_file',
        type=str,
        default='training_data_export.csv',
        help='Input CSV file with pre-exported training data.'
    )
    
    # Define filter parameters
    parser.add_argument(
        '--profit_threshold',
        type=float,
        default=0.5,
        help='Minimum probability threshold for trade execution (0.0-1.0).'
    )
    
    parser.add_argument(
        '--min_history_required',
        type=int,
        default=20,
        help='Minimum number of trades required before filtering is applied.'
    )
    
    parser.add_argument(
        '--volatility_weight',
        type=float,
        default=0.5,
        help='Weight given to volatility in the probability calculation (0.0-1.0).'
    )
    
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='debug_results',
        help='Prefix for output files.'
    )
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    input_file = args.input_file
    profit_threshold = args.profit_threshold
    min_history_required = args.min_history_required
    volatility_weight = args.volatility_weight
    output_prefix = args.output_prefix
    
    logging.info(f"Starting debug process with pre-exported data from {input_file}")
    logging.info(f"Filter settings: profit_threshold={profit_threshold}, min_history_required={min_history_required}, volatility_weight={volatility_weight}")
    
    try:
        data = pd.read_csv(input_file)
        logging.info(f"Loaded {len(data)} data points from {input_file}")
    except FileNotFoundError:
        logging.error(f"File {input_file} not found. Please run export_training_data.py first.")
        return
    except Exception as e:
        logging.error(f"Error reading {input_file}: {e}")
        return
    
    # Create mock data manager
    data_manager = MockDataManager(data)
    
    # Initialize filter configuration
    filter_config = {
        "filter_lookback_window": 500,
        "min_history_required": min_history_required,
        "profit_threshold": profit_threshold,
        "volatility_weight": volatility_weight,
        "state_weight": 1.0 - volatility_weight,
    }
    
    # Initialize the profitability filter
    profitability_filter = ProfitabilityFilter(data_manager, filter_config)
    
    # Process each data point
    logging.info("Processing data points...")
    trade_decisions = []
    probabilities = []
    
    for i in range(len(data)):
        state = data['refined_state'].iloc[i]
        vol_bid = data['forecasted_vol_bid'].iloc[i]
        vol_ask = data['forecasted_vol_ask'].iloc[i]
        
        # Simulate PnL updates for the filter's history
        if i > 0:
            # Create a simple mock PnL based on previous state and current price movement
            prev_state = data['refined_state'].iloc[i-1]
            price_change = data['bid'].iloc[i] - data['bid'].iloc[i-1]
            mock_pnl = 0.0
            
            if prev_state == 1 and price_change > 0:  # Bullish state and price went up
                mock_pnl = abs(price_change) * 10000  # Arbitrary scaling
            elif prev_state == -1 and price_change < 0:  # Bearish state and price went down
                mock_pnl = abs(price_change) * 10000  # Arbitrary scaling
            elif prev_state != 0:  # Wrong prediction
                mock_pnl = -abs(price_change) * 10000  # Arbitrary scaling
                
            if mock_pnl != 0:
                profitability_filter.update_performance_history(
                    prev_state, 
                    data['forecasted_vol_bid'].iloc[i-1],
                    data['forecasted_vol_ask'].iloc[i-1],
                    mock_pnl
                )
        
        # Get the filter's decision
        should_trade = profitability_filter.should_trade(state, vol_bid, vol_ask)
        probability = profitability_filter.calculate_combined_probability(state, vol_bid, vol_ask)
        
        trade_decisions.append(should_trade)
        probabilities.append(probability)
        
        # Log progress
        if i % 1000 == 0:
            logging.info(f"Processed {i}/{len(data)} data points")
    
    # Add results to the dataframe
    data['trade_approved'] = trade_decisions
    data['trade_probability'] = probabilities
    
    # Save results
    output_file = f"{output_prefix}_threshold_{profit_threshold}_history_{min_history_required}_volweight_{volatility_weight}.csv"
    data.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    
    # Get filter statistics
    filter_stats = profitability_filter.get_statistics()
    
    # Print summary
    logging.info("Filter Statistics:")
    logging.info(f"  Total trades evaluated: {filter_stats['total_trades_evaluated']}")
    logging.info(f"  Trades approved: {filter_stats['trades_approved']}")
    logging.info(f"  Trades rejected: {filter_stats['trades_rejected']}")
    logging.info(f"  Approval rate: {filter_stats['approval_rate']:.2f}")
    
    # Count approved trades by state
    approved_bullish = np.sum((data['refined_state'] == 1) & (data['trade_approved'] == 1))
    approved_bearish = np.sum((data['refined_state'] == -1) & (data['trade_approved'] == 1))
    approved_neutral = np.sum((data['refined_state'] == 0) & (data['trade_approved'] == 1))
    
    logging.info(f"Approved Bullish trades: {approved_bullish}")
    logging.info(f"Approved Bearish trades: {approved_bearish}")
    logging.info(f"Approved Neutral trades: {approved_neutral}")
    
    # Create visualization
    plt.figure(figsize=(14, 12))
    
    # Plot states and approved trades
    plt.subplot(4, 1, 1)
    plt.plot(data['refined_state'], 'b-', alpha=0.5, label='Market State')
    approved_indices = np.where(data['trade_approved'] == 1)[0]
    plt.scatter(approved_indices, data['refined_state'].iloc[approved_indices], 
                color='green', marker='o', label='Approved Trades')
    plt.title('Market State and Approved Trades')
    plt.grid(True)
    plt.legend()
    
    # Plot volatility
    plt.subplot(4, 1, 2)
    plt.plot(data['forecasted_vol_bid'], 'b-', alpha=0.5, label='Volatility Bid')
    plt.plot(data['forecasted_vol_ask'], 'r-', alpha=0.5, label='Volatility Ask')
    plt.scatter(approved_indices, data['forecasted_vol_bid'].iloc[approved_indices], 
                color='green', marker='o', label='Approved Trades')
    plt.title('Forecasted Volatility and Approved Trades')
    plt.grid(True)
    plt.legend()
    
    # Plot trade probabilities
    plt.subplot(4, 1, 3)
    plt.plot(data['trade_probability'], 'b-', label='Trade Probability')
    plt.axhline(y=profit_threshold, color='r', linestyle='--', label=f'Threshold ({profit_threshold})')
    plt.title('Trade Probability')
    plt.grid(True)
    plt.legend()
    
    # Plot price with approved trades
    plt.subplot(4, 1, 4)
    plt.plot(data['bid'], 'k-', alpha=0.7, label='Bid Price')
    
    # Mark bullish approved trades
    bullish_indices = np.where((data['refined_state'] == 1) & (data['trade_approved'] == 1))[0]
    plt.scatter(bullish_indices, data['bid'].iloc[bullish_indices], 
                color='green', marker='^', label='Approved Bullish')
    
    # Mark bearish approved trades
    bearish_indices = np.where((data['refined_state'] == -1) & (data['trade_approved'] == 1))[0]
    plt.scatter(bearish_indices, data['bid'].iloc[bearish_indices], 
                color='red', marker='v', label='Approved Bearish')
    
    plt.title('Price Chart with Approved Trades')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = f"{output_prefix}_viz_threshold_{profit_threshold}_history_{min_history_required}_volweight_{volatility_weight}.png"
    plt.savefig(viz_file, dpi=300)
    logging.info(f"Visualization saved to {viz_file}")
    
    logging.info("Debug process completed successfully.")

if __name__ == "__main__":
    main()
