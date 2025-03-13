import numpy as np
import pandas as pd
import logging

class ProfitabilityFilter:
    """
    A class to filter trading opportunities based on probability of profitability.
    Uses historical performance data to calculate probability scores for different
    market conditions based on state and volatility metrics.
    """
    def __init__(self, data_manager, config=None):
        """
        Initialize the ProfitabilityFilter class.
        
        Args:
            data_manager: Instance of DataManager.
            config: Dictionary with configuration parameters.
        """
        self.data_manager = data_manager
        self.config = config or {}
        
        # Default configuration parameters
        self.lookback_window = self.config.get("filter_lookback_window", 500)
        self.min_history_required = self.config.get("min_history_required", 100)
        self.profit_threshold = self.config.get("profit_threshold", 0.6)  # Minimum probability of profit
        self.volatility_weight = self.config.get("volatility_weight", 0.5)  # Weight for volatility in scoring
        self.state_weight = self.config.get("state_weight", 0.5)  # Weight for state in scoring
        
        # Performance tracking
        self.state_performance = {1: [], -1: [], 0: []}  # Track performance by state
        self.volatility_bins = 10  # Number of bins for volatility discretization
        self.volatility_performance = {}  # Track performance by volatility bin
        
        # Initialize performance history
        self.performance_history = []
        
        logging.info("ProfitabilityFilter initialized with profit threshold: {:.2f}".format(
            self.profit_threshold))

    def update_performance_history(self, state, volatility_bid, volatility_ask, pnl):
        """
        Update the performance history with a new trade result.
        
        Args:
            state: Market state (1 for Bullish, -1 for Bearish, 0 for Neutral).
            volatility_bid: Forecasted volatility for bid.
            volatility_ask: Forecasted volatility for ask.
            pnl: Profit and loss from the trade.
        """
        # Record state performance
        self.state_performance[state].append(pnl > 0)
        
        # Calculate average volatility
        avg_volatility = (volatility_bid + volatility_ask) / 2
        
        # Discretize volatility into bins
        volatility_bin = min(int(avg_volatility * self.volatility_bins), self.volatility_bins - 1)
        
        # Initialize bin if not exists
        if volatility_bin not in self.volatility_performance:
            self.volatility_performance[volatility_bin] = []
        
        # Record volatility performance
        self.volatility_performance[volatility_bin].append(pnl > 0)
        
        # Add to overall performance history
        self.performance_history.append({
            'state': state,
            'volatility_bid': volatility_bid,
            'volatility_ask': volatility_ask,
            'avg_volatility': avg_volatility,
            'volatility_bin': volatility_bin,
            'pnl': pnl,
            'profitable': pnl > 0
        })
        
        # Trim history if needed
        if len(self.performance_history) > self.lookback_window:
            oldest = self.performance_history.pop(0)
            self.state_performance[oldest['state']].pop(0)
            self.volatility_performance[oldest['volatility_bin']].pop(0)

    def calculate_state_probability(self, state):
        """
        Calculate the probability of profit based on historical performance for a given state.
        
        Args:
            state: Market state (1 for Bullish, -1 for Bearish, 0 for Neutral).
            
        Returns:
            Probability of profit for the given state.
        """
        if not self.state_performance[state]:
            return 0.5  # Default to neutral if no history
        
        return sum(self.state_performance[state]) / len(self.state_performance[state])

    def calculate_volatility_probability(self, volatility_bid, volatility_ask):
        """
        Calculate the probability of profit based on historical performance for a given volatility level.
        
        Args:
            volatility_bid: Forecasted volatility for bid.
            volatility_ask: Forecasted volatility for ask.
            
        Returns:
            Probability of profit for the given volatility level.
        """
        avg_volatility = (volatility_bid + volatility_ask) / 2
        volatility_bin = min(int(avg_volatility * self.volatility_bins), self.volatility_bins - 1)
        
        if volatility_bin not in self.volatility_performance or not self.volatility_performance[volatility_bin]:
            return 0.5  # Default to neutral if no history
        
        return sum(self.volatility_performance[volatility_bin]) / len(self.volatility_performance[volatility_bin])

    def calculate_combined_probability(self, state, volatility_bid, volatility_ask):
        """
        Calculate the combined probability of profit based on state and volatility.
        
        Args:
            state: Market state (1 for Bullish, -1 for Bearish, 0 for Neutral).
            volatility_bid: Forecasted volatility for bid.
            volatility_ask: Forecasted volatility for ask.
            
        Returns:
            Combined probability of profit.
        """
        state_prob = self.calculate_state_probability(state)
        volatility_prob = self.calculate_volatility_probability(volatility_bid, volatility_ask)
        
        # Weighted average of probabilities
        combined_prob = (self.state_weight * state_prob + 
                         self.volatility_weight * volatility_prob)
        
        return combined_prob

    def should_trade(self, state, volatility_bid, volatility_ask):
        """
        Determine if a trade should be executed based on the probability of profit.
        
        Args:
            state: Market state (1 for Bullish, -1 for Bearish, 0 for Neutral).
            volatility_bid: Forecasted volatility for bid.
            volatility_ask: Forecasted volatility for ask.
            
        Returns:
            Boolean indicating whether to execute the trade.
        """
        # Check if we have enough history
        if len(self.performance_history) < self.min_history_required:
            logging.debug("Not enough history to make filtering decision. Using default state logic.")
            return True  # Default to original strategy behavior
        
        # Calculate probability of profit
        probability = self.calculate_combined_probability(state, volatility_bid, volatility_ask)
        
        # Log the decision
        logging.debug(f"Trade probability: {probability:.4f}, Threshold: {self.profit_threshold:.4f}")
        
        # Return decision
        return probability >= self.profit_threshold

    def get_statistics(self):
        """
        Get statistics about the filter's performance.
        
        Returns:
            Dictionary with statistics.
        """
        if not self.performance_history:
            return {
                'total_trades_evaluated': 0,
                'trades_approved': 0,
                'trades_rejected': 0,
                'approval_rate': 0.0,
                'state_probabilities': {1: 0.5, -1: 0.5, 0: 0.5},
                'volatility_probabilities': {}
            }
        
        # Calculate statistics
        total_trades = len(self.performance_history)
        approved_trades = sum(1 for trade in self.performance_history 
                             if self.calculate_combined_probability(
                                 trade['state'], trade['volatility_bid'], trade['volatility_ask']
                             ) >= self.profit_threshold)
        
        state_probs = {
            state: self.calculate_state_probability(state)
            for state in [1, -1, 0]
        }
        
        volatility_probs = {
            bin_idx: sum(results) / len(results) if results else 0.5
            for bin_idx, results in self.volatility_performance.items()
        }
        
        return {
            'total_trades_evaluated': total_trades,
            'trades_approved': approved_trades,
            'trades_rejected': total_trades - approved_trades,
            'approval_rate': approved_trades / total_trades if total_trades > 0 else 0.0,
            'state_probabilities': state_probs,
            'volatility_probabilities': volatility_probs
        }

    def process(self, t, state, volatility_bid, volatility_ask):
        """
        Process the current market conditions and determine if a trade should be executed.
        
        Args:
            t: Current tick index.
            state: Market state (1 for Bullish, -1 for Bearish, 0 for Neutral).
            volatility_bid: Forecasted volatility for bid.
            volatility_ask: Forecasted volatility for ask.
            
        Returns:
            Boolean indicating whether to execute the trade.
        """
        # Get the latest PnL if available to update history
        try:
            latest_pnl = self.data_manager.get_latest_value("pnl")
            if not np.isnan(latest_pnl) and latest_pnl != 0:
                # Only update with actual trade results
                prev_state = self.data_manager.get("refined_state", 2)[-2] if t > 0 else 0
                prev_vol_bid = self.data_manager.get("forecasted_vol_bid", 2)[-2] if t > 0 else 0
                prev_vol_ask = self.data_manager.get("forecasted_vol_ask", 2)[-2] if t > 0 else 0
                
                self.update_performance_history(prev_state, prev_vol_bid, prev_vol_ask, latest_pnl)
        except (KeyError, IndexError) as e:
            logging.debug(f"Could not update performance history: {e}")
        
        # Determine if we should trade
        decision = self.should_trade(state, volatility_bid, volatility_ask)
        
        # Store the decision in DataManager
        self.data_manager.set("trade_approved", 1 if decision else 0)
        
        return decision
