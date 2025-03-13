import numpy as np
import logging
import pandas as pd
import threading


class PositionSizing:
    def __init__(self, data_manager, price_type):
        """
        Initialize the PositionSizing class for a specific price type.
        :param data_manager: Instance of DataManager.
        :param price_type: Either 'bid' or 'ask'.
        """
        self.data_manager = data_manager
        self.config = self.data_manager.config
        self.price_type = price_type  # Either 'bid' or 'ask'
        self.lock = threading.Lock()
        self.jj = 0

    def process(self, t, refined_state):
        """
        Orchestrate the position sizing workflow.
        :param t: Current tick index.
        :param refined_state: Refined market state (1: Bullish, -1: Bearish, 0: Neutral).
        """
        # Calculate position sizes
        position_size = self.calculate_position_sizes(refined_state)

        # Update PnL and equity
        pnl = self.update_equity_and_pnl(position_size)

        # Update margin
        self.update_margin(position_size)

        # Save results to DataManager
        self.data_manager.set(f"position_size_{self.price_type}", position_size)
        self.data_manager.set("pnl", pnl)

    def calculate_position_sizes(self, refined_state):
        """
        Calculate position sizes using Kelly Fraction and volatility thresholds.
        :param refined_state: Refined market state.
        :return: Position size.
        """
        with self.lock:
            if refined_state == 0:
                return 0.0  # Neutral state, no position

            # Retrieve necessary data
            forecasted_vol = self.data_manager.get_latest_value(f"forecasted_vol_{self.price_type}")
            arithmetic_return = self.data_manager.get_latest_value(f"arithmetic_return_{self.price_type}")
            log_return = self.data_manager.get(f"log_return_{self.price_type}",100)
            ema_price = self.data_manager.get(f"ema_{self.price_type}",100)
            # Adaptive sigma and k values (can be derived dynamically or predefined)
            adaptive_sigma,adaptive_k = self._calculate_adaptive_data(self.data_manager.get(f"forecasted_vol_{self.price_type}",100),log_return,ema_price)
            # adaptive_k = self.data_manager.get_config("k")
            # adaptive_sigma = 7e-5
            # adaptive_k = 5e-4

            # Volatility threshold
            volatility = max(forecasted_vol, adaptive_sigma)

            # Calculate Kelly Fraction
            if self.price_type == "bid" and refined_state == 1 and arithmetic_return > 0:
                kelly_fraction = arithmetic_return / (volatility ** 2)
            elif self.price_type == "ask" and refined_state == -1 and arithmetic_return < 0:
                kelly_fraction = abs(arithmetic_return) / (volatility ** 2)
            else:
                kelly_fraction = 0.0

            # Scale Kelly Fraction
            position_size = kelly_fraction * adaptive_k

            # Clamp position size
            # recent_data1 = self.data_manager.get(f"adjusted_position_size_{self.price_type}", window_size=100)
            # recent_data = recent_data1[recent_data1 != 0]
            recent_bid = self.data_manager.get(f"adjusted_position_size_bid", window_size=300)
            recent_bid = recent_bid[recent_bid != 0]
            recent_ask = self.data_manager.get(f"adjusted_position_size_ask", window_size=300)
            recent_ask = recent_ask[recent_ask != 0]
            f = 1
            percentile = 30
            if self.price_type == 'bid' and len(recent_bid)>50 and len(recent_ask)>50:
                percentile_bid = np.nanpercentile(recent_bid, percentile)
                # std_dev = np.nanstd(recent_data)
                mean_bid = np.nanmean(recent_bid)
                percentile_ask = np.nanpercentile(recent_ask, percentile)
                # std_dev = np.nanstd(recent_data)
                mean_ask = np.nanmean(recent_ask)

                # Use weighted combination of metrics
                # max_position_size = min(10 * percentile + 0 * mean + 0 * std_dev,1000)  # Weighted cap
                max_position_size_ask = f * percentile_ask / max(mean_ask,1e-16)
                max_position_size_bid = percentile_bid / max(mean_bid,1e-16)
                max_position_size = max_position_size_bid / max_position_size_ask
                # logging.info(f"++++++++++max_position_size={max_position_size}: percentile={percentile:.6e}, std_dev={std_dev:.6e}, mean={mean}")
            elif self.price_type == 'ask' and len(recent_bid)>50 and len(recent_ask)>50:
                percentile_bid = np.nanpercentile(recent_bid, percentile)
                # std_dev = np.nanstd(recent_data)
                mean_bid = np.nanmean(recent_bid)
                percentile_ask = np.nanpercentile(recent_ask, percentile)
                # std_dev = np.nanstd(recent_data)
                mean_ask = np.nanmean(recent_ask)

                # Use weighted combination of metrics
                # max_position_size = min(10 * percentile + 0 * mean + 0 * std_dev,1000)  # Weighted cap
                max_position_size_ask = f * percentile_ask / max(mean_ask,1e-16)
                max_position_size_bid = percentile_bid / max(mean_bid,1e-16)
                max_position_size = max_position_size_ask / max_position_size_bid
                # logging.info(f"++++++++++max_position_size={max_position_size}: percentile={percentile:.6e}, std_dev={std_dev:.6e}, mean={mean}")
            else:
                max_position_size = 1  # Default fallback
                self.jj=self.jj+1

            # max_position_size = min(max_position_size,100)  # Default fallback
            # max_position_size = .5  # Default fallback
            position_size = min(position_size, max_position_size)
            return position_size

    def update_equity_and_pnl(self, position_size):
        """
        Update equity and calculate PnL based on the current position.
        :param position_size: Position size.
        :return: PnL for the current tick.
        """
        equity = self.data_manager.get_latest_value("equity")
        arithmetic_return = self.data_manager.get_latest_value(f"arithmetic_return_{self.price_type}")

        # Calculate PnL
        pnl = position_size * arithmetic_return * equity

        # Update equity
        new_equity = equity + pnl
        self.data_manager.set("equity", new_equity)

        return pnl

    def update_margin(self, position_size):
        """
        Update margin used and free margin.
        :param position_size: Position size.
        """
        equity = self.data_manager.get_latest_value("equity")
        margin_requirement = self.data_manager.get_config("margin_requirement")

        # Calculate required margin
        required_margin = position_size * equity * margin_requirement

        # Update margin used and free margin
        margin_used = self.data_manager.get_latest_value("margin_used")
        margin_used += required_margin
        free_margin = equity - margin_used

        self.data_manager.set("margin_used", margin_used)
        self.data_manager.set("free_margin", free_margin)

    # def _calculate_adaptive_sigma(self,forecasted_vol,log_return):
    #     """
    #     Placeholder for adaptive sigma calculation.
    #     Integrate this function dynamically if adaptive sigma is not pre-computed.
    #     """
    #     long_term_vol = max(self.data_manager.get_latest_value(f"long_term_vol_{self.price_type}")* 0.1, 1e-8)
    #     log_return_std = max(np.nanstd(log_return) * 0.1, 1e-8)
    #     forecasted_vol_perc = max(np.nanpercentile(forecasted_vol, 10), 1e-8)
    #     adaptive_k = max(long_term_vol, log_return_std, forecasted_vol_perc)
    #     adaptive_min_sigma = min(long_term_vol, log_return_std, forecasted_vol_perc)
    #     return adaptive_min_sigma, adaptive_k

    def _calculate_adaptive_data(self,  forecasted_vol, log_return, price):
        """
        Calculate adaptive sigma and Kelly fraction based on multiple factors.
        :param t: Current tick index.
        :param forecasted_vol: Forecasted volatility.
        :param log_return: Array of log returns.
        :param prices: Array of price data.
        :return: adaptive_min_sigma, adaptive_k
        """
        # Existing Factors
        long_term_vol = max(self.data_manager.get_latest_value(f"long_term_vol_{self.price_type}") * 0.1, 1e-8)
        log_return_std = np.nanstd(log_return)
        # log_return_std = max(np.nanstd(log_return) * 0.1, 1e-8)
        forecasted_vol_perc = max(np.nanpercentile(forecasted_vol, 30), 1e-8)

        # Additional Factors
        ewma_vol = self.calculate_ewma_volatility(log_return, span=100)
        var = self.calculate_var(log_return, confidence_level=90)
        cvar = self.calculate_cvar(log_return, confidence_level=90)
        rsi = self.calculate_rsi(price, window=24) / 100
        # rsi = self.calculate_rsi_return(log_return, window=24) / 100
        # portfolio_vol = self.calculate_portfolio_volatility(self.data_manager.get("portfolio_returns", 1000))
        # max_drawdown = self.calculate_max_drawdown(self.data_manager.get("equity", 1000))
        # news_sentiment = self.data_manager.get_latest_value("news_sentiment") or 0.0

        # Example of integrating EWMA and VaR
        # ewma_vol_scaled = max(ewma_vol * 0.1, 1e-8)
        # var_scaled = max(var * 0.1, 1e-8)

        # Combine all factors
        # Assign weights
        weights = {
            "long_term_vol": .0,
            "log_return_std": .0,
            "forecasted_vol_perc": 1.0,
            "ewma_vol_scaled": .00,
            "var_scaled": .0,
            "cvar_scaled": 0.0,
            "rsi_scaled": 1  # Example weight for RSI
        }

        # Scale RSI to influence k and min_sigma
        if rsi > 70:
            rsi_scaled = 0.5  # Reduce position sizing
        elif rsi < 30:
            rsi_scaled = 1.15  # Increase position sizing
        else:
            rsi_scaled = 1.0  # No adjustment

        # Combine factors with weights
        adaptive_k = (
            weights["long_term_vol"] * long_term_vol +
            weights["log_return_std"] * log_return_std +
            weights["forecasted_vol_perc"] * forecasted_vol_perc +
            weights["ewma_vol_scaled"] * ewma_vol +
            weights["var_scaled"] * var +
            weights["cvar_scaled"] * cvar +
            weights["rsi_scaled"] * rsi
        )

        adaptive_min_sigma = (
            weights["long_term_vol"] * long_term_vol +
            weights["log_return_std"] * log_return_std +
            weights["forecasted_vol_perc"] * forecasted_vol_perc +
            weights["ewma_vol_scaled"] * ewma_vol +
            weights["var_scaled"] * var +
            weights["cvar_scaled"] * cvar +
            weights["rsi_scaled"] * rsi
        )

        # Optionally, apply scaling or normalization
        adaptive_k = max(adaptive_k, 1e-4)
        adaptive_min_sigma = min(adaptive_min_sigma, 1e-4)

        # Optionally, incorporate performance-based adjustments
        # sharpe_ratio = self.data_manager.get_latest_value("sharpe_ratio") or 1.0
        # adaptive_k = self.adjust_parameters_based_on_performance(sharpe_ratio)

        logging.debug(f"t=: Adaptive k={adaptive_k:.6e}, Adaptive min_sigma={adaptive_min_sigma:.6e}")
        return adaptive_min_sigma, adaptive_k

    # Additional Helper Functions
    def calculate_ewma_volatility(self, returns, span=100):
        """
        Calculate EWMA volatility.
        :param returns: Array of log returns.
        :param span: Span for EWMA.
        :return: EWMA volatility.
        """
        if len(returns) == 0:
            return 1e-8
        ewma_variance = pd.Series(returns).ewm(span=span).var().iloc[-1]
        return np.sqrt(ewma_variance)

    def calculate_var(self, returns, confidence_level=95):
        """
        Calculate Value at Risk (VaR).
        :param returns: Array of log returns.
        :param confidence_level: Confidence level for VaR.
        :return: VaR value.
        """
        if len(returns) == 0:
            return 1e-8
        var = np.nanpercentile(returns, 100 - confidence_level)
        return abs(var)

    def calculate_cvar(self, returns, confidence_level=95):
        """
        Calculate Conditional Value at Risk (CVaR).
        :param returns: Array of log returns.
        :param confidence_level: Confidence level for CVaR.
        :return: CVaR value.
        """
        if len(returns) == 0:
            return 1e-8
        var = np.nanpercentile(returns, 100 - confidence_level)
        cvar = returns[returns <= var].mean()
        return abs(cvar)

    def calculate_rsi_return(self, returns, window=14):
        """
        Calculate Relative Strength Index (RSI).
        :param returns: Array of log returns.
        :param window: Window size for RSI.
        :return: RSI value.
        """
        if len(returns) < window + 1:
            return 50  # Neutral RSI
        deltas = np.diff(returns)
        up = deltas[deltas > 0]
        down = -deltas[deltas < 0]
        if len(up) == 0:
            rs = 0
        else:
            avg_gain = np.nanmean(up[-window:])
            avg_loss = np.nanmean(down[-window:]) if len(down[-window:]) > 0 else 0
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 0
        return rsi
    def calculate_rsi(self, prices, window=24):
        """
        Calculate Relative Strength Index (RSI) using price data with Pandas.
        :param prices: Array-like of price data (e.g., closing prices).
        :param window: Window size for RSI calculation.
        :return: RSI value.
        """
        if len(prices) < window + 1:
            return 50  # Neutral RSI

        # Convert prices to a Pandas Series
        prices_series = pd.Series(prices)

        # Calculate price changes
        deltas = prices_series.diff()

        # Separate gains and losses
        gain = deltas.clip(lower=0)
        loss = -deltas.clip(upper=0)

        # Calculate the Exponential Weighted Moving Average
        avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

        # Calculate Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Return the latest RSI value, handling NaN
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def calculate_portfolio_volatility(self, portfolio_returns, window_size=1000):
        """
        Calculate portfolio volatility.
        :param portfolio_returns: Array of portfolio returns.
        :param window_size: Rolling window size.
        :return: Portfolio volatility.
        """
        data = portfolio_returns[-window_size:]
        data = data[~np.isnan(data)]
        return max(np.std(data), 1e-8) if len(data) > 0 else 1e-8

    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate the maximum drawdown of the equity curve.
        :param equity_curve: Array of equity values.
        :return: Maximum drawdown.
        """
        if len(equity_curve) == 0:
            return 0.0
        cumulative_max = np.maximum.accumulate(equity_curve)
        drawdowns = (cumulative_max - equity_curve) / cumulative_max
        return max(drawdowns) if len(drawdowns) > 0 else 0.0

    def adjust_parameters_based_on_performance(self, sharpe_ratio, target_sharpe=1.0):
        """
        Adjust adaptive_k based on the Sharpe ratio.
        :param sharpe_ratio: Current Sharpe ratio.
        :param target_sharpe: Desired Sharpe ratio.
        :return: Adjusted adaptive_k.
        """
        adjustment_factor = sharpe_ratio / target_sharpe
        adaptive_k = self.config.get("adaptive_k", 5e-4) * adjustment_factor
        adaptive_k = max(adaptive_k, 1e-4)  # Prevent it from becoming too small
        return adaptive_k



