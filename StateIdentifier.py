import numpy as np
import logging

class StateIdentifier:
    def __init__(self, data_manager):
        """
        Initialize the StateIdentifier class.
        :param data_manager: Instance of DataManager.
        """
        self.data_manager = data_manager
        self.config = self.data_manager.get_config  # Access config from DataManager

    def process(self, t, bid, ask):
        """
        Process the state identification workflow.
        Determines the initial states and refines them.
        :param t: Current tick index.
        """
        # Determine initial state
        initial_state = self.determine_initial_state(t, bid, ask)

        # Set the initial state in DataManager
        self.data_manager.set("initial_state", initial_state)

        # Refine the state
        refined_state = self.refine_state(t)
        # logging.info(f"-+-+-+-+-+-+-- ({refined_state}) completed -+-+-+-+----")

        # Set the refined state in DataManager
        self.data_manager.set("refined_state", refined_state)

    def determine_initial_state(self, t,  bid, ask):
        """
        Determine the initial market state (Bullish, Bearish, Neutral).
        :param t: Current tick index.
        :return: Initial market state (1 for Bullish, -1 for Bearish, 0 for Neutral).
        """
        try:
            # Retrieve required data
            forecasted_vol_bid = self.data_manager.get_latest_value("forecasted_vol_bid")
            forecasted_vol_ask = self.data_manager.get_latest_value("forecasted_vol_ask")
            log_return_bid = self.data_manager.get_latest_value("log_return_bid")
            log_return_ask = self.data_manager.get_latest_value("log_return_ask")
            forecasted_ask_change = self.data_manager.get_config("forecasted_ask_change")
            forecasted_bid_change = self.data_manager.get_config("forecasted_bid_change")

            # Calculate spread from DataManager
            # bid = self.data_manager.get_latest_value("bid")
            # ask = self.data_manager.get_latest_value("ask")
            # spread = self.data_manager.get_latest_value("spread")
            spread =  ask - bid
            percent = 80

            # Retrieve thresholds
            rolling_window_size = 500#self.data_manager.get_config("rolling_window_size")
            vol_bid_perc, vol_bid_mean, vol_bid_std = self._calculate_threshold("forecasted_vol_bid", t, rolling_window_size,percent=percent)
            vol_ask_perc, vol_ask_mean, vol_ask_std = self._calculate_threshold("forecasted_vol_ask", t, rolling_window_size,percent=percent)
            return_bid_perc, return_bid_mean, return_bid_std = self._calculate_threshold("log_return_bid", t, rolling_window_size, absolute=True,percent=percent)
            return_ask_perc, return_ask_mean, return_ask_std = self._calculate_threshold("log_return_ask", t, rolling_window_size, absolute=True,percent=percent)

            threshold_vol_bid = vol_bid_perc #vol_bid_mean+vol_bid_std
            threshold_vol_ask = vol_ask_perc #vol_ask_mean+vol_ask_std
            threshold_return_bid = return_bid_mean+return_bid_std
            threshold_return_ask = return_ask_mean-return_ask_std

            # Determine market state
            if (
                forecasted_vol_bid >= threshold_vol_bid
                and forecasted_vol_ask >= threshold_vol_ask
                and spread <= self.data_manager.get_config("max_spread")
            ):
                if log_return_bid > threshold_return_bid and forecasted_ask_change > spread:
                    initial_market_state = 1  # Bullish
                    logging.debug(f"t={t}: Market state determined as Bullish.")
                elif log_return_ask < -threshold_return_ask and forecasted_bid_change > spread:
                    initial_market_state = -1  # Bearish
                    logging.debug(f"t={t}: Market state determined as Bearish.")
                else:
                    initial_market_state = 0  # Neutral
                    logging.debug(f"t={t}: Market state determined as Neutral after conditions.")
            else:
                initial_market_state = 0  # Neutral due to low volatility
                logging.debug(f"t={t}: Market state determined as Neutral due to low volatility.")

            return initial_market_state

        except KeyError as e:
            logging.error(f"Missing key in DataManager: {e}")
            return 0  # Default to Neutral if data is missing
        except IndexError as e:
            logging.error(f"Index error in DataManager: {e}")
            return 0  # Default to Neutral if data retrieval fails

    def refine_state(self, t):
        """
        Refine the market state using a sliding window.
        :param t: Current tick index.
        :return: Refined market state (1 for Bullish, -1 for Bearish, 0 for Neutral).
        """
        refined_state = 0
        rolling_window_size = 500#self.data_manager.get_config("rolling_window_size")
        max_window_size = self.data_manager.get_config("max_window_size")
        dynamic_threshold_multiplier = 2 * np.sqrt(max_window_size)

        try:
            # Retrieve recent states
            rolling_states = self.data_manager.get("initial_state", rolling_window_size)

            if rolling_states.size > 0:
                median_state = np.nanmedian(rolling_states)
                std_dev = np.nanstd(rolling_states)
                deviation = dynamic_threshold_multiplier * std_dev

                dynamic_threshold_bullish = median_state + deviation
                dynamic_threshold_bearish = median_state - deviation

                for window_size in range(3, max_window_size + 1):
                    if window_size > rolling_states.size:
                        continue
                    window_data = rolling_states[-window_size:]
                    sum_window = np.nansum(window_data)

                    if sum_window >= dynamic_threshold_bullish:
                        refined_state = 1  # Bullish
                        logging.debug(f"t={t}: Refined state determined as Bullish with window size {window_size}.")
                        break
                    elif sum_window <= dynamic_threshold_bearish:
                        refined_state = -1  # Bearish
                        logging.debug(f"t={t}: Refined state determined as Bearish with window size {window_size}.")
                        break

        except KeyError as e:
            logging.error(f"Missing key in DataManager during refinement: {e}")
        except ValueError as e:
            logging.error(f"Value error during refinement: {e}")

        return refined_state

    def _calculate_threshold(self, key, t, window_size, absolute=False, percent=50):
        """
        Calculate a threshold based on historical data.
        :param key: Data key to fetch (e.g., 'forecasted_vol_bid').
        :param t: Current tick index.
        :param window_size: Rolling window size.
        :param absolute: Whether to use the absolute value of the data.
        :return: Calculated threshold.
        """
        try:
            data = self.data_manager.get(key, window_size)
            data = data[~np.isnan(data)]  # Filter out NaN values
            if absolute:
                data = np.abs(data)

            if data.size > 0:
                return np.percentile(data, percent), np.mean(data), np.std(data)
            else:
                return 0.0, 0.0, 0.0

        except KeyError as e:
            logging.error(f"Missing key in DataManager during threshold calculation: {e}")
            return 0.0
        except Exception as e:
            logging.error(f"Error calculating threshold for {key}: {e}")
            return 0.0