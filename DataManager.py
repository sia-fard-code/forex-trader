import numpy as np


class DataManager:
    def __init__(self, config):
        """
        Initialize the DataManager with a fixed-size buffer.
        :param config: Configuration dictionary.
        """
        self.config = config
        self.buffer_size = config['buffer_size']
        self.ema_period = config['ema_window']

        # Initialize fixed-size buffer
        self.data = np.full(
            self.buffer_size,
            fill_value=np.nan,
            dtype=[
                ('tick_id', 'i8'),
                ('timestamp', 'i4'),
                ('bid', 'f8'),
                ('ask', 'f8'),
                ('ema_bid', 'f8'),
                ('ema_ask', 'f8'),
                ('log_return_bid', 'f8'),
                ('log_return_ask', 'f8'),
                ('arithmetic_return_bid', 'f8'),
                ('arithmetic_return_ask', 'f8'),
                ('forecasted_vol_bid', 'f8'),
                ('forecasted_vol_ask', 'f8'),
                ('initial_state', 'f8'),
                ('refined_state', 'f8'),
                ('position_size_bid', 'f8'),
                ('position_size_ask', 'f8'),
                ('equity', 'f8'),
                ('balance', 'f8'),
                ('pnl', 'f8'),
                ('cumulative_pnl', 'f8'),
                ('margin_used', 'f8'),
                ('free_margin', 'f8'),
                ('long_term_vol_ask', 'f8'),
                ('long_term_vol_bid', 'f8'),
                ('adjusted_position_size_bid', 'f8'),
                ('adjusted_position_size_ask', 'f8'),
                ('spread', 'f8'),
                ('trade_approved', 'f8'),  # Added for ProfitabilityFilter
            ]
        )
        self.index = 0  # Points to the next position to write
        self.size = 0   # Tracks the number of valid rows in the buffer
        # Cache for storing computed log returns
        self.cache = {
            'log_return_bid': {'current_ema': 0.0, 'previous_ema': 0.0, 'log_return': 0.0},
            'log_return_ask': {'current_ema': 0.0, 'previous_ema': 0.0, 'log_return': 0.0}
        }

    def update(self, t, bid, ask,tick_id):
        """
        Append new market data, compute indicators, and store it in the buffer.
        :param t: Current tick index.
        :param bid: Bid price.
        :param ask: Ask price.
        """
        # Get the previous EMA values
        prev_ema_bid = self.data['ema_bid'][self.index - 1] if self.size > 0 else bid
        prev_ema_ask = self.data['ema_ask'][self.index - 1] if self.size > 0 else ask

        # Compute EMA
        current_ema_bid = self.calculate_ema(bid, prev_ema_bid)
        current_ema_ask = self.calculate_ema(ask, prev_ema_ask)

        # Compute returns
        log_return_bid = np.log(current_ema_bid / (prev_ema_bid + 1e-16)) if self.size > 0 else 0.0
        log_return_ask = np.log(current_ema_ask / (prev_ema_ask + 1e-16)) if self.size > 0 else 0.0
        # log_return_bid = self.compute_log_return(current_ema_bid, prev_ema_bid, 'log_return_bid')
        # log_return_ask = self.compute_log_return(current_ema_ask, prev_ema_ask, 'log_return_ask')
        arithmetic_return_bid = (current_ema_bid - prev_ema_bid) / (prev_ema_bid + 1e-16) if self.size > 0 else 0.0
        arithmetic_return_ask = (current_ema_ask - prev_ema_ask) / (prev_ema_ask + 1e-16) if self.size > 0 else 0.0
        spread = ask - bid

        # Write data to the buffer
        self.data[self.index] = (
            tick_id, t, bid, ask, current_ema_bid, current_ema_ask,
            log_return_bid, log_return_ask, arithmetic_return_bid, arithmetic_return_ask,
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, spread, 0  # Added 0 for trade_approved
        )

        # Update buffer metadata
        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_log_return(self, current_ema, previous_ema, cache_key, tolerance=1e-6):
        """
        Compute the log return and cache the result based on input values.
        If the same current_ema and previous_ema are provided, return the cached log return.
        :param current_ema: Current EMA value.
        :param previous_ema: Previous EMA value.
        :param cache_key: Cache key for the log return ('log_return_bid' or 'log_return_ask').
        :return: Log return value.
        """
        cache_entry = self.cache.get(cache_key)
        if cache_entry is not None:
            # if cache_entry['current_ema'] == current_ema and cache_entry['previous_ema'] == previous_ema:
            if (
                abs(cache_entry['current_ema'] - current_ema) < tolerance
                and abs(cache_entry['previous_ema'] - previous_ema) < tolerance
            ):
                # Return cached log return
                return cache_entry['log_return']
        
        # Compute the log return
        log_return = np.log(current_ema / (previous_ema + 1e-16)) if self.size > 0 else 0.0
        
        # Update the cache
        if cache_entry is not None:
            cache_entry['current_ema'] = current_ema
            cache_entry['previous_ema'] = previous_ema
            cache_entry['log_return'] = log_return
        
        return log_return
    
    def calculate_ema(self, current_price, previous_ema):
        """
        Compute the Exponential Moving Average (EMA).
        :param current_price: Current price.
        :param previous_ema: Previous EMA value.
        :return: Updated EMA value.
        """
        multiplier = 2 / (self.ema_period + 1)
        return (current_price - previous_ema) * multiplier + previous_ema

    def reset_cache(self):
        """
        Reset the cache for all cached values.
        """
        self.cache = {
            'log_return_bid': {'current_ema': None, 'previous_ema': None, 'log_return': None},
            'log_return_ask': {'current_ema': None, 'previous_ema': None, 'log_return': None}
        }

    def get_latest_data(self):
        """
        Retrieve the latest row of data.
        :return: Dictionary containing the latest row.
        """
        if self.size == 0:
            return None
        return self.data[(self.index - 1) % self.buffer_size]

    def get_size(self):
        """
        Retrieve the latest row of data.
        :return: Dictionary containing the latest row.
        """
        if self.size == 0:
            return None
        return self.size

    def get_latest_value(self, key):
        """
        Retrieve the latest value of a specific key (column).
        :param key: Column name to retrieve.
        :return: Latest value of the specified column.
        """
        if key not in self.data.dtype.names:
            raise KeyError(f"Column '{key}' does not exist in the DataManager.")
        if self.size == 0:
            raise ValueError("No data in the buffer.")
        
        latest_index = (self.index - 1) % self.buffer_size
        return self.data[key][latest_index]

    def get_window_data(self, window_size):
        """
        Retrieve the latest data within a window size.
        :param window_size: Number of recent rows to retrieve.
        :return: NumPy structured array containing the latest rows.
        """
        if self.size == 0:
            raise ValueError("No data in the buffer.")
        
        # Calculate the start index for the sliding window
        start_index = (self.index - min(window_size, self.size)) % self.buffer_size
        
        if start_index < self.index:
            # Continuous slice
            return self.data[start_index:self.index]
        else:
            # Wrap-around slice
            return np.concatenate((self.data[start_index:], self.data[:self.index]))
    
    def update_model_outputs(self, **kwargs):
        """
        Update model outputs like forecasted volatility, position size, or PnL in the latest row.
        :param kwargs: Key-value pairs of columns to update.
        """
        if self.size == 0:
            raise IndexError("Cannot update model outputs because the buffer is empty.")

        latest_index = (self.index - 1) % self.buffer_size
        for key, value in kwargs.items():
            if key not in self.data.dtype.names:
                raise KeyError(f"Column '{key}' does not exist in the DataManager.")
            self.data[key][latest_index] = value

    def get(self, key, window_size=None):
        """
        Retrieve a specific column from the buffer.
        Optionally, retrieve only the last `window_size` rows of the column.
        :param key: Column name to retrieve.
        :param window_size: Optional number of rows to retrieve from the end.
        :return: NumPy array containing the data for the specified column.
        """
        if key not in self.data.dtype.names:
            raise KeyError(f"Column '{key}' does not exist in the DataManager.")

        if window_size is not None:
            # Get the data within the specified window size
            window_data = self.get_window_data(window_size)
            return window_data[key]
        
        # Return the entire column up to the current size
        return self.data[key][:self.size]

    def get_config(self, param_name):
        """
        Fetch a specific configuration parameter.
        :param param_name: Name of the parameter.
        :return: Value of the parameter or None if not found.
        """
        return self.config[param_name]

    def set_config(self, param_name, value):
        """
        Set a specific configuration parameter.
        :param param_name: Name of the parameter.
        :param value: Value to set for the parameter.
        """
        self.config[param_name] = value

    def set(self, key, value):
        """
        Update a specific column in the latest row of the circular buffer.
        Assumes that rows are already added using the `update` method.
        :param key: The column name to update.
        :param value: The value to set for the column in the latest row.
        """
        if key not in self.data.dtype.names:
            raise KeyError(f"Column '{key}' does not exist in the DataManager.")
        
        if self.size == 0:
            raise IndexError("Cannot update an empty buffer. Please add a row first using the update method.")
        
        # Find the index of the latest row in the circular buffer
        last_index = (self.index - 1) % self.buffer_size
        
        # Update the specified column in the latest row
        self.data[key][last_index] = value
