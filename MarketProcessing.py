import numpy as np
import logging
from RecursiveEGARCH import RecursiveEGARCH  # Assuming this is implemented elsewhere

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MarketProcessing:
    def __init__(self, data_manager, price_type):
        """
        Initialize the MarketProcessing class for a specific price type.
        :param data_manager: Instance of DataManager.
        :param price_type: Either 'bid' or 'ask'.
        """
        self.data_manager = data_manager
        self.price_type = price_type  # Either 'bid' or 'ask'

        # Access configuration from the DataManager
        self.config = self.data_manager.config

        # EGARCH model for volatility forecasting
        self.model = RecursiveEGARCH(dist="Normal")
        self.training_done = False

        # Scaling factor for normalization
        self.scaling_factor = 1.0

    def process_tick(self, t):
        """
        Process a single tick for the specified price type.
        Includes training, updating, and simulating volatility.
        :param t: Current time index.
        """
        if not self.training_done:
            self._train_model()
        else:
            forecast_vol = self._update_model()

            # Save forecasted volatility to DataManager
            self.data_manager.set(f"forecasted_vol_{self.price_type}", forecast_vol)

            # Perform multi-step simulation and set forecasted change
            forecasted_change = self._simulate_volatility(t)
            self.data_manager.set_config(f"forecasted_{self.price_type}_change", forecasted_change)
    def _train_model(self):
        """
        Train the EGARCH model for the specified price type.
        """
        try:

            # Ensure sufficient data for training
            if self.data_manager.get_size() > self.data_manager.get_config("training_window_size"):
                log_returns = self.data_manager.get(f"log_return_{self.price_type}")
                log_returns = log_returns[~np.isnan(log_returns)]
                # Calculate scaling factor based on standard deviation
                std_dev = np.std(log_returns[1:])
                self.scaling_factor = 1 / max(std_dev, 1e-20)

                logging.info(f"Scaling Factor {self.price_type}: {self.scaling_factor}")
                # Scale training data
                scaled_data = log_returns[1:] * self.scaling_factor
                self.model.train(scaled_data)

                # Save model parameters to DataManager
                # self.data_manager.set_config(f"adaptive_min_sigma_{self.price_type}", self.model.adaptive_min_sigma_model)

                self.training_done = True
                self.data_manager.set_config("is_trained", True)
                self.data_manager.set(f"long_term_vol_{self.price_type}", self.model.long_term_vol)
                logging.info(f"EGARCH model for {self.price_type} trained successfully.")
            else:
                logging.debug(f"Insufficient data for training {self.price_type} EGARCH model.")
        except Exception as e:
            logging.error(f"Failed to train EGARCH model for {self.price_type}: {e}")

    def _update_model(self):
        """
        Update the EGARCH model for the specified price type and forecast volatility.
        :return: Forecasted volatility.
        """
        try:
            latest_return = self.data_manager.get_latest_value(f"log_return_{self.price_type}")

            # Scale the return
            scaled_return = latest_return * self.scaling_factor

            # Update the model and scale back the forecast
            forecasted_vol_scaled = self.model.update(scaled_return)
            forecasted_vol = forecasted_vol_scaled / self.scaling_factor
            self.data_manager.set(f"long_term_vol_{self.price_type}", self.model.long_term_vol)

            return forecasted_vol
        except Exception as e:
            logging.error(f"Failed to update EGARCH model for {self.price_type}: {e}")

        return np.nan

    def _simulate_volatility(self, t):
        """
        Simulate volatility for multi-step forecasting and calculate forecasted price changes.
        :param t: Current tick index.
        :return: Forecasted price change for the specified price type.
        """
        if self.training_done:
            try:
                avg_volatility = self.model.simulate(
                    steps=self.data_manager.get_config("forecast_steps"),
                    num_simulations=self.data_manager.get_config("num_simulations"),
                )

                # Retrieve EMA price from DataManager
                ema_price = self.data_manager.get_latest_value(f"ema_{self.price_type}")

                # Use the first step's average volatility for forecasted changes
                forecasted_change = avg_volatility[0] * ema_price

                logging.debug(
                    f"t={t}: Simulated volatility for {self.price_type}: {avg_volatility[0]}, "
                    f"Forecasted Change: {forecasted_change}"
                )
                return forecasted_change
            except Exception as e:
                logging.error(f"Failed to simulate volatility for {self.price_type} at tick {t}: {e}")
                return 0
        else:
            logging.warning(f"Simulation skipped for {self.price_type} as training is not complete.")
            return 0
