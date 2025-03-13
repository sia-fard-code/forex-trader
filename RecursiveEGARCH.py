

import numpy as np
from arch import arch_model


class RecursiveEGARCHOld:
    """
    A class to implement recursive updates for an EGARCH(1,1) model.
    """

    def __init__(self, p=1, o=1, q=1, dist="Normal"):
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.params = None
        self.last_volatility = 1.0  # Reasonable starting volatility
        self.model = None

    def train(self, returns):
        """
        Train the EGARCH model on a dataset and store the parameters.
        """
        if len(returns) == 0 or np.all(np.isnan(returns)):
            raise ValueError("Training data for EGARCH is empty or invalid.")

        self.model = arch_model(
            returns,
            mean="Zero",
            vol="EGARCH",
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist,
        )

        result = self.model.fit(disp="off")
        self.params = result.params

        # Initialize last_volatility based on the last fitted conditional volatility
        if 'omega' in self.params and 'alpha[1]' in self.params and 'beta[1]' in self.params:
            # Compute initial volatility based on the model
            # Alternatively, use the last known volatility from the training data
            self.last_volatility = np.sqrt(result.conditional_volatility[-1])
        else:
            self.last_volatility = 1.0  # Fallback
    def update(self, return_t):
        """
        Update the volatility estimate for a new return.
        """
        if self.params is None:
            raise ValueError("Model parameters not set. Train the model first.")

        # Replace NaN returns with zero
        return_t = np.nan_to_num(return_t, nan=0.0)

        # Retrieve parameters
        omega = self.params.get("omega", 0.0)
        alpha = self.params.get("alpha[1]", 0.0)
        gamma = self.params.get("gamma[1]", 0.0)
        beta = self.params.get("beta[1]", 0.0)

        # Compute standardized residual
        standardized_return = return_t / self.last_volatility

        # Compute expected value of |z_t| for a normal distribution
        expected_abs_standardized_return = np.sqrt(2 / np.pi)

        # Update log variance
        ln_sigma_sq = (
            omega
            + alpha * (abs(standardized_return) - expected_abs_standardized_return)
            + gamma * standardized_return
            + beta * np.log(self.last_volatility**2)
        )

        # Compute new volatility
        self.last_volatility = np.exp(ln_sigma_sq / 2)

        # Cap volatility to avoid overflow or underflow
        self.last_volatility = np.clip(self.last_volatility, 1e-8, 1e6)

        return self.last_volatility
    
    def simulate(self, steps, num_simulations=100):
        """
        Simulate future volatility for the given number of steps using Monte Carlo simulations.
        """
        if self.params is None:
            raise ValueError("Model parameters not set. Train the model first.")

        if steps <= 0:
            raise ValueError("Number of steps for simulation must be greater than 0.")

        # Retrieve model parameters
        omega = self.params.get("omega", 0.0)
        alpha = self.params.get("alpha[1]", 0.0)
        gamma = self.params.get("gamma[1]", 0.0)
        beta = self.params.get("beta[1]", 0.0)

        # Initialize the volatility simulations
        simulations = np.zeros((num_simulations, steps))
        initial_volatility = self.last_volatility

        # Generate random returns for all simulations at once
        random_returns = np.random.normal(0, 1, size=(num_simulations, steps))

        for sim in range(num_simulations):
            vol = initial_volatility
            for t in range(steps):
                # Compute standardized return
                standardized_return = random_returns[sim, t] / vol

                # Update log variance
                ln_sigma_sq = (
                    omega
                    + alpha * (abs(standardized_return) - np.sqrt(2 / np.pi))  # Expected value for normal distribution
                    + gamma * standardized_return
                    + beta * np.log(vol**2)
                )

                # Compute new volatility
                vol = np.exp(ln_sigma_sq / 2)
                vol = np.clip(vol, 1e-8, 1e6)  # Cap volatility for numerical stability

                # Store volatility
                simulations[sim, t] = vol

        # Calculate the average volatility across all simulations
        avg_volatility = np.mean(simulations, axis=0)
        return avg_volatility



class RecursiveEGARCH:
    def __init__(self, dist='normal'):
        self.dist = dist
        self.model = None
        self.fitted_model = None
        self.params = None
        self.conditional_volatility = []
        # self.long_term_vol = None
    
    def train(self, returns):
        """
        Train the EGARCH model on the provided returns.
        """
        # self.determine_scaling_factor(returns)
        # print(self.scaling_factor)
        best_aic = np.inf
        best_model = None

        for p in range(1, 3):
            for o in range(1, 2):
                for q in range(1, 3):
                    try:
                        model = arch_model(
                            returns,
                            mean="Zero",
                            vol="EGARCH",
                            p=p,
                            o=o,
                            q=q,
                            dist=self.dist,
                            rescale=False  # Already scaled
                        )
                        result = model.fit(disp="off")
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_model = model
                    except Exception as e:
                        print(f"Model fitting failed for p={p}, o={o}, q={q}: {e}")

        if best_model is None:
            raise ValueError("No suitable EGARCH model found during dynamic selection.")

        # Fit the best model
        self.fitted_model = best_model.fit(disp='off')
        self.params = self.fitted_model.params

        omega = self.params.get('omega', 0.0)
        alpha = self.params.get('alpha[1]', 0.0)
        beta = self.params.get('beta[1]', 0.0)
        self.long_term_vol = np.sqrt(omega / (1 - alpha - beta)) if (1 - alpha - beta) > 0 else 1e-4

        scaled_volatility = np.array(self.fitted_model.conditional_volatility)
        self.conditional_volatility = scaled_volatility.tolist()

        print("EGARCH model trained with parameters:", self.params)

    def compute_expected_abs_residual(self):
        if self.dist.lower() == 'normal':
            return np.sqrt(2 / np.pi)
        elif self.dist.lower() == 'studentst':
            dof = self.params.get('nu', 10)  # Default degrees of freedom
            return 2 * np.sqrt(dof - 2) / ((dof - 1) * np.sqrt(np.pi))
        else:
            raise NotImplementedError(f"Distribution {self.dist} not supported.")
    def update(self, return_t):
        """
        Update the volatility estimate for a new return.
        """
        if not self.fitted_model:
            raise ValueError("Model not trained yet.")
        
        # Retrieve parameters
        omega = self.params.get('omega', 0.0)
        alpha = self.params.get('alpha[1]', 0.0)
        gamma = self.params.get('gamma[1]', 0.0)
        beta = self.params.get('beta[1]', 0.0)
        
        # Compute standardized residual
        last_volatility = self.conditional_volatility[-1]
        standardized_return = return_t / last_volatility
        
        # Expected value of |z_t| for a normal distribution
        expected_abs_z = self.compute_expected_abs_residual() #np.sqrt(2 / np.pi)
        
        # Update log variance
        ln_sigma_sq = (
            omega
            + alpha * (abs(standardized_return) - expected_abs_z)
            + gamma * standardized_return
            + beta * np.log(last_volatility**2)
        )
        
        # Compute new volatility
        ln_sigma_sq = np.clip(ln_sigma_sq,-1e3,1e3)
        new_volatility = np.exp(ln_sigma_sq / 2) 
        
        # Cap volatility to avoid extreme values
        new_volatility = np.clip(new_volatility, 1e-8, 1e6)
        
        # Append to conditional volatility
        self.conditional_volatility.append(new_volatility)
        self.long_term_vol = np.sqrt(omega / (1 - alpha - beta)) if (1 - alpha - beta) > 0 else 1e-4
        
        return new_volatility
    
    def simulate(self, steps=1, num_simulations = 1):
        """
        Simulate future volatility for the given number of steps using Monte Carlo simulations.
        """
        if not self.fitted_model:
            raise ValueError("Model not trained yet.")

        last_volatility = self.conditional_volatility[-1]
        simulations = np.zeros((num_simulations, steps))
        vol = np.full((num_simulations,), last_volatility)

        for step in range(steps):
            z = np.random.normal(0, 1, size=num_simulations)
            ln_sigma_sq = (
                self.params.get('omega', 0.0)
                + self.params.get('alpha[1]', 0.0) * (np.abs(z) - np.sqrt(2 / np.pi))
                + self.params.get('gamma[1]', 0.0) * z
                + self.params.get('beta[1]', 0.0) * np.log(vol**2)
            )
            ln_sigma_sq = np.clip(ln_sigma_sq,-1e4,1e4)
            vol = np.exp(ln_sigma_sq / 2)
            vol = np.clip(vol, 1e-8, 1e6)  # Cap volatility
            simulations[:, step] = vol

        avg_volatility = simulations.mean(axis=0)
        return avg_volatility


