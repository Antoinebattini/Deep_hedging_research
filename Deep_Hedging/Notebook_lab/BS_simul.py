import numpy as np
from scipy.stats import norm

class DeltaHedging:
    def __init__(self, S0, K, T, r, sigma,n_steps):
        """
        Initialize the parameters for the option and simulation.

        Parameters:
        S0: float : Initial stock price
        K: float : Strike price
        T: float : Time to maturity in years
        r: float : Risk-free interest rate
        sigma: float : Volatility of the stock
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps

    def simulate_stock_path(self,n_simulations=1):
        """
        Simulate stock price paths using Geometric Brownian Motion.

        Parameters:
        n_steps: int : Number of time steps
        n_simulations: int : Number of simulation paths

        Returns:
        np.ndarray : Simulated stock price paths
        """
        dt = self.T / self.n_steps
        stock_paths = np.zeros((n_simulations, self.n_steps + 1))
        stock_paths[:, 0] = self.S0

        for t in range(1, self.n_steps + 1):
            z = np.random.standard_normal(n_simulations)
            stock_paths[:, t] = stock_paths[:, t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * z)

        return stock_paths

    def black_scholes_price(self, S, t, option_type="call"):
        """
        Calculate the Black-Scholes price of a European option.

        Parameters:
        S: float : Current stock price
        t: float : Current time in years
        option_type: str : 'call' or 'put'

        Returns:
        float : Option price
        """
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        d2 = d1 - self.sigma * np.sqrt(self.T - t)

        if option_type == "call":
            return S * norm.cdf(d1) - self.K * np.exp(-self.r * (self.T - t)) * norm.cdf(d2)
        elif option_type == "put":
            return self.K * np.exp(-self.r * (self.T - t)) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def delta(self, S, t, option_type="call"):
        """
        Calculate the Delta of a European option.

        Parameters:
        S: float : Current stock price
        t: float : Current time in years
        option_type: str : 'call' or 'put'

        Returns:
        float : Delta of the option
        """
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))

        if option_type == "call":
            return norm.cdf(d1)
        elif option_type == "put":
            return norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")


