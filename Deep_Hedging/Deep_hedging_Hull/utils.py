""" Utility Functions """
import random

import numpy as np
from scipy.stats import norm



def brownian_sim(num_path, num_period, mu, std, init_p, dt):
    z = np.random.normal(size=(num_path, num_period))

    a_price = np.zeros((num_path, num_period))
    a_price[:, 0] = init_p

    for t in range(num_period - 1):
        a_price[:, t + 1] = a_price[:, t] * np.exp(
            (mu - (std ** 2) / 2) * dt + std * np.sqrt(dt) * z[:, t]
        )
    return a_price


# BSM Call Option Pricing Formula & BS Delta formula
# T here is time to maturity
def bs_call(iv, T, S, K, r, q):
    d1 = (np.log(S / K) + (r - q + iv * iv / 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    bs_delta = np.exp(-q * T) * norm.cdf(d1)
    return bs_price, bs_delta


def get_sim_path(M, freq,num_sim):
    """ Return simulated data: a tuple of three arrays
        M: initial time to maturity
        freq: trading freq in unit of day, e.g. freq=2: every 2 day; freq=0.5 twice a day;
        np_seed: numpy random seed
        num_sim: number of simulation path

        1) asset price paths (num_path x num_period)
        2) option price paths (num_path x num_period)
        3) delta (num_path x num_period)
    """
    # set the np random seed

    # Trading Freq per day; passed from function parameter
    # freq = 2
    # Annual Trading Day
    T = 250
    # Simulation Time Step
    dt = 0.004 * freq
    # Option Day to Maturity; passed from function parameter
    # M = 60
    # Number of period
    num_period = int(M / freq)
    # Number of simulations; passed from function parameter
    # num_sim = 1000000
    # Annual Return
    mu = 0.05
    # Annual Volatility
    vol = 0.2
    # Initial Asset Value
    S = 100
    # Option Strike Price
    K = 100
    # Annual Risk Free Rate
    r = 0
    # Annual Dividend
    q = 0
    # asset price 2-d array
    print("1. generate asset price paths")
    a_price = brownian_sim(num_sim, num_period + 1, mu, vol, S, dt)

    # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
    ttm = np.arange(M, -freq, -freq)

    # BS price 2-d array and bs delta 2-d array
    print("2. generate BS price and delta")
    bs_price, bs_delta = bs_call(vol, ttm / T, a_price, K, r, q)

    print("simulation done!")

    return a_price, bs_price, bs_delta



def bs_call(iv, T, S, K, r, q):
    d1 = (np.log(S / K) + (r - q + iv * iv / 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    bs_delta = np.exp(-q * T) * norm.cdf(d1)
    return bs_price, bs_delta


def compute_rewards(
    path: np.ndarray,
    option_price_path: np.ndarray,
    delta_path: np.ndarray,
    strike_price: float,
    spread: float,
):

    N, total_T = path.shape
    print(N)
    T = total_T - 1 
    
    # Prepare an array to store the rewards: shape (N, T)
    rewards = np.zeros((N, T), dtype=np.float64)
    
    # Loop over each path
    for i in range(N):
        # For each time step t from 0 to T-1
        for t in range(T):
            current_price = path.iloc[i, t]
            current_option_price = option_price_path.iloc[i, t]
            current_position = delta_path.iloc[i, t]

            next_price = path.iloc[i, t+1]
            next_option_price = option_price_path.iloc[i, t+1]
            next_position = delta_path.iloc[i, t+1]

            # ---------------------
            # Reward calculation
            # ---------------------
            # 1) PnL from underlying movement:
            #    (next_price - current_price) * next_position / 100
            # 2) Minus transaction costs from changing the hedge position:
            #    |current_position - next_position| * current_price * (spread / 100)
            reward_t = (next_price - current_price) * next_position \
                       - abs(current_position - next_position) * current_price * spread

            # Final step adjustments:
            if t == T - 1:
                payoff_diff = (max(next_price - strike_price, 0.0) - next_option_price)
                cost_to_close = next_position * next_price * spread
                reward_t -= payoff_diff + cost_to_close
            else:

                reward_t -= (next_option_price - current_option_price)

            rewards[i, t] = reward_t

    return rewards
