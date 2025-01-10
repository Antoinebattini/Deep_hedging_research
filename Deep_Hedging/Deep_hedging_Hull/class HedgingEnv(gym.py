class HedgingEnv:
    def __init__(self, initial_stock_price=100, strike_price=100, maturity=1, volatility=0.2, risk_free_rate=0.01, time_steps=252):
        self.initial_stock_price = initial_stock_price
        self.strike_price = strike_price
        self.maturity = maturity
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.time_steps = time_steps

        self.dt = maturity / time_steps
        self.current_step = 0

        self.stock_paths = None
        self.hedge_position = 0  # H (action to find)
        self.call_option_value = 0

        self.state_space = 3  # [time to maturity, current stock price, current hedge position]
        self.action_space = 21  # Discrete actions: fractions of the stock [-1, -0.9, ..., 0.9, 1]
        self.actions = np.linspace(-1, 1, self.action_space)  # Fractions of stock to buy/sell

    def reset(self):
        self.current_step = 0
        self.stock_paths = generate_black_scholes_paths(self.initial_stock_price, self.risk_free_rate, self.volatility, self.maturity, self.dt,1)
        self.hedge_position = 0
        time_to_maturity = self.maturity - self.current_step * self.dt
        current_stock_price = self.stock_paths[self.current_step]
        self.call_option_value = call_option_price(self.stock_paths[self.current_step],self.strike_price,self.risk_free_rate,self.volatility,self.maturity)
        return np.array([time_to_maturity, current_stock_price, self.hedge_position])

    def step(self, action_index):
        action = self.actions[action_index]  # Convert discrete action index to fraction

        # Update time and state
        self.current_step += 1
        time_to_maturity = self.maturity - self.current_step * self.dt
        current_stock_price = self.stock_paths[self.current_step]
        next_call_option_value = call_option_price(current_stock_price,self.strike_price,self.risk_free_rate,self.volatility, time_to_maturity)

        # Calculate reward based on the provided formula
        reward = (next_call_option_value - self.call_option_value) + (current_stock_price - self.stock_paths[self.current_step - 1]) * self.hedge_position - 0.01 * abs(current_stock_price * (action - self.hedge_position))

        # Update hedge position
        self.hedge_position = action
        self.call_option_value = next_call_option_value

        # Check if the episode is done
        done = self.current_step >= self.time_steps - 1

        # State is [time to maturity, current stock price, current hedge position]
        state = np.array([time_to_maturity, current_stock_price, self.hedge_position])

        return state, reward, done
