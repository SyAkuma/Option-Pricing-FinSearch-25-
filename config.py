"""
Configuration file for Black-Scholes Nifty50 Assessment
======================================================

Different market scenarios and parameters for testing
"""

# Current market parameters (August 2025)
CURRENT_NIFTY_PARAMS = {
    "spot_price": 24500,
    "risk_free_rate": 0.065,  # 6.5% (Indian 10Y G-Sec)
    "base_volatility": 0.18,  # 18% (typical Nifty volatility)
}

# Different volatility regimes
VOLATILITY_SCENARIOS = {
    "low_vol": 0.12,      # Calm market conditions
    "normal_vol": 0.18,   # Typical market conditions  
    "high_vol": 0.25,     # Stressed market conditions
    "extreme_vol": 0.35   # Crisis conditions (like COVID-19)
}

# Strike price scenarios (relative to spot)
MONEYNESS_SCENARIOS = {
    "deep_otm": 0.90,     # 10% out-of-the-money
    "otm": 0.97,          # 3% out-of-the-money
    "atm": 1.00,          # At-the-money
    "itm": 1.03,          # 3% in-the-money
    "deep_itm": 1.10      # 10% in-the-money
}

# Time to expiration scenarios
MATURITY_SCENARIOS = {
    "weekly": 7,          # 1 week
    "monthly": 30,        # 1 month
    "quarterly": 90,      # 3 months
    "semi_annual": 180,   # 6 months
    "annual": 365         # 1 year
}

# Risk-free rate scenarios
INTEREST_RATE_SCENARIOS = {
    "low_rates": 0.04,    # 4% (developed market rates)
    "normal_rates": 0.065, # 6.5% (current Indian rates)
    "high_rates": 0.09    # 9% (high inflation scenario)
}

# Backtesting parameters
BACKTEST_CONFIG = {
    "n_samples": 500,
    "random_seed": 42,
    "confidence_level": 0.95
}
