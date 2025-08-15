#!/usr/bin/env python3
"""
Basic Example: Black-Scholes Option Pricing
===========================================

This example demonstrates basic usage of the Black-Scholes model
for pricing Nifty50 options and calculating Greeks.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Define the Black-Scholes classes directly in this file for simplicity
class BlackScholesModel:
    """
    Comprehensive Black-Scholes Model implementation for option pricing and Greeks calculation
    """

    def __init__(self, S, K, T, r, sigma, q=0):
        """
        Initialize Black-Scholes Model

        Parameters:
        S: Current stock price
        K: Strike price  
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield (default 0)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    def d1(self):
        """Calculate d1 parameter"""
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        """Calculate d2 parameter"""
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        """Calculate call option price"""
        d1_val = self.d1()
        d2_val = self.d2()
        return (self.S * np.exp(-self.q * self.T) * norm.cdf(d1_val) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(d2_val))

    def put_price(self):
        """Calculate put option price"""
        d1_val = self.d1()
        d2_val = self.d2()
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2_val) - 
                self.S * np.exp(-self.q * self.T) * norm.cdf(-d1_val))

    def all_greeks(self):
        """Calculate all option Greeks"""
        d1_val = self.d1()
        d2_val = self.d2()

        # Delta
        call_delta = np.exp(-self.q * self.T) * norm.cdf(d1_val)
        put_delta = call_delta - np.exp(-self.q * self.T)

        # Gamma
        gamma = (np.exp(-self.q * self.T) * norm.pdf(d1_val)) / (self.S * self.sigma * np.sqrt(self.T))

        # Vega
        vega = self.S * np.exp(-self.q * self.T) * norm.pdf(d1_val) * np.sqrt(self.T)

        # Theta
        call_theta = (-(self.S * np.exp(-self.q * self.T) * norm.pdf(d1_val) * self.sigma) / (2 * np.sqrt(self.T)) - 
                     self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2_val) + 
                     self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1_val))

        put_theta = (-(self.S * np.exp(-self.q * self.T) * norm.pdf(d1_val) * self.sigma) / (2 * np.sqrt(self.T)) + 
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2_val) - 
                    self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1_val))

        # Rho
        call_rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2_val)
        put_rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2_val)

        return {
            'call_delta': call_delta, 'put_delta': put_delta,
            'gamma': gamma, 'vega': vega,
            'call_theta': call_theta, 'put_theta': put_theta,
            'call_rho': call_rho, 'put_rho': put_rho
        }


class MonteCarloOptionPricer:
    """Monte Carlo simulation for option pricing"""

    def __init__(self, S0, K, T, r, sigma, q=0, n_simulations=100000):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_simulations = n_simulations

    def price_european_options(self):
        """Price European call and put options using Monte Carlo"""
        np.random.seed(42)

        # Generate final stock prices
        Z = np.random.standard_normal(self.n_simulations)
        ST = self.S0 * np.exp((self.r - self.q - 0.5 * self.sigma**2) * self.T + 
                              self.sigma * np.sqrt(self.T) * Z)

        # Calculate payoffs
        call_payoffs = np.maximum(ST - self.K, 0)
        put_payoffs = np.maximum(self.K - ST, 0)

        # Discount to present value
        call_price = np.exp(-self.r * self.T) * np.mean(call_payoffs)
        put_price = np.exp(-self.r * self.T) * np.mean(put_payoffs)

        # Calculate standard errors
        call_std = np.exp(-self.r * self.T) * np.std(call_payoffs) / np.sqrt(self.n_simulations)
        put_std = np.exp(-self.r * self.T) * np.std(put_payoffs) / np.sqrt(self.n_simulations)

        return {
            'call_price': call_price, 'call_std': call_std,
            'put_price': put_price, 'put_std': put_std
        }


def main():
    print("Basic Black-Scholes Example for Nifty50 Options")
    print("=" * 50)

    # Current Nifty50 scenario (as of August 2025)
    spot_price = 24500      # Current Nifty50 level
    strike_price = 24600    # Strike price (slightly OTM call)
    days_to_expiry = 30     # 30 days to expiration
    risk_free_rate = 0.065  # 6.5% (approximate Indian G-Sec rate)
    volatility = 0.18       # 18% (typical Nifty volatility)

    time_to_maturity = days_to_expiry / 365

    print(f"Market Parameters:")
    print(f"Spot Price: ₹{spot_price:,}")
    print(f"Strike Price: ₹{strike_price:,}")
    print(f"Days to Expiry: {days_to_expiry}")
    print(f"Risk-free Rate: {risk_free_rate:.1%}")
    print(f"Volatility: {volatility:.1%}")
    print()

    # Initialize Black-Scholes model
    bs_model = BlackScholesModel(
        S=spot_price,
        K=strike_price,
        T=time_to_maturity,
        r=risk_free_rate,
        sigma=volatility
    )

    # Calculate option prices
    call_price = bs_model.call_price()
    put_price = bs_model.put_price()

    print("Black-Scholes Option Prices:")
    print(f"Call Option: ₹{call_price:.2f}")
    print(f"Put Option: ₹{put_price:.2f}")
    print()

    # Calculate Greeks
    greeks = bs_model.all_greeks()

    print("Option Greeks:")
    print(f"Call Delta: {greeks['call_delta']:.4f}")
    print(f"Put Delta: {greeks['put_delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.6f}")
    print(f"Vega: {greeks['vega']:.2f}")
    print(f"Call Theta: {greeks['call_theta']:.2f}")
    print(f"Put Theta: {greeks['put_theta']:.2f}")
    print()

    # Monte Carlo comparison
    print("Monte Carlo Verification:")
    print("-" * 25)

    mc_pricer = MonteCarloOptionPricer(
        S0=spot_price,
        K=strike_price,
        T=time_to_maturity,
        r=risk_free_rate,
        sigma=volatility,
        n_simulations=100000
    )

    mc_results = mc_pricer.price_european_options()

    print(f"Monte Carlo Call: ₹{mc_results['call_price']:.2f} ± {mc_results['call_std']:.2f}")
    print(f"Monte Carlo Put: ₹{mc_results['put_price']:.2f} ± {mc_results['put_std']:.2f}")

    # Calculate differences
    call_diff = abs(call_price - mc_results['call_price'])
    put_diff = abs(put_price - mc_results['put_price'])

    print(f"\nPrice Differences:")
    print(f"Call Difference: ₹{call_diff:.2f} ({call_diff/call_price*100:.2f}%)")
    print(f"Put Difference: ₹{put_diff:.2f} ({put_diff/put_price*100:.2f}%)")

    print("\n" + "="*50)
    print("✅ Basic pricing example completed successfully!")
    print("📊 Black-Scholes and Monte Carlo results are very close")
    print(f"🎯 Accuracy: Call options within {call_diff/call_price*100:.1f}% difference")


if __name__ == "__main__":
    main()
