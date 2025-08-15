
#!/usr/bin/env python3
"""
Black-Scholes Model Accuracy Assessment for Nifty50 Options
============================================================

This script assesses the accuracy of the Black-Scholes Model by comparing
calculated option prices with real market prices for Nifty50 index options.

Author: Quantitative Analysis Team
Date: August 2025
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


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


class AccuracyAssessment:
    """Framework to assess Black-Scholes model accuracy"""

    @staticmethod
    def calculate_error_metrics(market_prices, bs_prices):
        """Calculate comprehensive error metrics"""
        market_prices = np.array(market_prices)
        bs_prices = np.array(bs_prices)

        errors = bs_prices - market_prices
        abs_errors = np.abs(errors)
        percentage_errors = (errors / market_prices) * 100

        return {
            'mean_error': np.mean(errors),
            'mean_absolute_error': np.mean(abs_errors),
            'root_mean_squared_error': np.sqrt(np.mean(errors**2)),
            'mean_absolute_percentage_error': np.mean(np.abs(percentage_errors)),
            'correlation': np.corrcoef(market_prices, bs_prices)[0, 1],
            'theil_u': AccuracyAssessment.theil_u_statistic(market_prices, bs_prices)
        }

    @staticmethod
    def theil_u_statistic(actual, predicted):
        """Calculate Theil's U statistic"""
        numerator = np.sqrt(np.mean((predicted - actual)**2))
        denominator = np.sqrt(np.mean(actual**2)) + np.sqrt(np.mean(predicted**2))
        return numerator / denominator if denominator != 0 else np.inf


def load_nifty_options_data(file_path):
    """
    Load Nifty50 options data from CSV file

    Expected columns:
    - spot_price: Current Nifty50 level
    - strike_price: Option strike price
    - time_to_maturity: Time to expiration in years
    - volatility: Implied volatility
    - risk_free_rate: Risk-free rate
    - market_call_price: Market price of call option
    - market_put_price: Market price of put option
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} option contracts from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Data file {file_path} not found. Generating synthetic data...")
        return generate_synthetic_data()


def generate_synthetic_data(n_samples=500):
    """Generate synthetic Nifty50 options data for demonstration"""
    np.random.seed(42)

    current_nifty = 24500
    spot_prices = np.random.normal(current_nifty, current_nifty * 0.02, n_samples)
    strike_prices = np.random.choice([current_nifty - 500, current_nifty - 250, current_nifty, 
                                     current_nifty + 250, current_nifty + 500], n_samples)
    days_to_expiry = np.random.choice([7, 15, 30, 45, 60, 90], n_samples)
    volatilities = np.random.uniform(0.12, 0.25, n_samples)
    risk_free_rate = 0.065

    data = []
    for i in range(n_samples):
        S = spot_prices[i]
        K = strike_prices[i]
        T = days_to_expiry[i] / 365
        sigma = volatilities[i]

        bs_model = BlackScholesModel(S, K, T, risk_free_rate, sigma)
        bs_call = bs_model.call_price()
        bs_put = bs_model.put_price()

        # Add market noise
        market_call = bs_call + np.random.normal(0, bs_call * 0.05)
        market_put = bs_put + np.random.normal(0, bs_put * 0.05)
        market_call = max(market_call, 0.5)
        market_put = max(market_put, 0.5)

        data.append({
            'spot_price': S, 'strike_price': K,
            'time_to_maturity': T, 'volatility': sigma,
            'risk_free_rate': risk_free_rate,
            'market_call_price': market_call,
            'market_put_price': market_put,
            'days_to_expiry': days_to_expiry[i]
        })

    return pd.DataFrame(data)


def main():
    """Main analysis function"""
    print("Black-Scholes Model Accuracy Assessment for Nifty50 Options")
    print("=" * 65)

    # Load or generate data
    data = load_nifty_options_data('nifty50_options_data.csv')

    # Initialize results storage
    results = {'call_options': [], 'put_options': []}

    print("\nCalculating Black-Scholes prices and comparing with market data...")

    # Calculate Black-Scholes prices for each contract
    bs_call_prices = []
    bs_put_prices = []

    for _, row in data.iterrows():
        bs_model = BlackScholesModel(
            S=row['spot_price'],
            K=row['strike_price'], 
            T=row['time_to_maturity'],
            r=row['risk_free_rate'],
            sigma=row['volatility']
        )

        bs_call_prices.append(bs_model.call_price())
        bs_put_prices.append(bs_model.put_price())

    # Calculate accuracy metrics
    call_metrics = AccuracyAssessment.calculate_error_metrics(
        data['market_call_price'], bs_call_prices
    )
    put_metrics = AccuracyAssessment.calculate_error_metrics(
        data['market_put_price'], bs_put_prices
    )

    # Display results
    print("\nACCURACY ASSESSMENT RESULTS")
    print("-" * 40)
    print("CALL OPTIONS:")
    for metric, value in call_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    print("\nPUT OPTIONS:")
    for metric, value in put_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

    # Monte Carlo comparison (sample)
    sample_row = data.iloc[0]
    print(f"\nMONTE CARLO COMPARISON (Sample Contract)")
    print("-" * 45)
    print(f"Spot: {sample_row['spot_price']:.2f}, Strike: {sample_row['strike_price']:.2f}")

    bs_model = BlackScholesModel(
        sample_row['spot_price'], sample_row['strike_price'],
        sample_row['time_to_maturity'], sample_row['risk_free_rate'],
        sample_row['volatility']
    )

    mc_pricer = MonteCarloOptionPricer(
        sample_row['spot_price'], sample_row['strike_price'],
        sample_row['time_to_maturity'], sample_row['risk_free_rate'],
        sample_row['volatility'], n_simulations=50000
    )

    bs_call = bs_model.call_price()
    bs_put = bs_model.put_price()
    mc_results = mc_pricer.price_european_options()

    print(f"Black-Scholes Call Price: ₹{bs_call:.2f}")
    print(f"Monte Carlo Call Price: ₹{mc_results['call_price']:.2f} ± {mc_results['call_std']:.2f}")
    print(f"Black-Scholes Put Price: ₹{bs_put:.2f}")
    print(f"Monte Carlo Put Price: ₹{mc_results['put_price']:.2f} ± {mc_results['put_std']:.2f}")

    # Save results
    results_df = data.copy()
    results_df['bs_call_price'] = bs_call_prices
    results_df['bs_put_price'] = bs_put_prices
    results_df['call_error'] = np.array(bs_call_prices) - data['market_call_price']
    results_df['put_error'] = np.array(bs_put_prices) - data['market_put_price']

    results_df.to_csv('black_scholes_accuracy_results.csv', index=False)
    print(f"\nResults saved to: black_scholes_accuracy_results.csv")

    return call_metrics, put_metrics


if __name__ == "__main__":
    call_results, put_results = main()
