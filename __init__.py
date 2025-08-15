"""
Black-Scholes Model Accuracy Assessment for Nifty50 Options
===========================================================

A comprehensive Python implementation for assessing the accuracy of the 
Black-Scholes Model against Nifty50 options market data.

Main Components:
- BlackScholesModel: Complete option pricing with Greeks
- MonteCarloOptionPricer: Alternative pricing method
- AccuracyAssessment: Statistical validation framework

Usage:
    from black_scholes_nifty_assessment import BlackScholesModel

    model = BlackScholesModel(S=24500, K=24600, T=30/365, r=0.065, sigma=0.18)
    call_price = model.call_price()
    put_price = model.put_price()
"""

__version__ = "1.0.0"
__author__ = "FinSearch Option Pricing Team"
__email__ = "asyed3627@gmail.com"

# Import main classes for easy access
try:
    from .black_scholes_nifty_assessment import (
        BlackScholesModel,
        MonteCarloOptionPricer, 
        AccuracyAssessment
    )
except ImportError:
    # Handle relative import issues when running as script
    pass
