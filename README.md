# Option Pricing Models — FinSearch 2025

**Comparing Black-Scholes, Binomial Tree, and Monte Carlo simulation for European option valuation on Nifty50 data**

Built as part of **FinSearch 2025**, a quantitative finance research programme run by the Finance Club, IIT Bombay.

---

## Overview

This project implements and benchmarks three classical option pricing models against real Nifty50 options market data. The goal was to understand where each model breaks down, how they diverge near-the-money vs. deep in/out-of-the-money, and what the Greeks tell us about sensitivity to market parameters.

**Models implemented:**
- Black-Scholes (closed-form, with full Greeks)
- Binomial Tree (BOPM — Cox-Ross-Rubinstein)
- Monte Carlo simulation (GBM-based, 1,000+ paths)

---

## Key Results

### Call Option Pricing Comparison (sample contract)

| Model | Price (INR) | vs. Market |
|---|---|---|
| Black-Scholes | 29.51 | — |
| Binomial (BOPM) | 27.65 | — |
| Market Payoff | 9.85 | intrinsic value only |

> The gap between model prices and intrinsic payoff isolates **time value** — a core concept the project explored in depth.

### Delta Comparison (near-the-money)

| Model | Delta |
|---|---|
| Black-Scholes | 0.501 |
| Binomial (BOPM) | 0.495 |

Near-the-money, both models converge closely on delta (~0.5), confirming the theoretical prediction that ATM options have approximately equal probability of expiring in or out of the money.

### Accuracy Metrics (Black-Scholes vs Nifty50 backtest data)

| Metric | Call Options | Put Options |
|---|---|---|
| MAPE | **4.04%** | **3.99%** |
| Correlation (Pearson's R) | **0.9951** | **0.9962** |
| Theil's U | **0.026** | — |

MAPE below 5% and correlation above 0.99 across both calls and puts indicates strong model fit under standard market conditions.

---

## Methodology

### Black-Scholes
Closed-form solution for European options under the assumptions of constant volatility, log-normal returns, and no dividends. Implemented with full Greeks (Δ, Γ, Θ, ν, ρ).

### Binomial Tree (CRR)
Discrete-time lattice model. The stock price at each node evolves with up-factor `u = e^(σ√Δt)` and down-factor `d = 1/u`. Option is valued by backward induction from expiry. Converges to Black-Scholes as the number of steps → ∞.

### Monte Carlo
Simulates 1,000+ stock price paths under Geometric Brownian Motion:

```
S(t+dt) = S(t) · exp((r - σ²/2)dt + σ√dt · Z)
```

where Z ~ N(0,1). Option payoff is averaged across all paths and discounted to present value. Used here primarily as a cross-validation check on the closed-form models.

---

## Repository Structure

```
Option-Pricing-FinSearch-25/
├── README.md
├── requirements.txt
├── config.py                          # Market scenario configurations
├── __init__.py
├── basic_pricing_example.py           # Self-contained BS + Monte Carlo demo
├── black_scholes_nifty_assessment.py  # Full backtest & accuracy pipeline
├── black_scholes_accuracy_results.csv # Generated results
├── nifty50_options_backtest_data.csv  # Nifty50 options dataset
├── monte_carlo_overview.md            # Monte Carlo methodology notes
└── test_implementation.py             # Validation tests (BS, put-call parity, MC)
```

---

## Setup

```bash
git clone https://github.com/SyAkuma/Option-Pricing-FinSearch-25-.git
cd Option-Pricing-FinSearch-25-
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, NumPy, Pandas, SciPy, Matplotlib

---

## Usage

### Run validation tests
```bash
python test_implementation.py
```
Checks: Black-Scholes basic pricing · Put-call parity · Monte Carlo convergence

### Quick pricing demo
```bash
python basic_pricing_example.py
```
Prints model prices, Greeks, and Monte Carlo estimate for a sample Nifty50 scenario.

### Full backtest
```bash
python black_scholes_nifty_assessment.py
```
Runs BS model across the full `nifty50_options_backtest_data.csv` dataset, computes accuracy metrics, and saves results to `black_scholes_accuracy_results.csv`.

---

## Input Data Format

| Column | Description |
|---|---|
| `spot_price` | Current Nifty50 index level |
| `strike_price` | Option strike |
| `time_to_maturity` | Time to expiry (years) |
| `volatility` | Implied volatility (decimal) |
| `risk_free_rate` | Risk-free rate (decimal, e.g. 0.065) |
| `market_call_price` | Observed market call price |
| `market_put_price` | Observed market put price |

---

## Context

Built during **FinSearch 2025** — a quantitative research programme by the Finance Club, IIT Bombay. The project scope covered option theory, model implementation, Greeks interpretation, and empirical validation against real market data.

**Author:** Syed Mohammad Ali · B.Tech EE, IIT Bombay · [github.com/SyAkuma](https://github.com/SyAkuma)
