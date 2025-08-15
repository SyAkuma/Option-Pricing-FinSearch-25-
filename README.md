# Option Pricing Models & Accuracy Assessment

This repository provides a comprehensive Python framework to:
- Implement the Black-Scholes Model (with Greeks)
- Validate against real or synthetic Nifty50 options data
- Cross-verify using Monte Carlo simulations
- Compute detailed accuracy metrics (MAE, RMSE, MAPE, Theil’s U, correlation)

---

## 📂 Repository Structure

Option-Pricing-FinSearch-25/
├── README.md
├── requirements.txt
├── config.py
├── test_implementation.py
├── basic_pricing_example.py
├── black_scholes_nifty_assessment.py
├── black_scholes_accuracy_results.csv
├── monte_carlo_overview.md
└── nifty50_options_backtest_data.csv

text

- **config.py**: Configuration for different market scenarios  
- **test_implementation.py**: Automated validation tests  
- **basic_pricing_example.py**: Self-contained Black-Scholes & Monte Carlo demo  
- **black_scholes_nifty_assessment.py**: Main backtest & reporting script  
- **black_scholes_accuracy_results.csv**: Generated accuracy results  
- **monte_carlo_overview.md**: Monte Carlo methodology documentation  
- **nifty50_options_backtest_data.csv**: Sample backtesting dataset  

---

## ⚙️ Installation

1. **Clone the repository**  
git clone https://github.com/YourUsername/Option-Pricing-FinSearch-25.git
cd Option-Pricing-FinSearch-25

text
2. **Create and activate a virtual environment**  
python3 -m venv venv
source venv/bin/activate

text
3. **Install dependencies**  
pip install -r requirements.txt

text

---

## 🔎 Usage

### 1. Run Automated Tests

Validate core implementation before analysis:
python test_implementation.py

text
All 3 tests must pass:
- Black-Scholes basic pricing  
- Put-Call parity  
- Monte Carlo convergence  

---

### 2. Demo Example

Quickly compare Black-Scholes vs Monte Carlo on a sample Nifty50 scenario:
python basic_pricing_example.py

text
Outputs:
- Market parameters (spot, strike, days, r, σ)  
- Black-Scholes call & put prices  
- Option Greeks (Δ, Γ, Θ, ν, ρ)  
- Monte Carlo price ± standard error  
- Percentage differences  

---

### 3. Full Backtest & Accuracy Assessment

Run the main analysis on `nifty50_options_backtest_data.csv` or your own data:
python black_scholes_nifty_assessment.py

text
This will:
- Load or generate synthetic data  
- Compute Black-Scholes prices for each contract  
- Calculate error metrics: MAE, RMSE, MAPE, Theil’s U, correlation  
- Perform Monte Carlo cross-validation on a sample contract  
- Save detailed results to `black_scholes_accuracy_results.csv`  

---

## 📄 Data Format

Your CSV must include these columns:

| Column               | Description                         |
|----------------------|-------------------------------------|
| spot_price           | Current Nifty50 index level         |
| strike_price         | Option strike price                 |
| time_to_maturity     | Time to expiration (in years)       |
| volatility           | Implied volatility (decimal)        |
| risk_free_rate       | Risk-free rate (decimal, e.g. 0.065)|
| market_call_price    | Observed call option market price   |
| market_put_price     | Observed put option market price    |
| days_to_expiry       | Time to expiration (in days)        |

---

## 📊 Results Interpretation

- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error (ideal < 5%)  
- **Correlation**: Pearson’s R (excellent > 0.99)  
- **Theil’s U**: Relative accuracy (very good < 0.05)  

Example:
CALL OPTIONS:
Mean Absolute Percentage Error: 4.04%
Correlation: 0.9951
PUT OPTIONS:
Mean Absolute Percentage Error: 3.99%
Correlation: 0.9962