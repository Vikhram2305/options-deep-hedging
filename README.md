# NIFTY Options: Greeks, Delta Hedging & Deep Hedging

A quantitative-finance research notebook for analyzing NIFTY options using **Black-Scholes**, **implied volatility**, **option Greeks**, **delta hedging**, and a neural-network-based **deep hedging** strategy.

The project works with minute-level NIFTY futures and weekly option prices for **5 February 2026**. It first derives market-implied parameters from real option prices, performs an intraday delta-hedging simulation, and then trains a neural network on synthetic Geometric Brownian Motion (GBM) price paths to learn a risk-aware hedging policy.

> **Notebook:** `HFT_CAMP_CODE_(3).ipynb`

## Project Overview

The notebook has three main objectives:

1. **Calculate option Greeks and implied volatility**
   - Black-Scholes option price
   - Delta
   - Gamma
   - Theta
   - Implied volatility using a bisection solver

2. **Perform intraday delta hedging**
   - Uses an at-the-money NIFTY call
   - Recalculates implied volatility and delta at each minute
   - Simulates the resulting hedging P&L through the trading day

3. **Build and evaluate a Deep Hedger**
   - Generates synthetic underlying-price paths using GBM
   - Uses a TensorFlow/Keras neural network to predict hedge deltas
   - Trains the model using a CVaR-based objective
   - Compares Deep Hedging against conventional Black-Scholes delta hedging

## Data

The notebook expects a minute-level CSV containing NIFTY futures and option prices.

The example dataset is:

```text
20260205_option_minute_prices_expiry (1).csv
```

The data contains fields such as:

| Column | Description |
|---|---|
| `date` | Trading date in `YYYYMMDD` format |
| `minute_end` | Time represented as `HHMMSS` |
| `symbol` | NIFTY futures or option symbol |
| `last_trade_price` | Last traded price |

All prices in the dataset are handled in **paisa**:

```text
₹1 = 100 paisa
```

The notebook keeps the calculations in paisa so that futures prices and option strikes use the same units.

## Important Expiry Assumption

For this exercise, the options are treated as expiring at:

```text
5 February 2026, 3:30 PM
```

This is the end of the supplied intraday data.

Although the option symbols indicate a weekly expiry of **10 February 2026**, the notebook intentionally uses 5 February 2026 as the expiry for the calculations. Consequently, the time to expiry at 11:00 AM is approximately **4.5 hours**.

This assumption is important when reproducing the implied-volatility and hedging results.

## Methodology

### 1. Data Preparation

At 11:00 AM, the notebook:

- Filters the dataset to `minute_end == 110000`
- Extracts the NIFTY futures price as the underlying `S`
- Separates futures from option contracts
- Parses option symbols to obtain:
  - Strike price
  - Call/put type
- Converts strikes from rupees to paisa
- Constructs observation timestamps
- Calculates time to expiry
- Uses a risk-free rate of:

```text
r = 5%
```

For the supplied data, the futures price at 11:00 AM is approximately:

```text
₹25,715.10
```

and 22 option contracts are available at that timestamp.

### 2. Black-Scholes Pricing and Greeks

The notebook implements Black-Scholes pricing directly.

For an option with:

- `S` = underlying price
- `K` = strike
- `T` = time to expiry
- `r` = risk-free rate
- `σ` = volatility

it computes:

- Option price
- Delta
- Gamma
- Theta

Both call and put options are supported.

The implementation also handles the expiry case where `T <= 0`.

### 3. Implied Volatility

Implied volatility is obtained by finding the volatility `σ` for which the Black-Scholes price matches the observed market price.

Because there is no closed-form solution for Black-Scholes implied volatility, the notebook uses **bisection**:

```text
Low volatility  = 0.001
High volatility = 5.0
Maximum iterations = 100
Tolerance = 1e-5
```

The calculated IV is then added to the option dataset.

The notebook also plots IV against strike price to visualize the resulting volatility smile/skew.

### 4. Intraday Delta Hedging

The target option for the real-market hedging simulation is:

```text
NIFTY2621025700CE
```

with:

```text
Strike = ₹25,700
Option type = Call
Risk-free rate = 5%
```

The simulation uses the full minute-by-minute dataset.

At each time step:

1. Read the current futures price.
2. Read the current option price.
3. Calculate the current implied volatility.
4. Calculate the Black-Scholes delta using that IV.
5. Hold the resulting delta over the next minute.
6. Calculate the trading P&L from the underlying price change.

The hedging P&L follows:

```text
PL_T = -Z_T + cumulative trading P&L
```

where `Z_T` is the option payoff at expiry.

Transaction costs are assumed to be zero.

### Example Intraday Result

For the supplied run, the dataset contains:

```text
375 price points
09:16:00 → 15:30:00
```

The final underlying price is:

```text
₹25,720.00
```

The option payoff is:

```text
₹20.00
```

The cumulative delta-hedging trading P&L is approximately:

```text
-₹48.76
```

giving a total hedging P&L of approximately:

```text
-₹68.76
```

These figures correspond to the notebook's stated assumptions and should not be interpreted as a live-trading result.

## Deep Hedging

The second part of the project explores whether a neural network can learn a dynamic hedging strategy that focuses on controlling tail risk.

### Synthetic Data Generation

The Deep Hedger is trained using synthetic underlying-price paths generated with **Geometric Brownian Motion (GBM)**.

The simulated paths:

- Start from the initial futures price observed in the real data
- Use the same number of discrete time steps as the real data
- Use a constant volatility of `0.6`

The synthetic paths provide a larger training environment than the single real-market day.

### Train/Test Split

The synthetic paths and their corresponding Black-Scholes quantities are split into:

```text
80% training
20% testing
```

with:

```text
random_state = 42
```

The corresponding Black-Scholes prices and deltas are split using the same partition.

## Deep Hedger Architecture

The model is implemented with **TensorFlow/Keras**.

At each time step, the model receives three inputs:

1. **Log-moneyness**

```text
log(S_t / K)
```

2. **Black-Scholes delta**

```text
BS_delta_t
```

3. **Previous hedge position**

```text
delta_(t-1)
```

The model outputs the new hedge position:

```text
delta_t
```

The architecture consists of:

```text
log(S_t / K) ─────┐
BS_delta_t ───────┼──> Concatenate
delta_(t-1) ──────┘
                       │
                       ▼
                 Dense(32, tanh)
                       │
                       ▼
                 Dense(1, tanh)
                       │
                       ▼
                    delta_t
```

The previous delta is fed back into the model at every time step, allowing the strategy to depend on its previous hedge decision.

## P&L and CVaR Objective

For a simulated price path, the notebook calculates hedging P&L as:

```text
PL_T = -Z_T + cumulative trading P&L
```

with no transaction costs.

Instead of optimizing average P&L directly, the Deep Hedger uses **Conditional Value at Risk (CVaR)** as its training objective.

At a significance level of:

```text
α = 0.05
```

the loss focuses on the worst 5% of P&L outcomes.

Conceptually:

```text
CVaR_α = E[PL | PL <= VaR_α]
```

The implementation converts the worst-tail P&L into a loss that can be minimized during neural-network training.

## Training

The notebook uses:

```text
Optimizer: Adam
Learning rate: 0.001
Epochs: 100
Batch size: 32
CVaR level: 5%
```

Training uses a custom TensorFlow loop with:

- Sequential delta rollout
- P&L calculation
- CVaR loss
- `GradientTape`
- Adam parameter updates

The rollout and training functions are compiled with `tf.function`.

## Evaluation

After training, the notebook evaluates both:

### Deep Hedger

- Train-set P&L
- Test-set P&L
- Mean P&L
- P&L standard deviation
- VaR
- CVaR

### Black-Scholes Delta Hedger

The same metrics are calculated for the conventional Black-Scholes strategy.

The notebook also produces P&L distributions to visually compare the two hedging approaches.

The main objective is to investigate whether the learned strategy can improve tail-risk characteristics relative to standard Black-Scholes delta hedging.

## Technologies

The project is implemented in Python and uses:

- **Python 3**
- **NumPy** — numerical computation
- **Pandas** — data manipulation
- **SciPy** — statistical functions, including the normal distribution
- **Matplotlib** — visualization
- **Seaborn** — plotting
- **Scikit-learn** — train/test splitting
- **TensorFlow / Keras** — Deep Hedger neural network
- **Google Colab** — notebook execution and CSV upload

## Running the Project

### Option 1: Google Colab

Open the notebook in Google Colab and run the cells from top to bottom.

The first cell provides a file-upload widget.

Upload the required CSV when prompted.

The expected workflow is:

```text
1. Upload CSV
      ↓
2. Load and inspect data
      ↓
3. Extract futures/options at 11 AM
      ↓
4. Calculate Black-Scholes Greeks
      ↓
5. Calculate implied volatility
      ↓
6. Plot volatility by strike
      ↓
7. Run minute-by-minute delta hedge
      ↓
8. Generate GBM paths
      ↓
9. Split synthetic data
      ↓
10. Build Deep Hedger
      ↓
11. Train using CVaR
      ↓
12. Evaluate Deep Hedger vs Black-Scholes
```

### Option 2: Local Jupyter Environment

Install the required dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn tensorflow jupyter
```

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
HFT_CAMP_CODE_(3).ipynb
```

and provide the required CSV file.

## Repository Structure

A clean repository can be organized as:

```text
.
├── README.md
├── HFT_CAMP_CODE_(3).ipynb
└── data/
    └── 20260205_option_minute_prices_expiry.csv
```

The CSV is not embedded inside the notebook; it is uploaded into the Colab session when the notebook is executed.

## Key Concepts Demonstrated

This project combines several quantitative-finance and machine-learning concepts:

- Black-Scholes option pricing
- Implied volatility
- Volatility smile/skew
- Delta
- Gamma
- Theta
- Dynamic delta hedging
- Hedging P&L
- Geometric Brownian Motion
- Neural-network hedging
- Sequential decision making
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Tail-risk optimization
- Model comparison on train/test data

## Limitations and Assumptions

This is a research/educational implementation rather than a production trading system.

Important assumptions include:

- The option expiry is manually set to 5 February 2026 at 3:30 PM for this exercise.
- The risk-free rate is fixed at 5%.
- Transaction costs are ignored.
- The Deep Hedger is trained on GBM-generated paths rather than a large historical dataset.
- The GBM volatility is fixed at 0.6.
- The hedging framework uses discrete minute-by-minute rebalancing.
- Market microstructure effects, bid/ask spreads, slippage, liquidity constraints, and execution latency are not modeled.
- The Black-Scholes framework assumes the model inputs and market assumptions used by the notebook.

Because of these assumptions, the results should be treated as an experimental comparison of hedging methodologies, not as evidence of a deployable trading strategy.

## Results Interpretation

The project is designed to answer two related questions:

**Classical hedging:**

> How well does dynamically recalculated Black-Scholes delta hedge an observed NIFTY option during the trading day?

**Deep hedging:**

> Can a neural network trained to minimize tail-risk through CVaR learn a hedge whose P&L/risk profile compares favorably with Black-Scholes delta hedging?

The final comparison should focus on the **test set**, particularly:

- Mean P&L
- P&L volatility
- VaR
- CVaR
- P&L distribution

A lower tail loss / more favorable CVaR is particularly relevant because CVaR is the objective used during Deep Hedger training.

## Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice, investment advice, or a recommendation to trade NIFTY or any other financial instrument.
