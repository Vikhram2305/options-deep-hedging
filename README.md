# NIFTY Options — Delta Hedging & Deep Hedging

A quantitative finance project that compares **classical Black–Scholes delta hedging** with a **neural-network-based Deep Hedging strategy** for NIFTY options.

The project uses minute-level NIFTY futures and options data to calculate option Greeks and implied volatility, perform intraday delta hedging, generate synthetic price paths using Geometric Brownian Motion (GBM), and train a Deep Hedger using a **CVaR-based objective**.

## Project Overview

The project is divided into three main parts:

### 1. Black–Scholes Pricing and Greeks

The project calculates:

* Call and Put option prices
* Delta
* Gamma
* Theta
* Implied Volatility

Implied volatility is calculated numerically using a **bisection method** by matching the Black–Scholes theoretical price with the observed market price.

The project also analyzes implied volatility across different strikes to study the volatility smile/skew.

### 2. Intraday Delta Hedging

A NIFTY call option is dynamically hedged using NIFTY futures.

At each minute:

1. The current futures price is obtained.
2. The current option price is obtained.
3. Implied volatility is calculated.
4. Black–Scholes delta is recalculated.
5. The corresponding hedge position is maintained for the next interval.
6. The trading P&L is accumulated.

The hedging P&L is calculated as:

```text
PL_T = -Option Payoff + Cumulative Trading P&L
```

Transaction costs are not included in the current implementation.

### 3. Deep Hedging

The project then explores whether a neural network can learn a dynamic hedging strategy focused on reducing **tail risk**.

Synthetic underlying-price paths are generated using **Geometric Brownian Motion (GBM)**. A neural network uses market information and the previous hedge position to predict the next hedge position.

The Deep Hedger is then compared with the conventional Black–Scholes delta hedge using:

* Mean P&L
* P&L Standard Deviation
* Value at Risk (VaR)
* Conditional Value at Risk (CVaR)
* P&L Distribution

## Deep Hedger

The Deep Hedger is implemented using TensorFlow/Keras.

At each time step, the model receives three inputs:

```text
log(S_t / K)
BS_delta_t
delta_(t-1)
```

The network architecture is:

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

The previous hedge position is fed back into the next step, allowing the strategy to depend on its previous decision.

## CVaR Objective

Instead of optimizing only average P&L, the Deep Hedger is trained using **Conditional Value at Risk (CVaR)**.

The model focuses on the worst 5% of P&L outcomes:

```text
CVaR level = 5%
```

Conceptually:

```text
CVaR_α = E[PL | PL ≤ VaR_α]
```

This makes the training objective more focused on controlling downside and tail risk.

## Training Configuration

```text
Optimizer       : Adam
Learning Rate   : 0.001
Epochs          : 100
Batch Size      : 32
CVaR Level      : 5%
```

The training process uses TensorFlow's `GradientTape` together with a sequential hedge rollout and custom P&L/CVaR calculations.

## Data

The project uses minute-level NIFTY futures and option prices.

The dataset contains fields such as:

| Column             | Description                |
| ------------------ | -------------------------- |
| `date`             | Trading date               |
| `minute_end`       | Time represented as HHMMSS |
| `symbol`           | Futures or option symbol   |
| `last_trade_price` | Last traded price          |

Prices are handled internally in **paisa**:

```text
₹1 = 100 paisa
```

This keeps the underlying price and option strike values in consistent units.

## Intraday Hedging Instrument

The example hedging experiment uses:

```text
Option : NIFTY2621025700CE
Strike : ₹25,700
Type   : Call
r      : 5%
```

For the supplied experiment:

```text
Price points        : 375
Time range          : 09:16 → 15:30
Final underlying    : ₹25,720
Option payoff       : ₹20
Delta hedge P&L     : approximately -₹48.76
Total hedging P&L   : approximately -₹68.76
```

These results depend on the supplied dataset and the assumptions used in the notebook.

## Synthetic Data Generation

For Deep Hedging, synthetic underlying-price paths are generated using **Geometric Brownian Motion**.

The simulation:

* Starts from the initial futures price observed in the dataset
* Uses the same number of discrete time steps as the market data
* Uses a constant volatility of `0.6`

The simulated paths are split into:

```text
80% Training
20% Testing
```

with:

```text
random_state = 42
```

## Evaluation

Both the Deep Hedger and Black–Scholes delta hedger are evaluated using:

| Metric           | Description                     |
| ---------------- | ------------------------------- |
| Mean P&L         | Average hedging outcome         |
| P&L Std. Dev.    | Variability of hedging outcomes |
| VaR              | Tail loss threshold             |
| CVaR             | Average loss in the worst tail  |
| P&L Distribution | Distribution of hedging results |

The main objective is to investigate whether the Deep Hedger can produce a more favorable **tail-risk profile** than conventional Black–Scholes delta hedging.

## Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Jupyter Notebook

## How to Run

Clone the repository:

```bash
git clone https://github.com/Vikhram2305/options-deep-hedging.git
cd options-deep-hedging
```

Install the required dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn tensorflow jupyter
```

Open the notebook:

```text
HFT_CAMP_CODE_(3).ipynb
```

Run the notebook cells sequentially and provide the required market-data CSV when prompted by the notebook.

## Important Assumptions

The current implementation is an experimental/research project and uses several simplifying assumptions:

* Risk-free rate is fixed at 5%
* Transaction costs are ignored
* Hedging is performed at discrete minute intervals
* Synthetic paths are generated using GBM
* GBM volatility is fixed at 0.6
* The Deep Hedger is trained on synthetic paths
* Bid/ask spreads and slippage are not modeled
* Market impact and liquidity constraints are not modeled
* Execution latency is not modeled

For this experiment, the option is treated as expiring at:

```text
5 February 2026, 3:30 PM
```

This is the end of the supplied intraday data and is used as the expiry convention for the calculations.

## Limitations

This project is intended for **educational and research purposes** rather than production trading.

The Deep Hedger is trained using a simplified GBM market model. Real financial markets contain additional effects such as stochastic volatility, jumps, transaction costs, liquidity constraints, changing market regimes, and market microstructure effects.

Therefore, the results should not be interpreted as evidence that the strategy will perform similarly in live markets.

## Key Concepts

This project demonstrates practical applications of:

* Black–Scholes option pricing
* Implied volatility
* Option Greeks
* Volatility smile/skew
* Dynamic delta hedging
* Hedging P&L
* Geometric Brownian Motion
* Neural-network hedging
* Sequential hedge decisions
* Value at Risk (VaR)
* Conditional Value at Risk (CVaR)
* Tail-risk optimization
* Train/test evaluation

## Disclaimer

This project is for **educational and research purposes only**.

It does not constitute financial advice, investment advice, or a recommendation to trade NIFTY, options, futures, or any other financial instrument.
