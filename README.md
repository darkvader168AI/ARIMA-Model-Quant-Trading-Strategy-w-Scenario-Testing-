This notebook is a **research/backtesting project for an “institutional-style” systematic equity strategy**. It pulls market data, builds **ARIMA-based price forecasts**, turns those forecasts (plus optional momentum/VWAP/fundamentals signals) into **position-sizing “gates,”** then **simulates portfolios** with **semi-annual rebalances, daily trading adjustments, and transaction costs**, and finally reports **performance + risk metrics** versus benchmarks.

Below is what each section is doing and how the whole pipeline fits together.

---

## 1) Data + “institutional setup” (Cell A)

**Goal:** Build a clean, daily dataset for a defined universe and benchmarks.

* **Universe:** 10 stocks (`SYMS`: NVDA, AMZN, GOOGL, LLY, V, FCX, XOM, CEG, COST, LMT)
* **Benchmarks:** SPY, DIA, QQQ, ACWI
* **Risk-free proxy:** `^IRX` (13-week T-Bill yield)

Key actions:

* Downloads **OHLCV** via `yfinance` from `START = 2010-01-01`.
* Regularizes everything to **business-day frequency** (`.asfreq("B").ffill()`), so the time index is consistent.
* Builds:

  * `px`: adjusted close prices for the stock universe
  * `bench_px`: benchmark prices
  * `rf_daily`: daily risk-free rate derived from `^IRX` (annual % → decimal → daily compounded)
* Creates a **train/test split**:

  * default is `TRAIN_FRAC = 0.9` (last ~10% of history becomes the OOS test window)
  * option to force a fixed last N-day test window via `TEST_DAYS`

Also defines realistic-ish implementation knobs:

* `TX_COST = 0.0002` (0.02% per unit turnover)
* `THRESH_PCT = 0.2` (signal threshold, later used conceptually)
* optional momentum lookback `MOM_LKBK`

---

## 2) ARIMA forecasting engine (Cell B)

**Goal:** For each stock, produce **walk-forward out-of-sample (OOS) forecasts** and diagnostics robustly.

What it does:

* Uses a small grid of ARIMA candidates:
  `CANDIDATES = [(1,1,0), (0,1,1), (1,1,1), (2,1,0)]`
* **Chooses ARIMA order once per ticker** on the training window using **AIC**.
* Runs a **walk-forward OOS forecast** across the test dates:

  * Periodically refits every `REFIT_EVERY` days (default 5) for speed
  * Forecasts “h steps ahead” between refits
* Works in **log prices** (stability) and exponentiates back to price levels.
* Computes and stores:

  * Forecast mean and **95% confidence interval** (`Lo95`, `Hi95`)
  * Actuals
  * OOS error metrics: **MAE, RMSE, MAPE**
  * Training diagnostics: Ljung–Box (autocorr), Jarque–Bera (normality), ARCH LM (heteroskedasticity), AIC/BIC/LLF, convergence flags

Important robustness feature:

* If ARIMA fails to fit/forecast, it falls back to a **simple drift model on log returns** with a 95% band.

**Output:** `fc_df`, `lo_df`, `hi_df`, `act_df` + `metrics_df`, `arima_stats_df`, `latest_fc_df`.

---

## 3) VWAP calculation utilities (Cell C)

**Goal:** Compute a daily VWAP proxy using **Typical Price (H+L+C)/3** and volume.

* Implements `anchored_or_rolling_vwap()`:

  * **Rolling VWAP** over `lookback` business days, or
  * **Anchored VWAP** starting from a chosen anchor (here, the start of the test window if `lookback=None`)

This VWAP is later used as a **mean-reversion / “price vs fair execution level” style feature** (price premium/discount to VWAP).

---

## 4) Baseline portfolio construction + rebalancing simulator (Cell D)

**Goal:** Create “core portfolios” and benchmark curves, with realistic frictions.

It defines:

* Initial capital: `INITIAL_CAPITAL = 100,000`
* Rebalance schedule: every `REBAL_MONTHS = 6`
* Transaction costs applied **at rebalances** using turnover (`TX_COST * gross_turnover * portfolio_value`)

Two baseline portfolio templates:

1. **Equal-weight**
2. **Cap-weight**

   * Uses shares outstanding from yfinance to approximate market cap weights
   * Supports **anchored** cap-weights (fixed at the first rebalance date) vs dynamic

Simulates both baseline portfolios and also builds benchmark equity curves to $100k.

---

## 5) Fundamentals “quality gate” (Cell E)

**Goal:** Add a coarse fundamental filter to avoid obviously weak names (optional).

* Pulls a snapshot of fundamentals from `yfinance.Ticker().info`:

  * profit margins, operating margins, ROE, revenue growth, free cash flow, debt/equity
* Creates a simple **score (0–6)**: +1 for “good” on each metric, +1 for reasonable leverage.
* Keeps either:

  * top `TOP_FUND_FRAC` (default top 70%), or
  * those above `MIN_FUND_SCORE`

This becomes a **static per-ticker gate** (broadcast through time) used in the composite signal scenarios.

---

## 6) Signal construction → “soft gates” → active daily trading between rebalances (Cell F)

**This is the core “strategy logic.”**

### Signals (features)

Built on the **test window** with **no look-ahead** (uses `shift(1)` so decisions use info available at the prior close):

1. **ARIMA expected return**
   [
   \text{exp_ret} = \frac{\text{forecast}}{\text{price_prev}} - 1
   ]
2. **Momentum** (optional)
3. **VWAP gap**
   [
   \text{price_prev}/\text{vwap_prev} - 1
   ]
4. **Fundamentals score** (static cross-section)

### Cross-sectional normalization

Each day, features are converted into **cross-sectional z-scores** across the 10 stocks (winsorized/clipped), i.e. “relative attractiveness today.”

### Composite score and soft gate

* Scenario 1 & 2 use: ARIMA + MOM + VWAP + FUND with weights:

  * `W_ARIMA=0.55, W_MOM=0.20, W_VWAP=0.15, W_FUND=0.10`
* Scenario 3 uses **ARIMA-only**.

Scores are converted into a **continuous gate in [0,1]** using a logistic function:

* Gate near 1 ⇒ keep/overweight the name within the template portfolio
* Gate near 0 ⇒ heavily underweight that name

### Portfolio execution model

* There is still a **6-month “base rebalance template”** (equal-weight or cap-weight).
* **Every trading day**, the template weights are multiplied by the gates and renormalized.
* The simulator trades toward that gated target daily and charges **transaction cost on daily turnover**.

This is basically:
**“Semi-annual strategic allocation + daily tactical tilts driven by model signals.”**

---

## 7) Reporting, benchmarking, and risk/performance attribution (Cell G)

**Goal:** Produce hedge-fund-style performance evaluation for the test window.

Outputs:

* Plot: cumulative $ value for:

  * Scenario 1: Equal-weight + (ARIMA/MOM/VWAP/FUND)
  * Scenario 2: Cap-weight + (ARIMA/MOM/VWAP/FUND)
  * Scenario 3: Equal-weight + ARIMA only
  * plus benchmark curves (SPY/DIA/QQQ/ACWI)

Metrics reported:

* End value, annualized return, annualized vol, Sharpe, max drawdown, win rate
* **CAPM alpha/beta/R² vs SPY**, using the T-bill-derived daily risk-free
* Extra manager metrics:

  * Sortino, Calmar
  * Information ratio & tracking error vs SPY
  * Up-capture / down-capture
  * Gross turnover and trade counts (implementation intensity)

---

## What the project “is,” in one sentence

A **systematic equity strategy research notebook** that uses **ARIMA forecasts** (optionally blended with momentum, VWAP mean-reversion, and fundamentals) to **tilt portfolio weights**, then **backtests** those rules with **transaction costs** and reports **professional performance metrics** against major benchmarks. It’s a long-only, buy-and-sell (active rebalancing) algorithm

---

## What it produces (deliverables)

* Per-stock ARIMA diagnostics + OOS forecast accuracy tables
* Forecast vs actual (with confidence bands) in dataframes
* Three strategy equity curves vs SPY/DIA/QQQ/ACWI
* A consolidated performance summary table (return/risk/alpha/turnover)

