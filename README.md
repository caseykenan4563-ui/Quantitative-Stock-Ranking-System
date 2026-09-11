# Quantitative Stock Ranking System

Research code for a technology-stock ranking pipeline that combines peer-relative valuation, price-trend scoring, market-regime logic, and portfolio simulation.

## What This Repository Contains

- Core ranking workflow in `comp.py`
- Peer-relative fair-value scoring
- Price-trend and regime-classification logic
- Historical data-layer adapters
- Point-in-time score/backtest utilities
- Portfolio simulation, construction, robustness, and alpha-quality scripts

## What This Repository Does Not Contain

- API keys or credentials
- `.env` files
- Raw paid vendor datasets
- SEC cache dumps
- Generated Excel workbooks
- Generated backtest result folders
- Portfolio website source code
- Unrelated portfolio, employment, or non-quantitative project material

## Data Requirements

The code expects market/fundamental data to be supplied locally. Vendor CSV paths can be configured with environment variables; raw data files are intentionally excluded from this public repository.

The SEC API requires a descriptive `SEC_USER_AGENT`. Set it before SEC-backed runs:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

For local CSV providers, set the relevant paths as needed:

```bash
export SHARADAR_TICKERS_CSV="/path/to/tickers.csv"
export SHARADAR_STOCKS_CSV="/path/to/stocks.csv"
export SHARADAR_FUNDS_CSV="/path/to/funds.csv"
export SHARADAR_FUNDAMENTALS_CSV="/path/to/fundamentals.csv"
export SHARADAR_DAILY_CSV="/path/to/daily.csv"
export SHARADAR_ACTIONS_CSV="/path/to/actions.csv"
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Entry Points

- `comp.py` builds the current feature table and ranking output.
- `historical_data_layer.py` builds canonical historical price/fundamental layers from local providers.
- `combined_score_backtest.py` tests historical combined-score performance.
- `combined_weight_tuning.py` tunes trend/fair-value/regime weights.
- `monthly_rebalanced_portfolio_simulator.py` runs sequential portfolio simulations with costs and constraints.
- `alpha_signal_quality.py` evaluates candidate alpha features against forward returns.
- `portfolio_robustness_stress_test.py` stress-tests costs, liquidity, slippage, and concentration rules.

## Research Disclaimer

This is a personal research project and educational prototype. It is not financial advice, not a live trading system, and not a recommendation to buy, sell, or hold any security.
