"""
Shared portfolio metrics for Stock Analysis backtests.

These helpers are intentionally independent of the SEC/fair-value pipeline.
They evaluate the return path after a ranking decision has already been made.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def finite_series(values: Iterable[float]) -> pd.Series:
    return (
        pd.Series(list(values), dtype="float64")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def finite_mean(values: Iterable[float]) -> float:
    series = finite_series(values)
    return float(series.mean()) if not series.empty else np.nan


def finite_median(values: Iterable[float]) -> float:
    series = finite_series(values)
    return float(series.median()) if not series.empty else np.nan


def finite_std(values: Iterable[float]) -> float:
    series = finite_series(values)
    return float(series.std(ddof=0)) if len(series) >= 2 else np.nan


def basket_daily_returns(
    price_wide: pd.DataFrame,
    tickers: Iterable[str],
    anchor: pd.Timestamp,
    future_date: pd.Timestamp,
    weights: dict[str, float] | pd.Series | None = None,
) -> pd.Series:
    columns = []
    seen = set()
    for ticker in tickers:
        clean = str(ticker).upper()
        if clean in seen or clean not in price_wide.columns:
            continue
        seen.add(clean)
        columns.append(clean)
    if not columns:
        return pd.Series(dtype="float64")

    index = pd.to_datetime(price_wide.index).normalize()
    mask = (index >= pd.to_datetime(anchor).normalize()) & (index <= pd.to_datetime(future_date).normalize())
    window = price_wide.loc[mask, columns].astype("float64")
    if len(window) < 2:
        return pd.Series(dtype="float64")

    returns = window.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if weights is None:
        return returns.mean(axis=1, skipna=True).dropna()

    if isinstance(weights, pd.Series):
        raw_weights = weights.copy()
        raw_weights.index = raw_weights.index.astype(str).str.upper()
    else:
        raw_weights = pd.Series(weights, dtype="float64")
        raw_weights.index = raw_weights.index.astype(str).str.upper()

    basket_weights = (
        pd.to_numeric(raw_weights.reindex(columns), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    if basket_weights.sum() <= 0:
        return returns.mean(axis=1, skipna=True).dropna()

    basket_weights = basket_weights / basket_weights.sum()
    weighted_sum = returns.multiply(basket_weights, axis=1).sum(axis=1, skipna=True)
    available_weight = returns.notna().multiply(basket_weights, axis=1).sum(axis=1)
    return (weighted_sum / available_weight.replace(0.0, np.nan)).dropna()


def asset_daily_returns(
    price_wide: pd.DataFrame,
    ticker: str,
    anchor: pd.Timestamp,
    future_date: pd.Timestamp,
) -> pd.Series:
    ticker = str(ticker).upper()
    if ticker not in price_wide.columns:
        return pd.Series(dtype="float64")

    index = pd.to_datetime(price_wide.index).normalize()
    mask = (index >= pd.to_datetime(anchor).normalize()) & (index <= pd.to_datetime(future_date).normalize())
    series = price_wide.loc[mask, ticker].astype("float64")
    if len(series) < 2:
        return pd.Series(dtype="float64")

    return series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()


def cumulative_return(daily_returns: pd.Series) -> float:
    clean = finite_series(daily_returns)
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() - 1.0)


def annualized_volatility(daily_returns: pd.Series) -> float:
    clean = finite_series(daily_returns)
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))


def annualized_sharpe(daily_returns: pd.Series) -> float:
    clean = finite_series(daily_returns)
    if len(clean) < 2:
        return np.nan
    std = clean.std(ddof=0)
    if std <= 0:
        return np.nan
    return float((clean.mean() / std) * math.sqrt(TRADING_DAYS_PER_YEAR))


def annualized_sortino(daily_returns: pd.Series) -> float:
    clean = finite_series(daily_returns)
    downside = clean[clean < 0]
    if len(clean) < 2 or len(downside) < 2:
        return np.nan
    downside_std = downside.std(ddof=0)
    if downside_std <= 0:
        return np.nan
    return float((clean.mean() / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(daily_returns: pd.Series) -> float:
    clean = finite_series(daily_returns)
    if clean.empty:
        return np.nan
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def return_to_drawdown(daily_returns: pd.Series) -> float:
    total_return = cumulative_return(daily_returns)
    drawdown = max_drawdown(daily_returns)
    if not np.isfinite(total_return) or not np.isfinite(drawdown) or drawdown == 0:
        return np.nan
    return float(total_return / abs(drawdown))


def capture_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    direction: str,
) -> float:
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return np.nan

    if direction == "downside":
        aligned = aligned[aligned["benchmark"] < 0]
    elif direction == "upside":
        aligned = aligned[aligned["benchmark"] > 0]
    else:
        raise ValueError("direction must be 'downside' or 'upside'")

    if aligned.empty:
        return np.nan

    strategy_total = cumulative_return(aligned["strategy"])
    benchmark_total = cumulative_return(aligned["benchmark"])
    if not np.isfinite(strategy_total) or not np.isfinite(benchmark_total) or abs(benchmark_total) < 1e-12:
        return np.nan
    return float(strategy_total / benchmark_total)


def tracking_error(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < 2:
        return np.nan

    excess = aligned["strategy"] - aligned["benchmark"]
    return float(excess.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))


def information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < 2:
        return np.nan

    excess = aligned["strategy"] - aligned["benchmark"]
    std = excess.std(ddof=0)
    if std <= 0:
        return np.nan
    return float((excess.mean() / std) * math.sqrt(TRADING_DAYS_PER_YEAR))


def cross_sectional_return_metrics(prefix: str, returns: pd.Series) -> dict:
    clean = finite_series(returns)
    if clean.empty:
        return {
            f"{prefix}_CrossSection_Return_Std": np.nan,
            f"{prefix}_Loss_Rate": np.nan,
            f"{prefix}_Worst_Holding_Return": np.nan,
            f"{prefix}_Best_Holding_Return": np.nan,
        }

    return {
        f"{prefix}_CrossSection_Return_Std": finite_std(clean),
        f"{prefix}_Loss_Rate": float((clean < 0).mean()),
        f"{prefix}_Worst_Holding_Return": float(clean.min()),
        f"{prefix}_Best_Holding_Return": float(clean.max()),
    }


def portfolio_risk_metrics(
    price_wide: pd.DataFrame,
    top_tickers: Iterable[str],
    universe_tickers: Iterable[str],
    anchor: pd.Timestamp,
    future_date: pd.Timestamp,
    benchmark_ticker: str = "XLK",
    top_weights: dict[str, float] | pd.Series | None = None,
    universe_weights: dict[str, float] | pd.Series | None = None,
) -> dict:
    top_daily = basket_daily_returns(price_wide, top_tickers, anchor, future_date, top_weights)
    universe_daily = basket_daily_returns(price_wide, universe_tickers, anchor, future_date, universe_weights)
    benchmark_daily = asset_daily_returns(price_wide, benchmark_ticker, anchor, future_date)

    return {
        "Top_Path_Cumulative_Return": cumulative_return(top_daily),
        "Top_Annualized_Volatility": annualized_volatility(top_daily),
        "Top_Sharpe": annualized_sharpe(top_daily),
        "Top_Sortino": annualized_sortino(top_daily),
        "Top_Max_Drawdown": max_drawdown(top_daily),
        "Top_Return_To_Drawdown": return_to_drawdown(top_daily),
        "Universe_Path_Cumulative_Return": cumulative_return(universe_daily),
        "Universe_Annualized_Volatility": annualized_volatility(universe_daily),
        "Universe_Sharpe": annualized_sharpe(universe_daily),
        "Universe_Max_Drawdown": max_drawdown(universe_daily),
        f"{benchmark_ticker}_Path_Cumulative_Return": cumulative_return(benchmark_daily),
        f"{benchmark_ticker}_Annualized_Volatility": annualized_volatility(benchmark_daily),
        f"{benchmark_ticker}_Sharpe": annualized_sharpe(benchmark_daily),
        f"{benchmark_ticker}_Max_Drawdown": max_drawdown(benchmark_daily),
        f"Top_Tracking_Error_vs_{benchmark_ticker}": tracking_error(top_daily, benchmark_daily),
        f"Top_Information_Ratio_vs_{benchmark_ticker}": information_ratio(top_daily, benchmark_daily),
        f"Top_Downside_Capture_vs_{benchmark_ticker}": capture_ratio(top_daily, benchmark_daily, "downside"),
        f"Top_Upside_Capture_vs_{benchmark_ticker}": capture_ratio(top_daily, benchmark_daily, "upside"),
    }
