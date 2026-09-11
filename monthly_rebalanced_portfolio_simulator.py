#!/usr/bin/env python3
"""
Sequential monthly portfolio simulator for the Stock Analysis ranking model.

This script turns historical point-in-time rankings into an investable-style
equity curve. It rebalances once per monthly anchor, applies liquidity and
concentration controls, estimates transaction costs and slippage, then measures
the compounded path through time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import combined_weight_tuning as weight_tuning
import historical_data_layer as historical_data
import portfolio_backtest_metrics as portfolio_metrics
import portfolio_construction_validation as construction
import regime_threshold_backtest as regime_bt


DEFAULT_SCORE_HISTORY = Path("backtests/combined_score/point_in_time_scores.csv")
DEFAULT_WEIGHT_TUNING_DIR = Path("backtests/weight_tuning")
DEFAULT_OUTPUT_DIR = Path("backtests/monthly_rebalanced_portfolio")
DEFAULT_TOP_N_LIST = "10,20,50,100"
DEFAULT_MODES = "equal,rank_weighted"
BENCHMARK_TICKERS = ["QQQ", "XLK"]
TRADING_DAYS_PER_YEAR = portfolio_metrics.TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class SimulationParams:
    initial_capital: float = 100_000.0
    rebalance_delay_days: int = 1
    min_dollar_volume: float = 30_000_000.0
    adv_window: int = 20
    max_position_weight: float = 0.15
    max_subindustry_weight: float = 0.35
    max_position_adv_pct: float = 0.02
    max_trade_adv_pct: float = 0.01
    transaction_cost_bps: float = 10.0
    slippage_bps_per_1pct_adv: float = 5.0


@dataclass
class PortfolioState:
    equity: float = 1.0
    weights: dict[str, float] = field(default_factory=dict)
    cash_weight: float = 1.0


@dataclass
class TargetBuild:
    weights: dict[str, float]
    raw_weights: dict[str, float]
    selected: pd.DataFrame
    eligible_count: int
    excluded_by_liquidity: int
    universe_start_count: int
    excluded_by_universe: int
    universe_source: str
    cash_weight: float
    position_cap_bind_count: int
    subindustry_cap_bind_count: int
    adv_cap_bind_count: int


@dataclass(frozen=True)
class DirectScoreConfig:
    config_id: str
    notes: str
    score_column: str
    score_kind: str = "direct_alpha_column"


@dataclass(frozen=True)
class WalkForwardSelectionParams:
    mode: str = "objective"
    min_excess_vs_xlk: float = 0.0
    max_drawdown: float = 0.35
    min_sharpe: float = 0.0
    max_turnover: float | None = 0.70
    recent_fold_count: int = 1
    min_recent_excess_vs_universe: float = 0.0
    min_recent_excess_vs_xlk: float = -0.10
    max_recent_drawdown: float = 0.22
    max_fold_drawdown: float = 0.28
    max_downside_capture_vs_xlk: float = 0.95
    min_positive_universe_fold_rate: float = 0.50
    min_positive_xlk_fold_rate: float = 0.25
    min_family_train_folds: int = 2
    min_family_recent_excess_vs_universe: float = 0.0
    min_family_recent_excess_vs_xlk: float = -0.05
    max_family_worst_drawdown: float = 0.30
    max_family_downside_capture_vs_xlk: float = 0.95
    min_family_positive_universe_fold_rate: float = 0.75
    min_family_positive_xlk_fold_rate: float = 0.50
    family_stability_weight: float = 0.35
    universe_weight: float = 0.60
    qqq_weight: float = 0.40
    xlk_weight: float = 1.20
    sharpe_weight: float = 0.10
    information_ratio_weight: float = 0.05
    drawdown_penalty: float = 0.65
    turnover_penalty: float = 0.05
    recent_universe_weight: float = 0.40
    recent_xlk_weight: float = 0.65
    positive_universe_rate_weight: float = 0.06
    positive_xlk_rate_weight: float = 0.12
    recent_sharpe_weight: float = 0.04
    worst_xlk_lag_penalty: float = 0.35
    worst_universe_lag_penalty: float = 0.20
    downside_capture_penalty: float = 0.18
    fold_drawdown_penalty: float = 0.35


@dataclass(frozen=True)
class FallbackPolicy:
    mode: str = "none"
    clean_guard_stages: tuple[str, ...] = (
        "family_stable_strict_recent_xlk_drawdown_downside",
        "family_stable_recent_defensive_downside",
        "strict_recent_xlk_drawdown_downside",
        "recent_defensive_downside",
    )
    exposure: float = 1.0


@dataclass
class HistoricalUniverse:
    membership: pd.DataFrame
    audit: pd.DataFrame
    source: str
    note: str
    external_path: str | None = None


def fetch_market_wide(
    tickers: list[str],
    start_date: str,
    end_date: str,
    batch_size: int,
    max_retries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import time
    import yfinance as yf

    tickers = sorted(set(str(ticker).upper() for ticker in tickers))
    price_frames: dict[str, pd.Series] = {}
    volume_frames: dict[str, pd.Series] = {}
    failed: list[str] = []

    print(f"[INFO] Fetching market history {start_date} -> {end_date}", flush=True)
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        batch_name = start // batch_size + 1

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"[INFO] Fetching market batch {batch_name} | "
                    f"{len(batch)} tickers | attempt {attempt}",
                    flush=True,
                )
                raw = yf.download(
                    tickers=batch,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if raw.empty:
                    raise ValueError("Yahoo returned empty DataFrame")

                loaded = parse_yahoo_batch(raw, batch)
                for ticker, pair in loaded.items():
                    price_frames[ticker] = pair["price"]
                    volume_frames[ticker] = pair["volume"]

                missing = sorted(set(batch) - set(loaded))
                if missing:
                    failed.extend(missing)
                break
            except Exception as exc:
                if attempt == max_retries:
                    print(f"[WARN] Market batch failed after retries: {batch} | {exc}", flush=True)
                    failed.extend(batch)
                else:
                    time.sleep(0.5 * attempt)

    failed = sorted(set(failed) - set(price_frames))
    if failed and batch_size != 1:
        print(f"[INFO] Retrying {len(failed)} missing market tickers individually", flush=True)
        retry_prices, retry_volumes = fetch_market_wide(
            tickers=failed,
            start_date=start_date,
            end_date=end_date,
            batch_size=1,
            max_retries=max_retries,
        )
        for ticker in retry_prices.columns:
            price_frames[ticker] = retry_prices[ticker].dropna()
        for ticker in retry_volumes.columns:
            volume_frames[ticker] = retry_volumes[ticker].dropna()

    price_wide = pd.DataFrame(price_frames).sort_index()
    volume_wide = pd.DataFrame(volume_frames).sort_index()
    price_wide.index = pd.to_datetime(price_wide.index).normalize()
    volume_wide.index = pd.to_datetime(volume_wide.index).normalize()
    price_wide = price_wide[~price_wide.index.duplicated(keep="last")]
    volume_wide = volume_wide[~volume_wide.index.duplicated(keep="last")]

    print("", flush=True)
    print("=== SANITY CHECK: MARKET FETCH ===", flush=True)
    print(f"Tickers requested: {len(tickers)}", flush=True)
    print(f"Tickers with price data: {len(price_wide.columns)}", flush=True)
    print(f"Tickers with volume data: {len(volume_wide.columns)}", flush=True)
    missing_final = sorted(set(tickers) - set(price_wide.columns))
    if missing_final:
        print(f"[WARN] Missing market data for {len(missing_final)} tickers", flush=True)
        print(f"Sample missing tickers: {missing_final[:20]}", flush=True)
    else:
        print("All tickers returned market data", flush=True)
    print("==================================", flush=True)
    print("", flush=True)

    return price_wide, volume_wide


def parse_yahoo_batch(raw: pd.DataFrame, tickers: list[str]) -> dict[str, dict[str, pd.Series]]:
    loaded: dict[str, dict[str, pd.Series]] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        first_level = set(raw.columns.get_level_values(0))
        for ticker in tickers:
            if ticker not in first_level:
                continue
            ticker_df = raw[ticker]
            pair = extract_price_volume_pair(ticker_df)
            if pair is not None:
                loaded[ticker] = pair
    else:
        pair = extract_price_volume_pair(raw)
        if pair is not None and tickers:
            loaded[tickers[0]] = pair
    return loaded


def extract_price_volume_pair(df: pd.DataFrame) -> dict[str, pd.Series] | None:
    if "Adj Close" in df.columns:
        price = df["Adj Close"]
    elif "Close" in df.columns:
        price = df["Close"]
    else:
        return None

    if "Volume" in df.columns:
        volume = df["Volume"]
    else:
        volume = pd.Series(index=df.index, dtype="float64")

    price = pd.to_numeric(price, errors="coerce").dropna()
    volume = pd.to_numeric(volume, errors="coerce").reindex(price.index)
    if price.empty:
        return None

    return {
        "price": price.astype("float64"),
        "volume": volume.astype("float64"),
    }


def next_trade_date(
    anchor: pd.Timestamp,
    price_index: pd.DatetimeIndex,
    rebalance_delay_days: int,
) -> pd.Timestamp | None:
    delay = max(int(rebalance_delay_days), 0)
    side = "left" if delay == 0 else "right"
    pos = int(price_index.searchsorted(pd.to_datetime(anchor).normalize(), side=side))
    if delay > 1:
        pos += delay - 1
    if pos >= len(price_index):
        return None
    return pd.to_datetime(price_index[pos]).normalize()


def build_rebalance_schedule(
    scored: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    rebalance_delay_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    schedule = []
    for anchor, day in scored.groupby("Date", sort=True):
        anchor = pd.to_datetime(anchor).normalize()
        trade_date = next_trade_date(anchor, price_index, rebalance_delay_days)
        if trade_date is None:
            continue
        schedule.append((anchor, trade_date, day.copy()))
    return schedule


def rolling_dollar_volume(
    price_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    aligned_volume = volume_wide.reindex(index=price_wide.index, columns=price_wide.columns)
    dollar_volume = price_wide * aligned_volume
    return dollar_volume.rolling(window=window, min_periods=max(5, window // 2)).mean().shift(1)


def first_valid_date(series: pd.Series) -> pd.Timestamp | pd.NaT:
    clean = series.dropna()
    if clean.empty:
        return pd.NaT
    return pd.to_datetime(clean.index.min()).normalize()


def last_valid_date(series: pd.Series) -> pd.Timestamp | pd.NaT:
    clean = series.dropna()
    if clean.empty:
        return pd.NaT
    return pd.to_datetime(clean.index.max()).normalize()


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        found = normalized.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def score_price_bounds(scores: pd.DataFrame, price_wide: pd.DataFrame) -> pd.DataFrame:
    score_bounds = (
        scores.groupby("Ticker")["Date"]
        .agg(First_Score_Date="min", Last_Score_Date="max")
        .reset_index()
    )

    price_rows = []
    score_tickers = set(score_bounds["Ticker"])
    for ticker in sorted(score_tickers):
        if ticker not in price_wide.columns:
            price_rows.append({
                "Ticker": ticker,
                "First_Price_Date": pd.NaT,
                "Last_Price_Date": pd.NaT,
            })
            continue
        price_rows.append({
            "Ticker": ticker,
            "First_Price_Date": first_valid_date(price_wide[ticker]),
            "Last_Price_Date": last_valid_date(price_wide[ticker]),
        })

    price_bounds = pd.DataFrame(price_rows)
    bounds = score_bounds.merge(price_bounds, on="Ticker", how="left")
    for col in ["First_Score_Date", "Last_Score_Date", "First_Price_Date", "Last_Price_Date"]:
        bounds[col] = pd.to_datetime(bounds[col], errors="coerce").dt.normalize()
    return bounds


def build_proxy_universe(scores: pd.DataFrame, price_wide: pd.DataFrame) -> HistoricalUniverse:
    bounds = score_price_bounds(scores, price_wide)
    membership = bounds.copy()
    membership["Membership_Start_Date"] = membership[["First_Score_Date", "First_Price_Date"]].max(axis=1)
    membership["Membership_End_Date"] = membership[["Last_Score_Date", "Last_Price_Date"]].min(axis=1)
    membership["Universe_Source"] = "score_price_availability_proxy"
    membership = membership.dropna(subset=["Membership_Start_Date", "Membership_End_Date"]).copy()
    membership = membership[membership["Membership_Start_Date"] <= membership["Membership_End_Date"]].copy()

    audit = bounds.merge(
        membership[["Ticker", "Membership_Start_Date", "Membership_End_Date", "Universe_Source"]],
        on="Ticker",
        how="left",
    )
    audit["Has_External_Membership"] = False
    audit["Survivorship_Status"] = np.where(
        audit["Membership_Start_Date"].notna(),
        "proxy_member_when_score_and_price_are_available",
        "excluded_no_score_price_overlap",
    )
    audit["Survivorship_Limitation"] = (
        "Proxy only: this prevents pre-IPO/no-price periods from entering the simulation, "
        "but it does not restore companies that were removed, acquired, or delisted before "
        "the current ticker universe was built."
    )

    return HistoricalUniverse(
        membership=membership[[
            "Ticker",
            "Membership_Start_Date",
            "Membership_End_Date",
            "Universe_Source",
        ]].copy(),
        audit=audit,
        source="score_price_availability_proxy",
        note=(
            "No external historical universe file was supplied. The simulator used score/price "
            "availability as a proxy universe. This is safer than projecting stocks before they "
            "have data, but unresolved survivorship bias remains."
        ),
    )


def load_external_universe(path: Path, scores: pd.DataFrame, price_wide: pd.DataFrame) -> HistoricalUniverse:
    if not path.exists():
        raise FileNotFoundError(f"Historical universe file not found: {path}")

    raw = pd.read_csv(path)
    ticker_col = find_column(raw.columns, ["Ticker", "Symbol", "ticker", "symbol"])
    if ticker_col is None:
        raise RuntimeError("Historical universe file must include a Ticker or Symbol column")

    start_col = find_column(raw.columns, [
        "Membership_Start_Date",
        "Start_Date",
        "Effective_Start_Date",
        "Added_Date",
        "Date_Added",
        "start",
    ])
    end_col = find_column(raw.columns, [
        "Membership_End_Date",
        "End_Date",
        "Effective_End_Date",
        "Removed_Date",
        "Date_Removed",
        "end",
    ])

    min_score_date = pd.to_datetime(scores["Date"].min()).normalize()
    max_price_date = pd.to_datetime(price_wide.index.max()).normalize()
    membership = pd.DataFrame({
        "Ticker": raw[ticker_col].astype(str).str.upper().str.strip(),
        "Membership_Start_Date": (
            pd.to_datetime(raw[start_col], errors="coerce").dt.normalize()
            if start_col else min_score_date
        ),
        "Membership_End_Date": (
            pd.to_datetime(raw[end_col], errors="coerce").dt.normalize()
            if end_col else max_price_date
        ),
    })
    membership["Membership_Start_Date"] = membership["Membership_Start_Date"].fillna(min_score_date)
    membership["Membership_End_Date"] = membership["Membership_End_Date"].fillna(max_price_date)
    membership["Universe_Source"] = "external_historical_universe"
    membership = membership.dropna(subset=["Ticker"]).copy()
    membership = membership[membership["Ticker"] != ""].copy()
    membership = membership[membership["Membership_Start_Date"] <= membership["Membership_End_Date"]].copy()

    bounds = score_price_bounds(scores, price_wide)
    membership_rollup = (
        membership.groupby("Ticker")
        .agg(
            Membership_Start_Date=("Membership_Start_Date", "min"),
            Membership_End_Date=("Membership_End_Date", "max"),
            External_Membership_Rows=("Ticker", "count"),
        )
        .reset_index()
    )
    audit = bounds.merge(membership_rollup, on="Ticker", how="outer")
    audit["Universe_Source"] = np.where(
        audit["Membership_Start_Date"].notna(),
        "external_historical_universe",
        "external_file_missing_ticker",
    )
    audit["Has_External_Membership"] = audit["Membership_Start_Date"].notna()
    audit["Survivorship_Status"] = "excluded_missing_from_external_universe_file"
    audit.loc[
        audit["Has_External_Membership"] & audit["First_Score_Date"].notna(),
        "Survivorship_Status",
    ] = "external_membership_window_applied"
    audit.loc[
        audit["Has_External_Membership"] & audit["First_Score_Date"].isna(),
        "Survivorship_Status",
    ] = "external_member_but_no_score_rows_available"
    audit["Survivorship_Limitation"] = (
        "External historical membership file supplied. Accuracy depends on whether that file "
        "contains the full investable universe, including removed, acquired, and delisted names."
    )

    return HistoricalUniverse(
        membership=membership[[
            "Ticker",
            "Membership_Start_Date",
            "Membership_End_Date",
            "Universe_Source",
        ]].copy(),
        audit=audit,
        source="external_historical_universe",
        note=(
            "Historical universe membership was loaded from an external file and applied at "
            "each monthly score anchor."
        ),
        external_path=str(path),
    )


def load_historical_universe(
    path: Path | None,
    scores: pd.DataFrame,
    price_wide: pd.DataFrame,
) -> HistoricalUniverse:
    if path is not None:
        return load_external_universe(path, scores, price_wide)
    return build_proxy_universe(scores, price_wide)


def filter_ranked_by_universe(
    ranked: pd.DataFrame,
    anchor: pd.Timestamp,
    universe: HistoricalUniverse | None,
) -> tuple[pd.DataFrame, int, int, str]:
    start_count = int(len(ranked))
    if universe is None or universe.membership.empty:
        return ranked.copy(), start_count, 0, "none"

    anchor = pd.to_datetime(anchor).normalize()
    membership = universe.membership
    valid_rows = membership[
        (membership["Membership_Start_Date"] <= anchor)
        & (membership["Membership_End_Date"] >= anchor)
    ]
    valid_tickers = set(valid_rows["Ticker"].astype(str).str.upper())
    filtered = ranked[ranked["Ticker"].astype(str).str.upper().isin(valid_tickers)].copy()
    return filtered, start_count, int(start_count - len(filtered)), universe.source


def clean_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    out = {}
    for ticker, weight in weights.items():
        value = float(weight)
        if np.isfinite(value) and value > 1e-12:
            out[str(ticker).upper()] = value
    return out


def constrained_weights(
    raw_weights: dict[str, float],
    subindustries: dict[str, str],
    position_caps: dict[str, float],
    max_subindustry_weight: float,
) -> dict[str, float]:
    raw = clean_weight_dict(raw_weights)
    if not raw:
        return {}

    total_raw = sum(raw.values())
    weights = {ticker: weight / total_raw for ticker, weight in raw.items() if total_raw > 0}

    # Enforce individual position caps and redistribute only to uncapped selected names.
    for _ in range(len(weights) + 5):
        capped = {
            ticker: max(0.0, min(1.0, position_caps.get(ticker, 1.0)))
            for ticker in weights
        }
        over = [ticker for ticker, weight in weights.items() if weight > capped[ticker] + 1e-12]
        if not over:
            break

        excess = 0.0
        for ticker in over:
            excess += weights[ticker] - capped[ticker]
            weights[ticker] = capped[ticker]

        under = [
            ticker for ticker, weight in weights.items()
            if weight < capped[ticker] - 1e-12
        ]
        total_room = sum(capped[ticker] - weights[ticker] for ticker in under)
        if total_room <= 1e-12:
            break

        basis = sum(raw[ticker] for ticker in under)
        if basis <= 0:
            break

        for ticker in under:
            room = capped[ticker] - weights[ticker]
            addition = min(room, excess * raw[ticker] / basis)
            weights[ticker] += addition

    # Enforce subindustry caps conservatively. Any excess becomes cash instead
    # of being forced into lower-ranked names after the top-N selection.
    if max_subindustry_weight > 0:
        group_totals: dict[str, float] = {}
        for ticker, weight in weights.items():
            group = subindustries.get(ticker, "Unknown")
            group_totals[group] = group_totals.get(group, 0.0) + weight

        for group, total in group_totals.items():
            if total <= max_subindustry_weight + 1e-12:
                continue
            scale = max_subindustry_weight / total
            for ticker in list(weights):
                if subindustries.get(ticker, "Unknown") == group:
                    weights[ticker] *= scale

    return clean_weight_dict(weights)


def build_target(
    ranked: pd.DataFrame,
    anchor: pd.Timestamp,
    trade_date: pd.Timestamp,
    top_n: int | None,
    mode: str,
    price_wide: pd.DataFrame,
    adv20: pd.DataFrame,
    params: SimulationParams,
    equity_dollars: float,
    historical_universe: HistoricalUniverse | None,
) -> TargetBuild:
    ranked = ranked.sort_values("Tuned_Combined_Score", ascending=False).copy()
    ranked, universe_start_count, excluded_by_universe, universe_source = filter_ranked_by_universe(
        ranked=ranked,
        anchor=anchor,
        universe=historical_universe,
    )
    trade_date = pd.to_datetime(trade_date).normalize()
    price_row = price_wide.loc[trade_date] if trade_date in price_wide.index else pd.Series(dtype="float64")
    adv_row = adv20.loc[trade_date] if trade_date in adv20.index else pd.Series(dtype="float64")

    ranked["Trade_Price"] = pd.to_numeric(ranked["Ticker"].map(price_row), errors="coerce")
    ranked["ADV20_Dollar"] = pd.to_numeric(ranked["Ticker"].map(adv_row), errors="coerce")
    ranked = ranked[np.isfinite(ranked["Trade_Price"])].copy()

    if params.min_dollar_volume > 0:
        eligible = ranked[np.isfinite(ranked["ADV20_Dollar"]) & (ranked["ADV20_Dollar"] >= params.min_dollar_volume)].copy()
    else:
        eligible = ranked.copy()

    selected = eligible.head(top_n).copy() if top_n is not None else eligible.copy()
    if selected.empty:
        return TargetBuild(
            weights={},
            raw_weights={},
            selected=selected,
            eligible_count=int(len(eligible)),
            excluded_by_liquidity=int(len(ranked) - len(eligible)),
            universe_start_count=universe_start_count,
            excluded_by_universe=excluded_by_universe,
            universe_source=universe_source,
            cash_weight=1.0,
            position_cap_bind_count=0,
            subindustry_cap_bind_count=0,
            adv_cap_bind_count=0,
        )

    raw_weights = construction.portfolio_weights(selected["Ticker"], "equal" if top_n is None else mode)
    subindustries = selected.set_index("Ticker")["SubIndustry"].astype(str).to_dict()
    adv_values = selected.set_index("Ticker")["ADV20_Dollar"].to_dict()

    position_caps = {}
    adv_cap_bind_count = 0
    for ticker in raw_weights:
        cap = float(params.max_position_weight) if params.max_position_weight > 0 else 1.0
        adv = adv_values.get(ticker, np.nan)
        if params.max_position_adv_pct > 0 and np.isfinite(adv) and equity_dollars > 0:
            adv_cap = (float(adv) * params.max_position_adv_pct) / equity_dollars
            if adv_cap < cap:
                adv_cap_bind_count += 1
            cap = min(cap, adv_cap)
        position_caps[ticker] = max(0.0, min(1.0, cap))

    weights = constrained_weights(
        raw_weights=raw_weights,
        subindustries=subindustries,
        position_caps=position_caps,
        max_subindustry_weight=params.max_subindustry_weight,
    )
    selected["Raw_Weight"] = selected["Ticker"].map(raw_weights).fillna(0.0)
    selected["Target_Weight"] = selected["Ticker"].map(weights).fillna(0.0)
    selected["Position_Cap"] = selected["Ticker"].map(position_caps).fillna(0.0)
    selected["Target_Position_ADV_Pct"] = (
        selected["Target_Weight"] * equity_dollars / selected["ADV20_Dollar"]
    ).replace([np.inf, -np.inf], np.nan)

    position_cap_bind_count = int((selected["Target_Weight"] < selected["Raw_Weight"] - 1e-9).sum())
    group_weights = selected.groupby("SubIndustry")["Target_Weight"].sum()
    subindustry_cap_bind_count = int((group_weights >= params.max_subindustry_weight - 1e-9).sum())

    return TargetBuild(
        weights=weights,
        raw_weights=raw_weights,
        selected=selected,
        eligible_count=int(len(eligible)),
        excluded_by_liquidity=int(len(ranked) - len(eligible)),
        universe_start_count=universe_start_count,
        excluded_by_universe=excluded_by_universe,
        universe_source=universe_source,
        cash_weight=max(0.0, 1.0 - sum(weights.values())),
        position_cap_bind_count=position_cap_bind_count,
        subindustry_cap_bind_count=subindustry_cap_bind_count,
        adv_cap_bind_count=adv_cap_bind_count,
    )


def fallback_target(
    fallback_mode: str,
    trade_date: pd.Timestamp,
    price_wide: pd.DataFrame,
    equity_dollars: float,
    exposure: float = 1.0,
) -> tuple[TargetBuild, str, str]:
    mode = str(fallback_mode).lower().strip()
    exposure = max(0.0, min(1.0, float(exposure)))
    trade_date = pd.to_datetime(trade_date).normalize()

    if mode in {"none", "cash", ""}:
        return (
            TargetBuild(
                weights={},
                raw_weights={},
                selected=pd.DataFrame(),
                eligible_count=0,
                excluded_by_liquidity=0,
                universe_start_count=0,
                excluded_by_universe=0,
                universe_source="fallback_cash",
                cash_weight=1.0,
                position_cap_bind_count=0,
                subindustry_cap_bind_count=0,
                adv_cap_bind_count=0,
            ),
            "cash",
            "cash_fallback",
        )

    asset_map = {
        "qqq": "QQQ",
        "xlk": "XLK",
    }
    asset = asset_map.get(mode)
    if asset is None:
        raise ValueError(f"Unsupported fallback mode: {fallback_mode}")

    price = (
        price_wide.at[trade_date, asset]
        if trade_date in price_wide.index and asset in price_wide.columns
        else np.nan
    )
    if not np.isfinite(price):
        return (
            TargetBuild(
                weights={},
                raw_weights={},
                selected=pd.DataFrame(),
                eligible_count=0,
                excluded_by_liquidity=0,
                universe_start_count=1,
                excluded_by_universe=0,
                universe_source=f"fallback_{asset.lower()}_unavailable",
                cash_weight=1.0,
                position_cap_bind_count=0,
                subindustry_cap_bind_count=0,
                adv_cap_bind_count=0,
            ),
            "cash",
            f"{asset}_price_unavailable_cash_fallback",
        )

    selected = pd.DataFrame([{
        "Ticker": asset,
        "SubIndustry": "Benchmark ETF",
        "Tuned_Combined_Score": np.nan,
        "Trade_Price": float(price),
        "Raw_Weight": exposure,
        "Target_Weight": exposure,
        "Position_Cap": 1.0,
        "Target_Position_ADV_Pct": np.nan,
    }])
    return (
        TargetBuild(
            weights={asset: exposure} if exposure > 0 else {},
            raw_weights={asset: exposure} if exposure > 0 else {},
            selected=selected,
            eligible_count=1,
            excluded_by_liquidity=0,
            universe_start_count=1,
            excluded_by_universe=0,
            universe_source=f"fallback_{asset.lower()}",
            cash_weight=max(0.0, 1.0 - exposure),
            position_cap_bind_count=0,
            subindustry_cap_bind_count=0,
            adv_cap_bind_count=0,
        ),
        asset,
        f"{asset}_fallback",
    )


def exposure_label_fields(
    fallback_triggered: bool,
    fallback_asset: str,
    selected_config_id: str,
    selected_top_n: int,
    selected_mode: str,
    selection_guard_stage: str,
) -> dict[str, object]:
    asset = str(fallback_asset or "").strip().upper()
    if fallback_triggered:
        if asset in {"QQQ", "XLK"}:
            source = "benchmark_etf_fallback"
            label = f"{asset} benchmark ETF fallback"
            benchmark_fallback = True
            cash_fallback = False
        elif asset in {"", "NONE", "CASH"}:
            source = "cash_fallback"
            label = "Cash fallback"
            benchmark_fallback = False
            cash_fallback = True
        else:
            source = "other_fallback"
            label = f"{asset} fallback"
            benchmark_fallback = False
            cash_fallback = False
        detail = (
            f"{label}: defensive selector stage {selection_guard_stage} did not clear "
            "the clean guard stages, so stock-selection candidates were bypassed."
        )
        return {
            "Exposure_Source": source,
            "Exposure_Label": label,
            "Exposure_Detail": detail,
            "Stock_Selection_Active": False,
            "Benchmark_Fallback_Active": benchmark_fallback,
            "Cash_Fallback_Active": cash_fallback,
        }

    return {
        "Exposure_Source": "stock_selection_alpha",
        "Exposure_Label": "Stock-selection alpha",
        "Exposure_Detail": (
            f"Stock-selection alpha: {selected_config_id} / top {selected_top_n} / "
            f"{selected_mode}; guard stage {selection_guard_stage}."
        ),
        "Stock_Selection_Active": True,
        "Benchmark_Fallback_Active": False,
        "Cash_Fallback_Active": False,
    }


def parse_score_columns(raw: str) -> list[str]:
    columns = []
    for item in raw.split(","):
        column = item.strip()
        if column and column not in columns:
            columns.append(column)
    return columns


def parse_stage_list(raw: str) -> tuple[str, ...]:
    stages = []
    for item in raw.split(","):
        stage = item.strip()
        if stage and stage not in stages:
            stages.append(stage)
    return tuple(stages)


def score_column_config_id(column: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in column).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"alpha_{safe}"


def direct_score_frame(scores: pd.DataFrame, config: DirectScoreConfig) -> pd.DataFrame:
    if config.score_column not in scores.columns:
        raise RuntimeError(f"Direct score column not found: {config.score_column}")

    df = scores.copy()
    df["Tuned_Combined_Score"] = pd.to_numeric(df[config.score_column], errors="coerce")
    df = df[np.isfinite(df["Tuned_Combined_Score"])].copy()
    if df.empty:
        return df

    if "Trend_Score_100" not in df.columns:
        if "Stock_Price_Trend_Score" in df.columns:
            df["Trend_Score_100"] = pd.to_numeric(df["Stock_Price_Trend_Score"], errors="coerce")
        elif "Price_Trend_Score" in df.columns:
            trend = pd.to_numeric(df["Price_Trend_Score"], errors="coerce")
            df["Trend_Score_100"] = trend.where(trend > 1.0, trend * 100.0)
        else:
            df["Trend_Score_100"] = np.nan

    df["Tuned_Fair_Value_Score"] = pd.to_numeric(df.get("Fair_Value_Score"), errors="coerce")
    df["Trend_Weight"] = np.nan
    df["Fair_Value_Weight"] = np.nan
    df["Direct_Score_Source_Column"] = config.score_column
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def score_frame_for_config(
    scores: pd.DataFrame,
    config: weight_tuning.WeightConfig | DirectScoreConfig,
) -> pd.DataFrame:
    if isinstance(config, DirectScoreConfig):
        return direct_score_frame(scores, config)
    return weight_tuning.config_score_frame(scores, config)


def add_alpha_score_configs(
    scores: pd.DataFrame,
    configs: list[weight_tuning.WeightConfig | DirectScoreConfig],
    score_columns: list[str],
    price_wide: pd.DataFrame,
) -> tuple[pd.DataFrame, list[weight_tuning.WeightConfig | DirectScoreConfig]]:
    if not score_columns:
        return scores, configs

    import alpha_signal_quality

    enriched = alpha_signal_quality.attach_ranked_signals(scores, price_wide)
    missing = sorted(set(score_columns) - set(enriched.columns))
    if missing:
        raise RuntimeError(f"Alpha score columns were not generated: {missing}")

    augmented = list(configs)
    existing_ids = {config.config_id for config in augmented}
    for column in score_columns:
        config_id = score_column_config_id(column)
        if config_id in existing_ids:
            continue
        augmented.append(DirectScoreConfig(
            config_id=config_id,
            score_column=column,
            notes=(
                "Research-only alpha candidate. The simulator ranks directly by this "
                "precomputed point-in-time score column and keeps all portfolio mechanics unchanged."
            ),
        ))
        existing_ids.add(config_id)
    return enriched, augmented


def daily_return_and_drift(
    state: PortfolioState,
    asset_returns: pd.Series,
) -> float:
    if not state.weights:
        return 0.0

    growth_values = {}
    total_growth = state.cash_weight
    for ticker, weight in state.weights.items():
        value = asset_returns.get(ticker, np.nan)
        ret = float(value) if pd.notna(value) and np.isfinite(value) else 0.0
        grown = weight * (1.0 + ret)
        growth_values[ticker] = grown
        total_growth += grown

    if total_growth <= 0:
        state.weights = {}
        state.cash_weight = 1.0
        return -1.0

    state.weights = {
        ticker: value / total_growth
        for ticker, value in growth_values.items()
        if value > 1e-12
    }
    state.cash_weight = state.cash_weight / total_growth
    state.equity *= total_growth
    return float(total_growth - 1.0)


def rebalance_portfolio(
    state: PortfolioState,
    target_weights: dict[str, float],
    trade_date: pd.Timestamp,
    adv20: pd.DataFrame,
    params: SimulationParams,
) -> dict:
    target = clean_weight_dict(target_weights)
    target_cash = max(0.0, 1.0 - sum(target.values()))
    current = clean_weight_dict(state.weights)
    current_cash = max(0.0, state.cash_weight)
    tickers = sorted(set(target) | set(current))

    delta = {ticker: target.get(ticker, 0.0) - current.get(ticker, 0.0) for ticker in tickers}
    gross_trade_weight = sum(abs(value) for value in delta.values())
    turnover_value = 0.5 * (
        gross_trade_weight
        + abs(target_cash - current_cash)
    )

    trade_date = pd.to_datetime(trade_date).normalize()
    adv_row = adv20.loc[trade_date] if trade_date in adv20.index else pd.Series(dtype="float64")
    equity_dollars = state.equity * params.initial_capital
    transaction_cost = gross_trade_weight * params.transaction_cost_bps / 10000.0
    slippage_cost = 0.0
    trade_adv_pcts = []
    trade_adv_breach_count = 0

    for ticker, trade_weight in delta.items():
        if abs(trade_weight) <= 1e-12:
            continue
        adv = adv_row.get(ticker, np.nan)
        if not np.isfinite(adv) or adv <= 0:
            continue
        trade_adv_pct = abs(trade_weight) * equity_dollars / adv
        trade_adv_pcts.append(float(trade_adv_pct))
        if params.max_trade_adv_pct > 0 and trade_adv_pct > params.max_trade_adv_pct:
            trade_adv_breach_count += 1
        slippage_bps = params.slippage_bps_per_1pct_adv * (trade_adv_pct / 0.01)
        slippage_cost += abs(trade_weight) * slippage_bps / 10000.0

    total_cost = min(0.99, transaction_cost + slippage_cost)
    state.equity *= 1.0 - total_cost
    state.weights = target
    state.cash_weight = target_cash

    return {
        "Turnover": float(turnover_value),
        "Gross_Trade_Weight": float(gross_trade_weight),
        "Transaction_Cost": float(transaction_cost),
        "Slippage_Cost": float(slippage_cost),
        "Total_Cost": float(total_cost),
        "Max_Trade_ADV_Pct": max(trade_adv_pcts) if trade_adv_pcts else np.nan,
        "Avg_Trade_ADV_Pct": float(np.mean(trade_adv_pcts)) if trade_adv_pcts else np.nan,
        "Trade_ADV_Breach_Count": int(trade_adv_breach_count),
    }


def calculate_performance_metrics(daily_returns: pd.Series) -> dict:
    clean = portfolio_metrics.finite_series(daily_returns)
    if clean.empty:
        return {
            "Total_Return": np.nan,
            "CAGR": np.nan,
            "Annualized_Volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Max_Drawdown": np.nan,
            "Calmar": np.nan,
            "Daily_Win_Rate": np.nan,
        }

    total_return = portfolio_metrics.cumulative_return(clean)
    years = len(clean) / TRADING_DAYS_PER_YEAR
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 and total_return > -1.0 else np.nan
    drawdown = portfolio_metrics.max_drawdown(clean)
    return {
        "Total_Return": total_return,
        "CAGR": cagr,
        "Annualized_Volatility": portfolio_metrics.annualized_volatility(clean),
        "Sharpe": portfolio_metrics.annualized_sharpe(clean),
        "Sortino": portfolio_metrics.annualized_sortino(clean),
        "Max_Drawdown": drawdown,
        "Calmar": cagr / abs(drawdown) if np.isfinite(cagr) and np.isfinite(drawdown) and drawdown != 0 else np.nan,
        "Daily_Win_Rate": float((clean > 0).mean()),
    }


def summarize_simulation(daily: pd.DataFrame, rebalances: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["Config_ID", "Top_N", "Portfolio_Mode"]
    for key, group in daily.groupby(group_cols, dropna=False):
        config_id, top_n, mode = key
        strategy_returns = group["Strategy_Daily_Return"].astype(float)
        universe_returns = group["Universe_Daily_Return"].astype(float)
        qqq_returns = group["QQQ_Daily_Return"].astype(float)
        xlk_returns = group["XLK_Daily_Return"].astype(float)
        strategy_metrics = calculate_performance_metrics(strategy_returns)
        universe_metrics = calculate_performance_metrics(universe_returns)
        qqq_metrics = calculate_performance_metrics(qqq_returns)
        xlk_metrics = calculate_performance_metrics(xlk_returns)
        strategy_final_equity = (
            1.0 + strategy_metrics["Total_Return"]
            if np.isfinite(strategy_metrics["Total_Return"]) else np.nan
        )
        universe_final_equity = (
            1.0 + universe_metrics["Total_Return"]
            if np.isfinite(universe_metrics["Total_Return"]) else np.nan
        )
        qqq_final_equity = (
            1.0 + qqq_metrics["Total_Return"]
            if np.isfinite(qqq_metrics["Total_Return"]) else np.nan
        )
        xlk_final_equity = (
            1.0 + xlk_metrics["Total_Return"]
            if np.isfinite(xlk_metrics["Total_Return"]) else np.nan
        )

        if rebalances.empty or not {"Config_ID", "Top_N", "Portfolio_Mode"}.issubset(rebalances.columns):
            rebalance_group = pd.DataFrame()
        else:
            rebalance_group = rebalances[
                (rebalances["Config_ID"] == config_id)
                & (rebalances["Top_N"] == top_n)
                & (rebalances["Portfolio_Mode"] == mode)
            ]

        tracking_error_xlk = portfolio_metrics.tracking_error(strategy_returns, xlk_returns)
        information_ratio_xlk = portfolio_metrics.information_ratio(strategy_returns, xlk_returns)

        rows.append({
            "Config_ID": config_id,
            "Top_N": int(top_n),
            "Portfolio_Mode": mode,
            "Trading_Days": int(len(group)),
            "Start_Date": group["Date"].min(),
            "End_Date": group["Date"].max(),
            "Final_Equity": float(strategy_final_equity),
            "Universe_Final_Equity": float(universe_final_equity),
            "QQQ_Final_Equity": float(qqq_final_equity),
            "XLK_Final_Equity": float(xlk_final_equity),
            "Total_Return": strategy_metrics["Total_Return"],
            "Universe_Total_Return": universe_metrics["Total_Return"],
            "QQQ_Total_Return": qqq_metrics["Total_Return"],
            "XLK_Total_Return": xlk_metrics["Total_Return"],
            "Excess_Return_vs_Universe": strategy_metrics["Total_Return"] - universe_metrics["Total_Return"],
            "Excess_Return_vs_QQQ": strategy_metrics["Total_Return"] - qqq_metrics["Total_Return"],
            "Excess_Return_vs_XLK": strategy_metrics["Total_Return"] - xlk_metrics["Total_Return"],
            "CAGR": strategy_metrics["CAGR"],
            "Universe_CAGR": universe_metrics["CAGR"],
            "QQQ_CAGR": qqq_metrics["CAGR"],
            "XLK_CAGR": xlk_metrics["CAGR"],
            "Annualized_Volatility": strategy_metrics["Annualized_Volatility"],
            "Sharpe": strategy_metrics["Sharpe"],
            "Sortino": strategy_metrics["Sortino"],
            "Max_Drawdown": strategy_metrics["Max_Drawdown"],
            "Calmar": strategy_metrics["Calmar"],
            "Daily_Win_Rate": strategy_metrics["Daily_Win_Rate"],
            "Tracking_Error_vs_XLK": tracking_error_xlk,
            "Information_Ratio_vs_XLK": information_ratio_xlk,
            "Downside_Capture_vs_XLK": portfolio_metrics.capture_ratio(strategy_returns, xlk_returns, "downside"),
            "Upside_Capture_vs_XLK": portfolio_metrics.capture_ratio(strategy_returns, xlk_returns, "upside"),
            "Rebalance_Count": int(len(rebalance_group)),
            "Avg_Turnover": float(rebalance_group["Strategy_Turnover"].mean()) if not rebalance_group.empty else np.nan,
            "Total_Transaction_Cost": float(rebalance_group["Strategy_Transaction_Cost"].sum()) if not rebalance_group.empty else np.nan,
            "Total_Slippage_Cost": float(rebalance_group["Strategy_Slippage_Cost"].sum()) if not rebalance_group.empty else np.nan,
            "Total_Cost": float(rebalance_group["Strategy_Total_Cost"].sum()) if not rebalance_group.empty else np.nan,
            "Avg_Cash_Weight": float(group["Strategy_Cash_Weight"].mean()),
            "Max_Cash_Weight": float(group["Strategy_Cash_Weight"].max()),
            "Avg_Holdings": float(rebalance_group["Strategy_Holdings"].mean()) if not rebalance_group.empty else np.nan,
            "Avg_Excluded_By_Universe": float(rebalance_group["Strategy_Excluded_By_Universe"].mean()) if not rebalance_group.empty and "Strategy_Excluded_By_Universe" in rebalance_group.columns else np.nan,
            "Avg_Excluded_By_Liquidity": float(rebalance_group["Strategy_Excluded_By_Liquidity"].mean()) if not rebalance_group.empty else np.nan,
            "Position_Cap_Bind_Rebalances": int((rebalance_group["Strategy_Position_Cap_Bind_Count"] > 0).sum()) if not rebalance_group.empty else 0,
            "Subindustry_Cap_Bind_Rebalances": int((rebalance_group["Strategy_Subindustry_Cap_Bind_Count"] > 0).sum()) if not rebalance_group.empty else 0,
            "ADV_Cap_Bind_Rebalances": int((rebalance_group["Strategy_ADV_Cap_Bind_Count"] > 0).sum()) if not rebalance_group.empty else 0,
            "Trade_ADV_Breach_Rebalances": int((rebalance_group["Strategy_Trade_ADV_Breach_Count"] > 0).sum()) if not rebalance_group.empty else 0,
        })

    summary = pd.DataFrame(rows)
    summary["Simulator_Objective"] = (
        summary["Excess_Return_vs_Universe"].fillna(0.0)
        + 0.50 * summary["Excess_Return_vs_XLK"].fillna(0.0)
        + 0.05 * summary["Sharpe"].fillna(0.0)
        + 0.02 * summary["Information_Ratio_vs_XLK"].fillna(0.0)
        - 0.10 * summary["Max_Drawdown"].abs().fillna(0.0)
        - 0.02 * summary["Avg_Turnover"].fillna(0.0)
    )
    return summary.sort_values("Simulator_Objective", ascending=False).round(6)


def assign_fold_labels(
    frame: pd.DataFrame,
    date_col: str,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty or date_col not in frame.columns:
        return frame
    out = frame.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out["Fold"] = [
        regime_bt.fold_label(date, end_date) if pd.notna(date) else "Unknown"
        for date in dates
    ]
    return out


def filter_simulation_variant(frame: pd.DataFrame, selected: pd.Series | dict) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return frame[
        (frame["Config_ID"] == selected["Config_ID"])
        & (frame["Top_N"].astype(int) == int(selected["Top_N"]))
        & (frame["Portfolio_Mode"] == selected["Portfolio_Mode"])
    ].copy()


def metric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def ordered_fold_labels(folds: Iterable[str]) -> list[str]:
    fold_rank = {fold: idx for idx, fold in enumerate(weight_tuning.FOLD_ORDER)}
    return sorted(
        {str(fold) for fold in folds if pd.notna(fold)},
        key=lambda fold: (fold_rank.get(fold, len(fold_rank)), fold),
    )


def candidate_family_id(config_id: object) -> str:
    config_text = str(config_id)
    if config_text.startswith("alpha_"):
        return config_text
    if config_text.startswith("grid_"):
        return "regime_weight_grid"
    return config_text


def add_candidate_family_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "Candidate_Family_ID" not in out.columns:
        out["Candidate_Family_ID"] = out["Config_ID"].map(candidate_family_id)
    return out


def add_recent_fold_consistency_columns(
    summary: pd.DataFrame,
    train_daily: pd.DataFrame,
    train_rebalances: pd.DataFrame,
    train_folds: list[str],
    params: WalkForwardSelectionParams,
) -> pd.DataFrame:
    if summary.empty or train_daily.empty:
        return summary

    fold_summaries: list[pd.DataFrame] = []
    for fold in ordered_fold_labels(train_folds):
        fold_daily = train_daily[train_daily["Fold"] == fold].copy()
        if fold_daily.empty:
            continue
        if train_rebalances.empty or "Fold" not in train_rebalances.columns:
            fold_rebalances = pd.DataFrame()
        else:
            fold_rebalances = train_rebalances[train_rebalances["Fold"] == fold].copy()
        fold_summary = summarize_simulation(fold_daily, fold_rebalances)
        if fold_summary.empty:
            continue
        fold_summary["Training_Fold"] = fold
        fold_summaries.append(fold_summary)

    if not fold_summaries:
        return summary

    per_fold = pd.concat(fold_summaries, ignore_index=True)
    fold_rank = {fold: idx for idx, fold in enumerate(weight_tuning.FOLD_ORDER)}
    per_fold["_Fold_Order"] = per_fold["Training_Fold"].map(lambda fold: fold_rank.get(str(fold), len(fold_rank)))

    rows: list[dict] = []
    group_cols = ["Config_ID", "Top_N", "Portfolio_Mode"]
    for key, group in per_fold.groupby(group_cols, dropna=False):
        config_id, top_n, mode = key
        group = group.sort_values(["_Fold_Order", "Training_Fold"]).copy()
        recent_count = max(1, int(params.recent_fold_count))
        recent = group.tail(recent_count)

        rows.append({
            "Config_ID": config_id,
            "Top_N": int(top_n),
            "Portfolio_Mode": mode,
            "Train_Fold_Count": int(len(group)),
            "Recent_Training_Folds": ",".join(str(fold) for fold in recent["Training_Fold"].tolist()),
            "Recent_Fold_Total_Return": float(recent["Total_Return"].mean()),
            "Recent_Fold_Excess_Return_vs_Universe": float(recent["Excess_Return_vs_Universe"].mean()),
            "Recent_Fold_Excess_Return_vs_QQQ": float(recent["Excess_Return_vs_QQQ"].mean()),
            "Recent_Fold_Excess_Return_vs_XLK": float(recent["Excess_Return_vs_XLK"].mean()),
            "Recent_Fold_Sharpe": float(recent["Sharpe"].mean()),
            "Recent_Fold_Max_Drawdown": float(recent["Max_Drawdown"].min()),
            "Recent_Fold_Downside_Capture_vs_XLK": float(recent["Downside_Capture_vs_XLK"].mean()),
            "Worst_Fold_Excess_Return_vs_Universe": float(group["Excess_Return_vs_Universe"].min()),
            "Worst_Fold_Excess_Return_vs_QQQ": float(group["Excess_Return_vs_QQQ"].min()),
            "Worst_Fold_Excess_Return_vs_XLK": float(group["Excess_Return_vs_XLK"].min()),
            "Positive_Universe_Fold_Rate": float((group["Excess_Return_vs_Universe"] >= 0.0).mean()),
            "Positive_QQQ_Fold_Rate": float((group["Excess_Return_vs_QQQ"] >= 0.0).mean()),
            "Positive_XLK_Fold_Rate": float((group["Excess_Return_vs_XLK"] >= 0.0).mean()),
            "Worst_Fold_Drawdown": float(group["Max_Drawdown"].min()),
            "Average_Downside_Capture_vs_XLK": float(group["Downside_Capture_vs_XLK"].mean()),
        })

    consistency = pd.DataFrame(rows)
    out = summary.merge(consistency, on=group_cols, how="left")
    return out


def add_candidate_family_stability_columns(
    summary: pd.DataFrame,
    train_daily: pd.DataFrame,
    train_rebalances: pd.DataFrame,
    train_folds: list[str],
    params: WalkForwardSelectionParams,
) -> pd.DataFrame:
    out = add_candidate_family_column(summary)
    if out.empty or train_daily.empty:
        return out

    fold_summaries: list[pd.DataFrame] = []
    for fold in ordered_fold_labels(train_folds):
        fold_daily = train_daily[train_daily["Fold"] == fold].copy()
        if fold_daily.empty:
            continue
        if train_rebalances.empty or "Fold" not in train_rebalances.columns:
            fold_rebalances = pd.DataFrame()
        else:
            fold_rebalances = train_rebalances[train_rebalances["Fold"] == fold].copy()
        fold_summary = summarize_simulation(fold_daily, fold_rebalances)
        if fold_summary.empty:
            continue
        fold_summary["Training_Fold"] = fold
        fold_summaries.append(fold_summary)

    if not fold_summaries:
        return out

    per_fold = pd.concat(fold_summaries, ignore_index=True)
    per_fold = add_candidate_family_column(per_fold)
    fold_rank = {fold: idx for idx, fold in enumerate(weight_tuning.FOLD_ORDER)}
    per_fold["_Fold_Order"] = per_fold["Training_Fold"].map(lambda fold: fold_rank.get(str(fold), len(fold_rank)))
    per_fold = add_selection_guard_columns(per_fold, params)

    best_per_family_fold = (
        per_fold
        .sort_values(
            ["Candidate_Family_ID", "_Fold_Order", "Defensive_Simulator_Objective"],
            ascending=[True, True, False],
        )
        .groupby(["Candidate_Family_ID", "Training_Fold"], dropna=False)
        .head(1)
        .copy()
    )

    rows: list[dict] = []
    for family_id, group in best_per_family_fold.groupby("Candidate_Family_ID", dropna=False):
        group = group.sort_values(["_Fold_Order", "Training_Fold"]).copy()
        recent_count = max(1, int(params.recent_fold_count))
        recent = group.tail(recent_count)
        best_constructions = [
            (
                f"{row.Training_Fold}:"
                f"{row.Config_ID}/top{int(row.Top_N)}/{row.Portfolio_Mode}"
            )
            for row in group.itertuples(index=False)
        ]

        rows.append({
            "Candidate_Family_ID": family_id,
            "Family_Train_Fold_Count": int(len(group)),
            "Family_Recent_Training_Folds": ",".join(str(fold) for fold in recent["Training_Fold"].tolist()),
            "Family_Best_Config_Count": int(group["Config_ID"].nunique()),
            "Family_Best_Portfolio_Constructions": ";".join(best_constructions),
            "Family_Average_Total_Return": float(group["Total_Return"].mean()),
            "Family_Average_Excess_Return_vs_Universe": float(group["Excess_Return_vs_Universe"].mean()),
            "Family_Average_Excess_Return_vs_QQQ": float(group["Excess_Return_vs_QQQ"].mean()),
            "Family_Average_Excess_Return_vs_XLK": float(group["Excess_Return_vs_XLK"].mean()),
            "Family_Average_Sharpe": float(group["Sharpe"].mean()),
            "Family_Average_Turnover": float(group["Avg_Turnover"].mean()),
            "Family_Recent_Fold_Total_Return": float(recent["Total_Return"].mean()),
            "Family_Recent_Fold_Excess_Return_vs_Universe": float(recent["Excess_Return_vs_Universe"].mean()),
            "Family_Recent_Fold_Excess_Return_vs_QQQ": float(recent["Excess_Return_vs_QQQ"].mean()),
            "Family_Recent_Fold_Excess_Return_vs_XLK": float(recent["Excess_Return_vs_XLK"].mean()),
            "Family_Recent_Fold_Sharpe": float(recent["Sharpe"].mean()),
            "Family_Recent_Fold_Max_Drawdown": float(recent["Max_Drawdown"].min()),
            "Family_Recent_Fold_Downside_Capture_vs_XLK": float(recent["Downside_Capture_vs_XLK"].mean()),
            "Family_Worst_Fold_Excess_Return_vs_Universe": float(group["Excess_Return_vs_Universe"].min()),
            "Family_Worst_Fold_Excess_Return_vs_QQQ": float(group["Excess_Return_vs_QQQ"].min()),
            "Family_Worst_Fold_Excess_Return_vs_XLK": float(group["Excess_Return_vs_XLK"].min()),
            "Family_Positive_Universe_Fold_Rate": float((group["Excess_Return_vs_Universe"] >= 0.0).mean()),
            "Family_Positive_QQQ_Fold_Rate": float((group["Excess_Return_vs_QQQ"] >= 0.0).mean()),
            "Family_Positive_XLK_Fold_Rate": float((group["Excess_Return_vs_XLK"] >= 0.0).mean()),
            "Family_Worst_Fold_Drawdown": float(group["Max_Drawdown"].min()),
            "Family_Average_Downside_Capture_vs_XLK": float(group["Downside_Capture_vs_XLK"].mean()),
        })

    family_stability = pd.DataFrame(rows)
    if family_stability.empty:
        return out
    return out.merge(family_stability, on="Candidate_Family_ID", how="left")


def guarded_selection_objective(
    summary: pd.DataFrame,
    params: WalkForwardSelectionParams,
) -> pd.Series:
    return (
        params.universe_weight * summary["Excess_Return_vs_Universe"].fillna(0.0)
        + params.qqq_weight * summary["Excess_Return_vs_QQQ"].fillna(0.0)
        + params.xlk_weight * summary["Excess_Return_vs_XLK"].fillna(0.0)
        + params.sharpe_weight * summary["Sharpe"].fillna(0.0)
        + params.information_ratio_weight * summary["Information_Ratio_vs_XLK"].fillna(0.0)
        - params.drawdown_penalty * summary["Max_Drawdown"].abs().fillna(0.0)
        - params.turnover_penalty * summary["Avg_Turnover"].fillna(0.0)
    )


def defensive_selection_objective(
    summary: pd.DataFrame,
    params: WalkForwardSelectionParams,
) -> pd.Series:
    base = guarded_selection_objective(summary, params)
    recent_universe = metric_series(summary, "Recent_Fold_Excess_Return_vs_Universe", 0.0).fillna(0.0)
    recent_xlk = metric_series(summary, "Recent_Fold_Excess_Return_vs_XLK", 0.0).fillna(0.0)
    positive_universe_rate = metric_series(summary, "Positive_Universe_Fold_Rate", 0.0).fillna(0.0)
    positive_xlk_rate = metric_series(summary, "Positive_XLK_Fold_Rate", 0.0).fillna(0.0)
    recent_sharpe = metric_series(summary, "Recent_Fold_Sharpe", 0.0).fillna(0.0)
    worst_xlk_lag = metric_series(summary, "Worst_Fold_Excess_Return_vs_XLK", 0.0).fillna(0.0).clip(upper=0.0).abs()
    worst_universe_lag = metric_series(summary, "Worst_Fold_Excess_Return_vs_Universe", 0.0).fillna(0.0).clip(upper=0.0).abs()
    downside_capture = metric_series(summary, "Average_Downside_Capture_vs_XLK", 1.0).fillna(1.0).clip(lower=0.0)
    fold_drawdown = metric_series(summary, "Worst_Fold_Drawdown", 0.0).fillna(0.0).abs()

    return (
        base
        + params.recent_universe_weight * recent_universe
        + params.recent_xlk_weight * recent_xlk
        + params.positive_universe_rate_weight * positive_universe_rate
        + params.positive_xlk_rate_weight * positive_xlk_rate
        + params.recent_sharpe_weight * recent_sharpe
        - params.worst_xlk_lag_penalty * worst_xlk_lag
        - params.worst_universe_lag_penalty * worst_universe_lag
        - params.downside_capture_penalty * downside_capture
        - params.fold_drawdown_penalty * fold_drawdown
    )


def family_stability_objective(
    summary: pd.DataFrame,
    params: WalkForwardSelectionParams,
) -> pd.Series:
    avg_universe = metric_series(summary, "Family_Average_Excess_Return_vs_Universe", 0.0).fillna(0.0)
    avg_xlk = metric_series(summary, "Family_Average_Excess_Return_vs_XLK", 0.0).fillna(0.0)
    recent_universe = metric_series(summary, "Family_Recent_Fold_Excess_Return_vs_Universe", 0.0).fillna(0.0)
    recent_xlk = metric_series(summary, "Family_Recent_Fold_Excess_Return_vs_XLK", 0.0).fillna(0.0)
    avg_sharpe = metric_series(summary, "Family_Average_Sharpe", 0.0).fillna(0.0)
    positive_universe_rate = metric_series(summary, "Family_Positive_Universe_Fold_Rate", 0.0).fillna(0.0)
    positive_xlk_rate = metric_series(summary, "Family_Positive_XLK_Fold_Rate", 0.0).fillna(0.0)
    worst_xlk_lag = metric_series(summary, "Family_Worst_Fold_Excess_Return_vs_XLK", 0.0).fillna(0.0).clip(upper=0.0).abs()
    worst_universe_lag = metric_series(summary, "Family_Worst_Fold_Excess_Return_vs_Universe", 0.0).fillna(0.0).clip(upper=0.0).abs()
    downside_capture = metric_series(summary, "Family_Average_Downside_Capture_vs_XLK", 1.0).fillna(1.0).clip(lower=0.0)
    worst_drawdown = metric_series(summary, "Family_Worst_Fold_Drawdown", 0.0).fillna(0.0).abs()
    avg_turnover = metric_series(summary, "Family_Average_Turnover", 0.0).fillna(0.0)

    return (
        0.25 * avg_universe
        + 0.45 * avg_xlk
        + params.recent_universe_weight * recent_universe
        + params.recent_xlk_weight * recent_xlk
        + 0.08 * avg_sharpe
        + 0.12 * positive_universe_rate
        + 0.18 * positive_xlk_rate
        - params.worst_xlk_lag_penalty * worst_xlk_lag
        - params.worst_universe_lag_penalty * worst_universe_lag
        - params.downside_capture_penalty * downside_capture
        - params.fold_drawdown_penalty * worst_drawdown
        - params.turnover_penalty * avg_turnover
    )


def add_selection_guard_columns(
    summary: pd.DataFrame,
    params: WalkForwardSelectionParams,
) -> pd.DataFrame:
    out = add_candidate_family_column(summary)
    out["Guarded_Simulator_Objective"] = guarded_selection_objective(out, params)
    out["Defensive_Simulator_Objective"] = defensive_selection_objective(out, params)
    out["Family_Stability_Objective"] = family_stability_objective(out, params)
    out["Family_Stable_Defensive_Objective"] = (
        out["Defensive_Simulator_Objective"].fillna(0.0)
        + params.family_stability_weight * out["Family_Stability_Objective"].fillna(0.0)
    )
    out["Guard_Passes_XLK"] = out["Excess_Return_vs_XLK"].fillna(-np.inf) >= params.min_excess_vs_xlk
    out["Guard_Passes_Drawdown"] = out["Max_Drawdown"].fillna(-np.inf) >= -abs(params.max_drawdown)
    out["Guard_Passes_Sharpe"] = out["Sharpe"].fillna(-np.inf) >= params.min_sharpe
    if params.max_turnover is None:
        out["Guard_Passes_Turnover"] = True
    else:
        out["Guard_Passes_Turnover"] = out["Avg_Turnover"].fillna(np.inf) <= params.max_turnover
    out["Guard_Passes_Recent_Universe"] = (
        metric_series(out, "Recent_Fold_Excess_Return_vs_Universe", -np.inf).fillna(-np.inf)
        >= params.min_recent_excess_vs_universe
    )
    out["Guard_Passes_Recent_XLK"] = (
        metric_series(out, "Recent_Fold_Excess_Return_vs_XLK", -np.inf).fillna(-np.inf)
        >= params.min_recent_excess_vs_xlk
    )
    out["Guard_Passes_Recent_Drawdown"] = (
        metric_series(out, "Recent_Fold_Max_Drawdown", -np.inf).fillna(-np.inf)
        >= -abs(params.max_recent_drawdown)
    )
    out["Guard_Passes_Worst_Fold_Drawdown"] = (
        metric_series(out, "Worst_Fold_Drawdown", -np.inf).fillna(-np.inf)
        >= -abs(params.max_fold_drawdown)
    )
    out["Guard_Passes_Downside_Capture"] = (
        metric_series(out, "Average_Downside_Capture_vs_XLK", 0.0).fillna(0.0)
        <= params.max_downside_capture_vs_xlk
    )
    out["Guard_Passes_Positive_Universe_Rate"] = (
        metric_series(out, "Positive_Universe_Fold_Rate", 0.0).fillna(0.0)
        >= params.min_positive_universe_fold_rate
    )
    out["Guard_Passes_Positive_XLK_Rate"] = (
        metric_series(out, "Positive_XLK_Fold_Rate", 0.0).fillna(0.0)
        >= params.min_positive_xlk_fold_rate
    )
    train_fold_count = metric_series(out, "Train_Fold_Count", 1.0).fillna(1.0).clip(lower=1.0)
    required_family_folds = train_fold_count.map(
        lambda count: min(max(1, int(params.min_family_train_folds)), int(count))
    )
    out["Guard_Passes_Family_Fold_Count"] = (
        metric_series(out, "Family_Train_Fold_Count", 0.0).fillna(0.0)
        >= required_family_folds
    )
    out["Guard_Passes_Family_Recent_Universe"] = (
        metric_series(out, "Family_Recent_Fold_Excess_Return_vs_Universe", -np.inf).fillna(-np.inf)
        >= params.min_family_recent_excess_vs_universe
    )
    out["Guard_Passes_Family_Recent_XLK"] = (
        metric_series(out, "Family_Recent_Fold_Excess_Return_vs_XLK", -np.inf).fillna(-np.inf)
        >= params.min_family_recent_excess_vs_xlk
    )
    out["Guard_Passes_Family_Worst_Drawdown"] = (
        metric_series(out, "Family_Worst_Fold_Drawdown", -np.inf).fillna(-np.inf)
        >= -abs(params.max_family_worst_drawdown)
    )
    out["Guard_Passes_Family_Downside_Capture"] = (
        metric_series(out, "Family_Average_Downside_Capture_vs_XLK", 0.0).fillna(0.0)
        <= params.max_family_downside_capture_vs_xlk
    )
    out["Guard_Passes_Family_Positive_Universe_Rate"] = (
        metric_series(out, "Family_Positive_Universe_Fold_Rate", 0.0).fillna(0.0)
        >= params.min_family_positive_universe_fold_rate
    )
    out["Guard_Passes_Family_Positive_XLK_Rate"] = (
        metric_series(out, "Family_Positive_XLK_Fold_Rate", 0.0).fillna(0.0)
        >= params.min_family_positive_xlk_fold_rate
    )
    return out


def select_walk_forward_candidate(
    train_summary: pd.DataFrame,
    params: WalkForwardSelectionParams,
) -> tuple[pd.Series, str, int]:
    if train_summary.empty:
        raise ValueError("train_summary is empty")

    if params.mode == "objective":
        selected = train_summary.sort_values("Simulator_Objective", ascending=False).iloc[0]
        return selected, "legacy_objective", int(len(train_summary))

    guarded = add_selection_guard_columns(train_summary, params)
    aggregate_guard = (
        guarded["Guard_Passes_XLK"]
        & guarded["Guard_Passes_Drawdown"]
        & guarded["Guard_Passes_Sharpe"]
        & guarded["Guard_Passes_Turnover"]
    )
    consistency_guard = (
        guarded["Guard_Passes_Recent_Universe"]
        & guarded["Guard_Passes_Recent_XLK"]
        & guarded["Guard_Passes_Recent_Drawdown"]
        & guarded["Guard_Passes_Worst_Fold_Drawdown"]
        & guarded["Guard_Passes_Downside_Capture"]
        & guarded["Guard_Passes_Positive_Universe_Rate"]
        & guarded["Guard_Passes_Positive_XLK_Rate"]
    )
    recent_defensive_guard = (
        guarded["Guard_Passes_Turnover"]
        & guarded["Guard_Passes_Sharpe"]
        & (guarded["Excess_Return_vs_Universe"].fillna(-np.inf) >= 0.0)
        & guarded["Guard_Passes_Recent_Universe"]
        & guarded["Guard_Passes_Recent_Drawdown"]
        & guarded["Guard_Passes_Worst_Fold_Drawdown"]
        & guarded["Guard_Passes_Downside_Capture"]
    )
    family_guard = (
        guarded["Guard_Passes_Family_Fold_Count"]
        & guarded["Guard_Passes_Family_Recent_Universe"]
        & guarded["Guard_Passes_Family_Recent_XLK"]
        & guarded["Guard_Passes_Family_Worst_Drawdown"]
        & guarded["Guard_Passes_Family_Downside_Capture"]
        & guarded["Guard_Passes_Family_Positive_Universe_Rate"]
        & guarded["Guard_Passes_Family_Positive_XLK_Rate"]
    )
    family_defensive_guard = (
        guarded["Guard_Passes_Family_Fold_Count"]
        & guarded["Guard_Passes_Family_Recent_Universe"]
        & guarded["Guard_Passes_Family_Worst_Drawdown"]
        & guarded["Guard_Passes_Family_Downside_Capture"]
        & guarded["Guard_Passes_Family_Positive_Universe_Rate"]
    )

    if params.mode == "guarded":
        objective_col = "Guarded_Simulator_Objective"
        stages = [
            ("strict_xlk_drawdown_sharpe_turnover", guarded[aggregate_guard]),
            (
                "relaxed_xlk_drawdown_turnover",
                guarded[
                    guarded["Guard_Passes_XLK"]
                    & guarded["Guard_Passes_Drawdown"]
                    & guarded["Guard_Passes_Turnover"]
                ],
            ),
            (
                "relaxed_drawdown_sharpe_turnover",
                guarded[
                    guarded["Guard_Passes_Drawdown"]
                    & guarded["Guard_Passes_Sharpe"]
                    & guarded["Guard_Passes_Turnover"]
                ],
            ),
            (
                "relaxed_drawdown_turnover",
                guarded[
                    guarded["Guard_Passes_Drawdown"]
                    & guarded["Guard_Passes_Turnover"]
                ],
            ),
            ("relaxed_benchmark_only", guarded[guarded["Guard_Passes_XLK"]]),
            ("fallback_all", guarded),
        ]
    elif params.mode == "guarded_defensive":
        objective_col = "Defensive_Simulator_Objective"
        stages = [
            ("strict_recent_xlk_drawdown_downside", guarded[aggregate_guard & consistency_guard]),
            ("recent_defensive_downside", guarded[recent_defensive_guard]),
            ("benchmark_drawdown_guard", guarded[aggregate_guard]),
            (
                "relaxed_drawdown_turnover",
                guarded[
                    guarded["Guard_Passes_Drawdown"]
                    & guarded["Guard_Passes_Turnover"]
                ],
            ),
            ("fallback_all", guarded),
        ]
    elif params.mode == "guarded_family_stable":
        objective_col = "Family_Stable_Defensive_Objective"
        stages = [
            (
                "family_stable_strict_recent_xlk_drawdown_downside",
                guarded[aggregate_guard & consistency_guard & family_guard],
            ),
            (
                "family_stable_recent_defensive_downside",
                guarded[recent_defensive_guard & family_guard],
            ),
            (
                "family_stable_benchmark_drawdown_guard",
                guarded[aggregate_guard & family_defensive_guard],
            ),
            (
                "family_stable_relaxed_drawdown_turnover",
                guarded[
                    guarded["Guard_Passes_Drawdown"]
                    & guarded["Guard_Passes_Turnover"]
                    & guarded["Guard_Passes_Family_Fold_Count"]
                    & guarded["Guard_Passes_Family_Worst_Drawdown"]
                ],
            ),
            ("fallback_all", guarded),
        ]
    else:
        raise ValueError(f"Unsupported walk-forward selection mode: {params.mode}")

    for stage, candidates in stages:
        if not candidates.empty:
            selected = candidates.sort_values(objective_col, ascending=False).iloc[0]
            return selected, stage, int(len(candidates))

    selected = guarded.sort_values(objective_col, ascending=False).iloc[0]
    return selected, "fallback_all", int(len(guarded))


def walk_forward_selection_from_equity(
    daily: pd.DataFrame,
    rebalances: pd.DataFrame,
    selection_params: WalkForwardSelectionParams,
) -> pd.DataFrame:
    if daily.empty or "Fold" not in daily.columns:
        return pd.DataFrame()

    selection_rows: list[dict] = []
    available_folds = [fold for fold in weight_tuning.FOLD_ORDER if fold in set(daily["Fold"])]

    for validation_idx in range(1, len(available_folds)):
        validation_fold = available_folds[validation_idx]
        train_folds = available_folds[:validation_idx]
        train_daily = daily[daily["Fold"].isin(train_folds)].copy()
        train_rebalances = rebalances[rebalances["Fold"].isin(train_folds)].copy() if "Fold" in rebalances.columns else pd.DataFrame()
        validation_daily = daily[daily["Fold"] == validation_fold].copy()
        validation_rebalances = rebalances[rebalances["Fold"] == validation_fold].copy() if "Fold" in rebalances.columns else pd.DataFrame()
        if train_daily.empty or validation_daily.empty:
            continue

        train_summary = summarize_simulation(train_daily, train_rebalances)
        if train_summary.empty:
            continue
        if selection_params.mode in {"guarded_defensive", "guarded_family_stable"}:
            train_summary = add_recent_fold_consistency_columns(
                summary=train_summary,
                train_daily=train_daily,
                train_rebalances=train_rebalances,
                train_folds=train_folds,
                params=selection_params,
            )
        if selection_params.mode == "guarded_family_stable":
            train_summary = add_candidate_family_stability_columns(
                summary=train_summary,
                train_daily=train_daily,
                train_rebalances=train_rebalances,
                train_folds=train_folds,
                params=selection_params,
            )

        selected, guard_stage, guard_candidate_count = select_walk_forward_candidate(
            train_summary,
            selection_params,
        )
        selected_validation_daily = filter_simulation_variant(validation_daily, selected)
        selected_validation_rebalances = filter_simulation_variant(validation_rebalances, selected)
        validation_summary = summarize_simulation(
            selected_validation_daily,
            selected_validation_rebalances,
        )
        validation = validation_summary.iloc[0].to_dict() if not validation_summary.empty else {}

        round_number = len(selection_rows) + 1
        selection_rows.append({
            "Walk_Forward_Round": round_number,
            "Train_Folds": ",".join(train_folds),
            "Validation_Fold": validation_fold,
            "Selection_Mode": selection_params.mode,
            "Selection_Guard_Stage": guard_stage,
            "Selection_Guard_Candidate_Count": guard_candidate_count,
            "Selected_Config_ID": selected["Config_ID"],
            "Selected_Top_N": int(selected["Top_N"]),
            "Selected_Portfolio_Mode": selected["Portfolio_Mode"],
            "Selected_Candidate_Family_ID": selected.get(
                "Candidate_Family_ID",
                candidate_family_id(selected["Config_ID"]),
            ),
            "Train_Simulator_Objective": float(selected["Simulator_Objective"]),
            "Train_Guarded_Simulator_Objective": float(selected.get("Guarded_Simulator_Objective", np.nan)),
            "Train_Defensive_Simulator_Objective": float(selected.get("Defensive_Simulator_Objective", np.nan)),
            "Train_Family_Stability_Objective": float(selected.get("Family_Stability_Objective", np.nan)),
            "Train_Family_Stable_Defensive_Objective": float(selected.get("Family_Stable_Defensive_Objective", np.nan)),
            "Train_Guard_Passes_XLK": bool(selected.get("Guard_Passes_XLK", False)),
            "Train_Guard_Passes_Drawdown": bool(selected.get("Guard_Passes_Drawdown", False)),
            "Train_Guard_Passes_Sharpe": bool(selected.get("Guard_Passes_Sharpe", False)),
            "Train_Guard_Passes_Turnover": bool(selected.get("Guard_Passes_Turnover", False)),
            "Train_Guard_Passes_Recent_Universe": bool(selected.get("Guard_Passes_Recent_Universe", False)),
            "Train_Guard_Passes_Recent_XLK": bool(selected.get("Guard_Passes_Recent_XLK", False)),
            "Train_Guard_Passes_Recent_Drawdown": bool(selected.get("Guard_Passes_Recent_Drawdown", False)),
            "Train_Guard_Passes_Worst_Fold_Drawdown": bool(selected.get("Guard_Passes_Worst_Fold_Drawdown", False)),
            "Train_Guard_Passes_Downside_Capture": bool(selected.get("Guard_Passes_Downside_Capture", False)),
            "Train_Guard_Passes_Positive_Universe_Rate": bool(selected.get("Guard_Passes_Positive_Universe_Rate", False)),
            "Train_Guard_Passes_Positive_XLK_Rate": bool(selected.get("Guard_Passes_Positive_XLK_Rate", False)),
            "Train_Guard_Passes_Family_Fold_Count": bool(selected.get("Guard_Passes_Family_Fold_Count", False)),
            "Train_Guard_Passes_Family_Recent_Universe": bool(selected.get("Guard_Passes_Family_Recent_Universe", False)),
            "Train_Guard_Passes_Family_Recent_XLK": bool(selected.get("Guard_Passes_Family_Recent_XLK", False)),
            "Train_Guard_Passes_Family_Worst_Drawdown": bool(selected.get("Guard_Passes_Family_Worst_Drawdown", False)),
            "Train_Guard_Passes_Family_Downside_Capture": bool(selected.get("Guard_Passes_Family_Downside_Capture", False)),
            "Train_Guard_Passes_Family_Positive_Universe_Rate": bool(selected.get("Guard_Passes_Family_Positive_Universe_Rate", False)),
            "Train_Guard_Passes_Family_Positive_XLK_Rate": bool(selected.get("Guard_Passes_Family_Positive_XLK_Rate", False)),
            "Train_Total_Return": float(selected["Total_Return"]),
            "Train_Excess_Return_vs_Universe": float(selected["Excess_Return_vs_Universe"]),
            "Train_Excess_Return_vs_QQQ": float(selected["Excess_Return_vs_QQQ"]),
            "Train_Excess_Return_vs_XLK": float(selected["Excess_Return_vs_XLK"]),
            "Train_Sharpe": float(selected["Sharpe"]),
            "Train_Max_Drawdown": float(selected["Max_Drawdown"]),
            "Train_Avg_Turnover": float(selected["Avg_Turnover"]),
            "Train_Fold_Count": float(selected.get("Train_Fold_Count", np.nan)),
            "Recent_Training_Folds": selected.get("Recent_Training_Folds", ""),
            "Recent_Fold_Excess_Return_vs_Universe": float(selected.get("Recent_Fold_Excess_Return_vs_Universe", np.nan)),
            "Recent_Fold_Excess_Return_vs_QQQ": float(selected.get("Recent_Fold_Excess_Return_vs_QQQ", np.nan)),
            "Recent_Fold_Excess_Return_vs_XLK": float(selected.get("Recent_Fold_Excess_Return_vs_XLK", np.nan)),
            "Recent_Fold_Sharpe": float(selected.get("Recent_Fold_Sharpe", np.nan)),
            "Recent_Fold_Max_Drawdown": float(selected.get("Recent_Fold_Max_Drawdown", np.nan)),
            "Recent_Fold_Downside_Capture_vs_XLK": float(selected.get("Recent_Fold_Downside_Capture_vs_XLK", np.nan)),
            "Worst_Fold_Excess_Return_vs_Universe": float(selected.get("Worst_Fold_Excess_Return_vs_Universe", np.nan)),
            "Worst_Fold_Excess_Return_vs_QQQ": float(selected.get("Worst_Fold_Excess_Return_vs_QQQ", np.nan)),
            "Worst_Fold_Excess_Return_vs_XLK": float(selected.get("Worst_Fold_Excess_Return_vs_XLK", np.nan)),
            "Positive_Universe_Fold_Rate": float(selected.get("Positive_Universe_Fold_Rate", np.nan)),
            "Positive_QQQ_Fold_Rate": float(selected.get("Positive_QQQ_Fold_Rate", np.nan)),
            "Positive_XLK_Fold_Rate": float(selected.get("Positive_XLK_Fold_Rate", np.nan)),
            "Worst_Fold_Drawdown": float(selected.get("Worst_Fold_Drawdown", np.nan)),
            "Average_Downside_Capture_vs_XLK": float(selected.get("Average_Downside_Capture_vs_XLK", np.nan)),
            "Family_Train_Fold_Count": float(selected.get("Family_Train_Fold_Count", np.nan)),
            "Family_Recent_Training_Folds": selected.get("Family_Recent_Training_Folds", ""),
            "Family_Best_Config_Count": float(selected.get("Family_Best_Config_Count", np.nan)),
            "Family_Best_Portfolio_Constructions": selected.get("Family_Best_Portfolio_Constructions", ""),
            "Family_Average_Excess_Return_vs_Universe": float(selected.get("Family_Average_Excess_Return_vs_Universe", np.nan)),
            "Family_Average_Excess_Return_vs_QQQ": float(selected.get("Family_Average_Excess_Return_vs_QQQ", np.nan)),
            "Family_Average_Excess_Return_vs_XLK": float(selected.get("Family_Average_Excess_Return_vs_XLK", np.nan)),
            "Family_Average_Sharpe": float(selected.get("Family_Average_Sharpe", np.nan)),
            "Family_Average_Turnover": float(selected.get("Family_Average_Turnover", np.nan)),
            "Family_Recent_Fold_Excess_Return_vs_Universe": float(selected.get("Family_Recent_Fold_Excess_Return_vs_Universe", np.nan)),
            "Family_Recent_Fold_Excess_Return_vs_QQQ": float(selected.get("Family_Recent_Fold_Excess_Return_vs_QQQ", np.nan)),
            "Family_Recent_Fold_Excess_Return_vs_XLK": float(selected.get("Family_Recent_Fold_Excess_Return_vs_XLK", np.nan)),
            "Family_Recent_Fold_Sharpe": float(selected.get("Family_Recent_Fold_Sharpe", np.nan)),
            "Family_Recent_Fold_Max_Drawdown": float(selected.get("Family_Recent_Fold_Max_Drawdown", np.nan)),
            "Family_Recent_Fold_Downside_Capture_vs_XLK": float(selected.get("Family_Recent_Fold_Downside_Capture_vs_XLK", np.nan)),
            "Family_Worst_Fold_Excess_Return_vs_Universe": float(selected.get("Family_Worst_Fold_Excess_Return_vs_Universe", np.nan)),
            "Family_Worst_Fold_Excess_Return_vs_QQQ": float(selected.get("Family_Worst_Fold_Excess_Return_vs_QQQ", np.nan)),
            "Family_Worst_Fold_Excess_Return_vs_XLK": float(selected.get("Family_Worst_Fold_Excess_Return_vs_XLK", np.nan)),
            "Family_Positive_Universe_Fold_Rate": float(selected.get("Family_Positive_Universe_Fold_Rate", np.nan)),
            "Family_Positive_QQQ_Fold_Rate": float(selected.get("Family_Positive_QQQ_Fold_Rate", np.nan)),
            "Family_Positive_XLK_Fold_Rate": float(selected.get("Family_Positive_XLK_Fold_Rate", np.nan)),
            "Family_Worst_Fold_Drawdown": float(selected.get("Family_Worst_Fold_Drawdown", np.nan)),
            "Family_Average_Downside_Capture_vs_XLK": float(selected.get("Family_Average_Downside_Capture_vs_XLK", np.nan)),
            "Validation_Final_Equity": validation.get("Final_Equity", np.nan),
            "Validation_Total_Return": validation.get("Total_Return", np.nan),
            "Validation_Excess_Return_vs_Universe": validation.get("Excess_Return_vs_Universe", np.nan),
            "Validation_Excess_Return_vs_QQQ": validation.get("Excess_Return_vs_QQQ", np.nan),
            "Validation_Excess_Return_vs_XLK": validation.get("Excess_Return_vs_XLK", np.nan),
            "Validation_Sharpe": validation.get("Sharpe", np.nan),
            "Validation_Max_Drawdown": validation.get("Max_Drawdown", np.nan),
            "Validation_Avg_Turnover": validation.get("Avg_Turnover", np.nan),
            "Validation_Avg_Cash_Weight": validation.get("Avg_Cash_Weight", np.nan),
        })

    return pd.DataFrame(selection_rows).round(6)


def simulate_walk_forward_dynamic_portfolio(
    scores: pd.DataFrame,
    configs: list[weight_tuning.WeightConfig | DirectScoreConfig],
    selections: pd.DataFrame,
    price_wide: pd.DataFrame,
    adv20: pd.DataFrame,
    params: SimulationParams,
    historical_universe: HistoricalUniverse | None,
    fallback_policy: FallbackPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if selections.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    fallback_policy = fallback_policy or FallbackPolicy()
    fallback_mode = str(fallback_policy.mode).lower().strip()
    clean_guard_stages = set(fallback_policy.clean_guard_stages)
    config_lookup = {config.config_id: config for config in configs}
    scored_cache: dict[str, pd.DataFrame] = {}
    selection_by_fold = {
        str(row["Validation_Fold"]): row
        for _, row in selections.iterrows()
        if str(row["Selected_Config_ID"]) in config_lookup
    }
    end_date = pd.to_datetime(price_wide.index.max()).normalize()
    live_schedule = []
    for anchor, trade_date, _ in build_rebalance_schedule(scores, price_wide.index, params.rebalance_delay_days):
        if price_wide.index.get_loc(trade_date) >= len(price_wide.index) - 1:
            continue
        fold = regime_bt.fold_label(trade_date, end_date)
        selected = selection_by_fold.get(fold)
        if selected is not None:
            live_schedule.append((anchor, trade_date, fold, selected))

    if not live_schedule:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    strategy = PortfolioState()
    universe = PortfolioState()
    qqq_equity = 1.0
    xlk_equity = 1.0
    daily_rows: list[dict] = []
    rebalance_rows: list[dict] = []

    for idx, (anchor, trade_date, fold, selected) in enumerate(live_schedule):
        selected_config_id = str(selected["Selected_Config_ID"])
        selected_top_n = int(selected["Selected_Top_N"])
        selected_mode = str(selected["Selected_Portfolio_Mode"])
        selected_family_id = str(selected.get("Selected_Candidate_Family_ID", candidate_family_id(selected_config_id)))
        selection_mode = str(selected.get("Selection_Mode", "objective"))
        selection_guard_stage = str(selected.get("Selection_Guard_Stage", "legacy_objective"))
        fallback_triggered = fallback_mode != "none" and selection_guard_stage not in clean_guard_stages
        fallback_asset = "none"
        fallback_reason = ""
        round_number = int(selected["Walk_Forward_Round"])
        if selected_config_id not in scored_cache:
            scored_cache[selected_config_id] = score_frame_for_config(
                scores,
                config_lookup[selected_config_id],
            )
        selected_scores = scored_cache[selected_config_id]
        day = selected_scores[selected_scores["Date"] == anchor].copy()
        ranked = day.sort_values("Tuned_Combined_Score", ascending=False).copy()

        strategy_target = build_target(
            ranked=ranked,
            anchor=anchor,
            trade_date=trade_date,
            top_n=selected_top_n,
            mode=selected_mode,
            price_wide=price_wide,
            adv20=adv20,
            params=params,
            equity_dollars=strategy.equity * params.initial_capital,
            historical_universe=historical_universe,
        )
        if fallback_triggered:
            strategy_target, fallback_asset, fallback_reason = fallback_target(
                fallback_mode=fallback_mode,
                trade_date=trade_date,
                price_wide=price_wide,
                equity_dollars=strategy.equity * params.initial_capital,
                exposure=fallback_policy.exposure,
            )
            fallback_reason = (
                f"selection_stage_{selection_guard_stage}_not_in_clean_stages;"
                f"{fallback_reason}"
            )
        exposure_fields = exposure_label_fields(
            fallback_triggered=bool(fallback_triggered),
            fallback_asset=fallback_asset,
            selected_config_id=selected_config_id,
            selected_top_n=selected_top_n,
            selected_mode=selected_mode,
            selection_guard_stage=selection_guard_stage,
        )
        universe_target = build_target(
            ranked=ranked,
            anchor=anchor,
            trade_date=trade_date,
            top_n=None,
            mode="equal",
            price_wide=price_wide,
            adv20=adv20,
            params=params,
            equity_dollars=universe.equity * params.initial_capital,
            historical_universe=historical_universe,
        )
        strategy_rebalance = rebalance_portfolio(strategy, strategy_target.weights, trade_date, adv20, params)
        universe_rebalance = rebalance_portfolio(universe, universe_target.weights, trade_date, adv20, params)

        rebalance_rows.append({
            "Config_ID": "walk_forward_dynamic",
            "Top_N": 0,
            "Portfolio_Mode": "dynamic",
            "Walk_Forward_Round": round_number,
            "Fold": fold,
            "Selection_Mode": selection_mode,
            "Selection_Guard_Stage": selection_guard_stage,
            "Fallback_Mode": fallback_mode,
            "Fallback_Triggered": bool(fallback_triggered),
            "Fallback_Asset": fallback_asset,
            "Fallback_Reason": fallback_reason,
            **exposure_fields,
            "Train_Folds": selected["Train_Folds"],
            "Selected_Config_ID": selected_config_id,
            "Selected_Candidate_Family_ID": selected_family_id,
            "Selected_Top_N": selected_top_n,
            "Selected_Portfolio_Mode": selected_mode,
            "Rebalance_Number": idx + 1,
            "Anchor_Date": anchor.strftime("%Y-%m-%d"),
            "Trade_Date": trade_date.strftime("%Y-%m-%d"),
            "Strategy_Equity_After_Rebalance": strategy.equity,
            "Universe_Equity_After_Rebalance": universe.equity,
            "Strategy_Holdings": int(len(strategy.weights)),
            "Universe_Holdings": int(len(universe.weights)),
            "Strategy_Cash_Weight": strategy.cash_weight,
            "Universe_Cash_Weight": universe.cash_weight,
            "Strategy_Turnover": strategy_rebalance["Turnover"],
            "Strategy_Gross_Trade_Weight": strategy_rebalance["Gross_Trade_Weight"],
            "Strategy_Transaction_Cost": strategy_rebalance["Transaction_Cost"],
            "Strategy_Slippage_Cost": strategy_rebalance["Slippage_Cost"],
            "Strategy_Total_Cost": strategy_rebalance["Total_Cost"],
            "Strategy_Max_Trade_ADV_Pct": strategy_rebalance["Max_Trade_ADV_Pct"],
            "Strategy_Avg_Trade_ADV_Pct": strategy_rebalance["Avg_Trade_ADV_Pct"],
            "Strategy_Trade_ADV_Breach_Count": strategy_rebalance["Trade_ADV_Breach_Count"],
            "Strategy_Universe_Start_Count": strategy_target.universe_start_count,
            "Strategy_Excluded_By_Universe": strategy_target.excluded_by_universe,
            "Strategy_Universe_Filter_Source": strategy_target.universe_source,
            "Strategy_Eligible_Count": strategy_target.eligible_count,
            "Strategy_Excluded_By_Liquidity": strategy_target.excluded_by_liquidity,
            "Strategy_Position_Cap_Bind_Count": strategy_target.position_cap_bind_count,
            "Strategy_Subindustry_Cap_Bind_Count": strategy_target.subindustry_cap_bind_count,
            "Strategy_ADV_Cap_Bind_Count": strategy_target.adv_cap_bind_count,
            "Universe_Turnover": universe_rebalance["Turnover"],
            "Universe_Gross_Trade_Weight": universe_rebalance["Gross_Trade_Weight"],
            "Universe_Transaction_Cost": universe_rebalance["Transaction_Cost"],
            "Universe_Slippage_Cost": universe_rebalance["Slippage_Cost"],
            "Universe_Total_Cost": universe_rebalance["Total_Cost"],
            "Universe_Universe_Start_Count": universe_target.universe_start_count,
            "Universe_Excluded_By_Universe": universe_target.excluded_by_universe,
            "Universe_Universe_Filter_Source": universe_target.universe_source,
            "Universe_Eligible_Count": universe_target.eligible_count,
            "Universe_Excluded_By_Liquidity": universe_target.excluded_by_liquidity,
        })
        daily_rows.append({
            "Config_ID": "walk_forward_dynamic",
            "Top_N": 0,
            "Portfolio_Mode": "dynamic",
            "Walk_Forward_Round": round_number,
            "Fold": fold,
            "Selection_Mode": selection_mode,
            "Selection_Guard_Stage": selection_guard_stage,
            "Fallback_Mode": fallback_mode,
            "Fallback_Triggered": bool(fallback_triggered),
            "Fallback_Asset": fallback_asset,
            "Fallback_Reason": fallback_reason,
            **exposure_fields,
            "Selected_Config_ID": selected_config_id,
            "Selected_Candidate_Family_ID": selected_family_id,
            "Selected_Top_N": selected_top_n,
            "Selected_Portfolio_Mode": selected_mode,
            "Date": trade_date.strftime("%Y-%m-%d"),
            "Rebalanced": True,
            "Strategy_Daily_Return": -strategy_rebalance["Total_Cost"],
            "Universe_Daily_Return": -universe_rebalance["Total_Cost"],
            "QQQ_Daily_Return": 0.0,
            "XLK_Daily_Return": 0.0,
            "Strategy_Equity": strategy.equity,
            "Universe_Equity": universe.equity,
            "QQQ_Equity": qqq_equity,
            "XLK_Equity": xlk_equity,
            "Strategy_Cash_Weight": strategy.cash_weight,
            "Universe_Cash_Weight": universe.cash_weight,
            "Strategy_Holdings": int(len(strategy.weights)),
            "Universe_Holdings": int(len(universe.weights)),
            "Strategy_Total_Cost": strategy_rebalance["Total_Cost"],
            "Universe_Total_Cost": universe_rebalance["Total_Cost"],
        })

        current_pos = int(price_wide.index.get_loc(trade_date))
        if idx + 1 < len(live_schedule):
            next_trade_date_value = live_schedule[idx + 1][1]
            next_pos = int(price_wide.index.get_loc(next_trade_date_value))
        else:
            next_pos = len(price_wide.index)

        for pos in range(current_pos + 1, next_pos):
            date = price_wide.index[pos]
            returns = (price_wide.iloc[pos] / price_wide.iloc[pos - 1] - 1.0).replace([np.inf, -np.inf], np.nan)
            strategy_return = daily_return_and_drift(strategy, returns)
            universe_return = daily_return_and_drift(universe, returns)

            qqq_return = returns.get("QQQ", np.nan)
            xlk_return = returns.get("XLK", np.nan)
            qqq_return = float(qqq_return) if pd.notna(qqq_return) and np.isfinite(qqq_return) else 0.0
            xlk_return = float(xlk_return) if pd.notna(xlk_return) and np.isfinite(xlk_return) else 0.0
            qqq_equity *= 1.0 + qqq_return
            xlk_equity *= 1.0 + xlk_return

            daily_rows.append({
                "Config_ID": "walk_forward_dynamic",
                "Top_N": 0,
                "Portfolio_Mode": "dynamic",
                "Walk_Forward_Round": round_number,
                "Fold": regime_bt.fold_label(date, end_date),
                "Selection_Mode": selection_mode,
                "Selection_Guard_Stage": selection_guard_stage,
                "Fallback_Mode": fallback_mode,
                "Fallback_Triggered": bool(fallback_triggered),
                "Fallback_Asset": fallback_asset,
                "Fallback_Reason": fallback_reason,
                **exposure_fields,
                "Selected_Config_ID": selected_config_id,
                "Selected_Candidate_Family_ID": selected_family_id,
                "Selected_Top_N": selected_top_n,
                "Selected_Portfolio_Mode": selected_mode,
                "Date": date.strftime("%Y-%m-%d"),
                "Rebalanced": False,
                "Strategy_Daily_Return": strategy_return,
                "Universe_Daily_Return": universe_return,
                "QQQ_Daily_Return": qqq_return,
                "XLK_Daily_Return": xlk_return,
                "Strategy_Equity": strategy.equity,
                "Universe_Equity": universe.equity,
                "QQQ_Equity": qqq_equity,
                "XLK_Equity": xlk_equity,
                "Strategy_Cash_Weight": strategy.cash_weight,
                "Universe_Cash_Weight": universe.cash_weight,
                "Strategy_Holdings": int(len(strategy.weights)),
                "Universe_Holdings": int(len(universe.weights)),
                "Strategy_Total_Cost": 0.0,
                "Universe_Total_Cost": 0.0,
            })

    dynamic_daily = pd.DataFrame(daily_rows)
    dynamic_rebalances = pd.DataFrame(rebalance_rows)
    dynamic_summary = summarize_simulation(dynamic_daily, dynamic_rebalances)
    if not dynamic_summary.empty:
        fallback_count = (
            int(dynamic_rebalances["Fallback_Triggered"].fillna(False).astype(bool).sum())
            if "Fallback_Triggered" in dynamic_rebalances.columns else 0
        )
        fallback_assets = (
            ",".join(
                f"{asset}:{count}"
                for asset, count in dynamic_rebalances.loc[
                    dynamic_rebalances["Fallback_Triggered"].fillna(False).astype(bool),
                    "Fallback_Asset",
                ].value_counts().sort_index().items()
            )
            if fallback_count else ""
        )
        dynamic_summary["Fallback_Mode"] = fallback_mode
        dynamic_summary["Fallback_Rebalance_Count"] = fallback_count
        dynamic_summary["Fallback_Triggered_Pct"] = (
            fallback_count / len(dynamic_rebalances) if len(dynamic_rebalances) else 0.0
        )
        dynamic_summary["Fallback_Asset_Counts"] = fallback_assets
        if "Exposure_Source" in dynamic_rebalances.columns:
            exposure_counts = dynamic_rebalances["Exposure_Source"].value_counts().sort_index()
            stock_selection_count = int(exposure_counts.get("stock_selection_alpha", 0))
            benchmark_fallback_count = int(exposure_counts.get("benchmark_etf_fallback", 0))
            cash_fallback_count = int(exposure_counts.get("cash_fallback", 0))
            dynamic_summary["Stock_Selection_Rebalance_Count"] = stock_selection_count
            dynamic_summary["Stock_Selection_Rebalance_Pct"] = (
                stock_selection_count / len(dynamic_rebalances) if len(dynamic_rebalances) else 0.0
            )
            dynamic_summary["Benchmark_Fallback_Rebalance_Count"] = benchmark_fallback_count
            dynamic_summary["Benchmark_Fallback_Rebalance_Pct"] = (
                benchmark_fallback_count / len(dynamic_rebalances) if len(dynamic_rebalances) else 0.0
            )
            dynamic_summary["Cash_Fallback_Rebalance_Count"] = cash_fallback_count
            dynamic_summary["Cash_Fallback_Rebalance_Pct"] = (
                cash_fallback_count / len(dynamic_rebalances) if len(dynamic_rebalances) else 0.0
            )
            dynamic_summary["Exposure_Source_Counts"] = ",".join(
                f"{source}:{int(count)}" for source, count in exposure_counts.items()
            )
    return dynamic_daily, dynamic_rebalances, dynamic_summary


def build_latest_holdings(
    scored: pd.DataFrame,
    price_wide: pd.DataFrame,
    adv20: pd.DataFrame,
    params: SimulationParams,
    top_n_list: list[int],
    modes: list[str],
    config: weight_tuning.WeightConfig | DirectScoreConfig,
    historical_universe: HistoricalUniverse | None,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    latest_anchor = pd.to_datetime(scored["Date"].max()).normalize()
    latest_day = scored[scored["Date"] == latest_anchor].copy()
    latest_trade_date = next_trade_date(latest_anchor, price_wide.index, params.rebalance_delay_days)
    if latest_trade_date is None:
        latest_trade_date = price_wide.index.max()

    rows = []
    for top_n in top_n_list:
        for mode in modes:
            target = build_target(
                ranked=latest_day,
                anchor=latest_anchor,
                trade_date=latest_trade_date,
                top_n=top_n,
                mode=mode,
                price_wide=price_wide,
                adv20=adv20,
                params=params,
                equity_dollars=params.initial_capital,
                historical_universe=historical_universe,
            )
            selected = target.selected.copy()
            if selected.empty:
                continue
            selected["Universe_Filter_Source"] = target.universe_source
            selected["Universe_Start_Count"] = target.universe_start_count
            selected["Excluded_By_Universe"] = target.excluded_by_universe
            selected["Config_ID"] = config.config_id
            selected["Anchor_Date"] = latest_anchor.strftime("%Y-%m-%d")
            selected["Trade_Date"] = latest_trade_date.strftime("%Y-%m-%d")
            selected["Top_N"] = top_n
            selected["Portfolio_Mode"] = mode
            selected["Portfolio_Rank"] = np.arange(1, len(selected) + 1)
            rows.extend(selected[
                [
                    col for col in [
                        "Config_ID",
                        "Anchor_Date",
                        "Trade_Date",
                        "Top_N",
                        "Portfolio_Mode",
                        "Portfolio_Rank",
                        "Ticker",
                        "SubIndustry",
                        "Tuned_Combined_Score",
                        "Tuned_Fair_Value_Score",
                        "Trend_Score_100",
                        "Trend_Weight",
                        "Fair_Value_Weight",
                        "Raw_Weight",
                        "Target_Weight",
                        "Universe_Filter_Source",
                        "Universe_Start_Count",
                        "Excluded_By_Universe",
                        "ADV20_Dollar",
                        "Target_Position_ADV_Pct",
                        "Industry_Regime",
                        "SubIndustry_Regime",
                    ]
                    if col in selected.columns
                ]
            ].to_dict("records"))
    return pd.DataFrame(rows)


def simulate_variant(
    scored: pd.DataFrame,
    config_id: str,
    top_n: int,
    mode: str,
    price_wide: pd.DataFrame,
    adv20: pd.DataFrame,
    params: SimulationParams,
    historical_universe: HistoricalUniverse | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = build_rebalance_schedule(scored, price_wide.index, params.rebalance_delay_days)
    performance_schedule = [
        item for item in schedule
        if price_wide.index.get_loc(item[1]) < len(price_wide.index) - 1
    ]
    if not performance_schedule:
        return pd.DataFrame(), pd.DataFrame()

    strategy = PortfolioState()
    universe = PortfolioState()
    qqq_equity = 1.0
    xlk_equity = 1.0
    daily_rows: list[dict] = []
    rebalance_rows: list[dict] = []

    for idx, (anchor, trade_date, day) in enumerate(performance_schedule):
        ranked = day.sort_values("Tuned_Combined_Score", ascending=False).copy()
        strategy_target = build_target(
            ranked=ranked,
            anchor=anchor,
            trade_date=trade_date,
            top_n=top_n,
            mode=mode,
            price_wide=price_wide,
            adv20=adv20,
            params=params,
            equity_dollars=strategy.equity * params.initial_capital,
            historical_universe=historical_universe,
        )
        universe_target = build_target(
            ranked=ranked,
            anchor=anchor,
            trade_date=trade_date,
            top_n=None,
            mode="equal",
            price_wide=price_wide,
            adv20=adv20,
            params=params,
            equity_dollars=universe.equity * params.initial_capital,
            historical_universe=historical_universe,
        )
        strategy_rebalance = rebalance_portfolio(strategy, strategy_target.weights, trade_date, adv20, params)
        universe_rebalance = rebalance_portfolio(universe, universe_target.weights, trade_date, adv20, params)
        rebalance_rows.append({
            "Config_ID": config_id,
            "Top_N": top_n,
            "Portfolio_Mode": mode,
            "Rebalance_Number": idx + 1,
            "Anchor_Date": anchor.strftime("%Y-%m-%d"),
            "Trade_Date": trade_date.strftime("%Y-%m-%d"),
            "Strategy_Equity_After_Rebalance": strategy.equity,
            "Universe_Equity_After_Rebalance": universe.equity,
            "Strategy_Holdings": int(len(strategy.weights)),
            "Universe_Holdings": int(len(universe.weights)),
            "Strategy_Cash_Weight": strategy.cash_weight,
            "Universe_Cash_Weight": universe.cash_weight,
            "Strategy_Turnover": strategy_rebalance["Turnover"],
            "Strategy_Gross_Trade_Weight": strategy_rebalance["Gross_Trade_Weight"],
            "Strategy_Transaction_Cost": strategy_rebalance["Transaction_Cost"],
            "Strategy_Slippage_Cost": strategy_rebalance["Slippage_Cost"],
            "Strategy_Total_Cost": strategy_rebalance["Total_Cost"],
            "Strategy_Max_Trade_ADV_Pct": strategy_rebalance["Max_Trade_ADV_Pct"],
            "Strategy_Avg_Trade_ADV_Pct": strategy_rebalance["Avg_Trade_ADV_Pct"],
            "Strategy_Trade_ADV_Breach_Count": strategy_rebalance["Trade_ADV_Breach_Count"],
            "Strategy_Universe_Start_Count": strategy_target.universe_start_count,
            "Strategy_Excluded_By_Universe": strategy_target.excluded_by_universe,
            "Strategy_Universe_Filter_Source": strategy_target.universe_source,
            "Strategy_Eligible_Count": strategy_target.eligible_count,
            "Strategy_Excluded_By_Liquidity": strategy_target.excluded_by_liquidity,
            "Strategy_Position_Cap_Bind_Count": strategy_target.position_cap_bind_count,
            "Strategy_Subindustry_Cap_Bind_Count": strategy_target.subindustry_cap_bind_count,
            "Strategy_ADV_Cap_Bind_Count": strategy_target.adv_cap_bind_count,
            "Universe_Turnover": universe_rebalance["Turnover"],
            "Universe_Gross_Trade_Weight": universe_rebalance["Gross_Trade_Weight"],
            "Universe_Transaction_Cost": universe_rebalance["Transaction_Cost"],
            "Universe_Slippage_Cost": universe_rebalance["Slippage_Cost"],
            "Universe_Total_Cost": universe_rebalance["Total_Cost"],
            "Universe_Universe_Start_Count": universe_target.universe_start_count,
            "Universe_Excluded_By_Universe": universe_target.excluded_by_universe,
            "Universe_Universe_Filter_Source": universe_target.universe_source,
            "Universe_Eligible_Count": universe_target.eligible_count,
            "Universe_Excluded_By_Liquidity": universe_target.excluded_by_liquidity,
        })
        daily_rows.append({
            "Config_ID": config_id,
            "Top_N": top_n,
            "Portfolio_Mode": mode,
            "Date": trade_date.strftime("%Y-%m-%d"),
            "Rebalanced": True,
            "Strategy_Daily_Return": -strategy_rebalance["Total_Cost"],
            "Universe_Daily_Return": -universe_rebalance["Total_Cost"],
            "QQQ_Daily_Return": 0.0,
            "XLK_Daily_Return": 0.0,
            "Strategy_Equity": strategy.equity,
            "Universe_Equity": universe.equity,
            "QQQ_Equity": qqq_equity,
            "XLK_Equity": xlk_equity,
            "Strategy_Cash_Weight": strategy.cash_weight,
            "Universe_Cash_Weight": universe.cash_weight,
            "Strategy_Holdings": int(len(strategy.weights)),
            "Universe_Holdings": int(len(universe.weights)),
            "Strategy_Total_Cost": strategy_rebalance["Total_Cost"],
            "Universe_Total_Cost": universe_rebalance["Total_Cost"],
        })

        current_pos = int(price_wide.index.get_loc(trade_date))
        if idx + 1 < len(performance_schedule):
            next_trade_date_value = performance_schedule[idx + 1][1]
            next_pos = int(price_wide.index.get_loc(next_trade_date_value))
        else:
            next_pos = len(price_wide.index)

        for pos in range(current_pos + 1, next_pos):
            date = price_wide.index[pos]
            returns = (price_wide.iloc[pos] / price_wide.iloc[pos - 1] - 1.0).replace([np.inf, -np.inf], np.nan)
            strategy_return = daily_return_and_drift(strategy, returns)
            universe_return = daily_return_and_drift(universe, returns)

            qqq_return = returns.get("QQQ", np.nan)
            xlk_return = returns.get("XLK", np.nan)
            qqq_return = float(qqq_return) if pd.notna(qqq_return) and np.isfinite(qqq_return) else 0.0
            xlk_return = float(xlk_return) if pd.notna(xlk_return) and np.isfinite(xlk_return) else 0.0
            qqq_equity *= 1.0 + qqq_return
            xlk_equity *= 1.0 + xlk_return

            daily_rows.append({
                "Config_ID": config_id,
                "Top_N": top_n,
                "Portfolio_Mode": mode,
                "Date": date.strftime("%Y-%m-%d"),
                "Rebalanced": False,
                "Strategy_Daily_Return": strategy_return,
                "Universe_Daily_Return": universe_return,
                "QQQ_Daily_Return": qqq_return,
                "XLK_Daily_Return": xlk_return,
                "Strategy_Equity": strategy.equity,
                "Universe_Equity": universe.equity,
                "QQQ_Equity": qqq_equity,
                "XLK_Equity": xlk_equity,
                "Strategy_Cash_Weight": strategy.cash_weight,
                "Universe_Cash_Weight": universe.cash_weight,
                "Strategy_Holdings": int(len(strategy.weights)),
                "Universe_Holdings": int(len(universe.weights)),
                "Strategy_Total_Cost": 0.0,
                "Universe_Total_Cost": 0.0,
            })

    return pd.DataFrame(daily_rows), pd.DataFrame(rebalance_rows)


def simulate_all(
    scores: pd.DataFrame,
    configs: list[weight_tuning.WeightConfig | DirectScoreConfig],
    top_n_list: list[int],
    modes: list[str],
    price_wide: pd.DataFrame,
    adv20: pd.DataFrame,
    params: SimulationParams,
    historical_universe: HistoricalUniverse | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_frames = []
    rebalance_frames = []
    latest_frames = []

    total_variants = len(configs) * len(top_n_list) * len(modes)
    variant_idx = 0
    for config in configs:
        scored = score_frame_for_config(scores, config)
        if scored.empty:
            continue
        latest_frames.append(build_latest_holdings(
            scored=scored,
            price_wide=price_wide,
            adv20=adv20,
            params=params,
            top_n_list=top_n_list,
            modes=modes,
            config=config,
            historical_universe=historical_universe,
        ))
        for top_n in top_n_list:
            for mode in modes:
                variant_idx += 1
                print(
                    f"[INFO] Simulating {variant_idx}/{total_variants}: "
                    f"{config.config_id} | top {top_n} | {mode}",
                    flush=True,
                )
                daily, rebalances = simulate_variant(
                    scored=scored,
                    config_id=config.config_id,
                    top_n=top_n,
                    mode=mode,
                    price_wide=price_wide,
                    adv20=adv20,
                    params=params,
                    historical_universe=historical_universe,
                )
                daily_frames.append(daily)
                rebalance_frames.append(rebalances)

    daily = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()
    rebalances = pd.concat(rebalance_frames, ignore_index=True, sort=False) if rebalance_frames else pd.DataFrame()
    latest_holdings = pd.concat(latest_frames, ignore_index=True, sort=False) if latest_frames else pd.DataFrame()
    return daily, rebalances, latest_holdings


def format_pct(value: object) -> str:
    return construction.format_pct(value)


def format_num(value: object, digits: int = 4) -> str:
    return construction.format_num(value, digits)


def write_summary(
    output_dir: Path,
    summary: pd.DataFrame,
    rebalances: pd.DataFrame,
    historical_universe: HistoricalUniverse,
    walk_forward_selection: pd.DataFrame,
    walk_forward_summary: pd.DataFrame,
    metadata: dict,
) -> None:
    top = summary.iloc[0].to_dict() if not summary.empty else {}
    current = (
        summary[summary["Config_ID"] == "current_live"].iloc[0].to_dict()
        if not summary.empty and "current_live" in set(summary["Config_ID"])
        else {}
    )
    walk_forward = walk_forward_summary.iloc[0].to_dict() if not walk_forward_summary.empty else {}

    lines = [
        "# Monthly Rebalanced Portfolio Simulation Summary",
        "",
        "## What was completed",
        "",
        "This run converted the stock-ranking output into sequential monthly portfolio equity curves.",
        "Unlike the earlier forward-return tests, this simulator compounds one portfolio through time and applies rebalancing friction each month.",
        "",
        "The simulator tested:",
        "",
        f"- Candidate score-weight configs: {metadata.get('candidate_config_count')}",
        f"- Direct alpha score columns: {metadata.get('alpha_score_columns') or []}",
        f"- Direct alpha configs: {metadata.get('direct_alpha_config_count') or 0}",
        f"- Portfolio sizes: {metadata.get('top_n_list')}",
        f"- Portfolio modes: {metadata.get('portfolio_modes')}",
        f"- Initial capital assumption: ${metadata.get('initial_capital'):,.0f}",
        f"- Rebalance delay: {metadata.get('rebalance_delay_days')} trading day after each score anchor",
        f"- Minimum 20-day dollar volume: ${metadata.get('min_dollar_volume'):,.0f}",
        f"- Max position weight: {format_pct(metadata.get('max_position_weight'))}",
        f"- Max subindustry weight: {format_pct(metadata.get('max_subindustry_weight'))}",
        f"- Max position as percent of 20-day dollar volume: {format_pct(metadata.get('max_position_adv_pct'))}",
        f"- Transaction cost: {metadata.get('transaction_cost_bps')} bps of gross traded value",
        f"- Slippage model: {metadata.get('slippage_bps_per_1pct_adv')} bps per 1% of ADV traded",
        f"- Historical universe control: `{historical_universe.source}`",
        "",
        "## Best simulated construction",
        "",
    ]

    if top:
        lines.extend([
            f"- Construction: `{top.get('Config_ID')}` / top {int(top.get('Top_N'))} / `{top.get('Portfolio_Mode')}`",
            f"- Final equity: {format_num(top.get('Final_Equity'), 4)}",
            f"- Total return: {format_pct(top.get('Total_Return'))}",
            f"- CAGR: {format_pct(top.get('CAGR'))}",
            f"- Excess return vs simulated universe: {format_pct(top.get('Excess_Return_vs_Universe'))}",
            f"- Excess return vs QQQ: {format_pct(top.get('Excess_Return_vs_QQQ'))}",
            f"- Excess return vs XLK: {format_pct(top.get('Excess_Return_vs_XLK'))}",
            f"- Sharpe: {format_num(top.get('Sharpe'))}",
            f"- Max drawdown: {format_pct(top.get('Max_Drawdown'))}",
            f"- Average turnover: {format_pct(top.get('Avg_Turnover'))}",
            f"- Total estimated cost drag: {format_pct(top.get('Total_Cost'))}",
            f"- Average cash weight: {format_pct(top.get('Avg_Cash_Weight'))}",
            "",
        ])
    else:
        lines.extend(["No simulation summary was produced.", ""])

    lines.extend(["## Current production weights", ""])
    if current:
        lines.extend([
            f"- Best construction using `current_live`: top {int(current.get('Top_N'))} / `{current.get('Portfolio_Mode')}`",
            f"- Final equity: {format_num(current.get('Final_Equity'), 4)}",
            f"- Total return: {format_pct(current.get('Total_Return'))}",
            f"- CAGR: {format_pct(current.get('CAGR'))}",
            f"- Excess return vs simulated universe: {format_pct(current.get('Excess_Return_vs_Universe'))}",
            f"- Excess return vs XLK: {format_pct(current.get('Excess_Return_vs_XLK'))}",
            f"- Sharpe: {format_num(current.get('Sharpe'))}",
            f"- Max drawdown: {format_pct(current.get('Max_Drawdown'))}",
            f"- Average turnover: {format_pct(current.get('Avg_Turnover'))}",
            "",
        ])
    else:
        lines.extend(["The `current_live` config was not found in this simulation.", ""])

    lines.extend([
        "## Survivorship-bias control",
        "",
        historical_universe.note,
        "",
        f"- Membership rows: {metadata.get('historical_universe_membership_rows')}",
        f"- Audit rows: {metadata.get('historical_universe_audit_rows')}",
        f"- Tickers excluded by universe filter per rebalance, average: {format_num(summary['Avg_Excluded_By_Universe'].mean(), 2) if not summary.empty and 'Avg_Excluded_By_Universe' in summary.columns else 'n.a.'}",
        "",
    ])

    lines.extend(["## Sequential walk-forward validation", ""])
    if not walk_forward_selection.empty:
        lines.append(f"Selection mode: `{metadata.get('walk_forward_selection_mode', 'objective')}`")
        if metadata.get("walk_forward_selection_mode") in {"guarded", "guarded_defensive", "guarded_family_stable"}:
            lines.extend([
                "Guard defaults:",
                f"- Minimum training excess vs XLK when available: {format_pct(metadata.get('guard_min_excess_vs_xlk'))}",
                f"- Maximum allowed training drawdown when available: {format_pct(metadata.get('guard_max_drawdown'))}",
                f"- Minimum training Sharpe when available: {format_num(metadata.get('guard_min_sharpe'))}",
                f"- Maximum average turnover when available: {format_pct(metadata.get('guard_max_turnover')) if metadata.get('guard_max_turnover') is not None else 'none'}",
                "",
            ])
        if metadata.get("fallback_mode") and metadata.get("fallback_mode") != "none":
            lines.extend([
                "Fallback policy:",
                f"- Fallback mode: `{metadata.get('fallback_mode')}`",
                f"- Fallback exposure: {format_pct(metadata.get('fallback_exposure'))}",
                f"- Clean guard stages: `{','.join(metadata.get('fallback_clean_guard_stages', []))}`",
                "",
            ])
        if metadata.get("walk_forward_selection_mode") in {"guarded_defensive", "guarded_family_stable"}:
            lines.extend([
                "Recent-fold and defensive guard defaults:",
                f"- Recent training fold count: {metadata.get('guard_recent_fold_count')}",
                f"- Minimum recent excess vs universe: {format_pct(metadata.get('guard_min_recent_excess_vs_universe'))}",
                f"- Minimum recent excess vs XLK: {format_pct(metadata.get('guard_min_recent_excess_vs_xlk'))}",
                f"- Maximum recent-fold drawdown: {format_pct(metadata.get('guard_max_recent_drawdown'))}",
                f"- Maximum worst training-fold drawdown: {format_pct(metadata.get('guard_max_fold_drawdown'))}",
                f"- Maximum average downside capture vs XLK: {format_num(metadata.get('guard_max_downside_capture_vs_xlk'))}",
                f"- Minimum positive universe-fold rate: {format_pct(metadata.get('guard_min_positive_universe_fold_rate'))}",
                f"- Minimum positive XLK-fold rate: {format_pct(metadata.get('guard_min_positive_xlk_fold_rate'))}",
                "",
            ])
        if metadata.get("walk_forward_selection_mode") == "guarded_family_stable":
            selection_params = metadata.get("walk_forward_selection_params", {}) or {}
            lines.extend([
                "Candidate-family stability guard defaults:",
                f"- Minimum family training folds: {selection_params.get('min_family_train_folds')}",
                f"- Minimum family recent excess vs universe: {format_pct(selection_params.get('min_family_recent_excess_vs_universe'))}",
                f"- Minimum family recent excess vs XLK: {format_pct(selection_params.get('min_family_recent_excess_vs_xlk'))}",
                f"- Maximum family worst-fold drawdown: {format_pct(selection_params.get('max_family_worst_drawdown'))}",
                f"- Maximum family average downside capture vs XLK: {format_num(selection_params.get('max_family_downside_capture_vs_xlk'))}",
                f"- Minimum family positive universe-fold rate: {format_pct(selection_params.get('min_family_positive_universe_fold_rate'))}",
                f"- Minimum family positive XLK-fold rate: {format_pct(selection_params.get('min_family_positive_xlk_fold_rate'))}",
                f"- Family stability objective weight: {format_num(selection_params.get('family_stability_weight'))}",
                "",
            ])
        lines.append("Selected construction by unseen validation fold:")
        for _, row in walk_forward_selection.iterrows():
            lines.append(
                "- "
                f"{row['Validation_Fold']}: `{row['Selected_Config_ID']}` / "
                f"family `{row.get('Selected_Candidate_Family_ID', '')}` / "
                f"top {int(row['Selected_Top_N'])} / `{row['Selected_Portfolio_Mode']}` "
                f"after training on {row['Train_Folds']} "
                f"(guard stage: `{row.get('Selection_Guard_Stage', 'legacy_objective')}`)"
            )
        lines.append("")
    else:
        lines.extend(["Walk-forward selection did not produce enough fold data.", ""])

    if walk_forward:
        lines.extend([
            "Dynamic walk-forward portfolio result:",
            f"- Final equity: {format_num(walk_forward.get('Final_Equity'), 4)}",
            f"- Total return: {format_pct(walk_forward.get('Total_Return'))}",
            f"- CAGR: {format_pct(walk_forward.get('CAGR'))}",
            f"- Excess return vs simulated universe: {format_pct(walk_forward.get('Excess_Return_vs_Universe'))}",
            f"- Excess return vs XLK: {format_pct(walk_forward.get('Excess_Return_vs_XLK'))}",
            f"- Sharpe: {format_num(walk_forward.get('Sharpe'))}",
            f"- Max drawdown: {format_pct(walk_forward.get('Max_Drawdown'))}",
            f"- Average turnover: {format_pct(walk_forward.get('Avg_Turnover'))}",
            f"- Average cash weight: {format_pct(walk_forward.get('Avg_Cash_Weight'))}",
            f"- Fallback rebalances: {int(walk_forward.get('Fallback_Rebalance_Count', 0))}",
            f"- Stock-selection rebalances: {int(walk_forward.get('Stock_Selection_Rebalance_Count', 0))}",
            f"- Benchmark fallback rebalances: {int(walk_forward.get('Benchmark_Fallback_Rebalance_Count', 0))}",
            f"- Cash fallback rebalances: {int(walk_forward.get('Cash_Fallback_Rebalance_Count', 0))}",
            "",
        ])
    else:
        lines.extend(["The dynamic walk-forward portfolio did not produce an equity curve.", ""])

    lines.extend(["## Top simulated candidates", ""])
    if not summary.empty:
        for _, row in summary.head(10).iterrows():
            lines.append(
                "- "
                f"`{row['Config_ID']}` / top {int(row['Top_N'])} / `{row['Portfolio_Mode']}`: "
                f"total return {format_pct(row['Total_Return'])}, "
                f"excess vs universe {format_pct(row['Excess_Return_vs_Universe'])}, "
                f"excess vs XLK {format_pct(row['Excess_Return_vs_XLK'])}, "
                f"Sharpe {format_num(row['Sharpe'])}, "
                f"drawdown {format_pct(row['Max_Drawdown'])}, "
                f"turnover {format_pct(row['Avg_Turnover'])}"
            )
        lines.append("")

    if historical_universe.source == "external_historical_universe":
        universe_interpretation = (
            "This run used an external historical-universe file. That is an improvement over "
            "the earlier score/price proxy, but it is only as complete as the supplied universe. "
            "If the point-in-time score history was rebuilt with that same universe, missing "
            "score rows now usually indicate unavailable historical prices, too little usable "
            "daily history, or missing fair-value inputs rather than a simulator universe bug. "
            "The combined-score historical-universe audit should be checked before interpreting "
            "the result as survivorship-clean."
        )
    else:
        universe_interpretation = (
            "If no external historical universe file is supplied, the simulator can only use a "
            "score/price availability proxy. That documents and reduces one obvious form of "
            "lookback error, but it does not restore companies that disappeared before today's "
            "ticker list was built."
        )

    lines.extend([
        "## Interpretation",
        "",
        "This is the first test that behaves like an actual monthly strategy. It uses one-day-delayed rebalances, lets weights drift between rebalances, charges estimated trading costs, applies liquidity filters, and keeps cash when caps prevent full allocation.",
        "",
        "Because this simulator compounds through time, its result should carry more weight than the earlier overlapping forward-return averages. A construction that looked excellent in overlapping windows can weaken here if it relies on high turnover, illiquid names, or unstable one-period winners.",
        "",
        f"This still does not make the strategy production-ready. {universe_interpretation} The liquidity and slippage assumptions are also simplified and should be stress-tested with larger costs and different account sizes.",
        "",
        "When direct alpha columns are included, they are research-only candidates. They prove whether a candidate score survives portfolio mechanics, but they should not replace production scoring until walk-forward selection, drawdown behavior, and benchmark-relative performance are stable.",
        "",
        "## Files produced",
        "",
        "- `monthly_equity_curves.csv`",
        "- `rebalance_log.csv`",
        "- `simulation_summary.csv`",
        "- `latest_target_holdings.csv`",
        "- `historical_universe_membership.csv`",
        "- `survivorship_universe_audit.csv`",
        "- `walk_forward_selected_constructions.csv`",
        "- `walk_forward_dynamic_equity_curve.csv`",
        "- `walk_forward_dynamic_rebalance_log.csv`",
        "- `walk_forward_dynamic_summary.csv`",
        "- `recommended_rebalanced_portfolio.json`",
        "- `run_metadata.json`",
        "- `MONTHLY_REBALANCED_PORTFOLIO_SUMMARY.md`",
    ])

    if not rebalances.empty:
        lines.extend([
            "",
            "## Rebalance diagnostics",
            "",
            f"- Rebalance rows: {len(rebalances):,}",
            f"- Average selected holdings: {format_num(rebalances['Strategy_Holdings'].mean(), 2)}",
            f"- Average names excluded by universe filter: {format_num(rebalances['Strategy_Excluded_By_Universe'].mean(), 2) if 'Strategy_Excluded_By_Universe' in rebalances.columns else 'n.a.'}",
            f"- Average names excluded by liquidity: {format_num(rebalances['Strategy_Excluded_By_Liquidity'].mean(), 2)}",
            f"- Rebalances with position caps binding: {int((rebalances['Strategy_Position_Cap_Bind_Count'] > 0).sum()):,}",
            f"- Rebalances with subindustry caps binding: {int((rebalances['Strategy_Subindustry_Cap_Bind_Count'] > 0).sum()):,}",
            f"- Rebalances with ADV position caps binding: {int((rebalances['Strategy_ADV_Cap_Bind_Count'] > 0).sum()):,}",
            f"- Rebalances with trade ADV breaches: {int((rebalances['Strategy_Trade_ADV_Breach_Count'] > 0).sum()):,}",
        ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "MONTHLY_REBALANCED_PORTFOLIO_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequential monthly rebalanced portfolio simulations.")
    parser.add_argument("--score-history", type=Path, default=DEFAULT_SCORE_HISTORY)
    parser.add_argument("--weight-tuning-dir", type=Path, default=DEFAULT_WEIGHT_TUNING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--historical-universe",
        type=Path,
        help=(
            "Optional CSV with Ticker plus optional Membership_Start_Date and "
            "Membership_End_Date columns. If omitted, a score/price availability "
            "proxy universe is used and reported as an unresolved survivorship limitation."
        ),
    )
    parser.add_argument(
        "--historical-price-file",
        type=Path,
        help=(
            "Optional canonical daily price CSV from historical_data_layer.py. "
            "Use this for vendor-grade active/delisted historical prices instead of Yahoo."
        ),
    )
    parser.add_argument("--candidate-preset", choices=["compact", "expanded"], default="expanded")
    parser.add_argument("--config-scope", choices=["shortlist", "all"], default="shortlist")
    parser.add_argument("--shortlist-count", type=int, default=10)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument(
        "--alpha-score-columns",
        default="",
        help=(
            "Comma-separated research score columns to rank directly in addition to the "
            "normal trend/fair-value weight configs. These columns are generated from the "
            "point-in-time score history and canonical prices using alpha_signal_quality.py helpers."
        ),
    )
    parser.add_argument(
        "--walk-forward-selection-mode",
        choices=["objective", "guarded", "guarded_defensive", "guarded_family_stable"],
        default="objective",
        help=(
            "Use the legacy simulator objective, a benchmark/drawdown-aware guarded objective, "
            "a fold-consistency/downside-aware defensive guarded objective, or a candidate-family "
            "stability objective when selecting constructions on prior folds."
        ),
    )
    parser.add_argument("--guard-min-excess-vs-xlk", type=float, default=0.0)
    parser.add_argument("--guard-max-drawdown", type=float, default=0.35)
    parser.add_argument("--guard-min-sharpe", type=float, default=0.0)
    parser.add_argument(
        "--guard-max-turnover",
        type=float,
        default=0.70,
        help="Maximum average turnover for guarded selection. Use a negative value to disable this guard.",
    )
    parser.add_argument("--guard-recent-fold-count", type=int, default=1)
    parser.add_argument("--guard-min-recent-excess-vs-universe", type=float, default=0.0)
    parser.add_argument("--guard-min-recent-excess-vs-xlk", type=float, default=-0.10)
    parser.add_argument("--guard-max-recent-drawdown", type=float, default=0.22)
    parser.add_argument("--guard-max-fold-drawdown", type=float, default=0.28)
    parser.add_argument("--guard-max-downside-capture-vs-xlk", type=float, default=0.95)
    parser.add_argument("--guard-min-positive-universe-fold-rate", type=float, default=0.50)
    parser.add_argument("--guard-min-positive-xlk-fold-rate", type=float, default=0.25)
    parser.add_argument("--guard-min-family-train-folds", type=int, default=2)
    parser.add_argument("--guard-min-family-recent-excess-vs-universe", type=float, default=0.0)
    parser.add_argument("--guard-min-family-recent-excess-vs-xlk", type=float, default=-0.05)
    parser.add_argument("--guard-max-family-worst-drawdown", type=float, default=0.30)
    parser.add_argument("--guard-max-family-downside-capture-vs-xlk", type=float, default=0.95)
    parser.add_argument("--guard-min-family-positive-universe-fold-rate", type=float, default=0.75)
    parser.add_argument("--guard-min-family-positive-xlk-fold-rate", type=float, default=0.50)
    parser.add_argument("--guard-family-stability-weight", type=float, default=0.35)
    parser.add_argument(
        "--fallback-mode",
        choices=["none", "cash", "qqq", "xlk"],
        default="none",
        help=(
            "Optional dynamic walk-forward fallback. When enabled, validation folds whose "
            "selected guard stage is not in --fallback-clean-stages hold cash, QQQ, or XLK."
        ),
    )
    parser.add_argument(
        "--fallback-clean-stages",
        default=(
            "family_stable_strict_recent_xlk_drawdown_downside,"
            "family_stable_recent_defensive_downside,"
            "strict_recent_xlk_drawdown_downside,"
            "recent_defensive_downside"
        ),
        help="Comma-separated guard stages that are considered clean enough to avoid fallback.",
    )
    parser.add_argument("--fallback-exposure", type=float, default=1.0)
    parser.add_argument("--top-n-list", default=DEFAULT_TOP_N_LIST)
    parser.add_argument("--portfolio-modes", default=DEFAULT_MODES)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--rebalance-delay-days", type=int, default=1)
    parser.add_argument("--min-dollar-volume", type=float, default=30_000_000.0)
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--max-position-weight", type=float, default=0.15)
    parser.add_argument("--max-subindustry-weight", type=float, default=0.35)
    parser.add_argument("--max-position-adv-pct", type=float, default=0.02)
    parser.add_argument("--max-trade-adv-pct", type=float, default=0.01)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps-per-1pct-adv", type=float, default=5.0)
    args = parser.parse_args()

    if args.initial_capital <= 0:
        raise ValueError("initial capital must be positive")
    if args.adv_window <= 0:
        raise ValueError("ADV window must be positive")
    if args.guard_recent_fold_count <= 0:
        raise ValueError("guard recent fold count must be positive")
    if args.guard_min_family_train_folds <= 0:
        raise ValueError("guard minimum family training folds must be positive")
    if args.guard_family_stability_weight < 0:
        raise ValueError("guard family stability weight cannot be negative")
    if args.fallback_exposure < 0:
        raise ValueError("fallback exposure cannot be negative")

    params = SimulationParams(
        initial_capital=args.initial_capital,
        rebalance_delay_days=args.rebalance_delay_days,
        min_dollar_volume=args.min_dollar_volume,
        adv_window=args.adv_window,
        max_position_weight=args.max_position_weight,
        max_subindustry_weight=args.max_subindustry_weight,
        max_position_adv_pct=args.max_position_adv_pct,
        max_trade_adv_pct=args.max_trade_adv_pct,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps_per_1pct_adv=args.slippage_bps_per_1pct_adv,
    )
    top_n_list = construction.parse_top_n_list(args.top_n_list)
    modes = construction.parse_modes(args.portfolio_modes)
    alpha_score_columns = parse_score_columns(args.alpha_score_columns)
    selection_params = WalkForwardSelectionParams(
        mode=args.walk_forward_selection_mode,
        min_excess_vs_xlk=args.guard_min_excess_vs_xlk,
        max_drawdown=args.guard_max_drawdown,
        min_sharpe=args.guard_min_sharpe,
        max_turnover=None if args.guard_max_turnover < 0 else args.guard_max_turnover,
        recent_fold_count=args.guard_recent_fold_count,
        min_recent_excess_vs_universe=args.guard_min_recent_excess_vs_universe,
        min_recent_excess_vs_xlk=args.guard_min_recent_excess_vs_xlk,
        max_recent_drawdown=args.guard_max_recent_drawdown,
        max_fold_drawdown=args.guard_max_fold_drawdown,
        max_downside_capture_vs_xlk=args.guard_max_downside_capture_vs_xlk,
        min_positive_universe_fold_rate=args.guard_min_positive_universe_fold_rate,
        min_positive_xlk_fold_rate=args.guard_min_positive_xlk_fold_rate,
        min_family_train_folds=args.guard_min_family_train_folds,
        min_family_recent_excess_vs_universe=args.guard_min_family_recent_excess_vs_universe,
        min_family_recent_excess_vs_xlk=args.guard_min_family_recent_excess_vs_xlk,
        max_family_worst_drawdown=args.guard_max_family_worst_drawdown,
        max_family_downside_capture_vs_xlk=args.guard_max_family_downside_capture_vs_xlk,
        min_family_positive_universe_fold_rate=args.guard_min_family_positive_universe_fold_rate,
        min_family_positive_xlk_fold_rate=args.guard_min_family_positive_xlk_fold_rate,
        family_stability_weight=args.guard_family_stability_weight,
    )
    fallback_policy = FallbackPolicy(
        mode=args.fallback_mode,
        clean_guard_stages=parse_stage_list(args.fallback_clean_stages),
        exposure=min(1.0, args.fallback_exposure),
    )
    configs = construction.select_configs(
        config_scope=args.config_scope,
        preset=args.candidate_preset,
        weight_tuning_dir=args.weight_tuning_dir,
        shortlist_count=args.shortlist_count,
        max_configs=args.max_configs,
    )
    if not configs:
        raise RuntimeError("No candidate configs selected")

    scores = weight_tuning.prepare_score_history(args.score_history)
    tickers = sorted(set(scores["Ticker"]) | set(BENCHMARK_TICKERS))
    start_date = scores["Date"].min() - pd.Timedelta(days=max(40, params.adv_window * 2))
    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

    print(f"[INFO] Loaded {len(scores):,} score rows across {scores['Date'].nunique()} anchors", flush=True)
    if args.historical_price_file:
        print(f"[INFO] Loading canonical market history from {args.historical_price_file}", flush=True)
        price_wide = historical_data.load_canonical_price_wide(
            path=args.historical_price_file,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            value_column="Close",
        )
        volume_wide = historical_data.load_canonical_price_wide(
            path=args.historical_price_file,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            value_column="Volume",
        )
        market_data_source = str(args.historical_price_file)
    else:
        price_wide, volume_wide = fetch_market_wide(
            tickers=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            batch_size=args.batch_size,
        )
        market_data_source = "yfinance"
    if price_wide.empty:
        raise RuntimeError("No price history returned; cannot simulate portfolio")

    scores, configs = add_alpha_score_configs(
        scores=scores,
        configs=configs,
        score_columns=alpha_score_columns,
        price_wide=price_wide,
    )
    if alpha_score_columns:
        print(f"[INFO] Added alpha score columns: {alpha_score_columns}", flush=True)
    print(f"[INFO] Testing {len(configs)} configs, top-N {top_n_list}, modes {modes}", flush=True)

    adv20 = rolling_dollar_volume(price_wide, volume_wide, params.adv_window)
    historical_universe = load_historical_universe(args.historical_universe, scores, price_wide)
    print(f"[INFO] Historical universe control: {historical_universe.source}", flush=True)
    print(f"[INFO] {historical_universe.note}", flush=True)

    daily, rebalances, latest_holdings = simulate_all(
        scores=scores,
        configs=configs,
        top_n_list=top_n_list,
        modes=modes,
        price_wide=price_wide,
        adv20=adv20,
        params=params,
        historical_universe=historical_universe,
    )
    end_date = pd.to_datetime(price_wide.index.max()).normalize()
    daily = assign_fold_labels(daily, "Date", end_date)
    rebalances = assign_fold_labels(rebalances, "Trade_Date", end_date)
    summary = summarize_simulation(daily, rebalances)
    walk_forward_selection = walk_forward_selection_from_equity(daily, rebalances, selection_params)
    walk_forward_daily, walk_forward_rebalances, walk_forward_summary = simulate_walk_forward_dynamic_portfolio(
        scores=scores,
        configs=configs,
        selections=walk_forward_selection,
        price_wide=price_wide,
        adv20=adv20,
        params=params,
        historical_universe=historical_universe,
        fallback_policy=fallback_policy,
    )

    metadata = {
        "status": "complete",
        "score_history": str(args.score_history),
        "weight_tuning_dir": str(args.weight_tuning_dir),
        "candidate_preset": args.candidate_preset,
        "config_scope": args.config_scope,
        "shortlist_count": args.shortlist_count,
        "candidate_config_count": int(len(configs)),
        "alpha_score_columns": alpha_score_columns,
        "direct_alpha_config_count": int(sum(isinstance(config, DirectScoreConfig) for config in configs)),
        "walk_forward_selection_mode": selection_params.mode,
        "walk_forward_selection_params": asdict(selection_params),
        "guard_min_excess_vs_xlk": selection_params.min_excess_vs_xlk,
        "guard_max_drawdown": selection_params.max_drawdown,
        "guard_min_sharpe": selection_params.min_sharpe,
        "guard_max_turnover": selection_params.max_turnover,
        "guard_recent_fold_count": selection_params.recent_fold_count,
        "guard_min_recent_excess_vs_universe": selection_params.min_recent_excess_vs_universe,
        "guard_min_recent_excess_vs_xlk": selection_params.min_recent_excess_vs_xlk,
        "guard_max_recent_drawdown": selection_params.max_recent_drawdown,
        "guard_max_fold_drawdown": selection_params.max_fold_drawdown,
        "guard_max_downside_capture_vs_xlk": selection_params.max_downside_capture_vs_xlk,
        "guard_min_positive_universe_fold_rate": selection_params.min_positive_universe_fold_rate,
        "guard_min_positive_xlk_fold_rate": selection_params.min_positive_xlk_fold_rate,
        "guard_min_family_train_folds": selection_params.min_family_train_folds,
        "guard_min_family_recent_excess_vs_universe": selection_params.min_family_recent_excess_vs_universe,
        "guard_min_family_recent_excess_vs_xlk": selection_params.min_family_recent_excess_vs_xlk,
        "guard_max_family_worst_drawdown": selection_params.max_family_worst_drawdown,
        "guard_max_family_downside_capture_vs_xlk": selection_params.max_family_downside_capture_vs_xlk,
        "guard_min_family_positive_universe_fold_rate": selection_params.min_family_positive_universe_fold_rate,
        "guard_min_family_positive_xlk_fold_rate": selection_params.min_family_positive_xlk_fold_rate,
        "guard_family_stability_weight": selection_params.family_stability_weight,
        "fallback_mode": fallback_policy.mode,
        "fallback_clean_guard_stages": list(fallback_policy.clean_guard_stages),
        "fallback_exposure": fallback_policy.exposure,
        "top_n_list": top_n_list,
        "portfolio_modes": modes,
        "score_rows": int(len(scores)),
        "anchor_count": int(scores["Date"].nunique()),
        "daily_rows": int(len(daily)),
        "rebalance_rows": int(len(rebalances)),
        "latest_holdings_rows": int(len(latest_holdings)),
        "walk_forward_selection_rows": int(len(walk_forward_selection)),
        "walk_forward_daily_rows": int(len(walk_forward_daily)),
        "walk_forward_rebalance_rows": int(len(walk_forward_rebalances)),
        "market_start_date": price_wide.index.min().strftime("%Y-%m-%d"),
        "market_end_date": price_wide.index.max().strftime("%Y-%m-%d"),
        "market_data_source": market_data_source,
        "historical_universe_path": str(args.historical_universe) if args.historical_universe else None,
        "historical_universe_source": historical_universe.source,
        "historical_universe_note": historical_universe.note,
        "historical_universe_membership_rows": int(len(historical_universe.membership)),
        "historical_universe_audit_rows": int(len(historical_universe.audit)),
        **asdict(params),
        "simulator_note": (
            "Sequential monthly rebalance with one-day delayed trades, buy-and-hold drift "
            "between rebalances, transaction costs, slippage, liquidity filters, concentration caps, "
            "historical-universe filtering, and sequential walk-forward construction selection."
        ),
    }
    if not walk_forward_summary.empty:
        metadata["walk_forward_dynamic_summary"] = walk_forward_summary.iloc[0].to_dict()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(config) for config in configs]).to_csv(args.output_dir / "tested_weight_configs.csv", index=False)
    daily.to_csv(args.output_dir / "monthly_equity_curves.csv", index=False)
    rebalances.to_csv(args.output_dir / "rebalance_log.csv", index=False)
    summary.to_csv(args.output_dir / "simulation_summary.csv", index=False)
    latest_holdings.to_csv(args.output_dir / "latest_target_holdings.csv", index=False)
    historical_universe.membership.to_csv(args.output_dir / "historical_universe_membership.csv", index=False)
    historical_universe.audit.to_csv(args.output_dir / "survivorship_universe_audit.csv", index=False)
    walk_forward_selection.to_csv(args.output_dir / "walk_forward_selected_constructions.csv", index=False)
    walk_forward_daily.to_csv(args.output_dir / "walk_forward_dynamic_equity_curve.csv", index=False)
    walk_forward_rebalances.to_csv(args.output_dir / "walk_forward_dynamic_rebalance_log.csv", index=False)
    walk_forward_summary.to_csv(args.output_dir / "walk_forward_dynamic_summary.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    recommended = {
        "best_simulated_construction": summary.iloc[0].to_dict() if not summary.empty else {},
        "best_current_live_construction": (
            summary[summary["Config_ID"] == "current_live"].iloc[0].to_dict()
            if not summary.empty and "current_live" in set(summary["Config_ID"])
            else {}
        ),
        "walk_forward_dynamic_construction": walk_forward_summary.iloc[0].to_dict() if not walk_forward_summary.empty else {},
        "walk_forward_selected_constructions": walk_forward_selection.to_dict("records"),
        "parameters": metadata,
        "candidate_parameters": {config.config_id: asdict(config) for config in configs},
        "recommendation_note": (
            "Treat this as a stronger validation layer than overlapping forward returns, "
            "but do not use it as production proof until external historical universe membership, "
            "cost sensitivity, and live paper-trading behavior are addressed."
        ),
    }
    (args.output_dir / "recommended_rebalanced_portfolio.json").write_text(
        json.dumps(recommended, indent=2, default=str),
        encoding="utf-8",
    )
    write_summary(
        args.output_dir,
        summary,
        rebalances,
        historical_universe,
        walk_forward_selection,
        walk_forward_summary,
        metadata,
    )

    print("\n=== MONTHLY REBALANCED SIMULATION SCORECARD ===", flush=True)
    columns = [
        "Config_ID",
        "Top_N",
        "Portfolio_Mode",
        "Simulator_Objective",
        "Total_Return",
        "Excess_Return_vs_Universe",
        "Excess_Return_vs_XLK",
        "CAGR",
        "Sharpe",
        "Max_Drawdown",
        "Avg_Turnover",
        "Total_Cost",
    ]
    if not summary.empty:
        print(summary[[col for col in columns if col in summary.columns]].head(15).to_string(index=False), flush=True)

    print(f"\n[SUCCESS] Saved monthly rebalanced portfolio simulation outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
