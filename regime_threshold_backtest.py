#!/usr/bin/env python3
"""
Backtests regime-threshold settings for the Stock Analysis pipeline.

This is intentionally a price/regime calibration harness first. It rebuilds
historical subindustry and industry regimes from daily price data, then tests
whether regime-adjusted trend rankings performed better 3, 6, 9, and 12 months
later. Point-in-time fair-value backtesting should be added after SEC facts are
cached by filing date.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import comp


HORIZONS = {
    "3M": 63,
    "6M": 126,
    "9M": 189,
    "12M": 252,
}

BENCHMARK_TICKERS = ["QQQ", "XLK"]
DEFAULT_OUTPUT_DIR = Path("backtests/regime_thresholds")


@dataclass(frozen=True)
class RegimeParams:
    config_id: str
    notes: str

    rolling_window: int = 5

    structural_bull_above20_min: float = 0.65
    structural_bull_above50_min: float = 0.55
    structural_bull_hh50_min: float = 0.50
    structural_bull_slope_min: float = 0.00
    structural_bull_new_low20_max: float = 0.15

    early_median20_min: float = 0.00
    early_median50_min: float = 0.00
    early_above20_min: float = 0.55
    early_hh20_min: float = 0.40
    early_new_low20_max: float = 0.25

    bear_median20_max: float = 0.00
    bear_above20_max: float = 0.40
    bear_above50_max: float = 0.35
    bear_slope_max: float = 0.00
    bear_new_low20_min: float = 0.30

    stock_flow_bull_pts_min: float = 0.65
    stock_flow_bear_pts_max: float = 0.35
    stock_flow_core_bull_min: float = 0.40
    stock_flow_confirmer_bull_min: float = 0.40
    stock_flow_core_bear_min: float = 0.60

    industry_green_bear_pct_min: float = 0.50
    industry_bull_green_bullish_min: int = 3
    industry_bull_yellow_bullish_min: int = 1
    industry_early_green_bullish_min: int = 2
    industry_red_bullish_late_cycle_min: int = 1
    warning_bear_override: bool = True
    gdt_semi_earlybull_trigger: bool = True

    bear_override_mode: str = "strict"
    flow_only_strength_as_early_bull: bool = True
    structural_only_strength_as_early_bull: bool = True

    trend_w_bull_bull: float = 0.60
    trend_w_bull_early: float = 0.55
    trend_w_bull_neutral: float = 0.50
    trend_w_neutral_bull: float = 0.45
    trend_w_neutral_early: float = 0.45
    trend_w_neutral_neutral: float = 0.40
    trend_w_bull_bear: float = 0.40
    trend_w_bear_any: float = 0.30


def build_candidate_params(preset: str) -> list[RegimeParams]:
    current = RegimeParams(
        config_id="current",
        notes="Current live thresholds and trend-weight matrix",
    )

    compact = [
        current,
        replace(
            current,
            config_id="relaxed_higher_highs",
            notes="Lower higher-high gates that appear too strict in current history",
            structural_bull_hh50_min=0.20,
            early_hh20_min=0.20,
        ),
        replace(
            current,
            config_id="balanced_relaxed",
            notes="Moderately relax breadth and momentum gates while keeping bear filter intact",
            structural_bull_above20_min=0.60,
            structural_bull_above50_min=0.50,
            structural_bull_hh50_min=0.20,
            early_above20_min=0.50,
            early_hh20_min=0.20,
        ),
        replace(
            current,
            config_id="fast_3d_balanced",
            notes="Faster three-day confirmation window",
            rolling_window=3,
            structural_bull_above20_min=0.60,
            structural_bull_above50_min=0.50,
            structural_bull_hh50_min=0.20,
            early_above20_min=0.50,
            early_hh20_min=0.20,
        ),
        replace(
            current,
            config_id="slow_10d_balanced",
            notes="Slower ten-day confirmation window",
            rolling_window=10,
            structural_bull_above20_min=0.60,
            structural_bull_above50_min=0.50,
            structural_bull_hh50_min=0.20,
            early_above20_min=0.50,
            early_hh20_min=0.20,
        ),
        replace(
            current,
            config_id="bear_sensitive",
            notes="Earlier defensive trigger for weak structure",
            structural_bull_hh50_min=0.20,
            early_hh20_min=0.20,
            bear_above20_max=0.45,
            bear_above50_max=0.45,
            bear_new_low20_min=0.20,
            stock_flow_core_bear_min=0.50,
        ),
        replace(
            current,
            config_id="bear_conservative",
            notes="Requires clearer weakness before Bear labels",
            structural_bull_hh50_min=0.20,
            early_hh20_min=0.20,
            bear_above20_max=0.35,
            bear_above50_max=0.30,
            bear_new_low20_min=0.40,
            stock_flow_core_bear_min=0.70,
        ),
        replace(
            current,
            config_id="flow_sensitive",
            notes="Lets leader strength classify earlier",
            structural_bull_hh50_min=0.20,
            early_hh20_min=0.20,
            stock_flow_bull_pts_min=0.60,
            stock_flow_core_bull_min=0.30,
            stock_flow_confirmer_bull_min=0.30,
        ),
        replace(
            current,
            config_id="trend_aggressive",
            notes="Higher trend weight in bullish regimes, lower in Bear",
            structural_bull_hh50_min=0.20,
            early_hh20_min=0.20,
            trend_w_bull_bull=0.70,
            trend_w_bull_early=0.65,
            trend_w_bull_neutral=0.55,
            trend_w_neutral_bull=0.55,
            trend_w_neutral_early=0.50,
            trend_w_neutral_neutral=0.40,
            trend_w_bull_bear=0.30,
            trend_w_bear_any=0.20,
        ),
        replace(
            current,
            config_id="trend_conservative",
            notes="Smaller regime effect on trend rankings",
            structural_bull_hh50_min=0.20,
            early_hh20_min=0.20,
            trend_w_bull_bull=0.55,
            trend_w_bull_early=0.50,
            trend_w_bull_neutral=0.45,
            trend_w_neutral_bull=0.45,
            trend_w_neutral_early=0.40,
            trend_w_neutral_neutral=0.35,
            trend_w_bull_bear=0.35,
            trend_w_bear_any=0.30,
        ),
    ]

    if preset == "compact":
        return compact

    expanded = []
    windows = [3, 5, 10]
    bull_hh50 = [0.10, 0.20, 0.30]
    bull_above20 = [0.55, 0.60, 0.65]
    early_hh20 = [0.10, 0.20, 0.30]
    bear_new_low = [0.20, 0.30, 0.40]
    flow_core = [0.30, 0.40, 0.50]

    for window in windows:
        for hh50 in bull_hh50:
            for above20 in bull_above20:
                for early_hh in early_hh20:
                    for bear_low in bear_new_low:
                        for core_bull in flow_core:
                            config_id = (
                                f"grid_w{window}_a{int(above20*100)}_"
                                f"hh{int(hh50*100)}_ehh{int(early_hh*100)}_"
                                f"bear{int(bear_low*100)}_flow{int(core_bull*100)}"
                            )
                            expanded.append(
                                replace(
                                    current,
                                    config_id=config_id,
                                    notes="Expanded grid search candidate",
                                    rolling_window=window,
                                    structural_bull_above20_min=above20,
                                    structural_bull_above50_min=max(0.45, above20 - 0.10),
                                    structural_bull_hh50_min=hh50,
                                    early_above20_min=max(0.45, above20 - 0.10),
                                    early_hh20_min=early_hh,
                                    bear_new_low20_min=bear_low,
                                    stock_flow_core_bull_min=core_bull,
                                    stock_flow_confirmer_bull_min=max(0.25, core_bull - 0.05),
                                )
                            )

    return compact + expanded


def build_ticker_to_subindustry() -> dict[str, str]:
    mapping = {}
    for subindustry, group in comp.REGIME_GROUPS.items():
        subindustry = comp.canonical_subindustry_name(subindustry)
        for ticker in group.get("core", []):
            mapping[str(ticker).upper()] = subindustry
        for ticker in group.get("confirmers", []):
            mapping[str(ticker).upper()] = subindustry
    return mapping


def build_membership_table() -> pd.DataFrame:
    rows = []
    for subindustry, group in comp.REGIME_GROUPS.items():
        subindustry = comp.canonical_subindustry_name(subindustry)
        for role in ("core", "confirmers"):
            for ticker in group.get(role, []):
                rows.append({
                    "Ticker": str(ticker).upper(),
                    "SubIndustry": subindustry,
                    "Role": "core" if role == "core" else "confirmer",
                })
    return pd.DataFrame(rows).drop_duplicates()


def normalize_price_data(price_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for ticker, df in price_data.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        clean = df[["close"]].copy()
        clean.index = pd.to_datetime(clean.index).normalize()
        clean = clean.sort_index()
        clean = clean[~clean.index.duplicated(keep="last")]
        out[str(ticker).upper()] = clean
    return out


def build_price_wide(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({
        ticker: df["close"]
        for ticker, df in price_data.items()
        if df is not None and not df.empty and "close" in df.columns
    }).sort_index()


def build_stock_feature_history(
    price_data: dict[str, pd.DataFrame],
    ticker_to_subindustry: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for ticker, df in price_data.items():
        if ticker not in ticker_to_subindustry or df.empty:
            continue

        close = df["close"].astype(float).sort_index()
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        rolling_high20 = close.rolling(20).max()
        rolling_high50 = close.rolling(50).max()
        rolling_low20 = close.rolling(20).min()
        rolling_low50 = close.rolling(50).min()

        pct_from_sma20 = (close - sma20) / sma20
        pct_from_sma50 = (close - sma50) / sma50
        nd20 = (close / rolling_high20).clip(0.0, 1.0)
        nd50 = (close / rolling_high50).clip(0.0, 1.0)
        ma_stack = ((close > sma20) & (sma20 > sma50)).astype(float)
        hold50 = (close > sma50).rolling(50).mean()
        new_low20 = close.eq(rolling_low20)
        new_low50 = close.eq(rolling_low50)
        higher_high20 = close.gt(close.shift(1).rolling(20).max())
        higher_high50 = close.gt(close.shift(1).rolling(50).max())
        low_penalty = new_low20.rolling(20).mean()
        return_50d = close / close.shift(50) - 1.0

        pts = (
            0.25 * nd20.fillna(0.0)
            + 0.30 * nd50.fillna(0.0)
            + 0.20 * ma_stack
            + 0.25 * hold50
            - 0.20 * low_penalty
        ).clip(0.0, 1.0)

        ticker_df = pd.DataFrame({
            "Date": close.index,
            "Ticker": ticker,
            "SubIndustry": ticker_to_subindustry[ticker],
            "Close": close,
            "PTS": pts,
            "ND20": nd20,
            "ND50": nd50,
            "MA_Stack": ma_stack,
            "Hold50": hold50,
            "LowPenalty": low_penalty,
            "Return_50D": return_50d,
            "Pct_From_SMA_20": pct_from_sma20,
            "Pct_From_SMA_50": pct_from_sma50,
            "New_Low_20D": new_low20.astype(int),
            "New_Low_50D": new_low50.astype(int),
            "Higher_High_20D": higher_high20.astype(int),
            "Higher_High_50D": higher_high50.astype(int),
        })
        rows.append(ticker_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, sort=False)


def build_member_feature_history(
    stock_features: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    base = stock_features.drop(columns=["SubIndustry"], errors="ignore")
    members = membership.copy()
    merged = members.merge(base, on="Ticker", how="inner")
    if merged.empty:
        return merged

    if "Membership_Start_Date" in merged.columns:
        starts = pd.to_datetime(merged["Membership_Start_Date"], errors="coerce").dt.normalize()
        dates = pd.to_datetime(merged["Date"], errors="coerce").dt.normalize()
        merged = merged[starts.isna() | (dates >= starts)].copy()

    if "Membership_End_Date" in merged.columns:
        ends = pd.to_datetime(merged["Membership_End_Date"], errors="coerce").dt.normalize()
        dates = pd.to_datetime(merged["Date"], errors="coerce").dt.normalize()
        merged = merged[ends.isna() | (dates <= ends)].copy()

    return merged


def build_subindustry_snapshots(
    member_features: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    df = member_features.copy()
    df["Above_SMA20"] = df["Pct_From_SMA_20"] > 0
    df["Above_SMA50"] = df["Pct_From_SMA_50"] > 0
    df["Higher_High_20"] = df["Higher_High_20D"].astype(bool)
    df["Higher_High_50"] = df["Higher_High_50D"].astype(bool)

    grouped = df.groupby(["Date", "SubIndustry"], dropna=False)
    snapshot = grouped.agg(
        Median_Pct_From_SMA_20=("Pct_From_SMA_20", "median"),
        Median_Pct_From_SMA_50=("Pct_From_SMA_50", "median"),
        New_Low_Ratio_20D=("New_Low_20D", "mean"),
        New_Low_Ratio_50D=("New_Low_50D", "mean"),
        Pct_Higher_Highs_20D=("Higher_High_20", "mean"),
        Pct_Higher_Highs_50D=("Higher_High_50", "mean"),
        Pct_Above_SMA_20=("Above_SMA20", "mean"),
        Pct_Above_SMA_50=("Above_SMA50", "mean"),
        Stock_Count=("Ticker", "nunique"),
    ).reset_index()

    snapshot = snapshot.sort_values(["SubIndustry", "Date"])
    rolling_cols = [
        "Pct_Above_SMA_20",
        "Pct_Above_SMA_50",
        "New_Low_Ratio_20D",
        "Pct_Higher_Highs_20D",
        "Pct_Higher_Highs_50D",
    ]
    for col in rolling_cols:
        snapshot[f"{col}_5D"] = (
            snapshot.groupby("SubIndustry")[col]
            .rolling(window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    snapshot["Slope_Median_Pct_From_SMA_20"] = (
        snapshot.groupby("SubIndustry")["Median_Pct_From_SMA_20"].diff(window)
    )
    return snapshot


def classify_structural(row: pd.Series, params: RegimeParams) -> str:
    required = [
        "Median_Pct_From_SMA_20",
        "Median_Pct_From_SMA_50",
        "Pct_Above_SMA_20_5D",
        "Pct_Above_SMA_50_5D",
        "New_Low_Ratio_20D_5D",
        "Pct_Higher_Highs_20D_5D",
        "Pct_Higher_Highs_50D_5D",
        "Slope_Median_Pct_From_SMA_20",
    ]
    if row[required].isna().any():
        return "Neutral"

    if (
        row["Pct_Above_SMA_20_5D"] >= params.structural_bull_above20_min
        and row["Pct_Above_SMA_50_5D"] >= params.structural_bull_above50_min
        and row["Pct_Higher_Highs_50D_5D"] >= params.structural_bull_hh50_min
        and row["Slope_Median_Pct_From_SMA_20"] >= params.structural_bull_slope_min
        and row["New_Low_Ratio_20D_5D"] <= params.structural_bull_new_low20_max
    ):
        return "Bull"

    if (
        row["Median_Pct_From_SMA_20"] > params.early_median20_min
        and row["Median_Pct_From_SMA_50"] > params.early_median50_min
        and row["Pct_Above_SMA_20_5D"] >= params.early_above20_min
        and row["Pct_Higher_Highs_20D_5D"] >= params.early_hh20_min
        and row["New_Low_Ratio_20D_5D"] <= params.early_new_low20_max
    ):
        return "EarlyBull"

    if (
        row["Median_Pct_From_SMA_20"] < params.bear_median20_max
        and row["Pct_Above_SMA_20_5D"] < params.bear_above20_max
        and row["Pct_Above_SMA_50_5D"] < params.bear_above50_max
        and row["Slope_Median_Pct_From_SMA_20"] < params.bear_slope_max
        and row["New_Low_Ratio_20D_5D"] > params.bear_new_low20_min
    ):
        return "Bear"

    return "Neutral"


def build_stock_flow_history(
    member_features: pd.DataFrame,
    params: RegimeParams,
) -> pd.DataFrame:
    rows = []
    for (date, subindustry), group in member_features.groupby(["Date", "SubIndustry"]):
        core = group[group["Role"] == "core"]
        confirmers = group[group["Role"] == "confirmer"]

        if core.empty:
            regime = "Neutral"
            core_bull_pct = 0.0
            confirmer_bull_pct = 0.0
            core_bear_pct = 0.0
            confirmer_bear_pct = 0.0
        else:
            core_bull_pct = float((core["PTS"] >= params.stock_flow_bull_pts_min).mean())
            confirmer_bull_pct = (
                float((confirmers["PTS"] >= params.stock_flow_bull_pts_min).mean())
                if not confirmers.empty else 0.0
            )
            core_bear_pct = float((core["PTS"] <= params.stock_flow_bear_pts_max).mean())
            confirmer_bear_pct = (
                float((confirmers["PTS"] <= params.stock_flow_bear_pts_max).mean())
                if not confirmers.empty else 0.0
            )

            if core_bear_pct >= params.stock_flow_core_bear_min:
                regime = "Bear"
            elif (
                core_bull_pct >= params.stock_flow_core_bull_min
                and confirmer_bull_pct >= params.stock_flow_confirmer_bull_min
            ):
                regime = "Bull"
            elif core_bull_pct >= params.stock_flow_core_bull_min:
                regime = "EarlyBull"
            else:
                regime = "Neutral"

        rows.append({
            "Date": date,
            "SubIndustry": subindustry,
            "StockFlow_Regime": regime,
            "Core_Bull_Pct": core_bull_pct,
            "Confirmer_Bull_Pct": confirmer_bull_pct,
            "Core_Bear_Pct": core_bear_pct,
            "Confirmer_Bear_Pct": confirmer_bear_pct,
        })

    return pd.DataFrame(rows)


def combine_regimes(structural: str, flow: str, params: RegimeParams) -> str:
    structural = structural if isinstance(structural, str) and structural else "Neutral"
    flow = flow if isinstance(flow, str) and flow else "Neutral"

    if params.bear_override_mode == "strict" and (structural == "Bear" or flow == "Bear"):
        return "Bear"
    if params.bear_override_mode == "structural_only" and structural == "Bear":
        return "Bear"
    if params.bear_override_mode == "both_required" and structural == "Bear" and flow == "Bear":
        return "Bear"

    if structural == "Bull" and flow == "Bull":
        return "Bull"

    if structural in ("Bull", "EarlyBull") and flow in ("Bull", "EarlyBull"):
        return "EarlyBull"

    if params.flow_only_strength_as_early_bull and structural == "Neutral" and flow in ("Bull", "EarlyBull"):
        return "EarlyBull"

    if params.structural_only_strength_as_early_bull and flow == "Neutral" and structural in ("Bull", "EarlyBull"):
        return "EarlyBull"

    return "Neutral"


def count_states(regimes: list[str]) -> dict[str, int]:
    return {
        "Bull": sum(r == "Bull" for r in regimes),
        "EarlyBull": sum(r == "EarlyBull" for r in regimes),
        "Neutral": sum(r == "Neutral" for r in regimes),
        "Bear": sum(r == "Bear" for r in regimes),
        "Total": len(regimes),
    }


def regime_at(regime_map: dict[str, str], subindustry: str) -> str:
    return regime_map.get(comp.canonical_subindustry_name(subindustry), "Neutral")


def classify_industry_regime(regime_map: dict[str, str], params: RegimeParams) -> tuple[str, dict]:
    gdt = regime_at(regime_map, comp.TECH_CORE)
    semi = regime_at(regime_map, "Semiconductors")
    warning = regime_at(regime_map, comp.TECH_WARNING)

    green = [regime_at(regime_map, s) for s in comp.TECH_GREEN]
    yellow = [regime_at(regime_map, s) for s in comp.TECH_YELLOW]
    red = [regime_at(regime_map, s) for s in comp.TECH_RED]

    green_counts = count_states(green)
    yellow_counts = count_states(yellow)
    red_counts = count_states(red)

    green_total = max(green_counts["Total"], 1)
    green_bullish = green_counts["Bull"] + green_counts["EarlyBull"]
    yellow_bullish = yellow_counts["Bull"] + yellow_counts["EarlyBull"]
    green_bear_pct = green_counts["Bear"] / green_total
    red_bullish = red_counts["Bull"] + red_counts["EarlyBull"]

    if gdt == "Bear":
        regime = "Bear"
    elif green_bear_pct >= params.industry_green_bear_pct_min:
        regime = "Bear"
    elif (
        params.warning_bear_override
        and warning == "Bear"
        and red_bullish >= params.industry_red_bullish_late_cycle_min
    ):
        regime = "Bear"
    elif (
        gdt != "Bear"
        and green_bullish >= params.industry_bull_green_bullish_min
        and yellow_bullish >= params.industry_bull_yellow_bullish_min
        and warning != "Bear"
    ):
        regime = "Bull"
    elif params.gdt_semi_earlybull_trigger and gdt == "Bull" and semi == "Bull":
        regime = "EarlyBull"
    elif (
        gdt != "Bear"
        and green_bullish >= params.industry_early_green_bullish_min
        and warning != "Bear"
    ):
        regime = "EarlyBull"
    else:
        regime = "Neutral"

    diagnostics = {
        "GDT_Regime": gdt,
        "Semis_Regime": semi,
        "Warning_Regime": warning,
        "Green_Bullish": green_bullish,
        "Green_Bear": green_counts["Bear"],
        "Yellow_Bullish": yellow_bullish,
        "Red_Bullish": red_bullish,
    }
    return regime, diagnostics


def build_industry_history(subindustry_history: pd.DataFrame, params: RegimeParams) -> pd.DataFrame:
    rows = []
    for date, group in subindustry_history.groupby("Date"):
        regime_map = dict(zip(group["SubIndustry"], group["SubIndustry_Regime"]))
        regime, diagnostics = classify_industry_regime(regime_map, params)
        rows.append({
            "Date": date,
            "Industry_Regime": regime,
            **diagnostics,
        })
    return pd.DataFrame(rows)


def trend_weight(industry_regime: str, subindustry_regime: str, params: RegimeParams) -> float:
    industry = industry_regime if isinstance(industry_regime, str) else "Neutral"
    subindustry = subindustry_regime if isinstance(subindustry_regime, str) else "Neutral"
    industry = "Bull" if industry == "EarlyBull" else industry

    if industry == "Bull" and subindustry == "Bull":
        return params.trend_w_bull_bull
    if industry == "Bull" and subindustry == "EarlyBull":
        return params.trend_w_bull_early
    if industry == "Bull" and subindustry == "Neutral":
        return params.trend_w_bull_neutral
    if industry == "Neutral" and subindustry == "Bull":
        return params.trend_w_neutral_bull
    if industry == "Neutral" and subindustry == "EarlyBull":
        return params.trend_w_neutral_early
    if industry == "Neutral" and subindustry == "Neutral":
        return params.trend_w_neutral_neutral
    if industry == "Bull" and subindustry == "Bear":
        return params.trend_w_bull_bear
    return params.trend_w_bear_any


def classify_regimes_for_params(
    stock_features: pd.DataFrame,
    member_features: pd.DataFrame,
    params: RegimeParams,
    evaluation_dates: list[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structural = build_subindustry_snapshots(member_features, params.rolling_window)
    structural["Structural_Regime"] = structural.apply(
        lambda row: classify_structural(row, params),
        axis=1,
    )
    structural["Structural_Regime_Persist"] = (
        structural
        .groupby("SubIndustry")["Structural_Regime"]
        .apply(comp.apply_regime_persistence)
        .reset_index(level=0, drop=True)
    )

    if evaluation_dates is not None:
        eval_dates = set(pd.to_datetime(pd.Series(evaluation_dates)).dt.normalize())
        structural = structural[structural["Date"].isin(eval_dates)].copy()
        member_features = member_features[member_features["Date"].isin(eval_dates)].copy()
        stock_features = stock_features[stock_features["Date"].isin(eval_dates)].copy()

    flow = build_stock_flow_history(member_features, params)
    subindustry = structural.merge(
        flow,
        on=["Date", "SubIndustry"],
        how="left",
    )
    subindustry["StockFlow_Regime"] = subindustry["StockFlow_Regime"].fillna("Neutral")
    subindustry["SubIndustry_Regime"] = subindustry.apply(
        lambda row: combine_regimes(
            row["Structural_Regime_Persist"],
            row["StockFlow_Regime"],
            params,
        ),
        axis=1,
    )

    industry = build_industry_history(subindustry, params)

    ranked = stock_features.merge(
        subindustry[
            [
                "Date",
                "SubIndustry",
                "Structural_Regime",
                "Structural_Regime_Persist",
                "StockFlow_Regime",
                "SubIndustry_Regime",
            ]
        ],
        on=["Date", "SubIndustry"],
        how="left",
    )
    ranked = ranked.merge(industry[["Date", "Industry_Regime"]], on="Date", how="left")
    ranked["SubIndustry_Regime"] = ranked["SubIndustry_Regime"].fillna("Neutral")
    ranked["Industry_Regime"] = ranked["Industry_Regime"].fillna("Neutral")
    ranked["Trend_Weight"] = ranked.apply(
        lambda row: trend_weight(row["Industry_Regime"], row["SubIndustry_Regime"], params),
        axis=1,
    )
    ranked["Regime_Adjusted_Trend_Score"] = ranked["PTS"] * 100.0 * ranked["Trend_Weight"]
    return ranked, subindustry


def choose_anchor_dates(price_wide: pd.DataFrame, years: int, frequency: str) -> list[pd.Timestamp]:
    if price_wide.empty:
        return []

    dates = pd.Series(pd.to_datetime(price_wide.index).normalize()).drop_duplicates().sort_values()
    end_date = dates.iloc[-1]
    start_date = end_date - pd.DateOffset(years=years)
    dates = dates[dates >= start_date]

    if frequency == "weekly":
        return dates.groupby(dates.dt.to_period("W-FRI")).max().tolist()
    if frequency == "quarterly":
        return dates.groupby(dates.dt.to_period("Q")).max().tolist()
    return dates.groupby(dates.dt.to_period("M")).max().tolist()


def fold_label(anchor: pd.Timestamp, end_date: pd.Timestamp) -> str:
    if anchor >= end_date - pd.DateOffset(years=1):
        return "1-current"
    if anchor >= end_date - pd.DateOffset(years=2):
        return "2-1"
    if anchor >= end_date - pd.DateOffset(years=3):
        return "3-2"
    if anchor >= end_date - pd.DateOffset(years=4):
        return "4-3"
    return "5-4"


def future_return_series(
    price_wide: pd.DataFrame,
    anchor: pd.Timestamp,
    horizon_days: int,
) -> tuple[pd.Timestamp | None, pd.Series]:
    if anchor not in price_wide.index:
        return None, pd.Series(dtype=float)
    pos = price_wide.index.get_loc(anchor)
    if isinstance(pos, slice) or isinstance(pos, np.ndarray):
        return None, pd.Series(dtype=float)
    future_pos = int(pos) + horizon_days
    if future_pos >= len(price_wide.index):
        return None, pd.Series(dtype=float)
    future_date = price_wide.index[future_pos]
    returns = price_wide.loc[future_date] / price_wide.loc[anchor] - 1.0
    return future_date, returns.replace([np.inf, -np.inf], np.nan)


def benchmark_value(returns: pd.Series, ticker: str) -> float:
    if ticker not in returns.index:
        return np.nan
    value = returns.loc[ticker]
    return float(value) if pd.notna(value) and np.isfinite(value) else np.nan


def evaluate_config(
    params: RegimeParams,
    ranked: pd.DataFrame,
    subindustry_history: pd.DataFrame,
    price_wide: pd.DataFrame,
    anchor_dates: list[pd.Timestamp],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_rows = []
    regime_rows = []
    end_date = price_wide.index.max()

    for anchor in anchor_dates:
        day = ranked[ranked["Date"] == anchor].copy()
        day = day[np.isfinite(day["Regime_Adjusted_Trend_Score"])]
        if day.empty:
            continue

        day = day.sort_values("Regime_Adjusted_Trend_Score", ascending=False)
        top = day.head(top_n)
        universe = day.copy()

        sub_day = subindustry_history[subindustry_history["Date"] == anchor].copy()

        for horizon_name, horizon_days in HORIZONS.items():
            future_date, returns = future_return_series(price_wide, anchor, horizon_days)
            if future_date is None or returns.empty:
                continue

            top_returns = returns.reindex(top["Ticker"]).dropna()
            universe_returns = returns.reindex(universe["Ticker"]).dropna()
            if top_returns.empty or universe_returns.empty:
                continue

            qqq_return = benchmark_value(returns, "QQQ")
            xlk_return = benchmark_value(returns, "XLK")
            anchor_rows.append({
                "Config_ID": params.config_id,
                "Fold": fold_label(anchor, end_date),
                "Anchor_Date": anchor.strftime("%Y-%m-%d"),
                "Future_Date": future_date.strftime("%Y-%m-%d"),
                "Horizon": horizon_name,
                "Top_N": top_n,
                "Top_Avg_Return": float(top_returns.mean()),
                "Top_Median_Return": float(top_returns.median()),
                "Universe_Avg_Return": float(universe_returns.mean()),
                "Universe_Median_Return": float(universe_returns.median()),
                "QQQ_Return": qqq_return,
                "XLK_Return": xlk_return,
                "Top_Excess_vs_Universe": float(top_returns.mean() - universe_returns.mean()),
                "Top_Excess_vs_QQQ": float(top_returns.mean() - qqq_return) if np.isfinite(qqq_return) else np.nan,
                "Top_Excess_vs_XLK": float(top_returns.mean() - xlk_return) if np.isfinite(xlk_return) else np.nan,
                "Hit_Rate_vs_Universe_Avg": float((top_returns > universe_returns.mean()).mean()),
                "Top_Count_With_Return": int(top_returns.count()),
                "Universe_Count_With_Return": int(universe_returns.count()),
                "Bullish_Subindustry_Count": int((sub_day["SubIndustry_Regime"].isin(["Bull", "EarlyBull"])).sum()),
                "Bear_Subindustry_Count": int((sub_day["SubIndustry_Regime"] == "Bear").sum()),
                "Neutral_Subindustry_Count": int((sub_day["SubIndustry_Regime"] == "Neutral").sum()),
            })

            for _, sub_row in sub_day.iterrows():
                tickers = day.loc[day["SubIndustry"] == sub_row["SubIndustry"], "Ticker"]
                sub_returns = returns.reindex(tickers).dropna()
                if sub_returns.empty:
                    continue
                regime_rows.append({
                    "Config_ID": params.config_id,
                    "Fold": fold_label(anchor, end_date),
                    "Anchor_Date": anchor.strftime("%Y-%m-%d"),
                    "Future_Date": future_date.strftime("%Y-%m-%d"),
                    "Horizon": horizon_name,
                    "SubIndustry": sub_row["SubIndustry"],
                    "SubIndustry_Regime": sub_row["SubIndustry_Regime"],
                    "Structural_Regime": sub_row["Structural_Regime"],
                    "StockFlow_Regime": sub_row["StockFlow_Regime"],
                    "SubIndustry_Avg_Return": float(sub_returns.mean()),
                    "SubIndustry_Median_Return": float(sub_returns.median()),
                    "SubIndustry_Count_With_Return": int(sub_returns.count()),
                })

    return pd.DataFrame(anchor_rows), pd.DataFrame(regime_rows)


def summarize_anchor_results(anchor_results: pd.DataFrame) -> pd.DataFrame:
    if anchor_results.empty:
        return pd.DataFrame()

    summary = (
        anchor_results
        .groupby(["Config_ID", "Horizon"], dropna=False)
        .agg(
            Observations=("Top_Avg_Return", "count"),
            Top_Avg_Return=("Top_Avg_Return", "mean"),
            Universe_Avg_Return=("Universe_Avg_Return", "mean"),
            QQQ_Return=("QQQ_Return", "mean"),
            XLK_Return=("XLK_Return", "mean"),
            Top_Excess_vs_Universe=("Top_Excess_vs_Universe", "mean"),
            Top_Excess_vs_QQQ=("Top_Excess_vs_QQQ", "mean"),
            Top_Excess_vs_XLK=("Top_Excess_vs_XLK", "mean"),
            Hit_Rate_vs_Universe_Avg=("Hit_Rate_vs_Universe_Avg", "mean"),
            Avg_Bullish_Subindustry_Count=("Bullish_Subindustry_Count", "mean"),
            Avg_Bear_Subindustry_Count=("Bear_Subindustry_Count", "mean"),
            Avg_Neutral_Subindustry_Count=("Neutral_Subindustry_Count", "mean"),
        )
        .reset_index()
    )
    for col in [
        "Top_Avg_Return",
        "Universe_Avg_Return",
        "QQQ_Return",
        "XLK_Return",
        "Top_Excess_vs_Universe",
        "Top_Excess_vs_QQQ",
        "Top_Excess_vs_XLK",
        "Hit_Rate_vs_Universe_Avg",
    ]:
        summary[col] = summary[col].round(6)
    return summary


def summarize_regime_timing(regime_results: pd.DataFrame) -> pd.DataFrame:
    if regime_results.empty:
        return pd.DataFrame()
    out = (
        regime_results
        .groupby(["Config_ID", "Horizon", "SubIndustry_Regime"], dropna=False)
        .agg(
            Observations=("SubIndustry_Avg_Return", "count"),
            Avg_Return=("SubIndustry_Avg_Return", "mean"),
            Median_Return=("SubIndustry_Median_Return", "median"),
            Avg_Subindustry_Count=("SubIndustry_Count_With_Return", "mean"),
        )
        .reset_index()
    )
    out["Avg_Return"] = out["Avg_Return"].round(6)
    out["Median_Return"] = out["Median_Return"].round(6)
    return out


def score_configs(anchor_summary: pd.DataFrame, regime_timing: pd.DataFrame) -> pd.DataFrame:
    if anchor_summary.empty:
        return pd.DataFrame()

    base = (
        anchor_summary
        .groupby("Config_ID", dropna=False)
        .agg(
            Observations=("Observations", "sum"),
            Avg_Top_Return=("Top_Avg_Return", "mean"),
            Avg_Excess_vs_Universe=("Top_Excess_vs_Universe", "mean"),
            Avg_Excess_vs_QQQ=("Top_Excess_vs_QQQ", "mean"),
            Avg_Excess_vs_XLK=("Top_Excess_vs_XLK", "mean"),
            Avg_Hit_Rate_vs_Universe=("Hit_Rate_vs_Universe_Avg", "mean"),
            Avg_Bullish_Subindustry_Count=("Avg_Bullish_Subindustry_Count", "mean"),
            Avg_Bear_Subindustry_Count=("Avg_Bear_Subindustry_Count", "mean"),
            Avg_Neutral_Subindustry_Count=("Avg_Neutral_Subindustry_Count", "mean"),
        )
        .reset_index()
    )

    regime_spreads = []
    if not regime_timing.empty:
        pivot = regime_timing.pivot_table(
            index=["Config_ID", "Horizon"],
            columns="SubIndustry_Regime",
            values="Avg_Return",
            aggfunc="mean",
        )
        for (config_id, horizon), row in pivot.iterrows():
            bullish = np.nanmean([row.get("Bull", np.nan), row.get("EarlyBull", np.nan)])
            bearish = row.get("Bear", np.nan)
            neutral = row.get("Neutral", np.nan)
            regime_spreads.append({
                "Config_ID": config_id,
                "Horizon": horizon,
                "Bullish_minus_Bear": bullish - bearish if np.isfinite(bullish) and np.isfinite(bearish) else np.nan,
                "Bullish_minus_Neutral": bullish - neutral if np.isfinite(bullish) and np.isfinite(neutral) else np.nan,
            })

    spread_df = pd.DataFrame(regime_spreads)
    if not spread_df.empty:
        spread_summary = (
            spread_df
            .groupby("Config_ID", dropna=False)
            .agg(
                Avg_Bullish_minus_Bear=("Bullish_minus_Bear", "mean"),
                Avg_Bullish_minus_Neutral=("Bullish_minus_Neutral", "mean"),
            )
            .reset_index()
        )
        base = base.merge(spread_summary, on="Config_ID", how="left")
    else:
        base["Avg_Bullish_minus_Bear"] = np.nan
        base["Avg_Bullish_minus_Neutral"] = np.nan

    base["Objective_Score"] = (
        base["Avg_Excess_vs_Universe"].fillna(0.0)
        + 0.5 * base["Avg_Excess_vs_XLK"].fillna(0.0)
        + 0.25 * base["Avg_Bullish_minus_Bear"].fillna(0.0)
    )
    return base.sort_values("Objective_Score", ascending=False).reset_index(drop=True)


def save_outputs(
    output_dir: Path,
    params_list: list[RegimeParams],
    anchor_results: pd.DataFrame,
    anchor_summary: pd.DataFrame,
    regime_results: pd.DataFrame,
    regime_timing: pd.DataFrame,
    config_scores: pd.DataFrame,
    run_metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(p) for p in params_list]).to_csv(
        output_dir / "tested_parameter_configs.csv",
        index=False,
    )
    anchor_results.to_csv(output_dir / "anchor_results.csv", index=False)
    anchor_summary.to_csv(output_dir / "anchor_summary.csv", index=False)
    regime_results.to_csv(output_dir / "regime_timing_observations.csv", index=False)
    regime_timing.to_csv(output_dir / "regime_timing_summary.csv", index=False)
    config_scores.to_csv(output_dir / "config_scores.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest regime-threshold settings.")
    parser.add_argument("--preset", choices=["compact", "expanded"], default="compact")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--anchor-frequency", choices=["monthly", "weekly", "quarterly"], default="monthly")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--skip-configs", type=int, default=0)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    args = parser.parse_args()

    full_params_list = build_candidate_params(args.preset)
    params_list = full_params_list[args.skip_configs:]
    if args.max_configs is not None:
        params_list = params_list[:args.max_configs]

    if not params_list:
        raise RuntimeError("No parameter configs selected for testing")

    ticker_to_subindustry = build_ticker_to_subindustry()
    membership = build_membership_table()
    tickers = sorted(set(ticker_to_subindustry) | set(BENCHMARK_TICKERS))

    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start_date = end_date - pd.DateOffset(years=args.years + 1)

    print(f"[INFO] Fetching {len(tickers)} tickers from {start_date.date()} to {end_date.date()}", flush=True)
    raw_price_data = comp.fetch_all_prices(
        tickers=tickers,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        batch_size=args.batch_size,
    )
    price_data = normalize_price_data(raw_price_data)
    price_wide = build_price_wide(price_data)
    if price_wide.empty:
        raise RuntimeError("No price data returned; cannot run backtest")

    stock_price_data = {ticker: df for ticker, df in price_data.items() if ticker in ticker_to_subindustry}
    stock_features = build_stock_feature_history(stock_price_data, ticker_to_subindustry)
    if stock_features.empty:
        raise RuntimeError("No stock features were generated")

    member_features = build_member_feature_history(stock_features, membership)
    anchor_dates = choose_anchor_dates(price_wide, args.years, args.anchor_frequency)
    print(f"[INFO] Built {len(stock_features):,} stock-day feature rows", flush=True)
    print(f"[INFO] Anchor dates: {len(anchor_dates)}", flush=True)
    print(
        f"[INFO] Testing {len(params_list)} selected parameter configs "
        f"from {len(full_params_list)} total candidates",
        flush=True,
    )

    all_anchor_results = []
    all_regime_results = []

    metadata_base = {
        "preset": args.preset,
        "years": args.years,
        "anchor_frequency": args.anchor_frequency,
        "top_n": args.top_n,
        "price_start_date": start_date.strftime("%Y-%m-%d"),
        "price_end_date": end_date.strftime("%Y-%m-%d"),
        "latest_price_date": price_wide.index.max().strftime("%Y-%m-%d"),
        "ticker_count_requested": len(tickers),
        "ticker_count_with_price_data": len(price_data),
        "stock_feature_rows": len(stock_features),
        "anchor_count": len(anchor_dates),
        "candidate_config_count": len(full_params_list),
        "selected_config_count": len(params_list),
        "skip_configs": args.skip_configs,
        "max_configs": args.max_configs,
        "mode_note": (
            "Price/regime calibration only. Full point-in-time fair-value "
            "backtesting requires SEC fact caching by filed date."
        ),
    }

    for idx, params in enumerate(params_list, start=1):
        print(f"[INFO] Testing config {idx}/{len(params_list)}: {params.config_id}", flush=True)
        ranked, subindustry_history = classify_regimes_for_params(
            stock_features=stock_features,
            member_features=member_features,
            params=params,
            evaluation_dates=anchor_dates,
        )
        anchor_results, regime_results = evaluate_config(
            params=params,
            ranked=ranked,
            subindustry_history=subindustry_history,
            price_wide=price_wide,
            anchor_dates=anchor_dates,
            top_n=args.top_n,
        )
        all_anchor_results.append(anchor_results)
        all_regime_results.append(regime_results)

        if args.checkpoint_every and idx % args.checkpoint_every == 0:
            checkpoint_anchor_results = pd.concat(all_anchor_results, ignore_index=True, sort=False)
            checkpoint_regime_results = pd.concat(all_regime_results, ignore_index=True, sort=False)
            checkpoint_anchor_summary = summarize_anchor_results(checkpoint_anchor_results)
            checkpoint_regime_timing = summarize_regime_timing(checkpoint_regime_results)
            checkpoint_config_scores = score_configs(checkpoint_anchor_summary, checkpoint_regime_timing)
            save_outputs(
                output_dir=args.output_dir,
                params_list=params_list[:idx],
                anchor_results=checkpoint_anchor_results,
                anchor_summary=checkpoint_anchor_summary,
                regime_results=checkpoint_regime_results,
                regime_timing=checkpoint_regime_timing,
                config_scores=checkpoint_config_scores,
                run_metadata={
                    **metadata_base,
                    "status": "partial",
                    "completed_config_count": idx,
                },
            )
            print(f"[INFO] Checkpoint saved after {idx} configs", flush=True)

    anchor_results = pd.concat(all_anchor_results, ignore_index=True, sort=False)
    regime_results = pd.concat(all_regime_results, ignore_index=True, sort=False)
    anchor_summary = summarize_anchor_results(anchor_results)
    regime_timing = summarize_regime_timing(regime_results)
    config_scores = score_configs(anchor_summary, regime_timing)

    metadata = {
        **metadata_base,
        "status": "complete",
        "completed_config_count": len(params_list),
    }
    save_outputs(
        output_dir=args.output_dir,
        params_list=params_list,
        anchor_results=anchor_results,
        anchor_summary=anchor_summary,
        regime_results=regime_results,
        regime_timing=regime_timing,
        config_scores=config_scores,
        run_metadata=metadata,
    )

    print("\n=== CONFIG SCORE LEADERBOARD ===")
    cols = [
        "Config_ID",
        "Objective_Score",
        "Avg_Excess_vs_Universe",
        "Avg_Excess_vs_XLK",
        "Avg_Hit_Rate_vs_Universe",
        "Avg_Bullish_Subindustry_Count",
        "Avg_Bear_Subindustry_Count",
        "Avg_Neutral_Subindustry_Count",
    ]
    if not config_scores.empty:
        print(config_scores[cols].head(10).to_string(index=False))
    print(f"\n[SUCCESS] Saved outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
