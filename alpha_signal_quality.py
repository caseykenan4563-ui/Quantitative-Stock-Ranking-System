#!/usr/bin/env python3
"""Analyze whether Stock Analysis ranking signals have forward alpha.

This is a research/validation harness. It does not change production scoring.
It asks whether higher point-in-time scores predicted stronger future returns,
using canonical historical prices and only score rows that existed at each
anchor date.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import historical_data_layer as historical_data
import regime_threshold_backtest as regime_bt


DEFAULT_SCORE_HISTORY = Path("backtests/combined_score_sharadar/point_in_time_scores.csv")
DEFAULT_PRICE_FILE = Path("data/historical_data_layer/prices_daily.csv")
DEFAULT_OUTPUT_DIR = Path("backtests/alpha_signal_quality_sharadar")

HORIZONS = regime_bt.HORIZONS
BENCHMARKS = ["QQQ", "XLK"]
TOP_N_VALUES = [10, 20, 50, 100]
RELATIVE_MOMENTUM_WINDOWS = [21, 63, 126]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate predictive quality of Stock Analysis ranking signals."
    )
    parser.add_argument("--score-history", type=Path, default=DEFAULT_SCORE_HISTORY)
    parser.add_argument("--historical-price-file", type=Path, default=DEFAULT_PRICE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-observations", type=int, default=25)
    return parser.parse_args()


def pct_rank_high(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() <= 1:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return clean.rank(pct=True, method="average") * 100.0


def finite_mean(values: pd.Series | list[float]) -> float:
    clean = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def finite_std(values: pd.Series | list[float]) -> float:
    clean = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.std(ddof=0)) if len(clean) >= 2 else np.nan


def nearest_index_at_or_before(index: pd.DatetimeIndex, date: pd.Timestamp) -> int | None:
    loc = index.searchsorted(pd.to_datetime(date).normalize(), side="right") - 1
    if loc < 0:
        return None
    return int(loc)


def trailing_return_row(
    price_wide: pd.DataFrame,
    anchor: pd.Timestamp,
    lookback_days: int,
) -> pd.Series:
    index = pd.to_datetime(price_wide.index).normalize()
    current_loc = nearest_index_at_or_before(index, anchor)
    if current_loc is None or current_loc < lookback_days:
        return pd.Series(dtype="float64")
    start_loc = current_loc - lookback_days
    current = price_wide.iloc[current_loc].astype("float64")
    start = price_wide.iloc[start_loc].astype("float64")
    return (current / start - 1.0).replace([np.inf, -np.inf], np.nan)


def trailing_volatility_row(
    price_wide: pd.DataFrame,
    anchor: pd.Timestamp,
    lookback_days: int,
) -> pd.Series:
    index = pd.to_datetime(price_wide.index).normalize()
    current_loc = nearest_index_at_or_before(index, anchor)
    if current_loc is None or current_loc < lookback_days:
        return pd.Series(dtype="float64")
    start_loc = current_loc - lookback_days
    window = price_wide.iloc[start_loc : current_loc + 1].astype("float64")
    returns = window.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return returns.std(axis=0, ddof=0) * np.sqrt(252.0)


def trailing_downside_volatility_row(
    price_wide: pd.DataFrame,
    anchor: pd.Timestamp,
    lookback_days: int,
) -> pd.Series:
    index = pd.to_datetime(price_wide.index).normalize()
    current_loc = nearest_index_at_or_before(index, anchor)
    if current_loc is None or current_loc < lookback_days:
        return pd.Series(dtype="float64")
    start_loc = current_loc - lookback_days
    window = price_wide.iloc[start_loc : current_loc + 1].astype("float64")
    returns = window.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    downside = returns.mask(returns > 0.0, 0.0)
    return downside.std(axis=0, ddof=0) * np.sqrt(252.0)


def trailing_max_drawdown_row(
    price_wide: pd.DataFrame,
    anchor: pd.Timestamp,
    lookback_days: int,
) -> pd.Series:
    index = pd.to_datetime(price_wide.index).normalize()
    current_loc = nearest_index_at_or_before(index, anchor)
    if current_loc is None or current_loc < lookback_days:
        return pd.Series(dtype="float64")
    start_loc = current_loc - lookback_days
    window = price_wide.iloc[start_loc : current_loc + 1].astype("float64")
    running_peak = window.cummax()
    drawdowns = (window / running_peak - 1.0).replace([np.inf, -np.inf], np.nan)
    return drawdowns.min(axis=0)


def weighted_average_columns(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    present = {
        column: float(weight)
        for column, weight in weights.items()
        if column in frame.columns and float(weight) > 0
    }
    if not present:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    values = pd.DataFrame({
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in present
    })
    weight_series = pd.Series(present, dtype="float64")
    weighted_values = values.mul(weight_series, axis=1)
    valid_weights = values.notna().mul(weight_series, axis=1)
    denominator = valid_weights.sum(axis=1).replace(0.0, np.nan)
    return weighted_values.sum(axis=1, skipna=True) / denominator


def future_return_row(
    price_wide: pd.DataFrame,
    anchor: pd.Timestamp,
    horizon_days: int,
) -> tuple[pd.Timestamp | pd.NaT, pd.Series]:
    index = pd.to_datetime(price_wide.index).normalize()
    start_loc = nearest_index_at_or_before(index, anchor)
    if start_loc is None:
        return pd.NaT, pd.Series(dtype="float64")
    future_loc = start_loc + horizon_days
    if future_loc >= len(index):
        return pd.NaT, pd.Series(dtype="float64")
    start = price_wide.iloc[start_loc].astype("float64")
    future = price_wide.iloc[future_loc].astype("float64")
    return index[future_loc], (future / start - 1.0).replace([np.inf, -np.inf], np.nan)


def clean_signal_name(name: str) -> str:
    return (
        name.replace("_", " ")
        .replace("Pct", "%")
        .replace("Xlk", "XLK")
        .replace("Qqq", "QQQ")
    )


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Score history file not found: {path}")
    scores = pd.read_csv(path)
    required = {"Date", "Ticker", "SubIndustry"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise RuntimeError(f"Score history missing required columns: {missing}")

    scores["Date"] = pd.to_datetime(scores["Date"], errors="coerce").dt.normalize()
    scores["Ticker"] = scores["Ticker"].astype(str).str.upper().str.strip()
    scores["SubIndustry"] = scores["SubIndustry"].astype(str).str.strip()
    scores = scores.dropna(subset=["Date", "Ticker"]).copy()
    scores = scores[scores["Ticker"] != ""].copy()

    numeric_candidates = [
        "Combined_Score",
        "Fair_Value_Score",
        "Price_Trend_Score",
        "Stock_Price_Trend_Score",
        "SubIndustry_Price_Trend_Score",
        "Industry_Price_Trend_Score",
        "Fair_Value_Metrics_Used",
        "Fair_Value_Usable_Metrics",
    ]
    for col in numeric_candidates:
        if col in scores.columns:
            scores[col] = pd.to_numeric(scores[col], errors="coerce")

    return scores.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def attach_ranked_signals(scores: pd.DataFrame, price_wide: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()

    if "Stock_Price_Trend_Score" not in out.columns and "Price_Trend_Score" in out.columns:
        trend = pd.to_numeric(out["Price_Trend_Score"], errors="coerce")
        out["Stock_Price_Trend_Score"] = np.where(trend <= 1.0, trend * 100.0, trend)

    rank_source_cols = [
        "Combined_Score",
        "Fair_Value_Score",
        "Stock_Price_Trend_Score",
        "SubIndustry_Price_Trend_Score",
        "Industry_Price_Trend_Score",
    ]
    for col in rank_source_cols:
        if col in out.columns:
            out[f"{col}_RankPct"] = out.groupby("Date", group_keys=False)[col].apply(pct_rank_high)

    if {"Date", "SubIndustry", "Combined_Score"}.issubset(out.columns):
        out["SubIndustry_Neutral_Combined_RankPct"] = (
            out.groupby(["Date", "SubIndustry"], group_keys=False)["Combined_Score"].apply(pct_rank_high)
        )
    if {"Date", "SubIndustry", "Fair_Value_Score"}.issubset(out.columns):
        out["SubIndustry_Neutral_Fair_Value_RankPct"] = (
            out.groupby(["Date", "SubIndustry"], group_keys=False)["Fair_Value_Score"].apply(pct_rank_high)
        )

    if "Fair_Value_Metrics_Used" in out.columns and "Combined_Score_RankPct" in out.columns:
        coverage_gap = (5.0 - pd.to_numeric(out["Fair_Value_Metrics_Used"], errors="coerce")).clip(lower=0.0)
        out["Coverage_Adjusted_Combined_RankPct"] = (out["Combined_Score_RankPct"] - coverage_gap * 5.0).clip(0.0, 100.0)

    if "Fair_Value_Score" in out.columns:
        metrics_used = pd.to_numeric(
            out.get("Fair_Value_Metrics_Used", pd.Series(0.0, index=out.index)),
            errors="coerce",
        ).fillna(0.0)
        fair_status = (
            out.get("Fair_Value_Data_Status", pd.Series("", index=out.index))
            .astype(str)
            .str.upper()
            .str.strip()
        )
        share_source = (
            out.get("Share_Count_Source_Category", pd.Series("", index=out.index))
            .astype(str)
            .str.upper()
            .str.strip()
        )
        confidence = (metrics_used.clip(lower=0.0, upper=7.0) / 7.0).where(fair_status.eq("OK"), 0.0)
        share_source_multiplier = pd.Series(0.90, index=out.index, dtype="float64")
        share_source_multiplier = share_source_multiplier.where(
            ~share_source.str.contains("WEIGHTED AVERAGE", na=False),
            0.85,
        )
        share_source_multiplier = share_source_multiplier.where(
            ~share_source.str.contains("MISSING USABLE", na=False),
            0.0,
        )
        share_source_multiplier = share_source_multiplier.where(
            ~share_source.str.contains("PREFERRED", na=False),
            1.0,
        )
        out["Fair_Value_Confidence"] = (confidence * share_source_multiplier).clip(0.0, 1.0)
        fair_value = pd.to_numeric(out["Fair_Value_Score"], errors="coerce").fillna(50.0)
        out["Confidence_Adjusted_Fair_Value_Score"] = (
            50.0 + (fair_value - 50.0) * out["Fair_Value_Confidence"]
        ).clip(0.0, 100.0)
        out["Confidence_Adjusted_Fair_Value_RankPct"] = (
            out.groupby("Date", group_keys=False)["Confidence_Adjusted_Fair_Value_Score"].apply(pct_rank_high)
        )

    if {"Combined_Score_RankPct", "Confidence_Adjusted_Fair_Value_RankPct", "Stock_Price_Trend_Score_RankPct"}.issubset(out.columns):
        out["Confidence_Adjusted_Combined_RankPct"] = weighted_average_columns(out, {
            "Combined_Score_RankPct": 0.60,
            "Confidence_Adjusted_Fair_Value_RankPct": 0.25,
            "Stock_Price_Trend_Score_RankPct": 0.15,
        })

    relative_rows = []
    anchors = sorted(out["Date"].dropna().unique())
    for anchor in anchors:
        anchor = pd.to_datetime(anchor).normalize()
        row: dict[str, pd.Series] = {}
        for window in RELATIVE_MOMENTUM_WINDOWS:
            trailing = trailing_return_row(price_wide, anchor, window)
            volatility = trailing_volatility_row(price_wide, anchor, window)
            downside_volatility = trailing_downside_volatility_row(price_wide, anchor, window)
            max_drawdown = trailing_max_drawdown_row(price_wide, anchor, window)
            if trailing.empty:
                row[f"Return_{window}D"] = pd.Series(dtype="float64")
                row[f"Excess_Return_{window}D_vs_XLK"] = pd.Series(dtype="float64")
                row[f"Volatility_{window}D"] = pd.Series(dtype="float64")
                row[f"Downside_Volatility_{window}D"] = pd.Series(dtype="float64")
                row[f"Max_Drawdown_{window}D"] = pd.Series(dtype="float64")
                row[f"Risk_Adjusted_Return_{window}D"] = pd.Series(dtype="float64")
                continue
            row[f"Return_{window}D"] = trailing
            xlk_return = trailing.get("XLK", np.nan)
            row[f"Excess_Return_{window}D_vs_XLK"] = trailing - xlk_return
            row[f"Volatility_{window}D"] = volatility
            row[f"Downside_Volatility_{window}D"] = downside_volatility
            row[f"Max_Drawdown_{window}D"] = max_drawdown
            row[f"Risk_Adjusted_Return_{window}D"] = trailing / volatility.replace(0.0, np.nan)

        tmp = out.loc[out["Date"] == anchor, ["Ticker", "SubIndustry"]].drop_duplicates().copy()
        tmp["Date"] = anchor
        for name, series in row.items():
            tmp[name] = tmp["Ticker"].map(series)
        for window in RELATIVE_MOMENTUM_WINDOWS:
            return_col = f"Return_{window}D"
            if return_col in tmp.columns:
                tmp[f"Excess_Return_{window}D_vs_SubIndustry"] = (
                    tmp[return_col]
                    - tmp.groupby("SubIndustry")[return_col].transform("mean")
                )
        tmp = tmp.drop(columns=["SubIndustry"])
        relative_rows.append(tmp)

    if relative_rows:
        relative = pd.concat(relative_rows, ignore_index=True, sort=False)
        out = out.merge(relative, on=["Date", "Ticker"], how="left")

        rel_cols = [f"Excess_Return_{window}D_vs_XLK" for window in RELATIVE_MOMENTUM_WINDOWS]
        sub_rel_cols = [f"Excess_Return_{window}D_vs_SubIndustry" for window in RELATIVE_MOMENTUM_WINDOWS]
        abs_cols = [f"Return_{window}D" for window in RELATIVE_MOMENTUM_WINDOWS]
        risk_adj_cols = [f"Risk_Adjusted_Return_{window}D" for window in RELATIVE_MOMENTUM_WINDOWS]
        vol_cols = [f"Volatility_{window}D" for window in RELATIVE_MOMENTUM_WINDOWS]
        downside_vol_cols = [f"Downside_Volatility_{window}D" for window in RELATIVE_MOMENTUM_WINDOWS]
        drawdown_cols = [f"Max_Drawdown_{window}D" for window in RELATIVE_MOMENTUM_WINDOWS]
        for col in rel_cols + sub_rel_cols + abs_cols + risk_adj_cols:
            if col in out.columns:
                out[f"{col}_RankPct"] = out.groupby("Date", group_keys=False)[col].apply(pct_rank_high)
        for col in vol_cols:
            if col in out.columns:
                out[f"Inverse_{col}_RankPct"] = out.groupby("Date", group_keys=False)[col].apply(lambda s: pct_rank_high(-s))
        for col in downside_vol_cols:
            if col in out.columns:
                out[f"Inverse_{col}_RankPct"] = out.groupby("Date", group_keys=False)[col].apply(lambda s: pct_rank_high(-s))
        for col in drawdown_cols:
            if col in out.columns:
                out[f"{col}_RankPct"] = out.groupby("Date", group_keys=False)[col].apply(pct_rank_high)

        rank_cols = [f"{col}_RankPct" for col in rel_cols if f"{col}_RankPct" in out.columns]
        if rank_cols:
            out["Relative_Momentum_vs_XLK_RankPct"] = out[rank_cols].mean(axis=1, skipna=True)
        sub_rank_cols = [f"{col}_RankPct" for col in sub_rel_cols if f"{col}_RankPct" in out.columns]
        if sub_rank_cols:
            out["Relative_Momentum_vs_SubIndustry_RankPct"] = out[sub_rank_cols].mean(axis=1, skipna=True)
        abs_rank_cols = [f"{col}_RankPct" for col in abs_cols if f"{col}_RankPct" in out.columns]
        if abs_rank_cols:
            out["Absolute_Momentum_RankPct"] = out[abs_rank_cols].mean(axis=1, skipna=True)
        risk_rank_cols = [f"{col}_RankPct" for col in risk_adj_cols if f"{col}_RankPct" in out.columns]
        if risk_rank_cols:
            out["Risk_Adjusted_Momentum_RankPct"] = out[risk_rank_cols].mean(axis=1, skipna=True)
        inverse_vol_rank_cols = [f"Inverse_{col}_RankPct" for col in vol_cols if f"Inverse_{col}_RankPct" in out.columns]
        if inverse_vol_rank_cols:
            out["Low_Volatility_RankPct"] = out[inverse_vol_rank_cols].mean(axis=1, skipna=True)
        out["SubIndustry_Relative_Momentum_MidLong_RankPct"] = weighted_average_columns(out, {
            "Excess_Return_63D_vs_SubIndustry_RankPct": 0.35,
            "Excess_Return_126D_vs_SubIndustry_RankPct": 0.65,
        })
        out["SubIndustry_Relative_Momentum_Quality_RankPct"] = weighted_average_columns(out, {
            "SubIndustry_Relative_Momentum_MidLong_RankPct": 0.60,
            "Risk_Adjusted_Return_126D_RankPct": 0.25,
            "Max_Drawdown_126D_RankPct": 0.15,
        })
        out["Downside_Quality_RankPct"] = weighted_average_columns(out, {
            "Max_Drawdown_63D_RankPct": 0.30,
            "Max_Drawdown_126D_RankPct": 0.30,
            "Inverse_Downside_Volatility_63D_RankPct": 0.20,
            "Inverse_Downside_Volatility_126D_RankPct": 0.20,
        })

    if {"Combined_Score_RankPct", "Relative_Momentum_vs_XLK_RankPct"}.issubset(out.columns):
        out["Alpha_Overlay_70Combined_30RelMom"] = (
            0.70 * out["Combined_Score_RankPct"] + 0.30 * out["Relative_Momentum_vs_XLK_RankPct"]
        )

    if {"Fair_Value_Score_RankPct", "Stock_Price_Trend_Score_RankPct", "Relative_Momentum_vs_XLK_RankPct"}.issubset(out.columns):
        out["Alpha_Prototype_ValueTrendRelMom"] = (
            0.40 * out["Fair_Value_Score_RankPct"]
            + 0.30 * out["Stock_Price_Trend_Score_RankPct"]
            + 0.30 * out["Relative_Momentum_vs_XLK_RankPct"]
        )

    if {"SubIndustry_Neutral_Combined_RankPct", "Relative_Momentum_vs_XLK_RankPct"}.issubset(out.columns):
        out["Alpha_Prototype_SubIndustryNeutral"] = (
            0.70 * out["SubIndustry_Neutral_Combined_RankPct"]
            + 0.30 * out["Relative_Momentum_vs_XLK_RankPct"]
        )

    if {"Combined_Score_RankPct", "Relative_Momentum_vs_SubIndustry_RankPct", "Risk_Adjusted_Momentum_RankPct"}.issubset(out.columns):
        out["Alpha_Prototype_SubIndustryRelRisk"] = (
            0.60 * out["Combined_Score_RankPct"]
            + 0.20 * out["Relative_Momentum_vs_SubIndustry_RankPct"]
            + 0.20 * out["Risk_Adjusted_Momentum_RankPct"]
        )

    if {"Fair_Value_Score_RankPct", "Relative_Momentum_vs_SubIndustry_RankPct", "Risk_Adjusted_Momentum_RankPct"}.issubset(out.columns):
        out["Alpha_Prototype_ValueSubRelRisk"] = (
            0.50 * out["Fair_Value_Score_RankPct"]
            + 0.25 * out["Relative_Momentum_vs_SubIndustry_RankPct"]
            + 0.25 * out["Risk_Adjusted_Momentum_RankPct"]
        )

    if {"Confidence_Adjusted_Fair_Value_RankPct", "SubIndustry_Relative_Momentum_MidLong_RankPct", "Downside_Quality_RankPct"}.issubset(out.columns):
        out["Alpha_Prototype_ConfidenceValueSubRelRisk"] = (
            0.45 * out["Confidence_Adjusted_Fair_Value_RankPct"]
            + 0.35 * out["SubIndustry_Relative_Momentum_MidLong_RankPct"]
            + 0.20 * out["Downside_Quality_RankPct"]
        )

    if {"Combined_Score_RankPct", "SubIndustry_Relative_Momentum_MidLong_RankPct", "Risk_Adjusted_Momentum_RankPct", "Downside_Quality_RankPct"}.issubset(out.columns):
        out["Alpha_Prototype_BenchmarkAwareQuality"] = (
            0.35 * out["Combined_Score_RankPct"]
            + 0.35 * out["SubIndustry_Relative_Momentum_MidLong_RankPct"]
            + 0.15 * out["Risk_Adjusted_Momentum_RankPct"]
            + 0.15 * out["Downside_Quality_RankPct"]
        )

    if {"Confidence_Adjusted_Fair_Value_RankPct", "SubIndustry_Relative_Momentum_MidLong_RankPct", "Risk_Adjusted_Momentum_RankPct", "Downside_Quality_RankPct"}.issubset(out.columns):
        out["Alpha_Prototype_XLKCompetitive"] = (
            0.30 * out["Confidence_Adjusted_Fair_Value_RankPct"]
            + 0.45 * out["SubIndustry_Relative_Momentum_MidLong_RankPct"]
            + 0.15 * out["Risk_Adjusted_Momentum_RankPct"]
            + 0.10 * out["Downside_Quality_RankPct"]
        )

    if {"SubIndustry_Relative_Momentum_MidLong_RankPct", "Risk_Adjusted_Momentum_RankPct", "Downside_Quality_RankPct"}.issubset(out.columns):
        out["Alpha_Prototype_MomentumQuality"] = (
            0.60 * out["SubIndustry_Relative_Momentum_MidLong_RankPct"]
            + 0.25 * out["Risk_Adjusted_Momentum_RankPct"]
            + 0.15 * out["Downside_Quality_RankPct"]
        )

    if {"Combined_Score_RankPct", "Confidence_Adjusted_Fair_Value_RankPct", "SubIndustry_Relative_Momentum_MidLong_RankPct", "Downside_Quality_RankPct"}.issubset(out.columns):
        out["Alpha_Prototype_ConservativeComposite"] = (
            0.40 * out["Combined_Score_RankPct"]
            + 0.25 * out["Confidence_Adjusted_Fair_Value_RankPct"]
            + 0.25 * out["SubIndustry_Relative_Momentum_MidLong_RankPct"]
            + 0.10 * out["Downside_Quality_RankPct"]
        )

    return out


def signal_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "Combined_Score",
        "Combined_Score_RankPct",
        "Fair_Value_Score",
        "Fair_Value_Score_RankPct",
        "Stock_Price_Trend_Score",
        "Stock_Price_Trend_Score_RankPct",
        "SubIndustry_Price_Trend_Score",
        "Industry_Price_Trend_Score",
        "SubIndustry_Neutral_Combined_RankPct",
        "SubIndustry_Neutral_Fair_Value_RankPct",
        "Coverage_Adjusted_Combined_RankPct",
        "Fair_Value_Confidence",
        "Confidence_Adjusted_Fair_Value_Score",
        "Confidence_Adjusted_Fair_Value_RankPct",
        "Confidence_Adjusted_Combined_RankPct",
        "Relative_Momentum_vs_XLK_RankPct",
        "Relative_Momentum_vs_SubIndustry_RankPct",
        "SubIndustry_Relative_Momentum_MidLong_RankPct",
        "SubIndustry_Relative_Momentum_Quality_RankPct",
        "Absolute_Momentum_RankPct",
        "Risk_Adjusted_Momentum_RankPct",
        "Low_Volatility_RankPct",
        "Downside_Quality_RankPct",
        "Alpha_Overlay_70Combined_30RelMom",
        "Alpha_Prototype_ValueTrendRelMom",
        "Alpha_Prototype_SubIndustryNeutral",
        "Alpha_Prototype_SubIndustryRelRisk",
        "Alpha_Prototype_ValueSubRelRisk",
        "Alpha_Prototype_ConfidenceValueSubRelRisk",
        "Alpha_Prototype_BenchmarkAwareQuality",
        "Alpha_Prototype_XLKCompetitive",
        "Alpha_Prototype_MomentumQuality",
        "Alpha_Prototype_ConservativeComposite",
    ]
    trailing = []
    for window in RELATIVE_MOMENTUM_WINDOWS:
        trailing.extend([
            f"Return_{window}D_RankPct",
            f"Excess_Return_{window}D_vs_XLK_RankPct",
            f"Excess_Return_{window}D_vs_SubIndustry_RankPct",
            f"Risk_Adjusted_Return_{window}D_RankPct",
            f"Inverse_Volatility_{window}D_RankPct",
            f"Inverse_Downside_Volatility_{window}D_RankPct",
            f"Max_Drawdown_{window}D_RankPct",
        ])
    return [col for col in preferred + trailing if col in frame.columns]


def build_forward_return_panel(scores: pd.DataFrame, price_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for anchor in sorted(scores["Date"].dropna().unique()):
        anchor = pd.to_datetime(anchor).normalize()
        anchor_scores = scores[scores["Date"] == anchor]
        tickers = anchor_scores["Ticker"].unique()
        for horizon_name, horizon_days in HORIZONS.items():
            future_date, returns = future_return_row(price_wide, anchor, horizon_days)
            if pd.isna(future_date) or returns.empty:
                continue
            tmp = anchor_scores[["Date", "Ticker", "SubIndustry"]].copy()
            tmp["Horizon"] = horizon_name
            tmp["Future_Date"] = future_date
            tmp["Forward_Return"] = tmp["Ticker"].map(returns)
            tmp["Universe_Equal_Return"] = finite_mean(returns.reindex(tickers))
            for benchmark in BENCHMARKS:
                tmp[f"{benchmark}_Return"] = returns.get(benchmark, np.nan)
                tmp[f"Excess_vs_{benchmark}"] = tmp["Forward_Return"] - tmp[f"{benchmark}_Return"]
            tmp["Excess_vs_Universe"] = tmp["Forward_Return"] - tmp["Universe_Equal_Return"]
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def calculate_ic(
    analysis: pd.DataFrame,
    signals: list[str],
    min_observations: int,
) -> pd.DataFrame:
    rows = []
    for (anchor, horizon), group in analysis.groupby(["Date", "Horizon"], sort=True):
        target = pd.to_numeric(group["Forward_Return"], errors="coerce")
        for signal in signals:
            values = pd.to_numeric(group[signal], errors="coerce")
            valid = pd.concat([values.rename("signal"), target.rename("target")], axis=1).dropna()
            if len(valid) < min_observations or valid["signal"].nunique() <= 1 or valid["target"].nunique() <= 1:
                continue
            rows.append({
                "Date": anchor,
                "Horizon": horizon,
                "Signal": signal,
                "Observations": int(len(valid)),
                "Spearman_IC": float(valid["signal"].rank().corr(valid["target"].rank())),
                "Pearson_IC": float(valid["signal"].corr(valid["target"])),
            })
    return pd.DataFrame(rows)


def summarize_ic(ic: pd.DataFrame) -> pd.DataFrame:
    if ic.empty:
        return pd.DataFrame()
    grouped = ic.groupby(["Signal", "Horizon"], sort=True)
    out = grouped.agg(
        Anchor_Count=("Date", "nunique"),
        Mean_Spearman_IC=("Spearman_IC", "mean"),
        Median_Spearman_IC=("Spearman_IC", "median"),
        Spearman_IC_Std=("Spearman_IC", lambda s: s.std(ddof=0)),
        Positive_IC_Rate=("Spearman_IC", lambda s: float((s > 0).mean())),
        Mean_Pearson_IC=("Pearson_IC", "mean"),
        Mean_Observations=("Observations", "mean"),
    ).reset_index()
    out["IC_TStat"] = out["Mean_Spearman_IC"] / (
        out["Spearman_IC_Std"] / np.sqrt(out["Anchor_Count"].clip(lower=1))
    )
    out.loc[out["Spearman_IC_Std"] <= 0, "IC_TStat"] = np.nan
    return out.sort_values(["Horizon", "Mean_Spearman_IC"], ascending=[True, False])


def calculate_quintiles(
    analysis: pd.DataFrame,
    signals: list[str],
    min_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bucket_rows = []
    spread_rows = []
    for (anchor, horizon), group in analysis.groupby(["Date", "Horizon"], sort=True):
        for signal in signals:
            valid = group[[signal, "Forward_Return", "Excess_vs_Universe", "Excess_vs_XLK"]].copy()
            valid[signal] = pd.to_numeric(valid[signal], errors="coerce")
            valid = valid.dropna(subset=[signal, "Forward_Return"])
            if len(valid) < min_observations or valid[signal].nunique() < 5:
                continue
            try:
                valid["Quintile"] = pd.qcut(
                    valid[signal].rank(method="first"),
                    q=5,
                    labels=[1, 2, 3, 4, 5],
                ).astype(int)
            except ValueError:
                continue

            grouped = valid.groupby("Quintile", sort=True)
            means = grouped["Forward_Return"].mean()
            for quintile, bucket in grouped:
                bucket_rows.append({
                    "Date": anchor,
                    "Horizon": horizon,
                    "Signal": signal,
                    "Quintile": int(quintile),
                    "Count": int(len(bucket)),
                    "Mean_Forward_Return": float(bucket["Forward_Return"].mean()),
                    "Mean_Excess_vs_Universe": float(bucket["Excess_vs_Universe"].mean()),
                    "Mean_Excess_vs_XLK": float(bucket["Excess_vs_XLK"].mean()),
                })

            if 1 in means.index and 5 in means.index:
                adjacent_increases = int(sum(means.loc[i + 1] > means.loc[i] for i in range(1, 5)))
                spread_rows.append({
                    "Date": anchor,
                    "Horizon": horizon,
                    "Signal": signal,
                    "Top_Bottom_Spread": float(means.loc[5] - means.loc[1]),
                    "Adjacent_Increases": adjacent_increases,
                    "Monotonicity_Rate": adjacent_increases / 4.0,
                    "Top_Quintile_Return": float(means.loc[5]),
                    "Bottom_Quintile_Return": float(means.loc[1]),
                })
    return pd.DataFrame(bucket_rows), pd.DataFrame(spread_rows)


def summarize_quintiles(bucket: pd.DataFrame, spread: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bucket.empty:
        bucket_summary = pd.DataFrame()
    else:
        bucket_summary = (
            bucket.groupby(["Signal", "Horizon", "Quintile"], sort=True)
            .agg(
                Anchor_Count=("Date", "nunique"),
                Mean_Forward_Return=("Mean_Forward_Return", "mean"),
                Mean_Excess_vs_Universe=("Mean_Excess_vs_Universe", "mean"),
                Mean_Excess_vs_XLK=("Mean_Excess_vs_XLK", "mean"),
                Mean_Count=("Count", "mean"),
            )
            .reset_index()
        )

    if spread.empty:
        spread_summary = pd.DataFrame()
    else:
        spread_summary = (
            spread.groupby(["Signal", "Horizon"], sort=True)
            .agg(
                Anchor_Count=("Date", "nunique"),
                Mean_Top_Bottom_Spread=("Top_Bottom_Spread", "mean"),
                Median_Top_Bottom_Spread=("Top_Bottom_Spread", "median"),
                Positive_Spread_Rate=("Top_Bottom_Spread", lambda s: float((s > 0).mean())),
                Mean_Monotonicity_Rate=("Monotonicity_Rate", "mean"),
                Mean_Top_Quintile_Return=("Top_Quintile_Return", "mean"),
                Mean_Bottom_Quintile_Return=("Bottom_Quintile_Return", "mean"),
            )
            .reset_index()
        )
    return bucket_summary, spread_summary


def calculate_topn(
    analysis: pd.DataFrame,
    signals: list[str],
    min_observations: int,
) -> pd.DataFrame:
    rows = []
    for (anchor, horizon), group in analysis.groupby(["Date", "Horizon"], sort=True):
        universe_return = finite_mean(group["Forward_Return"])
        qqq_return = finite_mean(group["QQQ_Return"])
        xlk_return = finite_mean(group["XLK_Return"])
        for signal in signals:
            valid = group[[signal, "Ticker", "Forward_Return"]].copy()
            valid[signal] = pd.to_numeric(valid[signal], errors="coerce")
            valid = valid.dropna(subset=[signal, "Forward_Return"])
            if len(valid) < min_observations:
                continue
            valid = valid.sort_values(signal, ascending=False)
            for top_n in TOP_N_VALUES:
                if len(valid) < top_n:
                    continue
                basket = valid.head(top_n)
                basket_return = finite_mean(basket["Forward_Return"])
                rows.append({
                    "Date": anchor,
                    "Horizon": horizon,
                    "Signal": signal,
                    "Top_N": top_n,
                    "Holdings": int(len(basket)),
                    "Mean_Forward_Return": basket_return,
                    "Universe_Equal_Return": universe_return,
                    "QQQ_Return": qqq_return,
                    "XLK_Return": xlk_return,
                    "Excess_vs_Universe": basket_return - universe_return,
                    "Excess_vs_QQQ": basket_return - qqq_return,
                    "Excess_vs_XLK": basket_return - xlk_return,
                })
    return pd.DataFrame(rows)


def summarize_topn(topn: pd.DataFrame) -> pd.DataFrame:
    if topn.empty:
        return pd.DataFrame()
    return (
        topn.groupby(["Signal", "Horizon", "Top_N"], sort=True)
        .agg(
            Anchor_Count=("Date", "nunique"),
            Mean_Forward_Return=("Mean_Forward_Return", "mean"),
            Mean_Excess_vs_Universe=("Excess_vs_Universe", "mean"),
            Mean_Excess_vs_QQQ=("Excess_vs_QQQ", "mean"),
            Mean_Excess_vs_XLK=("Excess_vs_XLK", "mean"),
            Universe_Hit_Rate=("Excess_vs_Universe", lambda s: float((s > 0).mean())),
            QQQ_Hit_Rate=("Excess_vs_QQQ", lambda s: float((s > 0).mean())),
            XLK_Hit_Rate=("Excess_vs_XLK", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values(["Horizon", "Top_N", "Mean_Excess_vs_Universe"], ascending=[True, True, False])
    )


def summarize_ic_by_year(ic: pd.DataFrame) -> pd.DataFrame:
    if ic.empty:
        return pd.DataFrame()
    out = ic.copy()
    out["Anchor_Year"] = pd.to_datetime(out["Date"], errors="coerce").dt.year
    return (
        out.groupby(["Anchor_Year", "Signal", "Horizon"], sort=True)
        .agg(
            Anchor_Count=("Date", "nunique"),
            Mean_Spearman_IC=("Spearman_IC", "mean"),
            Positive_IC_Rate=("Spearman_IC", lambda s: float((s > 0).mean())),
            Mean_Pearson_IC=("Pearson_IC", "mean"),
            Mean_Observations=("Observations", "mean"),
        )
        .reset_index()
        .sort_values(["Anchor_Year", "Horizon", "Mean_Spearman_IC"], ascending=[True, True, False])
    )


def summarize_topn_by_year(topn: pd.DataFrame) -> pd.DataFrame:
    if topn.empty:
        return pd.DataFrame()
    out = topn.copy()
    out["Anchor_Year"] = pd.to_datetime(out["Date"], errors="coerce").dt.year
    return (
        out.groupby(["Anchor_Year", "Signal", "Horizon", "Top_N"], sort=True)
        .agg(
            Anchor_Count=("Date", "nunique"),
            Mean_Forward_Return=("Mean_Forward_Return", "mean"),
            Mean_Excess_vs_Universe=("Excess_vs_Universe", "mean"),
            Mean_Excess_vs_QQQ=("Excess_vs_QQQ", "mean"),
            Mean_Excess_vs_XLK=("Excess_vs_XLK", "mean"),
            Universe_Hit_Rate=("Excess_vs_Universe", lambda s: float((s > 0).mean())),
            XLK_Hit_Rate=("Excess_vs_XLK", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values(["Anchor_Year", "Horizon", "Top_N", "Mean_Excess_vs_Universe"], ascending=[True, True, True, False])
    )


def build_overall_scorecard(
    ic_summary: pd.DataFrame,
    spread_summary: pd.DataFrame,
    topn_summary: pd.DataFrame,
) -> pd.DataFrame:
    signals = sorted(set(ic_summary.get("Signal", [])) | set(spread_summary.get("Signal", [])) | set(topn_summary.get("Signal", [])))
    rows = []
    for signal in signals:
        ic_part = ic_summary[ic_summary["Signal"] == signal] if not ic_summary.empty else pd.DataFrame()
        spread_part = spread_summary[spread_summary["Signal"] == signal] if not spread_summary.empty else pd.DataFrame()
        top100_part = topn_summary[(topn_summary["Signal"] == signal) & (topn_summary["Top_N"] == 100)] if not topn_summary.empty else pd.DataFrame()
        top50_part = topn_summary[(topn_summary["Signal"] == signal) & (topn_summary["Top_N"] == 50)] if not topn_summary.empty else pd.DataFrame()
        rows.append({
            "Signal": signal,
            "Mean_Spearman_IC_All_Horizons": finite_mean(ic_part.get("Mean_Spearman_IC", [])),
            "Mean_Positive_IC_Rate": finite_mean(ic_part.get("Positive_IC_Rate", [])),
            "Mean_Top_Bottom_Spread": finite_mean(spread_part.get("Mean_Top_Bottom_Spread", [])),
            "Mean_Positive_Spread_Rate": finite_mean(spread_part.get("Positive_Spread_Rate", [])),
            "Top100_Mean_Excess_vs_Universe": finite_mean(top100_part.get("Mean_Excess_vs_Universe", [])),
            "Top100_Mean_Excess_vs_XLK": finite_mean(top100_part.get("Mean_Excess_vs_XLK", [])),
            "Top50_Mean_Excess_vs_Universe": finite_mean(top50_part.get("Mean_Excess_vs_Universe", [])),
            "Top50_Mean_Excess_vs_XLK": finite_mean(top50_part.get("Mean_Excess_vs_XLK", [])),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Quality_Score"] = (
        out["Mean_Spearman_IC_All_Horizons"].fillna(0.0) * 100.0
        + out["Mean_Top_Bottom_Spread"].fillna(0.0) * 100.0
        + out["Top100_Mean_Excess_vs_Universe"].fillna(0.0) * 100.0
        + out["Mean_Positive_IC_Rate"].fillna(0.0) * 10.0
    )
    return out.sort_values("Quality_Score", ascending=False).reset_index(drop=True)


def format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def format_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_summary(
    output_dir: Path,
    scores: pd.DataFrame,
    signal_list: list[str],
    ic_summary: pd.DataFrame,
    spread_summary: pd.DataFrame,
    topn_summary: pd.DataFrame,
    yearly_topn_summary: pd.DataFrame,
    scorecard: pd.DataFrame,
    metadata: dict,
) -> None:
    lines = []
    lines.append("# Alpha Signal Quality Summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This run tests whether the existing Stock Analysis scores and several research-only alpha overlays "
        "predicted later returns. It uses point-in-time score rows and canonical Sharadar adjusted prices. "
        "The script does not change production scoring."
    )
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Score rows: {len(scores):,}")
    lines.append(f"- Monthly anchors: {scores['Date'].nunique():,}")
    lines.append(f"- Date range: {scores['Date'].min().date()} to {scores['Date'].max().date()}")
    lines.append(f"- Unique tickers: {scores['Ticker'].nunique():,}")
    lines.append(f"- Signals tested: {len(signal_list):,}")
    lines.append("")

    lines.append("## Best Overall Signals")
    lines.append("")
    if scorecard.empty:
        lines.append("No signal scorecard rows were produced.")
    else:
        for _, row in scorecard.head(10).iterrows():
            lines.append(
                "- "
                f"`{row['Signal']}`: mean IC {format_float(row['Mean_Spearman_IC_All_Horizons'])}, "
                f"top-bottom spread {format_pct(row['Mean_Top_Bottom_Spread'])}, "
                f"top-100 excess vs universe {format_pct(row['Top100_Mean_Excess_vs_Universe'])}, "
                f"top-100 excess vs XLK {format_pct(row['Top100_Mean_Excess_vs_XLK'])}"
            )
    lines.append("")

    lines.append("## Current Production Combined Score")
    lines.append("")
    current = scorecard[scorecard["Signal"] == "Combined_Score"] if not scorecard.empty else pd.DataFrame()
    if current.empty:
        lines.append("`Combined_Score` did not produce enough complete observations for the aggregate scorecard.")
    else:
        row = current.iloc[0]
        lines.append(
            f"`Combined_Score` averaged mean IC {format_float(row['Mean_Spearman_IC_All_Horizons'])}, "
            f"top-bottom spread {format_pct(row['Mean_Top_Bottom_Spread'])}, "
            f"top-100 excess vs universe {format_pct(row['Top100_Mean_Excess_vs_Universe'])}, "
            f"and top-100 excess vs XLK {format_pct(row['Top100_Mean_Excess_vs_XLK'])}."
        )
    lines.append("")

    lines.append("## Horizon Details")
    lines.append("")
    if topn_summary.empty:
        lines.append("No top-N summary rows were produced.")
    else:
        focus_signals = [
            "Combined_Score",
            "Fair_Value_Score",
            "Confidence_Adjusted_Fair_Value_RankPct",
            "Stock_Price_Trend_Score",
            "Relative_Momentum_vs_XLK_RankPct",
            "Relative_Momentum_vs_SubIndustry_RankPct",
            "SubIndustry_Relative_Momentum_MidLong_RankPct",
            "Risk_Adjusted_Momentum_RankPct",
            "Downside_Quality_RankPct",
            "Alpha_Overlay_70Combined_30RelMom",
            "Alpha_Prototype_ValueTrendRelMom",
            "Alpha_Prototype_SubIndustryNeutral",
            "Alpha_Prototype_SubIndustryRelRisk",
            "Alpha_Prototype_ValueSubRelRisk",
            "Alpha_Prototype_ConfidenceValueSubRelRisk",
            "Alpha_Prototype_BenchmarkAwareQuality",
            "Alpha_Prototype_XLKCompetitive",
            "Alpha_Prototype_MomentumQuality",
            "Alpha_Prototype_ConservativeComposite",
        ]
        focus = topn_summary[
            topn_summary["Signal"].isin(focus_signals) & (topn_summary["Top_N"] == 100)
        ].sort_values(["Horizon", "Mean_Excess_vs_Universe"], ascending=[True, False])
        for _, row in focus.iterrows():
            lines.append(
                "- "
                f"{row['Horizon']} `{row['Signal']}` top 100: "
                f"return {format_pct(row['Mean_Forward_Return'])}, "
                f"excess vs universe {format_pct(row['Mean_Excess_vs_Universe'])}, "
                f"excess vs XLK {format_pct(row['Mean_Excess_vs_XLK'])}, "
                f"universe hit rate {format_pct(row['Universe_Hit_Rate'])}"
            )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "A healthy alpha signal should show positive average rank correlation, positive top-bottom quintile spread, "
        "and top-N portfolios that beat the investable universe out of sample. Beating XLK is harder here because "
        "the universe is technology-heavy and XLK has been an exceptionally strong benchmark over much of the test."
    )
    lines.append(
        "The research-only overlay signals are candidates for model-quality work. They should be promoted into "
        "production only after they improve walk-forward portfolio results, not merely this single diagnostic report."
    )
    lines.append(
        "XLK-relative momentum has the same cross-sectional rank as absolute momentum when every stock is "
        "compared against the same XLK return on the same anchor date. The more meaningful relative signal is "
        "subindustry-relative momentum, because that changes each stock's rank inside its peer group."
    )
    lines.append("")

    lines.append("## Year-By-Year Stability")
    lines.append("")
    if yearly_topn_summary.empty:
        lines.append("No yearly top-N summary rows were produced.")
    else:
        focus_signals = [
            "Combined_Score",
            "Alpha_Prototype_ValueTrendRelMom",
            "Alpha_Prototype_SubIndustryRelRisk",
            "Alpha_Prototype_ValueSubRelRisk",
            "Alpha_Prototype_ConfidenceValueSubRelRisk",
            "Alpha_Prototype_BenchmarkAwareQuality",
            "Alpha_Prototype_XLKCompetitive",
            "Alpha_Prototype_MomentumQuality",
            "Alpha_Prototype_ConservativeComposite",
        ]
        focus = yearly_topn_summary[
            yearly_topn_summary["Signal"].isin(focus_signals)
            & (yearly_topn_summary["Top_N"] == 100)
            & (yearly_topn_summary["Horizon"].isin(["6M", "12M"]))
        ].sort_values(["Anchor_Year", "Horizon", "Signal"])
        for _, row in focus.iterrows():
            lines.append(
                "- "
                f"{int(row['Anchor_Year'])} {row['Horizon']} `{row['Signal']}` top 100: "
                f"excess vs universe {format_pct(row['Mean_Excess_vs_Universe'])}, "
                f"excess vs XLK {format_pct(row['Mean_Excess_vs_XLK'])}, "
                f"anchors {int(row['Anchor_Count'])}"
            )
    lines.append("")

    lines.append("## Files Produced")
    lines.append("")
    for name in [
        "analysis_panel.csv",
        "signal_ic_by_anchor.csv",
        "signal_ic_summary.csv",
        "yearly_signal_ic_summary.csv",
        "quintile_returns_by_anchor.csv",
        "quintile_return_summary.csv",
        "quintile_spread_summary.csv",
        "topn_returns_by_anchor.csv",
        "topn_return_summary.csv",
        "yearly_topn_return_summary.csv",
        "overall_signal_scorecard.csv",
        "run_metadata.json",
    ]:
        lines.append(f"- `{name}`")

    (output_dir / "ALPHA_SIGNAL_QUALITY_SUMMARY.md").write_text("\n".join(lines) + "\n")
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(args.score_history)
    tickers = sorted(set(scores["Ticker"]) | set(BENCHMARKS))
    start_date = scores["Date"].min() - pd.Timedelta(days=260)
    end_date = scores["Date"].max() + pd.Timedelta(days=370)
    price_wide = historical_data.load_canonical_price_wide(
        args.historical_price_file,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        value_column="Close",
    )

    enriched_scores = attach_ranked_signals(scores, price_wide)
    signals = signal_columns(enriched_scores)
    forward = build_forward_return_panel(enriched_scores, price_wide)
    if forward.empty:
        raise RuntimeError("No forward-return rows were generated. Check price coverage and anchor dates.")

    analysis = forward.merge(
        enriched_scores,
        on=["Date", "Ticker", "SubIndustry"],
        how="left",
        suffixes=("", "_Score"),
    )
    analysis = analysis.dropna(subset=["Forward_Return"]).copy()

    ic = calculate_ic(analysis, signals, args.min_observations)
    ic_summary = summarize_ic(ic)
    yearly_ic_summary = summarize_ic_by_year(ic)
    quintile, quintile_spread = calculate_quintiles(analysis, signals, args.min_observations)
    quintile_summary, spread_summary = summarize_quintiles(quintile, quintile_spread)
    topn = calculate_topn(analysis, signals, args.min_observations)
    topn_summary = summarize_topn(topn)
    yearly_topn_summary = summarize_topn_by_year(topn)
    scorecard = build_overall_scorecard(ic_summary, spread_summary, topn_summary)

    analysis.to_csv(args.output_dir / "analysis_panel.csv", index=False)
    ic.to_csv(args.output_dir / "signal_ic_by_anchor.csv", index=False)
    ic_summary.to_csv(args.output_dir / "signal_ic_summary.csv", index=False)
    yearly_ic_summary.to_csv(args.output_dir / "yearly_signal_ic_summary.csv", index=False)
    quintile.to_csv(args.output_dir / "quintile_returns_by_anchor.csv", index=False)
    quintile_summary.to_csv(args.output_dir / "quintile_return_summary.csv", index=False)
    spread_summary.to_csv(args.output_dir / "quintile_spread_summary.csv", index=False)
    topn.to_csv(args.output_dir / "topn_returns_by_anchor.csv", index=False)
    topn_summary.to_csv(args.output_dir / "topn_return_summary.csv", index=False)
    yearly_topn_summary.to_csv(args.output_dir / "yearly_topn_return_summary.csv", index=False)
    scorecard.to_csv(args.output_dir / "overall_signal_scorecard.csv", index=False)

    metadata = {
        "status": "complete",
        "score_history": str(args.score_history),
        "historical_price_file": str(args.historical_price_file),
        "output_dir": str(args.output_dir),
        "score_rows": int(len(scores)),
        "analysis_rows": int(len(analysis)),
        "anchor_count": int(scores["Date"].nunique()),
        "ticker_count": int(scores["Ticker"].nunique()),
        "signals_tested": signals,
        "horizons": HORIZONS,
        "benchmarks": BENCHMARKS,
        "relative_momentum_windows": RELATIVE_MOMENTUM_WINDOWS,
        "min_observations": args.min_observations,
        "research_only_candidate_signals": [
            "Relative_Momentum_vs_XLK_RankPct",
            "Relative_Momentum_vs_SubIndustry_RankPct",
            "Risk_Adjusted_Momentum_RankPct",
            "Alpha_Overlay_70Combined_30RelMom",
            "Alpha_Prototype_ValueTrendRelMom",
            "Alpha_Prototype_SubIndustryNeutral",
            "Alpha_Prototype_SubIndustryRelRisk",
            "Alpha_Prototype_ValueSubRelRisk",
            "Coverage_Adjusted_Combined_RankPct",
            "Fair_Value_Confidence",
            "Confidence_Adjusted_Fair_Value_RankPct",
            "Confidence_Adjusted_Combined_RankPct",
            "SubIndustry_Relative_Momentum_MidLong_RankPct",
            "SubIndustry_Relative_Momentum_Quality_RankPct",
            "Downside_Quality_RankPct",
            "Alpha_Prototype_ConfidenceValueSubRelRisk",
            "Alpha_Prototype_BenchmarkAwareQuality",
            "Alpha_Prototype_XLKCompetitive",
            "Alpha_Prototype_MomentumQuality",
            "Alpha_Prototype_ConservativeComposite",
        ],
    }
    write_summary(
        args.output_dir,
        scores,
        signals,
        ic_summary,
        spread_summary,
        topn_summary,
        yearly_topn_summary,
        scorecard,
        metadata,
    )

    print(f"Wrote alpha signal quality report to {args.output_dir}")
    if not scorecard.empty:
        print(scorecard.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
