#!/usr/bin/env python3
"""
Point-in-time combined-score backtest for the Stock Analysis pipeline.

This harness tests the actual ranking stack:
Price Trend Score + Fair Value Score + regime-conditioned score weights.

SEC companyfacts are cached once per CIK, while the fair-value builder filters
individual facts by SEC filed date at each historical anchor. That means an
anchor dated 2023-08-31 can only see facts filed on or before 2023-08-31.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import comp
import historical_data_layer as historical_data
import portfolio_backtest_metrics as portfolio_metrics
import regime_threshold_backtest as regime_bt


DEFAULT_OUTPUT_DIR = Path("backtests/combined_score")
DEFAULT_HISTORICAL_UNIVERSE = Path("data/historical_tech_universe.csv")
HORIZONS = regime_bt.HORIZONS
BENCHMARK_TICKERS = regime_bt.BENCHMARK_TICKERS

ANCHOR_RESULTS_FILE = "anchor_results.csv"
POINT_IN_TIME_SCORES_FILE = "point_in_time_scores.csv"
TOP_HOLDINGS_FILE = "top_holdings_by_anchor.csv"
ANCHOR_DIAGNOSTICS_FILE = "anchor_diagnostics.csv"
HISTORICAL_UNIVERSE_SCORE_AUDIT_FILE = "historical_universe_score_audit.csv"
SUMMARY_FILE = "anchor_summary.csv"
FOLD_SUMMARY_FILE = "fold_summary.csv"
METADATA_FILE = "run_metadata.json"
PARAMS_FILE = "production_regime_parameters.json"
VALUATION_LOG_FILE = "valuation_warnings.log"


def production_regime_params() -> regime_bt.RegimeParams:
    current = regime_bt.RegimeParams(
        config_id="production_calibrated_combined",
        notes=(
            "Current production regime settings after expanded 5-year threshold "
            "calibration, used for combined-score fair-value backtesting."
        ),
    )
    return replace(
        current,
        rolling_window=5,
        structural_bull_hh50_min=0.30,
        early_hh20_min=0.30,
        bear_new_low20_min=0.40,
        stock_flow_core_bull_min=0.30,
        stock_flow_confirmer_bull_min=0.25,
    )


def finite_mean(values: Iterable[float]) -> float:
    arr = pd.Series(list(values), dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if not arr.empty else np.nan


def finite_median(values: Iterable[float]) -> float:
    arr = pd.Series(list(values), dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.median()) if not arr.empty else np.nan


def mode_or_neutral(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return "Neutral"
    mode = clean.mode()
    return str(mode.iloc[0]) if not mode.empty else "Neutral"


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def append_frame(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new.copy()
    if new.empty:
        return existing.copy()
    return pd.concat([existing, new], ignore_index=True, sort=False)


def normalize_universe_role(value: object) -> str:
    role = str(value).strip().lower()
    if role == "core":
        return "core"
    return "confirmer"


def prepare_current_universe() -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame, dict]:
    ticker_to_subindustry = regime_bt.build_ticker_to_subindustry()
    membership = regime_bt.build_membership_table()
    audit = membership.copy()
    audit["Universe_Source"] = "active_comp_regime_groups"
    audit["Universe_Inclusion"] = "included"
    audit["Historical_Universe_Status"] = "Active_Current_Model_Universe"
    metadata = {
        "historical_universe_source": "active_comp_regime_groups",
        "historical_universe_path": None,
        "historical_universe_rows": int(len(membership)),
        "historical_universe_unique_tickers": int(membership["Ticker"].nunique()),
        "historical_universe_note": "No external historical universe file was loaded.",
    }
    return ticker_to_subindustry, membership, audit, metadata


def augment_benchmark_groups_from_universe(universe: pd.DataFrame) -> int:
    if "Benchmark_Bucket" not in universe.columns:
        return 0

    additions = 0
    for _, row in universe.iterrows():
        bucket = str(row.get("Benchmark_Bucket", "")).strip()
        if not bucket:
            continue
        ticker = str(row.get("Ticker", "")).upper().strip()
        subindustry = comp.canonical_subindustry_name(row.get("SubIndustry"))
        if not ticker or not subindustry:
            continue

        groups = comp.TECH_BENCHMARK_GROUPS.setdefault(subindustry, {})
        bucket_tickers = groups.setdefault(bucket, [])
        if ticker not in bucket_tickers:
            bucket_tickers.append(ticker)
            additions += 1
    return additions


def prepare_historical_universe(
    path: Path,
    backtest_end_date: pd.Timestamp,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame, dict]:
    if not path.exists():
        return prepare_current_universe()

    raw = pd.read_csv(path)
    required = {"Ticker", "SubIndustry"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"Historical universe file is missing required columns: {missing}")

    universe = raw.copy()
    if "Universe_Inclusion" in universe.columns:
        universe = universe[universe["Universe_Inclusion"].fillna("included").astype(str).str.lower() == "included"].copy()
    universe["Ticker"] = universe["Ticker"].astype(str).str.upper().str.strip()
    universe["SubIndustry"] = universe["SubIndustry"].apply(comp.canonical_subindustry_name)
    universe = universe.dropna(subset=["Ticker", "SubIndustry"]).copy()
    universe = universe[universe["Ticker"] != ""].copy()
    universe["Role"] = universe.get("Universe_Role", "confirmer").apply(normalize_universe_role)
    universe["Membership_Start_Date"] = pd.to_datetime(
        universe.get("Membership_Start_Date", pd.NaT),
        errors="coerce",
    ).dt.normalize()
    universe["Membership_End_Date"] = pd.to_datetime(
        universe.get("Membership_End_Date", pd.NaT),
        errors="coerce",
    ).dt.normalize()
    universe["Membership_End_Date"] = universe["Membership_End_Date"].fillna(
        pd.to_datetime(backtest_end_date).normalize()
    )

    universe = (
        universe.sort_values(["Ticker", "Membership_Start_Date"])
        .drop_duplicates(["Ticker", "SubIndustry", "Role", "Membership_Start_Date", "Membership_End_Date"])
        .reset_index(drop=True)
    )
    ticker_to_subindustry = (
        universe.sort_values(["Ticker", "Membership_Start_Date"])
        .drop_duplicates("Ticker", keep="last")
        .set_index("Ticker")["SubIndustry"]
        .to_dict()
    )
    membership = universe[[
        "Ticker",
        "SubIndustry",
        "Role",
        "Membership_Start_Date",
        "Membership_End_Date",
    ]].copy()

    benchmark_additions = augment_benchmark_groups_from_universe(universe)
    audit_cols = [
        "Ticker",
        "SubIndustry",
        "Role",
        "Benchmark_Bucket",
        "Membership_Start_Date",
        "Membership_End_Date",
        "Status",
        "Membership_Source",
        "Source_Confidence",
        "Data_Limitation",
    ]
    audit = universe[[col for col in audit_cols if col in universe.columns]].copy()
    audit["Universe_Source"] = "historical_tech_universe"
    audit["Universe_Inclusion"] = "included"
    audit = audit.rename(columns={"Status": "Historical_Universe_Status"})

    metadata = {
        "historical_universe_source": "historical_tech_universe",
        "historical_universe_path": str(path),
        "historical_universe_rows": int(len(membership)),
        "historical_universe_unique_tickers": int(membership["Ticker"].nunique()),
        "historical_universe_benchmark_bucket_additions": int(benchmark_additions),
        "historical_universe_status_counts": (
            audit["Historical_Universe_Status"].value_counts(dropna=False).to_dict()
            if "Historical_Universe_Status" in audit.columns else {}
        ),
        "historical_universe_note": (
            "Point-in-time scores are filtered by historical universe membership windows. "
            "Historical universe quality depends on the supplied CSV."
        ),
    }
    return ticker_to_subindustry, membership, audit, metadata


def annotate_score_rows_with_universe(scores: pd.DataFrame, universe_audit: pd.DataFrame) -> pd.DataFrame:
    if scores.empty or universe_audit.empty:
        return scores
    annotation_cols = [
        "Ticker",
        "Historical_Universe_Status",
        "Membership_Source",
        "Source_Confidence",
        "Membership_Start_Date",
        "Membership_End_Date",
    ]
    available = [col for col in annotation_cols if col in universe_audit.columns]
    if "Ticker" not in available:
        return scores
    annotations = universe_audit[available].drop_duplicates("Ticker")
    return scores.merge(annotations, on="Ticker", how="left")


def price_history_bounds(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, frame in price_data.items():
        ticker = str(ticker).upper().strip()
        if frame is None or frame.empty:
            rows.append({
                "Ticker": ticker,
                "Has_Price_Data": False,
                "Price_Data_First_Date": pd.NaT,
                "Price_Data_Last_Date": pd.NaT,
                "Price_Data_Row_Count": 0,
            })
            continue

        dates = pd.to_datetime(frame.index, errors="coerce")
        dates = dates[~pd.isna(dates)]
        rows.append({
            "Ticker": ticker,
            "Has_Price_Data": len(dates) > 0,
            "Price_Data_First_Date": dates.min().normalize() if len(dates) else pd.NaT,
            "Price_Data_Last_Date": dates.max().normalize() if len(dates) else pd.NaT,
            "Price_Data_Row_Count": int(len(frame)),
        })
    return pd.DataFrame(rows)


def ticker_row_counts(frame: pd.DataFrame, count_col: str) -> pd.DataFrame:
    if frame.empty or "Ticker" not in frame.columns:
        return pd.DataFrame(columns=["Ticker", count_col])

    values = frame.copy()
    values["Ticker"] = values["Ticker"].astype(str).str.upper().str.strip()
    return values.groupby("Ticker", dropna=False).size().rename(count_col).reset_index()


def point_in_time_score_summary(point_in_time_scores: pd.DataFrame) -> pd.DataFrame:
    if point_in_time_scores.empty or "Ticker" not in point_in_time_scores.columns:
        return pd.DataFrame(columns=[
            "Ticker",
            "Point_In_Time_Score_Row_Count",
            "Latest_Point_In_Time_Score_Date",
        ])

    scores = point_in_time_scores.copy()
    scores["Ticker"] = scores["Ticker"].astype(str).str.upper().str.strip()
    if "Date" in scores.columns:
        scores["Date"] = pd.to_datetime(scores["Date"], errors="coerce").dt.normalize()
        summary = (
            scores.groupby("Ticker", dropna=False)
            .agg(
                Point_In_Time_Score_Row_Count=("Ticker", "size"),
                Latest_Point_In_Time_Score_Date=("Date", "max"),
            )
            .reset_index()
        )
        summary["Latest_Point_In_Time_Score_Date"] = summary["Latest_Point_In_Time_Score_Date"].dt.strftime("%Y-%m-%d")
        return summary

    return (
        scores.groupby("Ticker", dropna=False)
        .size()
        .rename("Point_In_Time_Score_Row_Count")
        .reset_index()
        .assign(Latest_Point_In_Time_Score_Date="")
    )


def score_availability_status(row: pd.Series) -> str:
    if int(row.get("Point_In_Time_Score_Row_Count", 0)) > 0:
        return "scored"
    if int(row.get("Membership_Filtered_Feature_Row_Count", 0)) > 0:
        return "features_available_but_no_score_rows"
    if int(row.get("Stock_Feature_Row_Count", 0)) > 0:
        return "price_features_outside_membership_window"
    if not bool(row.get("Has_Price_Data", False)):
        return "no_price_data_returned"
    return "price_data_without_usable_features"


def build_historical_universe_score_audit(
    universe_audit: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    stock_features: pd.DataFrame,
    scorable_features: pd.DataFrame,
    point_in_time_scores: pd.DataFrame,
) -> pd.DataFrame:
    if universe_audit.empty:
        return pd.DataFrame()

    audit = universe_audit.copy()
    audit["Ticker"] = audit["Ticker"].astype(str).str.upper().str.strip()
    audit = audit.merge(price_history_bounds(price_data), on="Ticker", how="left")
    audit = audit.merge(ticker_row_counts(stock_features, "Stock_Feature_Row_Count"), on="Ticker", how="left")
    audit = audit.merge(ticker_row_counts(scorable_features, "Membership_Filtered_Feature_Row_Count"), on="Ticker", how="left")
    audit = audit.merge(point_in_time_score_summary(point_in_time_scores), on="Ticker", how="left")

    count_cols = [
        "Price_Data_Row_Count",
        "Stock_Feature_Row_Count",
        "Membership_Filtered_Feature_Row_Count",
        "Point_In_Time_Score_Row_Count",
    ]
    for col in count_cols:
        audit[col] = audit[col].fillna(0).astype(int)

    audit["Has_Price_Data"] = audit["Has_Price_Data"].astype("boolean").fillna(False).astype(bool)
    date_cols = [
        "Price_Data_First_Date",
        "Price_Data_Last_Date",
        "Latest_Point_In_Time_Score_Date",
    ]
    for col in date_cols:
        if col in audit.columns:
            audit[col] = audit[col].fillna("")

    audit["Score_Availability_Status"] = audit.apply(score_availability_status, axis=1)
    return audit


def add_score_audit_metadata(metadata: dict, score_audit: pd.DataFrame) -> dict:
    enriched = dict(metadata)
    if score_audit.empty or "Score_Availability_Status" not in score_audit.columns:
        return enriched

    status_counts = score_audit["Score_Availability_Status"].value_counts(dropna=False).to_dict()
    enriched["historical_universe_score_audit_rows"] = int(len(score_audit))
    enriched["historical_universe_score_status_counts"] = status_counts
    enriched["historical_universe_scored_ticker_count"] = int(
        (score_audit["Score_Availability_Status"] == "scored").sum()
    )
    enriched["historical_universe_no_price_data_ticker_count"] = int(
        (score_audit["Score_Availability_Status"] == "no_price_data_returned").sum()
    )
    return enriched


@contextlib.contextmanager
def valuation_log_context(output_dir: Path, verbose: bool):
    if verbose:
        yield
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / VALUATION_LOG_FILE).open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            yield


def summarize_anchor_results(anchor_results: pd.DataFrame) -> pd.DataFrame:
    if anchor_results.empty:
        return pd.DataFrame()

    aggregations = {
        "Observations": ("Top_Avg_Return", "count"),
        "Top_Avg_Return": ("Top_Avg_Return", "mean"),
        "Top_Median_Return": ("Top_Median_Return", "mean"),
        "Universe_Avg_Return": ("Universe_Avg_Return", "mean"),
        "QQQ_Return": ("QQQ_Return", "mean"),
        "XLK_Return": ("XLK_Return", "mean"),
        "Top_Excess_vs_Universe": ("Top_Excess_vs_Universe", "mean"),
        "Top_Excess_vs_QQQ": ("Top_Excess_vs_QQQ", "mean"),
        "Top_Excess_vs_XLK": ("Top_Excess_vs_XLK", "mean"),
        "Hit_Rate_vs_Universe_Avg": ("Hit_Rate_vs_Universe_Avg", "mean"),
        "Avg_Ranked_Count": ("Ranked_Count", "mean"),
        "Avg_Top_Count_With_Return": ("Top_Count_With_Return", "mean"),
        "Avg_Fair_Value_Coverage": ("Fair_Value_Coverage", "mean"),
        "Avg_Top_Combined_Score": ("Top_Avg_Combined_Score", "mean"),
        "Avg_Top_Fair_Value_Score": ("Top_Avg_Fair_Value_Score", "mean"),
        "Avg_Top_Price_Trend_Score": ("Top_Avg_Price_Trend_Score_100", "mean"),
        "Avg_Top_Annualized_Volatility": ("Top_Annualized_Volatility", "mean"),
        "Avg_Top_Sharpe": ("Top_Sharpe", "mean"),
        "Avg_Top_Sortino": ("Top_Sortino", "mean"),
        "Avg_Top_Max_Drawdown": ("Top_Max_Drawdown", "mean"),
        "Avg_Top_Return_To_Drawdown": ("Top_Return_To_Drawdown", "mean"),
        "Avg_Top_Downside_Capture_vs_XLK": ("Top_Downside_Capture_vs_XLK", "mean"),
        "Avg_Top_Tracking_Error_vs_XLK": ("Top_Tracking_Error_vs_XLK", "mean"),
        "Avg_Top_Information_Ratio_vs_XLK": ("Top_Information_Ratio_vs_XLK", "mean"),
        "Avg_Bullish_Subindustry_Count": ("Bullish_Subindustry_Count", "mean"),
        "Avg_Bear_Subindustry_Count": ("Bear_Subindustry_Count", "mean"),
        "Avg_Neutral_Subindustry_Count": ("Neutral_Subindustry_Count", "mean"),
    }
    aggregations = {
        out_col: (in_col, agg_func)
        for out_col, (in_col, agg_func) in aggregations.items()
        if in_col in anchor_results.columns
    }
    summary = (
        anchor_results
        .groupby("Horizon", dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    return summary.round(6)


def summarize_fold_results(anchor_results: pd.DataFrame) -> pd.DataFrame:
    if anchor_results.empty:
        return pd.DataFrame()

    aggregations = {
        "Observations": ("Top_Avg_Return", "count"),
        "Top_Avg_Return": ("Top_Avg_Return", "mean"),
        "Universe_Avg_Return": ("Universe_Avg_Return", "mean"),
        "QQQ_Return": ("QQQ_Return", "mean"),
        "XLK_Return": ("XLK_Return", "mean"),
        "Top_Excess_vs_Universe": ("Top_Excess_vs_Universe", "mean"),
        "Top_Excess_vs_QQQ": ("Top_Excess_vs_QQQ", "mean"),
        "Top_Excess_vs_XLK": ("Top_Excess_vs_XLK", "mean"),
        "Hit_Rate_vs_Universe_Avg": ("Hit_Rate_vs_Universe_Avg", "mean"),
        "Avg_Fair_Value_Coverage": ("Fair_Value_Coverage", "mean"),
        "Avg_Top_Annualized_Volatility": ("Top_Annualized_Volatility", "mean"),
        "Avg_Top_Sharpe": ("Top_Sharpe", "mean"),
        "Avg_Top_Max_Drawdown": ("Top_Max_Drawdown", "mean"),
        "Avg_Top_Downside_Capture_vs_XLK": ("Top_Downside_Capture_vs_XLK", "mean"),
    }
    aggregations = {
        out_col: (in_col, agg_func)
        for out_col, (in_col, agg_func) in aggregations.items()
        if in_col in anchor_results.columns
    }
    summary = (
        anchor_results
        .groupby(["Fold", "Horizon"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    return summary.round(6)


def save_outputs(
    output_dir: Path,
    params: regime_bt.RegimeParams,
    metadata: dict,
    anchor_results: pd.DataFrame,
    point_in_time_scores: pd.DataFrame,
    top_holdings: pd.DataFrame,
    anchor_diagnostics: pd.DataFrame,
    historical_universe_audit: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_results.to_csv(output_dir / ANCHOR_RESULTS_FILE, index=False)
    point_in_time_scores.to_csv(output_dir / POINT_IN_TIME_SCORES_FILE, index=False)
    top_holdings.to_csv(output_dir / TOP_HOLDINGS_FILE, index=False)
    anchor_diagnostics.to_csv(output_dir / ANCHOR_DIAGNOSTICS_FILE, index=False)
    historical_universe_audit.to_csv(output_dir / HISTORICAL_UNIVERSE_SCORE_AUDIT_FILE, index=False)
    summarize_anchor_results(anchor_results).to_csv(output_dir / SUMMARY_FILE, index=False)
    summarize_fold_results(anchor_results).to_csv(output_dir / FOLD_SUMMARY_FILE, index=False)
    (output_dir / PARAMS_FILE).write_text(
        json.dumps(asdict(params), indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


def point_in_time_score_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "Date",
        "Ticker",
        "SubIndustry",
        "Combined_Score",
        "Fair_Value_Score",
        "Price_Trend_Score",
        "Stock_Price_Trend_Score",
        "SubIndustry_Price_Trend_Score",
        "Industry_Price_Trend_Score",
        "Industry_Regime",
        "SubIndustry_Regime",
        "Fair_Value_Data_Status",
        "Fair_Value_Metrics_Used",
        "Fair_Value_Benchmark_Bucket",
        "Fair_Value_Usable_Metrics",
        "Fair_Value_Missing_Reason",
        "Fair_Value_Missing_Metrics",
        "Share_Count_Source_Category",
        "Market_Price_USD",
        "Market_Price_Date",
        "Market_Data_Source",
        "Historical_Universe_Status",
        "Membership_Source",
        "Source_Confidence",
        "Membership_Start_Date",
        "Membership_End_Date",
        "Combined_Rank",
        "Tier",
    ]
    return [col for col in preferred if col in df.columns]


def build_anchor_scores(
    anchor: pd.Timestamp,
    ranked: pd.DataFrame,
    ticker_to_subindustry: dict[str, str],
    price_data: dict[str, pd.DataFrame],
    output_dir: Path,
    refresh_sec_cache: bool,
    verbose_valuation: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = ranked[ranked["Date"] == anchor].copy()
    day = day[np.isfinite(day["PTS"])]
    if day.empty:
        return pd.DataFrame(), pd.DataFrame()

    day_tickers = sorted(day["Ticker"].dropna().astype(str).str.upper().unique())
    market_data = comp.build_market_data_map(
        tickers=day_tickers,
        asof_date=anchor,
        price_data=price_data,
    )

    comp.reset_fair_value_issue_log()
    with valuation_log_context(output_dir, verbose_valuation):
        valuation_df = comp.build_valuation_dataframe_for_universe(
            tickers=day_tickers,
            asof_date=anchor,
            price_data=price_data,
            market_data=market_data,
            use_sec_cache=True,
            refresh_sec_cache=refresh_sec_cache,
        )

    centers = comp.build_benchmark_centers(
        valuation_df=valuation_df,
        valuation_metrics=comp.BASE_VALUATION_WEIGHTS,
    )

    subindustry_regime_map = (
        day.drop_duplicates("SubIndustry")
        .set_index("SubIndustry")["SubIndustry_Regime"]
        .to_dict()
    )
    industry_regime = mode_or_neutral(day["Industry_Regime"])
    industry_regime_by_subindustry = {
        sub: industry_regime
        for sub in set(ticker_to_subindustry.values())
    }
    fair_value_scores = comp.build_fair_value_scores_for_universe(
        valuation_df=valuation_df,
        centers=centers,
        ticker_to_subindustry={ticker: ticker_to_subindustry[ticker] for ticker in day_tickers if ticker in ticker_to_subindustry},
        industry_regime_by_subindustry=industry_regime_by_subindustry,
        subindustry_regime_map=subindustry_regime_map,
    )
    price_trend_scores = day.set_index("Ticker")["PTS"].to_dict()

    combined = comp.build_combined_score_dataframe(
        valuation_df=valuation_df,
        fair_value_scores=fair_value_scores,
        price_trend_scores=price_trend_scores,
        ticker_to_subindustry={ticker: ticker_to_subindustry[ticker] for ticker in day_tickers if ticker in ticker_to_subindustry},
        industry_regime=industry_regime,
        subindustry_regimes=subindustry_regime_map,
        centers=centers,
        asof_date=anchor,
        market_data=market_data,
    )
    combined = combined[combined["Ticker"].isin(day_tickers)].copy()
    combined["Industry_Regime"] = industry_regime
    combined["SubIndustry_Regime"] = combined["SubIndustry"].map(subindustry_regime_map).fillna("Neutral")

    if "Combined_Score" in combined.columns:
        combined = combined.sort_values("Combined_Score", ascending=False, na_position="last")
        combined["Combined_Rank"] = (
            combined["Combined_Score"]
            .rank(method="first", ascending=False, na_option="bottom")
            .astype(int)
        )

    return combined, valuation_df


def evaluate_anchor(
    anchor: pd.Timestamp,
    combined: pd.DataFrame,
    subindustry_history: pd.DataFrame,
    price_wide: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if combined.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ranked = combined[np.isfinite(combined["Combined_Score"])].copy()
    ranked = ranked.sort_values("Combined_Score", ascending=False)
    top = ranked.head(top_n).copy()
    end_date = price_wide.index.max()
    sub_day = subindustry_history[subindustry_history["Date"] == anchor].copy()

    fair_value_coverage = (
        float(np.isfinite(combined["Fair_Value_Score"]).mean())
        if "Fair_Value_Score" in combined.columns and not combined.empty else np.nan
    )
    data_status_counts = (
        combined["Fair_Value_Data_Status"].fillna("Missing").value_counts().to_dict()
        if "Fair_Value_Data_Status" in combined.columns else {}
    )

    anchor_rows = []
    for horizon_name, horizon_days in HORIZONS.items():
        future_date, returns = regime_bt.future_return_series(price_wide, anchor, horizon_days)
        if future_date is None or returns.empty:
            continue

        top_returns = returns.reindex(top["Ticker"]).dropna()
        ranked_returns = returns.reindex(ranked["Ticker"]).dropna()
        if top_returns.empty or ranked_returns.empty:
            continue

        qqq_return = regime_bt.benchmark_value(returns, "QQQ")
        xlk_return = regime_bt.benchmark_value(returns, "XLK")
        ranked_avg = float(ranked_returns.mean())
        top_avg = float(top_returns.mean())
        risk_metrics = portfolio_metrics.portfolio_risk_metrics(
            price_wide=price_wide,
            top_tickers=top["Ticker"],
            universe_tickers=ranked["Ticker"],
            anchor=anchor,
            future_date=future_date,
            benchmark_ticker="XLK",
        )
        top_cross_section = portfolio_metrics.cross_sectional_return_metrics(
            prefix="Top",
            returns=top_returns,
        )
        anchor_rows.append({
            "Fold": regime_bt.fold_label(anchor, end_date),
            "Anchor_Date": anchor.strftime("%Y-%m-%d"),
            "Future_Date": future_date.strftime("%Y-%m-%d"),
            "Horizon": horizon_name,
            "Top_N": top_n,
            "Ranked_Count": int(len(ranked)),
            "Top_Count_With_Return": int(top_returns.count()),
            "Ranked_Count_With_Return": int(ranked_returns.count()),
            "Top_Avg_Return": top_avg,
            "Top_Median_Return": float(top_returns.median()),
            "Universe_Avg_Return": ranked_avg,
            "Universe_Median_Return": float(ranked_returns.median()),
            "QQQ_Return": qqq_return,
            "XLK_Return": xlk_return,
            "Top_Excess_vs_Universe": top_avg - ranked_avg,
            "Top_Excess_vs_QQQ": top_avg - qqq_return if np.isfinite(qqq_return) else np.nan,
            "Top_Excess_vs_XLK": top_avg - xlk_return if np.isfinite(xlk_return) else np.nan,
            "Hit_Rate_vs_Universe_Avg": float((top_returns > ranked_avg).mean()),
            "Fair_Value_Coverage": fair_value_coverage,
            "Ranked_Avg_Combined_Score": finite_mean(ranked["Combined_Score"]),
            "Top_Avg_Combined_Score": finite_mean(top["Combined_Score"]),
            "Top_Avg_Fair_Value_Score": finite_mean(top["Fair_Value_Score"]),
            "Top_Avg_Price_Trend_Score_100": finite_mean(top["Stock_Price_Trend_Score"]),
            **risk_metrics,
            **top_cross_section,
            "Industry_Regime": mode_or_neutral(combined["Industry_Regime"]),
            "Bullish_Subindustry_Count": int(sub_day["SubIndustry_Regime"].isin(["Bull", "EarlyBull"]).sum()),
            "Bear_Subindustry_Count": int((sub_day["SubIndustry_Regime"] == "Bear").sum()),
            "Neutral_Subindustry_Count": int((sub_day["SubIndustry_Regime"] == "Neutral").sum()),
        })

    holdings = top.copy()
    holdings["Anchor_Date"] = anchor.strftime("%Y-%m-%d")
    holding_cols = [
        "Anchor_Date",
        "Combined_Rank",
        "Ticker",
        "SubIndustry",
        "Combined_Score",
        "Fair_Value_Score",
        "Stock_Price_Trend_Score",
        "Industry_Regime",
        "SubIndustry_Regime",
        "Fair_Value_Data_Status",
        "Fair_Value_Metrics_Used",
        "Share_Count_Source_Category",
        "Market_Price_USD",
        "Market_Price_Date",
    ]
    holdings = holdings[[col for col in holding_cols if col in holdings.columns]]

    diagnostics = pd.DataFrame([{
        "Anchor_Date": anchor.strftime("%Y-%m-%d"),
        "Ranked_Count": int(len(ranked)),
        "Scored_Universe_Count": int(len(combined)),
        "Fair_Value_Coverage": fair_value_coverage,
        "Fair_Value_OK_Count": int(data_status_counts.get("OK", 0)),
        "Fair_Value_No_Data_Count": int(len(combined) - data_status_counts.get("OK", 0)),
        "Fair_Value_Status_Counts_JSON": json.dumps(data_status_counts, sort_keys=True),
        "Industry_Regime": mode_or_neutral(combined["Industry_Regime"]),
        "Bullish_Subindustry_Count": int(sub_day["SubIndustry_Regime"].isin(["Bull", "EarlyBull"]).sum()),
        "Bear_Subindustry_Count": int((sub_day["SubIndustry_Regime"] == "Bear").sum()),
        "Neutral_Subindustry_Count": int((sub_day["SubIndustry_Regime"] == "Neutral").sum()),
    }])

    return pd.DataFrame(anchor_rows), holdings, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest point-in-time combined stock rankings.")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--anchor-frequency", choices=["monthly", "weekly", "quarterly"], default="monthly")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--historical-universe", type=Path, default=DEFAULT_HISTORICAL_UNIVERSE)
    parser.add_argument(
        "--historical-price-file",
        type=Path,
        help=(
            "Optional canonical daily price CSV from historical_data_layer.py. "
            "Use this for vendor-grade active/delisted historical prices instead of Yahoo."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--skip-anchors", type=int, default=0)
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--tickers-limit", type=int)
    parser.add_argument("--refresh-sec-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose-valuation", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = production_regime_params()
    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start_date = end_date - pd.DateOffset(years=args.years + 1)

    ticker_to_subindustry, membership, historical_universe_audit, universe_metadata = prepare_historical_universe(
        path=args.historical_universe,
        backtest_end_date=end_date,
    )
    if args.tickers_limit is not None:
        selected = sorted(ticker_to_subindustry)[:args.tickers_limit]
        ticker_to_subindustry = {ticker: ticker_to_subindustry[ticker] for ticker in selected}
        membership = membership[membership["Ticker"].isin(selected)].copy()
        historical_universe_audit = historical_universe_audit[historical_universe_audit["Ticker"].isin(selected)].copy()

    tickers = sorted(set(ticker_to_subindustry) | set(BENCHMARK_TICKERS))

    if args.historical_price_file:
        print(
            f"[INFO] Loading canonical historical prices for {len(tickers)} tickers "
            f"from {args.historical_price_file}",
            flush=True,
        )
        price_data = historical_data.load_canonical_prices(
            path=args.historical_price_file,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
        universe_metadata["historical_price_source"] = str(args.historical_price_file)
    else:
        print(
            f"[INFO] Fetching prices for {len(tickers)} tickers "
            f"from {start_date.date()} to {end_date.date()}",
            flush=True,
        )
        raw_price_data = comp.fetch_all_prices(
            tickers=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            batch_size=args.batch_size,
        )
        price_data = regime_bt.normalize_price_data(raw_price_data)
        universe_metadata["historical_price_source"] = "yfinance"
    price_wide = regime_bt.build_price_wide(price_data)
    if price_wide.empty:
        raise RuntimeError("No price data returned; cannot run combined-score backtest")

    stock_price_data = {
        ticker: df
        for ticker, df in price_data.items()
        if ticker in ticker_to_subindustry
    }
    stock_features = regime_bt.build_stock_feature_history(stock_price_data, ticker_to_subindustry)
    if stock_features.empty:
        raise RuntimeError("No stock features were generated")

    member_features = regime_bt.build_member_feature_history(stock_features, membership)
    scorable_features = (
        member_features
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )
    if scorable_features.empty:
        raise RuntimeError("No membership-filtered stock features were generated")

    anchor_dates = regime_bt.choose_anchor_dates(price_wide, args.years, args.anchor_frequency)
    anchor_dates = anchor_dates[args.skip_anchors:]
    if args.max_anchors is not None:
        anchor_dates = anchor_dates[:args.max_anchors]
    if not anchor_dates:
        raise RuntimeError("No anchor dates selected")

    print(f"[INFO] Historical universe source: {universe_metadata.get('historical_universe_source')}", flush=True)
    print(f"[INFO] Built {len(stock_features):,} stock-day feature rows", flush=True)
    print(f"[INFO] Built {len(scorable_features):,} membership-filtered stock-day feature rows", flush=True)
    print(f"[INFO] Selected {len(anchor_dates)} anchor dates", flush=True)
    ranked, subindustry_history = regime_bt.classify_regimes_for_params(
        stock_features=scorable_features,
        member_features=member_features,
        params=params,
        evaluation_dates=anchor_dates,
    )

    anchor_results = load_csv_if_exists(args.output_dir / ANCHOR_RESULTS_FILE) if args.resume else pd.DataFrame()
    point_in_time_scores = load_csv_if_exists(args.output_dir / POINT_IN_TIME_SCORES_FILE) if args.resume else pd.DataFrame()
    top_holdings = load_csv_if_exists(args.output_dir / TOP_HOLDINGS_FILE) if args.resume else pd.DataFrame()
    anchor_diagnostics = load_csv_if_exists(args.output_dir / ANCHOR_DIAGNOSTICS_FILE) if args.resume else pd.DataFrame()

    completed_anchors = set()
    if args.resume and not point_in_time_scores.empty and "Date" in point_in_time_scores.columns:
        completed_anchors = set(point_in_time_scores["Date"].dropna().astype(str).unique())

    metadata_base = {
        "status": "partial",
        "model": "Price Trend Score + Fair Value Score + Regime Weights",
        "years": args.years,
        "anchor_frequency": args.anchor_frequency,
        "top_n": args.top_n,
        "price_start_date": start_date.strftime("%Y-%m-%d"),
        "price_end_date": end_date.strftime("%Y-%m-%d"),
        "latest_price_date": price_wide.index.max().strftime("%Y-%m-%d"),
        "ticker_count_requested": len(tickers),
        "ticker_count_with_price_data": len(price_data),
        "stock_feature_rows": len(stock_features),
        "membership_filtered_stock_feature_rows": len(scorable_features),
        "anchor_count_selected": len(anchor_dates),
        "sec_cache_dir": str(comp.SEC_COMPANYFACTS_CACHE_DIR),
        "point_in_time_rule": "SEC facts are filtered to filed <= anchor date before fair-value scoring.",
        "regime_params": asdict(params),
        **universe_metadata,
    }

    processed_count = 0
    for idx, anchor in enumerate(anchor_dates, start=1):
        anchor_key = anchor.strftime("%Y-%m-%d")
        if anchor_key in completed_anchors:
            print(f"[INFO] Skipping completed anchor {idx}/{len(anchor_dates)}: {anchor_key}", flush=True)
            continue

        print(f"[INFO] Processing anchor {idx}/{len(anchor_dates)}: {anchor_key}", flush=True)
        combined, _valuation_df = build_anchor_scores(
            anchor=anchor,
            ranked=ranked,
            ticker_to_subindustry=ticker_to_subindustry,
            price_data=price_data,
            output_dir=args.output_dir,
            refresh_sec_cache=args.refresh_sec_cache,
            verbose_valuation=args.verbose_valuation,
        )
        if combined.empty:
            print(f"[WARN] No combined scores for {anchor_key}", flush=True)
            continue
        combined = annotate_score_rows_with_universe(combined, historical_universe_audit)

        new_anchor_results, new_top_holdings, new_diagnostics = evaluate_anchor(
            anchor=anchor,
            combined=combined,
            subindustry_history=subindustry_history,
            price_wide=price_wide,
            top_n=args.top_n,
        )

        score_cols = point_in_time_score_columns(combined)
        new_scores = combined[score_cols].copy()
        anchor_results = append_frame(anchor_results, new_anchor_results)
        point_in_time_scores = append_frame(point_in_time_scores, new_scores)
        top_holdings = append_frame(top_holdings, new_top_holdings)
        anchor_diagnostics = append_frame(anchor_diagnostics, new_diagnostics)
        processed_count += 1

        metadata = {
            **metadata_base,
            "status": "partial",
            "completed_anchor_count_this_run": processed_count,
            "last_completed_anchor": anchor_key,
            "anchor_results_observations": int(len(anchor_results)),
        }
        score_audit = build_historical_universe_score_audit(
            universe_audit=historical_universe_audit,
            price_data=price_data,
            stock_features=stock_features,
            scorable_features=scorable_features,
            point_in_time_scores=point_in_time_scores,
        )
        metadata = add_score_audit_metadata(metadata, score_audit)
        save_outputs(
            output_dir=args.output_dir,
            params=params,
            metadata=metadata,
            anchor_results=anchor_results,
            point_in_time_scores=point_in_time_scores,
            top_holdings=top_holdings,
            anchor_diagnostics=anchor_diagnostics,
            historical_universe_audit=score_audit,
        )
        print(
            f"[INFO] Saved checkpoint for {anchor_key}: "
            f"{len(combined)} scored rows, {len(new_anchor_results)} horizon observations",
            flush=True,
        )

    metadata = {
        **metadata_base,
        "status": "complete",
        "completed_anchor_count_this_run": processed_count,
        "anchor_results_observations": int(len(anchor_results)),
    }
    score_audit = build_historical_universe_score_audit(
        universe_audit=historical_universe_audit,
        price_data=price_data,
        stock_features=stock_features,
        scorable_features=scorable_features,
        point_in_time_scores=point_in_time_scores,
    )
    metadata = add_score_audit_metadata(metadata, score_audit)
    save_outputs(
        output_dir=args.output_dir,
        params=params,
        metadata=metadata,
        anchor_results=anchor_results,
        point_in_time_scores=point_in_time_scores,
        top_holdings=top_holdings,
        anchor_diagnostics=anchor_diagnostics,
        historical_universe_audit=score_audit,
    )

    summary = summarize_anchor_results(anchor_results)
    print("\n=== COMBINED SCORE BACKTEST SUMMARY ===", flush=True)
    if summary.empty:
        print("[WARN] No horizon observations were produced.", flush=True)
    else:
        cols = [
            "Horizon",
            "Observations",
            "Top_Avg_Return",
            "Universe_Avg_Return",
            "XLK_Return",
            "Top_Excess_vs_Universe",
            "Top_Excess_vs_XLK",
            "Hit_Rate_vs_Universe_Avg",
            "Avg_Fair_Value_Coverage",
        ]
        print(summary[[col for col in cols if col in summary.columns]].to_string(index=False), flush=True)

    print(f"\n[SUCCESS] Saved combined-score backtest outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
