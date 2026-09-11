from __future__ import annotations

import argparse
import io
import os
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import comp


DEFAULT_OUTPUT_DIR = Path("practice_tests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated 30-day practice test of the stock ranking pipeline."
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--evaluation-horizon", type=int, default=5)
    parser.add_argument("--price-lookback-days", type=int, default=260)
    parser.add_argument("--regime-warmup-days", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--sec-scrape-sleep", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


@contextmanager
def maybe_silence(enabled: bool):
    if enabled:
        with redirect_stdout(io.StringIO()):
            yield
    else:
        yield


@contextmanager
def cached_sec_companyfacts():
    original_get_json = comp._sec_get_json
    cache = {}

    def cached_get_json(url: str, *args, **kwargs):
        if url not in cache:
            cache[url] = original_get_json(url, *args, **kwargs)
        return cache[url]

    comp._sec_get_json = cached_get_json
    try:
        yield cache
    finally:
        comp._sec_get_json = original_get_json


def normalize_price_data(price_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    normalized = {}
    for ticker, df in price_data.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        out = df.copy()
        idx = pd.to_datetime(out.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        out.index = idx.normalize()
        out = out[~out.index.duplicated(keep="last")].sort_index()
        normalized[ticker] = out
    return normalized


def build_ticker_to_subindustry() -> dict[str, str]:
    mapping = {}
    for subindustry, group in comp.REGIME_GROUPS.items():
        subindustry = comp.canonical_subindustry_name(subindustry)
        for ticker in group.get("core", []):
            mapping[ticker] = subindustry
        for ticker in group.get("confirmers", []):
            mapping[ticker] = subindustry
    return mapping


def collect_trading_dates(
    price_data: dict[str, pd.DataFrame],
    min_coverage: float = 0.50
) -> pd.DatetimeIndex:
    counts = {}
    for df in price_data.values():
        for date in df.index.unique():
            date = pd.Timestamp(date).normalize()
            counts[date] = counts.get(date, 0) + 1

    min_count = max(3, int(max(len(price_data), 1) * min_coverage))
    dates = sorted(date for date, count in counts.items() if count >= min_count)
    return pd.DatetimeIndex(dates)


def select_test_dates(
    all_dates: pd.DatetimeIndex,
    days: int,
    evaluation_horizon: int,
    regime_warmup_days: int
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    if all_dates.empty:
        raise RuntimeError("No usable trading dates were found in price data")

    end_pos = len(all_dates) - 1 - max(evaluation_horizon, 0)
    if end_pos < 0:
        raise RuntimeError("Not enough trading dates for the requested evaluation horizon")

    start_pos = max(0, end_pos - days + 1)
    test_dates = list(all_dates[start_pos:end_pos + 1])
    if len(test_dates) < days:
        print(f"[WARN] Only {len(test_dates)} eligible test dates are available")

    regime_start_pos = max(0, start_pos - regime_warmup_days)
    regime_dates = list(all_dates[regime_start_pos:end_pos + 1])
    return test_dates, regime_dates


def add_regime_features(history_df: pd.DataFrame) -> pd.DataFrame:
    history_df = history_df.copy()
    history_df["Date"] = pd.to_datetime(history_df["Date"]).dt.normalize()
    history_df["SubIndustry"] = history_df["SubIndustry"].apply(comp.canonical_subindustry_name)
    history_df = (
        history_df
        .sort_values(["SubIndustry", "Date"])
        .drop_duplicates(["Date", "SubIndustry"], keep="last")
        .reset_index(drop=True)
    )

    roll = 5
    history_df["Pct_Above_SMA_20_5D"] = (
        history_df.groupby("SubIndustry")["Pct_Above_SMA_20"]
        .rolling(roll, min_periods=roll).mean()
        .reset_index(level=0, drop=True)
    )
    history_df["Pct_Above_SMA_50_5D"] = (
        history_df.groupby("SubIndustry")["Pct_Above_SMA_50"]
        .rolling(roll, min_periods=roll).mean()
        .reset_index(level=0, drop=True)
    )
    history_df["New_Low_Ratio_20D_5D"] = (
        history_df.groupby("SubIndustry")["New_Low_Ratio_20D"]
        .rolling(roll, min_periods=roll).mean()
        .reset_index(level=0, drop=True)
    )
    history_df["Pct_Higher_Highs_20D_5D"] = (
        history_df.groupby("SubIndustry")["Pct_Higher_Highs_20D"]
        .rolling(roll, min_periods=roll).mean()
        .reset_index(level=0, drop=True)
    )
    history_df["Pct_Higher_Highs_50D_5D"] = (
        history_df.groupby("SubIndustry")["Pct_Higher_Highs_50D"]
        .rolling(roll, min_periods=roll).mean()
        .reset_index(level=0, drop=True)
    )
    history_df["Slope_Median_Pct_From_SMA_20"] = (
        history_df.groupby("SubIndustry")["Median_Pct_From_SMA_20"].diff(roll)
    )

    history_df["Structural_Regime"] = history_df.apply(
        comp.classify_subindustry_regime,
        axis=1
    )
    history_df["Structural_Regime_Persist"] = (
        history_df.groupby("SubIndustry")["Structural_Regime"]
        .apply(comp.apply_regime_persistence)
        .reset_index(level=0, drop=True)
    )
    return history_df


def build_historical_regimes(
    price_data: dict[str, pd.DataFrame],
    regime_dates: list[pd.Timestamp],
    ticker_to_subindustry: dict[str, str],
    verbose: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, pd.DataFrame]]:
    snapshots = []
    daily_by_date = {}

    for idx, asof_date in enumerate(regime_dates, start=1):
        date_str = asof_date.strftime("%Y-%m-%d")
        print(f"[PRACTICE] Building regime snapshot {idx}/{len(regime_dates)} for {date_str}")

        for subindustry, group in comp.REGIME_GROUPS.items():
            subindustry = comp.canonical_subindustry_name(subindustry)
            with maybe_silence(not verbose):
                snap = comp.compute_subindustry_snapshot(
                    date_str=date_str,
                    subindustry_name=subindustry,
                    subindustry_group=group,
                    price_data=price_data
                )
            if snap is not None:
                snapshots.append(snap)

        daily = comp.build_daily_stock_pts(
            price_data=price_data,
            asof_date=asof_date,
            ticker_to_subindustry=ticker_to_subindustry
        )
        if not daily.empty:
            daily_by_date[asof_date] = daily

    if not snapshots:
        raise RuntimeError("No subindustry snapshots were built")

    history_df = add_regime_features(pd.DataFrame(snapshots))

    stock_flow_rows = []
    for asof_date, daily in daily_by_date.items():
        for subindustry in comp.REGIME_GROUPS.keys():
            canonical = comp.canonical_subindustry_name(subindustry)
            stock_flow_rows.append({
                "Date": asof_date,
                "SubIndustry": canonical,
                "StockFlow_Regime": comp.classify_subindustry_stock_flow(
                    subindustry_name=subindustry,
                    daily_stock_pts=daily
                )
            })

    flow_df = pd.DataFrame(stock_flow_rows)
    if not flow_df.empty:
        flow_df["Date"] = pd.to_datetime(flow_df["Date"]).dt.normalize()
        history_df = history_df.merge(
            flow_df,
            on=["Date", "SubIndustry"],
            how="left"
        )
    else:
        history_df["StockFlow_Regime"] = "Neutral"

    history_df["StockFlow_Regime"] = history_df["StockFlow_Regime"].fillna("Neutral")
    history_df["SubIndustry_Regime"] = history_df.apply(
        lambda row: comp.combine_subindustry_regimes(
            row["Structural_Regime_Persist"],
            row["StockFlow_Regime"]
        ),
        axis=1
    )

    industry_rows = []
    for asof_date in sorted(history_df["Date"].unique()):
        industry_rows.append(
            comp.compute_tech_industry_snapshot(
                history_df=history_df,
                date=pd.Timestamp(asof_date)
            )
        )

    industry_df = pd.DataFrame(industry_rows)
    industry_df["Date"] = pd.to_datetime(industry_df["Date"]).dt.normalize()
    industry_df = industry_df.sort_values(["Industry", "Date"]).reset_index(drop=True)
    industry_df["Industry_Regime_Persist"] = (
        industry_df.groupby("Industry")["Tech_Regime"]
        .apply(comp.apply_regime_persistence)
        .reset_index(level=0, drop=True)
    )

    return history_df, industry_df, daily_by_date


def attach_regimes_to_daily(
    daily: pd.DataFrame,
    asof_date: pd.Timestamp,
    history_df: pd.DataFrame,
    industry_df: pd.DataFrame
) -> pd.DataFrame:
    out = daily.copy()
    lookup = history_df.loc[
        history_df["Date"] == asof_date,
        [
            "SubIndustry",
            "SubIndustry_Regime",
            "Structural_Regime_Persist",
            "StockFlow_Regime",
        ]
    ].drop_duplicates("SubIndustry")

    out = out.merge(lookup, on="SubIndustry", how="left")
    out["SubIndustry_Regime"] = out["SubIndustry_Regime"].fillna("Neutral")
    out["Structural_Regime_Persist"] = out["Structural_Regime_Persist"].fillna("Neutral")
    out["StockFlow_Regime"] = out["StockFlow_Regime"].fillna("Neutral")

    industry_regime = industry_df.loc[
        industry_df["Date"] == asof_date,
        "Industry_Regime_Persist"
    ]
    out["Industry_Regime"] = (
        industry_regime.iloc[0]
        if not industry_regime.empty
        else "Neutral"
    )
    return out


def build_sec_item_cache_for_tickers(
    tickers: list[str],
    latest_asof_date: pd.Timestamp,
    verbose: bool
) -> tuple[dict[str, dict[str, list[dict]]], list[dict]]:
    cik_map = comp.load_sec_cik_map()
    latest_asof_date = comp.normalize_asof_date(latest_asof_date)
    cache = {}
    failures = []

    for idx, ticker in enumerate(tickers, start=1):
        if idx == 1 or idx % 25 == 0:
            print(f"[PRACTICE] SEC fact parse {idx}/{len(tickers)} through {latest_asof_date.date()}")

        cik = cik_map.get(ticker.upper())
        if not cik:
            failures.append({
                "Date": latest_asof_date.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Error": "No SEC CIK found",
            })
            cache[ticker] = {}
            continue

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"
        data = comp._sec_get_json(url)
        if not data:
            failures.append({
                "Date": latest_asof_date.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Error": "SEC companyfacts fetch failed",
            })
            cache[ticker] = {}
            continue

        facts_by_taxonomy = data.get("facts", {})
        metric_cache = {}
        for metric_name, possible_tags in comp.RELEVANT_LABELS_VALUATION.items():
            if not isinstance(metric_name, str):
                continue
            unit_preferences = comp.metric_unit_preferences(metric_name)
            metric_items = []
            for tag in possible_tags:
                if not isinstance(tag, str):
                    continue
                with maybe_silence(not verbose):
                    metric_items.extend(
                        comp.collect_sec_fact_items(
                            facts_by_taxonomy=facts_by_taxonomy,
                            tag=tag,
                            unit_preferences=unit_preferences,
                            asof_date=latest_asof_date
                        )
                    )
            if metric_items:
                metric_cache[metric_name] = metric_items
        cache[ticker] = metric_cache

    return cache, failures


def build_raw_from_sec_item_cache(
    ticker: str,
    asof_date: pd.Timestamp,
    sec_item_cache: dict[str, dict[str, list[dict]]],
    years_needed: int = 5
) -> dict:
    asof_date = comp.normalize_asof_date(asof_date)
    raw = {}

    for metric_name, items in sec_item_cache.get(ticker, {}).items():
        filtered_items = [
            item for item in items
            if item.get("_filed_ts") is not None
            and pd.notna(item.get("_filed_ts"))
            and item["_filed_ts"].normalize() <= asof_date
        ]
        if not filtered_items:
            continue

        series = comp.select_metric_series(
            metric_name=metric_name,
            items=filtered_items,
            years_needed=years_needed
        )
        if series:
            raw[metric_name] = comp.series_to_raw_payload(series)

    return raw


def build_practice_valuation_dataframe(
    tickers: list[str],
    asof_date: pd.Timestamp,
    price_data: dict[str, pd.DataFrame],
    market_data: dict[str, dict],
    sec_item_cache: dict[str, dict[str, list[dict]]],
    verbose: bool
) -> tuple[pd.DataFrame, list[dict]]:
    rows = {}
    failures = []

    for idx, ticker in enumerate(tickers, start=1):
        if idx == 1 or idx % 25 == 0:
            print(f"[PRACTICE] Valuation {idx}/{len(tickers)} for {asof_date.date()}")

        try:
            raw = build_raw_from_sec_item_cache(
                ticker=ticker,
                asof_date=asof_date,
                sec_item_cache=sec_item_cache
            )
            raw = comp.enrich_raw_with_market_data(
                raw=raw,
                ticker=ticker,
                asof_date=asof_date,
                market_snapshot=market_data.get(ticker, {})
            )
            raw = comp.apply_stock_split_adjustment(raw)

            normalized_raw = comp.ensure_optional_zero_metrics(
                comp.normalize_raw_sec_data(raw)
            )
            data = comp.run_calculated_equations(
                normalized_raw,
                comp.CALCULATED_EQUATIONS_VALUATION
            )
        except Exception as exc:
            failures.append({
                "Date": asof_date.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Error": str(exc),
            })
            continue

        for metric, year_map in data.items():
            if not isinstance(year_map, dict):
                continue
            valid_years = [
                y for y in year_map.keys()
                if isinstance(y, (int, np.integer))
                and comp.is_finite_number(year_map.get(y))
            ]
            if not valid_years:
                continue
            latest_year = max(valid_years)
            rows.setdefault(metric, {})[ticker] = float(year_map[latest_year])

    return pd.DataFrame(rows).T.sort_index(), failures


def realized_return(
    price_data: dict[str, pd.DataFrame],
    ticker: str,
    asof_date: pd.Timestamp,
    horizon: int
) -> float:
    df = price_data.get(ticker)
    if df is None or df.empty or "close" not in df.columns or asof_date not in df.index:
        return np.nan

    loc = df.index.get_loc(asof_date)
    if isinstance(loc, slice):
        loc = loc.start
    if not isinstance(loc, (int, np.integer)):
        return np.nan

    future_loc = int(loc) + horizon
    if future_loc >= len(df):
        return np.nan

    current = df["close"].iloc[int(loc)]
    future = df["close"].iloc[future_loc]
    if not comp.is_finite_number(current) or not comp.is_finite_number(future) or current <= 0:
        return np.nan
    return float(future / current - 1.0)


def build_combined_for_date(
    asof_date: pd.Timestamp,
    daily: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    ticker_to_subindustry: dict[str, str],
    sec_item_cache: dict[str, dict[str, list[dict]]],
    verbose: bool
) -> tuple[pd.DataFrame, list[dict]]:
    tickers = sorted(daily["Ticker"].unique())
    market_data = comp.build_market_data_map(
        tickers=tickers,
        asof_date=asof_date,
        price_data=price_data
    )

    valuation_df, failures = build_practice_valuation_dataframe(
        tickers=tickers,
        asof_date=asof_date,
        price_data=price_data,
        market_data=market_data,
        sec_item_cache=sec_item_cache,
        verbose=verbose
    )

    centers = comp.build_benchmark_centers(
        valuation_df=valuation_df,
        valuation_metrics=comp.BASE_VALUATION_WEIGHTS
    )

    industry_regime = (
        daily["Industry_Regime"].dropna().iloc[0]
        if "Industry_Regime" in daily.columns and not daily["Industry_Regime"].dropna().empty
        else "Neutral"
    )
    subindustry_regimes = (
        daily.groupby("SubIndustry")["SubIndustry_Regime"].first().to_dict()
        if "SubIndustry_Regime" in daily.columns
        else {}
    )

    fair_value_scores = comp.build_fair_value_scores(
        valuation_df=valuation_df,
        ticker_to_subindustry=ticker_to_subindustry,
        centers=centers,
        industry_regime_by_subindustry={
            sub: industry_regime for sub in subindustry_regimes.keys()
        },
        subindustry_regime_map=subindustry_regimes
    )

    combined = daily.copy()
    combined["Fair_Value_Score"] = combined["Ticker"].map(fair_value_scores)
    combined["Fair_Value_Metrics_Used"] = combined.apply(
        lambda row: comp.count_fair_value_metrics_available(
            ticker=row["Ticker"],
            subindustry=row["SubIndustry"],
            valuation_df=valuation_df,
            centers=centers
        ),
        axis=1
    )
    combined["Fair_Value_Data_Status"] = combined.apply(
        lambda row: comp.fair_value_data_status(
            ticker=row["Ticker"],
            subindustry=row["SubIndustry"],
            fair_value_score=row["Fair_Value_Score"],
            metrics_used=int(row["Fair_Value_Metrics_Used"])
        ),
        axis=1
    )
    combined["Market_Price_USD"] = combined["Ticker"].map(
        lambda ticker: market_data.get(ticker, {}).get("price", np.nan)
    )
    combined["Market_Price_Date"] = combined["Ticker"].map(
        lambda ticker: comp.format_market_price_date(market_data.get(ticker, {}))
    )
    combined["Market_Data_Source"] = combined["Ticker"].map(
        lambda ticker: market_data.get(ticker, {}).get("source", "missing")
    )
    combined["Price_Trend_Score_100"] = combined["PTS"].map(comp.price_trend_score_to_100)
    combined = comp.add_price_trend_context_columns(combined)
    combined["Combined_Score"] = combined.apply(
        lambda row: comp.compute_combined_score(
            trend_score=row["PTS"],
            benchmark_score=row["Fair_Value_Score"],
            industry_regime=row["Industry_Regime"],
            subindustry_regime=row["SubIndustry_Regime"]
        ),
        axis=1
    )
    combined["Next_1D_Return"] = combined["Ticker"].map(
        lambda ticker: realized_return(price_data, ticker, asof_date, 1)
    )
    combined["Next_5D_Return"] = combined["Ticker"].map(
        lambda ticker: realized_return(price_data, ticker, asof_date, 5)
    )
    combined = comp.add_tier_columns(combined)
    combined = comp.build_feature_table(combined, valuation_df=valuation_df)
    combined["Rank"] = combined["Overall_Rank"]
    return combined, failures


def rank_ic(group: pd.DataFrame, return_col: str) -> float:
    subset = group[["Combined_Score", return_col]].dropna()
    if len(subset) < 5:
        return np.nan
    if subset["Combined_Score"].nunique() < 2 or subset[return_col].nunique() < 2:
        return np.nan
    ranked = subset.rank(method="average")
    return float(ranked["Combined_Score"].corr(ranked[return_col]))


def build_daily_summary(all_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in all_scores.groupby("Date"):
        ranked = group.sort_values("Combined_Score", ascending=False, na_position="last")
        top10 = ranked.head(10)
        top10_next_1d = top10["Next_1D_Return"].mean()
        top10_next_5d = top10["Next_5D_Return"].mean()
        universe_next_1d = group["Next_1D_Return"].mean()
        universe_next_5d = group["Next_5D_Return"].mean()
        rows.append({
            "Date": date,
            "Ticker_Count": len(group),
            "Industry_Regime": group["Industry_Regime"].mode().iloc[0]
            if not group["Industry_Regime"].mode().empty else "Neutral",
            "Fair_Value_OK_Count": int((group["Fair_Value_Data_Status"] == "OK").sum()),
            "Fair_Value_OK_Pct": float((group["Fair_Value_Data_Status"] == "OK").mean()),
            "Trend_Only_Count": int(group["Fair_Value_Data_Status"].str.contains("Trend only", na=False).sum()),
            "Market_Missing_Count": int((group["Market_Data_Source"] == "missing").sum()),
            "Avg_Combined_Score": group["Combined_Score"].mean(),
            "Top10_Avg_Combined_Score": top10["Combined_Score"].mean(),
            "Top10_Tickers": ", ".join(top10["Ticker"].astype(str).tolist()),
            "Top10_Avg_Next_1D_Return": top10_next_1d,
            "Universe_Avg_Next_1D_Return": universe_next_1d,
            "Top10_Excess_1D_Return": top10_next_1d - universe_next_1d,
            "Top10_Avg_Next_5D_Return": top10_next_5d,
            "Universe_Avg_Next_5D_Return": universe_next_5d,
            "Top10_Excess_5D_Return": top10_next_5d - universe_next_5d,
            "Rank_IC_1D": rank_ic(group, "Next_1D_Return"),
            "Rank_IC_5D": rank_ic(group, "Next_5D_Return"),
        })
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def build_subindustry_summary(all_scores: pd.DataFrame) -> pd.DataFrame:
    return (
        all_scores
        .groupby(["Date", "SubIndustry"], as_index=False)
        .agg(
            Ticker_Count=("Ticker", "count"),
            Avg_Combined_Score=("Combined_Score", "mean"),
            Avg_Price_Trend_Score_100=("Price_Trend_Score_100", "mean"),
            Avg_Fair_Value_Score=("Fair_Value_Score", "mean"),
            Fair_Value_OK_Count=("Fair_Value_Data_Status", lambda s: int((s == "OK").sum())),
            Avg_Next_1D_Return=("Next_1D_Return", "mean"),
            Avg_Next_5D_Return=("Next_5D_Return", "mean"),
        )
        .sort_values(["Date", "Avg_Combined_Score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def write_outputs(
    all_scores: pd.DataFrame,
    valuation_failures: pd.DataFrame,
    output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_table = comp.build_feature_table(all_scores)
    daily_summary = build_daily_summary(all_scores)
    subindustry_summary = build_subindustry_summary(all_scores)
    latest_date = all_scores["Date"].max()
    latest_top = (
        all_scores[all_scores["Date"] == latest_date]
        .sort_values("Combined_Score", ascending=False, na_position="last")
        .head(25)
        .reset_index(drop=True)
    )
    latest_tier_list = (
        all_scores[all_scores["Date"] == latest_date]
        .sort_values("Combined_Score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    paths = {
        "all_scores": output_dir / "last_30_day_combined_scores.csv",
        "daily_summary": output_dir / "last_30_day_daily_summary.csv",
        "subindustry_summary": output_dir / "last_30_day_subindustry_summary.csv",
        "feature_table": output_dir / "last_30_day_feature_table.csv",
        "valuation_failures": output_dir / "last_30_day_valuation_failures.csv",
        "workbook": output_dir / "last_30_day_practice_run.xlsx",
    }

    all_scores.to_csv(paths["all_scores"], index=False)
    feature_table.to_csv(paths["feature_table"], index=False)
    daily_summary.to_csv(paths["daily_summary"], index=False)
    subindustry_summary.to_csv(paths["subindustry_summary"], index=False)
    valuation_failures.to_csv(paths["valuation_failures"], index=False)

    with pd.ExcelWriter(paths["workbook"], engine="xlsxwriter") as writer:
        daily_summary.to_excel(writer, sheet_name="Daily Summary", index=False)
        feature_table.to_excel(writer, sheet_name="Feature Table", index=False)
        latest_tier_list.to_excel(writer, sheet_name="Latest Tier List", index=False)
        latest_top.to_excel(writer, sheet_name="Latest Top 25", index=False)
        subindustry_summary.to_excel(writer, sheet_name="Subindustry Summary", index=False)
        valuation_failures.to_excel(writer, sheet_name="Valuation Failures", index=False)
        all_scores.to_excel(writer, sheet_name="All Scores", index=False)

    return paths


def main() -> int:
    args = parse_args()
    comp.EXPORT_RAW_SEC_DEBUG = False
    comp.SEC_SCRAPE_SLEEP_SEC = max(args.sec_scrape_sleep, 0.0)
    os.environ.setdefault("STOCK_ANALYSIS_AUTO_OPEN", "0")

    ticker_to_subindustry = build_ticker_to_subindustry()
    tickers = sorted(ticker_to_subindustry.keys())
    if args.max_tickers is not None:
        tickers = tickers[:args.max_tickers]
        ticker_to_subindustry = {
            ticker: sub for ticker, sub in ticker_to_subindustry.items()
            if ticker in tickers
        }

    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=args.price_lookback_days)).strftime("%Y-%m-%d")

    print(f"[PRACTICE] Fetching prices for {len(tickers)} tickers")
    price_data = comp.fetch_all_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        batch_size=args.batch_size
    )
    price_data = normalize_price_data(price_data)

    all_dates = collect_trading_dates(price_data)
    test_dates, regime_dates = select_test_dates(
        all_dates=all_dates,
        days=args.days,
        evaluation_horizon=args.evaluation_horizon,
        regime_warmup_days=args.regime_warmup_days
    )
    print(
        "[PRACTICE] Test window: "
        f"{test_dates[0].date()} to {test_dates[-1].date()} "
        f"({len(test_dates)} trading dates)"
    )

    history_df, industry_df, daily_by_date = build_historical_regimes(
        price_data=price_data,
        regime_dates=regime_dates,
        ticker_to_subindustry=ticker_to_subindustry,
        verbose=args.verbose
    )

    combined_frames = []
    failure_rows = []

    with cached_sec_companyfacts() as sec_cache:
        sec_item_cache, sec_failures = build_sec_item_cache_for_tickers(
            tickers=sorted({ticker for frame in daily_by_date.values() for ticker in frame["Ticker"].unique()}),
            latest_asof_date=test_dates[-1],
            verbose=args.verbose
        )
        failure_rows.extend(sec_failures)

        for idx, asof_date in enumerate(test_dates, start=1):
            print(f"[PRACTICE] Scoring test date {idx}/{len(test_dates)}: {asof_date.date()}")
            daily = daily_by_date.get(asof_date)
            if daily is None or daily.empty:
                print(f"[WARN] No stock PTS rows for {asof_date.date()}, skipping")
                continue

            daily = attach_regimes_to_daily(
                daily=daily,
                asof_date=asof_date,
                history_df=history_df,
                industry_df=industry_df
            )
            combined, failures = build_combined_for_date(
                asof_date=asof_date,
                daily=daily,
                price_data=price_data,
                ticker_to_subindustry=ticker_to_subindustry,
                sec_item_cache=sec_item_cache,
                verbose=args.verbose
            )
            combined_frames.append(combined)
            failure_rows.extend(failures)

        print(f"[PRACTICE] SEC companyfacts responses fetched: {len(sec_cache)}")

    if not combined_frames:
        raise RuntimeError("No combined score rows were created")

    all_scores = pd.concat(combined_frames, ignore_index=True, sort=False)
    valuation_failures = pd.DataFrame(failure_rows, columns=["Date", "Ticker", "Error"])
    paths = write_outputs(all_scores, valuation_failures, args.output_dir)

    daily_summary = build_daily_summary(all_scores)
    print("\n[PRACTICE] Summary tail")
    print(daily_summary.tail(5).to_string(index=False))
    print("\n[PRACTICE] Outputs")
    for label, path in paths.items():
        print(f"{label}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
