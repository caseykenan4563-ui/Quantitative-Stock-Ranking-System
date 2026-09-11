#!/usr/bin/env python3
"""
Portfolio-construction validation for the Stock Analysis ranking model.

This layer does not rebuild SEC fair-value scores and does not change the
production ranking formula. It uses the point-in-time combined-score history,
re-scores anchors under shortlisted weight configs, then tests how portfolio
size, rank weighting, turnover, and transaction costs affect realized returns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import combined_weight_tuning as weight_tuning
import historical_data_layer as historical_data
import portfolio_backtest_metrics as portfolio_metrics
import regime_threshold_backtest as regime_bt


DEFAULT_SCORE_HISTORY = Path("backtests/combined_score/point_in_time_scores.csv")
DEFAULT_WEIGHT_TUNING_DIR = Path("backtests/weight_tuning")
DEFAULT_OUTPUT_DIR = Path("backtests/portfolio_construction")
DEFAULT_TOP_N_LIST = "10,20,50,100"
DEFAULT_MODES = "equal,rank_weighted"
FOLD_ORDER = weight_tuning.FOLD_ORDER
HORIZONS = regime_bt.HORIZONS


def parse_top_n_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("top-n values must be positive integers")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one top-n value is required")
    return values


def parse_modes(raw: str) -> list[str]:
    valid = {"equal", "rank_weighted"}
    modes = []
    for item in raw.split(","):
        mode = item.strip()
        if not mode:
            continue
        if mode not in valid:
            raise ValueError(f"unsupported portfolio mode: {mode}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("at least one portfolio mode is required")
    return modes


def add_unique(items: list[str], value: object) -> None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return
    text = str(value).strip()
    if text and text not in items:
        items.append(text)


def load_shortlisted_config_ids(weight_tuning_dir: Path, shortlist_count: int) -> list[str]:
    selected: list[str] = []
    add_unique(selected, "current_live")

    scorecard_path = weight_tuning_dir / "candidate_scorecard.csv"
    if scorecard_path.exists():
        scorecard = pd.read_csv(scorecard_path)
        if "Risk_Adjusted_Objective" in scorecard.columns:
            scorecard = scorecard.sort_values("Risk_Adjusted_Objective", ascending=False)
        for config_id in scorecard.get("Config_ID", pd.Series(dtype="object")).head(shortlist_count):
            add_unique(selected, config_id)

    recommended_path = weight_tuning_dir / "recommended_weights.json"
    if recommended_path.exists():
        try:
            recommended = json.loads(recommended_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            recommended = {}
        add_unique(selected, recommended.get("best_full_sample", {}).get("Config_ID"))
        for config_id in recommended.get("walk_forward_selected_config_counts", {}):
            add_unique(selected, config_id)

    walk_forward_path = weight_tuning_dir / "walk_forward_selected_configs.csv"
    if walk_forward_path.exists():
        walk_forward = pd.read_csv(walk_forward_path)
        for config_id in walk_forward.get("Selected_Config_ID", pd.Series(dtype="object")).dropna():
            add_unique(selected, config_id)

    return selected


def select_configs(
    config_scope: str,
    preset: str,
    weight_tuning_dir: Path,
    shortlist_count: int,
    max_configs: int | None,
) -> list[weight_tuning.WeightConfig]:
    configs = weight_tuning.build_candidate_configs(preset)
    if config_scope == "shortlist":
        wanted = load_shortlisted_config_ids(weight_tuning_dir, shortlist_count)
        by_id = {config.config_id: config for config in configs}
        configs = [by_id[config_id] for config_id in wanted if config_id in by_id]
        if not configs:
            configs = weight_tuning.curated_configs()

    if max_configs is not None:
        configs = configs[:max_configs]
    return configs


def portfolio_weights(tickers: Iterable[str], mode: str) -> dict[str, float]:
    names = [str(ticker).upper() for ticker in tickers if pd.notna(ticker)]
    if not names:
        return {}

    if mode == "equal":
        weight = 1.0 / len(names)
        return {ticker: weight for ticker in names}

    if mode == "rank_weighted":
        raw = np.arange(len(names), 0, -1, dtype="float64")
        raw = raw / raw.sum()
        return {ticker: float(raw[idx]) for idx, ticker in enumerate(names)}

    raise ValueError(f"unsupported portfolio mode: {mode}")


def weighted_average_return(
    returns: pd.Series,
    weights: dict[str, float],
) -> tuple[float, int]:
    if not weights:
        return np.nan, 0

    clean_returns = (
        pd.to_numeric(returns.reindex(weights.keys()), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if clean_returns.empty:
        return np.nan, 0

    clean_weights = (
        pd.Series(weights, dtype="float64")
        .reindex(clean_returns.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    total_weight = clean_weights.sum()
    if total_weight <= 0:
        return np.nan, 0

    clean_weights = clean_weights / total_weight
    return float((clean_returns * clean_weights).sum()), int(clean_returns.count())


def weighted_average_column(
    frame: pd.DataFrame,
    column: str,
    weights: dict[str, float],
) -> float:
    if column not in frame.columns or not weights:
        return np.nan
    values = (
        pd.to_numeric(frame.set_index("Ticker")[column].reindex(weights.keys()), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if values.empty:
        return np.nan
    aligned_weights = (
        pd.Series(weights, dtype="float64")
        .reindex(values.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    if aligned_weights.sum() <= 0:
        return np.nan
    aligned_weights = aligned_weights / aligned_weights.sum()
    return float((values * aligned_weights).sum())


def turnover(
    current_weights: dict[str, float],
    previous_weights: dict[str, float] | None,
    initial_turnover: float,
) -> tuple[float, bool]:
    if previous_weights is None:
        return float(initial_turnover), True
    tickers = set(current_weights) | set(previous_weights)
    value = 0.5 * sum(
        abs(current_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0))
        for ticker in tickers
    )
    return float(value), False


def regime_counts(day: pd.DataFrame) -> dict:
    if "SubIndustry" not in day.columns or "SubIndustry_Regime" not in day.columns:
        return {
            "Bullish_Subindustry_Count": 0,
            "Bear_Subindustry_Count": 0,
            "Neutral_Subindustry_Count": 0,
        }
    subindustry_regimes = day.groupby("SubIndustry")["SubIndustry_Regime"].agg(weight_tuning.mode_or_neutral)
    return {
        "Bullish_Subindustry_Count": int(subindustry_regimes.isin(["Bull", "EarlyBull"]).sum()),
        "Bear_Subindustry_Count": int((subindustry_regimes == "Bear").sum()),
        "Neutral_Subindustry_Count": int((subindustry_regimes == "Neutral").sum()),
    }


def evaluate_config_portfolios(
    scores: pd.DataFrame,
    config: weight_tuning.WeightConfig,
    price_wide: pd.DataFrame,
    top_n_list: list[int],
    modes: list[str],
    transaction_cost_bps: float,
    initial_turnover: float,
    include_risk_metrics: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = weight_tuning.config_score_frame(scores, config)
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()

    end_date = price_wide.index.max()
    cost_rate = float(transaction_cost_bps) / 10000.0
    previous_by_portfolio: dict[tuple[int, str], dict[str, float] | None] = {}
    result_rows: list[dict] = []
    latest_holding_rows: list[dict] = []
    latest_anchor = pd.to_datetime(scored["Date"].max()).normalize()

    for anchor, day in scored.groupby("Date", sort=True):
        anchor = pd.to_datetime(anchor).normalize()
        ranked = day.sort_values("Tuned_Combined_Score", ascending=False).copy()
        if ranked.empty:
            continue

        counts = regime_counts(ranked)
        industry_regime = weight_tuning.mode_or_neutral(ranked["Industry_Regime"])

        for top_n in top_n_list:
            top = ranked.head(top_n).copy()
            if top.empty:
                continue

            for mode in modes:
                weights = portfolio_weights(top["Ticker"], mode)
                key = (top_n, mode)
                rebalance_turnover, is_initial = turnover(
                    current_weights=weights,
                    previous_weights=previous_by_portfolio.get(key),
                    initial_turnover=initial_turnover,
                )
                previous_by_portfolio[key] = weights
                transaction_cost = rebalance_turnover * cost_rate

                if anchor == latest_anchor:
                    latest = top.copy()
                    latest["Config_ID"] = config.config_id
                    latest["Anchor_Date"] = anchor.strftime("%Y-%m-%d")
                    latest["Top_N"] = top_n
                    latest["Portfolio_Mode"] = mode
                    latest["Portfolio_Rank"] = np.arange(1, len(latest) + 1)
                    latest["Portfolio_Weight"] = latest["Ticker"].map(weights)
                    latest_holding_rows.extend(latest[
                        [
                            col for col in [
                                "Config_ID",
                                "Anchor_Date",
                                "Top_N",
                                "Portfolio_Mode",
                                "Portfolio_Rank",
                                "Portfolio_Weight",
                                "Ticker",
                                "SubIndustry",
                                "Tuned_Combined_Score",
                                "Tuned_Fair_Value_Score",
                                "Trend_Score_100",
                                "Trend_Weight",
                                "Fair_Value_Weight",
                                "Industry_Regime",
                                "SubIndustry_Regime",
                            ]
                            if col in latest.columns
                        ]
                    ].to_dict("records"))

                for horizon_name, horizon_days in HORIZONS.items():
                    future_date, returns = regime_bt.future_return_series(price_wide, anchor, horizon_days)
                    if future_date is None or returns.empty:
                        continue

                    top_gross_return, top_count = weighted_average_return(returns, weights)
                    universe_returns = returns.reindex(ranked["Ticker"]).dropna()
                    if not np.isfinite(top_gross_return) or universe_returns.empty:
                        continue

                    universe_avg = float(universe_returns.mean())
                    qqq_return = regime_bt.benchmark_value(returns, "QQQ")
                    xlk_return = regime_bt.benchmark_value(returns, "XLK")
                    top_net_return = top_gross_return - transaction_cost
                    top_return_coverage = top_count / len(top) if len(top) else np.nan

                    risk_metrics = {}
                    if include_risk_metrics:
                        risk_metrics = portfolio_metrics.portfolio_risk_metrics(
                            price_wide=price_wide,
                            top_tickers=top["Ticker"],
                            universe_tickers=ranked["Ticker"],
                            anchor=anchor,
                            future_date=future_date,
                            benchmark_ticker="XLK",
                            top_weights=weights,
                        )
                        if np.isfinite(risk_metrics.get("Top_Path_Cumulative_Return", np.nan)):
                            risk_metrics["Top_Path_Cumulative_Return_Net"] = (
                                risk_metrics["Top_Path_Cumulative_Return"] - transaction_cost
                            )

                    top_forward_returns = returns.reindex(top["Ticker"]).dropna()
                    result_rows.append({
                        "Config_ID": config.config_id,
                        "Fold": regime_bt.fold_label(anchor, end_date),
                        "Anchor_Date": anchor.strftime("%Y-%m-%d"),
                        "Future_Date": future_date.strftime("%Y-%m-%d"),
                        "Horizon": horizon_name,
                        "Top_N": top_n,
                        "Portfolio_Mode": mode,
                        "Ranked_Count": int(len(ranked)),
                        "Top_Count": int(len(top)),
                        "Top_Count_With_Return": int(top_count),
                        "Top_Return_Coverage": float(top_return_coverage),
                        "Ranked_Count_With_Return": int(universe_returns.count()),
                        "Initial_Portfolio": bool(is_initial),
                        "Turnover": rebalance_turnover,
                        "Transaction_Cost_Bps": transaction_cost_bps,
                        "Transaction_Cost": transaction_cost,
                        "Top_Gross_Return": top_gross_return,
                        "Top_Net_Return": top_net_return,
                        "Top_Median_Return": float(top_forward_returns.median()) if not top_forward_returns.empty else np.nan,
                        "Universe_Avg_Return": universe_avg,
                        "Universe_Median_Return": float(universe_returns.median()),
                        "QQQ_Return": qqq_return,
                        "XLK_Return": xlk_return,
                        "Top_Gross_Excess_vs_Universe": top_gross_return - universe_avg,
                        "Top_Net_Excess_vs_Universe": top_net_return - universe_avg,
                        "Top_Gross_Excess_vs_QQQ": top_gross_return - qqq_return if np.isfinite(qqq_return) else np.nan,
                        "Top_Net_Excess_vs_QQQ": top_net_return - qqq_return if np.isfinite(qqq_return) else np.nan,
                        "Top_Gross_Excess_vs_XLK": top_gross_return - xlk_return if np.isfinite(xlk_return) else np.nan,
                        "Top_Net_Excess_vs_XLK": top_net_return - xlk_return if np.isfinite(xlk_return) else np.nan,
                        "Net_Hit_Rate_vs_Universe_Avg": float((top_forward_returns - transaction_cost > universe_avg).mean()) if not top_forward_returns.empty else np.nan,
                        "Top_Avg_Tuned_Combined_Score": portfolio_metrics.finite_mean(top["Tuned_Combined_Score"]),
                        "Top_Weighted_Avg_Tuned_Combined_Score": weighted_average_column(top, "Tuned_Combined_Score", weights),
                        "Top_Avg_Fair_Value_Score": portfolio_metrics.finite_mean(top["Tuned_Fair_Value_Score"]),
                        "Top_Weighted_Avg_Fair_Value_Score": weighted_average_column(top, "Tuned_Fair_Value_Score", weights),
                        "Top_Avg_Price_Trend_Score": portfolio_metrics.finite_mean(top["Trend_Score_100"]),
                        "Top_Weighted_Avg_Price_Trend_Score": weighted_average_column(top, "Trend_Score_100", weights),
                        "Top_Avg_Trend_Weight": portfolio_metrics.finite_mean(top["Trend_Weight"]),
                        "Top_Weighted_Avg_Trend_Weight": weighted_average_column(top, "Trend_Weight", weights),
                        "Industry_Regime": industry_regime,
                        **counts,
                        **risk_metrics,
                        **portfolio_metrics.cross_sectional_return_metrics("Top", top_forward_returns),
                    })

    return pd.DataFrame(result_rows), pd.DataFrame(latest_holding_rows)


def aggregate_results(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    aggregations = {
        "Observations": ("Top_Net_Return", "count"),
        "Top_Gross_Return": ("Top_Gross_Return", "mean"),
        "Top_Net_Return": ("Top_Net_Return", "mean"),
        "Universe_Avg_Return": ("Universe_Avg_Return", "mean"),
        "QQQ_Return": ("QQQ_Return", "mean"),
        "XLK_Return": ("XLK_Return", "mean"),
        "Top_Gross_Excess_vs_Universe": ("Top_Gross_Excess_vs_Universe", "mean"),
        "Top_Net_Excess_vs_Universe": ("Top_Net_Excess_vs_Universe", "mean"),
        "Top_Gross_Excess_vs_QQQ": ("Top_Gross_Excess_vs_QQQ", "mean"),
        "Top_Net_Excess_vs_QQQ": ("Top_Net_Excess_vs_QQQ", "mean"),
        "Top_Gross_Excess_vs_XLK": ("Top_Gross_Excess_vs_XLK", "mean"),
        "Top_Net_Excess_vs_XLK": ("Top_Net_Excess_vs_XLK", "mean"),
        "Net_Hit_Rate_vs_Universe_Avg": ("Net_Hit_Rate_vs_Universe_Avg", "mean"),
        "Avg_Turnover": ("Turnover", "mean"),
        "Avg_Transaction_Cost": ("Transaction_Cost", "mean"),
        "Avg_Top_Return_Coverage": ("Top_Return_Coverage", "mean"),
        "Avg_Ranked_Count": ("Ranked_Count", "mean"),
        "Avg_Top_Trend_Weight": ("Top_Avg_Trend_Weight", "mean"),
        "Avg_Top_Weighted_Trend_Weight": ("Top_Weighted_Avg_Trend_Weight", "mean"),
        "Avg_Top_Annualized_Volatility": ("Top_Annualized_Volatility", "mean"),
        "Avg_Top_Sharpe": ("Top_Sharpe", "mean"),
        "Avg_Top_Sortino": ("Top_Sortino", "mean"),
        "Avg_Top_Max_Drawdown": ("Top_Max_Drawdown", "mean"),
        "Avg_Top_Return_To_Drawdown": ("Top_Return_To_Drawdown", "mean"),
        "Avg_Top_Downside_Capture_vs_XLK": ("Top_Downside_Capture_vs_XLK", "mean"),
        "Avg_Top_Upside_Capture_vs_XLK": ("Top_Upside_Capture_vs_XLK", "mean"),
        "Avg_Top_Tracking_Error_vs_XLK": ("Top_Tracking_Error_vs_XLK", "mean"),
        "Avg_Top_Information_Ratio_vs_XLK": ("Top_Information_Ratio_vs_XLK", "mean"),
        "Avg_Top_Loss_Rate": ("Top_Loss_Rate", "mean"),
        "Avg_Top_CrossSection_Return_Std": ("Top_CrossSection_Return_Std", "mean"),
    }
    aggregations = {
        out_col: (in_col, agg_func)
        for out_col, (in_col, agg_func) in aggregations.items()
        if in_col in results.columns
    }

    return (
        results
        .groupby(group_cols, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .round(6)
    )


def add_objective_columns(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return scorecard

    downside_capture = scorecard.get("Avg_Top_Downside_Capture_vs_XLK", pd.Series(np.nan, index=scorecard.index))
    downside_penalty = (downside_capture - 1.0).clip(lower=0.0).fillna(0.0)
    drawdown_penalty = scorecard.get("Avg_Top_Max_Drawdown", pd.Series(np.nan, index=scorecard.index)).abs().fillna(0.0)
    turnover_penalty = scorecard.get("Avg_Turnover", pd.Series(0.0, index=scorecard.index)).fillna(0.0)

    scorecard["Net_Return_Objective"] = (
        scorecard["Top_Net_Excess_vs_Universe"].fillna(0.0)
        + 0.50 * scorecard["Top_Net_Excess_vs_XLK"].fillna(0.0)
    )
    scorecard["Portfolio_Objective"] = (
        scorecard["Net_Return_Objective"]
        + 0.010 * scorecard.get("Avg_Top_Sharpe", pd.Series(0.0, index=scorecard.index)).fillna(0.0)
        + 0.005 * scorecard.get("Avg_Top_Information_Ratio_vs_XLK", pd.Series(0.0, index=scorecard.index)).fillna(0.0)
        - 0.050 * downside_penalty
        - 0.020 * drawdown_penalty
        - 0.001 * turnover_penalty
    )
    return scorecard


def score_portfolios(results: pd.DataFrame) -> pd.DataFrame:
    scorecard = aggregate_results(results, ["Config_ID", "Top_N", "Portfolio_Mode"])
    if scorecard.empty:
        return scorecard

    scorecard = add_objective_columns(scorecard)
    return scorecard.sort_values("Portfolio_Objective", ascending=False).reset_index(drop=True)


def score_overall_results(results: pd.DataFrame, label: str) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    summary = aggregate_results(results.assign(Portfolio_Set=label), ["Portfolio_Set"])
    if summary.empty:
        return summary
    return add_objective_columns(summary).round(6)


def filter_selected(results: pd.DataFrame, selected: pd.Series | dict) -> pd.DataFrame:
    config_id = selected["Config_ID"]
    top_n = int(selected["Top_N"])
    mode = selected["Portfolio_Mode"]
    return results[
        (results["Config_ID"] == config_id)
        & (results["Top_N"] == top_n)
        & (results["Portfolio_Mode"] == mode)
    ].copy()


def walk_forward_selection(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()

    selection_rows: list[dict] = []
    validation_frames: list[pd.DataFrame] = []
    available_folds = [fold for fold in FOLD_ORDER if fold in set(results["Fold"])]

    for validation_idx in range(1, len(available_folds)):
        validation_fold = available_folds[validation_idx]
        train_folds = available_folds[:validation_idx]
        train_results = results[results["Fold"].isin(train_folds)].copy()
        validation_results = results[results["Fold"] == validation_fold].copy()
        if train_results.empty or validation_results.empty:
            continue

        train_scorecard = score_portfolios(train_results)
        if train_scorecard.empty:
            continue

        selected = train_scorecard.iloc[0]
        selected_validation = filter_selected(validation_results, selected)
        validation_scorecard = score_portfolios(selected_validation)
        validation_summary = validation_scorecard.iloc[0].to_dict() if not validation_scorecard.empty else {}

        round_number = len(selection_rows) + 1
        selected_validation["Walk_Forward_Round"] = round_number
        selected_validation["Train_Folds"] = ",".join(train_folds)
        validation_frames.append(selected_validation)

        selection_rows.append({
            "Walk_Forward_Round": round_number,
            "Train_Folds": ",".join(train_folds),
            "Validation_Fold": validation_fold,
            "Selected_Config_ID": selected["Config_ID"],
            "Selected_Top_N": int(selected["Top_N"]),
            "Selected_Portfolio_Mode": selected["Portfolio_Mode"],
            "Train_Portfolio_Objective": float(selected["Portfolio_Objective"]),
            "Train_Net_Return_Objective": float(selected["Net_Return_Objective"]),
            "Validation_Portfolio_Objective": validation_summary.get("Portfolio_Objective", np.nan),
            "Validation_Net_Return_Objective": validation_summary.get("Net_Return_Objective", np.nan),
            "Validation_Top_Net_Excess_vs_Universe": validation_summary.get("Top_Net_Excess_vs_Universe", np.nan),
            "Validation_Top_Net_Excess_vs_XLK": validation_summary.get("Top_Net_Excess_vs_XLK", np.nan),
            "Validation_Avg_Top_Sharpe": validation_summary.get("Avg_Top_Sharpe", np.nan),
            "Validation_Avg_Top_Max_Drawdown": validation_summary.get("Avg_Top_Max_Drawdown", np.nan),
            "Validation_Avg_Turnover": validation_summary.get("Avg_Turnover", np.nan),
        })

    validation = (
        pd.concat(validation_frames, ignore_index=True, sort=False)
        if validation_frames else pd.DataFrame()
    )
    return pd.DataFrame(selection_rows).round(6), validation


def format_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n.a."
    if not np.isfinite(number):
        return "n.a."
    return f"{number:.2%}"


def format_num(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n.a."
    if not np.isfinite(number):
        return "n.a."
    return f"{number:.{digits}f}"


def construction_label(row: dict | pd.Series) -> str:
    return f"`{row.get('Config_ID')}` / top {int(row.get('Top_N'))} / `{row.get('Portfolio_Mode')}`"


def write_summary(
    output_dir: Path,
    scorecard: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    walk_forward: pd.DataFrame,
    walk_forward_overall: pd.DataFrame,
    current_live_scorecard: pd.DataFrame,
    metadata: dict,
) -> None:
    top = scorecard.iloc[0].to_dict() if not scorecard.empty else {}
    current_top = current_live_scorecard.iloc[0].to_dict() if not current_live_scorecard.empty else {}
    selections = (
        walk_forward[["Selected_Config_ID", "Selected_Top_N", "Selected_Portfolio_Mode"]]
        .value_counts()
        .reset_index(name="Count")
        if not walk_forward.empty else pd.DataFrame()
    )

    lines = [
        "# Portfolio Construction Validation Summary",
        "",
        "## What was completed",
        "",
        "This run tested the portfolio construction layer that sits after the stock-ranking algorithm.",
        "It did not rebuild SEC fair-value scores and did not modify production score weights in `comp.py`.",
        "",
        "The tested construction choices were:",
        "",
        f"- Portfolio sizes: {metadata.get('top_n_list')}",
        f"- Portfolio modes: {metadata.get('portfolio_modes')}",
        f"- Transaction cost assumption: {metadata.get('transaction_cost_bps')} bps per 100% turnover",
        f"- Initial anchor turnover charged: {metadata.get('initial_turnover')}",
        "",
        "Equal weight gives every selected stock the same allocation. Rank weight gives the highest-ranked stock the largest allocation and tapers linearly through the selected list.",
        "",
        "Transaction cost is estimated as `turnover * transaction_cost_bps / 10000` and is subtracted from each forward-return observation.",
        "Turnover is measured as one half of the absolute change in portfolio weights from the prior monthly anchor.",
        "",
        "## Run configuration",
        "",
        f"- Score rows loaded: {metadata.get('score_rows')}",
        f"- Anchor dates loaded: {metadata.get('anchor_count')}",
        f"- Candidate weight configs tested: {metadata.get('candidate_config_count')}",
        f"- Result observations: {metadata.get('result_rows')}",
        f"- Latest price date: {metadata.get('latest_price_date')}",
        "",
        "## Best full-sample construction",
        "",
    ]

    if top:
        lines.extend([
            f"- Construction: {construction_label(top)}",
            f"- Portfolio objective: {format_num(top.get('Portfolio_Objective'), 6)}",
            f"- Net return objective: {format_num(top.get('Net_Return_Objective'), 6)}",
            f"- Avg net return: {format_pct(top.get('Top_Net_Return'))}",
            f"- Net excess vs universe: {format_pct(top.get('Top_Net_Excess_vs_Universe'))}",
            f"- Net excess vs XLK: {format_pct(top.get('Top_Net_Excess_vs_XLK'))}",
            f"- Avg turnover: {format_pct(top.get('Avg_Turnover'))}",
            f"- Avg Sharpe: {format_num(top.get('Avg_Top_Sharpe'))}",
            f"- Avg max drawdown: {format_pct(top.get('Avg_Top_Max_Drawdown'))}",
            "",
        ])
    else:
        lines.extend(["No portfolio scorecard was produced.", ""])

    lines.extend(["## Current-live construction check", ""])
    if current_top:
        lines.extend([
            f"- Best construction using current production weights: {construction_label(current_top)}",
            f"- Portfolio objective: {format_num(current_top.get('Portfolio_Objective'), 6)}",
            f"- Avg net return: {format_pct(current_top.get('Top_Net_Return'))}",
            f"- Net excess vs universe: {format_pct(current_top.get('Top_Net_Excess_vs_Universe'))}",
            f"- Net excess vs XLK: {format_pct(current_top.get('Top_Net_Excess_vs_XLK'))}",
            f"- Avg turnover: {format_pct(current_top.get('Avg_Turnover'))}",
            f"- Avg Sharpe: {format_num(current_top.get('Avg_Top_Sharpe'))}",
            "",
        ])
    else:
        lines.extend(["The current live config was not found in the portfolio scorecard.", ""])

    lines.extend(["## Top full-sample candidates", ""])
    if not scorecard.empty:
        for _, row in scorecard.head(10).iterrows():
            lines.append(
                "- "
                f"{construction_label(row)}: objective {format_num(row.get('Portfolio_Objective'), 6)}, "
                f"net excess vs universe {format_pct(row.get('Top_Net_Excess_vs_Universe'))}, "
                f"net excess vs XLK {format_pct(row.get('Top_Net_Excess_vs_XLK'))}, "
                f"turnover {format_pct(row.get('Avg_Turnover'))}"
            )
        lines.append("")
    else:
        lines.extend(["No candidates to summarize.", ""])

    lines.extend(["## Walk-forward validation", ""])
    if not selections.empty:
        lines.append("Selected construction counts:")
        for _, row in selections.iterrows():
            lines.append(
                f"- `{row['Selected_Config_ID']}` / top {int(row['Selected_Top_N'])} / "
                f"`{row['Selected_Portfolio_Mode']}`: {int(row['Count'])}"
            )
        lines.append("")

    if not walk_forward_overall.empty:
        wf = walk_forward_overall.iloc[0].to_dict()
        lines.extend([
            "Aggregate validation result of constructions selected only from older folds:",
            f"- Observations: {wf.get('Observations')}",
            f"- Avg net return: {format_pct(wf.get('Top_Net_Return'))}",
            f"- Avg universe return: {format_pct(wf.get('Universe_Avg_Return'))}",
            f"- Avg XLK return: {format_pct(wf.get('XLK_Return'))}",
            f"- Net excess vs universe: {format_pct(wf.get('Top_Net_Excess_vs_Universe'))}",
            f"- Net excess vs XLK: {format_pct(wf.get('Top_Net_Excess_vs_XLK'))}",
            f"- Avg Sharpe: {format_num(wf.get('Avg_Top_Sharpe'))}",
            f"- Avg max drawdown: {format_pct(wf.get('Avg_Top_Max_Drawdown'))}",
            f"- Avg turnover: {format_pct(wf.get('Avg_Turnover'))}",
            "",
        ])
    else:
        lines.extend(["Walk-forward validation did not produce enough fold data.", ""])

    lines.extend([
        "## Interpretation",
        "",
        "This is a portfolio-construction test, not a new alpha claim. It answers how concentrated the final list should be and whether higher ranks deserve larger weights after transaction costs.",
        "",
        "Use the walk-forward section as the production gate. The full-sample winner is useful research evidence, but the walk-forward winner is closer to the real question: would a construction rule chosen from older data work on the next unseen year?",
        "",
        "If top-10 or rank-weighted portfolios win only in the full sample but fail in walk-forward validation, they are likely too concentrated for the current signal quality. If top-50 or top-100 equal weight is steadier, the ranking model may be better as a broad screener than as a highly concentrated portfolio selector until fair-value weights are improved.",
        "",
        "## Files produced",
        "",
        "- `portfolio_results.csv`",
        "- `portfolio_scorecard.csv`",
        "- `portfolio_summary_by_horizon.csv`",
        "- `portfolio_walk_forward_selected.csv`",
        "- `portfolio_walk_forward_validation_results.csv`",
        "- `portfolio_walk_forward_summary.csv`",
        "- `latest_portfolio_holdings.csv`",
        "- `recommended_portfolio_construction.json`",
        "- `run_metadata.json`",
        "- `PORTFOLIO_CONSTRUCTION_SUMMARY.md`",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "PORTFOLIO_CONSTRUCTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate portfolio construction choices for stock rankings.")
    parser.add_argument("--score-history", type=Path, default=DEFAULT_SCORE_HISTORY)
    parser.add_argument("--weight-tuning-dir", type=Path, default=DEFAULT_WEIGHT_TUNING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--historical-price-file",
        type=Path,
        help=(
            "Optional canonical daily price CSV from historical_data_layer.py. "
            "Use this for Sharadar active/delisted prices instead of Yahoo."
        ),
    )
    parser.add_argument("--candidate-preset", choices=["compact", "expanded"], default="expanded")
    parser.add_argument("--config-scope", choices=["shortlist", "all"], default="shortlist")
    parser.add_argument("--shortlist-count", type=int, default=10)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--top-n-list", default=DEFAULT_TOP_N_LIST)
    parser.add_argument("--portfolio-modes", default=DEFAULT_MODES)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--initial-turnover", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--skip-risk-metrics", action="store_true")
    args = parser.parse_args()

    top_n_list = parse_top_n_list(args.top_n_list)
    portfolio_modes = parse_modes(args.portfolio_modes)
    configs = select_configs(
        config_scope=args.config_scope,
        preset=args.candidate_preset,
        weight_tuning_dir=args.weight_tuning_dir,
        shortlist_count=args.shortlist_count,
        max_configs=args.max_configs,
    )
    if not configs:
        raise RuntimeError("No candidate configs selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = weight_tuning.prepare_score_history(args.score_history)

    print(f"[INFO] Loaded {len(scores):,} point-in-time score rows", flush=True)
    print(f"[INFO] Testing {len(configs)} configs, top-N {top_n_list}, modes {portfolio_modes}", flush=True)
    if args.historical_price_file:
        price_wide = historical_data.load_canonical_price_wide(
            path=args.historical_price_file,
            tickers=sorted(set(scores["Ticker"]) | set(regime_bt.BENCHMARK_TICKERS)),
            start_date=scores["Date"].min() - pd.Timedelta(days=5),
            end_date=pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
            value_column="Close",
        )
        missing = sorted((set(scores["Ticker"]) | set(regime_bt.BENCHMARK_TICKERS)) - set(price_wide.columns))
        if missing:
            print(
                f"[WARN] Canonical price file is missing {len(missing)} requested tickers",
                flush=True,
            )
            print(f"Sample missing tickers: {missing[:20]}", flush=True)
        if price_wide.empty:
            raise RuntimeError("No canonical price data returned; cannot validate portfolio construction")
    else:
        price_wide = weight_tuning.fetch_price_wide(scores, args.batch_size)
    print(f"[INFO] Price matrix: {len(price_wide):,} dates x {len(price_wide.columns):,} tickers", flush=True)

    all_results = []
    all_latest_holdings = []
    for idx, config in enumerate(configs, start=1):
        print(f"[INFO] Evaluating portfolio construction {idx}/{len(configs)}: {config.config_id}", flush=True)
        results, latest_holdings = evaluate_config_portfolios(
            scores=scores,
            config=config,
            price_wide=price_wide,
            top_n_list=top_n_list,
            modes=portfolio_modes,
            transaction_cost_bps=args.transaction_cost_bps,
            initial_turnover=args.initial_turnover,
            include_risk_metrics=not args.skip_risk_metrics,
        )
        all_results.append(results)
        all_latest_holdings.append(latest_holdings)

    results = pd.concat(all_results, ignore_index=True, sort=False) if all_results else pd.DataFrame()
    latest_holdings = (
        pd.concat(all_latest_holdings, ignore_index=True, sort=False)
        if all_latest_holdings else pd.DataFrame()
    )
    scorecard = score_portfolios(results)
    horizon_summary = aggregate_results(results, ["Config_ID", "Top_N", "Portfolio_Mode", "Horizon"])
    walk_forward, validation_results = walk_forward_selection(results)
    walk_forward_overall = score_overall_results(validation_results, "walk_forward_selected")
    current_live_scorecard = (
        scorecard[scorecard["Config_ID"] == "current_live"].copy()
        if not scorecard.empty else pd.DataFrame()
    )

    metadata = {
        "status": "complete",
        "score_history": str(args.score_history),
        "weight_tuning_dir": str(args.weight_tuning_dir),
        "historical_price_file": str(args.historical_price_file) if args.historical_price_file else None,
        "market_data_source": str(args.historical_price_file) if args.historical_price_file else "yfinance",
        "candidate_preset": args.candidate_preset,
        "config_scope": args.config_scope,
        "shortlist_count": args.shortlist_count,
        "candidate_config_count": int(len(configs)),
        "top_n_list": top_n_list,
        "portfolio_modes": portfolio_modes,
        "transaction_cost_bps": args.transaction_cost_bps,
        "initial_turnover": args.initial_turnover,
        "score_rows": int(len(scores)),
        "anchor_count": int(scores["Date"].nunique()),
        "result_rows": int(len(results)),
        "latest_price_date": price_wide.index.max().strftime("%Y-%m-%d"),
        "risk_metrics_included": not args.skip_risk_metrics,
        "objective_note": (
            "Portfolio_Objective = net excess vs universe + 0.5 * net excess vs XLK "
            "+ 0.010 * Sharpe + 0.005 * information ratio - downside, drawdown, and turnover penalties."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(config) for config in configs]).to_csv(args.output_dir / "tested_weight_configs.csv", index=False)
    results.to_csv(args.output_dir / "portfolio_results.csv", index=False)
    scorecard.to_csv(args.output_dir / "portfolio_scorecard.csv", index=False)
    horizon_summary.to_csv(args.output_dir / "portfolio_summary_by_horizon.csv", index=False)
    walk_forward.to_csv(args.output_dir / "portfolio_walk_forward_selected.csv", index=False)
    validation_results.to_csv(args.output_dir / "portfolio_walk_forward_validation_results.csv", index=False)
    walk_forward_overall.to_csv(args.output_dir / "portfolio_walk_forward_summary.csv", index=False)
    latest_holdings.to_csv(args.output_dir / "latest_portfolio_holdings.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    recommended = {
        "best_full_sample": scorecard.iloc[0].to_dict() if not scorecard.empty else {},
        "best_current_live_construction": current_live_scorecard.iloc[0].to_dict() if not current_live_scorecard.empty else {},
        "walk_forward_overall": walk_forward_overall.iloc[0].to_dict() if not walk_forward_overall.empty else {},
        "walk_forward_selected_construction_counts": (
            walk_forward[["Selected_Config_ID", "Selected_Top_N", "Selected_Portfolio_Mode"]]
            .value_counts()
            .reset_index(name="Count")
            .to_dict("records")
            if not walk_forward.empty else []
        ),
        "tested_portfolio_sizes": top_n_list,
        "tested_portfolio_modes": portfolio_modes,
        "transaction_cost_bps": args.transaction_cost_bps,
        "candidate_parameters": {config.config_id: asdict(config) for config in configs},
        "recommendation_note": (
            "Use walk-forward validation before changing production portfolio concentration or weighting rules."
        ),
    }
    (args.output_dir / "recommended_portfolio_construction.json").write_text(
        json.dumps(recommended, indent=2, default=str),
        encoding="utf-8",
    )

    write_summary(
        output_dir=args.output_dir,
        scorecard=scorecard,
        horizon_summary=horizon_summary,
        walk_forward=walk_forward,
        walk_forward_overall=walk_forward_overall,
        current_live_scorecard=current_live_scorecard,
        metadata=metadata,
    )

    print("\n=== PORTFOLIO CONSTRUCTION SCORECARD ===", flush=True)
    columns = [
        "Config_ID",
        "Top_N",
        "Portfolio_Mode",
        "Portfolio_Objective",
        "Net_Return_Objective",
        "Top_Net_Return",
        "Top_Net_Excess_vs_Universe",
        "Top_Net_Excess_vs_XLK",
        "Avg_Turnover",
        "Avg_Top_Sharpe",
        "Avg_Top_Max_Drawdown",
    ]
    if not scorecard.empty:
        print(scorecard[[col for col in columns if col in scorecard.columns]].head(15).to_string(index=False), flush=True)

    print("\n=== WALK-FORWARD PORTFOLIO SELECTIONS ===", flush=True)
    if walk_forward.empty:
        print("[WARN] No walk-forward selections were produced.", flush=True)
    else:
        print(walk_forward.to_string(index=False), flush=True)

    print(f"\n[SUCCESS] Saved portfolio construction outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
