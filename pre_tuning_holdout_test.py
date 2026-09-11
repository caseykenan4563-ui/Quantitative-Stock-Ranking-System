#!/usr/bin/env python3
"""Run a pre-tuning historical holdout test for the current default model.

The current production-facing research default was selected from the modern
Sharadar-backed validation work. This script runs a deliberately earlier window
using the frozen current construction:

- Alpha_Prototype_XLKCompetitive
- top 10
- rank_weighted
- $30M minimum 20-day dollar volume

Important limitation: the local historical universe was originally seeded from
the current model universe plus a small stale/delisted recovery list. For this
holdout, membership starts are backfilled from vendor price availability so we
can test signal behavior before 2021. That is useful, but it is not the same as
a complete 2019 technology universe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import historical_data_layer as historical_data
import monthly_rebalanced_portfolio_simulator as monthly_sim
import combined_weight_tuning as weight_tuning


DEFAULT_OUTPUT_DIR = Path("backtests/pre_tuning_holdout_2019_2021")
DEFAULT_BASE_UNIVERSE = Path("data/historical_tech_universe.csv")
DEFAULT_SECURITY_MASTER = Path("data/historical_data_layer/security_master.csv")
DEFAULT_PRICE_FILE = Path("data/historical_data_layer/prices_daily.csv")
DEFAULT_HOLDOUT_START = "2019-09-30"
DEFAULT_HOLDOUT_END = "2021-08-31"
DEFAULT_SCORE_YEARS = 7
DEFAULT_MAX_ANCHORS = 24
PRODUCTION_ALPHA_COLUMN = "Alpha_Prototype_XLKCompetitive"
PRODUCTION_CONFIG_ID = "alpha_Alpha_Prototype_XLKCompetitive"
CONSTRUCTION_VARIANTS = [
    {"rebalance_frequency": "monthly", "months_between_rebalances": 1, "top_n": 10, "mode": "rank_weighted"},
    {"rebalance_frequency": "quarterly", "months_between_rebalances": 3, "top_n": 10, "mode": "rank_weighted"},
    {"rebalance_frequency": "semiannual", "months_between_rebalances": 6, "top_n": 10, "mode": "rank_weighted"},
    {"rebalance_frequency": "quarterly", "months_between_rebalances": 3, "top_n": 20, "mode": "rank_weighted"},
    {"rebalance_frequency": "semiannual", "months_between_rebalances": 6, "top_n": 20, "mode": "rank_weighted"},
    {"rebalance_frequency": "quarterly", "months_between_rebalances": 3, "top_n": 50, "mode": "rank_weighted"},
    {"rebalance_frequency": "quarterly", "months_between_rebalances": 3, "top_n": 10, "mode": "equal"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-tuning holdout validation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-universe", type=Path, default=DEFAULT_BASE_UNIVERSE)
    parser.add_argument("--security-master", type=Path, default=DEFAULT_SECURITY_MASTER)
    parser.add_argument("--historical-price-file", type=Path, default=DEFAULT_PRICE_FILE)
    parser.add_argument("--holdout-start", default=DEFAULT_HOLDOUT_START)
    parser.add_argument("--holdout-end", default=DEFAULT_HOLDOUT_END)
    parser.add_argument("--score-years", type=int, default=DEFAULT_SCORE_YEARS)
    parser.add_argument("--max-anchors", type=int, default=DEFAULT_MAX_ANCHORS)
    parser.add_argument("--resume-scores", action="store_true")
    parser.add_argument("--skip-score-rebuild", action="store_true")
    return parser.parse_args()


def fmt_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number * 100:,.2f}%"


def fmt_num(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}"


def build_holdout_universe(
    base_universe_path: Path,
    security_master_path: Path,
    output_path: Path,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> pd.DataFrame:
    if not base_universe_path.exists():
        raise FileNotFoundError(f"Base universe not found: {base_universe_path}")
    if not security_master_path.exists():
        raise FileNotFoundError(f"Security master not found: {security_master_path}")

    base = pd.read_csv(base_universe_path)
    security = pd.read_csv(security_master_path)
    base["Ticker"] = base["Ticker"].astype(str).str.upper().str.strip()
    security["Ticker"] = security["Ticker"].astype(str).str.upper().str.strip()

    security_cols = [
        "Ticker",
        "First_Price_Date",
        "Last_Price_Date",
        "Company_Name",
        "Sector",
        "Industry",
        "SubIndustry",
        "Is_Delisted",
        "Source_Quality",
    ]
    security = security[[col for col in security_cols if col in security.columns]].drop_duplicates("Ticker")
    merged = base.merge(security, on="Ticker", how="left", suffixes=("", "_Security"))
    merged["First_Price_Date"] = pd.to_datetime(merged["First_Price_Date"], errors="coerce").dt.normalize()
    merged["Last_Price_Date"] = pd.to_datetime(merged["Last_Price_Date"], errors="coerce").dt.normalize()

    eligible = merged[
        (merged["First_Price_Date"].notna())
        & (merged["First_Price_Date"] <= holdout_end)
        & (merged["Last_Price_Date"].fillna(holdout_end) >= holdout_start)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No eligible holdout tickers found after price-date filtering")

    eligible["Membership_Start_Date"] = eligible["First_Price_Date"].map(
        lambda value: max(pd.to_datetime(value).normalize(), holdout_start)
    )
    eligible["Membership_End_Date"] = eligible["Last_Price_Date"].fillna(holdout_end).map(
        lambda value: min(pd.to_datetime(value).normalize(), holdout_end)
    )
    eligible = eligible[eligible["Membership_Start_Date"] <= eligible["Membership_End_Date"]].copy()

    is_delisted = (
        eligible["Is_Delisted"].map(
            lambda value: False if pd.isna(value) else str(value).strip().lower() in {"true", "1", "yes"}
        )
        if "Is_Delisted" in eligible.columns
        else pd.Series(False, index=eligible.index)
    )
    eligible["Status"] = np.where(
        is_delisted,
        "Holdout_Backfilled_Price_Window_Delisted",
        "Holdout_Backfilled_Price_Window",
    )
    eligible["Membership_Source"] = "holdout_backfilled_from_vendor_price_bounds"
    eligible["Source_Confidence"] = "medium"
    eligible["Data_Limitation"] = (
        "Pre-tuning holdout uses model-seeded historical universe with membership starts "
        "backfilled from vendor price availability; not a complete historical tech universe."
    )
    eligible["Universe_Inclusion"] = "included"

    preferred_cols = [
        "Ticker",
        "Industry",
        "SubIndustry",
        "Universe_Role",
        "Benchmark_Bucket",
        "Membership_Start_Date",
        "Membership_End_Date",
        "Status",
        "Is_Current_Model_Ticker",
        "Membership_Source",
        "Universe_Inclusion",
        "Source_Confidence",
        "First_Score_Date",
        "Last_Score_Date",
        "First_Price_Date",
        "Last_Price_Date",
        "Price_Row_Count",
        "Alternate_SubIndustries",
        "Removal_Reason",
        "Source_Note",
        "Data_Limitation",
    ]
    for col in preferred_cols:
        if col not in eligible.columns:
            eligible[col] = ""
    output = eligible[preferred_cols].copy()
    for col in ["Membership_Start_Date", "Membership_End_Date", "First_Price_Date", "Last_Price_Date"]:
        output[col] = pd.to_datetime(output[col], errors="coerce").dt.strftime("%Y-%m-%d")
    output = output.sort_values(["SubIndustry", "Ticker"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def run_score_rebuild(
    output_dir: Path,
    holdout_universe: Path,
    historical_price_file: Path,
    years: int,
    max_anchors: int,
    resume: bool,
) -> None:
    cmd = [
        sys.executable,
        "combined_score_backtest.py",
        "--years",
        str(years),
        "--anchor-frequency",
        "monthly",
        "--top-n",
        "100",
        "--output-dir",
        str(output_dir),
        "--historical-universe",
        str(holdout_universe),
        "--historical-price-file",
        str(historical_price_file),
        "--max-anchors",
        str(max_anchors),
    ]
    if resume:
        cmd.append("--resume")
    print("[INFO] Running holdout score rebuild", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def construction_config_id(base_config_id: str, frequency: str, top_n: int, mode: str) -> str:
    return f"{base_config_id}__{frequency}__top{top_n}__{mode}"


def select_rebalance_anchors(scored: pd.DataFrame, months_between_rebalances: int) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    months_between_rebalances = max(int(months_between_rebalances), 1)
    out = scored.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out = out[out["Date"].notna()].copy()
    if months_between_rebalances <= 1:
        return out

    kept_dates: list[pd.Timestamp] = []
    last_kept: pd.Timestamp | None = None
    for anchor in sorted(out["Date"].dropna().unique()):
        anchor = pd.to_datetime(anchor).normalize()
        if last_kept is None:
            kept_dates.append(anchor)
            last_kept = anchor
            continue
        month_delta = (anchor.year - last_kept.year) * 12 + (anchor.month - last_kept.month)
        if month_delta >= months_between_rebalances:
            kept_dates.append(anchor)
            last_kept = anchor

    return out[out["Date"].isin(kept_dates)].copy()


def run_frozen_construction_simulations(
    score_history: Path,
    holdout_universe: Path,
    historical_price_file: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = weight_tuning.prepare_score_history(score_history)
    tickers = sorted(set(scores["Ticker"]) | set(monthly_sim.BENCHMARK_TICKERS))
    start_date = scores["Date"].min() - pd.Timedelta(days=260)
    # The combined-score holdout keeps 12-month forward-return diagnostics, but
    # the compounded monthly portfolio should stop near the final holdout anchor
    # instead of passively carrying the last basket for another year.
    end_date = scores["Date"].max() + pd.Timedelta(days=45)
    price_wide = historical_data.load_canonical_price_wide(
        path=historical_price_file,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        value_column="Close",
    )
    volume_wide = historical_data.load_canonical_price_wide(
        path=historical_price_file,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        value_column="Volume",
    )
    if price_wide.empty:
        raise RuntimeError("No holdout market prices loaded")

    config = monthly_sim.DirectScoreConfig(
        config_id=PRODUCTION_CONFIG_ID,
        score_column=PRODUCTION_ALPHA_COLUMN,
        notes="Frozen production-facing alpha family used for pre-tuning holdout validation.",
    )
    scores, configs = monthly_sim.add_alpha_score_configs(
        scores=scores,
        configs=[],
        score_columns=[PRODUCTION_ALPHA_COLUMN],
        price_wide=price_wide,
    )
    config = next(config for config in configs if config.config_id == PRODUCTION_CONFIG_ID)
    scored = monthly_sim.score_frame_for_config(scores, config)
    params = monthly_sim.SimulationParams(min_dollar_volume=30_000_000.0)
    adv20 = monthly_sim.rolling_dollar_volume(price_wide, volume_wide, params.adv_window)
    historical_universe = monthly_sim.load_historical_universe(holdout_universe, scores, price_wide)

    daily_frames: list[pd.DataFrame] = []
    rebalance_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for variant in CONSTRUCTION_VARIANTS:
        frequency = str(variant["rebalance_frequency"])
        top_n = int(variant["top_n"])
        mode = str(variant["mode"])
        months_between = int(variant["months_between_rebalances"])
        variant_scored = select_rebalance_anchors(scored, months_between)
        variant_config_id = construction_config_id(PRODUCTION_CONFIG_ID, frequency, top_n, mode)

        daily, rebalances = monthly_sim.simulate_variant(
            scored=variant_scored,
            config_id=variant_config_id,
            top_n=top_n,
            mode=mode,
            price_wide=price_wide,
            adv20=adv20,
            params=params,
            historical_universe=historical_universe,
        )
        if daily.empty or rebalances.empty:
            continue

        daily = monthly_sim.assign_fold_labels(daily, "Date", price_wide.index.max())
        rebalances = monthly_sim.assign_fold_labels(rebalances, "Trade_Date", price_wide.index.max())
        for frame in (daily, rebalances):
            frame["Base_Config_ID"] = PRODUCTION_CONFIG_ID
            frame["Rebalance_Frequency"] = frequency
            frame["Months_Between_Rebalances"] = months_between
            frame["Construction_ID"] = variant_config_id

        summary = monthly_sim.summarize_simulation(daily, rebalances)
        if not summary.empty:
            summary["Base_Config_ID"] = PRODUCTION_CONFIG_ID
            summary["Rebalance_Frequency"] = frequency
            summary["Months_Between_Rebalances"] = months_between
            summary["Construction_ID"] = variant_config_id
            summary["Anchor_Count"] = int(variant_scored["Date"].nunique())
            summary_frames.append(summary)
        daily_frames.append(daily)
        rebalance_frames.append(rebalances)

    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    rebalances_all = pd.concat(rebalance_frames, ignore_index=True) if rebalance_frames else pd.DataFrame()
    summary_all = (
        pd.concat(summary_frames, ignore_index=True)
        .sort_values("Simulator_Objective", ascending=False)
        .reset_index(drop=True)
        if summary_frames else pd.DataFrame()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_all.to_csv(output_dir / "holdout_construction_equity_curve.csv", index=False)
    rebalances_all.to_csv(output_dir / "holdout_construction_rebalance_log.csv", index=False)
    summary_all.to_csv(output_dir / "holdout_construction_summary.csv", index=False)

    default_config_id = construction_config_id(PRODUCTION_CONFIG_ID, "monthly", 10, "rank_weighted")
    default_daily = daily_all[daily_all["Config_ID"] == default_config_id].copy() if not daily_all.empty else pd.DataFrame()
    default_rebalances = (
        rebalances_all[rebalances_all["Config_ID"] == default_config_id].copy()
        if not rebalances_all.empty else pd.DataFrame()
    )
    default_summary = (
        summary_all[summary_all["Config_ID"] == default_config_id].copy()
        if not summary_all.empty else pd.DataFrame()
    )
    default_daily.to_csv(output_dir / "holdout_monthly_equity_curve.csv", index=False)
    default_rebalances.to_csv(output_dir / "holdout_rebalance_log.csv", index=False)
    default_summary.to_csv(output_dir / "holdout_simulation_summary.csv", index=False)
    return daily_all, rebalances_all, summary_all


def write_summary(
    output_path: Path,
    args: argparse.Namespace,
    holdout_universe: pd.DataFrame,
    score_dir: Path,
    simulation_dir: Path,
    simulation_summary: pd.DataFrame,
) -> None:
    score_path = score_dir / "point_in_time_scores.csv"
    scores = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    anchor_results_path = score_dir / "anchor_results.csv"
    anchor_results = pd.read_csv(anchor_results_path) if anchor_results_path.exists() else pd.DataFrame()
    default_config_id = construction_config_id(PRODUCTION_CONFIG_ID, "monthly", 10, "rank_weighted")
    default_rows = (
        simulation_summary[simulation_summary["Config_ID"] == default_config_id]
        if not simulation_summary.empty and "Config_ID" in simulation_summary.columns else pd.DataFrame()
    )
    default = default_rows.iloc[0].to_dict() if not default_rows.empty else {}
    best = simulation_summary.iloc[0].to_dict() if not simulation_summary.empty else {}

    lines = [
        "# Pre-Tuning Holdout Test Summary",
        "",
        "## Purpose",
        "",
        "This run tests the current frozen Stock Analysis default on an older historical window that was not the primary tuning period.",
        "",
        "Frozen construction:",
        "",
        f"- Score family: `{PRODUCTION_ALPHA_COLUMN}`",
        "- Portfolio: top `10`, `rank_weighted`",
        "- Liquidity: `$30M` minimum 20-day dollar volume",
        "- Transaction/cost mechanics: same monthly simulator defaults as the production-facing research output",
        "",
        "## Holdout Window",
        "",
        f"- Holdout start target: `{args.holdout_start}`",
        f"- Holdout end target: `{args.holdout_end}`",
        f"- Score anchors generated: `{scores['Date'].nunique() if not scores.empty and 'Date' in scores.columns else 0}`",
        f"- Score rows generated: `{len(scores):,}`",
        f"- Holdout universe rows: `{len(holdout_universe):,}`",
        f"- Output folder: `{args.output_dir}`",
        "",
        "## Frozen Monthly Portfolio Result",
        "",
    ]
    if default:
        lines.extend([
            f"- Final strategy equity: `{fmt_num(default.get('Final_Equity'))}`",
            f"- Total return: `{fmt_pct(default.get('Total_Return'))}`",
            f"- Universe total return: `{fmt_pct(default.get('Universe_Total_Return'))}`",
            f"- QQQ total return: `{fmt_pct(default.get('QQQ_Total_Return'))}`",
            f"- XLK total return: `{fmt_pct(default.get('XLK_Total_Return'))}`",
            f"- Excess return vs universe: `{fmt_pct(default.get('Excess_Return_vs_Universe'))}`",
            f"- Excess return vs QQQ: `{fmt_pct(default.get('Excess_Return_vs_QQQ'))}`",
            f"- Excess return vs XLK: `{fmt_pct(default.get('Excess_Return_vs_XLK'))}`",
            f"- Sharpe: `{fmt_num(default.get('Sharpe'))}`",
            f"- Sortino: `{fmt_num(default.get('Sortino'))}`",
            f"- Max drawdown: `{fmt_pct(default.get('Max_Drawdown'))}`",
            f"- Average turnover: `{fmt_pct(default.get('Avg_Turnover'))}`",
            f"- Average cash weight: `{fmt_pct(default.get('Avg_Cash_Weight'))}`",
            "",
        ])
    else:
        lines.extend(["No monthly simulation summary was produced.", ""])

    lines.extend([
        "## Construction Fix Test",
        "",
        "The first holdout warning was that the signal looked more competitive over `9M` and `12M` horizons than in the monthly compounded top-10 portfolio. This pass therefore keeps the score formula frozen and tests whether slower, less reactive portfolio construction improves the same signal.",
        "",
    ])
    if best:
        lines.extend([
            "Best construction by the simulator objective:",
            "",
            f"- Construction: `{best.get('Rebalance_Frequency')}` / top `{int(best.get('Top_N', 0))}` / `{best.get('Portfolio_Mode')}`",
            f"- Total return: `{fmt_pct(best.get('Total_Return'))}`",
            f"- Excess return vs universe: `{fmt_pct(best.get('Excess_Return_vs_Universe'))}`",
            f"- Excess return vs QQQ: `{fmt_pct(best.get('Excess_Return_vs_QQQ'))}`",
            f"- Excess return vs XLK: `{fmt_pct(best.get('Excess_Return_vs_XLK'))}`",
            f"- Sharpe: `{fmt_num(best.get('Sharpe'))}`",
            f"- Max drawdown: `{fmt_pct(best.get('Max_Drawdown'))}`",
            f"- Average turnover: `{fmt_pct(best.get('Avg_Turnover'))}`",
            "",
            "Construction variants tested:",
            "",
        ])
        for _, row in simulation_summary.iterrows():
            lines.append(
                f"- `{row.get('Rebalance_Frequency')}` top `{int(row.get('Top_N', 0))}` `{row.get('Portfolio_Mode')}`: "
                f"return `{fmt_pct(row.get('Total_Return'))}`, "
                f"excess vs XLK `{fmt_pct(row.get('Excess_Return_vs_XLK'))}`, "
                f"drawdown `{fmt_pct(row.get('Max_Drawdown'))}`, "
                f"turnover `{fmt_pct(row.get('Avg_Turnover'))}`, "
                f"rebalances `{int(row.get('Rebalance_Count', 0))}`"
            )
        lines.append("")
    else:
        lines.extend(["No construction-variant simulation summary was produced.", ""])

    lines.extend([
        "## Forward-Return Cross-Check",
        "",
    ])
    if not anchor_results.empty:
        forward_summary = (
            anchor_results.groupby("Horizon", dropna=False)
            .agg(
                Observations=("Top_Avg_Return", "count"),
                Top_Avg_Return=("Top_Avg_Return", "mean"),
                Universe_Avg_Return=("Universe_Avg_Return", "mean"),
                QQQ_Return=("QQQ_Return", "mean"),
                XLK_Return=("XLK_Return", "mean"),
                Top_Excess_vs_Universe=("Top_Excess_vs_Universe", "mean"),
                Top_Excess_vs_QQQ=("Top_Excess_vs_QQQ", "mean"),
                Top_Excess_vs_XLK=("Top_Excess_vs_XLK", "mean"),
                Avg_Fair_Value_Coverage=("Fair_Value_Coverage", "mean"),
            )
            .reset_index()
        )
        forward_summary.to_csv(output_path.parent / "holdout_forward_return_summary.csv", index=False)
        for _, row in forward_summary.iterrows():
            lines.append(
                f"- `{row['Horizon']}`: observations `{int(row['Observations'])}`, "
                f"top avg `{fmt_pct(row['Top_Avg_Return'])}`, "
                f"excess vs universe `{fmt_pct(row['Top_Excess_vs_Universe'])}`, "
                f"excess vs QQQ `{fmt_pct(row['Top_Excess_vs_QQQ'])}`, "
                f"excess vs XLK `{fmt_pct(row['Top_Excess_vs_XLK'])}`, "
                f"fair-value coverage `{fmt_pct(row['Avg_Fair_Value_Coverage'])}`"
            )
        lines.append("")
    else:
        lines.extend(["No forward-return anchor results were produced.", ""])

    lines.extend([
        "## Interpretation Rules",
        "",
        "Do not retune the model from this result. This is a holdout check of the frozen current default.",
        "",
        "A good result would show positive excess return versus the model universe and competitive behavior versus `QQQ`/`XLK` without extreme drawdown.",
        "",
        "A weak result does not automatically invalidate the model because this holdout uses a coverage-limited universe, but it does mean the model still needs stronger out-of-sample evidence before it can be described as robust.",
        "",
        "If slower construction beats monthly construction, treat that as a portfolio-construction fix candidate rather than proof that the alpha weights are solved. It should be re-run on the Sharadar validation period and in forward paper testing before replacing the website/default construction.",
        "",
        "## Known Limitations",
        "",
        "- The holdout universe is backfilled from the current model-seeded universe plus recovered stale/delisted names, not a complete historical 2019 technology universe.",
        "- Membership starts are based on vendor price availability, so this can still contain survivorship and model-universe hindsight.",
        "- The test is useful for signal behavior and stress testing, but the live forward paper test remains the cleanest future proof.",
        "- Sharadar fundamentals are filtered point-in-time by filing date at each anchor.",
        "",
        "## Files Produced",
        "",
        f"- `{score_dir / 'point_in_time_scores.csv'}`",
        f"- `{score_dir / 'anchor_results.csv'}`",
        f"- `{simulation_dir / 'holdout_construction_equity_curve.csv'}`",
        f"- `{simulation_dir / 'holdout_construction_rebalance_log.csv'}`",
        f"- `{simulation_dir / 'holdout_construction_summary.csv'}`",
        f"- `{simulation_dir / 'holdout_monthly_equity_curve.csv'}`",
        f"- `{simulation_dir / 'holdout_rebalance_log.csv'}`",
        f"- `{simulation_dir / 'holdout_simulation_summary.csv'}`",
        f"- `{output_path.parent / 'holdout_forward_return_summary.csv'}`",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdout_start = pd.to_datetime(args.holdout_start).normalize()
    holdout_end = pd.to_datetime(args.holdout_end).normalize()
    holdout_universe_path = args.output_dir / "holdout_backfilled_universe.csv"
    score_dir = args.output_dir / "combined_score"
    simulation_dir = args.output_dir / "monthly_simulation"

    holdout_universe = build_holdout_universe(
        base_universe_path=args.base_universe,
        security_master_path=args.security_master,
        output_path=holdout_universe_path,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )

    if not args.skip_score_rebuild:
        run_score_rebuild(
            output_dir=score_dir,
            holdout_universe=holdout_universe_path,
            historical_price_file=args.historical_price_file,
            years=args.score_years,
            max_anchors=args.max_anchors,
            resume=args.resume_scores,
        )

    score_history = score_dir / "point_in_time_scores.csv"
    if not score_history.exists():
        raise FileNotFoundError(f"Score history not found after rebuild: {score_history}")

    daily, rebalances, simulation_summary = run_frozen_construction_simulations(
        score_history=score_history,
        holdout_universe=holdout_universe_path,
        historical_price_file=args.historical_price_file,
        output_dir=simulation_dir,
    )
    metadata = {
        "status": "complete",
        "holdout_start": args.holdout_start,
        "holdout_end": args.holdout_end,
        "holdout_universe_rows": int(len(holdout_universe)),
        "score_history": str(score_history),
        "daily_rows": int(len(daily)),
        "rebalance_rows": int(len(rebalances)),
        "simulation_summary_rows": int(len(simulation_summary)),
        "construction_variants": CONSTRUCTION_VARIANTS,
        "best_construction": (
            simulation_summary.iloc[0].to_dict()
            if not simulation_summary.empty else {}
        ),
        "frozen_alpha_column": PRODUCTION_ALPHA_COLUMN,
        "frozen_config_id": PRODUCTION_CONFIG_ID,
        "default_top_n": 10,
        "default_portfolio_mode": "rank_weighted",
        "min_dollar_volume": 30_000_000,
        "limitation": "Coverage-limited pre-tuning holdout; not a complete historical tech universe.",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    write_summary(
        output_path=args.output_dir / "PRE_TUNING_HOLDOUT_SUMMARY.md",
        args=args,
        holdout_universe=holdout_universe,
        score_dir=score_dir,
        simulation_dir=simulation_dir,
        simulation_summary=simulation_summary,
    )
    print("[SUCCESS] Pre-tuning holdout test complete")
    print(json.dumps(metadata, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
