#!/usr/bin/env python3
"""
Candidate-family stability and liquidity-floor test harness.

This is a compact production-readiness pass for the Sharadar-backed monthly
portfolio simulator. It keeps the same score history and candidate set, then
tests whether walk-forward selection remains durable as the minimum dollar
volume floor becomes stricter.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import historical_data_layer as historical_data
import monthly_rebalanced_portfolio_simulator as monthly_sim
import portfolio_construction_validation as construction
import portfolio_robustness_stress_test as robustness
import combined_weight_tuning as weight_tuning


DEFAULT_SCORE_HISTORY = Path("backtests/combined_score_sharadar/point_in_time_scores.csv")
DEFAULT_WEIGHT_TUNING_DIR = Path("backtests/weight_tuning_sharadar")
DEFAULT_PRICE_FILE = Path("data/historical_data_layer/prices_daily.csv")
DEFAULT_HISTORICAL_UNIVERSE = Path("data/historical_tech_universe.csv")
DEFAULT_OUTPUT_DIR = Path("backtests/candidate_family_liquidity_stability_sharadar")
DEFAULT_LIQUIDITY_FLOORS = "5000000,25000000,50000000,100000000"
DEFAULT_TOP_N_LIST = "10,20,50,100"
DEFAULT_MODES = "equal,rank_weighted"
DEFAULT_FALLBACK_MODES = "none,cash,qqq,xlk"
DEFAULT_CLEAN_GUARD_STAGES = robustness.DEFAULT_CLEAN_GUARD_STAGES


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        value = item.strip()
        if not value:
            continue
        parsed = float(value)
        if parsed <= 0:
            raise ValueError(f"Liquidity floor must be positive: {parsed}")
        if parsed not in values:
            values.append(parsed)
    return values


def parse_list(raw: str) -> list[str]:
    return robustness.parse_list(raw)


def load_inputs(args: argparse.Namespace) -> tuple[
    pd.DataFrame,
    list[weight_tuning.WeightConfig | monthly_sim.DirectScoreConfig],
    pd.DataFrame,
    pd.DataFrame,
    monthly_sim.HistoricalUniverse,
]:
    scores = weight_tuning.prepare_score_history(args.score_history)
    base_tickers = sorted(set(scores["Ticker"]) | set(monthly_sim.BENCHMARK_TICKERS))
    start_date = scores["Date"].min() - pd.Timedelta(days=80)
    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

    print(f"[INFO] Loaded {len(scores):,} score rows across {scores['Date'].nunique()} anchors", flush=True)
    print(f"[INFO] Loading canonical prices from {args.historical_price_file}", flush=True)
    price_wide = historical_data.load_canonical_price_wide(
        path=args.historical_price_file,
        tickers=base_tickers,
        start_date=start_date,
        end_date=end_date,
        value_column="Close",
    )
    volume_wide = historical_data.load_canonical_price_wide(
        path=args.historical_price_file,
        tickers=base_tickers,
        start_date=start_date,
        end_date=end_date,
        value_column="Volume",
    )
    if price_wide.empty:
        raise RuntimeError("No canonical price history loaded")

    configs = construction.select_configs(
        config_scope=args.config_scope,
        preset=args.candidate_preset,
        weight_tuning_dir=args.weight_tuning_dir,
        shortlist_count=args.shortlist_count,
        max_configs=args.max_configs,
    )
    alpha_score_columns = parse_list(args.alpha_score_columns)
    scores, configs = monthly_sim.add_alpha_score_configs(
        scores=scores,
        configs=configs,
        score_columns=alpha_score_columns,
        price_wide=price_wide,
    )
    historical_universe = monthly_sim.load_historical_universe(
        args.historical_universe,
        scores,
        price_wide,
    )
    print(f"[INFO] Testing {len(configs)} configs with alpha columns: {alpha_score_columns}", flush=True)
    print(f"[INFO] Historical universe control: {historical_universe.source}", flush=True)
    return scores, configs, price_wide, volume_wide, historical_universe


def selection_params(args: argparse.Namespace) -> monthly_sim.WalkForwardSelectionParams:
    return monthly_sim.WalkForwardSelectionParams(
        mode="guarded_family_stable",
        min_family_train_folds=args.guard_min_family_train_folds,
        min_family_recent_excess_vs_universe=args.guard_min_family_recent_excess_vs_universe,
        min_family_recent_excess_vs_xlk=args.guard_min_family_recent_excess_vs_xlk,
        max_family_worst_drawdown=args.guard_max_family_worst_drawdown,
        max_family_downside_capture_vs_xlk=args.guard_max_family_downside_capture_vs_xlk,
        min_family_positive_universe_fold_rate=args.guard_min_family_positive_universe_fold_rate,
        min_family_positive_xlk_fold_rate=args.guard_min_family_positive_xlk_fold_rate,
        family_stability_weight=args.guard_family_stability_weight,
    )


def base_simulation_params(args: argparse.Namespace, liquidity_floor: float) -> monthly_sim.SimulationParams:
    return monthly_sim.SimulationParams(
        initial_capital=args.initial_capital,
        rebalance_delay_days=args.rebalance_delay_days,
        min_dollar_volume=liquidity_floor,
        adv_window=args.adv_window,
        max_position_weight=args.max_position_weight,
        max_subindustry_weight=args.max_subindustry_weight,
        max_position_adv_pct=args.max_position_adv_pct,
        max_trade_adv_pct=args.max_trade_adv_pct,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps_per_1pct_adv=args.slippage_bps_per_1pct_adv,
    )


def value_counts_string(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    counts = frame[column].fillna("").astype(str).value_counts().sort_index()
    return ",".join(f"{key}:{int(value)}" for key, value in counts.items() if key)


def run_liquidity_floor(
    liquidity_floor: float,
    args: argparse.Namespace,
    scores: pd.DataFrame,
    configs: list[weight_tuning.WeightConfig | monthly_sim.DirectScoreConfig],
    price_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    historical_universe: monthly_sim.HistoricalUniverse,
    top_n_list: list[int],
    modes: list[str],
    fallback_modes: list[str],
    clean_guard_stages: tuple[str, ...],
    selector: monthly_sim.WalkForwardSelectionParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = base_simulation_params(args, liquidity_floor)
    print(f"[INFO] Liquidity floor ${liquidity_floor:,.0f}", flush=True)
    adv20 = monthly_sim.rolling_dollar_volume(price_wide, volume_wide, params.adv_window)
    daily, rebalances, _ = monthly_sim.simulate_all(
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
    daily = monthly_sim.assign_fold_labels(daily, "Date", end_date)
    rebalances = monthly_sim.assign_fold_labels(rebalances, "Trade_Date", end_date)

    static_summary = monthly_sim.summarize_simulation(daily, rebalances).copy()
    selections = monthly_sim.walk_forward_selection_from_equity(
        daily=daily,
        rebalances=rebalances,
        selection_params=selector,
    ).copy()

    for frame in (static_summary, selections):
        frame["Liquidity_Floor"] = liquidity_floor
        frame["Selection_Mode"] = selector.mode

    clean_rounds = (
        int(selections["Selection_Guard_Stage"].isin(clean_guard_stages).sum())
        if not selections.empty and "Selection_Guard_Stage" in selections.columns
        else 0
    )
    selection_rounds = int(len(selections))
    family_counts = value_counts_string(selections, "Selected_Candidate_Family_ID")
    config_counts = value_counts_string(selections, "Selected_Config_ID")
    stage_counts = value_counts_string(selections, "Selection_Guard_Stage")

    result_rows: list[dict] = []
    for fallback_mode in fallback_modes:
        fallback_policy = monthly_sim.FallbackPolicy(
            mode=fallback_mode,
            clean_guard_stages=clean_guard_stages,
            exposure=args.fallback_exposure,
        )
        wf_daily, wf_rebalances, wf_summary = monthly_sim.simulate_walk_forward_dynamic_portfolio(
            scores=scores,
            configs=configs,
            selections=selections,
            price_wide=price_wide,
            adv20=adv20,
            params=params,
            historical_universe=historical_universe,
            fallback_policy=fallback_policy,
        )
        if wf_summary.empty:
            continue

        row = wf_summary.iloc[0].to_dict()
        row.update({
            "Liquidity_Floor": liquidity_floor,
            "Fallback_Mode": fallback_mode,
            "Clean_Guard_Stages": ",".join(clean_guard_stages),
            "Selection_Rounds": selection_rounds,
            "Clean_Selection_Rounds": clean_rounds,
            "Fallback_Would_Trigger_Rounds": max(0, selection_rounds - clean_rounds),
            "Selected_Family_Counts": family_counts,
            "Selected_Config_Counts": config_counts,
            "Selection_Guard_Stage_Counts": stage_counts,
            "Unique_Selected_Family_Count": (
                int(selections["Selected_Candidate_Family_ID"].nunique())
                if not selections.empty and "Selected_Candidate_Family_ID" in selections.columns
                else 0
            ),
            "Average_Eligible_Count": (
                float(wf_rebalances["Strategy_Eligible_Count"].mean())
                if not wf_rebalances.empty and "Strategy_Eligible_Count" in wf_rebalances.columns
                else np.nan
            ),
            "Average_Excluded_By_Liquidity": (
                float(wf_rebalances["Strategy_Excluded_By_Liquidity"].mean())
                if not wf_rebalances.empty and "Strategy_Excluded_By_Liquidity" in wf_rebalances.columns
                else np.nan
            ),
            "Average_Position_Cap_Binds": (
                float(wf_rebalances["Strategy_Position_Cap_Bind_Count"].mean())
                if not wf_rebalances.empty and "Strategy_Position_Cap_Bind_Count" in wf_rebalances.columns
                else np.nan
            ),
            "Average_Subindustry_Cap_Binds": (
                float(wf_rebalances["Strategy_Subindustry_Cap_Bind_Count"].mean())
                if not wf_rebalances.empty and "Strategy_Subindustry_Cap_Bind_Count" in wf_rebalances.columns
                else np.nan
            ),
            "Family_Liquidity_Robustness_Objective": robustness.robustness_objective(row),
        })
        if not static_summary.empty:
            best_static = static_summary.iloc[0].to_dict()
            row.update({
                "Best_Static_Config_ID": best_static.get("Config_ID", ""),
                "Best_Static_Top_N": best_static.get("Top_N", np.nan),
                "Best_Static_Portfolio_Mode": best_static.get("Portfolio_Mode", ""),
                "Best_Static_Total_Return": best_static.get("Total_Return", np.nan),
                "Best_Static_Excess_Return_vs_XLK": best_static.get("Excess_Return_vs_XLK", np.nan),
            })
        result_rows.append(row)

    return static_summary, selections, pd.DataFrame(result_rows)


def format_pct(value: object) -> str:
    return monthly_sim.format_pct(value)


def format_num(value: object, digits: int = 4) -> str:
    return monthly_sim.format_num(value, digits)


def write_summary(
    output_dir: Path,
    dynamic_summary: pd.DataFrame,
    selections: pd.DataFrame,
    selector: monthly_sim.WalkForwardSelectionParams,
    liquidity_floors: list[float],
    fallback_modes: list[str],
) -> None:
    lines = [
        "# Candidate-Family Liquidity Stability Summary",
        "",
        "## What was completed",
        "",
        "This pass tested whether the walk-forward selector stays stable at stricter liquidity floors before any live promotion.",
        "The selector used `guarded_family_stable`, which adds score-family consistency on top of the prior recent-fold, drawdown, turnover, and XLK/downside guards.",
        "",
        "Liquidity floors tested:",
    ]
    for floor in liquidity_floors:
        lines.append(f"- ${floor:,.0f} minimum 20-day dollar volume")

    lines.extend(["", "Fallback modes tested:"])
    for mode in fallback_modes:
        lines.append(f"- `{mode}`")

    lines.extend([
        "",
        "Family-stability settings:",
        f"- Minimum family training folds: {selector.min_family_train_folds}",
        f"- Minimum family recent excess vs universe: {format_pct(selector.min_family_recent_excess_vs_universe)}",
        f"- Minimum family recent excess vs XLK: {format_pct(selector.min_family_recent_excess_vs_xlk)}",
        f"- Maximum family worst-fold drawdown: {format_pct(selector.max_family_worst_drawdown)}",
        f"- Maximum family average downside capture vs XLK: {format_num(selector.max_family_downside_capture_vs_xlk)}",
        f"- Minimum family positive universe-fold rate: {format_pct(selector.min_family_positive_universe_fold_rate)}",
        f"- Minimum family positive XLK-fold rate: {format_pct(selector.min_family_positive_xlk_fold_rate)}",
        f"- Family stability objective weight: {format_num(selector.family_stability_weight)}",
        "",
    ])

    if dynamic_summary.empty:
        lines.append("No dynamic liquidity-stability results were produced.")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "CANDIDATE_FAMILY_LIQUIDITY_STABILITY_SUMMARY.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        return

    ranked = dynamic_summary.sort_values("Family_Liquidity_Robustness_Objective", ascending=False)
    best = ranked.iloc[0]
    lines.extend([
        "## Best Dynamic Result",
        "",
        f"- Liquidity floor: ${best.get('Liquidity_Floor'):,.0f}",
        f"- Fallback mode: `{best.get('Fallback_Mode')}`",
        f"- Total return: {format_pct(best.get('Total_Return'))}",
        f"- Excess return vs simulated universe: {format_pct(best.get('Excess_Return_vs_Universe'))}",
        f"- Excess return vs QQQ: {format_pct(best.get('Excess_Return_vs_QQQ'))}",
        f"- Excess return vs XLK: {format_pct(best.get('Excess_Return_vs_XLK'))}",
        f"- Sharpe: {format_num(best.get('Sharpe'))}",
        f"- Sortino: {format_num(best.get('Sortino'))}",
        f"- Max drawdown: {format_pct(best.get('Max_Drawdown'))}",
        f"- Average turnover: {format_pct(best.get('Avg_Turnover'))}",
        f"- Fallback rebalances: {int(best.get('Fallback_Rebalance_Count', 0))}",
        f"- Selected families: `{best.get('Selected_Family_Counts', '')}`",
        "",
    ])

    lines.extend(["## Liquidity Floor Comparison", ""])
    floor_best = (
        dynamic_summary
        .sort_values("Family_Liquidity_Robustness_Objective", ascending=False)
        .groupby("Liquidity_Floor", as_index=False)
        .head(1)
        .sort_values("Liquidity_Floor")
    )
    for _, row in floor_best.iterrows():
        lines.append(
            "- "
            f"${row['Liquidity_Floor']:,.0f}: `{row['Fallback_Mode']}` fallback, "
            f"return {format_pct(row.get('Total_Return'))}, "
            f"excess vs XLK {format_pct(row.get('Excess_Return_vs_XLK'))}, "
            f"Sharpe {format_num(row.get('Sharpe'))}, "
            f"drawdown {format_pct(row.get('Max_Drawdown'))}, "
            f"avg eligible {format_num(row.get('Average_Eligible_Count'), 1)}, "
            f"fallback rebalances {int(row.get('Fallback_Rebalance_Count', 0))}"
        )

    lines.extend(["", "## No-Fallback Production Readiness View", ""])
    no_fallback = dynamic_summary[dynamic_summary["Fallback_Mode"] == "none"].sort_values("Liquidity_Floor")
    if no_fallback.empty:
        lines.append("No no-fallback runs were produced.")
    else:
        for _, row in no_fallback.iterrows():
            lines.append(
                "- "
                f"${row['Liquidity_Floor']:,.0f}: return {format_pct(row.get('Total_Return'))}, "
                f"excess vs universe {format_pct(row.get('Excess_Return_vs_Universe'))}, "
                f"excess vs XLK {format_pct(row.get('Excess_Return_vs_XLK'))}, "
                f"Sharpe {format_num(row.get('Sharpe'))}, "
                f"drawdown {format_pct(row.get('Max_Drawdown'))}, "
                f"avg excluded by liquidity {format_num(row.get('Average_Excluded_By_Liquidity'), 1)}, "
                f"families `{row.get('Selected_Family_Counts', '')}`"
            )

    lines.extend(["", "## Selected Families By Fold", ""])
    if selections.empty:
        lines.append("No walk-forward selections were produced.")
    else:
        for floor, group in selections.groupby("Liquidity_Floor"):
            selected = [
                (
                    f"{row.Validation_Fold}:"
                    f"{row.Selected_Candidate_Family_ID}/"
                    f"top{int(row.Selected_Top_N)}/"
                    f"{row.Selected_Portfolio_Mode}/"
                    f"{row.Selection_Guard_Stage}"
                )
                for row in group.itertuples(index=False)
            ]
            lines.append(f"- ${floor:,.0f}: {'; '.join(selected)}")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "A live candidate should not only win the full period. It should keep producing acceptable folds as the liquidity floor rises, avoid relying on a single unstable score family, and stay competitive with XLK after turnover and friction.",
        "",
        "If the strongest result only appears at a very restrictive liquidity floor, that is useful but also means the strategy is becoming a large-cap technology stock selector. If performance collapses at stricter floors, the signal is too dependent on smaller or less-liquid stocks for live use.",
        "",
        "## Files produced",
        "",
        "- `liquidity_floor_dynamic_summary.csv`",
        "- `liquidity_floor_static_summary.csv`",
        "- `liquidity_floor_selected_constructions.csv`",
        "- `candidate_family_selection_counts.csv`",
        "- `run_metadata.json`",
        "- `CANDIDATE_FAMILY_LIQUIDITY_STABILITY_SUMMARY.md`",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "CANDIDATE_FAMILY_LIQUIDITY_STABILITY_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run candidate-family stability and liquidity-floor tests.")
    parser.add_argument("--score-history", type=Path, default=DEFAULT_SCORE_HISTORY)
    parser.add_argument("--weight-tuning-dir", type=Path, default=DEFAULT_WEIGHT_TUNING_DIR)
    parser.add_argument("--historical-price-file", type=Path, default=DEFAULT_PRICE_FILE)
    parser.add_argument("--historical-universe", type=Path, default=DEFAULT_HISTORICAL_UNIVERSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--liquidity-floors", default=DEFAULT_LIQUIDITY_FLOORS)
    parser.add_argument("--fallback-modes", default=DEFAULT_FALLBACK_MODES)
    parser.add_argument("--fallback-clean-stages", default=DEFAULT_CLEAN_GUARD_STAGES)
    parser.add_argument("--fallback-exposure", type=float, default=1.0)
    parser.add_argument("--candidate-preset", choices=["compact", "expanded"], default="expanded")
    parser.add_argument("--config-scope", choices=["shortlist", "all"], default="shortlist")
    parser.add_argument("--shortlist-count", type=int, default=10)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--alpha-score-columns", default=robustness.DEFAULT_ALPHA_SCORE_COLUMNS)
    parser.add_argument("--top-n-list", default=DEFAULT_TOP_N_LIST)
    parser.add_argument("--portfolio-modes", default=DEFAULT_MODES)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--rebalance-delay-days", type=int, default=1)
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--max-position-weight", type=float, default=0.15)
    parser.add_argument("--max-subindustry-weight", type=float, default=0.35)
    parser.add_argument("--max-position-adv-pct", type=float, default=0.02)
    parser.add_argument("--max-trade-adv-pct", type=float, default=0.01)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps-per-1pct-adv", type=float, default=5.0)
    parser.add_argument("--guard-min-family-train-folds", type=int, default=2)
    parser.add_argument("--guard-min-family-recent-excess-vs-universe", type=float, default=0.0)
    parser.add_argument("--guard-min-family-recent-excess-vs-xlk", type=float, default=-0.05)
    parser.add_argument("--guard-max-family-worst-drawdown", type=float, default=0.30)
    parser.add_argument("--guard-max-family-downside-capture-vs-xlk", type=float, default=0.95)
    parser.add_argument("--guard-min-family-positive-universe-fold-rate", type=float, default=0.75)
    parser.add_argument("--guard-min-family-positive-xlk-fold-rate", type=float, default=0.50)
    parser.add_argument("--guard-family-stability-weight", type=float, default=0.35)
    args = parser.parse_args()

    liquidity_floors = parse_float_list(args.liquidity_floors)
    fallback_modes = parse_list(args.fallback_modes)
    unknown_fallback = sorted(set(fallback_modes) - {"none", "cash", "qqq", "xlk"})
    if unknown_fallback:
        raise ValueError(f"Unknown fallback mode(s): {unknown_fallback}")
    if args.fallback_exposure < 0:
        raise ValueError("fallback exposure cannot be negative")
    if args.guard_min_family_train_folds <= 0:
        raise ValueError("minimum family train folds must be positive")
    if args.guard_family_stability_weight < 0:
        raise ValueError("family stability weight cannot be negative")

    clean_guard_stages = tuple(parse_list(args.fallback_clean_stages))
    top_n_list = construction.parse_top_n_list(args.top_n_list)
    modes = construction.parse_modes(args.portfolio_modes)
    selector = selection_params(args)

    scores, configs, price_wide, volume_wide, historical_universe = load_inputs(args)

    static_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    dynamic_frames: list[pd.DataFrame] = []
    for floor in liquidity_floors:
        static_summary, selections, dynamic_summary = run_liquidity_floor(
            liquidity_floor=floor,
            args=args,
            scores=scores,
            configs=configs,
            price_wide=price_wide,
            volume_wide=volume_wide,
            historical_universe=historical_universe,
            top_n_list=top_n_list,
            modes=modes,
            fallback_modes=fallback_modes,
            clean_guard_stages=clean_guard_stages,
            selector=selector,
        )
        static_frames.append(static_summary)
        selection_frames.append(selections)
        dynamic_frames.append(dynamic_summary)

    static_all = pd.concat(static_frames, ignore_index=True) if static_frames else pd.DataFrame()
    selections_all = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    dynamic_all = pd.concat(dynamic_frames, ignore_index=True) if dynamic_frames else pd.DataFrame()

    family_counts = pd.DataFrame()
    if not selections_all.empty and "Selected_Candidate_Family_ID" in selections_all.columns:
        family_counts = (
            selections_all
            .groupby(["Liquidity_Floor", "Selected_Candidate_Family_ID"], dropna=False)
            .agg(
                Selection_Count=("Selected_Candidate_Family_ID", "size"),
                Avg_Validation_Excess_vs_Universe=("Validation_Excess_Return_vs_Universe", "mean"),
                Avg_Validation_Excess_vs_XLK=("Validation_Excess_Return_vs_XLK", "mean"),
                Worst_Validation_Drawdown=("Validation_Max_Drawdown", "min"),
                Avg_Family_Positive_XLK_Fold_Rate=("Family_Positive_XLK_Fold_Rate", "mean"),
                Avg_Family_Worst_Drawdown=("Family_Worst_Fold_Drawdown", "mean"),
            )
            .reset_index()
            .sort_values(["Liquidity_Floor", "Selection_Count"], ascending=[True, False])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dynamic_all.sort_values("Family_Liquidity_Robustness_Objective", ascending=False).round(6).to_csv(
        args.output_dir / "liquidity_floor_dynamic_summary.csv",
        index=False,
    )
    static_all.round(6).to_csv(args.output_dir / "liquidity_floor_static_summary.csv", index=False)
    selections_all.round(6).to_csv(args.output_dir / "liquidity_floor_selected_constructions.csv", index=False)
    family_counts.round(6).to_csv(args.output_dir / "candidate_family_selection_counts.csv", index=False)

    metadata = {
        "status": "complete",
        "score_history": str(args.score_history),
        "weight_tuning_dir": str(args.weight_tuning_dir),
        "historical_price_file": str(args.historical_price_file),
        "historical_universe": str(args.historical_universe),
        "liquidity_floors": liquidity_floors,
        "fallback_modes": fallback_modes,
        "fallback_clean_guard_stages": list(clean_guard_stages),
        "selection_params": asdict(selector),
        "candidate_preset": args.candidate_preset,
        "config_scope": args.config_scope,
        "shortlist_count": args.shortlist_count,
        "max_configs": args.max_configs,
        "alpha_score_columns": parse_list(args.alpha_score_columns),
        "top_n_list": top_n_list,
        "portfolio_modes": modes,
        "dynamic_summary_rows": int(len(dynamic_all)),
        "static_summary_rows": int(len(static_all)),
        "selection_rows": int(len(selections_all)),
        "candidate_family_selection_rows": int(len(family_counts)),
        "simulation_params_without_liquidity_floor": {
            key: value
            for key, value in asdict(base_simulation_params(args, liquidity_floors[0])).items()
            if key != "min_dollar_volume"
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    write_summary(
        output_dir=args.output_dir,
        dynamic_summary=dynamic_all,
        selections=selections_all,
        selector=selector,
        liquidity_floors=liquidity_floors,
        fallback_modes=fallback_modes,
    )

    print("\n=== CANDIDATE FAMILY LIQUIDITY STABILITY SUMMARY ===", flush=True)
    columns = [
        "Liquidity_Floor",
        "Fallback_Mode",
        "Family_Liquidity_Robustness_Objective",
        "Total_Return",
        "Excess_Return_vs_Universe",
        "Excess_Return_vs_XLK",
        "Sharpe",
        "Max_Drawdown",
        "Avg_Turnover",
        "Fallback_Rebalance_Count",
        "Selected_Family_Counts",
    ]
    if not dynamic_all.empty:
        print(
            dynamic_all.sort_values("Family_Liquidity_Robustness_Objective", ascending=False)[columns]
            .head(20)
            .to_string(index=False),
            flush=True,
        )
    print(f"\n[SUCCESS] Saved candidate-family liquidity outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
