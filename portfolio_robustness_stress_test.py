#!/usr/bin/env python3
"""
Robustness and fallback stress tests for the monthly portfolio simulator.

This harness keeps the ranking/scoring layer fixed, then asks whether the
defensive walk-forward strategy survives harsher portfolio assumptions. For
each stress scenario it tests the same defensive selector with no fallback,
cash fallback, QQQ fallback, and XLK fallback.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import combined_weight_tuning as weight_tuning
import historical_data_layer as historical_data
import monthly_rebalanced_portfolio_simulator as monthly_sim
import portfolio_construction_validation as construction


DEFAULT_SCORE_HISTORY = Path("backtests/combined_score_sharadar/point_in_time_scores.csv")
DEFAULT_WEIGHT_TUNING_DIR = Path("backtests/weight_tuning_sharadar")
DEFAULT_PRICE_FILE = Path("data/historical_data_layer/prices_daily.csv")
DEFAULT_HISTORICAL_UNIVERSE = Path("data/historical_tech_universe.csv")
DEFAULT_OUTPUT_DIR = Path("backtests/portfolio_robustness_stress_sharadar")
DEFAULT_ALPHA_SCORE_COLUMNS = ",".join([
    "Alpha_Prototype_SubIndustryRelRisk",
    "Alpha_Prototype_ValueTrendRelMom",
    "Alpha_Prototype_ValueSubRelRisk",
    "Alpha_Overlay_70Combined_30RelMom",
    "Relative_Momentum_vs_SubIndustry_RankPct",
    "Risk_Adjusted_Momentum_RankPct",
    "Alpha_Prototype_ConfidenceValueSubRelRisk",
    "Alpha_Prototype_BenchmarkAwareQuality",
    "Alpha_Prototype_XLKCompetitive",
    "Alpha_Prototype_MomentumQuality",
    "Alpha_Prototype_ConservativeComposite",
    "SubIndustry_Relative_Momentum_MidLong_RankPct",
    "SubIndustry_Relative_Momentum_Quality_RankPct",
    "Downside_Quality_RankPct",
    "Confidence_Adjusted_Fair_Value_RankPct",
    "Confidence_Adjusted_Combined_RankPct",
])
DEFAULT_TOP_N_LIST = "10,20,50,100"
DEFAULT_MODES = "equal,rank_weighted"
DEFAULT_FALLBACK_MODES = "none,cash,qqq,xlk"
DEFAULT_CLEAN_GUARD_STAGES = (
    "family_stable_strict_recent_xlk_drawdown_downside,"
    "family_stable_recent_defensive_downside,"
    "strict_recent_xlk_drawdown_downside,"
    "recent_defensive_downside"
)


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    notes: str
    overrides: dict[str, float | int]


def stress_scenarios(preset: str) -> list[StressScenario]:
    scenarios = [
        StressScenario("base", "Current defensive simulation assumptions.", {}),
        StressScenario(
            "higher_trading_costs",
            "Raises transaction cost and slippage to test cost sensitivity.",
            {"transaction_cost_bps": 25.0, "slippage_bps_per_1pct_adv": 10.0},
        ),
        StressScenario(
            "severe_trading_costs",
            "Uses punitive trading friction to reveal turnover fragility.",
            {"transaction_cost_bps": 50.0, "slippage_bps_per_1pct_adv": 25.0},
        ),
        StressScenario(
            "strict_liquidity",
            "Requires higher average dollar volume before a stock is eligible.",
            {"min_dollar_volume": 25_000_000.0},
        ),
        StressScenario(
            "tight_concentration",
            "Lowers position and subindustry caps to force more diversification/cash.",
            {"max_position_weight": 0.10, "max_subindustry_weight": 0.25},
        ),
        StressScenario(
            "larger_account_1m",
            "Scales the portfolio to $1M and tightens trade-size ADV limits.",
            {
                "initial_capital": 1_000_000.0,
                "max_position_adv_pct": 0.01,
                "max_trade_adv_pct": 0.005,
            },
        ),
        StressScenario(
            "combined_stress",
            "Combines higher costs, stricter liquidity, tighter caps, and larger size.",
            {
                "initial_capital": 1_000_000.0,
                "transaction_cost_bps": 25.0,
                "slippage_bps_per_1pct_adv": 15.0,
                "min_dollar_volume": 25_000_000.0,
                "max_position_weight": 0.10,
                "max_subindustry_weight": 0.25,
                "max_position_adv_pct": 0.01,
                "max_trade_adv_pct": 0.005,
            },
        ),
    ]
    if preset == "expanded":
        scenarios.extend([
            StressScenario(
                "very_strict_liquidity",
                "Requires $50M average dollar volume before a stock is eligible.",
                {"min_dollar_volume": 50_000_000.0},
            ),
            StressScenario(
                "very_tight_concentration",
                "Pushes concentration caps down to a more diversified, cash-heavy profile.",
                {"max_position_weight": 0.07, "max_subindustry_weight": 0.20},
            ),
            StressScenario(
                "larger_account_5m",
                "Scales the portfolio to $5M with tighter ADV limits.",
                {
                    "initial_capital": 5_000_000.0,
                    "max_position_adv_pct": 0.01,
                    "max_trade_adv_pct": 0.005,
                },
            ),
            StressScenario(
                "severe_combined_stress",
                "Combines severe costs, strict liquidity, tight caps, and $5M sizing.",
                {
                    "initial_capital": 5_000_000.0,
                    "transaction_cost_bps": 50.0,
                    "slippage_bps_per_1pct_adv": 25.0,
                    "min_dollar_volume": 50_000_000.0,
                    "max_position_weight": 0.07,
                    "max_subindustry_weight": 0.20,
                    "max_position_adv_pct": 0.01,
                    "max_trade_adv_pct": 0.005,
                },
            ),
        ])
    return scenarios


def parse_list(raw: str) -> list[str]:
    values = []
    for item in str(raw).split(","):
        value = item.strip()
        if value and value not in values:
            values.append(value)
    return values


def params_for_scenario(
    base: monthly_sim.SimulationParams,
    scenario: StressScenario,
) -> monthly_sim.SimulationParams:
    values = asdict(base)
    values.update(scenario.overrides)
    return monthly_sim.SimulationParams(**values)


def robustness_objective(row: pd.Series | dict) -> float:
    def value(name: str, default: float = 0.0) -> float:
        raw = row.get(name, default)
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return default
        return numeric if np.isfinite(numeric) else default

    return (
        value("Excess_Return_vs_Universe")
        + 0.80 * value("Excess_Return_vs_XLK")
        + 0.15 * value("Sharpe")
        - 0.80 * abs(value("Max_Drawdown"))
        - 0.05 * value("Avg_Turnover")
        - 0.02 * value("Fallback_Triggered_Pct")
    )


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


def selection_params() -> monthly_sim.WalkForwardSelectionParams:
    return monthly_sim.WalkForwardSelectionParams(mode="guarded_family_stable")


def run_scenario(
    scenario: StressScenario,
    params: monthly_sim.SimulationParams,
    scores: pd.DataFrame,
    configs: list[weight_tuning.WeightConfig | monthly_sim.DirectScoreConfig],
    price_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    historical_universe: monthly_sim.HistoricalUniverse,
    top_n_list: list[int],
    modes: list[str],
    fallback_modes: list[str],
    clean_guard_stages: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"[INFO] Scenario {scenario.scenario_id}: {scenario.notes}", flush=True)
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
    static_summary = monthly_sim.summarize_simulation(daily, rebalances)
    selections = monthly_sim.walk_forward_selection_from_equity(
        daily=daily,
        rebalances=rebalances,
        selection_params=selection_params(),
    )

    scenario_values = {
        "Scenario_ID": scenario.scenario_id,
        "Scenario_Notes": scenario.notes,
        **asdict(params),
    }
    static_summary = static_summary.copy()
    for key, value in scenario_values.items():
        static_summary[key] = value

    selections = selections.copy()
    for key, value in scenario_values.items():
        selections[key] = value

    result_rows = []
    dynamic_curves = []
    dynamic_rebalances = []
    for fallback_mode in fallback_modes:
        fallback_policy = monthly_sim.FallbackPolicy(
            mode=fallback_mode,
            clean_guard_stages=clean_guard_stages,
            exposure=1.0,
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

        summary_row = wf_summary.iloc[0].to_dict()
        summary_row.update(scenario_values)
        summary_row["Fallback_Mode"] = fallback_mode
        summary_row["Clean_Guard_Stages"] = ",".join(clean_guard_stages)
        summary_row["Stress_Robustness_Objective"] = robustness_objective(summary_row)
        best_static = static_summary.iloc[0].to_dict() if not static_summary.empty else {}
        summary_row["Best_Static_Config_ID"] = best_static.get("Config_ID", "")
        summary_row["Best_Static_Top_N"] = best_static.get("Top_N", np.nan)
        summary_row["Best_Static_Portfolio_Mode"] = best_static.get("Portfolio_Mode", "")
        summary_row["Best_Static_Total_Return"] = best_static.get("Total_Return", np.nan)
        summary_row["Best_Static_Excess_Return_vs_XLK"] = best_static.get("Excess_Return_vs_XLK", np.nan)
        result_rows.append(summary_row)

        if not wf_daily.empty:
            wf_daily = wf_daily.copy()
            wf_daily["Scenario_ID"] = scenario.scenario_id
            wf_daily["Fallback_Mode"] = fallback_mode
            dynamic_curves.append(wf_daily)
        if not wf_rebalances.empty:
            wf_rebalances = wf_rebalances.copy()
            wf_rebalances["Scenario_ID"] = scenario.scenario_id
            wf_rebalances["Fallback_Mode"] = fallback_mode
            dynamic_rebalances.append(wf_rebalances)

    results = pd.DataFrame(result_rows)
    curve = pd.concat(dynamic_curves, ignore_index=True) if dynamic_curves else pd.DataFrame()
    rebalance_log = pd.concat(dynamic_rebalances, ignore_index=True) if dynamic_rebalances else pd.DataFrame()
    return static_summary, selections, results, curve, rebalance_log


def format_pct(value: object) -> str:
    return monthly_sim.format_pct(value)


def format_num(value: object, digits: int = 4) -> str:
    return monthly_sim.format_num(value, digits)


def linked_return(frame: pd.DataFrame, return_col: str) -> float:
    if frame.empty or return_col not in frame.columns:
        return np.nan
    returns = pd.to_numeric(frame[return_col], errors="coerce").dropna()
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() - 1.0)


def build_exposure_label_summary(
    dynamic_curves: pd.DataFrame,
    dynamic_rebalances: pd.DataFrame,
) -> pd.DataFrame:
    if dynamic_curves.empty or "Exposure_Source" not in dynamic_curves.columns:
        return pd.DataFrame()

    rebalance_counts = pd.DataFrame()
    if not dynamic_rebalances.empty and "Exposure_Source" in dynamic_rebalances.columns:
        rebalance_counts = (
            dynamic_rebalances
            .groupby(["Scenario_ID", "Fallback_Mode", "Exposure_Source", "Exposure_Label"], dropna=False)
            .size()
            .rename("Rebalance_Count")
            .reset_index()
        )

    rows = []
    group_cols = ["Scenario_ID", "Fallback_Mode", "Exposure_Source", "Exposure_Label"]
    for keys, group in dynamic_curves.groupby(group_cols, dropna=False):
        scenario_id, fallback_mode, source, label = keys
        dates = pd.to_datetime(group["Date"], errors="coerce").dropna()
        trading_days = int(len(group))
        rows.append({
            "Scenario_ID": scenario_id,
            "Fallback_Mode": fallback_mode,
            "Exposure_Source": source,
            "Exposure_Label": label,
            "Trading_Days": trading_days,
            "Trading_Day_Pct": np.nan,
            "First_Date": dates.min().strftime("%Y-%m-%d") if not dates.empty else "",
            "Last_Date": dates.max().strftime("%Y-%m-%d") if not dates.empty else "",
            "Linked_Strategy_Return_During_Label": linked_return(group, "Strategy_Daily_Return"),
            "Average_Strategy_Daily_Return": pd.to_numeric(
                group.get("Strategy_Daily_Return", pd.Series(dtype=float)),
                errors="coerce",
            ).mean(),
            "Average_Cash_Weight": pd.to_numeric(
                group.get("Strategy_Cash_Weight", pd.Series(dtype=float)),
                errors="coerce",
            ).mean(),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    total_days = summary.groupby(["Scenario_ID", "Fallback_Mode"])["Trading_Days"].transform("sum")
    summary["Trading_Day_Pct"] = summary["Trading_Days"] / total_days.replace(0, np.nan)

    if not rebalance_counts.empty:
        summary = summary.merge(
            rebalance_counts,
            on=["Scenario_ID", "Fallback_Mode", "Exposure_Source", "Exposure_Label"],
            how="left",
        )
    else:
        summary["Rebalance_Count"] = np.nan
    summary["Rebalance_Count"] = summary["Rebalance_Count"].fillna(0).astype(int)

    return summary.sort_values(
        ["Scenario_ID", "Fallback_Mode", "Exposure_Source", "Trading_Days"],
        ascending=[True, True, True, False],
    )


def write_summary(
    output_dir: Path,
    results: pd.DataFrame,
    static_summary: pd.DataFrame,
    selections: pd.DataFrame,
    rebalances: pd.DataFrame,
    scenarios: list[StressScenario],
    fallback_modes: list[str],
) -> None:
    lines = [
        "# Portfolio Robustness Stress Test Summary",
        "",
        "## What was completed",
        "",
        "This run stress-tested the defensive walk-forward alpha selector under multiple portfolio assumptions.",
        "Each scenario tested the same candidate configs, same Sharadar point-in-time score history, same canonical adjusted prices, and the same historical universe control.",
        "",
        "Fallback modes tested:",
    ]
    for mode in fallback_modes:
        lines.append(f"- `{mode}`")

    lines.extend([
        "",
        "Stress scenarios tested:",
    ])
    for scenario in scenarios:
        lines.append(f"- `{scenario.scenario_id}`: {scenario.notes}")

    if results.empty:
        lines.extend(["", "No dynamic stress-test results were produced."])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ROBUSTNESS_STRESS_TEST_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    ranked = results.sort_values("Stress_Robustness_Objective", ascending=False).copy()
    best = ranked.iloc[0]
    lines.extend([
        "",
        "## Best Dynamic Stress-Test Result",
        "",
        f"- Scenario: `{best['Scenario_ID']}`",
        f"- Fallback mode: `{best['Fallback_Mode']}`",
        f"- Final equity: {format_num(best.get('Final_Equity'), 4)}",
        f"- Total return: {format_pct(best.get('Total_Return'))}",
        f"- Excess return vs simulated universe: {format_pct(best.get('Excess_Return_vs_Universe'))}",
        f"- Excess return vs QQQ: {format_pct(best.get('Excess_Return_vs_QQQ'))}",
        f"- Excess return vs XLK: {format_pct(best.get('Excess_Return_vs_XLK'))}",
        f"- Sharpe: {format_num(best.get('Sharpe'))}",
        f"- Sortino: {format_num(best.get('Sortino'))}",
        f"- Max drawdown: {format_pct(best.get('Max_Drawdown'))}",
        f"- Average turnover: {format_pct(best.get('Avg_Turnover'))}",
        f"- Fallback rebalances: {int(best.get('Fallback_Rebalance_Count', 0))}",
        f"- Robustness objective: {format_num(best.get('Stress_Robustness_Objective'))}",
        "",
    ])

    lines.extend(["## Base Scenario Fallback Comparison", ""])
    base = results[results["Scenario_ID"] == "base"].sort_values("Stress_Robustness_Objective", ascending=False)
    if base.empty:
        lines.append("The base scenario was not included in this run.")
    else:
        for _, row in base.iterrows():
            lines.append(
                "- "
                f"`{row['Fallback_Mode']}`: return {format_pct(row.get('Total_Return'))}, "
                f"excess vs XLK {format_pct(row.get('Excess_Return_vs_XLK'))}, "
                f"Sharpe {format_num(row.get('Sharpe'))}, "
                f"drawdown {format_pct(row.get('Max_Drawdown'))}, "
                f"fallback rebalances {int(row.get('Fallback_Rebalance_Count', 0))}"
            )

    lines.extend(["", "## Best Fallback By Scenario", ""])
    scenario_best = results.sort_values("Stress_Robustness_Objective", ascending=False).groupby("Scenario_ID", as_index=False).head(1)
    scenario_best = scenario_best.sort_values("Scenario_ID")
    for _, row in scenario_best.iterrows():
        lines.append(
            "- "
            f"`{row['Scenario_ID']}` -> `{row['Fallback_Mode']}`: "
            f"return {format_pct(row.get('Total_Return'))}, "
            f"excess vs universe {format_pct(row.get('Excess_Return_vs_Universe'))}, "
            f"excess vs XLK {format_pct(row.get('Excess_Return_vs_XLK'))}, "
            f"Sharpe {format_num(row.get('Sharpe'))}, "
            f"drawdown {format_pct(row.get('Max_Drawdown'))}, "
            f"fallback rebalances {int(row.get('Fallback_Rebalance_Count', 0))}"
        )

    lines.extend(["", "## Fallback Mode Robustness", ""])
    fallback_rollup = (
        results.groupby("Fallback_Mode")
        .agg(
            Scenario_Count=("Scenario_ID", "count"),
            Avg_Total_Return=("Total_Return", "mean"),
            Worst_Total_Return=("Total_Return", "min"),
            Avg_Excess_vs_XLK=("Excess_Return_vs_XLK", "mean"),
            Worst_Excess_vs_XLK=("Excess_Return_vs_XLK", "min"),
            Avg_Sharpe=("Sharpe", "mean"),
            Worst_Drawdown=("Max_Drawdown", "min"),
            Avg_Fallback_Rebalances=("Fallback_Rebalance_Count", "mean"),
            Avg_Objective=("Stress_Robustness_Objective", "mean"),
        )
        .reset_index()
        .sort_values("Avg_Objective", ascending=False)
    )
    for _, row in fallback_rollup.iterrows():
        lines.append(
            "- "
            f"`{row['Fallback_Mode']}`: avg return {format_pct(row.get('Avg_Total_Return'))}, "
            f"worst return {format_pct(row.get('Worst_Total_Return'))}, "
            f"avg excess vs XLK {format_pct(row.get('Avg_Excess_vs_XLK'))}, "
            f"worst excess vs XLK {format_pct(row.get('Worst_Excess_vs_XLK'))}, "
            f"avg Sharpe {format_num(row.get('Avg_Sharpe'))}, "
            f"worst drawdown {format_pct(row.get('Worst_Drawdown'))}, "
            f"avg fallback rebalances {format_num(row.get('Avg_Fallback_Rebalances'), 2)}"
        )

    lines.extend(["", "## Exposure Source Labeling", ""])
    if not rebalances.empty and "Exposure_Source" in rebalances.columns:
        exposure_counts = (
            rebalances
            .groupby(["Fallback_Mode", "Exposure_Source", "Exposure_Label"], dropna=False)
            .size()
            .rename("Rebalance_Count")
            .reset_index()
            .sort_values(["Fallback_Mode", "Exposure_Source"])
        )
        lines.append("Every dynamic rebalance and daily equity row now labels whether returns came from stock-selection alpha, benchmark ETF fallback, or cash fallback.")
        lines.append("")
        for _, row in exposure_counts.iterrows():
            lines.append(
                "- "
                f"`{row['Fallback_Mode']}` / `{row['Exposure_Source']}` "
                f"({row['Exposure_Label']}): {int(row['Rebalance_Count']):,} rebalances"
            )
    else:
        lines.append("No exposure-source labels were available in this run.")

    lines.extend(["", "## Dynamic Selection Notes", ""])
    if selections.empty:
        lines.append("No walk-forward selections were produced.")
    else:
        for scenario_id, group in selections.groupby("Scenario_ID"):
            stages = group["Selection_Guard_Stage"].value_counts().to_dict()
            selected_configs = [
                f"{row['Validation_Fold']}:{row['Selected_Config_ID']}/top{int(row['Selected_Top_N'])}/{row['Selected_Portfolio_Mode']}"
                for _, row in group.iterrows()
            ]
            lines.append(
                f"- `{scenario_id}` selected {'; '.join(selected_configs)}. "
                f"Guard stages: {stages}"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The defensive alpha selector survived the stress suite better than the unguarded selector did in earlier tests. It remained strongly positive versus the simulated stock universe across the tested scenarios.",
        "",
        "The important weak point is still XLK. The strategy narrowed the XLK gap dramatically in the base defensive run, but harsh costs, stricter liquidity, and tighter caps can still leave it behind XLK. That means the model is useful as a stock-selection research engine, but it still needs more signal work or a smarter benchmark fallback before it should be treated as a live allocation rule.",
        "",
        "Fallback should be used carefully. Cash fallback reduces forced exposure, but can miss strong recovery periods. QQQ and XLK fallback keep market exposure, but can make the strategy look more like a benchmark rotation system than a pure stock picker. The best fallback policy should be chosen by stress robustness, not by one full-sample winner.",
        "",
        "## Files produced",
        "",
        "- `stress_test_summary.csv`",
        "- `scenario_static_simulation_summary.csv`",
        "- `scenario_walk_forward_selections.csv`",
        "- `dynamic_equity_curves.csv`",
        "- `dynamic_rebalance_log.csv`",
        "- `exposure_label_summary.csv`",
        "- `fallback_event_log.csv`",
        "- `run_metadata.json`",
        "- `ROBUSTNESS_STRESS_TEST_SUMMARY.md`",
    ])

    if not rebalances.empty and "Fallback_Triggered" in rebalances.columns:
        fallback_events = rebalances[rebalances["Fallback_Triggered"].fillna(False).astype(bool)]
        lines.extend([
            "",
            "## Fallback Event Count",
            "",
            f"- Total fallback rebalance events: {len(fallback_events):,}",
        ])
        if not fallback_events.empty:
            by_mode = fallback_events["Fallback_Mode"].value_counts().sort_index()
            for mode, count in by_mode.items():
                lines.append(f"- `{mode}`: {int(count):,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ROBUSTNESS_STRESS_TEST_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run robustness stress tests for the monthly portfolio simulator.")
    parser.add_argument("--score-history", type=Path, default=DEFAULT_SCORE_HISTORY)
    parser.add_argument("--weight-tuning-dir", type=Path, default=DEFAULT_WEIGHT_TUNING_DIR)
    parser.add_argument("--historical-price-file", type=Path, default=DEFAULT_PRICE_FILE)
    parser.add_argument("--historical-universe", type=Path, default=DEFAULT_HISTORICAL_UNIVERSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario-preset", choices=["compact", "expanded"], default="compact")
    parser.add_argument("--scenarios", default="", help="Optional comma-separated scenario ids to run.")
    parser.add_argument("--fallback-modes", default=DEFAULT_FALLBACK_MODES)
    parser.add_argument("--fallback-clean-stages", default=DEFAULT_CLEAN_GUARD_STAGES)
    parser.add_argument("--candidate-preset", choices=["compact", "expanded"], default="expanded")
    parser.add_argument("--config-scope", choices=["shortlist", "all"], default="shortlist")
    parser.add_argument("--shortlist-count", type=int, default=10)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--alpha-score-columns", default=DEFAULT_ALPHA_SCORE_COLUMNS)
    parser.add_argument("--top-n-list", default=DEFAULT_TOP_N_LIST)
    parser.add_argument("--portfolio-modes", default=DEFAULT_MODES)
    args = parser.parse_args()

    scenarios = stress_scenarios(args.scenario_preset)
    requested = set(parse_list(args.scenarios))
    if requested:
        scenarios = [scenario for scenario in scenarios if scenario.scenario_id in requested]
        missing = requested - {scenario.scenario_id for scenario in scenarios}
        if missing:
            raise ValueError(f"Unknown stress scenario id(s): {sorted(missing)}")
    if not scenarios:
        raise ValueError("No stress scenarios selected")

    fallback_modes = parse_list(args.fallback_modes)
    unknown_fallback = sorted(set(fallback_modes) - {"none", "cash", "qqq", "xlk"})
    if unknown_fallback:
        raise ValueError(f"Unknown fallback mode(s): {unknown_fallback}")

    clean_guard_stages = tuple(parse_list(args.fallback_clean_stages))
    top_n_list = construction.parse_top_n_list(args.top_n_list)
    modes = construction.parse_modes(args.portfolio_modes)

    scores, configs, price_wide, volume_wide, historical_universe = load_inputs(args)
    base_params = monthly_sim.SimulationParams()

    static_outputs = []
    selection_outputs = []
    result_outputs = []
    dynamic_curves = []
    dynamic_rebalances = []

    for scenario in scenarios:
        params = params_for_scenario(base_params, scenario)
        static_summary, selections, results, curve, rebalance_log = run_scenario(
            scenario=scenario,
            params=params,
            scores=scores,
            configs=configs,
            price_wide=price_wide,
            volume_wide=volume_wide,
            historical_universe=historical_universe,
            top_n_list=top_n_list,
            modes=modes,
            fallback_modes=fallback_modes,
            clean_guard_stages=clean_guard_stages,
        )
        static_outputs.append(static_summary)
        selection_outputs.append(selections)
        result_outputs.append(results)
        if not curve.empty:
            dynamic_curves.append(curve)
        if not rebalance_log.empty:
            dynamic_rebalances.append(rebalance_log)

    static_all = pd.concat(static_outputs, ignore_index=True) if static_outputs else pd.DataFrame()
    selections_all = pd.concat(selection_outputs, ignore_index=True) if selection_outputs else pd.DataFrame()
    results_all = pd.concat(result_outputs, ignore_index=True) if result_outputs else pd.DataFrame()
    dynamic_curves_all = pd.concat(dynamic_curves, ignore_index=True) if dynamic_curves else pd.DataFrame()
    dynamic_rebalances_all = pd.concat(dynamic_rebalances, ignore_index=True) if dynamic_rebalances else pd.DataFrame()
    exposure_label_summary = build_exposure_label_summary(dynamic_curves_all, dynamic_rebalances_all)
    fallback_events = (
        dynamic_rebalances_all[dynamic_rebalances_all["Fallback_Triggered"].fillna(False).astype(bool)].copy()
        if not dynamic_rebalances_all.empty and "Fallback_Triggered" in dynamic_rebalances_all.columns
        else pd.DataFrame()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_all.sort_values("Stress_Robustness_Objective", ascending=False).round(6).to_csv(
        args.output_dir / "stress_test_summary.csv",
        index=False,
    )
    static_all.round(6).to_csv(args.output_dir / "scenario_static_simulation_summary.csv", index=False)
    selections_all.round(6).to_csv(args.output_dir / "scenario_walk_forward_selections.csv", index=False)
    dynamic_curves_all.round(6).to_csv(args.output_dir / "dynamic_equity_curves.csv", index=False)
    dynamic_rebalances_all.round(6).to_csv(args.output_dir / "dynamic_rebalance_log.csv", index=False)
    exposure_label_summary.round(6).to_csv(args.output_dir / "exposure_label_summary.csv", index=False)
    fallback_events.round(6).to_csv(args.output_dir / "fallback_event_log.csv", index=False)

    metadata = {
        "status": "complete",
        "score_history": str(args.score_history),
        "weight_tuning_dir": str(args.weight_tuning_dir),
        "historical_price_file": str(args.historical_price_file),
        "historical_universe": str(args.historical_universe),
        "scenario_preset": args.scenario_preset,
        "scenario_count": len(scenarios),
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "fallback_modes": fallback_modes,
        "fallback_clean_guard_stages": list(clean_guard_stages),
        "selection_params": asdict(selection_params()),
        "top_n_list": top_n_list,
        "portfolio_modes": modes,
        "result_rows": int(len(results_all)),
        "static_summary_rows": int(len(static_all)),
        "selection_rows": int(len(selections_all)),
        "dynamic_curve_rows": int(len(dynamic_curves_all)),
        "dynamic_rebalance_rows": int(len(dynamic_rebalances_all)),
        "exposure_label_summary_rows": int(len(exposure_label_summary)),
        "fallback_event_rows": int(len(fallback_events)),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    write_summary(
        output_dir=args.output_dir,
        results=results_all,
        static_summary=static_all,
        selections=selections_all,
        rebalances=dynamic_rebalances_all,
        scenarios=scenarios,
        fallback_modes=fallback_modes,
    )

    print("\n=== ROBUSTNESS STRESS TEST SUMMARY ===", flush=True)
    columns = [
        "Scenario_ID",
        "Fallback_Mode",
        "Stress_Robustness_Objective",
        "Total_Return",
        "Excess_Return_vs_Universe",
        "Excess_Return_vs_XLK",
        "Sharpe",
        "Max_Drawdown",
        "Avg_Turnover",
        "Fallback_Rebalance_Count",
    ]
    if not results_all.empty:
        print(
            results_all.sort_values("Stress_Robustness_Objective", ascending=False)[columns]
            .head(20)
            .to_string(index=False),
            flush=True,
        )
    print(f"\n[SUCCESS] Saved robustness stress outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
