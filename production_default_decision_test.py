#!/usr/bin/env python3
"""
Production-default decision harness for the Sharadar-backed stock model.

This script answers three linked questions before live promotion:
1. Should $25M minimum 20-day dollar volume become the default liquidity floor?
2. Should ConfidenceValueSubRelRisk become the stable alpha family?
3. Which XLK fallback trigger policy is best supported by walk-forward evidence?
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import candidate_family_liquidity_stability_test as liquidity_test
import combined_weight_tuning as weight_tuning
import monthly_rebalanced_portfolio_simulator as monthly_sim
import portfolio_construction_validation as construction
import portfolio_robustness_stress_test as robustness


DEFAULT_SCORE_HISTORY = Path("backtests/combined_score_sharadar/point_in_time_scores.csv")
DEFAULT_WEIGHT_TUNING_DIR = Path("backtests/weight_tuning_sharadar")
DEFAULT_PRICE_FILE = Path("data/historical_data_layer/prices_daily.csv")
DEFAULT_HISTORICAL_UNIVERSE = Path("data/historical_tech_universe.csv")
DEFAULT_OUTPUT_DIR = Path("backtests/production_default_decision_sharadar")
DEFAULT_LIQUIDITY_FLOORS = "5000000,15000000,20000000,25000000,30000000,40000000,50000000,100000000"
DEFAULT_TOP_N_LIST = "10,20,50,100"
DEFAULT_MODES = "equal,rank_weighted"

CONFIDENCE_VALUE_FAMILY = "alpha_Alpha_Prototype_ConfidenceValueSubRelRisk"
XLK_COMPETITIVE_FAMILY = "alpha_Alpha_Prototype_XLKCompetitive"
BENCHMARK_AWARE_FAMILY = "alpha_Alpha_Prototype_BenchmarkAwareQuality"
CONFIDENCE_FAIR_VALUE_FAMILY = "alpha_Confidence_Adjusted_Fair_Value_RankPct"

STRICT_STAGE = "family_stable_strict_recent_xlk_drawdown_downside"
RECENT_DEFENSIVE_STAGE = "family_stable_recent_defensive_downside"
BENCHMARK_STAGE = "family_stable_benchmark_drawdown_guard"
RELAXED_STAGE = "family_stable_relaxed_drawdown_turnover"
LEGACY_STRICT_STAGE = "strict_recent_xlk_drawdown_downside"
LEGACY_RECENT_DEFENSIVE_STAGE = "recent_defensive_downside"

STRICT_CLEAN = (STRICT_STAGE, LEGACY_STRICT_STAGE)
DEFENSIVE_CLEAN = (
    STRICT_STAGE,
    RECENT_DEFENSIVE_STAGE,
    LEGACY_STRICT_STAGE,
    LEGACY_RECENT_DEFENSIVE_STAGE,
)
BENCHMARK_OK_CLEAN = (
    STRICT_STAGE,
    RECENT_DEFENSIVE_STAGE,
    BENCHMARK_STAGE,
    LEGACY_STRICT_STAGE,
    LEGACY_RECENT_DEFENSIVE_STAGE,
)
RELAXED_OK_CLEAN = (
    STRICT_STAGE,
    RECENT_DEFENSIVE_STAGE,
    BENCHMARK_STAGE,
    RELAXED_STAGE,
    LEGACY_STRICT_STAGE,
    LEGACY_RECENT_DEFENSIVE_STAGE,
)


@dataclass(frozen=True)
class DecisionPolicy:
    policy_id: str
    notes: str
    candidate_families: tuple[str, ...]
    fallback_mode: str
    fallback_trigger_policy: str
    clean_guard_stages: tuple[str, ...]


def decision_policies() -> list[DecisionPolicy]:
    return [
        DecisionPolicy(
            policy_id="dynamic_no_fallback",
            notes="Let the family-stable selector choose across all candidate families; never fallback.",
            candidate_families=(),
            fallback_mode="none",
            fallback_trigger_policy="none",
            clean_guard_stages=RELAXED_OK_CLEAN,
        ),
        DecisionPolicy(
            policy_id="dynamic_xlk_relaxed_only",
            notes="Dynamic family selection; fallback to XLK only when the selector reaches relaxed or fallback-all stages.",
            candidate_families=(),
            fallback_mode="xlk",
            fallback_trigger_policy="fallback_on_relaxed_or_weaker",
            clean_guard_stages=BENCHMARK_OK_CLEAN,
        ),
        DecisionPolicy(
            policy_id="dynamic_xlk_benchmark_or_relaxed",
            notes="Dynamic family selection; fallback to XLK when the selector cannot clear strict/recent defensive standards.",
            candidate_families=(),
            fallback_mode="xlk",
            fallback_trigger_policy="fallback_on_benchmark_or_weaker",
            clean_guard_stages=DEFENSIVE_CLEAN,
        ),
        DecisionPolicy(
            policy_id="dynamic_xlk_strict_only",
            notes="Dynamic family selection; fallback to XLK unless the strict family-stable stage clears.",
            candidate_families=(),
            fallback_mode="xlk",
            fallback_trigger_policy="fallback_on_any_non_strict_stage",
            clean_guard_stages=STRICT_CLEAN,
        ),
        DecisionPolicy(
            policy_id="confidence_family_no_fallback",
            notes="Lock the selector to ConfidenceValueSubRelRisk; tune top-N/allocation by walk-forward only.",
            candidate_families=(CONFIDENCE_VALUE_FAMILY,),
            fallback_mode="none",
            fallback_trigger_policy="none",
            clean_guard_stages=RELAXED_OK_CLEAN,
        ),
        DecisionPolicy(
            policy_id="confidence_family_xlk_relaxed_only",
            notes="Lock to ConfidenceValueSubRelRisk; fallback to XLK only on relaxed or fallback-all stages.",
            candidate_families=(CONFIDENCE_VALUE_FAMILY,),
            fallback_mode="xlk",
            fallback_trigger_policy="fallback_on_relaxed_or_weaker",
            clean_guard_stages=BENCHMARK_OK_CLEAN,
        ),
        DecisionPolicy(
            policy_id="confidence_family_xlk_benchmark_or_relaxed",
            notes="Lock to ConfidenceValueSubRelRisk; fallback to XLK when strict/recent defensive standards fail.",
            candidate_families=(CONFIDENCE_VALUE_FAMILY,),
            fallback_mode="xlk",
            fallback_trigger_policy="fallback_on_benchmark_or_weaker",
            clean_guard_stages=DEFENSIVE_CLEAN,
        ),
        DecisionPolicy(
            policy_id="confidence_family_xlk_strict_only",
            notes="Lock to ConfidenceValueSubRelRisk; fallback to XLK unless strict family-stable standards clear.",
            candidate_families=(CONFIDENCE_VALUE_FAMILY,),
            fallback_mode="xlk",
            fallback_trigger_policy="fallback_on_any_non_strict_stage",
            clean_guard_stages=STRICT_CLEAN,
        ),
        DecisionPolicy(
            policy_id="xlkcompetitive_family_no_fallback",
            notes="Lock the selector to XLKCompetitive; tune top-N/allocation by walk-forward only.",
            candidate_families=(XLK_COMPETITIVE_FAMILY,),
            fallback_mode="none",
            fallback_trigger_policy="none",
            clean_guard_stages=RELAXED_OK_CLEAN,
        ),
        DecisionPolicy(
            policy_id="benchmarkaware_family_no_fallback",
            notes="Lock the selector to BenchmarkAwareQuality; tune top-N/allocation by walk-forward only.",
            candidate_families=(BENCHMARK_AWARE_FAMILY,),
            fallback_mode="none",
            fallback_trigger_policy="none",
            clean_guard_stages=RELAXED_OK_CLEAN,
        ),
        DecisionPolicy(
            policy_id="confidence_fair_value_family_no_fallback",
            notes="Lock the selector to confidence-adjusted fair value; tune top-N/allocation by walk-forward only.",
            candidate_families=(CONFIDENCE_FAIR_VALUE_FAMILY,),
            fallback_mode="none",
            fallback_trigger_policy="none",
            clean_guard_stages=RELAXED_OK_CLEAN,
        ),
    ]


def parse_list(raw: str) -> list[str]:
    return robustness.parse_list(raw)


def candidate_config_ids(
    configs: list[weight_tuning.WeightConfig | monthly_sim.DirectScoreConfig],
    families: tuple[str, ...],
) -> list[str]:
    if not families:
        return [config.config_id for config in configs]
    allowed = set(families)
    return [
        config.config_id
        for config in configs
        if monthly_sim.candidate_family_id(config.config_id) in allowed
    ]


def filter_by_config_ids(frame: pd.DataFrame, config_ids: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[frame["Config_ID"].isin(config_ids)].copy()


def value_counts_string(frame: pd.DataFrame, column: str) -> str:
    return liquidity_test.value_counts_string(frame, column)


def metric(row: pd.Series | dict, name: str, default: float = 0.0) -> float:
    raw = row.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def production_default_objective(row: pd.Series | dict) -> float:
    fallback_pct = metric(row, "Fallback_Triggered_Pct")
    unique_family_count = metric(row, "Unique_Selected_Family_Count", 1.0)
    clean_rate = metric(row, "Clean_Selection_Rate")
    return (
        0.95 * metric(row, "Excess_Return_vs_XLK")
        + 0.45 * metric(row, "Excess_Return_vs_Universe")
        + 0.12 * metric(row, "Sharpe")
        + 0.08 * clean_rate
        - 0.70 * abs(metric(row, "Max_Drawdown"))
        - 0.05 * metric(row, "Avg_Turnover")
        - 0.20 * fallback_pct
        - 0.03 * max(0.0, unique_family_count - 1.0)
    )


def run_simulate_all(
    output_dir: Path,
    liquidity_floor: float,
    scores: pd.DataFrame,
    configs: list[weight_tuning.WeightConfig | monthly_sim.DirectScoreConfig],
    top_n_list: list[int],
    modes: list[str],
    price_wide: pd.DataFrame,
    adv20: pd.DataFrame,
    params: monthly_sim.SimulationParams,
    historical_universe: monthly_sim.HistoricalUniverse,
    quiet: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not quiet:
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
        return daily, rebalances

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "simulation_progress.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[INFO] Liquidity floor ${liquidity_floor:,.0f}\n")
        with contextlib.redirect_stdout(log_file):
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
    return daily, rebalances


def run_policy(
    policy: DecisionPolicy,
    liquidity_floor: float,
    scores: pd.DataFrame,
    configs: list[weight_tuning.WeightConfig | monthly_sim.DirectScoreConfig],
    price_wide: pd.DataFrame,
    adv20: pd.DataFrame,
    params: monthly_sim.SimulationParams,
    historical_universe: monthly_sim.HistoricalUniverse,
    daily: pd.DataFrame,
    rebalances: pd.DataFrame,
    selector: monthly_sim.WalkForwardSelectionParams,
) -> tuple[dict, pd.DataFrame]:
    config_ids = candidate_config_ids(configs, policy.candidate_families)
    policy_daily = filter_by_config_ids(daily, config_ids)
    policy_rebalances = filter_by_config_ids(rebalances, config_ids)
    selections = monthly_sim.walk_forward_selection_from_equity(
        daily=policy_daily,
        rebalances=policy_rebalances,
        selection_params=selector,
    ).copy()

    if selections.empty:
        return {}, selections

    fallback_policy = monthly_sim.FallbackPolicy(
        mode=policy.fallback_mode,
        clean_guard_stages=policy.clean_guard_stages,
        exposure=1.0,
    )
    _, wf_rebalances, wf_summary = monthly_sim.simulate_walk_forward_dynamic_portfolio(
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
        return {}, selections

    selection_rounds = int(len(selections))
    clean_rounds = int(selections["Selection_Guard_Stage"].isin(policy.clean_guard_stages).sum())
    clean_selection_rate = clean_rounds / selection_rounds if selection_rounds else np.nan
    stage_counts = value_counts_string(selections, "Selection_Guard_Stage")
    has_fallback_all_selection = bool(
        "Selection_Guard_Stage" in selections.columns
        and selections["Selection_Guard_Stage"].astype(str).eq("fallback_all").any()
    )
    row = wf_summary.iloc[0].to_dict()
    row.update({
        "Policy_ID": policy.policy_id,
        "Policy_Notes": policy.notes,
        "Liquidity_Floor": liquidity_floor,
        "Candidate_Family_Filter": ",".join(policy.candidate_families) if policy.candidate_families else "dynamic_all",
        "Fallback_Mode": policy.fallback_mode,
        "Fallback_Trigger_Policy": policy.fallback_trigger_policy,
        "Clean_Guard_Stages": ",".join(policy.clean_guard_stages),
        "Selection_Rounds": selection_rounds,
        "Clean_Selection_Rounds": clean_rounds,
        "Clean_Selection_Rate": clean_selection_rate,
        "Fallback_Would_Trigger_Rounds": (
            max(0, selection_rounds - clean_rounds)
            if policy.fallback_mode != "none"
            else 0
        ),
        "Selected_Family_Counts": value_counts_string(selections, "Selected_Candidate_Family_ID"),
        "Selected_Config_Counts": value_counts_string(selections, "Selected_Config_ID"),
        "Selection_Guard_Stage_Counts": stage_counts,
        "Has_Fallback_All_Selection": has_fallback_all_selection,
        "Production_Clean_Qualified": (
            policy.fallback_mode == "none"
            and selection_rounds > 0
            and clean_rounds == selection_rounds
            and not has_fallback_all_selection
        ),
        "Unique_Selected_Family_Count": (
            int(selections["Selected_Candidate_Family_ID"].nunique())
            if "Selected_Candidate_Family_ID" in selections.columns
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
    })
    row["Production_Default_Objective"] = production_default_objective(row)
    selections["Policy_ID"] = policy.policy_id
    selections["Policy_Notes"] = policy.notes
    selections["Candidate_Family_Filter"] = row["Candidate_Family_Filter"]
    selections["Fallback_Mode"] = policy.fallback_mode
    selections["Fallback_Trigger_Policy"] = policy.fallback_trigger_policy
    selections["Clean_Guard_Stages"] = row["Clean_Guard_Stages"]
    selections["Liquidity_Floor"] = liquidity_floor
    return row, selections


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
    policies: list[DecisionPolicy],
    selector: monthly_sim.WalkForwardSelectionParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = liquidity_test.base_simulation_params(args, liquidity_floor)
    print(f"[INFO] Decision floor ${liquidity_floor:,.0f}", flush=True)
    adv20 = monthly_sim.rolling_dollar_volume(price_wide, volume_wide, params.adv_window)
    daily, rebalances = run_simulate_all(
        output_dir=args.output_dir,
        liquidity_floor=liquidity_floor,
        scores=scores,
        configs=configs,
        top_n_list=top_n_list,
        modes=modes,
        price_wide=price_wide,
        adv20=adv20,
        params=params,
        historical_universe=historical_universe,
        quiet=args.quiet_simulation_logs,
    )

    end_date = pd.to_datetime(price_wide.index.max()).normalize()
    daily = monthly_sim.assign_fold_labels(daily, "Date", end_date)
    rebalances = monthly_sim.assign_fold_labels(rebalances, "Trade_Date", end_date)
    static_summary = monthly_sim.summarize_simulation(daily, rebalances).copy()
    if not static_summary.empty:
        static_summary["Liquidity_Floor"] = liquidity_floor

    result_rows: list[dict] = []
    selection_frames: list[pd.DataFrame] = []
    for policy in policies:
        row, selections = run_policy(
            policy=policy,
            liquidity_floor=liquidity_floor,
            scores=scores,
            configs=configs,
            price_wide=price_wide,
            adv20=adv20,
            params=params,
            historical_universe=historical_universe,
            daily=daily,
            rebalances=rebalances,
            selector=selector,
        )
        if row:
            result_rows.append(row)
        if not selections.empty:
            selection_frames.append(selections)

    dynamic_summary = pd.DataFrame(result_rows)
    selections_all = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    return static_summary, selections_all, dynamic_summary


def format_pct(value: object) -> str:
    return monthly_sim.format_pct(value)


def format_num(value: object, digits: int = 4) -> str:
    return monthly_sim.format_num(value, digits)


def first_row(frame: pd.DataFrame, sort_col: str = "Production_Default_Objective") -> dict:
    if frame.empty:
        return {}
    return frame.sort_values(sort_col, ascending=False).iloc[0].to_dict()


def production_qualified_no_fallback(dynamic_summary: pd.DataFrame) -> pd.DataFrame:
    if dynamic_summary.empty:
        return dynamic_summary.copy()
    clean_rate = pd.to_numeric(
        dynamic_summary.get("Clean_Selection_Rate", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    fallback_all = (
        dynamic_summary.get("Selection_Guard_Stage_Counts", pd.Series(dtype=str))
        .astype(str)
        .str.contains("fallback_all", regex=False, na=False)
    )
    return dynamic_summary[
        (dynamic_summary["Fallback_Mode"] == "none")
        & (clean_rate >= 0.999)
        & ~fallback_all
    ].copy()


def write_summary(
    output_dir: Path,
    dynamic_summary: pd.DataFrame,
    selections: pd.DataFrame,
    static_summary: pd.DataFrame,
    policies: list[DecisionPolicy],
    liquidity_floors: list[float],
) -> None:
    lines = [
        "# Production Default Decision Summary",
        "",
        "## What Was Tested",
        "",
        "This pass used the Sharadar-backed point-in-time score history and canonical prices to decide three production-default questions:",
        "",
        "- whether `$25M` should become the live research liquidity floor",
        "- whether `Alpha_Prototype_ConfidenceValueSubRelRisk` should become the stable alpha family",
        "- when XLK fallback should trigger",
        "",
        "Liquidity floors tested:",
    ]
    for floor in liquidity_floors:
        lines.append(f"- ${floor:,.0f}")
    lines.extend(["", "Policies tested:"])
    for policy in policies:
        families = ",".join(policy.candidate_families) if policy.candidate_families else "dynamic_all"
        lines.append(f"- `{policy.policy_id}`: {families}; fallback `{policy.fallback_mode}`; trigger `{policy.fallback_trigger_policy}`")

    if dynamic_summary.empty:
        lines.extend(["", "No dynamic decision rows were produced."])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "PRODUCTION_DEFAULT_DECISION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    best = first_row(dynamic_summary)
    best_no_fallback = first_row(dynamic_summary[dynamic_summary["Fallback_Mode"] == "none"])
    best_xlk = first_row(dynamic_summary[dynamic_summary["Fallback_Mode"] == "xlk"])
    qualified_no_fallback = production_qualified_no_fallback(dynamic_summary)
    best_qualified_no_fallback = first_row(qualified_no_fallback)
    best_qualified_fixed = first_row(
        qualified_no_fallback[qualified_no_fallback["Candidate_Family_Filter"] != "dynamic_all"]
    )
    confidence_rows = dynamic_summary[
        dynamic_summary["Candidate_Family_Filter"].astype(str).str.contains(CONFIDENCE_VALUE_FAMILY, regex=False)
    ]
    best_confidence = first_row(confidence_rows)
    confidence_no_fallback = confidence_rows[confidence_rows["Fallback_Mode"] == "none"]
    best_confidence_no_fallback = first_row(confidence_no_fallback)
    dynamic_rows = dynamic_summary[dynamic_summary["Candidate_Family_Filter"] == "dynamic_all"]
    best_dynamic = first_row(dynamic_rows)
    floor_25 = dynamic_summary[dynamic_summary["Liquidity_Floor"] == 25_000_000.0]
    best_25 = first_row(floor_25)

    lines.extend([
        "",
        "## Best Overall Row",
        "",
        f"- Policy: `{best.get('Policy_ID')}`",
        f"- Liquidity floor: ${best.get('Liquidity_Floor'):,.0f}",
        f"- Fallback trigger: `{best.get('Fallback_Trigger_Policy')}`",
        f"- Candidate family filter: `{best.get('Candidate_Family_Filter')}`",
        f"- Total return: {format_pct(best.get('Total_Return'))}",
        f"- Excess vs universe: {format_pct(best.get('Excess_Return_vs_Universe'))}",
        f"- Excess vs QQQ: {format_pct(best.get('Excess_Return_vs_QQQ'))}",
        f"- Excess vs XLK: {format_pct(best.get('Excess_Return_vs_XLK'))}",
        f"- Sharpe: {format_num(best.get('Sharpe'))}",
        f"- Max drawdown: {format_pct(best.get('Max_Drawdown'))}",
        f"- Average turnover: {format_pct(best.get('Avg_Turnover'))}",
        f"- Fallback rebalances: {int(best.get('Fallback_Rebalance_Count', 0))}",
        f"- Clean-selection rate: {format_pct(best.get('Clean_Selection_Rate'))}",
        f"- Selection guard stages: `{best.get('Selection_Guard_Stage_Counts', '')}`",
        f"- Selected families: `{best.get('Selected_Family_Counts', '')}`",
        "",
        "Note: the best raw row is not automatically the live default. A row is treated as production-qualified only when it uses no fallback and clears the clean selector in every walk-forward round.",
        "",
        "## Best Production-Qualified Row",
        "",
        f"- Policy: `{best_qualified_no_fallback.get('Policy_ID')}`",
        f"- Liquidity floor: ${best_qualified_no_fallback.get('Liquidity_Floor'):,.0f}",
        f"- Candidate family filter: `{best_qualified_no_fallback.get('Candidate_Family_Filter')}`",
        f"- Total return: {format_pct(best_qualified_no_fallback.get('Total_Return'))}",
        f"- Excess vs universe: {format_pct(best_qualified_no_fallback.get('Excess_Return_vs_Universe'))}",
        f"- Excess vs XLK: {format_pct(best_qualified_no_fallback.get('Excess_Return_vs_XLK'))}",
        f"- Sharpe: {format_num(best_qualified_no_fallback.get('Sharpe'))}",
        f"- Max drawdown: {format_pct(best_qualified_no_fallback.get('Max_Drawdown'))}",
        f"- Average turnover: {format_pct(best_qualified_no_fallback.get('Avg_Turnover'))}",
        f"- Selection guard stages: `{best_qualified_no_fallback.get('Selection_Guard_Stage_Counts', '')}`",
        "",
        "## Liquidity Default Read",
        "",
    ])

    no_fallback_by_floor = (
        dynamic_summary[dynamic_summary["Fallback_Mode"] == "none"]
        .sort_values("Production_Default_Objective", ascending=False)
        .groupby("Liquidity_Floor", as_index=False)
        .head(1)
        .sort_values("Liquidity_Floor")
    )
    for _, row in no_fallback_by_floor.iterrows():
        lines.append(
            "- "
            f"${row['Liquidity_Floor']:,.0f}: `{row['Policy_ID']}`, "
            f"return {format_pct(row.get('Total_Return'))}, "
            f"excess vs XLK {format_pct(row.get('Excess_Return_vs_XLK'))}, "
            f"Sharpe {format_num(row.get('Sharpe'))}, "
            f"drawdown {format_pct(row.get('Max_Drawdown'))}, "
            f"clean rate {format_pct(row.get('Clean_Selection_Rate'))}, "
            f"avg eligible {format_num(row.get('Average_Eligible_Count'), 1)}, "
            f"families `{row.get('Selected_Family_Counts', '')}`"
        )

    lines.extend([
        "",
        f"Best no-fallback policy: `{best_no_fallback.get('Policy_ID')}` at ${best_no_fallback.get('Liquidity_Floor'):,.0f}.",
        f"Best production-qualified no-fallback policy: `{best_qualified_no_fallback.get('Policy_ID')}` at ${best_qualified_no_fallback.get('Liquidity_Floor'):,.0f}.",
        f"Best `$25M` row: `{best_25.get('Policy_ID')}` with excess vs XLK {format_pct(best_25.get('Excess_Return_vs_XLK'))}.",
        "",
        "## Stable Alpha Family Read",
        "",
        f"Best confidence-family row: `{best_confidence.get('Policy_ID')}` at ${best_confidence.get('Liquidity_Floor'):,.0f}, "
        f"return {format_pct(best_confidence.get('Total_Return'))}, excess vs XLK {format_pct(best_confidence.get('Excess_Return_vs_XLK'))}, "
        f"Sharpe {format_num(best_confidence.get('Sharpe'))}, drawdown {format_pct(best_confidence.get('Max_Drawdown'))}.",
        f"Best confidence-family no-fallback row: `{best_confidence_no_fallback.get('Policy_ID')}` at ${best_confidence_no_fallback.get('Liquidity_Floor'):,.0f}, "
        f"return {format_pct(best_confidence_no_fallback.get('Total_Return'))}, excess vs XLK {format_pct(best_confidence_no_fallback.get('Excess_Return_vs_XLK'))}.",
        f"Best production-qualified fixed-family row: `{best_qualified_fixed.get('Policy_ID')}` at ${best_qualified_fixed.get('Liquidity_Floor'):,.0f}, "
        f"return {format_pct(best_qualified_fixed.get('Total_Return'))}, excess vs XLK {format_pct(best_qualified_fixed.get('Excess_Return_vs_XLK'))}.",
        f"Best dynamic-family row: `{best_dynamic.get('Policy_ID')}` at ${best_dynamic.get('Liquidity_Floor'):,.0f}, "
        f"return {format_pct(best_dynamic.get('Total_Return'))}, excess vs XLK {format_pct(best_dynamic.get('Excess_Return_vs_XLK'))}, "
        f"Sharpe {format_num(best_dynamic.get('Sharpe'))}, drawdown {format_pct(best_dynamic.get('Max_Drawdown'))}.",
        "",
    ])

    family_no_fallback = dynamic_summary[
        (dynamic_summary["Fallback_Mode"] == "none")
        & (dynamic_summary["Candidate_Family_Filter"] != "dynamic_all")
    ].sort_values("Production_Default_Objective", ascending=False)
    if not family_no_fallback.empty:
        lines.append("Best fixed-family no-fallback rows:")
        for _, row in family_no_fallback.head(8).iterrows():
            lines.append(
                "- "
                f"`{row['Policy_ID']}` at ${row['Liquidity_Floor']:,.0f}: "
                f"return {format_pct(row.get('Total_Return'))}, "
                f"excess vs XLK {format_pct(row.get('Excess_Return_vs_XLK'))}, "
                f"Sharpe {format_num(row.get('Sharpe'))}, "
                f"drawdown {format_pct(row.get('Max_Drawdown'))}"
            )
        lines.append("")

    lines.extend([
        "## XLK Fallback Trigger Read",
        "",
        f"Best XLK-fallback row: `{best_xlk.get('Policy_ID')}` at ${best_xlk.get('Liquidity_Floor'):,.0f}, "
        f"trigger `{best_xlk.get('Fallback_Trigger_Policy')}`, "
        f"return {format_pct(best_xlk.get('Total_Return'))}, "
        f"excess vs XLK {format_pct(best_xlk.get('Excess_Return_vs_XLK'))}, "
        f"fallback rebalances {int(best_xlk.get('Fallback_Rebalance_Count', 0))}.",
        "",
    ])

    xlk_25 = floor_25[floor_25["Fallback_Mode"] == "xlk"].sort_values("Production_Default_Objective", ascending=False)
    if not xlk_25.empty:
        lines.append("XLK fallback policies at `$25M`:")
        for _, row in xlk_25.iterrows():
            lines.append(
                "- "
                f"`{row['Policy_ID']}` / `{row['Fallback_Trigger_Policy']}`: "
                f"return {format_pct(row.get('Total_Return'))}, "
                f"excess vs XLK {format_pct(row.get('Excess_Return_vs_XLK'))}, "
                f"Sharpe {format_num(row.get('Sharpe'))}, "
                f"drawdown {format_pct(row.get('Max_Drawdown'))}, "
                f"fallback rebalances {int(row.get('Fallback_Rebalance_Count', 0))}"
            )
        lines.append("")

    lines.extend([
        "## Provisional Decision",
        "",
        "Use this as a research-default decision, not a fully automated trading rule.",
        "",
        f"- Liquidity default candidate: ${best_qualified_no_fallback.get('Liquidity_Floor'):,.0f} based on the best no-fallback row that cleared the selector in every walk-forward round.",
        f"- Stable-family candidate: `{best_qualified_fixed.get('Candidate_Family_Filter')}` based on the best production-qualified fixed-family row.",
        f"- `Alpha_Prototype_ConfidenceValueSubRelRisk` should remain a useful ingredient, but should not be promoted as the stable default family yet; its best no-fallback row had excess vs XLK of {format_pct(best_confidence_no_fallback.get('Excess_Return_vs_XLK'))}.",
        "- XLK fallback should trigger only when the dynamic selector reaches `family_stable_relaxed_drawdown_turnover` or `fallback_all`. Do not trigger fallback merely because the selector used the benchmark/drawdown guard; the stricter fallback rules gave up too much return.",
        "- The raw `$100M` / `Alpha_Prototype_XLKCompetitive` result should stay in research review because it had excellent returns but reached `fallback_all` in most walk-forward selection rounds.",
        "",
        "## Files Produced",
        "",
        "- `production_default_decision_summary.csv`",
        "- `production_default_selected_constructions.csv`",
        "- `production_default_static_summary.csv`",
        "- `production_default_policy_rankings.csv`",
        "- `run_metadata.json`",
        "- `simulation_progress.log`",
        "- `PRODUCTION_DEFAULT_DECISION_SUMMARY.md`",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "PRODUCTION_DEFAULT_DECISION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_policy_rankings(dynamic_summary: pd.DataFrame) -> pd.DataFrame:
    if dynamic_summary.empty:
        return pd.DataFrame()
    return (
        dynamic_summary
        .groupby("Policy_ID", dropna=False)
        .agg(
            Tested_Floors=("Liquidity_Floor", "count"),
            Best_Liquidity_Floor=("Liquidity_Floor", lambda values: np.nan),
            Avg_Total_Return=("Total_Return", "mean"),
            Best_Total_Return=("Total_Return", "max"),
            Avg_Excess_vs_XLK=("Excess_Return_vs_XLK", "mean"),
            Best_Excess_vs_XLK=("Excess_Return_vs_XLK", "max"),
            Worst_Excess_vs_XLK=("Excess_Return_vs_XLK", "min"),
            Avg_Sharpe=("Sharpe", "mean"),
            Worst_Drawdown=("Max_Drawdown", "min"),
            Avg_Turnover=("Avg_Turnover", "mean"),
            Avg_Fallback_Rebalances=("Fallback_Rebalance_Count", "mean"),
            Avg_Objective=("Production_Default_Objective", "mean"),
            Best_Objective=("Production_Default_Objective", "max"),
        )
        .reset_index()
        .sort_values(["Best_Objective", "Avg_Objective"], ascending=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide production liquidity, alpha family, and XLK fallback defaults.")
    parser.add_argument("--score-history", type=Path, default=DEFAULT_SCORE_HISTORY)
    parser.add_argument("--weight-tuning-dir", type=Path, default=DEFAULT_WEIGHT_TUNING_DIR)
    parser.add_argument("--historical-price-file", type=Path, default=DEFAULT_PRICE_FILE)
    parser.add_argument("--historical-universe", type=Path, default=DEFAULT_HISTORICAL_UNIVERSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--liquidity-floors", default=DEFAULT_LIQUIDITY_FLOORS)
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
    parser.add_argument("--quiet-simulation-logs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    liquidity_floors = liquidity_test.parse_float_list(args.liquidity_floors)
    top_n_list = construction.parse_top_n_list(args.top_n_list)
    modes = construction.parse_modes(args.portfolio_modes)
    policies = decision_policies()
    selector = monthly_sim.WalkForwardSelectionParams(mode="guarded_family_stable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores, configs, price_wide, volume_wide, historical_universe = liquidity_test.load_inputs(args)

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
            policies=policies,
            selector=selector,
        )
        static_frames.append(static_summary)
        selection_frames.append(selections)
        dynamic_frames.append(dynamic_summary)

    static_all = pd.concat(static_frames, ignore_index=True) if static_frames else pd.DataFrame()
    selections_all = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    dynamic_all = pd.concat(dynamic_frames, ignore_index=True) if dynamic_frames else pd.DataFrame()
    policy_rankings = build_policy_rankings(dynamic_all)

    if not policy_rankings.empty and not dynamic_all.empty:
        best_by_policy = (
            dynamic_all.sort_values("Production_Default_Objective", ascending=False)
            .groupby("Policy_ID", as_index=False)
            .head(1)
            [["Policy_ID", "Liquidity_Floor"]]
            .rename(columns={"Liquidity_Floor": "Best_Liquidity_Floor"})
        )
        policy_rankings = policy_rankings.drop(columns=["Best_Liquidity_Floor"]).merge(
            best_by_policy,
            on="Policy_ID",
            how="left",
        )

    dynamic_all.sort_values("Production_Default_Objective", ascending=False).round(6).to_csv(
        args.output_dir / "production_default_decision_summary.csv",
        index=False,
    )
    selections_all.round(6).to_csv(args.output_dir / "production_default_selected_constructions.csv", index=False)
    static_all.round(6).to_csv(args.output_dir / "production_default_static_summary.csv", index=False)
    policy_rankings.round(6).to_csv(args.output_dir / "production_default_policy_rankings.csv", index=False)

    metadata = {
        "status": "complete",
        "score_history": str(args.score_history),
        "weight_tuning_dir": str(args.weight_tuning_dir),
        "historical_price_file": str(args.historical_price_file),
        "historical_universe": str(args.historical_universe),
        "liquidity_floors": liquidity_floors,
        "selection_params": asdict(selector),
        "policies": [asdict(policy) for policy in policies],
        "candidate_preset": args.candidate_preset,
        "config_scope": args.config_scope,
        "shortlist_count": args.shortlist_count,
        "max_configs": args.max_configs,
        "alpha_score_columns": parse_list(args.alpha_score_columns),
        "top_n_list": top_n_list,
        "portfolio_modes": modes,
        "dynamic_summary_rows": int(len(dynamic_all)),
        "selection_rows": int(len(selections_all)),
        "static_summary_rows": int(len(static_all)),
        "policy_ranking_rows": int(len(policy_rankings)),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    write_summary(
        output_dir=args.output_dir,
        dynamic_summary=dynamic_all,
        selections=selections_all,
        static_summary=static_all,
        policies=policies,
        liquidity_floors=liquidity_floors,
    )

    print("\n=== PRODUCTION DEFAULT DECISION SUMMARY ===", flush=True)
    columns = [
        "Policy_ID",
        "Liquidity_Floor",
        "Fallback_Trigger_Policy",
        "Production_Default_Objective",
        "Total_Return",
        "Excess_Return_vs_Universe",
        "Excess_Return_vs_XLK",
        "Sharpe",
        "Max_Drawdown",
        "Fallback_Rebalance_Count",
        "Selected_Family_Counts",
    ]
    if not dynamic_all.empty:
        print(
            dynamic_all.sort_values("Production_Default_Objective", ascending=False)[columns]
            .head(25)
            .to_string(index=False),
            flush=True,
        )
    print(f"\n[SUCCESS] Saved production-default decision outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
