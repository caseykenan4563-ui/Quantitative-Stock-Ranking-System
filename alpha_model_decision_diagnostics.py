#!/usr/bin/env python3
"""Summarize whether the current alpha model is strong enough versus XLK.

This is a decision/reporting harness. It does not change scoring or portfolio
construction. It reads the latest signal-quality, monthly simulator, robustness,
and universe-audit outputs and writes a plain-English attribution report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OLD_SIGNAL_SCORECARD = Path("backtests/alpha_signal_quality_sharadar/overall_signal_scorecard.csv")
DEFAULT_NEW_SIGNAL_SCORECARD = Path("backtests/alpha_signal_quality_signal_improvements_sharadar/overall_signal_scorecard.csv")
DEFAULT_BASE_DYNAMIC_SUMMARY = Path(
    "backtests/monthly_rebalanced_portfolio_alpha_defensive_guarded_sharadar/walk_forward_dynamic_summary.csv"
)
DEFAULT_NEW_DYNAMIC_SUMMARY = Path(
    "backtests/monthly_rebalanced_portfolio_alpha_signal_improvements_sharadar/walk_forward_dynamic_summary.csv"
)
DEFAULT_NEW_STATIC_SUMMARY = Path(
    "backtests/monthly_rebalanced_portfolio_alpha_signal_improvements_sharadar/simulation_summary.csv"
)
DEFAULT_ROBUSTNESS_SUMMARY = Path("backtests/portfolio_robustness_stress_sharadar/stress_test_summary.csv")
DEFAULT_UNIVERSE_AUDIT = Path("backtests/combined_score_sharadar/historical_universe_score_audit.csv")
DEFAULT_OUTPUT_DIR = Path("backtests/alpha_model_decision_diagnostics")

NEW_SIGNAL_CANDIDATES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write alpha model decision diagnostics.")
    parser.add_argument("--old-signal-scorecard", type=Path, default=DEFAULT_OLD_SIGNAL_SCORECARD)
    parser.add_argument("--new-signal-scorecard", type=Path, default=DEFAULT_NEW_SIGNAL_SCORECARD)
    parser.add_argument("--base-dynamic-summary", type=Path, default=DEFAULT_BASE_DYNAMIC_SUMMARY)
    parser.add_argument("--new-dynamic-summary", type=Path, default=DEFAULT_NEW_DYNAMIC_SUMMARY)
    parser.add_argument("--new-static-summary", type=Path, default=DEFAULT_NEW_STATIC_SUMMARY)
    parser.add_argument("--robustness-summary", type=Path, default=DEFAULT_ROBUSTNESS_SUMMARY)
    parser.add_argument("--universe-audit", type=Path, default=DEFAULT_UNIVERSE_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def numeric(row: pd.Series | dict, column: str, default: float = np.nan) -> float:
    try:
        value = row.get(column, default)
    except AttributeError:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def format_pct(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric_value):
        return "n/a"
    return f"{numeric_value * 100:.2f}%"


def format_num(value: object, digits: int = 4) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric_value):
        return "n/a"
    return f"{numeric_value:.{digits}f}"


def first_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    return frame.iloc[0]


def best_signal_rows(scorecard: pd.DataFrame, candidates: list[str] | None = None) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()
    frame = scorecard.copy()
    if candidates is not None:
        frame = frame[frame["Signal"].isin(candidates)].copy()
    if frame.empty or "Quality_Score" not in frame.columns:
        return frame.head(0)
    return frame.sort_values("Quality_Score", ascending=False).reset_index(drop=True)


def build_signal_comparison(old_scorecard: pd.DataFrame, new_scorecard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in [("prior", old_scorecard), ("improved_candidate_pass", new_scorecard)]:
        if frame.empty:
            continue
        ranked = best_signal_rows(frame)
        for rank, (_, row) in enumerate(ranked.head(20).iterrows(), start=1):
            rows.append({
                "Run": label,
                "Rank": rank,
                "Signal": row.get("Signal"),
                "Quality_Score": numeric(row, "Quality_Score"),
                "Mean_Spearman_IC_All_Horizons": numeric(row, "Mean_Spearman_IC_All_Horizons"),
                "Mean_Top_Bottom_Spread": numeric(row, "Mean_Top_Bottom_Spread"),
                "Top100_Mean_Excess_vs_Universe": numeric(row, "Top100_Mean_Excess_vs_Universe"),
                "Top100_Mean_Excess_vs_XLK": numeric(row, "Top100_Mean_Excess_vs_XLK"),
                "Top50_Mean_Excess_vs_XLK": numeric(row, "Top50_Mean_Excess_vs_XLK"),
            })
    return pd.DataFrame(rows)


def build_portfolio_comparison(
    base_dynamic: pd.DataFrame,
    new_dynamic: pd.DataFrame,
    new_static: pd.DataFrame,
    robustness: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if not base_dynamic.empty:
        row = first_row(base_dynamic)
        rows.append({
            "Run": "prior_defensive_dynamic",
            "Config_ID": row.get("Config_ID"),
            "Fallback_Mode": row.get("Fallback_Mode", "none"),
            "Total_Return": numeric(row, "Total_Return"),
            "Excess_Return_vs_Universe": numeric(row, "Excess_Return_vs_Universe"),
            "Excess_Return_vs_XLK": numeric(row, "Excess_Return_vs_XLK"),
            "Sharpe": numeric(row, "Sharpe"),
            "Sortino": numeric(row, "Sortino"),
            "Max_Drawdown": numeric(row, "Max_Drawdown"),
            "Avg_Turnover": numeric(row, "Avg_Turnover"),
            "Total_Cost": numeric(row, "Total_Cost"),
            "Fallback_Rebalance_Count": numeric(row, "Fallback_Rebalance_Count", 0.0),
        })
    if not new_dynamic.empty:
        row = first_row(new_dynamic)
        rows.append({
            "Run": "improved_candidate_dynamic",
            "Config_ID": row.get("Config_ID"),
            "Fallback_Mode": row.get("Fallback_Mode", "none"),
            "Total_Return": numeric(row, "Total_Return"),
            "Excess_Return_vs_Universe": numeric(row, "Excess_Return_vs_Universe"),
            "Excess_Return_vs_XLK": numeric(row, "Excess_Return_vs_XLK"),
            "Sharpe": numeric(row, "Sharpe"),
            "Sortino": numeric(row, "Sortino"),
            "Max_Drawdown": numeric(row, "Max_Drawdown"),
            "Avg_Turnover": numeric(row, "Avg_Turnover"),
            "Total_Cost": numeric(row, "Total_Cost"),
            "Fallback_Rebalance_Count": numeric(row, "Fallback_Rebalance_Count", 0.0),
        })
    if not new_static.empty:
        ranked = new_static.sort_values("Simulator_Objective", ascending=False)
        for rank, (_, row) in enumerate(ranked.head(10).iterrows(), start=1):
            rows.append({
                "Run": f"improved_candidate_static_rank_{rank}",
                "Config_ID": row.get("Config_ID"),
                "Fallback_Mode": row.get("Fallback_Mode", "none"),
                "Total_Return": numeric(row, "Total_Return"),
                "Excess_Return_vs_Universe": numeric(row, "Excess_Return_vs_Universe"),
                "Excess_Return_vs_XLK": numeric(row, "Excess_Return_vs_XLK"),
                "Sharpe": numeric(row, "Sharpe"),
                "Sortino": numeric(row, "Sortino"),
                "Max_Drawdown": numeric(row, "Max_Drawdown"),
                "Avg_Turnover": numeric(row, "Avg_Turnover"),
                "Total_Cost": numeric(row, "Total_Cost"),
                "Fallback_Rebalance_Count": numeric(row, "Fallback_Rebalance_Count", 0.0),
            })
    if not robustness.empty:
        ranked = robustness.sort_values("Stress_Robustness_Objective", ascending=False)
        for rank, (_, row) in enumerate(ranked.head(10).iterrows(), start=1):
            rows.append({
                "Run": f"robustness_rank_{rank}_{row.get('Scenario_ID')}_{row.get('Fallback_Mode')}",
                "Config_ID": row.get("Config_ID"),
                "Fallback_Mode": row.get("Fallback_Mode", "none"),
                "Total_Return": numeric(row, "Total_Return"),
                "Excess_Return_vs_Universe": numeric(row, "Excess_Return_vs_Universe"),
                "Excess_Return_vs_XLK": numeric(row, "Excess_Return_vs_XLK"),
                "Sharpe": numeric(row, "Sharpe"),
                "Sortino": numeric(row, "Sortino"),
                "Max_Drawdown": numeric(row, "Max_Drawdown"),
                "Avg_Turnover": numeric(row, "Avg_Turnover"),
                "Total_Cost": numeric(row, "Total_Cost"),
                "Fallback_Rebalance_Count": numeric(row, "Fallback_Rebalance_Count", 0.0),
            })
    return pd.DataFrame(rows)


def build_driver_attribution(
    old_scorecard: pd.DataFrame,
    new_scorecard: pd.DataFrame,
    base_dynamic: pd.DataFrame,
    new_dynamic: pd.DataFrame,
    robustness: pd.DataFrame,
    universe_audit: pd.DataFrame,
) -> pd.DataFrame:
    active_scorecard = new_scorecard if not new_scorecard.empty else old_scorecard
    best_signal = first_row(best_signal_rows(active_scorecard))
    best_new_signal = first_row(best_signal_rows(active_scorecard, NEW_SIGNAL_CANDIDATES))
    dynamic = first_row(new_dynamic if not new_dynamic.empty else base_dynamic)

    dynamic_xlk_gap = numeric(dynamic, "Excess_Return_vs_XLK")
    dynamic_universe_gap = numeric(dynamic, "Excess_Return_vs_Universe")
    total_cost = numeric(dynamic, "Total_Cost", 0.0)

    base_none = pd.DataFrame()
    base_xlk = pd.DataFrame()
    strict_none = pd.DataFrame()
    best_robustness = pd.Series(dtype="object")
    if not robustness.empty:
        best_robustness = first_row(robustness.sort_values("Stress_Robustness_Objective", ascending=False))
        base_none = robustness[
            (robustness["Scenario_ID"].astype(str) == "base")
            & (robustness["Fallback_Mode"].astype(str) == "none")
        ]
        base_xlk = robustness[
            (robustness["Scenario_ID"].astype(str) == "base")
            & (robustness["Fallback_Mode"].astype(str) == "xlk")
        ]
        strict_none = robustness[
            (robustness["Scenario_ID"].astype(str) == "strict_liquidity")
            & (robustness["Fallback_Mode"].astype(str) == "none")
        ]

    scored_count = missing_price = stale_or_missing = 0
    if not universe_audit.empty and "Score_Availability_Status" in universe_audit.columns:
        status_counts = universe_audit["Score_Availability_Status"].value_counts(dropna=False)
        scored_count = int(status_counts.get("scored", 0))
        missing_price = int(status_counts.get("no_price_data_returned", 0))
        stale_or_missing = int(len(universe_audit) - scored_count)

    rows = []
    if not best_signal.empty:
        best_top100_xlk = numeric(best_signal, "Top100_Mean_Excess_vs_XLK")
        rows.append({
            "Driver": "Signal strength versus XLK",
            "Evidence": (
                f"Best current signal `{best_signal.get('Signal')}` has top-100 excess vs XLK "
                f"{format_pct(best_top100_xlk)} and top-100 excess vs universe "
                f"{format_pct(numeric(best_signal, 'Top100_Mean_Excess_vs_Universe'))}."
            ),
            "Severity": "high" if best_top100_xlk < 0 else "medium",
            "Conclusion": (
                "The model is still mainly proving stock-universe alpha, not durable XLK alpha."
                if best_top100_xlk < 0
                else "The best cross-sectional signal clears XLK in this diagnostic, pending portfolio validation."
            ),
        })

    if not best_new_signal.empty:
        rows.append({
            "Driver": "Improved signal candidate quality",
            "Evidence": (
                f"Best new candidate `{best_new_signal.get('Signal')}` has quality score "
                f"{format_num(numeric(best_new_signal, 'Quality_Score'), 2)}, mean IC "
                f"{format_num(numeric(best_new_signal, 'Mean_Spearman_IC_All_Horizons'))}, "
                f"and top-50 excess vs XLK {format_pct(numeric(best_new_signal, 'Top50_Mean_Excess_vs_XLK'))}."
            ),
            "Severity": "medium",
            "Conclusion": (
                "Promote only if this also improves the sequential monthly simulation; signal diagnostics alone are not enough."
            ),
        })

    if not dynamic.empty:
        rows.append({
            "Driver": "Portfolio result versus XLK",
            "Evidence": (
                f"Dynamic portfolio excess vs universe is {format_pct(dynamic_universe_gap)}, "
                f"excess vs XLK is {format_pct(dynamic_xlk_gap)}, total cost drag is {format_pct(total_cost)}, "
                f"and max drawdown is {format_pct(numeric(dynamic, 'Max_Drawdown'))}."
            ),
            "Severity": "high" if dynamic_xlk_gap < 0 else "low",
            "Conclusion": (
                "If XLK gap is larger than cost drag, the shortfall is not just trading friction."
                if dynamic_xlk_gap < -abs(total_cost)
                else "Costs could plausibly explain a meaningful part of the XLK gap."
            ),
        })

    if not best_robustness.empty:
        rows.append({
            "Driver": "Best robustness scenario",
            "Evidence": (
                f"Best stress row was `{best_robustness.get('Scenario_ID')}` / "
                f"`{best_robustness.get('Fallback_Mode')}` with return "
                f"{format_pct(numeric(best_robustness, 'Total_Return'))}, "
                f"excess vs XLK {format_pct(numeric(best_robustness, 'Excess_Return_vs_XLK'))}, "
                f"Sharpe {format_num(numeric(best_robustness, 'Sharpe'))}, and drawdown "
                f"{format_pct(numeric(best_robustness, 'Max_Drawdown'))}."
            ),
            "Severity": "medium",
            "Conclusion": (
                "There is a promising configuration under cleaner liquidity assumptions, but it is not stable across base and larger-account assumptions."
            ),
        })

    if not base_none.empty and not strict_none.empty:
        base = base_none.iloc[0]
        strict = strict_none.iloc[0]
        rows.append({
            "Driver": "Liquidity filter sensitivity",
            "Evidence": (
                f"Base/no-fallback returned {format_pct(numeric(base, 'Total_Return'))} with "
                f"excess vs XLK {format_pct(numeric(base, 'Excess_Return_vs_XLK'))}; "
                f"strict-liquidity/no-fallback returned {format_pct(numeric(strict, 'Total_Return'))} with "
                f"excess vs XLK {format_pct(numeric(strict, 'Excess_Return_vs_XLK'))}."
            ),
            "Severity": "high",
            "Conclusion": (
                "The model appears highly sensitive to investability filters. Stricter liquidity may remove noisy/fragile names and improve the stock-selection signal."
            ),
        })

    if not base_none.empty and not base_xlk.empty:
        none = base_none.iloc[0]
        xlk = base_xlk.iloc[0]
        improvement = numeric(xlk, "Total_Return") - numeric(none, "Total_Return")
        fallback_count = numeric(xlk, "Fallback_Rebalance_Count", 0.0)
        rows.append({
            "Driver": "Fallback layer",
            "Evidence": (
                f"Base stress run with no fallback returned {format_pct(numeric(none, 'Total_Return'))}; "
                f"base stress run with XLK fallback returned {format_pct(numeric(xlk, 'Total_Return'))}, "
                f"a difference of {format_pct(improvement)}, with {format_num(fallback_count, 0)} fallback rebalances."
            ),
            "Severity": "medium",
            "Conclusion": (
                "Fallback did not explain the base result when it never triggered; fallback-heavy wins should still be treated as benchmark exposure rather than stock-picking proof."
            ),
        })

    if not universe_audit.empty:
        rows.append({
            "Driver": "Historical universe completeness",
            "Evidence": (
                f"Universe audit has {len(universe_audit):,} rows, {scored_count:,} scored tickers, "
                f"{missing_price:,} no-price rows, and {stale_or_missing:,} rows not fully scored."
            ),
            "Severity": "medium" if stale_or_missing else "low",
            "Conclusion": (
                "The universe is usable for research, but it is still not a complete historical tech universe."
            ),
        })

    return pd.DataFrame(rows)


def choose_decision(driver_attribution: pd.DataFrame, portfolio_comparison: pd.DataFrame) -> str:
    if portfolio_comparison.empty:
        return "Insufficient portfolio evidence. Run the monthly simulator before promoting any signal."
    dynamic = portfolio_comparison[portfolio_comparison["Run"].isin([
        "improved_candidate_dynamic",
        "prior_defensive_dynamic",
    ])]
    row = first_row(dynamic.tail(1))
    xlk_gap = numeric(row, "Excess_Return_vs_XLK")
    universe_gap = numeric(row, "Excess_Return_vs_Universe")
    drawdown = numeric(row, "Max_Drawdown")
    sharpe = numeric(row, "Sharpe")
    if xlk_gap > 0 and universe_gap > 0 and sharpe > 1.0 and drawdown > -0.30:
        return "Promising enough for a broader robustness rerun, but still not live-ready without paper trading."
    if universe_gap > 0 and xlk_gap < 0:
        return "Promising research model, but not strong enough to replace XLK as a production allocation rule yet."
    if universe_gap < 0 and xlk_gap < 0:
        return "Promising raw signals exist, but the current walk-forward selection layer is not strong enough yet."
    return "Not strong enough yet; improve the signal and universe before relying on this model."


def write_markdown(
    output_dir: Path,
    driver_attribution: pd.DataFrame,
    signal_comparison: pd.DataFrame,
    portfolio_comparison: pd.DataFrame,
    decision: str,
) -> None:
    lines = [
        "# Alpha Model Decision Diagnostics",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Driver Attribution",
        "",
    ]
    if driver_attribution.empty:
        lines.append("No attribution rows were generated.")
    else:
        for _, row in driver_attribution.iterrows():
            lines.append(
                f"- `{row['Driver']}` ({row['Severity']}): {row['Evidence']} {row['Conclusion']}"
            )

    lines.extend(["", "## Best Signal Rows", ""])
    if signal_comparison.empty:
        lines.append("No signal comparison was available.")
    else:
        for _, row in signal_comparison.head(12).iterrows():
            lines.append(
                "- "
                f"{row['Run']} rank {int(row['Rank'])} `{row['Signal']}`: "
                f"quality {format_num(row['Quality_Score'], 2)}, "
                f"mean IC {format_num(row['Mean_Spearman_IC_All_Horizons'])}, "
                f"top-100 excess vs universe {format_pct(row['Top100_Mean_Excess_vs_Universe'])}, "
                f"top-100 excess vs XLK {format_pct(row['Top100_Mean_Excess_vs_XLK'])}"
            )

    lines.extend(["", "## Portfolio Rows", ""])
    if portfolio_comparison.empty:
        lines.append("No portfolio comparison was available.")
    else:
        for _, row in portfolio_comparison.head(14).iterrows():
            lines.append(
                "- "
                f"{row['Run']} `{row['Config_ID']}`: "
                f"return {format_pct(row['Total_Return'])}, "
                f"excess vs universe {format_pct(row['Excess_Return_vs_Universe'])}, "
                f"excess vs XLK {format_pct(row['Excess_Return_vs_XLK'])}, "
                f"Sharpe {format_num(row['Sharpe'])}, "
                f"drawdown {format_pct(row['Max_Drawdown'])}, "
                f"cost {format_pct(row['Total_Cost'])}"
            )

    lines.extend([
        "",
        "## Next Work",
        "",
        "The next promotion step is to rerun the robustness suite with any improved candidate that survives the monthly simulator.",
        "If the improved candidates still trail XLK, the next alpha work should focus on new information rather than more threshold tuning: fair-value confidence, subindustry-specific trend features, downside penalties, broader historical universe coverage, and later Google/news/search-interest features.",
        "",
        "## Files Produced",
        "",
        "- `driver_attribution.csv`",
        "- `signal_scorecard_comparison.csv`",
        "- `portfolio_result_comparison.csv`",
        "- `alpha_model_decision_summary.csv`",
        "- `run_metadata.json`",
        "- `ALPHA_MODEL_DECISION_SUMMARY.md`",
    ])
    (output_dir / "ALPHA_MODEL_DECISION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    old_scorecard = read_csv(args.old_signal_scorecard)
    new_scorecard = read_csv(args.new_signal_scorecard)
    base_dynamic = read_csv(args.base_dynamic_summary)
    new_dynamic = read_csv(args.new_dynamic_summary)
    new_static = read_csv(args.new_static_summary)
    robustness = read_csv(args.robustness_summary)
    universe_audit = read_csv(args.universe_audit)

    signal_comparison = build_signal_comparison(old_scorecard, new_scorecard)
    portfolio_comparison = build_portfolio_comparison(base_dynamic, new_dynamic, new_static, robustness)
    driver_attribution = build_driver_attribution(
        old_scorecard=old_scorecard,
        new_scorecard=new_scorecard,
        base_dynamic=base_dynamic,
        new_dynamic=new_dynamic,
        robustness=robustness,
        universe_audit=universe_audit,
    )
    decision = choose_decision(driver_attribution, portfolio_comparison)

    summary = pd.DataFrame([{
        "Decision": decision,
        "Old_Signal_Scorecard": str(args.old_signal_scorecard),
        "New_Signal_Scorecard": str(args.new_signal_scorecard),
        "Base_Dynamic_Summary": str(args.base_dynamic_summary),
        "New_Dynamic_Summary": str(args.new_dynamic_summary),
        "Robustness_Summary": str(args.robustness_summary),
        "Universe_Audit": str(args.universe_audit),
        "Signal_Comparison_Rows": int(len(signal_comparison)),
        "Portfolio_Comparison_Rows": int(len(portfolio_comparison)),
        "Driver_Attribution_Rows": int(len(driver_attribution)),
    }])

    signal_comparison.round(6).to_csv(output_dir / "signal_scorecard_comparison.csv", index=False)
    portfolio_comparison.round(6).to_csv(output_dir / "portfolio_result_comparison.csv", index=False)
    driver_attribution.to_csv(output_dir / "driver_attribution.csv", index=False)
    summary.to_csv(output_dir / "alpha_model_decision_summary.csv", index=False)
    metadata = {
        "status": "complete",
        "decision": decision,
        "inputs": {
            "old_signal_scorecard": str(args.old_signal_scorecard),
            "new_signal_scorecard": str(args.new_signal_scorecard),
            "base_dynamic_summary": str(args.base_dynamic_summary),
            "new_dynamic_summary": str(args.new_dynamic_summary),
            "new_static_summary": str(args.new_static_summary),
            "robustness_summary": str(args.robustness_summary),
            "universe_audit": str(args.universe_audit),
        },
        "new_signal_candidates": NEW_SIGNAL_CANDIDATES,
        "rows": {
            "signal_comparison": int(len(signal_comparison)),
            "portfolio_comparison": int(len(portfolio_comparison)),
            "driver_attribution": int(len(driver_attribution)),
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    write_markdown(output_dir, driver_attribution, signal_comparison, portfolio_comparison, decision)

    print(f"Wrote alpha model decision diagnostics to {output_dir}")
    print(decision)
    if not driver_attribution.empty:
        print(driver_attribution.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
