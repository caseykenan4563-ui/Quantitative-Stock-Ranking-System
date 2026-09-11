#!/usr/bin/env python3
"""
Build the historical technology universe used by the portfolio simulator.

This creates the first reproducible historical-universe CSV for the project. It
combines the current hardcoded model universe with the stale/removed ticker seed
list recovered from the git commit that cleaned the active universe.

The output is intentionally explicit about source quality. This is a practical
historical-universe seed, not a vendor-grade delisted security master.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import comp


DEFAULT_SCORE_HISTORY = Path("backtests/combined_score/point_in_time_scores.csv")
DEFAULT_OUTPUT = Path("data/historical_tech_universe.csv")
DEFAULT_AUDIT = Path("data/historical_tech_universe_audit.csv")
DEFAULT_NOTES = Path("data/historical_tech_universe_notes.md")
DEFAULT_START_DATE = "2000-01-01"
MODEL_REMOVAL_DATE = "2026-09-07"
CURRENT_STATUS_BUFFER_DAYS = 21


@dataclass(frozen=True)
class LegacyTicker:
    ticker: str
    subindustry: str
    universe_role: str
    removal_reason: str
    source_note: str
    benchmark_bucket: str = ""


LEGACY_REMOVED_TICKERS = [
    LegacyTicker("CFLT", "Application Software", "core", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("BASE", "Application Software", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("INFA", "Application Software", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("JAMF", "Application Software", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup.", "Small"),
    LegacyTicker("MLNK", "Application Software", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("ONTF", "Application Software", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup.", "Small"),
    LegacyTicker("PRO", "Application Software", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("SEMR", "Application Software", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup.", "Small"),
    LegacyTicker("RTEC", "Semiconductor Equipment", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("COMM", "Hardware and Storage", "core", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup.", "Mid"),
    LegacyTicker("IAS", "Search and Digital Media", "core", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("VMW", "System Software", "core", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("CYBR", "Cybersecurity", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup.", "Large"),
    LegacyTicker("WISH", "E-Commerce Marketplace", "core", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("SKLZ", "Interactive Home Entertainment", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup.", "Small"),
    LegacyTicker("ASGN", "IT Consulting and Services", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup.", "Small"),
    LegacyTicker("PRFT", "IT Consulting and Services", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup."),
    LegacyTicker("FYBR", "Integrated Telecom", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup.", "Large"),
    LegacyTicker("WOW", "Integrated Telecom", "confirmers", "removed_from_active_model_universe", "Removed by commit 1879e1b as stale/delisted cleanup.", "Small"),
]

INVALID_TICKER_EVENTS = [
    {
        "Ticker": "LIFL",
        "SubIndustry": "Application Software",
        "Universe_Role": "benchmark_bucket_typo",
        "Status": "Excluded_Invalid_Ticker",
        "Source_Note": "Historical typo in TECH_BENCHMARK_GROUPS. Corrected to LIF and excluded from membership.",
    }
]


def flatten_current_universe() -> pd.DataFrame:
    rows = []
    for subindustry, group in comp.REGIME_GROUPS.items():
        for role in ("core", "confirmers"):
            for ticker in group.get(role, []):
                rows.append({
                    "Ticker": str(ticker).upper(),
                    "Industry": "Tech",
                    "SubIndustry": comp.canonical_subindustry_name(subindustry) or subindustry,
                    "Universe_Role": role,
                    "Source_Note": "Current active ticker from comp.REGIME_GROUPS.",
                })

    current = pd.DataFrame(rows)
    alternates = (
        current.groupby("Ticker")["SubIndustry"]
        .apply(lambda values: ";".join(sorted(set(map(str, values)))))
        .rename("Alternate_SubIndustries")
        .reset_index()
    )
    current = (
        current.sort_values(["Ticker", "Universe_Role", "SubIndustry"])
        .drop_duplicates("Ticker", keep="first")
        .merge(alternates, on="Ticker", how="left")
    )
    return current


def benchmark_bucket_lookup() -> dict[str, str]:
    lookup = {}
    for subindustry, buckets in comp.TECH_BENCHMARK_GROUPS.items():
        for bucket, tickers in buckets.items():
            for ticker in tickers:
                lookup[str(ticker).upper()] = str(bucket)
    return lookup


def score_first_dates(score_history: Path) -> pd.DataFrame:
    if not score_history.exists():
        return pd.DataFrame(columns=["Ticker", "First_Score_Date", "Last_Score_Date"])
    scores = pd.read_csv(score_history, usecols=["Date", "Ticker"])
    scores["Date"] = pd.to_datetime(scores["Date"], errors="coerce").dt.normalize()
    scores["Ticker"] = scores["Ticker"].astype(str).str.upper()
    return (
        scores.dropna(subset=["Date", "Ticker"])
        .groupby("Ticker")["Date"]
        .agg(First_Score_Date="min", Last_Score_Date="max")
        .reset_index()
    )


def extract_price_bounds(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    multi = isinstance(raw.columns, pd.MultiIndex)
    for ticker in tickers:
        price = pd.Series(dtype="float64")
        try:
            ticker_df = raw[ticker] if multi else raw
            if "Adj Close" in ticker_df.columns:
                price = ticker_df["Adj Close"]
            elif "Close" in ticker_df.columns:
                price = ticker_df["Close"]
            price = pd.to_numeric(price, errors="coerce").dropna()
        except Exception:
            price = pd.Series(dtype="float64")

        rows.append({
            "Ticker": ticker,
            "First_Price_Date": pd.to_datetime(price.index.min()).normalize() if not price.empty else pd.NaT,
            "Last_Price_Date": pd.to_datetime(price.index.max()).normalize() if not price.empty else pd.NaT,
            "Price_Row_Count": int(len(price)),
        })
    return pd.DataFrame(rows)


def fetch_price_bounds(tickers: list[str], start_date: str, end_date: str, batch_size: int) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    tickers = sorted(set(tickers))
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        print(f"[INFO] Fetching listing bounds batch {start // batch_size + 1}: {len(batch)} tickers", flush=True)
        try:
            raw = yf.download(
                tickers=batch,
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            frames.append(extract_price_bounds(raw, batch))
        except Exception as exc:
            print(f"[WARN] Failed listing-bounds batch {batch}: {exc}", flush=True)
            frames.append(pd.DataFrame({
                "Ticker": batch,
                "First_Price_Date": pd.NaT,
                "Last_Price_Date": pd.NaT,
                "Price_Row_Count": 0,
            }))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def membership_start(backtest_start: pd.Timestamp, first_price: pd.Timestamp | pd.NaT) -> pd.Timestamp:
    if pd.isna(first_price):
        return backtest_start
    return max(backtest_start, pd.to_datetime(first_price).normalize())


def row_status(is_active_current: bool, last_price: pd.Timestamp | pd.NaT, latest_price: pd.Timestamp) -> tuple[str, str, str | None]:
    if is_active_current:
        return "Active_Current_Model_Universe", "active_current_comp_regime_groups", None
    if pd.isna(last_price):
        return "Historical_Seed_No_Price_Returned", "legacy_removed_from_git_history_no_yahoo_price", MODEL_REMOVAL_DATE

    last_price = pd.to_datetime(last_price).normalize()
    days_stale = (latest_price - last_price).days
    if days_stale <= CURRENT_STATUS_BUFFER_DAYS:
        return (
            "Removed_From_Model_Universe_Price_Still_Active",
            "legacy_removed_from_git_history_price_still_active",
            MODEL_REMOVAL_DATE,
        )
    return (
        "Inactive_Historical_Price_Ended",
        "legacy_removed_from_git_history_price_ended",
        last_price.strftime("%Y-%m-%d"),
    )


def build_universe(score_history: Path, start_date: str, end_date: str, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    current = flatten_current_universe()
    legacy = pd.DataFrame([{
        "Ticker": item.ticker,
        "Industry": "Tech",
        "SubIndustry": item.subindustry,
        "Universe_Role": item.universe_role,
        "Alternate_SubIndustries": item.subindustry,
        "Benchmark_Bucket": item.benchmark_bucket,
        "Source_Note": item.source_note,
        "Removal_Reason": item.removal_reason,
    } for item in LEGACY_REMOVED_TICKERS])
    current["Removal_Reason"] = ""

    combined = pd.concat([current, legacy], ignore_index=True, sort=False)
    combined["Ticker"] = combined["Ticker"].astype(str).str.upper()
    combined["Is_Current_Model_Ticker"] = combined["Ticker"].isin(set(current["Ticker"]))
    combined = (
        combined.sort_values(["Ticker", "Is_Current_Model_Ticker"], ascending=[True, False])
        .drop_duplicates("Ticker", keep="first")
        .reset_index(drop=True)
    )

    score_bounds = score_first_dates(score_history)
    price_bounds = fetch_price_bounds(
        tickers=combined["Ticker"].tolist(),
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
    )
    combined = (
        combined
        .merge(score_bounds, on="Ticker", how="left")
        .merge(price_bounds, on="Ticker", how="left")
    )

    latest_price = pd.to_datetime(combined["Last_Price_Date"], errors="coerce").max()
    backtest_start = (
        pd.to_datetime(score_bounds["First_Score_Date"], errors="coerce").min()
        if not score_bounds.empty else pd.Timestamp("2021-09-30")
    )
    if pd.isna(backtest_start):
        backtest_start = pd.Timestamp("2021-09-30")

    status_rows = [
        row_status(
            is_active_current=bool(row["Is_Current_Model_Ticker"]),
            last_price=row.get("Last_Price_Date", pd.NaT),
            latest_price=latest_price,
        )
        for _, row in combined.iterrows()
    ]
    combined["Status"] = [item[0] for item in status_rows]
    combined["Membership_Source"] = [item[1] for item in status_rows]
    combined["Membership_End_Date"] = [item[2] for item in status_rows]
    combined["Membership_Start_Date"] = [
        membership_start(backtest_start, row.get("First_Price_Date", pd.NaT)).strftime("%Y-%m-%d")
        for _, row in combined.iterrows()
    ]

    bucket_lookup = benchmark_bucket_lookup()
    existing_bucket = combined.get("Benchmark_Bucket", pd.Series("", index=combined.index)).fillna("").astype(str)
    current_bucket = combined["Ticker"].map(bucket_lookup).fillna("").astype(str)
    combined["Benchmark_Bucket"] = existing_bucket.where(existing_bucket.str.strip() != "", current_bucket)
    combined["Membership_End_Date"] = combined["Membership_End_Date"].fillna("")
    combined["Universe_Inclusion"] = "included"
    combined["Source_Confidence"] = np.where(
        combined["Is_Current_Model_Ticker"],
        "high_current_model_membership",
        "medium_legacy_seed_price_window",
    )
    combined["Data_Limitation"] = np.where(
        combined["Is_Current_Model_Ticker"],
        "Current model membership is known, but historical subindustry changes are not tracked.",
        "Legacy membership is seeded from the prior model list and Yahoo price bounds, not a vendor delisted security master.",
    )

    invalid = pd.DataFrame(INVALID_TICKER_EVENTS)
    invalid["Industry"] = "Tech"
    invalid["Membership_Start_Date"] = ""
    invalid["Membership_End_Date"] = ""
    invalid["Benchmark_Bucket"] = ""
    invalid["Is_Current_Model_Ticker"] = False
    invalid["Membership_Source"] = "historical_typo_exclusion"
    invalid["Universe_Inclusion"] = "excluded"
    invalid["Source_Confidence"] = "high_invalid_ticker_typo"
    invalid["Data_Limitation"] = "Excluded from membership because it was a typo rather than a tradable ticker."

    column_order = [
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
    for col in column_order:
        if col not in combined.columns:
            combined[col] = ""
        if col not in invalid.columns:
            invalid[col] = ""

    audit = pd.concat([combined[column_order], invalid[column_order]], ignore_index=True, sort=False)
    universe = audit[audit["Universe_Inclusion"] == "included"].copy()
    universe = universe.sort_values(["SubIndustry", "Status", "Ticker"]).reset_index(drop=True)
    audit = audit.sort_values(["Universe_Inclusion", "SubIndustry", "Ticker"]).reset_index(drop=True)

    metadata = {
        "status": "complete",
        "score_history": str(score_history),
        "price_start_date": start_date,
        "price_end_date": end_date,
        "backtest_start_date": backtest_start.strftime("%Y-%m-%d"),
        "latest_observed_price_date": latest_price.strftime("%Y-%m-%d") if pd.notna(latest_price) else None,
        "current_model_tickers": int(current["Ticker"].nunique()),
        "legacy_removed_tickers": int(len(LEGACY_REMOVED_TICKERS)),
        "included_universe_rows": int(len(universe)),
        "audit_rows": int(len(audit)),
        "invalid_excluded_rows": int((audit["Universe_Inclusion"] == "excluded").sum()),
        "status_counts": audit["Status"].value_counts(dropna=False).to_dict(),
        "note": (
            "This is a reproducible project historical-universe seed. It is better than a "
            "current-only universe, but it is not a complete vendor-grade security master."
        ),
    }
    return universe, audit, metadata


def write_notes(path: Path, metadata: dict, universe: pd.DataFrame, audit: pd.DataFrame) -> None:
    status_counts = audit["Status"].value_counts(dropna=False)
    lines = [
        "# Historical Tech Universe Notes",
        "",
        "This file documents the first historical-universe source for the Stock Analysis project.",
        "",
        "## What This Contains",
        "",
        "- Current active model tickers from `comp.REGIME_GROUPS`.",
        "- Legacy tickers recovered from git commit `1879e1b Remove stale stock tickers from universe`.",
        "- Membership windows bounded by observed Yahoo Finance daily price history.",
        "- Source/status fields so future backtests can separate active, removed, inactive, and invalid tickers.",
        "",
        "## Files",
        "",
        "- `data/historical_tech_universe.csv`: included membership rows for simulator input.",
        "- `data/historical_tech_universe_audit.csv`: included rows plus excluded invalid ticker events.",
        "- `data/historical_tech_universe_notes.md`: this explanation.",
        "",
        "## Run Summary",
        "",
        f"- Current model tickers: {metadata.get('current_model_tickers')}",
        f"- Legacy removed ticker seeds: {metadata.get('legacy_removed_tickers')}",
        f"- Included universe rows: {metadata.get('included_universe_rows')}",
        f"- Audit rows: {metadata.get('audit_rows')}",
        f"- Invalid excluded rows: {metadata.get('invalid_excluded_rows')}",
        f"- Backtest start date: {metadata.get('backtest_start_date')}",
        f"- Latest observed price date: {metadata.get('latest_observed_price_date')}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in status_counts.items():
        lines.append(f"- `{status}`: {int(count)}")

    legacy_sample = universe[~universe["Is_Current_Model_Ticker"].astype(bool)].head(30)
    lines.extend([
        "",
        "## Legacy Seed Rows",
        "",
    ])
    if legacy_sample.empty:
        lines.append("No legacy seed rows were included.")
    else:
        for _, row in legacy_sample.iterrows():
            end = row["Membership_End_Date"] if str(row["Membership_End_Date"]).strip() else "open"
            lines.append(
                f"- `{row['Ticker']}` | {row['SubIndustry']} | {row['Status']} | "
                f"{row['Membership_Start_Date']} to {end}"
            )

    lines.extend([
        "",
        "## Important Limitation",
        "",
        "This is not a full survivorship-bias fix by itself. It adds known legacy tickers from this project's prior universe and prevents the simulator from pretending a stock existed before its observed price history. It does not yet include every technology company that was delisted, acquired, merged, renamed, or removed before this project tracked it.",
        "",
        "The next stronger version should use a vendor-grade historical security master or an exchange/security dataset with historical listings, delist dates, corporate-action mappings, and sector/subindustry history.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build historical technology universe CSV.")
    parser.add_argument("--score-history", type=Path, default=DEFAULT_SCORE_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=(pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    parser.add_argument("--batch-size", type=int, default=35)
    args = parser.parse_args()

    universe, audit, metadata = build_universe(
        score_history=args.score_history,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(universe.to_csv(index=False), encoding="utf-8")
    args.audit.write_text(audit.to_csv(index=False), encoding="utf-8")
    write_notes(args.notes, metadata, universe, audit)
    (args.output.with_suffix(".metadata.json")).write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== HISTORICAL TECH UNIVERSE ===", flush=True)
    print(json.dumps(metadata, indent=2, default=str), flush=True)
    print(f"[SUCCESS] Saved {args.output}", flush=True)
    print(f"[SUCCESS] Saved {args.audit}", flush=True)
    print(f"[SUCCESS] Saved {args.notes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
