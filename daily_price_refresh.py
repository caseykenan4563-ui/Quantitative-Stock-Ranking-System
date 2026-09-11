#!/usr/bin/env python3
"""Refresh recent canonical prices for the active forward paper test.

This updater is intentionally narrow. It updates only the active paper-test
holdings plus QQQ and XLK so the forward ledger can accumulate unseen daily
marks without rebuilding the whole historical Sharadar data layer.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_sources.base import CANONICAL_PRICE_COLUMNS, ensure_columns, normalize_ticker_list
from historical_data_layer import price_dict_to_canonical
from data_sources.yahoo import YahooDataProvider


DEFAULT_PRICE_FILE = Path("data/historical_data_layer/prices_daily.csv")
DEFAULT_POSITIONS_FILE = Path("paper_tests/production_default_paper_positions.csv")
DEFAULT_HOLDINGS_FILE = Path("website_assets/production-default-latest-holdings.csv")
DEFAULT_REPORT_FILE = Path("paper_tests/latest_daily_price_refresh.json")
DEFAULT_BENCHMARKS = "QQQ,XLK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh recent paper-test daily prices.")
    parser.add_argument("--price-file", type=Path, default=DEFAULT_PRICE_FILE)
    parser.add_argument("--positions-file", type=Path, default=DEFAULT_POSITIONS_FILE)
    parser.add_argument("--holdings-file", type=Path, default=DEFAULT_HOLDINGS_FILE)
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--start-date", default="", help="Optional inclusive override, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="", help="Optional exclusive override, YYYY-MM-DD.")
    parser.add_argument("--min-new-date", default="", help="Optional inclusive minimum date for accepted new rows.")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def read_active_tickers(positions_file: Path, holdings_file: Path, benchmarks: Iterable[str]) -> list[str]:
    tickers: list[str] = []
    if positions_file.exists():
        positions = pd.read_csv(positions_file)
        if "Ticker" in positions.columns:
            tickers.extend(positions["Ticker"].dropna().astype(str).tolist())

    if not tickers and holdings_file.exists():
        holdings = pd.read_csv(holdings_file)
        if "Ticker" in holdings.columns:
            tickers.extend(holdings["Ticker"].dropna().astype(str).tolist())

    tickers.extend(list(benchmarks))
    out = normalize_ticker_list(tickers)
    if not out:
        raise RuntimeError("No active tickers found. Run generate_website_assets.py and paper_test_tracker.py first.")
    return out


def load_price_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Canonical price file not found: {path}")
    frame = pd.read_csv(path)
    required = {"Ticker", "Date", "Close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Canonical price file missing required columns: {missing}")
    frame["Ticker"] = frame["Ticker"].astype(str).str.upper().str.strip()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["Ticker", "Date", "Close"]).copy()
    return frame


def latest_existing_dates(frame: pd.DataFrame, tickers: Iterable[str]) -> dict[str, pd.Timestamp]:
    selected = normalize_ticker_list(tickers)
    subset = frame[frame["Ticker"].isin(selected)].copy()
    if subset.empty:
        return {}
    return subset.groupby("Ticker")["Date"].max().to_dict()


def default_start_date(existing_dates: dict[str, pd.Timestamp]) -> pd.Timestamp:
    if not existing_dates:
        return pd.Timestamp(datetime.now(UTC).date() - timedelta(days=10))
    earliest_latest = min(pd.to_datetime(value).normalize() for value in existing_dates.values())
    return earliest_latest + pd.Timedelta(days=1)


def default_end_date() -> pd.Timestamp:
    # yfinance treats end as exclusive, so request through tomorrow.
    return pd.Timestamp(datetime.now(UTC).date() + timedelta(days=1))


def fetch_yahoo_prices(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    batch_size: int,
) -> pd.DataFrame:
    provider = YahooDataProvider(batch_size=batch_size)
    price_data = provider.load_prices(
        tickers=tickers,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        adjusted=True,
    )
    prices = price_dict_to_canonical(price_data, "yahoo_daily_refresh", adjusted=True)
    if prices.empty:
        return prices
    prices["Last_Updated"] = utc_now()
    return ensure_columns(prices, CANONICAL_PRICE_COLUMNS)


def filter_new_rows(
    new_prices: pd.DataFrame,
    existing_dates: dict[str, pd.Timestamp],
    min_new_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if new_prices.empty:
        return new_prices

    out = new_prices.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out = out.dropna(subset=["Ticker", "Date", "Close"]).copy()

    def is_new(row: pd.Series) -> bool:
        ticker = row["Ticker"]
        row_date = pd.to_datetime(row["Date"]).normalize()
        if min_new_date is not None and row_date < min_new_date:
            return False
        latest = existing_dates.get(ticker)
        return latest is None or row_date > pd.to_datetime(latest).normalize()

    return out[out.apply(is_new, axis=1)].copy()


def merge_prices(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if new_rows.empty:
        return existing

    all_columns = list(dict.fromkeys(list(existing.columns) + list(new_rows.columns)))
    merged = pd.concat(
        [
            existing.reindex(columns=all_columns),
            new_rows.reindex(columns=all_columns),
        ],
        ignore_index=True,
        sort=False,
    )
    merged["Ticker"] = merged["Ticker"].astype(str).str.upper().str.strip()
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = (
        merged
        .dropna(subset=["Ticker", "Date", "Close"])
        .drop_duplicates(["Ticker", "Date"], keep="last")
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )
    return merged


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def write_prices_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    benchmarks = normalize_ticker_list(args.benchmarks.split(","))
    tickers = read_active_tickers(args.positions_file, args.holdings_file, benchmarks)
    existing = load_price_file(args.price_file)
    existing_dates = latest_existing_dates(existing, tickers)

    start_date = (
        pd.to_datetime(args.start_date).normalize()
        if args.start_date else default_start_date(existing_dates)
    )
    end_date = (
        pd.to_datetime(args.end_date).normalize()
        if args.end_date else default_end_date()
    )
    min_new_date = pd.to_datetime(args.min_new_date).normalize() if args.min_new_date else start_date
    if pd.isna(start_date) or pd.isna(end_date):
        raise RuntimeError("Invalid start or end date")
    if end_date <= start_date:
        print("[INFO] No refresh needed; end date is not after start date")
        return 0

    fetched = fetch_yahoo_prices(tickers, start_date, end_date, batch_size=args.batch_size)
    new_rows = filter_new_rows(fetched, existing_dates, min_new_date=min_new_date)
    merged = merge_prices(existing, new_rows)

    latest_after = latest_existing_dates(merged, tickers)
    report = {
        "status": "dry_run" if args.dry_run else "complete",
        "updated_at": utc_now(),
        "provider": "yahoo_daily_refresh",
        "price_file": str(args.price_file),
        "positions_file": str(args.positions_file),
        "tickers": tickers,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date_exclusive": end_date.strftime("%Y-%m-%d"),
        "fetched_rows": int(len(fetched)),
        "accepted_new_rows": int(len(new_rows)),
        "existing_rows": int(len(existing)),
        "merged_rows": int(len(merged)),
        "latest_dates_before": {
            ticker: pd.to_datetime(value).strftime("%Y-%m-%d")
            for ticker, value in existing_dates.items()
        },
        "latest_dates_after": {
            ticker: pd.to_datetime(value).strftime("%Y-%m-%d")
            for ticker, value in latest_after.items()
        },
    }

    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    write_prices_atomic(args.price_file, merged)
    write_report(args.report_file, report)

    print("[SUCCESS] Daily paper-test prices refreshed")
    print(json.dumps({
        "tickers": len(tickers),
        "fetched_rows": report["fetched_rows"],
        "accepted_new_rows": report["accepted_new_rows"],
        "price_file": str(args.price_file),
        "report_file": str(args.report_file),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
