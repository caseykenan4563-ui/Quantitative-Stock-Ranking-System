#!/usr/bin/env python3
"""Build canonical historical data files for the Stock Analysis pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_sources.base import (
    CANONICAL_ACTION_COLUMNS,
    CANONICAL_DAILY_FUNDAMENTAL_COLUMNS,
    CANONICAL_FUNDAMENTAL_COLUMNS,
    CANONICAL_PRICE_COLUMNS,
    CANONICAL_SECURITY_MASTER_COLUMNS,
    empty_frame,
    ensure_columns,
    filter_by_date_range,
    filter_by_tickers,
    normalize_ticker_list,
    write_csv,
)
from data_sources.sharadar_csv import SharadarCsvPaths, SharadarCsvProvider
from data_sources.yahoo import YahooDataProvider


DEFAULT_OUTPUT_DIR = Path("data/historical_data_layer")
DEFAULT_HISTORICAL_UNIVERSE = Path("data/historical_tech_universe.csv")
DEFAULT_BENCHMARK_TICKERS = "QQQ,XLK"
SECURITY_MASTER_FILE = "security_master.csv"
PRICE_FILE = "prices_daily.csv"
FUNDAMENTALS_FILE = "fundamentals_quarterly.csv"
DAILY_FUNDAMENTALS_FILE = "daily_fundamentals.csv"
CORPORATE_ACTIONS_FILE = "corporate_actions.csv"
QUALITY_REPORT_FILE = "data_quality_report.csv"
METADATA_FILE = "provider_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build provider-neutral historical market data files. Use sharadar-csv "
            "for the real production historical layer; yahoo is a fallback smoke-test provider."
        )
    )
    parser.add_argument("--provider", choices=["sharadar-csv", "yahoo"], default="sharadar-csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--historical-universe", type=Path, default=DEFAULT_HISTORICAL_UNIVERSE)
    parser.add_argument("--tickers", default="", help="Comma-separated ticker override.")
    parser.add_argument(
        "--benchmark-tickers",
        default=DEFAULT_BENCHMARK_TICKERS,
        help="Comma-separated benchmark tickers to include in canonical prices, not fair-value fundamentals.",
    )
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--dimension", default="ART", help="Sharadar fundamentals dimension. Prefer ART for fair value.")
    parser.add_argument("--all-provider-tickers", action="store_true")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--adjusted", action="store_true", default=True)
    parser.add_argument("--unadjusted", dest="adjusted", action="store_false")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--sharadar-tickers-csv", type=Path)
    parser.add_argument("--sharadar-stocks-csv", type=Path)
    parser.add_argument("--sharadar-funds-csv", type=Path)
    parser.add_argument("--sharadar-fundamentals-csv", type=Path)
    parser.add_argument("--sharadar-daily-csv", type=Path)
    parser.add_argument("--sharadar-actions-csv", type=Path)
    return parser.parse_args()


def build_provider(args: argparse.Namespace):
    if args.provider == "yahoo":
        return YahooDataProvider(universe_path=args.historical_universe, batch_size=args.batch_size)

    paths = SharadarCsvPaths.from_env().with_overrides(
        tickers=args.sharadar_tickers_csv,
        stocks=args.sharadar_stocks_csv,
        funds=args.sharadar_funds_csv,
        fundamentals=args.sharadar_fundamentals_csv,
        daily=args.sharadar_daily_csv,
        actions=args.sharadar_actions_csv,
    )
    return SharadarCsvProvider(paths=paths)


def selected_tickers(args: argparse.Namespace, security_master: pd.DataFrame | None = None) -> list[str]:
    if args.tickers.strip():
        return normalize_ticker_list(args.tickers.split(","))
    if args.all_provider_tickers and security_master is not None and "Ticker" in security_master.columns:
        return normalize_ticker_list(security_master["Ticker"].dropna().astype(str).tolist())
    if args.historical_universe.exists():
        universe = pd.read_csv(args.historical_universe)
        if "Ticker" in universe.columns:
            return normalize_ticker_list(universe["Ticker"].dropna().astype(str).tolist())
    return []


def price_dict_to_canonical(
    price_data: dict[str, pd.DataFrame],
    provider_name: str,
    adjusted: bool,
) -> pd.DataFrame:
    frames = []
    for ticker, frame in price_data.items():
        if frame.empty:
            continue
        tmp = frame.copy()
        tmp.index = pd.to_datetime(tmp.index, errors="coerce").normalize()
        tmp = tmp[~tmp.index.isna()].copy()
        tmp["Provider"] = provider_name
        tmp["Ticker"] = str(ticker).upper().strip()
        tmp["Date"] = tmp.index
        tmp["Open"] = pd.to_numeric(tmp.get("open"), errors="coerce")
        tmp["High"] = pd.to_numeric(tmp.get("high"), errors="coerce")
        tmp["Low"] = pd.to_numeric(tmp.get("low"), errors="coerce")
        tmp["Close"] = pd.to_numeric(tmp.get("close"), errors="coerce")
        tmp["Close_Unadjusted"] = pd.to_numeric(tmp.get("close_unadjusted"), errors="coerce")
        tmp["Volume"] = pd.to_numeric(tmp.get("volume"), errors="coerce")
        tmp["Adjustment_Mode"] = "adjusted" if adjusted else "unadjusted"
        tmp["Last_Updated"] = pd.NA
        frames.append(ensure_columns(tmp.reset_index(drop=True), CANONICAL_PRICE_COLUMNS))
    if not frames:
        return empty_frame(CANONICAL_PRICE_COLUMNS)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["Ticker", "Date", "Close"]).copy()
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def load_canonical_prices(
    path: Path,
    tickers: Iterable[str] | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Canonical historical price file not found: {path}")
    frame = pd.read_csv(path)
    required = {"Ticker", "Date", "Close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Canonical historical price file missing required columns: {missing}")

    frame["Ticker"] = frame["Ticker"].astype(str).str.upper().str.strip()
    frame = filter_by_tickers(frame, tickers)
    frame = filter_by_date_range(frame, "Date", start_date, end_date)
    result: dict[str, pd.DataFrame] = {}
    for ticker, group in frame.groupby("Ticker", sort=True):
        dates = pd.to_datetime(group["Date"], errors="coerce").dt.normalize()
        out = pd.DataFrame(index=dates)
        out["open"] = pd.to_numeric(group.get("Open"), errors="coerce").to_numpy()
        out["high"] = pd.to_numeric(group.get("High"), errors="coerce").to_numpy()
        out["low"] = pd.to_numeric(group.get("Low"), errors="coerce").to_numpy()
        out["close"] = pd.to_numeric(group["Close"], errors="coerce").to_numpy()
        out["close_unadjusted"] = pd.to_numeric(group.get("Close_Unadjusted"), errors="coerce").to_numpy()
        out["volume"] = pd.to_numeric(group.get("Volume"), errors="coerce").to_numpy()
        out["provider"] = group.get("Provider", pd.Series("unknown", index=group.index)).astype(str).to_numpy()
        out = out.dropna(subset=["close"])
        if not out.empty:
            result[str(ticker)] = out[~out.index.duplicated(keep="last")].sort_index()
    return result


def load_canonical_price_wide(
    path: Path,
    tickers: Iterable[str] | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
    value_column: str = "Close",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Canonical historical price file not found: {path}")
    frame = pd.read_csv(path)
    required = {"Ticker", "Date", value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Canonical historical price file missing required columns: {missing}")
    frame["Ticker"] = frame["Ticker"].astype(str).str.upper().str.strip()
    frame = filter_by_tickers(frame, tickers)
    frame = filter_by_date_range(frame, "Date", start_date, end_date)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    wide = frame.pivot_table(index="Date", columns="Ticker", values=value_column, aggfunc="last")
    wide.index = pd.to_datetime(wide.index, errors="coerce").normalize()
    return wide.sort_index()


def write_schema_only(output_dir: Path, provider_status: dict, args: argparse.Namespace) -> None:
    write_csv(empty_frame(CANONICAL_SECURITY_MASTER_COLUMNS), output_dir / SECURITY_MASTER_FILE)
    write_csv(empty_frame(CANONICAL_PRICE_COLUMNS), output_dir / PRICE_FILE)
    write_csv(empty_frame(CANONICAL_FUNDAMENTAL_COLUMNS), output_dir / FUNDAMENTALS_FILE)
    write_csv(empty_frame(CANONICAL_DAILY_FUNDAMENTAL_COLUMNS), output_dir / DAILY_FUNDAMENTALS_FILE)
    write_csv(empty_frame(CANONICAL_ACTION_COLUMNS), output_dir / CORPORATE_ACTIONS_FILE)
    quality = pd.DataFrame([
        {"Metric": "schema_only", "Value": True},
        {"Metric": "provider", "Value": provider_status.get("provider")},
        {"Metric": "point_in_time_fundamentals", "Value": provider_status.get("point_in_time_fundamentals")},
        {"Metric": "delisted_prices", "Value": provider_status.get("delisted_prices")},
    ])
    write_csv(quality, output_dir / QUALITY_REPORT_FILE)
    write_metadata(output_dir, provider_status, args, tickers=[], files_written=schema_files(output_dir))


def schema_files(output_dir: Path) -> dict[str, str]:
    return {
        "security_master": str(output_dir / SECURITY_MASTER_FILE),
        "prices_daily": str(output_dir / PRICE_FILE),
        "fundamentals_quarterly": str(output_dir / FUNDAMENTALS_FILE),
        "daily_fundamentals": str(output_dir / DAILY_FUNDAMENTALS_FILE),
        "corporate_actions": str(output_dir / CORPORATE_ACTIONS_FILE),
        "quality_report": str(output_dir / QUALITY_REPORT_FILE),
        "metadata": str(output_dir / METADATA_FILE),
    }


def build_quality_report(
    security_master: pd.DataFrame,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    daily_fundamentals: pd.DataFrame,
    actions: pd.DataFrame,
    tickers: list[str],
    provider_status: dict,
) -> pd.DataFrame:
    rows = [
        ("provider", provider_status.get("provider")),
        ("survivorship_bias_free", provider_status.get("survivorship_bias_free")),
        ("point_in_time_fundamentals", provider_status.get("point_in_time_fundamentals")),
        ("delisted_prices", provider_status.get("delisted_prices")),
        ("tickers_requested", len(tickers)),
        ("security_master_rows", len(security_master)),
        ("security_master_unique_tickers", _nunique(security_master, "Ticker")),
        ("security_master_delisted_rows", _sum_bool(security_master, "Is_Delisted")),
        ("price_rows", len(prices)),
        ("price_unique_tickers", _nunique(prices, "Ticker")),
        ("price_first_date", _min_date(prices, "Date")),
        ("price_last_date", _max_date(prices, "Date")),
        ("fundamental_rows", len(fundamentals)),
        ("fundamental_unique_tickers", _nunique(fundamentals, "Ticker")),
        ("fundamental_first_filing_date", _min_date(fundamentals, "Filing_Date")),
        ("fundamental_last_filing_date", _max_date(fundamentals, "Filing_Date")),
        ("daily_fundamental_rows", len(daily_fundamentals)),
        ("corporate_action_rows", len(actions)),
    ]
    if tickers and not prices.empty and "Ticker" in prices.columns:
        missing_prices = sorted(set(tickers) - set(prices["Ticker"].dropna().astype(str)))
        rows.append(("requested_tickers_missing_prices", len(missing_prices)))
        rows.append(("sample_missing_price_tickers", ", ".join(missing_prices[:25])))
    if tickers and not fundamentals.empty and "Ticker" in fundamentals.columns:
        missing_fundamentals = sorted(set(tickers) - set(fundamentals["Ticker"].dropna().astype(str)))
        rows.append(("requested_tickers_missing_fundamentals", len(missing_fundamentals)))
        rows.append(("sample_missing_fundamental_tickers", ", ".join(missing_fundamentals[:25])))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def write_metadata(
    output_dir: Path,
    provider_status: dict,
    args: argparse.Namespace,
    tickers: list[str],
    files_written: dict[str, str],
) -> None:
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_status": provider_status,
        "provider": args.provider,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "dimension": args.dimension,
        "adjusted_prices": args.adjusted,
        "ticker_count": len(tickers),
        "benchmark_tickers": provider_status.get("benchmark_tickers", []),
        "historical_universe": str(args.historical_universe),
        "all_provider_tickers": bool(args.all_provider_tickers),
        "schema_only": bool(args.schema_only),
        "files": files_written,
        "point_in_time_rule": (
            "Use Filing_Date <= anchor date and ARQ/ART dimensions for as-reported fundamentals."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / METADATA_FILE).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _nunique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].nunique(dropna=True))


def _sum_bool(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _min_date(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    value = pd.to_datetime(frame[column], errors="coerce").min()
    return "" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _max_date(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    value = pd.to_datetime(frame[column], errors="coerce").max()
    return "" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def main() -> None:
    args = parse_args()
    provider = build_provider(args)
    provider_status = provider.provider_status()

    if args.schema_only:
        write_schema_only(args.output_dir, provider_status, args)
        print(f"[SUCCESS] Wrote canonical schema templates to {args.output_dir}", flush=True)
        return

    print(f"[INFO] Loading security master from {provider.provider_name}", flush=True)
    security_master = provider.load_security_master()
    tickers = selected_tickers(args, security_master)
    benchmark_tickers = normalize_ticker_list(args.benchmark_tickers.split(","))
    price_tickers = normalize_ticker_list(tickers + benchmark_tickers)
    provider_status["benchmark_tickers"] = benchmark_tickers
    provider_status["price_ticker_count"] = len(price_tickers)
    if not tickers:
        raise RuntimeError("No tickers selected. Pass --tickers or provide data/historical_tech_universe.csv.")

    if not args.all_provider_tickers:
        security_master = filter_by_tickers(security_master, price_tickers)

    print(f"[INFO] Selected {len(tickers)} tickers", flush=True)
    if benchmark_tickers:
        print(f"[INFO] Including benchmark price tickers: {', '.join(benchmark_tickers)}", flush=True)
    print(f"[INFO] Loading daily prices {args.start_date} -> {args.end_date}", flush=True)
    price_data = provider.load_prices(price_tickers, args.start_date, args.end_date, adjusted=args.adjusted)
    prices = price_dict_to_canonical(price_data, provider.provider_name, adjusted=args.adjusted)

    print(f"[INFO] Loading fundamentals dimension={args.dimension}", flush=True)
    fundamentals = provider.load_fundamentals(tickers, args.start_date, args.end_date, dimension=args.dimension)

    print("[INFO] Loading daily fundamentals", flush=True)
    daily_fundamentals = provider.load_daily_fundamentals(tickers, args.start_date, args.end_date)

    print("[INFO] Loading corporate actions", flush=True)
    actions = provider.load_corporate_actions(args.start_date, args.end_date)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(ensure_columns(security_master, CANONICAL_SECURITY_MASTER_COLUMNS), args.output_dir / SECURITY_MASTER_FILE)
    write_csv(ensure_columns(prices, CANONICAL_PRICE_COLUMNS), args.output_dir / PRICE_FILE)
    write_csv(ensure_columns(fundamentals, CANONICAL_FUNDAMENTAL_COLUMNS), args.output_dir / FUNDAMENTALS_FILE)
    write_csv(
        ensure_columns(daily_fundamentals, CANONICAL_DAILY_FUNDAMENTAL_COLUMNS),
        args.output_dir / DAILY_FUNDAMENTALS_FILE,
    )
    write_csv(ensure_columns(actions, CANONICAL_ACTION_COLUMNS), args.output_dir / CORPORATE_ACTIONS_FILE)

    quality = build_quality_report(
        security_master=security_master,
        prices=prices,
        fundamentals=fundamentals,
        daily_fundamentals=daily_fundamentals,
        actions=actions,
        tickers=tickers,
        provider_status=provider_status,
    )
    write_csv(quality, args.output_dir / QUALITY_REPORT_FILE)
    write_metadata(args.output_dir, provider_status, args, tickers, schema_files(args.output_dir))

    print("", flush=True)
    print("=== HISTORICAL DATA LAYER SUMMARY ===", flush=True)
    print(f"Provider: {provider.provider_name}", flush=True)
    print(f"Tickers selected: {len(tickers)}", flush=True)
    print(f"Security master rows: {len(security_master)}", flush=True)
    print(f"Price rows: {len(prices)}", flush=True)
    print(f"Fundamental rows: {len(fundamentals)}", flush=True)
    print(f"Daily fundamental rows: {len(daily_fundamentals)}", flush=True)
    print(f"Corporate action rows: {len(actions)}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    print("=====================================", flush=True)


if __name__ == "__main__":
    main()
