"""Yahoo Finance fallback adapter for the historical data layer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .base import (
    CANONICAL_ACTION_COLUMNS,
    CANONICAL_DAILY_FUNDAMENTAL_COLUMNS,
    CANONICAL_FUNDAMENTAL_COLUMNS,
    CANONICAL_SECURITY_MASTER_COLUMNS,
    HistoricalDataProvider,
    empty_frame,
    ensure_columns,
    normalize_ticker,
    normalize_ticker_list,
)


class YahooDataProvider(HistoricalDataProvider):
    provider_name = "yahoo"

    def __init__(self, universe_path: Path | None = None, batch_size: int = 25):
        self.universe_path = universe_path
        self.batch_size = batch_size

    def provider_status(self) -> dict:
        return {
            "provider": self.provider_name,
            "survivorship_bias_free": False,
            "point_in_time_fundamentals": False,
            "delisted_prices": False,
            "required_files_present": True,
            "notes": (
                "Compatibility fallback only. Yahoo/yfinance is useful for current prices "
                "and smoke tests, but not for vendor-grade historical universe backtests."
            ),
        }

    def load_security_master(self) -> pd.DataFrame:
        if self.universe_path is None or not self.universe_path.exists():
            return empty_frame(CANONICAL_SECURITY_MASTER_COLUMNS)
        raw = pd.read_csv(self.universe_path)
        if "Ticker" not in raw.columns:
            return empty_frame(CANONICAL_SECURITY_MASTER_COLUMNS)

        out = pd.DataFrame(index=raw.index)
        out["Provider"] = self.provider_name
        out["Ticker"] = raw["Ticker"].map(normalize_ticker)
        out["Provider_Security_ID"] = out["Ticker"]
        out["Company_Name"] = raw.get("Company_Name", pd.Series(pd.NA, index=raw.index))
        out["Exchange"] = raw.get("Exchange", pd.Series(pd.NA, index=raw.index))
        out["Currency"] = "USD"
        out["Sector"] = raw.get("Sector", "Technology")
        out["Industry"] = raw.get("Industry", pd.Series(pd.NA, index=raw.index))
        out["SubIndustry"] = raw.get("SubIndustry", pd.Series(pd.NA, index=raw.index))
        out["SIC_Code"] = pd.NA
        out["SIC_Sector"] = pd.NA
        out["SIC_Industry"] = pd.NA
        out["CIK"] = raw.get("CIK", pd.Series(pd.NA, index=raw.index))
        out["FIGI"] = pd.NA
        status = raw.get("Status", pd.Series("", index=raw.index)).astype(str).str.lower()
        out["Is_Delisted"] = status.str.contains("delist|legacy|removed", regex=True)
        out["First_Price_Date"] = raw.get("Membership_Start_Date", pd.Series(pd.NA, index=raw.index))
        out["Last_Price_Date"] = raw.get("Membership_End_Date", pd.Series(pd.NA, index=raw.index))
        out["First_Fundamental_Date"] = pd.NA
        out["Last_Fundamental_Date"] = pd.NA
        out["Source_Quality"] = "project_universe_proxy_not_vendor_grade"
        out = out[out["Ticker"] != ""].drop_duplicates("Ticker", keep="last")
        return ensure_columns(out.reset_index(drop=True), CANONICAL_SECURITY_MASTER_COLUMNS)

    def load_prices(
        self,
        tickers: Iterable[str],
        start_date: object,
        end_date: object,
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        import time
        import yfinance as yf

        ticker_list = normalize_ticker_list(tickers)
        result: dict[str, pd.DataFrame] = {}
        if not ticker_list:
            return result

        for start in range(0, len(ticker_list), self.batch_size):
            batch = ticker_list[start:start + self.batch_size]
            for attempt in range(1, 4):
                try:
                    raw = yf.download(
                        tickers=batch,
                        start=str(start_date),
                        end=str(end_date),
                        interval="1d",
                        group_by="ticker",
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                    )
                    loaded = _parse_yahoo_download(raw, batch, adjusted=adjusted)
                    result.update(loaded)
                    break
                except Exception:
                    if attempt == 3:
                        break
                    time.sleep(0.5 * attempt)
        return result

    def load_fundamentals(
        self,
        tickers: Iterable[str],
        start_date: object,
        end_date: object,
        dimension: str = "ARQ",
    ) -> pd.DataFrame:
        return empty_frame(CANONICAL_FUNDAMENTAL_COLUMNS)

    def load_daily_fundamentals(
        self,
        tickers: Iterable[str],
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        return empty_frame(CANONICAL_DAILY_FUNDAMENTAL_COLUMNS)

    def load_corporate_actions(
        self,
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        return empty_frame(CANONICAL_ACTION_COLUMNS)


def _parse_yahoo_download(
    raw: pd.DataFrame,
    tickers: list[str],
    adjusted: bool,
) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return loaded

    if isinstance(raw.columns, pd.MultiIndex):
        first_level = set(raw.columns.get_level_values(0))
        second_level = set(raw.columns.get_level_values(1))
        for ticker in tickers:
            ticker_df = None
            if ticker in first_level:
                ticker_df = raw[ticker]
            elif ticker in second_level:
                ticker_df = raw.xs(ticker, axis=1, level=1)
            if ticker_df is not None:
                parsed = _extract_price_frame(ticker_df, adjusted=adjusted)
                if parsed is not None:
                    loaded[ticker] = parsed
    elif tickers:
        parsed = _extract_price_frame(raw, adjusted=adjusted)
        if parsed is not None:
            loaded[tickers[0]] = parsed

    return loaded


def _extract_price_frame(df: pd.DataFrame, adjusted: bool) -> pd.DataFrame | None:
    close_column = "Adj Close" if adjusted and "Adj Close" in df.columns else "Close"
    if close_column not in df.columns:
        return None
    out = pd.DataFrame(index=pd.to_datetime(df.index).normalize())
    out["open"] = pd.to_numeric(df.get("Open"), errors="coerce")
    out["high"] = pd.to_numeric(df.get("High"), errors="coerce")
    out["low"] = pd.to_numeric(df.get("Low"), errors="coerce")
    out["close"] = pd.to_numeric(df[close_column], errors="coerce")
    out["close_unadjusted"] = pd.to_numeric(df.get("Close"), errors="coerce")
    out["volume"] = pd.to_numeric(df.get("Volume"), errors="coerce")
    out["provider"] = "yahoo"
    out = out.dropna(subset=["close"])
    if out.empty:
        return None
    return out[~out.index.duplicated(keep="last")].sort_index()
