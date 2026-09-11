"""Sharadar bulk CSV adapter for the historical data layer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .base import (
    CANONICAL_ACTION_COLUMNS,
    CANONICAL_DAILY_FUNDAMENTAL_COLUMNS,
    CANONICAL_FUNDAMENTAL_COLUMNS,
    CANONICAL_PRICE_COLUMNS,
    CANONICAL_SECURITY_MASTER_COLUMNS,
    HistoricalDataProvider,
    empty_frame,
    ensure_columns,
    filter_by_date_range,
    filter_by_tickers,
    normalize_ticker,
)


DEFAULT_SHARADAR_DIR = Path("data/vendor/sharadar")
DEFAULT_CHUNK_ROWS = 200_000


def _default_path(filename: str) -> Path:
    return DEFAULT_SHARADAR_DIR / filename


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


@dataclass(frozen=True)
class SharadarCsvPaths:
    tickers: Path | None = None
    stocks: Path | None = None
    funds: Path | None = None
    fundamentals: Path | None = None
    daily: Path | None = None
    actions: Path | None = None

    @classmethod
    def from_env(cls) -> "SharadarCsvPaths":
        return cls(
            tickers=_env_path("SHARADAR_TICKERS_CSV") or _default_path("tickers.csv"),
            stocks=_env_path("SHARADAR_STOCKS_CSV") or _default_path("stocks.csv"),
            funds=_env_path("SHARADAR_FUNDS_CSV") or _default_path("funds.csv"),
            fundamentals=_env_path("SHARADAR_FUNDAMENTALS_CSV") or _default_path("fundamentals.csv"),
            daily=_env_path("SHARADAR_DAILY_CSV") or _default_path("daily.csv"),
            actions=_env_path("SHARADAR_ACTIONS_CSV") or _default_path("actions.csv"),
        )

    def with_overrides(
        self,
        tickers: Path | None = None,
        stocks: Path | None = None,
        funds: Path | None = None,
        fundamentals: Path | None = None,
        daily: Path | None = None,
        actions: Path | None = None,
    ) -> "SharadarCsvPaths":
        return SharadarCsvPaths(
            tickers=tickers or self.tickers,
            stocks=stocks or self.stocks,
            funds=funds or self.funds,
            fundamentals=fundamentals or self.fundamentals,
            daily=daily or self.daily,
            actions=actions or self.actions,
        )

    def present(self) -> dict[str, str]:
        values = {}
        for name, path in self.__dict__.items():
            values[name] = str(path) if path else ""
        return values


class SharadarCsvProvider(HistoricalDataProvider):
    provider_name = "sharadar_csv"

    def __init__(self, paths: SharadarCsvPaths | None = None):
        self.paths = paths or SharadarCsvPaths.from_env()
        self.chunk_rows = _env_int("SHARADAR_CSV_CHUNK_ROWS", DEFAULT_CHUNK_ROWS)

    def provider_status(self) -> dict:
        return {
            "provider": self.provider_name,
            "survivorship_bias_free": True,
            "point_in_time_fundamentals": True,
            "delisted_prices": True,
            "required_files_present": self.required_files_present(),
            "csv_paths": self.paths.present(),
            "notes": (
                "Uses Sharadar bulk CSVs. Prefer fundamentals dimension ARQ/ART "
                "for point-in-time/as-reported backtests. Reads funds.csv when "
                "available so ETF benchmarks such as QQQ/XLK can be included."
            ),
        }

    def required_files_present(self) -> bool:
        required = [self.paths.tickers, self.paths.stocks, self.paths.fundamentals]
        return all(path is not None and path.exists() for path in required)

    def load_security_master(self) -> pd.DataFrame:
        frame = self._read_required(self.paths.tickers, "tickers")
        lookup = _column_lookup(frame)
        out = pd.DataFrame(index=frame.index)
        out["Provider"] = self.provider_name
        out["Provider_Security_ID"] = _col(frame, lookup, "permaticker")
        out["Ticker"] = _col(frame, lookup, "ticker").map(normalize_ticker)
        out["Company_Name"] = _col(frame, lookup, "name")
        out["Exchange"] = _col(frame, lookup, "exchange")
        out["Currency"] = _col(frame, lookup, "currency").fillna("USD")
        out["Sector"] = _col(frame, lookup, "sector").fillna(_col(frame, lookup, "sicsector"))
        out["Industry"] = _col(frame, lookup, "industry").fillna(_col(frame, lookup, "sicindustry"))
        out["SubIndustry"] = out["Industry"]
        out["SIC_Code"] = _col(frame, lookup, "siccode")
        out["SIC_Sector"] = _col(frame, lookup, "sicsector")
        out["SIC_Industry"] = _col(frame, lookup, "sicindustry")
        out["CIK"] = _col(frame, lookup, "cik")
        out["FIGI"] = _col(frame, lookup, "figi")
        out["Is_Delisted"] = _col(frame, lookup, "isdelisted").astype(str).str.upper().eq("Y")
        out["First_Price_Date"] = _date_text(_col(frame, lookup, "firstpricedate"))
        out["Last_Price_Date"] = _date_text(_col(frame, lookup, "lastpricedate"))
        out["First_Fundamental_Date"] = _date_text(_col(frame, lookup, "firstquarter"))
        out["Last_Fundamental_Date"] = _date_text(_col(frame, lookup, "lastquarter"))
        out["Source_Quality"] = "vendor_security_master_active_delisted"
        out = out[out["Ticker"] != ""].copy()
        out = out.sort_values(["Ticker", "Provider_Security_ID"]).drop_duplicates(
            ["Ticker", "Provider_Security_ID"],
            keep="last",
        )
        return ensure_columns(out.reset_index(drop=True), CANONICAL_SECURITY_MASTER_COLUMNS)

    def load_prices(
        self,
        tickers: Iterable[str],
        start_date: object,
        end_date: object,
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        frames = []
        stocks_frame = self._read_filtered_required(
            self.paths.stocks,
            "stocks",
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            date_column="date",
        )
        if not stocks_frame.empty:
            frames.append(stocks_frame)

        if self.paths.funds is not None and self.paths.funds.exists():
            funds_frame = self._read_filtered_required(
                self.paths.funds,
                "funds",
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                date_column="date",
            )
            if not funds_frame.empty:
                frames.append(funds_frame)

        if frames:
            frame = pd.concat(frames, ignore_index=True, sort=False)
        else:
            frame = stocks_frame

        lookup = _column_lookup(frame)
        frame = _standardize_ticker_date(frame, lookup)

        close_source = "closeadj" if adjusted and "closeadj" in lookup else "close"
        out = pd.DataFrame(index=frame.index)
        out["Provider"] = self.provider_name
        out["Ticker"] = frame["Ticker"]
        out["Date"] = frame["Date"]
        out["Open"] = _numeric(_col(frame, lookup, "open"))
        out["High"] = _numeric(_col(frame, lookup, "high"))
        out["Low"] = _numeric(_col(frame, lookup, "low"))
        out["Close"] = _numeric(_col(frame, lookup, close_source))
        out["Close_Unadjusted"] = _numeric(_col(frame, lookup, "closeunadj"))
        if out["Close_Unadjusted"].isna().all():
            out["Close_Unadjusted"] = _numeric(_col(frame, lookup, "close"))
        out["Volume"] = _numeric(_col(frame, lookup, "volume"))
        out["Adjustment_Mode"] = (
            "split_dividend_spinoff_adjusted" if close_source == "closeadj" else "split_adjusted"
        )
        out["Last_Updated"] = _date_text(_col(frame, lookup, "lastupdated"))
        out = ensure_columns(out, CANONICAL_PRICE_COLUMNS)

        result: dict[str, pd.DataFrame] = {}
        for ticker, group in out.dropna(subset=["Date"]).groupby("Ticker", sort=True):
            prices = (
                group.sort_values("Date")
                .drop_duplicates("Date", keep="last")
                .set_index("Date")
            )
            result[str(ticker)] = pd.DataFrame({
                "open": prices["Open"],
                "high": prices["High"],
                "low": prices["Low"],
                "close": prices["Close"],
                "close_unadjusted": prices["Close_Unadjusted"],
                "volume": prices["Volume"],
                "provider": self.provider_name,
            })
        return result

    def load_fundamentals(
        self,
        tickers: Iterable[str],
        start_date: object,
        end_date: object,
        dimension: str = "ARQ",
    ) -> pd.DataFrame:
        if self.paths.fundamentals is None:
            return empty_frame(CANONICAL_FUNDAMENTAL_COLUMNS)
        frame = self._read_filtered_required(
            self.paths.fundamentals,
            "fundamentals",
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            date_column="date",
            dimension=dimension,
        )
        lookup = _column_lookup(frame)
        frame = _standardize_ticker_date(frame, lookup, source_date="date")

        lookup = _column_lookup(frame)
        out = pd.DataFrame(index=frame.index)
        out["Provider"] = self.provider_name
        out["Ticker"] = frame["Ticker"]
        out["Dimension"] = _col(frame, lookup, "dimension").astype(str).str.upper()
        out["Data_View"] = out["Dimension"].map(_dimension_view)
        out["Filing_Date"] = _date_text(frame["Date"])
        out["Calendar_Date"] = _date_text(_col(frame, lookup, "calendardate"))
        out["Fiscal_Period_End"] = _date_text(_col(frame, lookup, "reportperiod"))
        out["Fiscal_Period"] = _col(frame, lookup, "fiscalperiod")
        out["Currency"] = _col(frame, lookup, "currency")
        out["FX_USD"] = _numeric(_col(frame, lookup, "fxusd"))
        out["Revenue_USD"] = _usd_metric(frame, lookup, "revenueusd", "revenue")
        out["Gross_Profit_USD"] = _usd_metric(frame, lookup, None, "gp")
        out["Operating_Income_USD"] = _usd_metric(frame, lookup, None, "opinc")
        out["Net_Income_USD"] = _usd_metric(frame, lookup, "netinccmnusd", "netinccmn")
        out["Income_Before_Tax_USD"] = _usd_metric(frame, lookup, None, "ebt")
        out["Income_Tax_Expense_USD"] = _usd_metric(frame, lookup, None, "taxexp")
        out["Interest_Expense_USD"] = _usd_metric(frame, lookup, None, "intexp")
        out["Depreciation_Amortization_USD"] = _usd_metric(frame, lookup, None, "depamor")
        out["EBIT_USD"] = _usd_metric(frame, lookup, "ebitusd", "ebit")
        out["EBITDA_USD"] = _usd_metric(frame, lookup, "ebitdausd", "ebitda")
        out["Cash_From_Operations_USD"] = _usd_metric(frame, lookup, None, "ncfo")
        out["Capital_Expenditure_USD"] = _usd_metric(frame, lookup, None, "capex").abs()
        out["Free_Cash_Flow_USD"] = _usd_metric(frame, lookup, None, "fcf")
        out["Cash_USD"] = _usd_metric(frame, lookup, "cashnequsd", "cashneq")
        out["Short_Term_Investments_USD"] = _usd_metric(frame, lookup, None, "investmentsc")
        out["Short_Term_Debt_USD"] = _usd_metric(frame, lookup, None, "debtc")
        out["Long_Term_Debt_USD"] = _usd_metric(frame, lookup, None, "debtnc")
        out["Debt_USD"] = _usd_metric(frame, lookup, "debtusd", "debt")
        out["Equity_USD"] = _usd_metric(frame, lookup, "equityusd", "equity")
        out["Assets_USD"] = _usd_metric(frame, lookup, None, "assets")
        out["Current_Assets_USD"] = _usd_metric(frame, lookup, None, "assetsc")
        out["Current_Liabilities_USD"] = _usd_metric(frame, lookup, None, "liabilitiesc")
        out["Net_PPE_USD"] = _usd_metric(frame, lookup, None, "ppnenet")
        out["Goodwill_USD"] = _usd_metric(frame, lookup, None, "goodwill")
        out["Net_Intangible_Assets_USD"] = _usd_metric(frame, lookup, None, "intangibles")
        out["Operating_Lease_Liability_USD"] = pd.NA
        out["Operating_Lease_ROU_Asset_USD"] = pd.NA
        out["Preferred_Stock_USD"] = pd.NA
        out["Minority_Interest_USD"] = pd.NA
        out["Weighted_Average_Lease_Discount_Rate"] = pd.NA
        out["Invested_Capital_USD"] = _usd_metric(frame, lookup, None, "invcap")
        out["ROIC"] = _numeric(_col(frame, lookup, "roic"))
        out["Market_Cap_USD"] = _numeric(_col(frame, lookup, "marketcap"))
        out["Enterprise_Value_USD"] = _numeric(_col(frame, lookup, "ev"))
        out["Shares_Basic"] = _numeric(_col(frame, lookup, "sharesbas"))
        out["Shares_Weighted_Average"] = _numeric(_col(frame, lookup, "shareswa"))
        out["Shares_Weighted_Average_Diluted"] = _numeric(_col(frame, lookup, "shareswadil"))
        out["Shares_Weighted_Average_Diluted"] = out["Shares_Weighted_Average_Diluted"].fillna(
            out["Shares_Weighted_Average"]
        )
        out["Price_USD"] = _numeric(_col(frame, lookup, "price"))
        out["PE"] = _numeric(_col(frame, lookup, "pe"))
        out["PB"] = _numeric(_col(frame, lookup, "pb"))
        out["PS"] = _numeric(_col(frame, lookup, "ps"))
        out["EV_EBITDA"] = _numeric(_col(frame, lookup, "evebitda"))
        out["Last_Updated"] = _date_text(_col(frame, lookup, "lastupdated"))
        return ensure_columns(out.sort_values(["Ticker", "Filing_Date"]), CANONICAL_FUNDAMENTAL_COLUMNS)

    def load_daily_fundamentals(
        self,
        tickers: Iterable[str],
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        if self.paths.daily is None:
            return empty_frame(CANONICAL_DAILY_FUNDAMENTAL_COLUMNS)
        frame = self._read_filtered_required(
            self.paths.daily,
            "daily",
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            date_column="date",
        )
        lookup = _column_lookup(frame)
        frame = _standardize_ticker_date(frame, lookup, source_date="date")
        lookup = _column_lookup(frame)

        out = pd.DataFrame(index=frame.index)
        out["Provider"] = self.provider_name
        out["Ticker"] = frame["Ticker"]
        out["Date"] = _date_text(frame["Date"])
        out["Market_Cap_USD"] = _numeric(_col(frame, lookup, "marketcap")) * 1_000_000.0
        out["Enterprise_Value_USD"] = _numeric(_col(frame, lookup, "ev")) * 1_000_000.0
        out["PE"] = _numeric(_col(frame, lookup, "pe"))
        out["PB"] = _numeric(_col(frame, lookup, "pb"))
        out["PS"] = _numeric(_col(frame, lookup, "ps"))
        out["EV_EBIT"] = _numeric(_col(frame, lookup, "evebit"))
        out["EV_EBITDA"] = _numeric(_col(frame, lookup, "evebitda"))
        out["Last_Updated"] = _date_text(_col(frame, lookup, "lastupdated"))
        return ensure_columns(out.sort_values(["Ticker", "Date"]), CANONICAL_DAILY_FUNDAMENTAL_COLUMNS)

    def load_corporate_actions(
        self,
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        if self.paths.actions is None:
            return empty_frame(CANONICAL_ACTION_COLUMNS)
        frame = self._read_filtered_required(
            self.paths.actions,
            "actions",
            start_date=start_date,
            end_date=end_date,
            date_column="date",
        )
        lookup = _column_lookup(frame)
        frame = _standardize_ticker_date(frame, lookup, source_date="date")
        lookup = _column_lookup(frame)

        out = pd.DataFrame(index=frame.index)
        out["Provider"] = self.provider_name
        out["Ticker"] = frame["Ticker"]
        out["Date"] = _date_text(frame["Date"])
        out["Action"] = _col(frame, lookup, "action")
        out["Value"] = _numeric(_col(frame, lookup, "value"))
        out["Contra_Ticker"] = _col(frame, lookup, "contraticker").map(normalize_ticker)
        out["Name"] = _col(frame, lookup, "name")
        out["Contra_Name"] = _col(frame, lookup, "contraname")
        return ensure_columns(out.sort_values(["Date", "Ticker"]), CANONICAL_ACTION_COLUMNS)

    def _read_required(self, path: Path | None, label: str) -> pd.DataFrame:
        if path is None:
            raise FileNotFoundError(
                f"Missing Sharadar {label} CSV path. Set SHARADAR_{label.upper()}_CSV "
                f"or pass --sharadar-{label}-csv."
            )
        if not path.exists():
            raise FileNotFoundError(f"Sharadar {label} CSV not found: {path}")
        return pd.read_csv(path, low_memory=False)

    def _read_filtered_required(
        self,
        path: Path | None,
        label: str,
        tickers: Iterable[str] | None = None,
        start_date: object | None = None,
        end_date: object | None = None,
        date_column: str = "date",
        dimension: str | None = None,
    ) -> pd.DataFrame:
        if path is None:
            raise FileNotFoundError(
                f"Missing Sharadar {label} CSV path. Set SHARADAR_{label.upper()}_CSV "
                f"or pass --sharadar-{label}-csv."
            )
        if not path.exists():
            raise FileNotFoundError(f"Sharadar {label} CSV not found: {path}")

        header = pd.read_csv(path, nrows=0)
        lookup = _column_lookup(header)
        ticker_col = lookup.get("ticker")
        date_col = lookup.get(date_column.lower())
        dimension_col = lookup.get("dimension")
        selected = {
            normalize_ticker(ticker)
            for ticker in (tickers or [])
            if normalize_ticker(ticker)
        }
        start = _to_date(start_date)
        end = _to_date(end_date)
        dimension_value = str(dimension).upper().strip() if dimension else ""

        if not selected and start is None and end is None and not dimension_value:
            return pd.read_csv(path, low_memory=False)

        frames = []
        rows_read = 0
        rows_kept = 0
        for chunk in pd.read_csv(path, chunksize=self.chunk_rows, low_memory=False):
            rows_read += len(chunk)
            mask = pd.Series(True, index=chunk.index)

            if selected and ticker_col in chunk.columns:
                mask &= chunk[ticker_col].map(normalize_ticker).isin(selected)

            if dimension_value and dimension_col in chunk.columns:
                mask &= chunk[dimension_col].astype(str).str.upper().str.strip().eq(dimension_value)

            if date_col in chunk.columns and (start is not None or end is not None):
                dates = pd.to_datetime(chunk[date_col], errors="coerce").dt.normalize()
                if start is not None:
                    mask &= dates >= start
                if end is not None:
                    mask &= dates <= end

            filtered = chunk.loc[mask].copy()
            if filtered.empty:
                continue
            rows_kept += len(filtered)
            frames.append(filtered)

        print(
            f"[INFO] Filtered Sharadar {label}: kept {rows_kept:,} of {rows_read:,} rows",
            flush=True,
        )
        if not frames:
            return header.copy()
        return pd.concat(frames, ignore_index=True, sort=False)


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).lower(): column for column in frame.columns}


def _col(frame: pd.DataFrame, lookup: dict[str, str], name: str | None) -> pd.Series:
    if not name:
        return pd.Series(pd.NA, index=frame.index)
    column = lookup.get(name.lower())
    if column is None:
        return pd.Series(pd.NA, index=frame.index)
    return frame[column]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _to_date(value: object | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _date_text(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _standardize_ticker_date(
    frame: pd.DataFrame,
    lookup: dict[str, str],
    source_date: str = "date",
) -> pd.DataFrame:
    out = frame.copy()
    out["Ticker"] = _col(out, lookup, "ticker").map(normalize_ticker)
    out["Date"] = pd.to_datetime(_col(out, lookup, source_date), errors="coerce").dt.normalize()
    return out


def _dimension_view(value: object) -> str:
    dimension = str(value).upper()
    if dimension.startswith("AR"):
        return "as_reported_point_in_time"
    if dimension.startswith("MR"):
        return "most_recent_reported_revised"
    return "unknown"


def _usd_metric(
    frame: pd.DataFrame,
    lookup: dict[str, str],
    usd_column: str | None,
    raw_column: str,
) -> pd.Series:
    usd_values = _numeric(_col(frame, lookup, usd_column))
    if usd_values.notna().any():
        return usd_values

    raw_values = _numeric(_col(frame, lookup, raw_column))
    fxusd = _numeric(_col(frame, lookup, "fxusd"))
    return raw_values.mul(fxusd).where(fxusd.notna())
