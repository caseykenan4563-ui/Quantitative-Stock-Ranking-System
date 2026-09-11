"""Shared interfaces and schemas for historical market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd


CANONICAL_SECURITY_MASTER_COLUMNS = [
    "Provider",
    "Provider_Security_ID",
    "Ticker",
    "Company_Name",
    "Exchange",
    "Currency",
    "Sector",
    "Industry",
    "SubIndustry",
    "SIC_Code",
    "SIC_Sector",
    "SIC_Industry",
    "CIK",
    "FIGI",
    "Is_Delisted",
    "First_Price_Date",
    "Last_Price_Date",
    "First_Fundamental_Date",
    "Last_Fundamental_Date",
    "Source_Quality",
]

CANONICAL_PRICE_COLUMNS = [
    "Provider",
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Close_Unadjusted",
    "Volume",
    "Adjustment_Mode",
    "Last_Updated",
]

CANONICAL_FUNDAMENTAL_COLUMNS = [
    "Provider",
    "Ticker",
    "Dimension",
    "Data_View",
    "Filing_Date",
    "Calendar_Date",
    "Fiscal_Period_End",
    "Fiscal_Period",
    "Currency",
    "FX_USD",
    "Revenue_USD",
    "Gross_Profit_USD",
    "Operating_Income_USD",
    "Net_Income_USD",
    "Income_Before_Tax_USD",
    "Income_Tax_Expense_USD",
    "Interest_Expense_USD",
    "Depreciation_Amortization_USD",
    "EBIT_USD",
    "EBITDA_USD",
    "Cash_From_Operations_USD",
    "Capital_Expenditure_USD",
    "Free_Cash_Flow_USD",
    "Cash_USD",
    "Short_Term_Investments_USD",
    "Short_Term_Debt_USD",
    "Long_Term_Debt_USD",
    "Debt_USD",
    "Equity_USD",
    "Assets_USD",
    "Current_Assets_USD",
    "Current_Liabilities_USD",
    "Net_PPE_USD",
    "Goodwill_USD",
    "Net_Intangible_Assets_USD",
    "Operating_Lease_Liability_USD",
    "Operating_Lease_ROU_Asset_USD",
    "Preferred_Stock_USD",
    "Minority_Interest_USD",
    "Weighted_Average_Lease_Discount_Rate",
    "Invested_Capital_USD",
    "ROIC",
    "Market_Cap_USD",
    "Enterprise_Value_USD",
    "Shares_Basic",
    "Shares_Weighted_Average",
    "Shares_Weighted_Average_Diluted",
    "Price_USD",
    "PE",
    "PB",
    "PS",
    "EV_EBITDA",
    "Last_Updated",
]

CANONICAL_DAILY_FUNDAMENTAL_COLUMNS = [
    "Provider",
    "Ticker",
    "Date",
    "Market_Cap_USD",
    "Enterprise_Value_USD",
    "PE",
    "PB",
    "PS",
    "EV_EBIT",
    "EV_EBITDA",
    "Last_Updated",
]

CANONICAL_ACTION_COLUMNS = [
    "Provider",
    "Ticker",
    "Date",
    "Action",
    "Value",
    "Contra_Ticker",
    "Name",
    "Contra_Name",
]


@dataclass(frozen=True)
class SecurityRecord:
    provider: str
    provider_security_id: str
    ticker: str
    company_name: str
    exchange: str = ""
    currency: str = "USD"
    sector: str = ""
    industry: str = ""
    subindustry: str = ""
    cik: str = ""
    figi: str = ""
    is_delisted: bool = False
    first_price_date: str = ""
    last_price_date: str = ""
    first_fundamental_date: str = ""
    last_fundamental_date: str = ""
    source_quality: str = ""

    def to_frame_row(self) -> dict:
        row = asdict(self)
        return {
            "Provider": row["provider"],
            "Provider_Security_ID": row["provider_security_id"],
            "Ticker": row["ticker"],
            "Company_Name": row["company_name"],
            "Exchange": row["exchange"],
            "Currency": row["currency"],
            "Sector": row["sector"],
            "Industry": row["industry"],
            "SubIndustry": row["subindustry"],
            "CIK": row["cik"],
            "FIGI": row["figi"],
            "Is_Delisted": row["is_delisted"],
            "First_Price_Date": row["first_price_date"],
            "Last_Price_Date": row["last_price_date"],
            "First_Fundamental_Date": row["first_fundamental_date"],
            "Last_Fundamental_Date": row["last_fundamental_date"],
            "Source_Quality": row["source_quality"],
        }


def normalize_ticker(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).upper().strip()


def normalize_ticker_list(tickers: Iterable[str] | None) -> list[str]:
    if tickers is None:
        return []
    return sorted({normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)})


def to_date(value: object) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return None
    return pd.Timestamp(date).normalize()


def parse_date_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in frame.columns:
        frame = frame.copy()
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    return frame


def filter_by_tickers(frame: pd.DataFrame, tickers: Iterable[str] | None, column: str = "Ticker") -> pd.DataFrame:
    selected = normalize_ticker_list(tickers)
    if not selected or column not in frame.columns:
        return frame
    return frame[frame[column].astype(str).str.upper().isin(selected)].copy()


def filter_by_date_range(
    frame: pd.DataFrame,
    date_column: str,
    start_date: object | None,
    end_date: object | None,
) -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns:
        return frame
    frame = parse_date_column(frame, date_column)
    start = to_date(start_date)
    end = to_date(end_date)
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame[date_column] >= start
    if end is not None:
        mask &= frame[date_column] <= end
    return frame.loc[mask].copy()


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[columns + [c for c in out.columns if c not in columns]]


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


class HistoricalDataProvider(ABC):
    """Provider boundary for survivorship-aware historical research data."""

    provider_name: str

    @abstractmethod
    def load_security_master(self) -> pd.DataFrame:
        """Return canonical active/delisted security reference data."""

    @abstractmethod
    def load_prices(
        self,
        tickers: Iterable[str],
        start_date: object,
        end_date: object,
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Return daily price frames keyed by ticker, indexed by date."""

    @abstractmethod
    def load_fundamentals(
        self,
        tickers: Iterable[str],
        start_date: object,
        end_date: object,
        dimension: str = "ARQ",
    ) -> pd.DataFrame:
        """Return canonical filing-date-aware financial statement rows."""

    @abstractmethod
    def load_daily_fundamentals(
        self,
        tickers: Iterable[str],
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        """Return canonical daily valuation ratios and market metrics."""

    @abstractmethod
    def load_corporate_actions(
        self,
        start_date: object,
        end_date: object,
    ) -> pd.DataFrame:
        """Return canonical splits, dividends, ticker changes, and delisting events."""

    def provider_status(self) -> dict:
        return {
            "provider": self.provider_name,
            "survivorship_bias_free": False,
            "point_in_time_fundamentals": False,
            "delisted_prices": False,
        }
