import requests
import pandas as pd
import os
import time
import yfinance as yf
from openpyxl import load_workbook, Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font
from datetime import date
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz
import numpy as np
import json
from pathlib import Path
import copy
import random
import sys
import subprocess

#======Set Stuff=======


STOCK_PTS_PATH = "data/stock_price_trend_history.csv"
COMBINED_SCORE_HISTORY_PATH = "data/combined_score_history.csv"
FEATURE_TABLE_HISTORY_PATH = "data/stock_feature_table_history.csv"
CANONICAL_FUNDAMENTALS_PATH = Path(
    os.getenv("STOCK_ANALYSIS_CANONICAL_FUNDAMENTALS", "data/historical_data_layer/fundamentals_quarterly.csv")
)
CANONICAL_FUNDAMENTAL_DIMENSION_PRIORITY = tuple(
    part.strip().upper()
    for part in os.getenv("STOCK_ANALYSIS_CANONICAL_FUNDAMENTAL_DIMENSIONS", "ART,ARY,ARQ").split(",")
    if part.strip()
)

SUBINDUSTRY_NAME_ALIASES = {
    "Semiconductor Equiptment": "Semiconductor Equipment",
    "IT Counsulting and Services": "IT Consulting and Services",
    "Travel and Accomodation": "Travel and Accommodation",
}


def canonical_subindustry_name(name: str | None) -> str | None:
    if not isinstance(name, str):
        return name
    clean = name.strip()
    return SUBINDUSTRY_NAME_ALIASES.get(clean, clean)


def apply_regime_persistence(regimes: pd.Series) -> pd.Series:
    """
    Normalizes classified regime labels. The structural inputs already use
    rolling windows, so this must not forward-fill stale regimes over current
    signals.
    """
    s = regimes.fillna("Neutral").astype(str).str.strip()
    valid = {"Bull", "EarlyBull", "Neutral", "Bear"}
    return s.where(s.isin(valid), "Neutral")


def price_trend_score_to_100(score: float) -> float:
    if not np.isfinite(score):
        return np.nan
    return float(score * 100.0) if 0.0 <= score <= 1.0 else float(score)


SEC_FACT_TAXONOMIES = ("us-gaap", "dei", "ifrs-full")
SEC_ALLOWED_FORMS = {"10-K", "10-Q", "20-F", "40-F", "6-K"}
FLOW_METRICS = {
    "Total Revenue",
    "Net Income",
    "Operating Income",
    "Gross Profit",
    "Depreciation, Depletion, and Amortization",
    "Income Tax Expense",
    "Income Before Tax",
    "Cash From Operations",
    "Payments To Acquire PP&E",
    "Interest Expense Non-Operating",
    "Interest Expense",
}
POINT_IN_TIME_METRICS = {
    "Total Assets",
    "Current Assets",
    "Current Liabilities",
    "Net PP&E",
    "Goodwill",
    "Net Intangible Assets",
    "Cash & Cash Equivelance",
    "Short Term Investments",
    "Short Term Debt (STD)",
    "Long Term Debt (LTD)",
    "Operating Lease Liability",
    "Preferred Stock",
    "Minority Interest",
    "Total Common Shares Outstanding",
    "Weighted Average Shares",
    "Operating Lease ROU Asset",
    "Weighted Average Lease Discount Rate",
}
SHARE_METRICS = {
    "Total Common Shares Outstanding",
    "Weighted Average Shares",
}
PURE_METRICS = {
    "Weighted Average Lease Discount Rate",
}
OPTIONAL_ZERO_METRICS = {
    "Short Term Investments",
    "Short Term Debt (STD)",
    "Long Term Debt (LTD)",
    "Operating Lease Liability",
    "Preferred Stock",
    "Minority Interest",
    "Goodwill",
    "Net Intangible Assets",
    "Operating Lease ROU Asset",
    "Weighted Average Lease Discount Rate",
    "Interest Expense Non-Operating",
    "Interest Expense",
}
MARKET_CAP_SHARE_COUNT_METRIC = "Market Cap Share Count"
SHARE_COUNT_FALLBACK_FLAG_METRIC = "Market Cap Share Count Fallback Used"
MIN_PEER_OBSERVATIONS = 3
MIN_FAIR_VALUE_METRICS_USED = 3
MIN_BENCHMARK_SUBINDUSTRY_SIZE = 9
MAX_MARKET_PRICE_STALENESS_DAYS = 7
SEC_SCRAPE_SLEEP_SEC = 0.12
TIER_PERCENTILE_CUTOFFS = (
    ("S", 0.10),
    ("A", 0.25),
    ("B", 0.50),
    ("C", 0.75),
    ("D", 1.00),
)

FEATURE_ID_COLUMNS = [
    "Date",
    "Ticker",
    "Industry",
    "SubIndustry",
]

FEATURE_OUTPUT_COLUMNS = [
    "Combined_Score",
    "Overall_Rank",
    "Tier",
    "Tier_Rank",
    "Tier_Percentile",
]

FEATURE_REGIME_COLUMNS = [
    "Industry_Regime",
    "SubIndustry_Regime",
    "Structural_Regime_Persist",
    "StockFlow_Regime",
    "Trend_Weight",
    "Benchmark_Weight",
]

FEATURE_PRICE_TREND_COLUMNS = [
    "PTS",
    "Price_Trend_Score",
    "Price_Trend_Score_100",
    "Stock_Price_Trend_Score",
    "SubIndustry_Price_Trend_Score",
    "Industry_Price_Trend_Score",
    "ND20",
    "ND50",
    "MA_Stack",
    "Hold50",
    "LowPenalty",
    "Return_50D",
    "Pct_From_SMA_20",
    "Pct_From_SMA_50",
    "New_Low_20D",
    "New_Low_50D",
    "Higher_High_20D",
    "Higher_High_50D",
]

FEATURE_VALUATION_SCORE_COLUMNS = [
    "Fair_Value_Score",
    "Fair_Value_Metrics_Used",
    "Fair_Value_Data_Status",
    "Fair_Value_Benchmark_Bucket",
    "Fair_Value_Usable_Metrics",
    "Fair_Value_Missing_Reason",
    "Fair_Value_Missing_Metrics",
]

FEATURE_MARKET_COLUMNS = [
    "Market_Price_USD",
    "Market_Price_Date",
    "Market_Data_Source",
    "Valuation_Data_Source",
    "Valuation_Data_Source_Detail",
]

FEATURE_SHARE_COUNT_COLUMNS = [
    "Share_Count_Used",
    "Share_Count_Source",
    "Share_Count_Source_Category",
    "Share_Count_Fallback_Used",
]

FEATURE_VALUATION_COLUMNS = [
    "Total Revenue",
    "Net Income",
    "Operating Income",
    "Gross Profit",
    "Cash From Operations",
    "Free Cash Flow",
    "Total Common Shares Outstanding",
    "Weighted Average Shares",
    MARKET_CAP_SHARE_COUNT_METRIC,
    "Market Cap",
    "Enterprise Value",
    "EBITDA",
    "Earnings Per Share (EPS)",
    "P/E Ratio (Calculated)",
    "Total Revenue CAGR (3-Year)",
    "Earnings Per Share (EPS) CAGR (3-Year) (%)",
    "FCF Forward (3Y)",
    "Free Cash Flow Margin",
    "EV / EBITDA",
    "EV / FCF Forward (3Y)",
    "Gross Profit / EV",
    "Price / Sales",
    "PEG Ratio (Calculated)",
    "ROIC (Heavy)",
    "ROIC (Light)",
    "Net Debt",
]

FEATURE_FUTURE_PLACEHOLDER_COLUMNS = [
    "Google_Search_Interest",
    "Google_Search_Interest_Change_30D",
    "Google_Search_Interest_Change_90D",
    "Google_News_Keyword_Count",
    "Google_News_Keyword_Growth_30D",
    "News_Sentiment_Score",
    "News_Sentiment_Volume",
    "Earnings_Surprise",
    "Analyst_EPS_Revision_30D",
    "Analyst_Revenue_Revision_30D",
    "Forward_Revenue_Growth_Estimate",
    "Forward_EPS_Growth_Estimate",
    "Insider_Net_Buying",
    "Institutional_Ownership_Change",
    "Short_Interest_Pct_Float",
    "Short_Interest_Change",
    "Options_Call_Put_Ratio",
    "Implied_Volatility",
    "Realized_Volatility_20D",
    "Liquidity_Dollar_Volume_20D",
    "Beta_1Y",
    "Max_Drawdown_6M",
    "Downside_Volatility_3M",
    "Sector_ETF_Relative_Strength",
    "Dividend_Yield",
    "Buyback_Yield",
    "Turnover_Penalty",
    "Transaction_Cost_Estimate",
    "Rate_Sensitivity",
    "Credit_Spread_Sensitivity",
    "USD_Sensitivity",
]

FEATURE_TARGET_COLUMNS = [
    "Next_1D_Return",
    "Next_5D_Return",
    "Next_21D_Return",
    "Next_63D_Return",
    "Next_1D_Excess_Return",
    "Next_5D_Excess_Return",
    "Next_21D_Excess_Return",
    "Next_63D_Excess_Return",
]

FEATURE_TABLE_COLUMNS = (
    FEATURE_ID_COLUMNS
    + FEATURE_OUTPUT_COLUMNS
    + FEATURE_REGIME_COLUMNS
    + FEATURE_PRICE_TREND_COLUMNS
    + FEATURE_VALUATION_SCORE_COLUMNS
    + FEATURE_MARKET_COLUMNS
    + FEATURE_SHARE_COUNT_COLUMNS
    + FEATURE_VALUATION_COLUMNS
    + FEATURE_FUTURE_PLACEHOLDER_COLUMNS
    + FEATURE_TARGET_COLUMNS
)

EXPORT_RAW_SEC_DEBUG = True   # ← turn OFF to disable entirely
AUTO_OPEN_OUTPUT = os.getenv("STOCK_ANALYSIS_AUTO_OPEN", "0") == "1"


# --- SEC API Configuration ---
# The SEC requires a User-Agent header for all API requests.
# Please replace 'YOUR_EMAIL@example.com' with your actual email address.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "StockAnalysisResearch/1.0 contact@example.com")
HEADERS = {
    'User-Agent': SEC_USER_AGENT
}
RISK_FREE_RATE = 0.04          # 4% risk-free rate (e.g., 10-year U.S. Treasury)
EQUITY_RISK_PREMIUM = 0.055    # 5.5% equity market risk premium
MIN_COST_OF_DEBT = 0.02        # 2% minimum cost of debt

# --- Peer Comparison and CAGR Labels ---
# These are the 9 metrics defined by the user for the combined CAGR sheet.
CAGR_LABELS_FOR_EXPORT = [
    "Total Revenue",
    "Net Income",
    "Total Assets",
    "Total Common Shares Outstanding",
    "Cash From Operations",
    "Earnings Per Share (EPS)",
    "Gross Profit Margin (%)",
    "Net Profit Margin (%)",
    "ROIC (Heavy)"
]

#====================================
#
#
#=========Relevent Labels & Calculations============
RELEVANT_LABELS_VALUATION = {

    # ==================================================
    # INCOME STATEMENT (CORE)
    # ==================================================
    "Total Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet"
    ],

    "Net Income": [
        "NetIncomeLoss",
        "IncomeLossFromContinuingOperations"
    ],

    "Operating Income": [
        "OperatingIncomeLoss"
    ],

    "Gross Profit": [
        "GrossProfit",
        ["CALCULATION:", "Revenues", "CostOfGoodsAndServicesSold", "-"],
        ["CALCULATION:", "Revenues", "CostOfRevenue", "-"],
        ["CALCULATION:", "Revenues", "CostOfGoodsSold", "-"],
        ["CALCULATION:", "RevenueFromContractWithCustomerExcludingAssessedTax", "CostOfRevenue", "-"],
        ["CALCULATION:", "RevenueFromContractWithCustomerExcludingAssessedTax", "CostOfGoodsAndServicesSold", "-"],
        ["CALCULATION:", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "-"]
    ],

    "Depreciation, Depletion, and Amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        ["CALCULATION:", "Depreciation", "+", "AmortizationOfIntangibleAssets", "+", "Depletion"],
        ["CALCULATION:", "AmortizationOfIntangibleAssets", "Depreciation", "+"]
    ],

    "Income Tax Expense": [
        "IncomeTaxExpenseBenefit"
    ],

    "Income Before Tax": [
        "IncomeLossBeforeIncomeTaxExpenseBenefit",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxExpenseBenefit"
    ],

    # ==================================================
    # BALANCE SHEET (EV + ROIC)
    # ==================================================
    "Total Assets": [
        "Assets",
        "AssetsTotal"
    ],

    "Current Assets": [
        "AssetsCurrent",
        ["CALCULATION:", "Assets", "AssetsNoncurrent", "-"],
        ["CALCULATION:", "Assets", "NoncurrentAssets", "-"]
    ],

    "Current Liabilities": [
        "LiabilitiesCurrent",
        ["CALCULATION:", "Liabilities", "LiabilitiesNoncurrent", "-"]
    ],

    "Net PP&E": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
        ["CALCULATION:", "PropertyPlantAndEquipment", "AccumulatedDepreciationDepletionAndAmortization", "-"]
    ],

    "Goodwill": [
        "Goodwill"
    ],

    "Net Intangible Assets": [
        "IntangibleAssetsNetExcludingGoodwill",
        "IntangibleAssetsOtherThanGoodwillNet",
        ["CALCULATION:", "IntangibleAssets", "Goodwill", "-"]
    ],

    # ==================================================
    # CAPITAL STRUCTURE (EV)
    # ==================================================
    "Cash & Cash Equivelance": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalents",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    ],

    "Short Term Investments": [
        "MarketableSecuritiesCurrent",
        "ShortTermInvestments",
        ["CALCULATION:", "CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalentsAtCarryingValue", "-"]
    ],

    "Short Term Debt (STD)": [
        "LongTermDebtCurrent",
        "ConvertibleSeniorNotesCurrent",
        "NotesPayableCurrent"
    ],

    "Long Term Debt (LTD)": [
        "ConvertibleSeniorNotes",
        "LongTermDebt",
        ["CALCULATION:", "ConvertibleSeniorNotes", "+", "LongTermDebt", "+", "OperatingLeaseLiability"]
    ],

    "Operating Lease Liability": [
        "OperatingLeaseLiability"
    ],

    "Preferred Stock": [
        "PreferredStockValueOutstanding",
        "PreferredStockValue"
    ],

    "Minority Interest": [
        "MinorityInterest",
        "NoncontrollingInterest"
    ],

    # ==================================================
    # SHARES / MARKET CAP
    # ==================================================
    "Total Common Shares Outstanding": [
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding"
    ],

    "Weighted Average Shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingDiluted",
        "WeightedAverageNumberOfSharesOutstandingBasic"
    ],

    # ==================================================
    # CASH FLOW (FCF)
    # ==================================================
    "Cash From Operations": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashFlowsFromUsedInOperatingActivities"
    ],

    "Payments To Acquire PP&E": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets"
    ],

    # ==================================================
    # LEASE SUPPORT (ROIC LIGHT)
    # ==================================================
    "Operating Lease ROU Asset": [
        "OperatingLeaseRightOfUseAsset"
    ],

    "Weighted Average Lease Discount Rate": [
        "WeightedAverageDiscountRate"
    ],
    "Interest Expense Non-Operating": [
        "InterestExpenseNonoperating"
    ],
    "Interest Expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestPaidNet",
        "InterestExpenseOperating",
    ],
}


# ===============================================================
# NEW CALCULATED EQUATION METRICS FOR VALUATION
# ===============================================================


CALCULATED_EQUATIONS_VALUATION = {

    # =======================================================
    # LEVEL 1 — FOUNDATIONS
    # =======================================================
    "Interest Expense Consolidated": {
        "components": ["Interest Expense", "Interest Expense Non-Operating"],
        "operation": "CONSOLIDATE_FILL",
        "description": "Fallback interest expense consolidation"
    },

    "Free Cash Flow": {
        "components": ["Cash From Operations", "Payments To Acquire PP&E"],
        "operation": "-",
        "description": "CFO - CapEx"
    },

    "Average Total Assets": {
        "components": ["Total Assets"],
        "operation": "AVERAGE_PRIOR",
        "description": "Avg assets for ROIC"
    },

    # =======================================================
    # LEVEL 2 — CAPITAL STRUCTURE
    # =======================================================
    "Interest Bearing Debt": {
        "components": ["Short Term Debt (STD)", "Long Term Debt (LTD)", "Operating Lease Liability"],
        "operation": "+",
        "description": "Total interest-bearing debt"
    },

    "Net Debt": {
        "components": ["Interest Bearing Debt", "Cash & Cash Equivelance"],
        "operation": "-",
        "description": "Net debt"
    },

    "Market Cap": {
        "components": ["Closing Price (USD)", MARKET_CAP_SHARE_COUNT_METRIC],
        "operation": "*",
        "description": "Equity market value"
    },

    "Enterprise Value": {
        "components": ["Market Cap", "Net Debt", "Preferred Stock", "Minority Interest"],
        "operation": "+",
        "description": "EV"
    },

    # =======================================================
    # LEVEL 3 — OPERATING PERFORMANCE
    # =======================================================
    "EBITDA": {
        "components": ["Operating Income", "Depreciation, Depletion, and Amortization"],
        "operation": "+",
        "description": "EBITDA"
    },

    "EBT": {
        "components": ["Operating Income", "Interest Expense Consolidated"],
        "operation": "-",
        "description": "Earnings before tax"
    },

    "Tax Rate": {
        "components": ["Income Tax Expense", "EBT"],
        "operation": "/",
        "description": "Effective tax rate"
    },

    "Tax Retention Ratio": {
        "components": [1, "Tax Rate"],
        "operation": "-",
        "description": "1 - tax rate"
    },

    # =======================================================
    # LEVEL 4 — ROIC (HEAVY & LIGHT)
    # =======================================================
    "Net Working Capital (Heavy)": {
        "components": ["Current Assets", "Current Liabilities"],
        "operation": "-",
        "description": "NWC (heavy)"
    },

    "Invested Capital (Heavy)": {
        "components": ["Net PP&E", "Net Working Capital (Heavy)"],
        "operation": "+",
        "description": "IC heavy"
    },

    "NOPAT (Heavy)": {
        "components": ["Operating Income", "Tax Retention Ratio"],
        "operation": "*",
        "description": "NOPAT heavy"
    },

    "ROIC (Heavy)": {
        "components": ["NOPAT (Heavy)", "Invested Capital (Heavy)"],
        "operation": "/",
        "description": "ROIC heavy"
    },

    # ----- LIGHT ADJUSTMENTS -----
    "Embedded Lease Interest": {
        "components": ["Operating Lease Liability", "Weighted Average Lease Discount Rate"],
        "operation": "*",
        "description": "Lease interest"
    },

    "NOPAT (Light)": {
        "components": ["NOPAT (Heavy)", "Embedded Lease Interest"],
        "operation": "+",
        "description": "NOPAT light"
    },

    "Operating Current Assets": {
        "components": ["Current Assets", "Cash & Cash Equivelance", "Short Term Investments"],
        "operation": "-",
        "description": "Operating current assets"
    },

    "Net Working Capital (Light)": {
        "components": ["Operating Current Assets", "Current Liabilities"],
        "operation": "-",
        "description": "NWC light"
    },

    "Invested Capital (Light)": {
        "components": ["Net PP&E", "Net Intangible Assets", "Goodwill", "Operating Lease ROU Asset", "Net Working Capital (Light)"],
        "operation": "+",
        "description": "IC light"
    },

    "ROIC (Light)": {
        "components": ["NOPAT (Light)", "Invested Capital (Light)"],
        "operation": "/",
        "description": "ROIC light"
    },

    # =======================================================
    # LEVEL 5 — GROWTH & FORWARD METRICS
    # =======================================================
    "Earnings Per Share (EPS)": {
        "components": ["Net Income", "Weighted Average Shares"],
        "operation": "/",
        "description": "EPS"
    },
    "P/E Ratio (Calculated)": {
        "components": ["Closing Price (USD)", "Earnings Per Share (EPS)"],
        "operation": "/",
        "description": "Closing Price (USD) / Earnings Per Share (EPS)"
    },

    "Earnings Per Share (EPS) CAGR (3-Year) (%)": {
        "components": ["Earnings Per Share (EPS)"],
        "operation": "CAGR",
        "years": 3,
        "multiplier": 100,
        "description": "EPS CAGR"
    },

    "Total Revenue CAGR (3-Year)": {
        "components": ["Total Revenue"],
        "operation": "CAGR",
        "years": 3,
        "multiplier": 1,
        "description": "Revenue CAGR"
    },

    "1 + Revenue CAGR": {
        "components": ["Total Revenue CAGR (3-Year)", 1],
        "operation": "+",
    },
        
    "(1 + Revenue CAGR)^3": {
        "components": ["1 + Revenue CAGR", "1 + Revenue CAGR", "1 + Revenue CAGR"],
        "operation": "*",
    },
    
    "Revenue 3-Year Forward": {
        "components": ["(1 + Revenue CAGR)^3", "Total Revenue"],
        "operation": "*",
    },

    "Free Cash Flow Margin": {
        "components": ["Free Cash Flow", "Total Revenue"],
        "operation": "/",
        "description": "FCF margin"
    },

    "FCF Forward (3Y)": {
        "components": ["Revenue 3-Year Forward", "Free Cash Flow Margin"],
        "operation": "*",
        "description": "Forward FCF"
    },

    # =======================================================
    # LEVEL 6 — FINAL VALUATION MULTIPLES
    # =======================================================
    "EV / EBITDA": {
        "components": ["Enterprise Value", "EBITDA"],
        "operation": "/",
        "description": "EV/EBITDA"
    },

    "EV / FCF Forward (3Y)": {
        "components": ["Enterprise Value", "FCF Forward (3Y)"],
        "operation": "/",
        "description": "EV/FCF forward"
    },

    "Gross Profit / EV": {
        "components": ["Gross Profit", "Enterprise Value"],
        "operation": "/",
        "description": "Gross profit to EV"
    },

    "Price / Sales": {
        "components": ["Market Cap", "Total Revenue"],
        "operation": "/",
        "description": "P/S"
    },

    "PEG Ratio (Calculated)": {
        "components": ["P/E Ratio (Calculated)", "Earnings Per Share (EPS) CAGR (3-Year) (%)"],
        "operation": "/",
        "description": "PEG ratio"
    },

    #"Value Spread (Heavy)": {
     #   "components": ["ROIC (Heavy)", "Weighted Average Cost of Capital"],
     #   "operation": "-",
      #  "description": "Value spread heavy"
    #},

    #"Value Spread (Light)": {
    #    "components": ["ROIC (Light)", "Weighted Average Cost of Capital"],
    #    "operation": "-",
    #    "description": "Value spread light"
    #},
}

#=================================================================================
#
#
#========Industry Stock Separation by Regime and Benchmark(Tech)==================

#====================Is Capital Shifting=======================
GLOBAL_DIVERSIFIED_TECH_PLATFORMS = {
    "MSFT","GOOGL","AMZN","AAPL","META","IBM"
}
SEMICONDUCTORS = [
    "NVDA", "AMD", "INTC", "MU", "MRVL", "QCOM", "TXN", "NXPI", "ON", "ADI", "MCHP", "MPWR", "SWKS", "QRVO", "LSCC", "SYNA", "CRUS", "POWI", "AOSL"
]

SEMICONDUCTORS_CONFIRMERS = [
    "SITM", "SLAB", "NVEC", "INDI", "MXL", "CRDO", "MX", "DIOD"
]

APPLICATION_SOFTWARE = [
    "ADBE", "ADSK", "CRM", "NOW", "DDOG", "SNOW", "MDB", "WDAY", "TEAM", "INTU", "HUBS", "GTLB", "DOCU", "ESTC", "FICO", "BSY", "MANH", "TYL", "QTWO", "PTC", "GWRE"
]

APPLICATION_SOFTWARE_CONFIRMERS = [
    "ACIW", "AGYS", "AI", "AIOT", "AKAM", "ALKT", "ALRM", "AMPL", "APPF", "APPN", "APPS", "BILL", "BL", "BLKB", "BLND", "BOX", "BRZE", "CXM", "CWAN", "DBX", "DOMO", "DT", "EXFY", "EXOD", "FIVN", "FROG", "FRSH", "GDDY", "IOT", "INTA", "LAW", "LIF", "MIR", "NABL", "NCNO", "NTNX", "PAR", "PATH", "PD", "PEGA", "PENG", "PLTR", "PRGS", "RAMP", "RDVT", "RNG", "SOUN", "SPT", "SPSC", "TDC", "VERX", "WEAV", "WK", "YEXT", "ZETA", "ZM", "APP", "ASAN", "AVPT"
]

SEMICONDUCTORS_EQUIPTMENT = [
    "AMAT", "LRCX", "KLAC", "ASML", "ENTG", "ACLS"
    ]

SEMICONDUCTORS_EQUIPTMENT_CONFIRMERS = [
    "ONTO", "FORM", "ACMR", "PLAB", "UCTT", "COHU", "PDFS", "VECO", "ICHR", "KLIC", "ASYS", "CAMT"
]

HARDWARE_AND_STORAGE = [
    "DELL", "HPQ", "HPE", "NTAP", "STX", "WDC", "SMCI", "GLW", "CIEN", "TEL", "SNX", "SANM", "JBL", "FLEX", "PLXS", "TTMI", "KEYS", "TDY", "VSH", "AVT", "APH"
]

HARDWARE_AND_STORAGE_CONFIRMATION = [
    "AAOI", "ADTN", "ARLO", "ATEN", "BELFA", "BELFB", "BMI", "CLFD", "CRSR", "DAKT", "ITRI", "KE", "LFUS", "LITE", "LWLG", "MEI", "MVIS", "NTGR", "OSIS", "OUST", "ROG", "SCSC", "VPG", "XRX", "ZBRA"
]

SEARCH_AND_DIGITAL_MEDIA = [
    "TTD", "MGNI"
]

SEARCH_AND_DIGITAL_MEDIA_CONFIRMATION = [
    "PERI", "DV"
]

#====================Is Capital Spreading======================


SYSTEM_SOFTWARE = [
    "ORCL", "NOW", "NTAP"
]

SYSTEM_SOFTWARE_CONFIRMATION = [
    "AKAM", "DOCN", "FFIV", "TDC", "VRSN", "NTCT"
]

CYBERSECURITY = [
    "CRWD", "PANW", "FTNT", "ZS", "OKTA"
]

CYBERSECURITY_CONFIRMATION = [
    "TENB", "QLYS", "RPD", "S", "GEN", "VRNS"
]

E_COMMERCE_MARKETPLACE = [
    "EBAY", "ETSY", "SHOP"
]

MOBILITY_AND_DELIVERY_PLATFORM = [
    "UBER", "LYFT", "DASH"
]

INTERACTIVE_HOME_ENTERTAINMENT = [
    "EA", "TTWO", "RBLX"
]

INTERACTIVE_HOME_ENTERTAINMENT_COMPARISON = [
    "PLTK"
]


#====================Is this Late / End-Cycle==================


IT_CONSULTING_AND_SERVICES = [
    "ACN", "CTSH", "DXC", "EPAM", "SAIC", "CACI", "G"
]

IT_CONSULTING_AND_SERVICES_CONFIRMATION = [
    "KD", "RXT", "UIS", "BGSF", "CNDT", "PSN"
]

INTEGRATED_TELECOM = [
    "VZ", "T", "TMUS", "LUMN", "CABO", "TDS"
]

INTEGRATED_TELECOM_CONFIRMATION = [
    "OPTU", "SHEN", "CCOI", "ATNI", "GOGO"
]

TRAVEL_AND_ACCOMODATION = [
    "ABNB", "BKNG", "EXPE"
]


#===========Regimine Groups identifier========


REGIME_GROUPS = {
#====================Is Capital Shifting=======================
    
    "Global Diversified Tech Platform": {
        "core": GLOBAL_DIVERSIFIED_TECH_PLATFORMS,
        "confirmers": []
    },
    "Semiconductors": {
        "core": SEMICONDUCTORS,
        "confirmers": SEMICONDUCTORS_CONFIRMERS
    },
    "Application Software": {
        "core": APPLICATION_SOFTWARE,
        "confirmers": APPLICATION_SOFTWARE_CONFIRMERS
    },
    "Semiconductor Equipment": {
        "core": SEMICONDUCTORS_EQUIPTMENT,
        "confirmers": SEMICONDUCTORS_EQUIPTMENT_CONFIRMERS
    },
    "Hardware and Storage": {
        "core": HARDWARE_AND_STORAGE,
        "confirmers": HARDWARE_AND_STORAGE_CONFIRMATION
    },
    "Search and Digital Media": {
        "core": SEARCH_AND_DIGITAL_MEDIA,
        "confirmers": SEARCH_AND_DIGITAL_MEDIA_CONFIRMATION
    },
#====================Is Capital Spreading======================

    "System Software": {
        "core": SYSTEM_SOFTWARE,
        "confirmers": SYSTEM_SOFTWARE_CONFIRMATION
    },
    "Cybersecurity": {
        "core": CYBERSECURITY,
        "confirmers": CYBERSECURITY_CONFIRMATION
    },
    "E-Commerce Marketplace": { #core only
        "core": E_COMMERCE_MARKETPLACE
    },
    "Mobility and Delivery Platform": { #core only
        "core": MOBILITY_AND_DELIVERY_PLATFORM
    },
    "Interactive Home Entertainment": {
        "core": INTERACTIVE_HOME_ENTERTAINMENT,
        "confirmers": INTERACTIVE_HOME_ENTERTAINMENT_COMPARISON
    },
#====================Is this Late / End-Cycle==================

    "IT Consulting and Services": {
        "core": IT_CONSULTING_AND_SERVICES,
        "confirmers": IT_CONSULTING_AND_SERVICES_CONFIRMATION
    },
    "Integrated Telecom": {
        "core": INTEGRATED_TELECOM,
        "confirmers": INTEGRATED_TELECOM_CONFIRMATION
    },
    "Travel and Accommodation": {
        "core": TRAVEL_AND_ACCOMODATION
    }
}
# ==========================================================
# Tech Benchmark Groups
# Used ONLY for Fair Value / Benchmark Scoring
# ==========================================================
# Rules:
# - >= 15 stocks → Small / Mid / Large (30 / 40 / 30)
# - 9–14 stocks → Small / Large (50 / 50)
# - < 9 stocks → Small only (trend-only, no benchmark scoring)
# ==========================================================

TECH_BENCHMARK_GROUPS = {

    # ===================== GREEN ==========================
    "Semiconductors": {
        "Small": [
            "MX","NVEC","AOSL","INDI","MXL","POWI","DIOD","SYNA","SLAB"
        ],
        "Mid": [
            "CRUS","QRVO","SWKS","SITM","LSCC","ON","CRDO","MCHP","MPWR","MRVL"
        ],
        "Large": [
            "ADI","TXN","QCOM","INTC","MU","AMD","NXPI","NVDA"
        ],
    },

    "Application Software": {
        "Small": [
            "AI","EXFY","DOMO","MIR","LAW","APPS","WEAV","BRZE",
            "AIOT","RDVT","BLND","YEXT","RNG","PENG","PD","NABL","AMPL","PAR",
            "FIVN","PRGS","CXM","EXOD","SOUN","SPT"
        ],
        "Mid": [
            "FRSH","RAMP","ALKT","APPN","ALRM","AVPT","NCNO","TDC","BLKB","ASAN",
            "BL","AGYS","SPSC","INTA","BOX","ZETA","QTWO","WK","ACIW","LIF",
            "BILL","GTLB","ESTC","FROG","DBX","APPF","PATH","PEGA","MANH","NTNX",
            "CWAN","VERX"
        ],
        "Large": [
            "BSY","AKAM","DT","DOCU","GWRE","GDDY","IOT","HUBS","PTC","TYL",
            "ZM","NOW","MDB","FICO","TEAM","ADSK","SNOW","ADBE","INTU",
            "APP","CRM","PLTR","DDOG","WDAY"
        ],
    },

    "Semiconductor Equipment": {
        "Small": [
            "ASYS","ICHR","PDFS","COHU","UCTT","VECO"
        ],
        "Mid": [
            "PLAB","KLIC","ACLS","ACMR","FORM","CAMT","ONTO"
        ],
        "Large": [
            "ENTG","KLAC","AMAT","LRCX","ASML"
        ],
    },

    "Hardware and Storage": {
        "Small": [
            "MEI","MVIS","XRX","CLFD","LWLG","VPG","CRSR","NTGR","ADTN","KE",
            "SCSC","DAKT","ATEN","ARLO","BELFA","BELFB"
        ],
        "Mid": [
            "OUST","ROG","VSH","AAOI","PLXS","AVT","ITRI","OSIS","SNX",
            "LFUS","BMI","TTMI","SANM"
        ],
        "Large": [
            "FLEX","TDY","JBL","LITE","HPE","CIEN","KEYS","STX","WDC","TEL",
            "GLW","DELL","APH","ZBRA","SMCI","HPQ","NTAP"
        ],
    },

    # Warning / Early signal — SMALL ONLY (trend-first, no benchmark)
    "Search and Digital Media": {
        "Small": ["PERI","DV","MGNI","TTD"]
    },

    # ===================== YELLOW ==========================
    "System Software": {
        "Small": ["NTCT","TDC","DOCN","AKAM","FFIV"],
        "Large": ["NTAP","VRSN","NOW","ORCL"]
    },

    "Cybersecurity": {
        "Small": ["RPD","TENB","VRNS","QLYS","S","OKTA"],
        "Large": ["GEN","ZS","FTNT","PANW","CRWD"]
    },

    "E-Commerce Marketplace": {
        "Small": ["ETSY","EBAY"]
    },

    "Mobility and Delivery Platform": {
        "Small": ["DASH","UBER"]
    },

    "Interactive Home Entertainment": {
        "Small": ["PLTK","TTWO","EA","RBLX"]
    },

    # ===================== RED ==========================
    "IT Consulting and Services": {
        "Small": ["BGSF","UIS","RXT","CNDT","DXC","PSN"],
        "Large": ["SAIC","KD","G","EPAM","CACI","CTSH","ACN"]
    },

    "Integrated Telecom": {
        "Small": ["ATNI","OPTU","CABO","SHEN","GOGO","CCOI"],
        "Large": ["TDS","LUMN","VZ","T","TMUS"]
    },

    "Travel and Accommodation": {
        "Small": ["EXPE","ABNB","BKNG"]
    },
}

#===========================================================
#
#
#============Industry Regime Groups==========================
#
# ==========================
# TECH INDUSTRY GROUPS
# ==========================
TECH_CORE = "Global Diversified Tech Platform"
TECH_WARNING = "Search and Digital Media"

TECH_GREEN = [ #Capital moves here first
    "Semiconductors",
    "Application Software",
    "Semiconductor Equipment",
    "Hardware and Storage",
    "Search and Digital Media",  # also used as warning
]

TECH_YELLOW = [ #Confirms regime is coming
    "System Software",
    "Cybersecurity",
    "E-Commerce Marketplace",
    "Mobility and Delivery Platform",
    "Interactive Home Entertainment",
]

TECH_RED = [ #late risk watcher
    "IT Consulting and Services",
    "Integrated Telecom",
    "Travel and Accommodation",
]

#=================================================================
#
#
#=====================Outline of functions=========================
#1. Universe & configuration setup
#   - Load ticker lists, benchmarks, regimes, parameters, debug paths
#
#2. Raw SEC data scraping
#   - Pull unmodified SEC/XBRL financial data
#   - Output in { years: [...], values: [...] } format
#
#3. Raw SEC debug export (side channel)
#   - Deep-copy raw SEC data
#   - Export to Excel (one workbook, one sheet per ticker)
#   - NO mutation, NO reuse downstream
#
#4. Data normalization (hard boundary)
#   - Convert { years, values } → { "YYYY": value }
#   - After this stage, "years"/"values" must never appear again
#
#5. History sufficiency enforcement
#   - Enforce minimum years of data (e.g., 5 years)
#  - Drop incomplete metrics or failing tickers
#
#6. Derived fundamental metric calculation
#   - Compute CAGR, margins, ROIC, FCF growth, stability metrics
#   - Inputs are normalized, year-keyed data ONLY
#
#7. Valuation multiple computation
#   - Compute EV/EBITDA, P/FCF, P/S, etc.
#   - Exclude negative EBITDA values
#
#8. Benchmark normalization
#   - Normalize metrics relative to index/sector/sub-industry benchmarks
#
#9. Regime-conditioned weighting
#   - Apply bull/bear regime weights to growth, valuation, stability metrics
#
#10. Composite scoring, ranking, and final exports
#    - Combine scores, rank stocks
#   - Export final tables/CSVs/Excel outputs
#
#
#======================================================================
#
#
#=====(1)==============Universe & Configuration Set-Up=============(1)========
#
#-----------------------------------------------------------------------
# Quick Summary:
# Defines the scope and rules of the valuation run before any data is fetched or calculated. This section determines which tickers are included, how they are grouped (sector, sub-industry, leaders/confirmers), what benchmarks and regimes apply, and whether benchmark data needs to be rebuilt. It establishes a stable, read-only configuration context so all downstream scraping, valuation, and scoring operate under consistent assumptions.
# Contect for benchmark and price/fair value scoring...does not score though
#-------------------------------------------------------------------------
#
#--------------------------List of Functions-------------------------------
#List of functions:
# A. flatten_ticker_groups
# B. load_sec_cik_map
# C. get_ticker_bucket
#--------------------------------------------------------------------------
#
#
#--------------------------Coded Functions---------------------------------------


SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT
}

SEC_COMPANYFACTS_CACHE_DIR = Path("data/sec_companyfacts_cache")

_CIK_MAP_CACHE = None
FAIR_VALUE_ISSUE_LOG: dict[str, list[str]] = {}

DEFAULT_CV_THRESHOLD = 0.50

CV_THRESHOLDS_BY_METRIC = {
    "EV / EBITDA": 0.70,
    "PEG Ratio (Calculated)": 1.20,
    "EV / FCF Forward (3Y)": 1.00,
    "Gross Profit / EV": 0.60,
    "Price / Sales": 0.70,
    "Value Spread (Heavy) (%)": 0.50,
    "Value Spread (Light) (%)": 0.50,
}


def sec_companyfacts_cache_path(cik: str) -> Path:
    cik_str = str(cik).zfill(10)
    return SEC_COMPANYFACTS_CACHE_DIR / f"CIK{cik_str}.json"


def load_sec_companyfacts_cache(cik: str) -> dict | None:
    path = sec_companyfacts_cache_path(cik)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read SEC companyfacts cache {path}: {exc}")
        return None

    if isinstance(data, dict) and "facts" in data:
        return data
    if isinstance(data, dict) and isinstance(data.get("companyfacts"), dict):
        return data["companyfacts"]
    return None


def save_sec_companyfacts_cache(cik: str, data: dict) -> None:
    if not isinstance(data, dict) or "facts" not in data:
        return

    path = sec_companyfacts_cache_path(cik)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    tmp_path.replace(path)


def load_sec_cik_map(force_reload: bool = False) -> dict:
    """
    Loads SEC ticker->CIK map once and caches it.
    """
    global _CIK_MAP_CACHE
    if _CIK_MAP_CACHE is not None and not force_reload:
        return _CIK_MAP_CACHE

    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()

    cik_map = {}
    for entry in data.values():
        t = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if t and cik:
            cik_map[t] = cik

    _CIK_MAP_CACHE = cik_map
    return cik_map


def reset_fair_value_issue_log():
    FAIR_VALUE_ISSUE_LOG.clear()


def log_fair_value_issue(ticker: str, message: str):
    if not ticker or not message:
        return
    ticker = ticker.upper()
    issues = FAIR_VALUE_ISSUE_LOG.setdefault(ticker, [])
    if message not in issues:
        issues.append(message)


def summarize_logged_fair_value_issues(ticker: str, limit: int = 4) -> str:
    issues = FAIR_VALUE_ISSUE_LOG.get(str(ticker).upper(), [])
    if not issues:
        return ""
    shown = issues[:limit]
    suffix = f"; plus {len(issues) - limit} more scrape notes" if len(issues) > limit else ""
    return "; ".join(shown) + suffix
   

def get_ticker_bucket(subindustry: str, ticker: str) -> str | None:
    """
    Returns 'Small'/'Mid'/'Large' bucket for ticker within subindustry.
    Returns None if not found.
    """
    subindustry = canonical_subindustry_name(subindustry)
    groups = TECH_BENCHMARK_GROUPS.get(subindustry, {})
    for bucket, tickers in groups.items():
        if isinstance(tickers, list) and ticker in tickers:
            return bucket
    return None
    
    
def flatten_ticker_groups(regime_groups: dict) -> list:
    tickers = set()
    for group in regime_groups.values():
        tickers.update(group.get("core", []))
        tickers.update(group.get("confirmers", []))
    return sorted(tickers)


def normalize_asof_date(asof_date: pd.Timestamp | str | None) -> pd.Timestamp:
    if asof_date is None:
        return pd.Timestamp.today().normalize()
    return pd.to_datetime(asof_date).normalize()


def sec_date_key(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        clean = value.strip()
        if len(clean) >= 10 and clean[4] == "-" and clean[7] == "-":
            return clean[:10]

    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def metric_unit_preferences(metric_name: str) -> tuple[str, ...]:
    if metric_name in SHARE_METRICS:
        return ("shares",)
    if metric_name in PURE_METRICS:
        return ("pure",)
    return ("USD",)


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))


def is_positive_finite_number(value) -> bool:
    return is_finite_number(value) and float(value) > 0


def sec_item_year(item: dict) -> int | None:
    fy = item.get("fy")
    if isinstance(fy, (int, np.integer)):
        return int(fy)

    end_key = item.get("_end_ts") or sec_date_key(item.get("end"))
    if end_key:
        try:
            return int(str(end_key)[:4])
        except ValueError:
            return None

    return None


def sec_expected_ytd_quarters(fp: str) -> int | None:
    return {"Q1": 1, "Q2": 2, "Q3": 3}.get(str(fp).upper())


def is_ytd_quarter_fact(item: dict) -> bool:
    expected_qtrs = sec_expected_ytd_quarters(item.get("fp", ""))
    if expected_qtrs is None:
        return False

    qtrs = item.get("qtrs")
    if isinstance(qtrs, (int, np.integer)):
        return int(qtrs) == expected_qtrs

    return qtrs is None


def normalize_sec_metric_value(metric_name: str, value: float) -> float:
    value = float(value)
    if metric_name in PURE_METRICS and abs(value) > 1.5:
        return value / 100.0
    return value


def collect_sec_fact_items(
    facts_by_taxonomy: dict,
    tag: str,
    unit_preferences: tuple[str, ...],
    asof_date: pd.Timestamp
) -> list[dict]:
    items = []
    asof_key = normalize_asof_date(asof_date).strftime("%Y-%m-%d")

    for taxonomy in SEC_FACT_TAXONOMIES:
        tag_data = facts_by_taxonomy.get(taxonomy, {}).get(tag)
        if not tag_data:
            continue

        units = tag_data.get("units", {})
        for unit in unit_preferences:
            unit_items = units.get(unit, [])
            if not isinstance(unit_items, list):
                continue

            for raw_item in unit_items:
                if raw_item.get("form") not in SEC_ALLOWED_FORMS:
                    continue
                if raw_item.get("fy") is None and raw_item.get("end") is None:
                    continue
                if not is_finite_number(raw_item.get("val")):
                    continue

                filed = sec_date_key(raw_item.get("filed"))
                if filed is None or filed > asof_key:
                    continue

                item = dict(raw_item)
                item["_taxonomy"] = taxonomy
                item["_tag"] = tag
                item["_unit"] = unit
                item["_filed_ts"] = filed
                item["_end_ts"] = sec_date_key(raw_item.get("end"))
                item["_start_ts"] = sec_date_key(raw_item.get("start"))
                items.append(item)

    return items


def latest_item(items: list[dict]) -> dict | None:
    if not items:
        return None
    return sorted(
        items,
        key=lambda item: (
            item.get("_end_ts") or "",
            item.get("_filed_ts") or "",
        )
    )[-1]


def annual_sec_series(metric_name: str, items: list[dict]) -> dict[int, float]:
    by_year = {}

    for item in items:
        form = item.get("form")
        fp = str(item.get("fp", "")).upper()
        qtrs = item.get("qtrs")

        if not (form in {"10-K", "20-F", "40-F"} or fp == "FY" or qtrs == 4):
            continue

        year = sec_item_year(item)
        if year is None:
            continue

        current = by_year.get(year)
        if current is None or (
            item.get("_filed_ts") or "",
            item.get("_end_ts") or ""
        ) > (
            current.get("_filed_ts") or "",
            current.get("_end_ts") or ""
        ):
            by_year[year] = item

    return {
        y: normalize_sec_metric_value(metric_name, item["val"])
        for y, item in by_year.items()
    }


def latest_point_in_time_series(metric_name: str, items: list[dict]) -> dict[int, float]:
    series = annual_sec_series(metric_name, items)
    item = latest_item(items)
    if item is None:
        return series

    year = sec_item_year(item)
    if year is not None:
        series[year] = normalize_sec_metric_value(metric_name, item["val"])

    return series


def quarterly_ttm_series(metric_name: str, items: list[dict]) -> dict[int, float]:
    series = annual_sec_series(metric_name, items)

    quarter_items = [
        item for item in items
        if item.get("form") in {"10-Q", "6-K"}
        and str(item.get("fp", "")).upper() in {"Q1", "Q2", "Q3"}
    ]
    if not quarter_items:
        return series

    ytd_items = [item for item in quarter_items if is_ytd_quarter_fact(item)]
    q_items = ytd_items if ytd_items else quarter_items

    current = latest_item(q_items)
    if current is None:
        return series

    fy = sec_item_year(current)
    fp = str(current.get("fp", "")).upper()
    qtrs = current.get("qtrs")
    if fy is None or not fp:
        return series

    prior_annual = series.get(fy - 1)
    if prior_annual is None:
        return series

    prior_candidates = [
        item for item in q_items
        if sec_item_year(item) == fy - 1
        and str(item.get("fp", "")).upper() == fp
        and (qtrs is None or item.get("qtrs") == qtrs)
    ]
    prior_ytd = latest_item(prior_candidates)
    if prior_ytd is None:
        return series

    ttm = (
        float(prior_annual)
        + normalize_sec_metric_value(metric_name, current["val"])
        - normalize_sec_metric_value(metric_name, prior_ytd["val"])
    )
    if np.isfinite(ttm):
        series[fy] = ttm

    return series


def select_metric_series(
    metric_name: str,
    items: list[dict],
    years_needed: int
) -> dict[int, float]:
    if metric_name in FLOW_METRICS:
        series = quarterly_ttm_series(metric_name, items)
    elif metric_name in POINT_IN_TIME_METRICS:
        series = latest_point_in_time_series(metric_name, items)
    else:
        series = annual_sec_series(metric_name, items)

    valid = {
        int(year): float(value)
        for year, value in series.items()
        if isinstance(year, (int, np.integer)) and np.isfinite(float(value))
    }

    selected_years = sorted(valid.keys(), reverse=True)[:years_needed]
    return {year: valid[year] for year in selected_years}


def series_to_raw_payload(series: dict[int, float]) -> dict:
    years = sorted(series.keys(), reverse=True)
    return {
        "years": years,
        "values": [series[y] for y in years]
    }


def latest_market_price_from_price_data(
    ticker: str,
    asof_date: pd.Timestamp,
    price_data: dict | None
) -> tuple[float, pd.Timestamp] | tuple[float, None]:
    if not price_data or ticker not in price_data:
        return np.nan, None

    df = price_data[ticker]
    if df is None or df.empty or "close" not in df.columns:
        return np.nan, None

    history = df.copy()
    history.index = pd.to_datetime(history.index).normalize()
    history = history.loc[history.index <= asof_date]
    if history.empty:
        return np.nan, None

    price_date = history.index.max()
    price = history.loc[price_date, "close"]
    return (float(price), price_date) if np.isfinite(price) else (np.nan, None)


def market_price_source_from_price_data(
    ticker: str,
    price_date: pd.Timestamp | None,
    price_data: dict | None
) -> str:
    if price_date is None or not price_data or ticker not in price_data:
        return "missing"

    df = price_data[ticker]
    if df is None or df.empty or "provider" not in df.columns:
        return "yfinance_adjusted_close"

    history = df.copy()
    history.index = pd.to_datetime(history.index).normalize()
    history = history.loc[history.index <= pd.to_datetime(price_date).normalize()]
    providers = history["provider"].dropna().astype(str)
    if providers.empty:
        return "canonical_price_file"
    provider = providers.iloc[-1].strip() or "canonical_price_file"
    return f"{provider}_adjusted_close"


def build_market_data_map(
    tickers: list[str],
    asof_date: pd.Timestamp,
    price_data: dict | None
) -> dict[str, dict]:
    market_data = {}
    asof_date = normalize_asof_date(asof_date)

    for ticker in tickers:
        price, price_date = latest_market_price_from_price_data(ticker, asof_date, price_data)
        staleness_days = (
            int((asof_date - price_date).days)
            if price_date is not None else None
        )
        if staleness_days is not None and staleness_days > MAX_MARKET_PRICE_STALENESS_DAYS:
            market_data[ticker] = {
                "price": np.nan,
                "price_date": price_date,
                "source": f"stale_{market_price_source_from_price_data(ticker, price_date, price_data)}",
                "staleness_days": staleness_days,
            }
            continue

        market_data[ticker] = {
            "price": price,
            "price_date": price_date,
            "source": market_price_source_from_price_data(ticker, price_date, price_data),
            "staleness_days": staleness_days,
        }

    return market_data


def format_market_price_date(market_snapshot: dict | None) -> str:
    if not market_snapshot:
        return ""
    price_date = market_snapshot.get("price_date")
    if price_date is None or pd.isna(price_date):
        return ""
    return pd.to_datetime(price_date).strftime("%Y-%m-%d")


def enrich_raw_with_market_data(
    raw: dict,
    ticker: str,
    asof_date: pd.Timestamp | None,
    market_snapshot: dict | None
) -> dict:
    if not market_snapshot:
        return raw

    price = market_snapshot.get("price", np.nan)
    if not np.isfinite(price):
        print(f"[WARN] No market price available for {ticker}; valuation multiples needing price will be skipped")
        log_fair_value_issue(
            ticker,
            "Market price unavailable or stale; price-based valuation metrics cannot be calculated"
        )
        return raw

    years = [
        int(y)
        for payload in raw.values()
        if isinstance(payload, dict)
        for y in payload.get("years", [])
        if isinstance(y, (int, np.integer))
    ]
    latest_year = max(years) if years else normalize_asof_date(asof_date).year

    raw["Closing Price (USD)"] = {
        "years": [latest_year],
        "values": [float(price)]
    }
    return raw


def ensure_optional_zero_metrics(data: dict) -> dict:
    years = sorted({
        int(y)
        for series in data.values()
        if isinstance(series, dict)
        for y in series.keys()
        if isinstance(y, (int, np.integer))
    })
    if not years:
        return data

    for metric in OPTIONAL_ZERO_METRICS:
        series = data.setdefault(metric, {})
        if not isinstance(series, dict):
            series = {}
            data[metric] = series
        for year in years:
            series.setdefault(year, 0.0)

    return data


def add_market_cap_share_count_fallback(data: dict) -> dict:
    """
    Builds the share-count series used for market cap.

    Preferred source is point-in-time common shares outstanding. If that is
    missing for a fiscal year, fall back to weighted-average shares so market
    cap and EV ratios can still be calculated with visible provenance.
    """
    direct = data.get("Total Common Shares Outstanding", {})
    weighted = data.get("Weighted Average Shares", {})
    if not isinstance(direct, dict):
        direct = {}
    if not isinstance(weighted, dict):
        weighted = {}

    years = sorted({
        int(y)
        for series in (direct, weighted)
        for y in series.keys()
        if isinstance(y, (int, np.integer))
    })
    if not years:
        return data

    share_count = {}
    fallback_flag = {}

    for year in years:
        direct_value = direct.get(year)
        weighted_value = weighted.get(year)

        if is_positive_finite_number(direct_value):
            share_count[year] = float(direct_value)
            fallback_flag[year] = 0.0
        elif is_positive_finite_number(weighted_value):
            share_count[year] = float(weighted_value)
            fallback_flag[year] = 1.0

    if share_count:
        data[MARKET_CAP_SHARE_COUNT_METRIC] = share_count
        data[SHARE_COUNT_FALLBACK_FLAG_METRIC] = fallback_flag

    return data

#===============================================================================
#
#
#=========(2)=================Raw SEC Data Scraping==============(2)==============
#
#---------------------------------------------------------------------------------
#Quick Summary:
#Responsible for retrieving unmodified financial statement data directly from SEC filings (XBRL/JSON) for each ticker. This stage parses and structures the data into a raw, inspection-friendly format (typically { years: [...], values: [...] }) without applying normalization, filtering, or calculations.
# Its sole purpose is to accurately capture source data so all downstream analysis is based on a faithful representation of the filings.
# No normalization, filtering, valuation, or scoring occurs here.
#----------------------------------------------------------------------------------
#
#-------------------------------List of Functions----------------------------------
# A. _sec_get_json
# B. get_latest_sec_filing_metadata
# C. scrape_sec_financials
# D. build_raw_financial_data
# E. build_valuation_financials
#----------------------------------------------------------------------------------
#
#----------------------------------Functions---------------------------------------

def _sec_get_json(url: str, sleep_sec: float = 0.2, max_retries: int = 3) -> dict | None:
    """
    SEC-safe GET with basic backoff.
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=20)
            if r.status_code in (429, 403):
                time.sleep(sleep_sec * (2 ** attempt))
                continue
            r.raise_for_status()
            time.sleep(sleep_sec)  # polite pacing
            return r.json()
        except Exception:
            time.sleep(sleep_sec * (2 ** attempt))
    return None


def get_sec_companyfacts(
    cik: str,
    use_sec_cache: bool = True,
    refresh_sec_cache: bool = False
) -> dict | None:
    """
    Loads SEC companyfacts once per CIK and reuses the raw filing history.

    Point-in-time behavior is handled later by collect_sec_fact_items(), which
    filters each fact by its SEC filed date before an as-of valuation is built.
    """
    cik_str = str(cik).zfill(10)

    if use_sec_cache and not refresh_sec_cache:
        cached = load_sec_companyfacts_cache(cik_str)
        if cached:
            return cached

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
    data = _sec_get_json(url)
    if data:
        if use_sec_cache:
            save_sec_companyfacts_cache(cik_str, data)
        return data

    if use_sec_cache:
        cached = load_sec_companyfacts_cache(cik_str)
        if cached:
            print(f"[WARN] SEC fetch failed for CIK{cik_str}; using cached companyfacts")
            return cached

    return None


def get_latest_sec_filing_metadata(ticker: str) -> dict | None:
    """
    Returns latest 10-K or 10-Q metadata for ticker.
    """
    cik_map = load_sec_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return None

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _sec_get_json(submissions_url)
    if not data:
        return None

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", []) or []
    dates = filings.get("filingDate", []) or []
    periods = filings.get("reportDate", []) or []

    for form, filed, period in zip(forms, dates, periods):
        if form in ("10-K", "10-Q"):
            filing_date = pd.to_datetime(filed, errors="coerce")
            fiscal_period = pd.to_datetime(period, errors="coerce")
            return {
                "ticker": ticker.upper(),
                "form": form,
                "filing_date": filing_date,
                "fiscal_period": fiscal_period,
            }

    return None


def scrape_sec_financials(
    ticker: str,
    cik_map: dict,
    relevant_labels: dict,
    years_needed: int = 5,
    asof_date: pd.Timestamp | str | None = None,
    use_sec_cache: bool = True,
    refresh_sec_cache: bool = False
) -> dict:
    """
    Scrapes SEC XBRL companyfacts using explicit unit rules.

    USD metrics only accept USD facts, share metrics only accept shares, and
    pure/rate metrics only accept pure facts. Flow metrics use annual history
    plus latest 10-Q TTM when enough YTD data exists; point-in-time metrics use
    the latest filed balance-sheet/share fact available by asof_date.

    CALCULATION entries are intentionally skipped here and handled later
    by the calculated-equations pipeline.
    """

    ticker = ticker.upper()
    asof_date = normalize_asof_date(asof_date)
    cik = cik_map.get(ticker)

    if not cik:
        print(f"[WARN] No CIK found for {ticker}")
        log_fair_value_issue(ticker, "No SEC CIK found for ticker")
        return {}

    cik_str = str(cik).zfill(10)
    data = get_sec_companyfacts(
        cik=cik_str,
        use_sec_cache=use_sec_cache,
        refresh_sec_cache=refresh_sec_cache
    )
    if not data:
        print(f"[WARN] SEC fetch failed for {ticker}")
        log_fair_value_issue(ticker, "SEC companyfacts request failed")
        return {}

    facts_by_taxonomy = data.get("facts", {})
    results = {}

    # --------------------------------------------------
    # Loop through valuation metrics
    # --------------------------------------------------
    for metric_name, possible_tags in relevant_labels.items():

        # Safety: metric names must be strings
        if not isinstance(metric_name, str):
            continue

        unit_preferences = metric_unit_preferences(metric_name)
        metric_items = []

        for tag in possible_tags:
            # Skip CALCULATION entries (lists)
            if not isinstance(tag, str):
                continue

            metric_items.extend(
                collect_sec_fact_items(
                    facts_by_taxonomy=facts_by_taxonomy,
                    tag=tag,
                    unit_preferences=unit_preferences,
                    asof_date=asof_date
                )
            )

        if not metric_items:
            print(f"[WARN] {ticker}: no SEC {unit_preferences} facts for {metric_name}")
            unit_label = "/".join(unit_preferences)
            log_fair_value_issue(
                ticker,
                f"No SEC {unit_label} facts found for {metric_name}"
            )
            continue

        series = select_metric_series(
            metric_name=metric_name,
            items=metric_items,
            years_needed=years_needed
        )

        if series:
            results[metric_name] = series_to_raw_payload(series)

    if SEC_SCRAPE_SLEEP_SEC > 0:
        time.sleep(SEC_SCRAPE_SLEEP_SEC)  # SEC rate-limit friendly
    return results
    
    
def build_raw_financial_data(
    ticker: str,
    asof_date: pd.Timestamp | str | None = None,
    market_snapshot: dict | None = None,
    debug_sec_writer: pd.ExcelWriter | None = None,
    use_sec_cache: bool = True,
    refresh_sec_cache: bool = False
) -> dict:
    cik_map = load_sec_cik_map()
    asof_date = normalize_asof_date(asof_date)

    raw = scrape_sec_financials(
        ticker=ticker,
        cik_map=cik_map,
        relevant_labels=RELEVANT_LABELS_VALUATION,
        asof_date=asof_date,
        use_sec_cache=use_sec_cache,
        refresh_sec_cache=refresh_sec_cache
    )

    raw = enrich_raw_with_market_data(
        raw=raw,
        ticker=ticker,
        asof_date=asof_date,
        market_snapshot=market_snapshot
    )

    if EXPORT_RAW_SEC_DEBUG and debug_sec_writer is not None:
        export_raw_sec_wide(
            raw_data=raw,
            ticker=ticker,
            writer=debug_sec_writer
        )


    # 🔍 DEBUG EXPORT (TEMPORARY)
    #export_raw_sec_to_excel(ticker, raw)

    # Normalize for splits
    raw = apply_stock_split_adjustment(raw)

    return raw


def build_valuation_financials(
    ticker: str,
    asof_date: pd.Timestamp | str | None = None,
    market_snapshot: dict | None = None,
    debug_sec_writer: pd.ExcelWriter | None = None,
    use_sec_cache: bool = True,
    refresh_sec_cache: bool = False
) -> dict:
    canonical = build_valuation_financials_from_canonical_fundamentals(
        ticker=ticker,
        asof_date=asof_date,
        market_snapshot=market_snapshot,
    )
    if canonical:
        record_valuation_data_source(
            ticker,
            "sharadar_canonical_fundamentals",
            canonical_fundamental_source_detail(ticker, asof_date),
        )
        return canonical

    # ----------------------------------
    # 1) Raw SEC scrape fallback
    # ----------------------------------
    raw = build_raw_financial_data(
        ticker=ticker,
        asof_date=asof_date,
        market_snapshot=market_snapshot,
        debug_sec_writer=debug_sec_writer,
        use_sec_cache=use_sec_cache,
        refresh_sec_cache=refresh_sec_cache
    )

    # ----------------------------------
    # 2) 🔒 CANONICAL NORMALIZATION (REQUIRED)
    # ----------------------------------
    raw = normalize_sec_to_year_map(raw)

    for metric, year_map in raw.items():
        if not isinstance(year_map, dict):
            continue
        for k in year_map.keys():
            if not isinstance(k, int):
                raise RuntimeError(
                    f"[FATAL] normalize_sec_to_year_map failed: {metric} → key={k}"
                )
    # ----------------------------------
    # 3) Enforce minimum history (POST-normalization)
    # ----------------------------------
    raw = {
        metric: series
        for metric, series in raw.items()
        if isinstance(series, dict) and (len(series) >= 5 or metric in POINT_IN_TIME_METRICS or metric == "Closing Price (USD)")
    }
    raw = ensure_optional_zero_metrics(raw)
    raw = add_market_cap_share_count_fallback(raw)

    # ----------------------------------
    # 4) Run calculated equations
    # ----------------------------------
    calculated = run_calculated_equations(
        raw,
        CALCULATED_EQUATIONS_VALUATION
    )

    return calculated


#===================================================================================
#
#
#=======(3)=============Raw SEC debug export (side channel)============(3)===========
#
#-------------------------------------------------------------------------------------
#Quick Summary:
#Responsible for writing a read-only snapshot of the raw, unnormalized SEC data to disk for inspection and validation.
#This stage exists purely to verify scraping correctness (completeness, year alignment, missing values) and must never mutate data or feed back into the valuation pipeline.
#It operates as an observational side channel so downstream normalization, valuation, and scoring remain logically isolated and reproducible

#--------------------------------List of Functions-------------------------------------
# A. export_raw_sec_to_excel
# B. export_raw_sec_wide
#--------------------------------------------------------------------------------------
#
#----------------------------------Functions-------------------------------------------

"""
def export_raw_sec_to_excel(
    ticker: str,
    raw_data: dict,
    excel_path: str
):
    
    #Appends raw SEC-scraped financial data for a single ticker
   # as ONE sheet in a shared debug workbook.
    

    sheet_name = ticker[:31]  # Excel sheet name limit
    mode = "a" if os.path.exists(excel_path) else "w"

    rows = []

    for metric, payload in raw_data.items():
        if not isinstance(payload, dict):
            continue

        years = payload.get("years", [])
        values = payload.get("values", [])

        if not years or not values:
            continue

        for y, v in zip(years, values):
            rows.append({
                "Metric": metric,
                "Year": y,
                "Value": v
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter", mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
"""
def export_raw_sec_wide(
    raw_data: dict,
    ticker: str,
    writer: pd.ExcelWriter
):
    """
    Exports raw SEC data in wide financial-model format.

    Rows = metrics
    Columns = years
    One sheet per ticker.
    """

    if not raw_data:
        return

    # --------------------------------------------------
    # Collect all valid numeric years
    # --------------------------------------------------
    all_years = sorted({
        int(y)
        for metric_data in raw_data.values()
        if isinstance(metric_data, dict)
        for y in metric_data.keys()
        if isinstance(y, (int, np.integer)) or (isinstance(y, str) and y.isdigit())
    })

    if not all_years:
        return

    # --------------------------------------------------
    # Build wide rows
    # --------------------------------------------------
    rows = []

    for metric, year_map in raw_data.items():
        if not isinstance(year_map, dict):
            continue

        row = {"Metric": metric}
        for y in all_years:
            row[y] = year_map.get(y, None)

        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows).set_index("Metric")

    # --------------------------------------------------
    # Write ONE sheet per ticker
    # --------------------------------------------------
    sheet_name = f"{ticker}_RAW"[:31]  # Excel limit

    df.to_excel(writer, sheet_name=sheet_name)

#=====================================================================================
#
#
#============(4)============Data normalization (hard boundary)=====(4)================
#
#-------------------------------------------------------------------------------------
# Quick Summary:
# converts raw SEC financial data from {years, values} arrays into a consistent, year-keyed format (e.g., { "2019": value, "2020": value }) that all downstream logic relies on.
# This stage enforces a strict contract: after normalization, no raw SEC shapes or non-year keys are allowed to pass forward.
# It exists to guarantee correctness and comparability so all calculations, valuations, and scores operate on a single, predictable data structure.
#------------------------------------------------------------------------------------
#
#-----------------------------------Function List------------------------------------
# A. normalize_raw_sec_data
# B. normalize_sec_to_year_map
# C. safe_get
#
#-------------------------------------------------------------------------------------
#
#-------------------------------------Functions---------------------------------------

def normalize_raw_sec_data(raw: dict) -> dict:
    """
    Converts raw SEC containers:
    { metric: { "years": [...], "values": [...] } }
    →
    { metric: { int_year: value } }

    Guarantees:
    - No "years"/"values" containers survive
    - Only integer years are kept
    """
    normalized = {}

    for metric, payload in raw.items():
        # Case 1: raw SEC container
        if (
            isinstance(payload, dict)
            and "years" in payload
            and "values" in payload
            and isinstance(payload["years"], list)
            and isinstance(payload["values"], list)
            and len(payload["years"]) == len(payload["values"])
        ):
            year_map = {}

            for y, v in zip(payload["years"], payload["values"]):
                if isinstance(y, (int, np.integer)):
                    year_map[y] = v
                elif isinstance(y, str) and y.isdigit():
                    year_map[int(y)] = v
                # else: drop ("TTM", "FY", "years", etc.)

            normalized[metric] = year_map

        # Case 2: already normalized metric
        else:
            normalized[metric] = payload

    return normalized


def normalize_sec_to_year_map(raw_data: dict) -> dict:
    """
    Converts SEC raw containers into:
    { metric: {int_year: value} }
    """
    out = {}

    for metric, payload in raw_data.items():
        if (
            isinstance(payload, dict)
            and "years" in payload
            and "values" in payload
            and isinstance(payload["years"], list)
            and isinstance(payload["values"], list)
        ):
            year_map = {}

            for y, v in zip(payload["years"], payload["values"]):
                try:
                    y = int(y)
                except Exception:
                    continue
                year_map[y] = v

            if year_map:
                out[metric] = year_map

    return out


def safe_get(metric):
    return data.get(metric, {})
    
#=================================================================================
#
#
#========(5)==============History Sufficiency Enforcement============(5)===========
#
#----------------------------------------------------------------------------------
# Quick Summary
# History Sufficiency Enforcement ensures that each metric and ticker has enough historical data to support reliable calculations before any growth rates or valuations are computed.
# This stage filters out metrics or entire tickers that fail minimum lookback requirements (e.g., insufficient years, gaps in history), preventing unstable CAGR, margin, or trend calculations.
# Its role is to protect downstream valuation and scoring from misleading results caused by sparse or incomplete data.
#----------------------------------------------------------------------------------
#
#--------------------------------Function List-------------------------------------
# A. compute_subindustry_filing_coverage
# B. count_states
#-----------------------------------------------------------------------------------
#
#-----------------------------------Functions----------------------------------------

def compute_subindustry_filing_coverage(
    subindustry: str,
    last_benchmark_date: pd.Timestamp,
    sleep_sec: float = 0.15
) -> dict:
    """
    Computes SEC filing coverage for a sub-industry benchmark group.
    """

    subindustry = canonical_subindustry_name(subindustry)
    groups = TECH_BENCHMARK_GROUPS.get(subindustry, {})
    tickers = set()

    for bucket in groups.values():
        if isinstance(bucket, list):
            tickers.update(bucket)

    if len(tickers) < 9:
        return {
            "benchmarkable": False,
            "coverage": 0.0,
            "eligible": False
        }

    filings = []
    for ticker in tickers:
        meta = get_latest_sec_filing_metadata(ticker)
        if meta:
            filings.append(meta)
        time.sleep(sleep_sec)  # SEC rate-limit safety

    if not filings:
        return {
            "benchmarkable": True,
            "coverage": 0.0,
            "eligible": False
        }

    df = pd.DataFrame(filings)

    updated = df["filing_date"] > last_benchmark_date
    coverage = updated.sum() / len(df)

    return {
        "benchmarkable": True,
        "coverage": round(coverage, 3),
        "eligible": coverage >= 0.75,
        "latest_filing": df["filing_date"].max(),
        "latest_fiscal_period": df["fiscal_period"].max()
    }

def count_states(regimes: list[str]) -> dict:
    return {
        "Bull": sum(r == "Bull" for r in regimes),
        "EarlyBull": sum(r == "EarlyBull" for r in regimes),
        "Neutral": sum(r == "Neutral" for r in regimes),
        "Bear": sum(r == "Bear" for r in regimes),
        "Total": len(regimes),
    }
   
#==================================================================================
#
#
#=========(6)============Derived Fundamental Metric Calculation=======(6)=============
#
#----------------------------------------------------------------------------------
# Quick Summary:
# Computes higher-level financial signals from normalized, history-validated fundamentals.
# This stage calculates measures such as growth rates (e.g., CAGR), margins, returns, stability/consistency metrics, and other derived indicators that describe business quality and performance.
# All outputs here are purely analytical inputs for valuation and scoring, with no benchmarking, price data, or regime weighting applied yet.
#----------------------------------------------------------------------------------
#
#---------------------------------Function List------------------------------------
# A. run_calculated_equations
# B. compute_metric_cvs
# C. dispersion_weight_multiplier
# D. compute_hold_ratio
#----------------------------------------------------------------------------------
#
#-----------------------------------Functions--------------------------------------


def run_calculated_equations(
    raw_data: dict,
    equations: dict
) -> dict:
    """
    Executes CALCULATED_EQUATIONS_VALUATION against raw SEC data.

    raw_data format:
    {
        "Metric": {
            "years": [...],
            "values": [...]
        }
    }

    Returns:
    {
        "Metric": {year: value}
    }
    """
    
            
    data = {}
    
    for metric, payload in raw_data.items():
        # Case 1: raw SEC container
        if (
            isinstance(payload, dict)
            and "years" in payload
            and "values" in payload
            and isinstance(payload["years"], list)
            and isinstance(payload["values"], list)
            and len(payload["years"]) == len(payload["values"])
        ):
            cleaned = {}

            for y, v in zip(payload["years"], payload["values"]):
                if isinstance(y, (int, np.integer)):
                    cleaned[y] = v
                elif isinstance(y, str) and y.isdigit():
                    cleaned[int(y)] = v
                # else: drop ("years", "FY", "TTM", etc.)

            if cleaned:
                data[metric] = cleaned

    
        # Case 2: already-normalized metric
        elif isinstance(payload, dict):
            # only keep int-like years
            cleaned = {
                y: v for y, v in payload.items()
                if isinstance(y, (int, np.integer))
            }
            if cleaned:
                data[metric] = cleaned

    # 🔒 FINAL SAFETY CHECK — normalized data only
    for metric, series in data.items():
        bad_keys = [k for k in series.keys() if not isinstance(k, (int, np.integer))]
        if bad_keys:
            raise RuntimeError(
                f"[FATAL] Non-integer years leaked for {metric}: {bad_keys}"
            )

    def safe_get(metric):
        return data.get(metric, {})

    for metric_name, rule in equations.items():
        op = rule.get("operation")
        comps = rule.get("components", [])
    
        # -------------------------------
        # CAGR
        # -------------------------------
        if op == "CAGR":
            base = safe_get(comps[0])
    
            # 🔒 HARD GUARD: only allow {int: value}
            base = {
                y: v for y, v in base.items()
                if isinstance(y, (int, np.integer))
            }
    
            yrs = rule.get("years", 3)
    
            # 🔒 HARD GUARD: years must be int
            if isinstance(yrs, str):
                if yrs.isdigit():
                    yrs = int(yrs)
                else:
                    raise RuntimeError(
                        f"[BAD RULE] {metric_name} has invalid years: {rule.get('years')}"
                    )
            elif not isinstance(yrs, int):
                raise RuntimeError(
                    f"[BAD RULE] {metric_name} has invalid years type: {type(yrs)}"
                )
    
            if len(base) < yrs + 1:
                continue
    
            sorted_years = sorted(base.keys())
            y0, y1 = sorted_years[-(yrs + 1)], sorted_years[-1]
    
            try:
                v0 = float(base[y0])
                v1 = float(base[y1])
            except (TypeError, ValueError, OverflowError):
                continue

            if not np.isfinite(v0) or not np.isfinite(v1):
                continue

            if v0 <= 0 or v1 <= 0:
                continue
    
            cagr = (v1 / v0) ** (1 / yrs) - 1
            data[metric_name] = {y1: cagr * rule.get("multiplier", 1)}




        # -------------------------------
        # AVERAGE PRIOR
        # -------------------------------
        elif op == "AVERAGE_PRIOR":
            base = safe_get(comps[0])
            out = {}
            for y in base:
                prev = base.get(y - 1)
                if prev is not None:
                    out[y] = (base[y] + prev) / 2
            if out:
                data[metric_name] = out

        # -------------------------------
        # CONSOLIDATE_FILL
        # -------------------------------
        elif op == "CONSOLIDATE_FILL":
            out = {}
            for comp in comps:
                series = safe_get(comp)
                for y, v in series.items():
                    if y not in out and v is not None:
                        out[y] = v
            if out:
                data[metric_name] = out

        # -------------------------------
        # Arithmetic (+ - * /)
        # -------------------------------
        else:
            out = {}
            series_list = []
            all_years = sorted({
                y
                for series in data.values()
                if isinstance(series, dict)
                for y in series.keys()
                if isinstance(y, (int, np.integer))
            })

            for c in comps:
                if isinstance(c, (int, float)):
                    series_list.append({y: c for y in all_years})
                else:
                    series_list.append(safe_get(c))

            year_sets = [set(s.keys()) for s in series_list if s]
            
            if len(year_sets) < 2:
                continue  # not enough data overlap
            
            years = set.intersection(*year_sets)

            for y in years:
                try:
                    vals = [s[y] for s in series_list if y in s]
                    if len(vals) != len(series_list):
                        continue  # missing component for this year
            
                    if op == "+":
                        out[y] = sum(vals)
                    elif op == "-":
                        r = vals[0]
                        for v in vals[1:]:
                            r -= v
                        out[y] = r
                    elif op == "*":
                        r = 1
                        for v in vals:
                            r *= v
                        out[y] = r
                    elif op == "/":
                        r = vals[0]
                        valid = True
                        for v in vals[1:]:
                            if v == 0:
                                valid = False
                                break
                            r /= v
                        if valid:
                            out[y] = r
                except Exception:
                    continue

            if out:
                data[metric_name] = out

    return data


def compute_metric_cvs(
    valuation_df: pd.DataFrame,
    subindustry: str,
    bucket: str,
    metrics: list[str]
) -> dict[str, float]:

    cvs = {}
    subindustry = canonical_subindustry_name(subindustry)
    tickers = TECH_BENCHMARK_GROUPS.get(subindustry, {}).get(bucket, [])

    if not isinstance(tickers, list) or len(tickers) < 3:
        return {m: np.nan for m in metrics}

    for metric in metrics:
        if metric not in valuation_df.index:
            cvs[metric] = np.nan
            continue

        vals = [
            float(valuation_df.at[metric, t])
            for t in tickers
            if t in valuation_df.columns
            and np.isfinite(valuation_df.at[metric, t])
        ]

        if len(vals) < 3:
            cvs[metric] = np.nan
            continue

        arr = np.array(vals)
        mean = np.mean(arr)

        cvs[metric] = np.nan if mean == 0 else np.std(arr) / abs(mean)

    return cvs


def dispersion_weight_multiplier(cv: float) -> float: #Uses Coefficient of Variation to give accuracy points toward specific stock.
    """
    Converts coefficient of variation into reliability weight.
    Missing CV defaults to neutral (1.0).
    """
    if not np.isfinite(cv):
        return 1.0  # IMPORTANT FIX

    if cv <= 0.25:
        return 1.15
    if cv <= 0.50:
        return 1.00
    if cv <= 0.75:
        return 0.85
    return 0.65


def compute_hold_ratio(close: pd.Series, window: int) -> float: #For price-trend (PTS) valuation
    """
    Percent of days price held above SMA over window.
    """
    if len(close) < window:
        return np.nan

    sma = close.rolling(window).mean()
    held = close > sma
    return held.iloc[-window:].mean()

#=====================================================================================
#
#
#=======(7)=================Market Data & Price Fetching===============(7)============
#
#------------------------------------------------------------------------------------
# Quick Summary:
# Retrieves current and historical market information needed to price securities, such as stock prices, shares outstanding, and related market inputs.
# This stage is strictly observational, providing clean market data for valuation and price trend analysis without performing scoring, benchmarking, or regime logic.
#------------------------------------------------------------------------------------
#
#-----------------------------------Functions List-------------------------------------
# A. fetch_all_prices
# B. apply_stock_split_adjustment
# C. _check_split_ratio
#------------------------------------------------------------------------------------
#
#--------------------------------------Functions--------------------------------------

def fetch_all_prices(
    tickers: list,
    start_date: str,
    end_date: str,
    batch_size: int = 25,
    max_retries: int = 3,
    sleep_seconds: float = 0.5,
) -> dict:

    import time
    import yfinance as yf
    import pandas as pd

    # Ensure valid date order (UNCHANGED)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    print(f"[DEBUG] Yahoo date range: {start_date} -> {end_date}")

    # Deduplicate + stabilize order
    tickers = sorted(set(tickers))

    price_data = {}
    failed = []

    # 🔴 CHANGED: batched Yahoo calls instead of one giant call
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"[DEBUG] Fetching price batch "
                    f"{i // batch_size + 1} | "
                    f"{len(batch)} tickers | attempt {attempt}"
                )

                df = yf.download(
                    tickers=batch,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=False,   # 🔴 CRITICAL FIX
                )

                if df.empty:
                    raise ValueError("Yahoo returned empty DataFrame")

                # ---- ORIGINAL LOGIC (PRESERVED) ----
                if isinstance(df.columns, pd.MultiIndex):
                    for ticker in batch:
                        if ticker not in df.columns.levels[0]:
                            continue

                        ticker_df = df[ticker]

                        if "Adj Close" in ticker_df.columns:
                            s = ticker_df["Adj Close"]
                        elif "Close" in ticker_df.columns:
                            s = ticker_df["Close"]
                        else:
                            continue

                        s = s.dropna()
                        if not s.empty:
                            price_data[ticker] = s.to_frame(name="close")

                else:
                    # Single-ticker fallback (rare but preserved)
                    if "Adj Close" in df.columns:
                        s = df["Adj Close"].dropna()
                    elif "Close" in df.columns:
                        s = df["Close"].dropna()
                    else:
                        s = None

                    if s is not None and not s.empty:
                        price_data[batch[0]] = s.to_frame(name="close")

                break  # success → exit retry loop

            except Exception as e:
                print(f"[WARN] Price batch failed ({attempt}/{max_retries}): {e}")
                time.sleep(2 ** attempt)

        else:
            # All retries failed
            failed.extend(batch)

        time.sleep(sleep_seconds)

    # ---- Sanity check (UNCHANGED STRUCTURE) ----
    print("\n=== SANITY CHECK: PRICE FETCH ===")
    print(f"Tickers requested: {len(tickers)}")
    print(f"Tickers with price data: {len(price_data)}")

    missing = sorted(set(tickers) - set(price_data.keys()))
    if missing:
        print(f"[WARN] Missing price data for {len(missing)} tickers")
        print(f"Sample missing tickers: {missing[:10]}")
    else:
        print("All tickers returned price data")

    print("================================\n")

    return price_data

    return price_data


def apply_stock_split_adjustment(financial_data: dict) -> dict:
    """
    Detects and adjusts historical share counts and closing prices
    based on stock splits. SEC data is NOT retrospectively adjusted.
    """

    shares_data = financial_data.get("Total Common Shares Outstanding", {})
    price_data = financial_data.get("Closing Price (USD)", {})

    if not shares_data or not price_data:
        return financial_data

    # Convert and sort years
    try:
        years = sorted(int(y) for y in shares_data.keys())
    except Exception:
        return financial_data

    split_factors = {}

    # --------------------------------------------------
    # 1) Detect valid splits
    # --------------------------------------------------
    for i in range(1, len(years)):
        y_cur = str(years[i])
        y_prev = str(years[i - 1])

        try:
            cur_shares = float(shares_data[y_cur])
            prev_shares = float(shares_data[y_prev])
            cur_price = float(price_data.get(y_cur))
            prev_price = float(price_data.get(y_prev))
        except Exception:
            continue

        if prev_shares <= 0 or cur_shares <= 0:
            continue

        ratio = cur_shares / prev_shares

        # Only consider large jumps
        if ratio < 1.5:
            continue

        split_factor = _check_split_ratio(ratio)
        if split_factor <= 1.0:
            continue

        # --------------------------------------------------
        # 2) Price sanity check (VERY IMPORTANT)
        # --------------------------------------------------
        if prev_price > 0 and cur_price > 0:
            implied_price_ratio = prev_price / cur_price

            # True split ≈ inverse relationship
            if abs(implied_price_ratio - split_factor) > 0.6:
                continue  # reject buybacks / noise

        split_factors[y_cur] = split_factor

    if not split_factors:
        return financial_data

    # --------------------------------------------------
    # 3) Build cumulative adjustment factors
    # --------------------------------------------------
    adjustment_by_year = {str(y): 1.0 for y in years}
    cumulative = 1.0

    for split_year in sorted(split_factors.keys(), key=int, reverse=True):
        cumulative *= split_factors[split_year]
        for y in years:
            if int(y) < int(split_year):
                adjustment_by_year[str(y)] = cumulative

    # --------------------------------------------------
    # 4) Apply adjustments
    # --------------------------------------------------
    for metric in [
        "Total Common Shares Outstanding",
        "Preferred Stock Issued",
        "Closing Price (USD)"
    ]:
        metric_data = financial_data.get(metric, {})
        if not metric_data:
            continue

        for y, factor in adjustment_by_year.items():
            if factor == 1.0 or y not in metric_data:
                continue

            try:
                val = float(metric_data[y])
            except Exception:
                continue

            if metric in ("Total Common Shares Outstanding", "Preferred Stock Issued"):
                metric_data[y] = val * factor
            else:  # Closing Price
                metric_data[y] = val / factor

    return financial_data


def _check_split_ratio(ratio: float) -> float:
    """
    Checks if a calculated ratio (new_shares / old_shares) corresponds
    to a common stock split (e.g., 2:1, 3:1, 4:1).
    Returns the integer factor or 1.0 if not a real split.
    """
    common_splits = [2, 3, 4, 5, 10, 20, 50, 100]

    for split in common_splits:
        if abs(ratio - split) < 0.05:
            return float(split)

    # Reverse splits (rare but possible)
    reverse_splits = [0.5, 0.333, 0.25, 0.2, 0.1, 0.01]
    for split in reverse_splits:
        if abs(ratio - split) < 0.05:
            return float(split)

    return 1.0
    
#=====================================================================================
#
#
#=======(8)=================Price Trend & Technical Scoring===============(8)============
#
#------------------------------------------------------------------------------------
# Quick Summary:
# Analyzes market price behavior to assess momentum, trend strength, and technical health independent of fundamentals.
# This stage converts price-based signals (such as moving-average structure, higher highs/lows, and breadth measures) into standardized trend scores that reflect how well a stock is behaving in the market.
# Is the stock at a good buy....
#------------------------------------------------------------------------------------
#
#-----------------------------------Functions List-------------------------------------
# A. build_daily_stock_pts
# B. build_price_features
# C. score_stock_price_trend
# D. compute_sma
# E. percent_from_sma
# F. compute_nd
# G. compute_ma_stack
# H. _slope
# I. is_higher_high
# J. is_new_low
# K. compute_new_low_penalty
#------------------------------------------------------------------------------------
#
#--------------------------------------Functions--------------------------------------
def build_daily_stock_pts(
    price_data: dict,
    asof_date: pd.Timestamp,
    ticker_to_subindustry: dict,
    fair_value_scores: dict | None = None
) -> pd.DataFrame:
    """
    Builds daily stock Price-Trend Scores (PTS).
    Price + valuation only. NO regime logic.
    """

    rows = []
    date_str = asof_date.strftime("%Y-%m-%d")

    for ticker, df in price_data.items():

        if df.empty or asof_date not in df.index:
            continue

        if ticker not in ticker_to_subindustry:
            continue

        subindustry = ticker_to_subindustry[ticker]

        # --- Price trend score ---
        pts, components = score_stock_price_trend(df, asof_date)

        # --- Benchmark / Fair Value score ---
        fair_value_score = (
            fair_value_scores.get(ticker, np.nan)
            if fair_value_scores else np.nan
        )

        rows.append({
            "Date": date_str,
            "Ticker": ticker,
            "SubIndustry": subindustry,

            # Raw scores ONLY
            "PTS": pts,
            "Fair_Value_Score": fair_value_score,

            **components
        })

    return pd.DataFrame(rows)


def build_price_features(df):
    """
    df must contain a 'close' column indexed by date
    """

    df = df.copy()

    # Moving Averages
    df["SMA_20"] = compute_sma(df["close"], 20)
    df["SMA_50"] = compute_sma(df["close"], 50)

    # Percent Distance
    df["Pct_From_SMA_20"] = percent_from_sma(df["close"], df["SMA_20"])
    df["Pct_From_SMA_50"] = percent_from_sma(df["close"], df["SMA_50"])

    # New Lows
    df["New_Low_20D"] = is_new_low(df["close"], 20)
    df["New_Low_50D"] = is_new_low(df["close"], 50)

    # Higher Highs
    df["Higher_High_20D"] = is_higher_high(df["close"], 20)
    df["Higher_High_50D"] = is_higher_high(df["close"], 50)

    return df


def score_stock_price_trend(
    df: pd.DataFrame,
    asof_date: pd.Timestamp
):
    """
    Computes full Price-Trend Score (PTS) and components.
    """

    df_slice = df.loc[:asof_date].copy()

    if "close" not in df_slice.columns:
        raise ValueError("Expected 'close' column in price data")

    # 🔴 Build indicators FIRST
    df_feat = build_price_features(df_slice)

    close = df_feat["close"]

    # --- Core metrics ---
    nd20 = compute_nd(close, 20)
    nd50 = compute_nd(close, 50)

    ma_stack = compute_ma_stack(df_feat)
    hold_50 = compute_hold_ratio(close, 50)
    low_penalty = compute_new_low_penalty(close, 20)

    # --- Weighted base score ---
    pts_base = (
        0.25 * np.nan_to_num(nd20) +
        0.30 * np.nan_to_num(nd50) +
        0.20 * ma_stack +
        0.25 * hold_50 -
        0.20 * low_penalty
    )

    pts_base = np.clip(pts_base, 0, 1)

    components = {
        "ND20": nd20,
        "ND50": nd50,
        "MA_Stack": ma_stack,
        "Hold50": hold_50,
        "LowPenalty": low_penalty,
        "Return_50D": close.iloc[-1] / close.iloc[-51] - 1 if len(close) > 50 else np.nan,
        "Pct_From_SMA_20": df_feat["Pct_From_SMA_20"].iloc[-1],
        "Pct_From_SMA_50": df_feat["Pct_From_SMA_50"].iloc[-1],
        "New_Low_20D": int(df_feat["New_Low_20D"].iloc[-1]),
        "New_Low_50D": int(df_feat["New_Low_50D"].iloc[-1]),
        "Higher_High_20D": int(df_feat["Higher_High_20D"].iloc[-1]),
        "Higher_High_50D": int(df_feat["Higher_High_50D"].iloc[-1]),
    }

    return pts_base, components
    
#Simple Moving Average
def compute_sma(series, window):
    return series.rolling(window).mean()

#Percent Difference from SMA
def percent_from_sma(price_series, sma_series):
    return (price_series - sma_series) / sma_series

#Low Detection
def is_new_low(price_series, lookback):
    rolling_min = price_series.rolling(lookback).min()
    return price_series == rolling_min


def is_higher_high(series, window):
    """
    True if today's close is higher than the max close
    over the prior `window` days.
    """
    return series > series.shift(1).rolling(window).max()


def compute_nd(close: pd.Series, window: int) -> float:
    """
    Proximity to the recent high over `window`, normalized to [0, 1].
    1.0 means the stock is at its window high; lower values mean larger drawdowns.
    """
    if len(close) < window:
        return np.nan

    recent_high = close.iloc[-window:].max()
    latest = close.iloc[-1]

    if not np.isfinite(recent_high) or recent_high <= 0 or not np.isfinite(latest):
        return np.nan

    return float(np.clip(latest / recent_high, 0.0, 1.0))
    
    
def compute_ma_stack(df: pd.DataFrame) -> float:
    """
    Structural MA confirmation.
    """
    if pd.isna(df["SMA_20"].iloc[-1]) or pd.isna(df["SMA_50"].iloc[-1]):
        return 0.0

    if df["close"].iloc[-1] > df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1]:
        return 1.0
    return 0.0
    
def _slope(series: np.ndarray) -> float:
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series))
    try:
        return np.polyfit(x, series, 1)[0]
    except Exception:
        return np.nan
        
def compute_new_low_penalty(close: pd.Series, window: int) -> float:
    """
    Penalizes recent breakdown behavior.
    """
    if len(close) < window:
        return np.nan

    lows = close == close.rolling(window).min()
    return lows.iloc[-window:].mean()


_CANONICAL_FUNDAMENTALS_CACHE = {}
_VALUATION_DATA_SOURCE_BY_TICKER = {}

CANONICAL_TO_RAW_METRICS = {
    "Total Revenue": ("Revenue_USD", "flow"),
    "Net Income": ("Net_Income_USD", "flow"),
    "Operating Income": ("Operating_Income_USD", "flow"),
    "Gross Profit": ("Gross_Profit_USD", "flow"),
    "Depreciation, Depletion, and Amortization": ("Depreciation_Amortization_USD", "flow"),
    "Income Tax Expense": ("Income_Tax_Expense_USD", "flow"),
    "Income Before Tax": ("Income_Before_Tax_USD", "flow"),
    "Interest Expense": ("Interest_Expense_USD", "flow"),
    "Total Assets": ("Assets_USD", "point"),
    "Current Assets": ("Current_Assets_USD", "point"),
    "Current Liabilities": ("Current_Liabilities_USD", "point"),
    "Net PP&E": ("Net_PPE_USD", "point"),
    "Goodwill": ("Goodwill_USD", "point"),
    "Net Intangible Assets": ("Net_Intangible_Assets_USD", "point"),
    "Cash & Cash Equivelance": ("Cash_USD", "point"),
    "Short Term Investments": ("Short_Term_Investments_USD", "point"),
    "Short Term Debt (STD)": ("Short_Term_Debt_USD", "point"),
    "Long Term Debt (LTD)": ("Long_Term_Debt_USD", "point"),
    "Operating Lease Liability": ("Operating_Lease_Liability_USD", "point"),
    "Preferred Stock": ("Preferred_Stock_USD", "point"),
    "Minority Interest": ("Minority_Interest_USD", "point"),
    "Total Common Shares Outstanding": ("Shares_Basic", "point"),
    "Weighted Average Shares": ("Shares_Weighted_Average_Diluted", "point"),
    "Cash From Operations": ("Cash_From_Operations_USD", "flow"),
    "Payments To Acquire PP&E": ("Capital_Expenditure_USD", "flow_abs"),
    "Operating Lease ROU Asset": ("Operating_Lease_ROU_Asset_USD", "point"),
    "Weighted Average Lease Discount Rate": ("Weighted_Average_Lease_Discount_Rate", "point"),
    "Closing Price (USD)": ("Price_USD", "point"),
    "Market Cap": ("Market_Cap_USD", "point"),
    "Enterprise Value": ("Enterprise_Value_USD", "point"),
    "EBITDA": ("EBITDA_USD", "flow"),
    "Free Cash Flow": ("Free_Cash_Flow_USD", "flow"),
    "EV / EBITDA": ("EV_EBITDA", "point"),
    "Price / Sales": ("PS", "point"),
    "P/E Ratio (Calculated)": ("PE", "point"),
    "ROIC (Heavy)": ("ROIC", "point"),
    "ROIC (Light)": ("ROIC", "point"),
}


def load_canonical_fundamentals(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else CANONICAL_FUNDAMENTALS_PATH
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _CANONICAL_FUNDAMENTALS_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    try:
        frame = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if frame.empty or "Ticker" not in frame.columns:
        return pd.DataFrame()

    frame = frame.copy()
    frame["Ticker"] = frame["Ticker"].astype(str).str.upper().str.strip()
    if "Dimension" in frame.columns:
        frame["Dimension"] = frame["Dimension"].fillna("").astype(str).str.upper().str.strip()

    for col in ["Filing_Date", "Calendar_Date", "Fiscal_Period_End"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.normalize()

    _CANONICAL_FUNDAMENTALS_CACHE.clear()
    _CANONICAL_FUNDAMENTALS_CACHE[key] = frame
    return frame.copy()


def select_canonical_fundamental_rows(
    fundamentals: pd.DataFrame,
    ticker: str,
    asof_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    if fundamentals.empty:
        return pd.DataFrame(), ""

    ticker = str(ticker).upper().strip()
    rows = fundamentals[fundamentals["Ticker"] == ticker].copy()
    if rows.empty or "Filing_Date" not in rows.columns:
        return pd.DataFrame(), ""

    rows = rows[rows["Filing_Date"].notna() & (rows["Filing_Date"] <= asof_date)].copy()
    if rows.empty:
        return pd.DataFrame(), ""

    if "Dimension" not in rows.columns:
        return rows, ""

    for dimension in CANONICAL_FUNDAMENTAL_DIMENSION_PRIORITY:
        selected = rows[rows["Dimension"] == dimension].copy()
        if not selected.empty:
            return selected, dimension

    dimension = str(rows["Dimension"].dropna().iloc[0]) if rows["Dimension"].notna().any() else ""
    return rows[rows["Dimension"] == dimension].copy() if dimension else rows, dimension


def reset_valuation_data_source_log() -> None:
    _VALUATION_DATA_SOURCE_BY_TICKER.clear()


def record_valuation_data_source(ticker: str, source: str, detail: str = "") -> None:
    _VALUATION_DATA_SOURCE_BY_TICKER[str(ticker).upper().strip()] = {
        "Valuation_Data_Source": source,
        "Valuation_Data_Source_Detail": detail,
    }


def canonical_fundamental_source_detail(
    ticker: str,
    asof_date: pd.Timestamp | str | None,
) -> str:
    asof_date = normalize_asof_date(asof_date)
    fundamentals = load_canonical_fundamentals()
    rows, dimension = select_canonical_fundamental_rows(
        fundamentals=fundamentals,
        ticker=ticker,
        asof_date=asof_date,
    )
    if rows.empty:
        return "No canonical fundamental row with Filing_Date <= as-of date"

    latest_filing = rows["Filing_Date"].max() if "Filing_Date" in rows.columns else pd.NaT
    latest_text = (
        pd.to_datetime(latest_filing).strftime("%Y-%m-%d")
        if pd.notna(latest_filing) else "unknown"
    )
    dimension_text = dimension or "unknown_dimension"
    return f"Dimension {dimension_text}; latest filing {latest_text}; rows {len(rows)}"


def valuation_data_source_detail(ticker: str) -> dict:
    return _VALUATION_DATA_SOURCE_BY_TICKER.get(
        str(ticker).upper().strip(),
        {
            "Valuation_Data_Source": "unknown",
            "Valuation_Data_Source_Detail": "",
        },
    )


def fiscal_year_from_canonical_rows(rows: pd.DataFrame) -> pd.Series:
    if "Fiscal_Period_End" in rows.columns:
        dates = rows["Fiscal_Period_End"]
    elif "Calendar_Date" in rows.columns:
        dates = rows["Calendar_Date"]
    else:
        dates = rows["Filing_Date"]
    return pd.to_datetime(dates, errors="coerce").dt.year


def period_order_from_canonical_rows(rows: pd.DataFrame) -> pd.Series:
    candidates = []
    for col in ["Fiscal_Period_End", "Calendar_Date", "Filing_Date"]:
        if col in rows.columns:
            candidates.append(pd.to_datetime(rows[col], errors="coerce"))
    if not candidates:
        return pd.Series(pd.NaT, index=rows.index)
    order = candidates[0].copy()
    for candidate in candidates[1:]:
        order = order.fillna(candidate)
    return order


def latest_canonical_rows_by_fiscal_year(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    out["_Fiscal_Year"] = fiscal_year_from_canonical_rows(out)
    out["_Period_Order"] = period_order_from_canonical_rows(out)
    out = out.dropna(subset=["_Fiscal_Year"]).copy()
    if out.empty:
        return out
    out["_Fiscal_Year"] = out["_Fiscal_Year"].astype(int)
    return (
        out.sort_values(["_Fiscal_Year", "_Period_Order", "Filing_Date"])
        .drop_duplicates("_Fiscal_Year", keep="last")
    )


def canonical_metric_series(
    rows: pd.DataFrame,
    metric_column: str,
    metric_kind: str,
    dimension: str,
) -> dict[int, float]:
    if rows.empty or metric_column not in rows.columns:
        return {}

    working = rows.copy()
    working["_Fiscal_Year"] = fiscal_year_from_canonical_rows(working)
    working["_Period_Order"] = period_order_from_canonical_rows(working)
    working = working.dropna(subset=["_Fiscal_Year", "_Period_Order"]).copy()
    if working.empty:
        return {}

    working["_Fiscal_Year"] = working["_Fiscal_Year"].astype(int)
    working["_Metric_Value"] = pd.to_numeric(working[metric_column], errors="coerce")
    if metric_kind == "flow_abs":
        working["_Metric_Value"] = working["_Metric_Value"].abs()
    working = working.dropna(subset=["_Metric_Value"]).copy()
    if working.empty:
        return {}

    if dimension == "ARQ" and metric_kind in ("flow", "flow_abs"):
        quarterly = (
            working.sort_values(["_Period_Order", "Filing_Date"])
            .drop_duplicates("_Period_Order", keep="last")
        )
        quarterly["_Metric_TTM"] = quarterly["_Metric_Value"].rolling(4, min_periods=4).sum()
        latest = (
            quarterly.dropna(subset=["_Metric_TTM"])
            .sort_values(["_Fiscal_Year", "_Period_Order", "Filing_Date"])
            .drop_duplicates("_Fiscal_Year", keep="last")
        )
        value_column = "_Metric_TTM"
    else:
        latest = (
            working.sort_values(["_Fiscal_Year", "_Period_Order", "Filing_Date"])
            .drop_duplicates("_Fiscal_Year", keep="last")
        )
        value_column = "_Metric_Value"

    series = {}
    for _, row in latest.iterrows():
        year = int(row["_Fiscal_Year"])
        value = row[value_column]
        if is_finite_number(value):
            series[year] = float(value)
    return series


def build_raw_financial_data_from_canonical_fundamentals(
    ticker: str,
    asof_date: pd.Timestamp | str | None = None,
    market_snapshot: dict | None = None,
    fundamentals_path: Path | str | None = None,
) -> dict:
    asof_date = normalize_asof_date(asof_date)
    fundamentals = load_canonical_fundamentals(fundamentals_path)
    rows, dimension = select_canonical_fundamental_rows(
        fundamentals=fundamentals,
        ticker=ticker,
        asof_date=asof_date,
    )
    if rows.empty:
        return {}

    raw = {}
    for metric_name, (column, kind) in CANONICAL_TO_RAW_METRICS.items():
        series = canonical_metric_series(
            rows=rows,
            metric_column=column,
            metric_kind=kind,
            dimension=dimension,
        )
        if series:
            raw[metric_name] = series

    if market_snapshot and is_finite_number(market_snapshot.get("price", np.nan)):
        latest_rows = latest_canonical_rows_by_fiscal_year(rows)
        latest_year = (
            int(latest_rows["_Fiscal_Year"].max())
            if not latest_rows.empty and "_Fiscal_Year" in latest_rows.columns
            else asof_date.year
        )
        raw["Closing Price (USD)"] = {latest_year: float(market_snapshot["price"])}

    return raw


def build_valuation_financials_from_canonical_fundamentals(
    ticker: str,
    asof_date: pd.Timestamp | str | None = None,
    market_snapshot: dict | None = None,
    fundamentals_path: Path | str | None = None,
) -> dict:
    raw = build_raw_financial_data_from_canonical_fundamentals(
        ticker=ticker,
        asof_date=asof_date,
        market_snapshot=market_snapshot,
        fundamentals_path=fundamentals_path,
    )
    if not raw:
        return {}

    raw = {
        metric: series
        for metric, series in raw.items()
        if isinstance(series, dict)
        and (
            len(series) >= 5
            or metric in POINT_IN_TIME_METRICS
            or metric == "Closing Price (USD)"
            or metric in BASE_VALUATION_WEIGHTS
            or metric in {"Market Cap", "Enterprise Value", "EBITDA", "Free Cash Flow"}
        )
    }
    raw = ensure_optional_zero_metrics(raw)
    raw = add_market_cap_share_count_fallback(raw)
    calculated = run_calculated_equations(raw, CALCULATED_EQUATIONS_VALUATION)
    return calculated

#=====================================================================================
#
#
#=======(9)================Valuation Multiple Computation===========(9)===============
# Quick Summary
# Combines normalized fundamentals with current market data to calculate pricing ratios that express how the market values a business.
# This stage computes metrics such as EV/EBITDA, P/FCF, P/S, and value spreads, while enforcing rules like excluding negative EBITDA or invalid inputs.
# The outputs are raw valuation ratios, not scores, and serve as inputs for benchmark comparison and fair value scoring in later stages.
#------------------------------------------------------------------------------------
#
#--------------------------------Functions List--------------------------------------
# A. build_valuation_dataframe
# B. build_valuation_dataframe_for_universe
# C. compute_fair_value_score
#------------------------------------------------------------------------------------
#
#------------------------------------Functions----------------------------------------


def build_valuation_financials(
    ticker: str,
    asof_date: pd.Timestamp | str | None = None,
    market_snapshot: dict | None = None,
    debug_sec_writer: pd.ExcelWriter | None = None,
    use_sec_cache: bool = True,
    refresh_sec_cache: bool = False
) -> dict:
    canonical = build_valuation_financials_from_canonical_fundamentals(
        ticker=ticker,
        asof_date=asof_date,
        market_snapshot=market_snapshot,
    )
    if canonical:
        record_valuation_data_source(
            ticker,
            "sharadar_canonical_fundamentals",
            canonical_fundamental_source_detail(ticker, asof_date),
        )
        return canonical

    # ----------------------------------
    # 1) Raw SEC scrape fallback
    # ----------------------------------
    raw = build_raw_financial_data(
        ticker=ticker,
        asof_date=asof_date,
        market_snapshot=market_snapshot,
        debug_sec_writer=debug_sec_writer,
        use_sec_cache=use_sec_cache,
        refresh_sec_cache=refresh_sec_cache
    )

    # ----------------------------------
    # 2) 🔒 CANONICAL NORMALIZATION (REQUIRED)
    # ----------------------------------
    raw = normalize_sec_to_year_map(raw)

    for metric, year_map in raw.items():
        if not isinstance(year_map, dict):
            continue
        for k in year_map.keys():
            if not isinstance(k, int):
                raise RuntimeError(
                    f"[FATAL] normalize_sec_to_year_map failed: {metric} → key={k}"
                )
    # ----------------------------------
    # 3) Enforce minimum history (POST-normalization)
    # ----------------------------------
    raw = {
        metric: series
        for metric, series in raw.items()
        if isinstance(series, dict) and (len(series) >= 5 or metric in POINT_IN_TIME_METRICS or metric == "Closing Price (USD)")
    }
    raw = ensure_optional_zero_metrics(raw)
    raw = add_market_cap_share_count_fallback(raw)

    # ----------------------------------
    # 4) Run calculated equations
    # ----------------------------------
    calculated = run_calculated_equations(
        raw,
        CALCULATED_EQUATIONS_VALUATION
    )

    record_valuation_data_source(
        ticker,
        "sec_companyfacts_fallback" if calculated else "missing_valuation_data",
        canonical_fundamental_source_detail(ticker, asof_date),
    )
    return calculated

def build_valuation_dataframe(
    tickers: list[str],
    asof_date: pd.Timestamp | None = None,
    debug_sec_writer: pd.ExcelWriter | None = None,
    price_data: dict | None = None,
    market_data: dict | None = None,
    use_sec_cache: bool = True,
    refresh_sec_cache: bool = False
) -> pd.DataFrame:
    """
    Builds valuation dataframe:
    - rows = valuation metrics
    - columns = tickers
    - values = most recent available value
    """

    rows = {}
    reset_valuation_data_source_log()
    DEBUG_ONE_TRACE = True  # local, safe, deterministic
    asof_date = normalize_asof_date(asof_date)
    if market_data is None:
        market_data = build_market_data_map(
            tickers=tickers,
            asof_date=asof_date,
            price_data=price_data
        )

    # ==================================================
    # MAIN VALUATION LOOP
    # ==================================================
    for ticker in tickers:
        try:
            # ------------------------------
            # Build valuation inputs.
            #
            # Prefer canonical vendor fundamentals when present. The canonical
            # path filters by filing date and avoids unnecessary SEC scraping;
            # SEC companyfacts remains the fallback for symbols not covered by
            # the local historical data layer.
            # ------------------------------
            data = build_valuation_financials(
                ticker=ticker,
                asof_date=asof_date,
                market_snapshot=market_data.get(ticker, {}),
                debug_sec_writer=debug_sec_writer,
                use_sec_cache=use_sec_cache,
                refresh_sec_cache=refresh_sec_cache
            )

        except Exception as e:
            print(f"[WARN] Valuation failed for {ticker}: {e}")
            log_fair_value_issue(ticker, f"Valuation calculation failed: {e}")

            if DEBUG_ONE_TRACE:
                DEBUG_ONE_TRACE = False
                raise  # full traceback ONCE

            continue

        # ------------------------------
        # Collect latest values
        # ------------------------------
        for metric, year_map in data.items():
            if not isinstance(year_map, dict):
                continue

            valid_years = [
                y for y in year_map.keys()
                if isinstance(y, (int, np.integer))
            ]
            if not valid_years:
                continue

            latest_year = max(valid_years)
            value = year_map.get(latest_year)

            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                continue

            if not np.isfinite(value):
                continue

            rows.setdefault(metric, {})[ticker] = value

    valuation_df = pd.DataFrame(rows).T.sort_index()
    return valuation_df


def build_valuation_dataframe_for_universe(
    tickers: list[str],
    asof_date: pd.Timestamp | str | None = None,
    price_data: dict | None = None,
    market_data: dict | None = None,
    use_sec_cache: bool = True,
    refresh_sec_cache: bool = False
) -> pd.DataFrame:
    rows = {}
    reset_valuation_data_source_log()
    asof_date = normalize_asof_date(asof_date)
    if market_data is None:
        market_data = build_market_data_map(
            tickers=tickers,
            asof_date=asof_date,
            price_data=price_data
        )

    for ticker in tickers:
        financials = build_valuation_financials(
            ticker=ticker,
            asof_date=asof_date,
            market_snapshot=market_data.get(ticker, {}),
            use_sec_cache=use_sec_cache,
            refresh_sec_cache=refresh_sec_cache
        )

        for metric, series in financials.items():
            if metric not in rows:
                rows[metric] = {}
            if isinstance(series, dict):
                valid_years = [
                    y for y in series.keys()
                    if isinstance(y, (int, np.integer))
                    and np.isfinite(series.get(y))
                ]
                rows[metric][ticker] = series.get(max(valid_years)) if valid_years else np.nan

    return pd.DataFrame(rows).T

def compute_fair_value_score(
    ticker: str,
    subindustry: str,
    valuation_df: pd.DataFrame,
    centers: dict,
    industry_regime: str = "Neutral",
    subindustry_regime: str = "Neutral"
) -> float:
    """
    Computes Fair Value Score using:
    - benchmark centers
    - base weights
    - subindustry structural multipliers
    - dispersion-aware reliability
    - regime multipliers
    """
    subindustry = canonical_subindustry_name(subindustry)

    # -----------------------------
    # Trend-only guard
    # -----------------------------
    if not is_subindustry_benchmarkable(subindustry):
        return np.nan

    bucket = get_ticker_bucket(subindustry, ticker)
    if bucket is None:
        return np.nan

    # -----------------------------
    # Compute metric CVs for weight building
    # -----------------------------
    metric_list = list(BASE_VALUATION_WEIGHTS.keys())
    metric_cvs = compute_metric_cvs(
        valuation_df=valuation_df,
        subindustry=subindustry,
        bucket=bucket,
        metrics=metric_list
    )

    # -----------------------------
    # Build FINAL dynamic weights (now includes regime)
    # -----------------------------
    weights = build_final_valuation_weights(
        subindustry=subindustry,
        metric_cvs=metric_cvs,
        industry_regime=industry_regime,
        subindustry_regime=subindustry_regime
    )

    if not weights:
        return np.nan

    score_sum = 0.0
    weight_sum = 0.0
    metrics_used = 0

    # -----------------------------
    # Metric loop
    # -----------------------------
    for metric, weight in weights.items():

        # Must exist
        if metric not in valuation_df.index:
            continue
        if ticker not in valuation_df.columns:
            continue

        # Ticker value
        try:
            ticker_val = float(valuation_df.at[metric, ticker])
        except Exception:
            continue
        if not np.isfinite(ticker_val):
            continue

        # Benchmark center (precomputed)
        benchmark_val = centers.get((subindustry, bucket, metric), np.nan)
        if not np.isfinite(benchmark_val):
            continue
        benchmark_val = float(benchmark_val)

        # -----------------------------
        # Metric-specific validity guards
        # -----------------------------
        if metric == "EV / EBITDA":
            if "EBITDA" not in valuation_df.index:
                continue
            try:
                ebitda = float(valuation_df.at["EBITDA", ticker])
            except Exception:
                continue
            if not np.isfinite(ebitda) or ebitda <= 0:
                continue

        if metric == "PEG Ratio (Calculated)":
            grow_row = "Earnings Per Share (EPS) CAGR (3-Year) (%)"
            if grow_row not in valuation_df.index:
                continue
            try:
                g = float(valuation_df.at[grow_row, ticker])
            except Exception:
                continue
            if not np.isfinite(g) or g <= 0:
                continue

        if metric == "EV / FCF Forward (3Y)":
            fcf_row = "FCF Forward (3Y)"
            if fcf_row not in valuation_df.index:
                continue
            try:
                f = float(valuation_df.at[fcf_row, ticker])
            except Exception:
                continue
            if not np.isfinite(f) or f <= 0:
                continue

        if metric == "Price / Sales":
            rev_row = "Total Revenue"
            if rev_row not in valuation_df.index:
                continue
            try:
                rev = float(valuation_df.at[rev_row, ticker])
            except Exception:
                continue
            if not np.isfinite(rev) or rev <= 0:
                continue

        if metric == "Gross Profit / EV":
            ev_row = "Enterprise Value"
            if ev_row not in valuation_df.index:
                continue
            try:
                ev = float(valuation_df.at[ev_row, ticker])
            except Exception:
                continue
            if not np.isfinite(ev) or ev <= 0:
                continue

        # -----------------------------
        # Direction logic
        # -----------------------------
        lower_better = metric in (
            "EV / EBITDA",
            "EV / FCF Forward (3Y)",
            "PEG Ratio (Calculated)",
            "Price / Sales",
        )

        if lower_better and ticker_val <= 0:
            continue

        denom = abs(benchmark_val)
        if denom < 1e-9:
            continue

        if lower_better:
            contribution = (benchmark_val - ticker_val) / denom
        else:
            contribution = (ticker_val - benchmark_val) / denom

        contribution = float(np.clip(contribution, -2.0, 2.0))

        score_sum += contribution * weight
        weight_sum += weight
        metrics_used += 1

    if metrics_used < MIN_FAIR_VALUE_METRICS_USED:
        return np.nan

    return (score_sum / weight_sum) if weight_sum > 0 else np.nan


def count_fair_value_metrics_available(
    ticker: str,
    subindustry: str,
    valuation_df: pd.DataFrame,
    centers: dict
) -> int:
    diagnostic = diagnose_fair_value_score(
        ticker=ticker,
        subindustry=subindustry,
        fair_value_score=np.nan,
        valuation_df=valuation_df,
        centers=centers
    )
    return int(diagnostic["Fair_Value_Metrics_Used"])


def subindustry_benchmark_ticker_count(subindustry: str) -> int:
    subindustry = canonical_subindustry_name(subindustry)
    groups = TECH_BENCHMARK_GROUPS.get(subindustry, {})
    return sum(len(tickers) for tickers in groups.values() if isinstance(tickers, list))


def valuation_metric_value(
    valuation_df: pd.DataFrame | None,
    ticker: str,
    metric: str
) -> float:
    if valuation_df is None or valuation_df.empty:
        return np.nan
    if metric not in valuation_df.index or ticker not in valuation_df.columns:
        return np.nan
    try:
        value = float(valuation_df.at[metric, ticker])
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def has_valuation_metric(
    valuation_df: pd.DataFrame | None,
    ticker: str,
    metric: str
) -> bool:
    return np.isfinite(valuation_metric_value(valuation_df, ticker, metric))


def share_count_source_detail(
    ticker: str,
    valuation_df: pd.DataFrame | None
) -> dict:
    share_count = valuation_metric_value(
        valuation_df=valuation_df,
        ticker=ticker,
        metric=MARKET_CAP_SHARE_COUNT_METRIC
    )
    fallback_flag = valuation_metric_value(
        valuation_df=valuation_df,
        ticker=ticker,
        metric=SHARE_COUNT_FALLBACK_FLAG_METRIC
    )

    if not np.isfinite(share_count):
        return {
            "Share_Count_Used": np.nan,
            "Share_Count_Source": "",
            "Share_Count_Source_Category": "Missing usable SEC share count",
            "Share_Count_Fallback_Used": np.nan,
        }

    fallback_used = np.isfinite(fallback_flag) and float(fallback_flag) >= 0.5
    if fallback_used:
        return {
            "Share_Count_Used": share_count,
            "Share_Count_Source": "Weighted Average Shares",
            "Share_Count_Source_Category": "Fallback - SEC/vendor weighted-average share count",
            "Share_Count_Fallback_Used": 1,
        }

    return {
        "Share_Count_Used": share_count,
        "Share_Count_Source": "Total Common Shares Outstanding",
        "Share_Count_Source_Category": "Preferred - SEC/vendor point-in-time common/basic shares",
        "Share_Count_Fallback_Used": 0,
    }


def diagnostic_source_area(metric: str) -> str:
    income_statement = {
        "Total Revenue",
        "Net Income",
        "Operating Income",
        "Gross Profit",
        "Depreciation, Depletion, and Amortization",
        "Income Tax Expense",
        "Income Before Tax",
        "Interest Expense Non-Operating",
        "Interest Expense",
    }
    balance_sheet = {
        "Total Assets",
        "Current Assets",
        "Current Liabilities",
        "Net PP&E",
        "Goodwill",
        "Net Intangible Assets",
        "Cash & Cash Equivelance",
        "Short Term Investments",
        "Short Term Debt (STD)",
        "Long Term Debt (LTD)",
        "Operating Lease Liability",
        "Preferred Stock",
        "Minority Interest",
        "Operating Lease ROU Asset",
        "Weighted Average Lease Discount Rate",
    }
    share_count = {
        "Total Common Shares Outstanding",
        "Weighted Average Shares",
        MARKET_CAP_SHARE_COUNT_METRIC,
    }
    cash_flow = {
        "Cash From Operations",
        "Payments To Acquire PP&E",
    }

    if metric == "Closing Price (USD)":
        return "market price data"
    if metric in cash_flow:
        return "SEC cash-flow statement data"
    if metric in share_count:
        return "SEC share-count data"
    if metric in balance_sheet:
        return "SEC balance sheet data"
    if metric in income_statement:
        return "SEC income statement data"
    return "derived valuation data"


def raw_metric_missing_message(metric: str) -> str:
    if metric == "Closing Price (USD)":
        return "Closing Price (USD) missing from market price data"
    if metric == MARKET_CAP_SHARE_COUNT_METRIC:
        return (
            "Market Cap Share Count missing: neither Total Common Shares Outstanding "
            "nor Weighted Average Shares produced a positive finite SEC share-count "
            "value for a matching fiscal year"
        )
    if metric == SHARE_COUNT_FALLBACK_FLAG_METRIC:
        return "Share-count fallback flag missing because no usable market-cap share count was generated"

    if metric in RELEVANT_LABELS_VALUATION:
        unit_label = "/".join(metric_unit_preferences(metric))
        return (
            f"{metric} missing from {diagnostic_source_area(metric)} "
            f"in required {unit_label} units or acceptable SEC forms"
        )

    return f"{metric} was not generated"


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def missing_metric_input_leaves(
    ticker: str,
    metric: str,
    valuation_df: pd.DataFrame | None,
    visited: set[str] | None = None
) -> list[str]:
    if has_valuation_metric(valuation_df, ticker, metric):
        return []

    if visited is None:
        visited = set()
    if metric in visited:
        return [f"{metric} diagnostic trace repeated a dependency"]
    visited = set(visited)
    visited.add(metric)

    rule = CALCULATED_EQUATIONS_VALUATION.get(metric)
    if not rule:
        return [raw_metric_missing_message(metric)]

    leaves = []
    for component in rule.get("components", []):
        if isinstance(component, (int, float, np.integer, np.floating)):
            continue
        if has_valuation_metric(valuation_df, ticker, component):
            continue
        leaves.extend(
            missing_metric_input_leaves(
                ticker=ticker,
                metric=component,
                valuation_df=valuation_df,
                visited=visited
            )
        )

    if leaves:
        return unique_preserve_order(leaves)

    return [
        (
            f"{metric} calculation produced no finite value from available inputs. "
            "This usually means required inputs do not overlap in the same fiscal year, "
            "a denominator is zero, a CAGR base is negative, or the result is non-finite"
        )
    ]


def summarize_missing_metric_inputs(
    ticker: str,
    metric: str,
    valuation_df: pd.DataFrame | None,
    limit: int = 5
) -> str:
    leaves = missing_metric_input_leaves(
        ticker=ticker,
        metric=metric,
        valuation_df=valuation_df
    )
    if not leaves:
        return "required inputs are present but the metric failed a scoring validity rule"

    shown = leaves[:limit]
    suffix = f"; plus {len(leaves) - limit} more missing inputs" if len(leaves) > limit else ""
    return "; ".join(shown) + suffix


def peer_observation_count(
    valuation_df: pd.DataFrame | None,
    subindustry: str,
    bucket: str,
    metric: str
) -> int:
    if valuation_df is None or valuation_df.empty or metric not in valuation_df.index:
        return 0

    peers = TECH_BENCHMARK_GROUPS.get(canonical_subindustry_name(subindustry), {}).get(bucket, [])
    if not isinstance(peers, list):
        return 0

    count = 0
    for peer in peers:
        if has_valuation_metric(valuation_df, peer, metric):
            count += 1
    return count


def fair_value_metric_diagnostic(
    ticker: str,
    subindustry: str,
    bucket: str,
    metric: str,
    valuation_df: pd.DataFrame,
    centers: dict
) -> tuple[bool, str]:
    ticker_val = valuation_metric_value(valuation_df, ticker, metric)
    if not np.isfinite(ticker_val):
        missing_inputs = summarize_missing_metric_inputs(ticker, metric, valuation_df)
        return False, f"{metric}: ticker metric unavailable. {missing_inputs}."

    benchmark_val = centers.get((subindustry, bucket, metric), np.nan)
    if not np.isfinite(benchmark_val):
        peer_count = peer_observation_count(valuation_df, subindustry, bucket, metric)
        return (
            False,
            (
                f"{metric}: peer benchmark unavailable for {subindustry} {bucket} bucket. "
                f"{peer_count} peers have usable values; {MIN_PEER_OBSERVATIONS} are required"
            )
        )

    if metric == "EV / EBITDA":
        ebitda = valuation_metric_value(valuation_df, ticker, "EBITDA")
        if not np.isfinite(ebitda):
            missing_inputs = summarize_missing_metric_inputs(ticker, "EBITDA", valuation_df)
            return False, f"{metric}: EBITDA unavailable. {missing_inputs}."
        if ebitda <= 0:
            return False, f"{metric}: EBITDA is zero or negative, so EV/EBITDA is not meaningful."

    if metric == "PEG Ratio (Calculated)":
        grow_row = "Earnings Per Share (EPS) CAGR (3-Year) (%)"
        growth = valuation_metric_value(valuation_df, ticker, grow_row)
        if not np.isfinite(growth):
            missing_inputs = summarize_missing_metric_inputs(ticker, grow_row, valuation_df)
            return False, f"{metric}: EPS CAGR unavailable. {missing_inputs}."
        if growth <= 0:
            return False, f"{metric}: EPS CAGR is zero or negative, so PEG is not meaningful."

    if metric == "EV / FCF Forward (3Y)":
        fcf = valuation_metric_value(valuation_df, ticker, "FCF Forward (3Y)")
        if not np.isfinite(fcf):
            missing_inputs = summarize_missing_metric_inputs(ticker, "FCF Forward (3Y)", valuation_df)
            return False, f"{metric}: forward FCF unavailable. {missing_inputs}."
        if fcf <= 0:
            return False, f"{metric}: forward FCF is zero or negative, so EV/FCF is not meaningful."

    if metric == "Price / Sales":
        revenue = valuation_metric_value(valuation_df, ticker, "Total Revenue")
        if not np.isfinite(revenue):
            missing_inputs = summarize_missing_metric_inputs(ticker, "Total Revenue", valuation_df)
            return False, f"{metric}: total revenue unavailable. {missing_inputs}."
        if revenue <= 0:
            return False, f"{metric}: total revenue is zero or negative, so price/sales is not meaningful."

    if metric == "Gross Profit / EV":
        ev = valuation_metric_value(valuation_df, ticker, "Enterprise Value")
        if not np.isfinite(ev):
            missing_inputs = summarize_missing_metric_inputs(ticker, "Enterprise Value", valuation_df)
            return False, f"{metric}: enterprise value unavailable. {missing_inputs}."
        if ev <= 0:
            return False, f"{metric}: enterprise value is zero or negative, so gross profit/EV is not meaningful."

    lower_better = metric in (
        "EV / EBITDA",
        "EV / FCF Forward (3Y)",
        "PEG Ratio (Calculated)",
        "Price / Sales",
    )
    if lower_better and ticker_val <= 0:
        return False, f"{metric}: valuation multiple is zero or negative and cannot be scored."

    if abs(float(benchmark_val)) < 1e-9:
        return False, f"{metric}: peer benchmark is effectively zero, so relative scoring is unstable."

    return True, f"{metric}: usable"


def diagnose_fair_value_score(
    ticker: str,
    subindustry: str,
    fair_value_score: float,
    valuation_df: pd.DataFrame,
    centers: dict,
    industry_regime: str = "Neutral",
    subindustry_regime: str = "Neutral"
) -> dict:
    ticker = str(ticker).upper()
    subindustry = canonical_subindustry_name(subindustry)
    bucket = get_ticker_bucket(subindustry, ticker)
    logged_issues = summarize_logged_fair_value_issues(ticker)

    diagnostic = {
        "Fair_Value_Metrics_Used": 0,
        "Fair_Value_Data_Status": "OK",
        "Fair_Value_Benchmark_Bucket": bucket or "",
        "Fair_Value_Usable_Metrics": "",
        "Fair_Value_Missing_Reason": "",
        "Fair_Value_Missing_Metrics": "",
    }

    peer_total = subindustry_benchmark_ticker_count(subindustry)
    if not is_subindustry_benchmarkable(subindustry):
        diagnostic["Fair_Value_Data_Status"] = "No fair value - subindustry too small"
        diagnostic["Fair_Value_Missing_Reason"] = (
            f"No fair value score for {ticker}: {subindustry} has {peer_total} benchmark peers, "
            f"below the required {MIN_BENCHMARK_SUBINDUSTRY_SIZE}. Fair-value peer scoring is disabled for this group."
        )
        return diagnostic

    if bucket is None:
        diagnostic["Fair_Value_Data_Status"] = "No fair value - benchmark bucket missing"
        diagnostic["Fair_Value_Missing_Reason"] = (
            f"No fair value score for {ticker}: ticker is not assigned to a Small, Mid, or Large "
            f"benchmark bucket inside {subindustry}."
        )
        return diagnostic

    if valuation_df is None or valuation_df.empty:
        diagnostic["Fair_Value_Data_Status"] = "No fair value - valuation data missing"
        diagnostic["Fair_Value_Missing_Reason"] = (
            f"No fair value score for {ticker}: no valuation dataframe was generated."
        )
        return diagnostic

    if ticker not in valuation_df.columns:
        diagnostic["Fair_Value_Data_Status"] = "No fair value - valuation data missing"
        diagnostic["Fair_Value_Missing_Reason"] = (
            f"No fair value score for {ticker}: no SEC or market-derived valuation metrics were generated."
        )
        if logged_issues:
            diagnostic["Fair_Value_Missing_Metrics"] = f"Scrape notes: {logged_issues}"
        return diagnostic

    metric_cvs = compute_metric_cvs(
        valuation_df=valuation_df,
        subindustry=subindustry,
        bucket=bucket,
        metrics=list(BASE_VALUATION_WEIGHTS.keys())
    )
    weights = build_final_valuation_weights(
        subindustry=subindustry,
        metric_cvs=metric_cvs,
        industry_regime=industry_regime,
        subindustry_regime=subindustry_regime
    )

    if not weights:
        diagnostic["Fair_Value_Data_Status"] = "No fair value - valuation weights missing"
        diagnostic["Fair_Value_Missing_Reason"] = (
            f"No fair value score for {ticker}: valuation weights could not be generated for {subindustry}."
        )
        return diagnostic

    usable_metrics = []
    missing_metric_reasons = []

    for metric in weights:
        usable, reason = fair_value_metric_diagnostic(
            ticker=ticker,
            subindustry=subindustry,
            bucket=bucket,
            metric=metric,
            valuation_df=valuation_df,
            centers=centers
        )
        if usable:
            usable_metrics.append(metric)
        else:
            missing_metric_reasons.append(reason)

    diagnostic["Fair_Value_Metrics_Used"] = len(usable_metrics)
    diagnostic["Fair_Value_Usable_Metrics"] = ", ".join(usable_metrics)
    diagnostic["Fair_Value_Missing_Metrics"] = "; ".join(missing_metric_reasons)

    if len(usable_metrics) < MIN_FAIR_VALUE_METRICS_USED:
        diagnostic["Fair_Value_Data_Status"] = "No fair value - insufficient usable metrics"
        diagnostic["Fair_Value_Missing_Reason"] = (
            f"No fair value score for {ticker}: only {len(usable_metrics)} of "
            f"{MIN_FAIR_VALUE_METRICS_USED} required fair-value metrics passed all SEC, market, "
            "peer benchmark, and sign-validity checks."
        )
        if logged_issues:
            diagnostic["Fair_Value_Missing_Metrics"] = (
                diagnostic["Fair_Value_Missing_Metrics"] + f"; Scrape notes: {logged_issues}"
            ).strip("; ")
        return diagnostic

    if not is_finite_number(fair_value_score):
        diagnostic["Fair_Value_Data_Status"] = "No fair value - score normalization failed"
        diagnostic["Fair_Value_Missing_Reason"] = (
            f"No fair value score for {ticker}: enough usable metrics were found, but final "
            "fair-value normalization returned a missing score."
        )
        return diagnostic

    diagnostic["Fair_Value_Missing_Metrics"] = ""
    return diagnostic


def fair_value_data_status(
    ticker: str,
    subindustry: str,
    fair_value_score: float,
    metrics_used: int
) -> str:
    if not is_subindustry_benchmarkable(subindustry):
        return "No fair value - subindustry too small"
    if metrics_used < MIN_FAIR_VALUE_METRICS_USED:
        return "No fair value - insufficient usable metrics"
    if not is_finite_number(fair_value_score):
        return "No fair value - score normalization failed"
    return "OK"
    

#=====================================================================================
#
#
#=======(10)======Benchmark Normalization (Relative valuation context)===========(10)=======
# Quick Summary
# Compares each company’s valuation metrics against appropriate peer benchmarks at the industry or sub-industry level.
# This stage establishes what “cheap” or “expensive” means in context by normalizing valuation ratios relative to benchmark centers or distributions.
# Its output provides benchmark-relative valuation signals that can be fairly compared across stocks and later converted into scores.
#------------------------------------------------------------------------------------
#
#--------------------------------Functions List--------------------------------------
# A. rebuild_benchmarks_if_needed
# B. build_benchmark_centers
# C. compute_benchmark_center
# D. should_rebuild_benchmark
# E. update_benchmark_metadata
# F. load_benchmark_metadata
# G. save_benchmark_metadata
# H. load_benchmark_centers
# I. save_benchmark_centers
# J. is_subindustry_benchmarkable
#------------------------------------------------------------------------------------
#
#------------------------------------Functions----------------------------------------

def rebuild_benchmarks_if_needed(
    valuation_df: pd.DataFrame,
    valuation_metrics: dict
) -> dict:
    """
    Rebuilds benchmark centers only when SEC coverage allows.
    Returns centers dict.
    """

    metadata = load_benchmark_metadata()
    centers = load_benchmark_centers()

    for subindustry in TECH_BENCHMARK_GROUPS.keys():

        if not is_subindustry_benchmarkable(subindustry):
            continue

        if not should_rebuild_benchmark(subindustry, metadata):
            continue

        filing_result = compute_subindustry_filing_coverage(
            subindustry=subindustry,
            last_benchmark_date=pd.to_datetime(
                metadata.get(subindustry, {}).get("last_update", "1900-01-01")
            )
        )

        if not filing_result.get("eligible", False):
            continue

        # rebuild centers for this subindustry only
        new_centers = build_benchmark_centers(
            valuation_df=valuation_df,
            valuation_metrics=valuation_metrics
        )

        # merge
        centers.update({
            k: v for k, v in new_centers.items()
            if k[0] == subindustry
        })

        update_benchmark_metadata(subindustry, filing_result, metadata)

    save_benchmark_centers(centers)
    save_benchmark_metadata(metadata)

    return centers


def build_benchmark_centers( # If Cv threshold holds true, uses average, else, uses median
    valuation_df: pd.DataFrame,
    valuation_metrics: dict,
    cv_thresholds: dict | None = None
) -> dict:
    """
    Builds benchmark centers for each (subindustry, bucket, metric).
    Returns:
        centers[(subindustry, bucket, metric_name)] = center_value
    """
    if cv_thresholds is None:
        cv_thresholds = {}

    centers = {}

    for subindustry, buckets in TECH_BENCHMARK_GROUPS.items():
        if not is_subindustry_benchmarkable(subindustry):
            continue  # trend-only, skip all centers

        for bucket, tickers in buckets.items():
            if not isinstance(tickers, list) or len(tickers) == 0:
                continue

            for metric_name in valuation_metrics.keys():
                # metric must exist in valuation_df index
                if metric_name not in valuation_df.index:
                    continue

                peer_vals = []
                for t in tickers:
                    if t not in valuation_df.columns:
                        continue
                    try:
                        v = float(valuation_df.at[metric_name, t])
                        if np.isfinite(v):
                            peer_vals.append(v)
                    except:
                        continue

                if len(peer_vals) < MIN_PEER_OBSERVATIONS:
                    continue

                # choose threshold
                thr = cv_thresholds.get(
                    metric_name,
                    CV_THRESHOLDS_BY_METRIC.get(metric_name, DEFAULT_CV_THRESHOLD)
                )

                center = compute_benchmark_center(peer_vals, cv_threshold=thr)
                centers[(subindustry, bucket, metric_name)] = center

    return centers


def compute_benchmark_center(values: list[float], cv_threshold: float) -> float:
    """
    Uses median if dispersion is high (CV > threshold), else mean.
    """
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return np.nan

    mean = float(np.mean(arr))
    std = float(np.std(arr))

    # If mean ~ 0, CV becomes unstable -> use median
    if mean == 0.0:
        return float(np.median(arr))

    cv = std / abs(mean)
    return float(np.median(arr)) if cv > cv_threshold else mean


def should_rebuild_benchmark(
    subindustry: str,
    benchmark_metadata: dict
) -> bool:
    """
    Determines whether benchmark recomputation is allowed.
    """

    last_update = benchmark_metadata.get(subindustry, {}).get(
        "last_update", pd.Timestamp("1900-01-01")
    )

    result = compute_subindustry_filing_coverage(
        subindustry=subindustry,
        last_benchmark_date=pd.to_datetime(last_update)
    )

    return result.get("eligible", False)
    
def update_benchmark_metadata(
    subindustry: str,
    filing_result: dict,
    benchmark_metadata: dict
):
    benchmark_metadata[subindustry] = {
        "last_update": filing_result["latest_filing"].strftime("%Y-%m-%d"),
        "fiscal_period": filing_result["latest_fiscal_period"].strftime("%Y-%m-%d"),
        "coverage": filing_result["coverage"]
    }


def is_subindustry_benchmarkable(
    subindustry: str,
    min_total: int = MIN_BENCHMARK_SUBINDUSTRY_SIZE
) -> bool:
    """
    True if sub-industry has >= min_total stocks across ALL size buckets.
    If False -> trend-only (no valuation benchmarking).
    """
    subindustry = canonical_subindustry_name(subindustry)
    groups = TECH_BENCHMARK_GROUPS.get(subindustry, {})
    total = 0

    for _, tickers in groups.items():
        if isinstance(tickers, list):
            total += len(tickers)

    return total >= min_total


def load_benchmark_metadata() -> dict:
    if BENCHMARK_META_PATH.exists():
        with open(BENCHMARK_META_PATH, "r") as f:
            return json.load(f)
    return {}


def save_benchmark_metadata(metadata: dict):
    with open(BENCHMARK_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

def save_benchmark_centers(centers: dict):
    serializable = {
        f"{canonical_subindustry_name(k[0])}||{k[1]}||{k[2]}": v
        for k, v in centers.items()
        if np.isfinite(v)
    }
    with open(BENCHMARK_CENTERS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)


def load_benchmark_centers() -> dict:
    if not BENCHMARK_CENTERS_PATH.exists():
        return {}

    with open(BENCHMARK_CENTERS_PATH, "r") as f:
        raw = json.load(f)

    centers = {}
    for k, v in raw.items():
        sub, bucket, metric = k.split("||")
        sub = canonical_subindustry_name(sub)
        centers[(sub, bucket, metric)] = float(v)

    return centers



#=====================================================================================
#
#
#=======(11)===========Score Normalization & Calibration (0–100)===========(11)=======
# Quick Summary
# Rescales raw valuation and price-based signals into a common, interpretable scoring range.
# This stage ensures different metrics and components are directly comparable by applying consistent scaling, clipping, or percentile-based normalization before final combination and weighting.
#------------------------------------------------------------------------------------
#
#--------------------------------Functions List--------------------------------------
# A. build_fair_value_scores
# B. build_fair_value_scores_for_universe
# C. normalize_fair_value_scores
# NOTE: PTS IS ALREADY NORMALIZED AND CALUBRATED WITHIN CODE...so this is just fair_value score
#------------------------------------------------------------------------------------
#
#------------------------------------Functions----------------------------------------
def build_fair_value_scores(
    valuation_df: pd.DataFrame,
    ticker_to_subindustry: dict,
    centers: dict,
    industry_regime_by_subindustry: dict | None = None,
    subindustry_regime_map: dict | None = None
) -> dict:
    """
    Returns dict[ticker -> Fair Value Score (0–100 or NaN)]

    industry_regime_by_subindustry:
        optional dict[subindustry -> "Bull"/"Neutral"/"Bear"]
        (if you only have ONE industry regime, you can just pass the same value everywhere)

    subindustry_regime_map:
        optional dict[subindustry -> "Bull"/"Neutral"/"Bear"]
    """
    if industry_regime_by_subindustry is None:
        industry_regime_by_subindustry = {}

    if subindustry_regime_map is None:
        subindustry_regime_map = {}

    raw_scores = {}

    for ticker in sorted(set(ticker_to_subindustry.keys()) | set(valuation_df.columns)):
        sub = ticker_to_subindustry.get(ticker)
        if not sub:
            continue

        ind_regime = industry_regime_by_subindustry.get(sub, "Neutral")
        sub_regime = subindustry_regime_map.get(sub, "Neutral")

        raw_scores[ticker] = compute_fair_value_score(
            ticker=ticker,
            subindustry=sub,
            valuation_df=valuation_df,
            centers=centers,
            industry_regime=ind_regime,
            subindustry_regime=sub_regime
        )

    scores = pd.Series(raw_scores, dtype="float64")
    scores = normalize_fair_value_scores(scores)
    return scores.to_dict()


def build_fair_value_scores_for_universe(
    valuation_df: pd.DataFrame,
    centers: dict,
    ticker_to_subindustry: dict,
    industry_regime_by_subindustry: dict | None = None,
    subindustry_regime_map: dict | None = None
) -> dict:
    return build_fair_value_scores(
        valuation_df=valuation_df,
        ticker_to_subindustry=ticker_to_subindustry,
        centers=centers,
        industry_regime_by_subindustry=industry_regime_by_subindustry,
        subindustry_regime_map=subindustry_regime_map
    )

def normalize_fair_value_scores(fv_scores: pd.Series) -> pd.Series: #This Calibrates FVS
    """
    Normalizes Fair Value Scores to 0-100 scale using robust bounds.
    """
    valid = fv_scores.dropna()

    if valid.empty:
        return fv_scores

    min_v = valid.quantile(0.05)
    max_v = valid.quantile(0.95)

    if min_v == max_v:
        return fv_scores.apply(lambda _: 50.0)

    clipped = fv_scores.clip(lower=min_v, upper=max_v)
    return ((clipped - min_v) / (max_v - min_v) * 100).round(2)

# --------------------------------------------------
# BASE VALUATION WEIGHTS (GLOBAL)
# --------------------------------------------------
BASE_VALUATION_WEIGHTS = {
    "EV / EBITDA": 0.22,
    "EV / FCF Forward (3Y)": 0.22,
    "Price / Sales": 0.16,
    "PEG Ratio (Calculated)": 0.14,
    "Gross Profit / EV": 0.14,
    "ROIC (Heavy)": 0.06,
    "ROIC (Light)": 0.06,
}
# --------------------------------------------------
# SUB-INDUSTRY STRUCTURAL MULTIPLIERS
# --------------------------------------------------
SUBINDUSTRY_VALUATION_MULTIPLIERS = {

    "Application Software": {
        "EV / EBITDA": 0.70,
        "EV / FCF Forward (3Y)": 1.20,
        "Price / Sales": 1.30,
        "PEG Ratio (Calculated)": 1.20,
        "Gross Profit / EV": 1.20,
        "ROIC (Heavy)": 0.90,
        "ROIC (Light)": 0.90,
    },

    "Semiconductors": {
        "EV / EBITDA": 1.25,
        "EV / FCF Forward (3Y)": 1.20,
        "Price / Sales": 0.70,
        "PEG Ratio (Calculated)": 0.80,
        "Gross Profit / EV": 1.10,
        "ROIC (Heavy)": 1.00,
        "ROIC (Light)": 1.00,
    },

    "Semiconductor Equipment": {
        "EV / EBITDA": 1.30,
        "EV / FCF Forward (3Y)": 1.15,
        "Price / Sales": 0.65,
        "PEG Ratio (Calculated)": 0.75,
        "Gross Profit / EV": 1.15,
        "ROIC (Heavy)": 1.05,
        "ROIC (Light)": 1.05,
    },

    "Hardware and Storage": {
        "EV / EBITDA": 1.10,
        "EV / FCF Forward (3Y)": 1.10,
        "Price / Sales": 0.85,
        "PEG Ratio (Calculated)": 0.90,
        "Gross Profit / EV": 1.15,
        "ROIC (Heavy)": 1.15,
        "ROIC (Light)": 1.15,
    },

    "System Software": {
        "EV / EBITDA": 0.85,
        "EV / FCF Forward (3Y)": 1.15,
        "Price / Sales": 1.15,
        "PEG Ratio (Calculated)": 1.10,
        "Gross Profit / EV": 1.10,
        "ROIC (Heavy)": 0.95,
        "ROIC (Light)": 0.95,
    },

    "Cybersecurity": {
        "EV / EBITDA": 0.80,
        "EV / FCF Forward (3Y)": 1.15,
        "Price / Sales": 1.20,
        "PEG Ratio (Calculated)": 1.25,
        "Gross Profit / EV": 1.15,
        "ROIC (Heavy)": 0.95,
        "ROIC (Light)": 0.95,
    },
}

REGIME_VALUATION_MULTIPLIERS = {
    "Industry": {
        "Bull": {},
        "Neutral": {},
        "Bear": {},
    },
    "SubIndustry": {
        "Bull": {},
        "Neutral": {},
        "Bear": {},
    },
}


#=====================================================================================
#
#
#========(12)===============Regime-Conditioned Weighting==============(12)============
# Quick Summary:
# Adjusts the importance of price, valuation, and fundamental signals based on the prevailing market, industry, or sub-industry regime.
# This stage dynamically reweights components (e.g., growth vs. stability, momentum vs. valuation) so final scores reflect what historically matters most in the current regime.
#------------------------------------------------------------------------------------
#
#--------------------------------Functions List--------------------------------------
# A. classify_tech_industry_regime
# B. compute_tech_industry_snapshot
# C. compute_subindustry_snapshot
# D. build_subindustry_regime_features
# E. classify_subindustry_stock_flow
# F. combine_subindustry_regimes
# G. get_subindustry_regime_on_date
# H. normalize_regime_for_valuation
# I. build_valuation_weights
# J. build_final_valuation_weights
# K. get_trend_weight
# L. get_regime_multipliers
# M. resolve_regime_multiplier
# N. apply_regime_multipliers
#------------------------------------------------------------------------------------
#
#------------------------------------Functions----------------------------------------

def classify_tech_industry_regime(
    gdt_regime: str,
    semi_regime: str,
    green_regimes: list[str],
    yellow_regimes: list[str],
    red_regimes: list[str],
    warning_regime: str
) -> str:
    """
    Returns: 'Bull', 'EarlyBull', 'Neutral', 'Bear'
    """

    green_counts = count_states(green_regimes)
    yellow_counts = count_states(yellow_regimes)
    red_counts = count_states(red_regimes)

    green_total = max(green_counts["Total"], 1)
    yellow_total = max(yellow_counts["Total"], 1)

    green_bullish = green_counts["Bull"] + green_counts["EarlyBull"]
    yellow_bullish = yellow_counts["Bull"] + yellow_counts["EarlyBull"]

    green_bear_pct = green_counts["Bear"] / green_total
    red_bullish = red_counts["Bull"] + red_counts["EarlyBull"]

    # -----------------------
    # BEAR (hard filters)
    # -----------------------
    if gdt_regime == "Bear":
        return "Bear"

    # Broad breakdown in leadership groups
    if green_bear_pct >= 0.50:
        return "Bear"

    # Warning rollover: if ad-tech is Bear AND reds still holding bullish -> late-cycle/bear risk
    if warning_regime == "Bear" and red_bullish >= 1:
        return "Bear"

    # -----------------------
    # CONFIRMED BULL
    # -----------------------
    # GDT not Bear + >=3 green bullish + >=1 yellow bullish + warning not Bear
    if (gdt_regime != "Bear") and (green_bullish >= 3) and (yellow_bullish >= 1) and (warning_regime != "Bear"):
        return "Bull"

    # -----------------------
    # EARLY BULL
    # -----------------------
    if gdt_regime == "Bull" and semi_regime == "Bull":
        return "EarlyBull"

    # Broader EarlyBull: GDT not Bear + >=2 green bullish + warning not Bear
    if (gdt_regime != "Bear") and (green_bullish >= 2) and (warning_regime != "Bear"):
        return "EarlyBull"

    return "Neutral"


def compute_tech_industry_snapshot(
    history_df: pd.DataFrame,
    date: pd.Timestamp,
    regime_col: str = "SubIndustry_Regime"
) -> dict:
    """
    Produces a single row for industry_regime_history.csv
    """

    gdt = get_subindustry_regime_on_date(history_df, date, TECH_CORE, regime_col=regime_col)
    semi = get_subindustry_regime_on_date(history_df, date, "Semiconductors", regime_col=regime_col)
    warning = get_subindustry_regime_on_date(history_df, date, TECH_WARNING, regime_col=regime_col)

    green_regimes = [get_subindustry_regime_on_date(history_df, date, s, regime_col=regime_col) for s in TECH_GREEN]
    yellow_regimes = [get_subindustry_regime_on_date(history_df, date, s, regime_col=regime_col) for s in TECH_YELLOW]
    red_regimes = [get_subindustry_regime_on_date(history_df, date, s, regime_col=regime_col) for s in TECH_RED]

    tech_regime = classify_tech_industry_regime(
        gdt_regime=gdt,
        semi_regime=semi,
        green_regimes=green_regimes,
        yellow_regimes=yellow_regimes,
        red_regimes=red_regimes,
        warning_regime=warning
    )

    green_counts = count_states(green_regimes)
    yellow_counts = count_states(yellow_regimes)
    red_counts = count_states(red_regimes)

    return {
        "Date": date.strftime("%Y-%m-%d"),
        "Industry": "Tech",
        "Tech_Regime": tech_regime,

        # Helpful diagnostics (optional but I recommend keeping)
        "GDT_Regime": gdt,
        "Semis_Regime": semi,
        "Warning_Regime": warning,

        "Green_Bullish": green_counts["Bull"] + green_counts["EarlyBull"],
        "Green_Bear": green_counts["Bear"],
        "Yellow_Bullish": yellow_counts["Bull"] + yellow_counts["EarlyBull"],
        "Red_Bullish": red_counts["Bull"] + red_counts["EarlyBull"],
    }


def compute_subindustry_snapshot(
    date_str: str,
    subindustry_name: str,
    subindustry_group: dict,
    price_data: dict
):
    tickers = list(
        set(subindustry_group.get("core", [])) |
        set(subindustry_group.get("confirmers", []))
    )

    rows = []

    # Normalize snapshot date
    date = pd.to_datetime(date_str).normalize()

    for ticker in tickers:
        df = price_data.get(ticker)

        if df is None or df.empty:
            continue

        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()

        if date not in df.index:
            continue

        df_feat = build_price_features(df)

        row = df_feat.loc[date]

        if pd.isna(row["SMA_20"]):
            continue

        rows.append({
            "Pct_From_SMA_20": row["Pct_From_SMA_20"],
            "Pct_From_SMA_50": row["Pct_From_SMA_50"],
            "New_Low_20D": int(row["New_Low_20D"]),
            "New_Low_50D": int(row["New_Low_50D"]),
            "Higher_High_20D": int(row["Higher_High_20D"]),
            "Higher_High_50D": int(row["Higher_High_50D"]),
        })

    if not rows:
        print(f"[DEBUG] {subindustry_name}: no stocks passed filters on {date_str}")
        return None

    if len(rows) < 3:
        print(
            f"[WARN] {subindustry_name} has too few valid stocks "
            f"({len(rows)}) on {date_str}"
        )
        return None

    agg = pd.DataFrame(rows)

    snapshot = {
        "Date": date.strftime("%Y-%m-%d"),
        "SubIndustry": subindustry_name,

        # Central tendency
        "Median_Pct_From_SMA_20": agg["Pct_From_SMA_20"].median(),
        "Median_Pct_From_SMA_50": agg["Pct_From_SMA_50"].median(),

        # Weakness / exhaustion
        "New_Low_Ratio_20D": agg["New_Low_20D"].mean(),
        "New_Low_Ratio_50D": agg["New_Low_50D"].mean(),

        # Momentum structure
        "Pct_Higher_Highs_20D": agg["Higher_High_20D"].mean(),
        "Pct_Higher_Highs_50D": agg["Higher_High_50D"].mean(),

        # Breadth
        "Pct_Above_SMA_20": (agg["Pct_From_SMA_20"] > 0).mean(),
        "Pct_Above_SMA_50": (agg["Pct_From_SMA_50"] > 0).mean(),

        "Stock_Count": len(agg)
    }

    print(
        f"[DEBUG] {subindustry_name} | {date.strftime('%Y-%m-%d')} | "
        f"Stocks used: {len(agg)} | "
        f"Median SMA20: {snapshot['Median_Pct_From_SMA_20']:.3f}"
    )

    return snapshot


def compute_subindustry_snapshot_history(
    subindustry_name: str,
    subindustry_group: dict,
    price_data: dict,
    min_stocks: int = 3
) -> pd.DataFrame:
    """
    Rebuilds daily sub-industry structural snapshots from fetched price history.
    This keeps rolling regime windows tied to trading days instead of sparse
    script-run dates.
    """
    tickers = sorted(
        set(subindustry_group.get("core", [])) |
        set(subindustry_group.get("confirmers", []))
    )

    rows = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue

        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()

        if "close" not in df.columns:
            continue

        df_feat = build_price_features(df)
        df_feat = df_feat.dropna(subset=["SMA_20"])
        if df_feat.empty:
            continue

        stock_rows = df_feat[[
            "Pct_From_SMA_20",
            "Pct_From_SMA_50",
            "New_Low_20D",
            "New_Low_50D",
            "Higher_High_20D",
            "Higher_High_50D",
        ]].copy()
        stock_rows["Date"] = stock_rows.index
        stock_rows["Ticker"] = ticker
        rows.append(stock_rows)

    if not rows:
        return pd.DataFrame()

    stock_df = pd.concat(rows, ignore_index=True, sort=False)
    grouped = stock_df.groupby("Date")

    snapshot = grouped.agg(
        Median_Pct_From_SMA_20=("Pct_From_SMA_20", "median"),
        Median_Pct_From_SMA_50=("Pct_From_SMA_50", "median"),
        New_Low_Ratio_20D=("New_Low_20D", "mean"),
        New_Low_Ratio_50D=("New_Low_50D", "mean"),
        Pct_Higher_Highs_20D=("Higher_High_20D", "mean"),
        Pct_Higher_Highs_50D=("Higher_High_50D", "mean"),
        Pct_Above_SMA_20=("Pct_From_SMA_20", lambda s: (s > 0).mean()),
        Pct_Above_SMA_50=("Pct_From_SMA_50", lambda s: (s > 0).mean()),
        Stock_Count=("Ticker", "nunique"),
    ).reset_index()

    snapshot = snapshot[snapshot["Stock_Count"] >= min_stocks].copy()
    if snapshot.empty:
        return pd.DataFrame()

    snapshot.insert(1, "SubIndustry", subindustry_name)
    snapshot["Date"] = pd.to_datetime(snapshot["Date"]).dt.strftime("%Y-%m-%d")
    return snapshot


#Subindustry Regime Condition =================
def classify_subindustry_regime(row: pd.Series) -> str:
    """
    Classifies sub-industry cycle regime.
    Returns: 'EarlyBull', 'Bull', 'Neutral', or 'Bear'
    """
    REQUIRED_COLS = [
        "Pct_Above_SMA_20_5D",
        "Pct_Above_SMA_50_5D",
        "New_Low_Ratio_20D_5D",
        "Pct_Higher_Highs_20D_5D",
        "Pct_Higher_Highs_50D_5D",
        "Slope_Median_Pct_From_SMA_20",
    ]
    
    if row[REQUIRED_COLS].isna().any():
        return "Neutral"

    # Thresholds calibrated by regime_threshold_backtest.py expanded 5-year test.
    # --- Confirmed Bull ---
    if (
        row["Pct_Above_SMA_20_5D"] >= 0.65 and
        row["Pct_Above_SMA_50_5D"] >= 0.55 and
        row["Pct_Higher_Highs_50D_5D"] >= 0.30 and
        row["Slope_Median_Pct_From_SMA_20"] >= 0 and
        row["New_Low_Ratio_20D_5D"] <= 0.15
    ):
        return "Bull"
        
    # --- Early Bull (indicator) ---
    if (
        row["Median_Pct_From_SMA_20"] > 0 and
        row["Median_Pct_From_SMA_50"] > 0 and
        row["Pct_Above_SMA_20_5D"] >= 0.55 and
        row["Pct_Higher_Highs_20D_5D"] >= 0.30 and
        row["New_Low_Ratio_20D_5D"] <= 0.25
    ):
        return "EarlyBull"
    # --- Bear / rollover ---
    
    if (
        row["Median_Pct_From_SMA_20"] < 0 and
        row["Pct_Above_SMA_20_5D"] < 0.40 and
        row["Pct_Above_SMA_50_5D"] < 0.35 and
        row["Slope_Median_Pct_From_SMA_20"] < 0 and
        row["New_Low_Ratio_20D_5D"] > 0.40
    ):
        return "Bear"
        
    return "Neutral"
#=======================
def classify_subindustry_stock_flow(
    subindustry_name: str,
    daily_stock_pts: pd.DataFrame,
    min_core_pct: float = 0.30,
    min_confirmer_pct: float = 0.25,
) -> str:
    """
    Determines sub-industry regime using core (leaders) and confirmers.
    """

    group = REGIME_GROUPS[subindustry_name]

    core_stocks = daily_stock_pts[
        daily_stock_pts["Ticker"].isin(group["core"])
    ]

    confirmer_stocks = daily_stock_pts[
        daily_stock_pts["Ticker"].isin(group.get("confirmers", []))
    ]

    if core_stocks.empty:
        return "Neutral"

    # ---- Bull detection ----
    core_bull_pct = (core_stocks["PTS"] >= 0.65).mean()

    confirmer_bull_pct = (
        (confirmer_stocks["PTS"] >= 0.65).mean()
        if not confirmer_stocks.empty
        else 0.0
    )

    # ---- Bear detection (leader failure) ----
    core_bear_pct = (core_stocks["PTS"] <= 0.35).mean()

    confirmer_bear_pct = (
        (confirmer_stocks["PTS"] <= 0.35).mean()
        if not confirmer_stocks.empty
        else 0.0
    )

    # ---- Regime rules (ORDER MATTERS) ----

    # 1️⃣ Distribution / Bear
    if core_bear_pct >= 0.60:
        return "Bear"

    # 2️⃣ Confirmed Bull
    if core_bull_pct >= min_core_pct and confirmer_bull_pct >= min_confirmer_pct:
        return "Bull"

    # 3️⃣ Early Bull (leaders only)
    if core_bull_pct >= min_core_pct:
        return "EarlyBull"
        
    if len(core_stocks) < 3:
        return "Neutral"

    return "Neutral"


def build_subindustry_regime_features(
    daily_stock_pts: pd.DataFrame,
    window: int = 5
) -> pd.DataFrame:
    """
    Builds sub-industry level regime features expected by
    classify_subindustry_regime().
    """

    df = daily_stock_pts.copy()
    df = df.sort_values(["SubIndustry", "Ticker", "Date"])

    # ----------------------------------
    # Binary conditions at stock level
    # ----------------------------------
    if "Pct_From_SMA_20" in df.columns:
        df["Above_SMA20"] = df["Pct_From_SMA_20"] > 0
    else:
        df["Above_SMA20"] = df["MA_Stack"] > 0

    if "Pct_From_SMA_50" in df.columns:
        df["Above_SMA50"] = df["Pct_From_SMA_50"] > 0
    else:
        df["Above_SMA50"] = df["MA_Stack"] > 0

    df["Higher_High_20"] = (
        df["Higher_High_20D"].astype(bool)
        if "Higher_High_20D" in df.columns
        else df.groupby("Ticker")["ND20"].diff() > 0
    )
    df["Higher_High_50"] = (
        df["Higher_High_50D"].astype(bool)
        if "Higher_High_50D" in df.columns
        else df.groupby("Ticker")["ND50"].diff() > 0
    )

    # ----------------------------------
    # Aggregate to sub-industry per date
    # ----------------------------------
    grouped = df.groupby(["Date", "SubIndustry"])

    snapshot = grouped.agg(
        Pct_Above_SMA_20=("Above_SMA20", "mean"),
        Pct_Above_SMA_50=("Above_SMA50", "mean"),
        Pct_Higher_Highs_20D=("Higher_High_20", "mean"),
        Pct_Higher_Highs_50D=("Higher_High_50", "mean"),
    ).reset_index()

    # ----------------------------------
    # Rolling smoothing (5D)
    # ----------------------------------
    snapshot = snapshot.sort_values(["SubIndustry", "Date"])

    for col in [
        "Pct_Above_SMA_20",
        "Pct_Above_SMA_50",
        "Pct_Higher_Highs_20D",
        "Pct_Higher_Highs_50D",
    ]:
        snapshot[f"{col}_5D"] = (
            snapshot
            .groupby("SubIndustry")[col]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # ----------------------------------
    # Median slope of ND20
    # ----------------------------------
    slope_df = (
        df.groupby(["SubIndustry", "Ticker"])
        .apply(
            lambda g: pd.Series({
                "Slope": _slope(g["ND20"].tail(window).values)
            })
        )
        .reset_index()
    )

    slope_snapshot = (
        slope_df.groupby("SubIndustry")["Slope"]
        .median()
        .rename("Slope_Median_Pct_From_SMA_20")
        .reset_index()
    )

    # ----------------------------------
    # Final merge
    # ----------------------------------
    snapshot = snapshot.merge(
        slope_snapshot,
        on="SubIndustry",
        how="left"
    )

    return snapshot
    
    
def combine_subindustry_regimes(
    structural: str,
    flow: str
) -> str:
    """
    Combines structural (breadth / trend) and stock-flow (leaders / confirmers)
    regimes into a single trusted sub-industry regime.
    """
    structural = structural if isinstance(structural, str) and structural else "Neutral"
    flow = flow if isinstance(flow, str) and flow else "Neutral"

    # --------------------------------------------------
    # 1️⃣ BEAR OVERRIDES (risk-first)
    # --------------------------------------------------
    # If either structure OR capital flow says Bear,
    # capital is exiting → protect first
    if structural == "Bear" or flow == "Bear":
        return "Bear"

    # --------------------------------------------------
    # 2️⃣ CONFIRMED BULL (highest conviction)
    # --------------------------------------------------
    # Requires:
    # - Broad participation
    # - Leaders + confirmers working
    if structural == "Bull" and flow == "Bull":
        return "Bull"

    # --------------------------------------------------
    # 3️⃣ EARLY BULL (capital returning)
    # --------------------------------------------------
    # Allows early positioning when:
    # - Structure is improving
    # - Leaders are working
    if structural in ["Bull", "EarlyBull"] and flow in ["Bull", "EarlyBull"]:
        return "EarlyBull"

    # A neutral structural read should not erase clear leader/confirmation
    # strength. Treat flow-only strength as early-cycle until breadth confirms.
    if flow == "Bull" and structural == "Neutral":
        return "EarlyBull"

    if flow == "EarlyBull" and structural == "Neutral":
        return "EarlyBull"

    if structural in ["Bull", "EarlyBull"] and flow == "Neutral":
        return "EarlyBull"

    # --------------------------------------------------
    # 4️⃣ NEUTRAL (default)
    # --------------------------------------------------
    # Choppy, rotational, or inconclusive environments
    return "Neutral"


def get_subindustry_regime_on_date(
    history_df: pd.DataFrame,
    date: pd.Timestamp,
    subindustry: str,
    regime_col: str = "SubIndustry_Regime"
) -> str:
    """
    Returns the subindustry regime at a given date (or 'Neutral' if missing).
    Assumes history_df already canonicalized (no duplicates per Date/SubIndustry).
    """
    subindustry = canonical_subindustry_name(subindustry)
    df = history_df[
        (history_df["SubIndustry"] == subindustry) &
        (history_df["Date"] == date)
    ]
    if df.empty or regime_col not in df.columns:
        return "Neutral"
    val = df.iloc[0][regime_col]
    return val if isinstance(val, str) and val else "Neutral"


def normalize_regime_for_valuation(regime: str) -> str:
    """
    Maps trend regimes into valuation-safe regimes.
    """
    if regime in ("Bull", "EarlyBull"):
        return "Bull"
    if regime == "Bear":
        return "Bear"
    return "Neutral"


def build_valuation_weights(
    subindustry: str,
    metric_cvs: dict[str, float]
) -> dict[str, float]:
    """
    Builds base valuation weights with:
    - base weights
    - sub-industry structural multipliers
    - dispersion (CV) reliability
    """

    weights = {}

    subindustry = canonical_subindustry_name(subindustry)
    sub_mods = SUBINDUSTRY_VALUATION_MULTIPLIERS.get(subindustry, {})

    for metric, base_weight in BASE_VALUATION_WEIGHTS.items():
        structural_mult = sub_mods.get(metric, 1.0)
        dispersion_mult = dispersion_weight_multiplier(
            metric_cvs.get(metric, np.nan)
        )

        w = base_weight * structural_mult * dispersion_mult
        if w > 0:
            weights[metric] = w

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def build_final_valuation_weights(
    subindustry: str,
    metric_cvs: dict[str, float],
    industry_regime: str = "Neutral",
    subindustry_regime: str = "Neutral"
) -> dict[str, float]:
    """
    Builds final valuation weights:
      base -> subindustry structural -> dispersion reliability -> regime overlays
    """

    # Base + structural + dispersion
    base_weights = build_valuation_weights(
        subindustry=subindustry,
        metric_cvs=metric_cvs
    )

    if not base_weights:
        return {}

    # Apply regime overlays (and re-normalize)
    return apply_regime_multipliers(
        weights=base_weights,
        industry_regime=industry_regime,
        subindustry_regime=subindustry_regime
    )



BENCHMARK_META_PATH = Path("benchmark_metadata.json")
BENCHMARK_CENTERS_PATH = Path("benchmark_centers.json")



def get_trend_weight(
    industry_regime: str | None,
    subindustry_regime: str | None
) -> float:
    """ 
    Regime-conditioned weight: how much to trust price/trend vs benchmark/fundamentals.
    Returns trend_weight in [0,1]. benchmark_weight = 1 - trend_weight.
    """

    # 🔒 Defensive normalization (preserves original logic)
    industry = (
        industry_regime
        if isinstance(industry_regime, str)
        else "Neutral"
    ).strip()

    subindustry = (
        subindustry_regime
        if isinstance(subindustry_regime, str)
        else "Neutral"
    ).strip()

    if industry == "EarlyBull":
        industry = "Bull"

    if industry == "Bull" and subindustry == "Bull":
        return 0.60

    if industry == "Bull" and subindustry == "EarlyBull":
        return 0.55

    if industry == "Bull" and subindustry == "Neutral":
        return 0.50

    if industry == "Neutral" and subindustry == "Bull":
        return 0.45

    if industry == "Neutral" and subindustry == "EarlyBull":
        return 0.45

    if industry == "Neutral" and subindustry == "Neutral":
        return 0.40

    if industry == "Bull" and subindustry == "Bear":
        return 0.40

    # Industry Bear (any subindustry)
    return 0.30


def get_regime_multipliers(
    industry_regime: str,
    subindustry_regime: str
) -> dict[str, float]:
    """
    Returns combined regime multipliers for valuation metrics.
    """

    industry_regime = normalize_regime_for_valuation(industry_regime)
    subindustry_regime = normalize_regime_for_valuation(subindustry_regime)

    multipliers = {}

    # Industry regime
    industry_layer = REGIME_VALUATION_MULTIPLIERS.get("Industry", {})
    for metric, m in industry_layer.get(industry_regime, {}).items():
        multipliers[metric] = multipliers.get(metric, 1.0) * m

    # Sub-industry regime
    sub_layer = REGIME_VALUATION_MULTIPLIERS.get("SubIndustry", {})
    for metric, m in sub_layer.get(subindustry_regime, {}).items():
        multipliers[metric] = multipliers.get(metric, 1.0) * m

    return multipliers


def resolve_regime_multiplier(metric: str, regime: str, level: str = "Industry") -> float:
    """
    Returns regime multiplier for a metric.
    level: "Industry" or "SubIndustry"
    regime: "Bull" | "Neutral" | "Bear"
    """
    return (
        REGIME_VALUATION_MULTIPLIERS
        .get(level, {})
        .get(regime, {})
        .get(metric, 1.0)
    )


def apply_regime_multipliers(
    weights: dict[str, float],
    industry_regime: str,
    subindustry_regime: str
) -> dict[str, float]:
    """
    Applies industry + subindustry regime multipliers to weights, then renormalizes.
    """
    adjusted = {}
    for metric, w in weights.items():
        ind_mult = resolve_regime_multiplier(metric, industry_regime, level="Industry")
        sub_mult = resolve_regime_multiplier(metric, subindustry_regime, level="SubIndustry")
        adjusted[metric] = w * ind_mult * sub_mult

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted



#=====================================================================================
#
#
#========(13)===============Composite Scoring, Ranking, Export==============(13)============
# Quick Summary:
# Computes and builds final score logic combining FVS and PTS score (QUANITATIVE SCORE)
# Ranks the scores, highest (1-#) being the highest score out of 100
#------------------------------------------------------------------------------------
#
#--------------------------------Functions List--------------------------------------
# A. compute_combined_score
# B. build_combined_score_dataframe
# C. export_combined_scores_to_excel
# D. open_excel_file
# E. run_full_ranking_pipeline
# F. run_regime_pipeline
# G. run_master_pipeline
#------------------------------------------------------------------------------------
#
#------------------------------------Functions----------------------------------------
def compute_combined_score(
    trend_score: float,
    benchmark_score: float,
    industry_regime: str,
    subindustry_regime: str
) -> float:
    """
    Combines Price-Trend Score (PTS) and Fair Value Score
    using regime-conditioned weighting. PTS is stored internally on a 0-1
    scale, while Fair Value is normalized to 0-100.
    """
    trend_score_100 = price_trend_score_to_100(trend_score)

    # -----------------------------
    # Safety guards
    # -----------------------------
    if not np.isfinite(trend_score_100) or not np.isfinite(benchmark_score):
        return np.nan

    # -----------------------------
    # Regime-based weighting
    # -----------------------------
    trend_weight = get_trend_weight(
        industry_regime=industry_regime,
        subindustry_regime=subindustry_regime
    )

    benchmark_weight = 1.0 - trend_weight

    # -----------------------------
    # Final combined score
    # -----------------------------
    combined = (
        trend_weight * trend_score_100 +
        benchmark_weight * benchmark_score
    )

    return combined


def add_price_trend_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds stock, sub-industry, and industry price-trend context on a 0-100 scale.
    The combined score still uses the stock-level PTS and regime-conditioned
    trend/benchmark weights; these columns make the trend layers auditable.
    """
    if df.empty:
        return df

    out = df.copy()
    source_col = "PTS" if "PTS" in out.columns else "Price_Trend_Score"
    if source_col not in out.columns:
        return out

    out["Stock_Price_Trend_Score"] = out[source_col].map(price_trend_score_to_100)

    sub_group_cols = ["SubIndustry"]
    if "Date" in out.columns:
        sub_group_cols = ["Date", "SubIndustry"]

    if "SubIndustry" in out.columns:
        out["SubIndustry_Price_Trend_Score"] = (
            out.groupby(sub_group_cols)["Stock_Price_Trend_Score"]
            .transform("mean")
        )

    industry_group_cols = []
    if "Date" in out.columns:
        industry_group_cols.append("Date")
    if "Industry" in out.columns:
        industry_group_cols.append("Industry")

    if industry_group_cols:
        out["Industry_Price_Trend_Score"] = (
            out.groupby(industry_group_cols)["Stock_Price_Trend_Score"]
            .transform("mean")
        )
    else:
        out["Industry_Price_Trend_Score"] = out["Stock_Price_Trend_Score"].mean()

    return out


def tier_for_percentile(percentile: float) -> str:
    if not np.isfinite(percentile):
        return "Unrated"
    for tier, cutoff in TIER_PERCENTILE_CUTOFFS:
        if percentile <= cutoff:
            return tier
    return "D"


def add_tier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts by Combined_Score and adds ordered tier-list fields.
    Tiers are relative to the current universe so the workbook always shows
    the best names first even when absolute score levels shift by regime.
    """
    if df.empty:
        return df

    if "Date" in df.columns and df["Date"].nunique(dropna=False) > 1:
        return (
            pd.concat(
                [add_tier_columns(group) for _, group in df.groupby("Date", sort=True, dropna=False)],
                ignore_index=True,
                sort=False
            )
            .reset_index(drop=True)
        )

    out = (
        df.copy()
        .sort_values("Combined_Score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    scores = pd.to_numeric(out.get("Combined_Score"), errors="coerce")
    valid_mask = scores.notna() & np.isfinite(scores)
    valid_count = int(valid_mask.sum())

    out["Overall_Rank"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out["Tier_Percentile"] = np.nan
    out["Tier"] = "Unrated"
    out["Tier_Rank"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    if valid_count == 0:
        return out

    valid_positions = list(np.flatnonzero(valid_mask.to_numpy()))
    ranks = np.arange(1, valid_count + 1)
    percentiles = ranks / valid_count

    out.loc[valid_positions, "Overall_Rank"] = ranks
    out.loc[valid_positions, "Tier_Percentile"] = percentiles
    out.loc[valid_positions, "Tier"] = [
        tier_for_percentile(pct) for pct in percentiles
    ]
    out["Tier_Rank"] = (
        out.groupby("Tier", sort=False).cumcount() + 1
    ).astype("Int64")
    out.loc[out["Tier"] == "Unrated", "Tier_Rank"] = pd.NA

    return out


def ordered_unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def add_valuation_feature_columns(
    feature_df: pd.DataFrame,
    valuation_df: pd.DataFrame | None
) -> pd.DataFrame:
    if valuation_df is None or valuation_df.empty or "Ticker" not in feature_df.columns:
        return feature_df

    out = feature_df.copy()
    for metric in FEATURE_VALUATION_COLUMNS:
        if metric not in out.columns:
            out[metric] = np.nan
        if metric not in valuation_df.index:
            continue

        metric_values = valuation_df.loc[metric].to_dict()
        mapped = out["Ticker"].map(metric_values)
        out[metric] = out[metric].where(out[metric].notna(), mapped)

    return out


def build_feature_table(
    combined_df: pd.DataFrame,
    valuation_df: pd.DataFrame | None = None,
    industry: str = "Tech"
) -> pd.DataFrame:
    """
    Builds the canonical feature table used for calibration/backtesting.
    Existing implemented fields are populated; planned future factors are
    included as blank columns so we can add them one at a time later.
    """
    if combined_df.empty:
        return pd.DataFrame(columns=ordered_unique(FEATURE_TABLE_COLUMNS))

    out = combined_df.copy()

    if "Industry" not in out.columns:
        out["Industry"] = industry

    out = add_price_trend_context_columns(out)
    out = add_tier_columns(out)
    out = add_valuation_feature_columns(out, valuation_df)

    if "Price_Trend_Score_100" not in out.columns:
        source_col = "PTS" if "PTS" in out.columns else "Price_Trend_Score"
        if source_col in out.columns:
            out["Price_Trend_Score_100"] = out[source_col].map(price_trend_score_to_100)

    if "Price_Trend_Score" not in out.columns and "PTS" in out.columns:
        out["Price_Trend_Score"] = out["PTS"]

    if {"Industry_Regime", "SubIndustry_Regime"}.issubset(out.columns):
        out["Trend_Weight"] = out.apply(
            lambda row: get_trend_weight(
                row.get("Industry_Regime"),
                row.get("SubIndustry_Regime")
            ),
            axis=1
        )
    elif "Trend_Weight" not in out.columns:
        out["Trend_Weight"] = np.nan

    out["Benchmark_Weight"] = out["Trend_Weight"].map(
        lambda weight: 1.0 - weight if is_finite_number(weight) else np.nan
    )

    for col in ordered_unique(FEATURE_TABLE_COLUMNS):
        if col not in out.columns:
            out[col] = np.nan

    first_cols = ordered_unique(FEATURE_TABLE_COLUMNS)
    remaining_cols = [col for col in out.columns if col not in first_cols]
    return out[first_cols + remaining_cols]


def save_feature_table_history(
    feature_df: pd.DataFrame,
    output_path: str = FEATURE_TABLE_HISTORY_PATH
):
    if feature_df.empty:
        return
    if not {"Date", "Ticker"}.issubset(feature_df.columns):
        print("[WARN] Feature table history skipped; Date/Ticker columns are required")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    history_df = pd.read_csv(output_path) if os.path.exists(output_path) else pd.DataFrame()
    out = pd.concat([history_df, feature_df], ignore_index=True, sort=False)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = (
        out
        .dropna(subset=["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .sort_values(["Date", "Overall_Rank"], ascending=[True, True], na_position="last")
        .reset_index(drop=True)
    )
    out.to_csv(output_path, index=False)
    print(f"[SUCCESS] Feature table history saved -> {output_path}")


def generate_website_assets_for_latest_output():
    """
    Refreshes the website/demo outputs after the normal ranking pipeline writes
    the latest feature table. This keeps the public-facing table aligned with
    the production-qualified research defaults tested in the backtest harness.
    """
    enabled = os.getenv("STOCK_ANALYSIS_GENERATE_WEBSITE_ASSETS", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        print("[INFO] Website asset refresh skipped by STOCK_ANALYSIS_GENERATE_WEBSITE_ASSETS")
        return

    try:
        from generate_website_assets import generate_website_assets

        metadata = generate_website_assets()
        print(
            "[SUCCESS] Website research assets refreshed -> "
            f"{metadata.get('output_dir', 'website_assets')} "
            f"({metadata.get('eligible_ranked_rows', 0)} eligible names, "
            f"model date {metadata.get('model_date', 'unknown')})"
        )
    except Exception as exc:
        print(f"[WARN] Website asset refresh failed: {exc}")
    
def export_combined_scores_to_excel(
    combined_df: pd.DataFrame,
    output_path: str = "TechIndustryMC.xlsx",
    feature_df: pd.DataFrame | None = None
):
    """
    Outputs:
    - One summary sheet ranked best → worst
    - One tier-list sheet with S/A/B/C/D grouping
    - One sheet per industry
    """

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        tier_list = add_tier_columns(combined_df)
        feature_table = (
            build_feature_table(tier_list)
            if feature_df is None else feature_df
        )

        # --------------------------------------------------
        # SUMMARY SHEET
        # --------------------------------------------------
        summary = tier_list.copy()
        summary["Rank"] = summary["Overall_Rank"]

        summary.to_excel(writer, sheet_name="Summary", index=False)
        tier_list.to_excel(writer, sheet_name="Tier List", index=False)
        feature_table.to_excel(writer, sheet_name="Feature Table", index=False)

        # --------------------------------------------------
        # PER-INDUSTRY SHEETS
        # --------------------------------------------------
        for industry, group in tier_list.groupby("SubIndustry"):
            sheet_name = industry[:31]  # Excel limit
            industry_df = (
                group.sort_values("Combined_Score", ascending=False)
                .reset_index(drop=True)
            )
            industry_df["SubIndustry_Rank"] = industry_df.index + 1

            industry_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Excel file saved: {output_path}")
    
    if AUTO_OPEN_OUTPUT:
        open_excel_file(output_path)


def save_combined_score_history(
    combined_df: pd.DataFrame,
    output_path: str = COMBINED_SCORE_HISTORY_PATH
):
    if combined_df.empty:
        return
    if not {"Date", "Ticker"}.issubset(combined_df.columns):
        print("[WARN] Combined score history skipped; Date/Ticker columns are required")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    history_df = pd.read_csv(output_path) if os.path.exists(output_path) else pd.DataFrame()
    out = pd.concat([history_df, combined_df], ignore_index=True, sort=False)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = (
        out
        .dropna(subset=["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .sort_values(["Date", "Combined_Score"], ascending=[True, False], na_position="last")
        .reset_index(drop=True)
    )
    out.to_csv(output_path, index=False)
    print(f"[SUCCESS] Combined score history saved -> {output_path}")


def build_combined_score_dataframe(
    valuation_df: pd.DataFrame,
    fair_value_scores: dict,
    price_trend_scores: dict,
    ticker_to_subindustry: dict,
    industry_regime: str,
    subindustry_regimes: dict,
    centers: dict | None = None,
    asof_date: pd.Timestamp | str | None = None,
    market_data: dict | None = None
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by ticker with all core scores.
    """

    rows = []
    date_str = (
        normalize_asof_date(asof_date).strftime("%Y-%m-%d")
        if asof_date is not None else ""
    )
    centers = centers or {}
    market_data = market_data or {}

    all_tickers = sorted(
        set(ticker_to_subindustry.keys())
        | set(valuation_df.columns)
        | set(price_trend_scores.keys())
    )

    for ticker in all_tickers:
        sub = ticker_to_subindustry.get(ticker)
        if not sub:
            continue

        fv = fair_value_scores.get(ticker, np.nan)
        pts = price_trend_scores.get(ticker, np.nan)
        subindustry_regime = subindustry_regimes.get(sub, "Neutral")
        fair_value_diagnostic = diagnose_fair_value_score(
            ticker=ticker,
            subindustry=sub,
            fair_value_score=fv,
            valuation_df=valuation_df,
            centers=centers,
            industry_regime=industry_regime,
            subindustry_regime=subindustry_regime
        )
        market_snapshot = market_data.get(ticker, {})
        share_count_detail = share_count_source_detail(ticker, valuation_df)
        valuation_source = valuation_data_source_detail(ticker)

        combined = compute_combined_score(
            trend_score=pts,
            benchmark_score=fv,
            industry_regime=industry_regime,
            subindustry_regime=subindustry_regime
        )

        rows.append({
            "Date": date_str,
            "Ticker": ticker,
            "SubIndustry": sub,
            "Fair_Value_Score": fv,
            **fair_value_diagnostic,
            "Price_Trend_Score": pts,
            "Market_Price_USD": market_snapshot.get("price", np.nan),
            "Market_Price_Date": format_market_price_date(market_snapshot),
            "Market_Data_Source": market_snapshot.get("source", "missing"),
            **valuation_source,
            **share_count_detail,
            "Combined_Score": combined
        })

    df = pd.DataFrame(rows)
    df = add_price_trend_context_columns(df)
    df = add_tier_columns(df)

    return df


def open_excel_file(path: str):
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", path], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(path)
        else:  # Linux
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        print(f"[WARN] Could not open Excel file: {e}")


def run_full_ranking_pipeline(
    tickers: list[str],
    ticker_to_subindustry: dict,
    price_data: dict,
    asof_date: pd.Timestamp,
    industry_regime: str,
    subindustry_regimes: dict
):
    reset_fair_value_issue_log()
    market_data = build_market_data_map(
        tickers=tickers,
        asof_date=asof_date,
        price_data=price_data
    )

    # 1) Valuation
    valuation_df = build_valuation_dataframe_for_universe(
        tickers=tickers,
        asof_date=asof_date,
        price_data=price_data,
        market_data=market_data
    )

    # 2) Benchmarks
    centers = rebuild_benchmarks_if_needed(
        valuation_df=valuation_df,
        valuation_metrics=BASE_VALUATION_WEIGHTS
    )

    # 3) Fair Value
    fair_value_scores = build_fair_value_scores_for_universe(
        valuation_df=valuation_df,
        centers=centers,
        ticker_to_subindustry=ticker_to_subindustry,
        industry_regime_by_subindustry={
            sub: industry_regime for sub in set(ticker_to_subindustry.values())
        },
        subindustry_regime_map=subindustry_regimes
    )

    # 4) Price Trend
    daily_pts = build_daily_stock_pts(
        price_data=price_data,
        asof_date=asof_date,
        ticker_to_subindustry=ticker_to_subindustry,
        fair_value_scores=fair_value_scores
    )

    price_trend_scores = (
        daily_pts
        .set_index("Ticker")["PTS"]
        .to_dict()
    )

    # 5) Combined Scores
    combined_df = build_combined_score_dataframe(
        valuation_df=valuation_df,
        fair_value_scores=fair_value_scores,
        price_trend_scores=price_trend_scores,
        ticker_to_subindustry=ticker_to_subindustry,
        industry_regime=industry_regime,
        subindustry_regimes=subindustry_regimes,
        centers=centers,
        asof_date=asof_date,
        market_data=market_data
    )

    # 6) Export
    feature_df = build_feature_table(combined_df, valuation_df=valuation_df)
    export_combined_scores_to_excel(combined_df, feature_df=feature_df)
    save_combined_score_history(combined_df)
    save_feature_table_history(feature_df)
    generate_website_assets_for_latest_output()

    return combined_df
    

def run_master_pipeline():
    print("\n==============================")
    print(" RUNNING MASTER PIPELINE")
    print("==============================\n")

    # --------------------------------------------------
    # DEBUG: Raw SEC export workbook (ONE per run)
    # --------------------------------------------------
    DEBUG_SEC_PATH = "debug_sec_exports/SEC_raw_debug.xlsx"
    os.makedirs(os.path.dirname(DEBUG_SEC_PATH), exist_ok=True)

    debug_sec_writer = pd.ExcelWriter(
        DEBUG_SEC_PATH,
        engine="xlsxwriter"
    )

    try:
        # ----------------------------------
        # 1) Run REGIME + PRICE TREND PIPELINE
        # ----------------------------------
        regime_results = run_regime_pipeline()

        daily_stock_pts = regime_results["daily_stock_pts"]
        industry_regime = regime_results["industry_regime"]
        subindustry_regimes = regime_results["subindustry_regimes"]
        asof_date = regime_results["asof_date"]
        price_data = regime_results["price_data"]

        if daily_stock_pts.empty:
            raise RuntimeError("Regime pipeline returned no stock data")

        print(f"[INFO] Loaded stock PTS for {asof_date.date()}")

        # ----------------------------------
        # 2) Build ticker universe + mappings
        # ----------------------------------
        tickers = sorted(daily_stock_pts["Ticker"].unique())
        market_data = build_market_data_map(
            tickers=tickers,
            asof_date=asof_date,
            price_data=price_data
        )

        ticker_to_subindustry = (
            daily_stock_pts
            .set_index("Ticker")["SubIndustry"]
            .to_dict()
        )

        # ----------------------------------
        # 3) RUN VALUATION PIPELINE
        # ----------------------------------
        reset_fair_value_issue_log()
        valuation_df = build_valuation_dataframe(
            tickers=tickers,
            asof_date=asof_date,
            debug_sec_writer=debug_sec_writer,
            price_data=price_data,
            market_data=market_data
        )

        centers = build_benchmark_centers(
            valuation_df=valuation_df,
            valuation_metrics=BASE_VALUATION_WEIGHTS
        )

        fair_value_scores = build_fair_value_scores(
            valuation_df=valuation_df,
            ticker_to_subindustry=ticker_to_subindustry,
            centers=centers,
            industry_regime_by_subindustry={
                sub: industry_regime for sub in subindustry_regimes.keys()
            },
            subindustry_regime_map=subindustry_regimes
        )

        # ----------------------------------
        # 4) MERGE ALL SCORES
        # ----------------------------------
        combined_df = daily_stock_pts.copy()

        combined_df["Fair_Value_Score"] = combined_df["Ticker"].map(fair_value_scores)
        fair_value_diagnostics = combined_df.apply(
            lambda r: diagnose_fair_value_score(
                ticker=r["Ticker"],
                subindustry=r["SubIndustry"],
                fair_value_score=r["Fair_Value_Score"],
                valuation_df=valuation_df,
                centers=centers,
                industry_regime=r["Industry_Regime"],
                subindustry_regime=r["SubIndustry_Regime"]
            ),
            axis=1
        )
        fair_value_diagnostics_df = pd.DataFrame(
            list(fair_value_diagnostics),
            index=combined_df.index
        )
        for col in FEATURE_VALUATION_SCORE_COLUMNS:
            if col == "Fair_Value_Score":
                continue
            combined_df[col] = fair_value_diagnostics_df.get(col, "")

        combined_df["Market_Price_USD"] = combined_df["Ticker"].map(
            lambda t: market_data.get(t, {}).get("price", np.nan)
        )
        combined_df["Market_Price_Date"] = combined_df["Ticker"].map(
            lambda t: format_market_price_date(market_data.get(t, {}))
        )
        combined_df["Market_Data_Source"] = combined_df["Ticker"].map(
            lambda t: market_data.get(t, {}).get("source", "missing")
        )
        valuation_source_details = combined_df["Ticker"].map(valuation_data_source_detail)
        valuation_source_details_df = pd.DataFrame(
            list(valuation_source_details),
            index=combined_df.index
        )
        combined_df["Valuation_Data_Source"] = valuation_source_details_df.get(
            "Valuation_Data_Source",
            "unknown"
        )
        combined_df["Valuation_Data_Source_Detail"] = valuation_source_details_df.get(
            "Valuation_Data_Source_Detail",
            ""
        )
        share_count_details = combined_df["Ticker"].map(
            lambda t: share_count_source_detail(t, valuation_df)
        )
        share_count_details_df = pd.DataFrame(
            list(share_count_details),
            index=combined_df.index
        )
        for col in FEATURE_SHARE_COUNT_COLUMNS:
            combined_df[col] = share_count_details_df.get(col, np.nan)

        combined_df["Combined_Score"] = combined_df.apply(
            lambda r: compute_combined_score(
                trend_score=r["PTS"],
                benchmark_score=r["Fair_Value_Score"],
                industry_regime=r["Industry_Regime"],
                subindustry_regime=r["SubIndustry_Regime"]
            ),
            axis=1
        )

        combined_df = add_price_trend_context_columns(combined_df)

        # ----------------------------------
        # 5) SORT + EXPORT
        # ----------------------------------
        combined_df = add_tier_columns(combined_df)
        feature_df = build_feature_table(combined_df, valuation_df=valuation_df)

        export_combined_scores_to_excel(
            combined_df=combined_df,
            output_path="Tech_Combined_Rankings.xlsx",
            feature_df=feature_df
        )

        save_combined_score_history(combined_df)
        save_feature_table_history(feature_df)
        generate_website_assets_for_latest_output()

        print("\n=== MASTER PIPELINE COMPLETE ===")

    finally:
        # --------------------------------------------------
        # 🔒 GUARANTEED writer close (even on error)
        # --------------------------------------------------
        debug_sec_writer.close()
        print("✅ Raw SEC debug workbook saved →", DEBUG_SEC_PATH)

def run_regime_pipeline():
    """
    Runs the Market / Industry / Sub-Industry regime detection pipeline
    and writes canonical regime history CSVs.
    """

    import os
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    os.makedirs("data", exist_ok=True)

    SUBIND_HISTORY_PATH = "data/subindustry_regime_history.csv"
    IND_HISTORY_PATH = "data/industry_regime_history.csv"
    STOCK_PTS_PATH = "data/stock_price_trend_history.csv"

    # --------------------------------------------------
    # Collect all tickers
    # --------------------------------------------------
    all_regime_tickers = flatten_ticker_groups(REGIME_GROUPS)

    print("\n=== SANITY CHECK: REGIME TICKERS ===")
    print(f"Total regime tickers: {len(all_regime_tickers)}")
    print("===================================\n")

    # --------------------------------------------------
    # Fetch price data
    # --------------------------------------------------
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")

    price_data = fetch_all_prices(
        tickers=all_regime_tickers,
        start_date=start_date,
        end_date=end_date
    )

    if not price_data:
        print("[ERROR] No price data fetched — aborting")
        return

    last_trading_date = max(df.index.max() for df in price_data.values())
    date_str = last_trading_date.strftime("%Y-%m-%d")

    print(f"[DEBUG] Snapshot date: {date_str}")

    # ==================================================
    # ============ SUB-INDUSTRY PIPELINE ================
    # ==================================================

    # --------------------------------------------------
    # Load sub-industry history
    # --------------------------------------------------
    if os.path.exists(SUBIND_HISTORY_PATH):
        history_df = pd.read_csv(SUBIND_HISTORY_PATH)
    else:
        history_df = pd.DataFrame()

    if not history_df.empty and "SubIndustry" in history_df.columns:
        history_df["SubIndustry"] = history_df["SubIndustry"].apply(canonical_subindustry_name)

    # --------------------------------------------------
    # Rebuild recent daily structural snapshots from price history
    # --------------------------------------------------
    snapshots = []

    for subindustry_name, group in REGIME_GROUPS.items():
        subindustry_name = canonical_subindustry_name(subindustry_name)
        snap_history = compute_subindustry_snapshot_history(
            subindustry_name=subindustry_name,
            subindustry_group=group,
            price_data=price_data
        )

        if not snap_history.empty:
            snapshots.append(snap_history)
            continue

        snap = compute_subindustry_snapshot(
            date_str=date_str,
            subindustry_name=subindustry_name,
            subindustry_group=group,
            price_data=price_data
        )
        if snap is not None:
            snapshots.append(pd.DataFrame([snap]))

    if snapshots:
        history_df = pd.concat(
            [history_df, pd.concat(snapshots, ignore_index=True, sort=False)],
            ignore_index=True
        )

    # --------------------------------------------------
    # Canonicalize
    # --------------------------------------------------
    if history_df.empty:
        raise RuntimeError("No sub-industry history rows available after snapshot build")

    history_df["SubIndustry"] = history_df["SubIndustry"].apply(canonical_subindustry_name)
    history_df["Date"] = pd.to_datetime(history_df["Date"])

    history_df = (
        history_df
        .sort_values(["SubIndustry", "Date"])
        .drop_duplicates(["Date", "SubIndustry"], keep="last")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Rolling persistence metrics
    # --------------------------------------------------
    ROLL = 5

    history_df["Pct_Above_SMA_20_5D"] = (
        history_df.groupby("SubIndustry")["Pct_Above_SMA_20"]
        .rolling(ROLL, min_periods=ROLL).mean()
        .reset_index(level=0, drop=True)
    )

    history_df["Pct_Above_SMA_50_5D"] = (
        history_df.groupby("SubIndustry")["Pct_Above_SMA_50"]
        .rolling(ROLL, min_periods=ROLL).mean()
        .reset_index(level=0, drop=True)
    )

    history_df["New_Low_Ratio_20D_5D"] = (
        history_df.groupby("SubIndustry")["New_Low_Ratio_20D"]
        .rolling(ROLL, min_periods=ROLL).mean()
        .reset_index(level=0, drop=True)
    )

    history_df["Pct_Higher_Highs_20D_5D"] = (
        history_df.groupby("SubIndustry")["Pct_Higher_Highs_20D"]
        .rolling(ROLL, min_periods=ROLL).mean()
        .reset_index(level=0, drop=True)
    )

    history_df["Pct_Higher_Highs_50D_5D"] = (
        history_df.groupby("SubIndustry")["Pct_Higher_Highs_50D"]
        .rolling(ROLL, min_periods=ROLL).mean()
        .reset_index(level=0, drop=True)
    )

    history_df["Slope_Median_Pct_From_SMA_20"] = (
        history_df.groupby("SubIndustry")["Median_Pct_From_SMA_20"]
        .diff(ROLL)
    )

    # --------------------------------------------------
    # Structural sub-industry regime (RAW)
    # --------------------------------------------------
    history_df["Structural_Regime"] = history_df.apply(
        classify_subindustry_regime,
        axis=1
    )

    # --------------------------------------------------
    # Structural persistence (KEY FIX ALREADY ADDED)
    # --------------------------------------------------
    history_df["Structural_Regime_Persist"] = (
        history_df
        .groupby("SubIndustry")["Structural_Regime"]
        .apply(apply_regime_persistence)
        .reset_index(level=0, drop=True)
    )

    # --------------------------------------------------
    # Stock-flow regime prep
    # --------------------------------------------------
    ticker_to_subindustry = {}
    for subindustry, group in REGIME_GROUPS.items():
        for t in group.get("core", []):
            ticker_to_subindustry[t] = subindustry
        for t in group.get("confirmers", []):
            ticker_to_subindustry[t] = subindustry

    daily_stock_pts = build_daily_stock_pts(
        price_data=price_data,
        asof_date=last_trading_date,
        ticker_to_subindustry=ticker_to_subindustry
    )

    if daily_stock_pts.empty:
        raise RuntimeError("No daily stock price-trend rows were built")

    # --------------------------------------------------
    # Stock-flow regime for the current date
    # --------------------------------------------------
    if "StockFlow_Regime" not in history_df.columns:
        history_df["StockFlow_Regime"] = "Neutral"

    stock_flow_today = {
        subindustry: classify_subindustry_stock_flow(
            subindustry_name=subindustry,
            daily_stock_pts=daily_stock_pts
        )
        for subindustry in REGIME_GROUPS.keys()
    }

    today_mask = history_df["Date"] == last_trading_date
    history_df.loc[today_mask, "StockFlow_Regime"] = (
        history_df.loc[today_mask, "SubIndustry"].map(stock_flow_today).fillna("Neutral")
    )
    history_df["StockFlow_Regime"] = history_df["StockFlow_Regime"].fillna("Neutral")

    # --------------------------------------------------
    # FINAL sub-industry regime
    # --------------------------------------------------
    history_df["SubIndustry_Regime"] = history_df.apply(
        lambda r: combine_subindustry_regimes(
            r["Structural_Regime_Persist"],
            r["StockFlow_Regime"]
        ),
        axis=1
    )

    # --------------------------------------------------
    # Inject fresh current-date regimes into stock-level table
    # --------------------------------------------------
    regime_lookup = history_df[
        history_df["Date"] == last_trading_date
    ][[
        "SubIndustry",
        "SubIndustry_Regime",
        "Structural_Regime_Persist",
        "StockFlow_Regime"
    ]].drop_duplicates("SubIndustry")

    daily_stock_pts = daily_stock_pts.merge(
        regime_lookup,
        on="SubIndustry",
        how="left"
    )
    daily_stock_pts["SubIndustry_Regime"] = daily_stock_pts["SubIndustry_Regime"].fillna("Neutral")
    daily_stock_pts["Structural_Regime_Persist"] = daily_stock_pts["Structural_Regime_Persist"].fillna("Neutral")
    daily_stock_pts["StockFlow_Regime"] = daily_stock_pts["StockFlow_Regime"].fillna("Neutral")

    # --------------------------------------------------
    # Save sub-industry history
    # --------------------------------------------------
    history_df.to_csv(SUBIND_HISTORY_PATH, index=False)
    print(f"[SUCCESS] Sub-industry history saved → {SUBIND_HISTORY_PATH}")

    # ==================================================
    # ================ INDUSTRY PIPELINE ===============
    # ==================================================
    industry_rows = []

    for date, _ in history_df.groupby("Date"):
        row = compute_tech_industry_snapshot(
            history_df=history_df,
            date=date
        )
        row["Date"] = date
        industry_rows.append(row)

    industry_df = pd.DataFrame(industry_rows)

    industry_df = (
        industry_df
        .sort_values(["Industry", "Date"])
        .drop_duplicates(["Industry", "Date"], keep="last")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Industry persistence
    # --------------------------------------------------
    industry_df["Industry_Regime_Persist"] = (
        industry_df
        .groupby("Industry")["Tech_Regime"]
        .apply(apply_regime_persistence)
        .reset_index(level=0, drop=True)
    )

    industry_df.to_csv(IND_HISTORY_PATH, index=False)
    print(f"[SUCCESS] Industry history saved → {IND_HISTORY_PATH}")

    # --------------------------------------------------
    # Attach industry regime to stocks
    # --------------------------------------------------
    industry_regime_today = industry_df.loc[
        industry_df["Date"] == last_trading_date,
        "Industry_Regime_Persist"
    ]

    industry_regime_today = (
        industry_regime_today.iloc[0]
        if not industry_regime_today.empty
        else "Neutral"
    )

    daily_stock_pts["Industry_Regime"] = industry_regime_today


    # --------------------------------------------------
    # Save daily stock PTS
    # --------------------------------------------------
    if os.path.exists(STOCK_PTS_PATH):
        stock_hist = pd.read_csv(STOCK_PTS_PATH)
    else:
        stock_hist = pd.DataFrame()

    if not stock_hist.empty and "Date" in stock_hist.columns:
        stock_hist = stock_hist[stock_hist["Date"].astype(str) != date_str]

    stock_hist = pd.concat([stock_hist, daily_stock_pts], ignore_index=True, sort=False)
    stock_hist = stock_hist.drop_duplicates(["Date", "Ticker"], keep="last")
    stock_hist.to_csv(STOCK_PTS_PATH, index=False)
    print(f"[SUCCESS] Stock price trend history saved → {STOCK_PTS_PATH}")

    print("\n=== PIPELINE COMPLETE ===")
    return {
    "daily_stock_pts": daily_stock_pts,
    "industry_regime": industry_regime_today,
    "subindustry_regimes": (
        history_df.loc[history_df["Date"] == last_trading_date]
        .set_index("SubIndustry")["SubIndustry_Regime"]
        .to_dict()
    ),
    "asof_date": last_trading_date,
    "price_data": price_data
}


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_master_pipeline()
