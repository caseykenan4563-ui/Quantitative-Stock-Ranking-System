#!/usr/bin/env python3
"""Generate website-facing Stock Analysis artifacts from the latest model state.

The website output intentionally uses the current production-qualified research
defaults:

- rank by Alpha_Prototype_XLKCompetitive
- require at least $30M 20-day average dollar volume
- show the top 100 ranked names and top 10 rank-weighted paper portfolio

This script does not place trades. It writes explanatory website assets only.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import alpha_signal_quality
import historical_data_layer as historical_data
import monthly_rebalanced_portfolio_simulator as monthly_sim


DEFAULT_FEATURE_HISTORY = Path("data/stock_feature_table_history.csv")
DEFAULT_SCORE_HISTORY = Path("backtests/combined_score_sharadar/point_in_time_scores.csv")
DEFAULT_PRICE_FILE = Path("data/historical_data_layer/prices_daily.csv")
DEFAULT_HISTORICAL_UNIVERSE = Path("data/historical_tech_universe.csv")
DEFAULT_DECISION_SELECTIONS = Path(
    "backtests/production_default_decision_sharadar/production_default_selected_constructions.csv"
)
DEFAULT_OUTPUT_DIR = Path("website_assets")

PRODUCTION_SCORE_COLUMN = "Alpha_Prototype_XLKCompetitive"
PRODUCTION_CONFIG_ID = "alpha_Alpha_Prototype_XLKCompetitive"
PRODUCTION_SCORE_LABEL = "XLKCompetitive Alpha Score"
PRODUCTION_MIN_DOLLAR_VOLUME = 30_000_000.0
PRODUCTION_TOP_N = 10
PRODUCTION_PORTFOLIO_MODE = "rank_weighted"

CHART_WIDTH = 1120
CHART_HEIGHT = 560
COLORS = {
    "strategy": "#0f6b63",
    "universe": "#536878",
    "qqq": "#375a9e",
    "xlk": "#9b5a2e",
    "accent": "#0f6b63",
    "warning": "#b45309",
    "line": "#d6dee4",
    "text": "#172026",
    "muted": "#64717b",
    "bg": "#ffffff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate website-facing Stock Analysis outputs.")
    parser.add_argument("--feature-history", type=Path, default=DEFAULT_FEATURE_HISTORY)
    parser.add_argument("--score-history", type=Path, default=DEFAULT_SCORE_HISTORY)
    parser.add_argument("--historical-price-file", type=Path, default=DEFAULT_PRICE_FILE)
    parser.add_argument("--historical-universe", type=Path, default=DEFAULT_HISTORICAL_UNIVERSE)
    parser.add_argument("--decision-selections", type=Path, default=DEFAULT_DECISION_SELECTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-dollar-volume", type=float, default=PRODUCTION_MIN_DOLLAR_VOLUME)
    parser.add_argument("--top-n", type=int, default=PRODUCTION_TOP_N)
    parser.add_argument("--portfolio-mode", default=PRODUCTION_PORTFOLIO_MODE)
    parser.add_argument("--skip-backtest", action="store_true")
    return parser.parse_args()


def format_date(value: object) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(dt) else dt.strftime("%Y-%m-%d")


def format_number(value: object, digits: int = 2) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(num):
        return ""
    return f"{num:,.{digits}f}"


def format_dollars(value: object) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(num):
        return ""
    if abs(num) >= 1_000_000_000:
        return f"${num / 1_000_000_000:,.2f}B"
    if abs(num) >= 1_000_000:
        return f"${num / 1_000_000:,.1f}M"
    return f"${num:,.0f}"


def format_pct(value: object, digits: int = 2) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(num):
        return ""
    return f"{num * 100:,.{digits}f}%"


def production_tier(rank: int, count: int) -> str:
    if count <= 0:
        return "Unrated"
    pct = rank / count
    if pct <= 0.10:
        return "S"
    if pct <= 0.25:
        return "A"
    if pct <= 0.50:
        return "B"
    if pct <= 0.75:
        return "C"
    return "D"


def load_latest_feature_rows(path: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    if not path.exists():
        raise FileNotFoundError(f"Feature history file not found: {path}")
    frame = pd.read_csv(path)
    required = {"Date", "Ticker", "SubIndustry", "Combined_Score", "Fair_Value_Score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Feature history missing required columns: {missing}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["Ticker"] = frame["Ticker"].astype(str).str.upper().str.strip()
    frame = frame.dropna(subset=["Date", "Ticker"]).copy()
    latest_date = frame["Date"].max()
    latest = frame[frame["Date"] == latest_date].copy()
    latest = latest.drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)
    return latest, pd.to_datetime(latest_date).normalize()


def nearest_on_or_before(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    clean = pd.to_datetime(index).normalize()
    loc = clean.searchsorted(pd.to_datetime(date).normalize(), side="right") - 1
    if loc < 0:
        return None
    return pd.to_datetime(clean[loc]).normalize()


def load_price_and_volume(
    price_file: Path,
    tickers: Iterable[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
    price_wide = historical_data.load_canonical_price_wide(
        path=price_file,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        value_column="Close",
    )
    volume_wide = historical_data.load_canonical_price_wide(
        path=price_file,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        value_column="Volume",
    )
    if price_wide.empty:
        raise RuntimeError("No canonical price data available for website asset generation")
    return price_wide, volume_wide


def enriched_latest_ranking(
    latest: pd.DataFrame,
    latest_date: pd.Timestamp,
    price_file: Path,
    min_dollar_volume: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    tickers = sorted(set(latest["Ticker"]) | {"QQQ", "XLK"})
    start_date = latest_date - pd.Timedelta(days=260)
    price_wide, volume_wide = load_price_and_volume(price_file, tickers, start_date=start_date)
    enriched = alpha_signal_quality.attach_ranked_signals(latest, price_wide)
    if PRODUCTION_SCORE_COLUMN not in enriched.columns:
        raise RuntimeError(f"{PRODUCTION_SCORE_COLUMN} was not generated")

    adv20 = monthly_sim.rolling_dollar_volume(price_wide, volume_wide, window=20)
    adv_date = nearest_on_or_before(adv20.index, latest_date)
    if adv_date is None:
        enriched["ADV20_Dollar"] = np.nan
    else:
        enriched["ADV20_Dollar"] = enriched["Ticker"].map(adv20.loc[adv_date])

    enriched["Production_Score"] = pd.to_numeric(enriched[PRODUCTION_SCORE_COLUMN], errors="coerce")
    enriched["Production_Eligible"] = (
        pd.to_numeric(enriched["ADV20_Dollar"], errors="coerce").ge(min_dollar_volume)
        & np.isfinite(enriched["Production_Score"])
    )
    ranked = (
        enriched[enriched["Production_Eligible"]]
        .sort_values("Production_Score", ascending=False)
        .reset_index(drop=True)
    )
    ranked["Production_Rank"] = np.arange(1, len(ranked) + 1)
    ranked["Production_Tier"] = [
        production_tier(int(rank), len(ranked)) for rank in ranked["Production_Rank"]
    ]
    ranked["Production_Tier_Rank"] = ranked.groupby("Production_Tier").cumcount() + 1
    ranked["Liquidity_Default_USD"] = min_dollar_volume
    ranked["Production_Score_Column"] = PRODUCTION_SCORE_COLUMN
    ranked["Model_Date"] = latest_date.strftime("%Y-%m-%d")
    ranked["ADV20_Date"] = format_date(adv_date)
    return ranked, price_wide, adv20, adv_date


def html_escape(value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return html.escape(str(value))


def top_table_columns() -> list[tuple[str, str, str]]:
    return [
        ("Production_Rank", "Rank", "int"),
        ("Ticker", "Ticker", "text"),
        ("SubIndustry", "Subindustry", "text"),
        ("Production_Tier", "Tier", "text"),
        ("Production_Score", "Production Alpha", "num"),
        ("Combined_Score", "Original Combined", "num"),
        ("Stock_Price_Trend_Score", "Trend", "num"),
        ("Fair_Value_Score", "Fair Value", "num"),
        ("ADV20_Dollar", "20D Dollar Volume", "dollar"),
        ("Share_Count_Source_Category", "Share Count Source", "text"),
        ("SubIndustry_Regime", "Subindustry Regime", "text"),
        ("Industry_Regime", "Industry Regime", "text"),
    ]


def render_rank_table_rows(frame: pd.DataFrame, limit: int) -> str:
    rows = []
    for _, row in frame.head(limit).iterrows():
        cells = []
        for column, _, kind in top_table_columns():
            value = row.get(column, "")
            if kind == "int":
                text = str(int(value)) if pd.notna(value) and np.isfinite(float(value)) else ""
            elif kind == "num":
                text = format_number(value, 2)
            elif kind == "dollar":
                text = format_dollars(value)
            else:
                text = html_escape(value)
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def write_top_table_html(ranked: pd.DataFrame, output_dir: Path, generated_at: str) -> None:
    headers = "".join(f"<th>{html_escape(label)}</th>" for _, label, _ in top_table_columns())
    rows = render_rank_table_rows(ranked, 100)
    model_date = html_escape(ranked["Model_Date"].iloc[0] if not ranked.empty else "")
    eligible_count = len(ranked)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Analysis Production Ranking</title>
<style>
:root {{ color-scheme: light; --ink:#172026; --muted:#64717b; --line:#d9e0e5; --head:#eef3f7; --row:#ffffff; --alt:#f8fafb; --accent:#0f6b63; --warn:#7c2d12; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; color:var(--ink); background:#f3f6f8; }}
.wrap {{ max-width:1240px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 8px; font-size:24px; line-height:1.2; font-weight:750; }}
.meta {{ margin:0 0 10px; color:var(--muted); font-size:14px; line-height:1.45; }}
.disclaimer {{ margin:16px 0; padding:12px 14px; border-left:4px solid var(--warn); background:#fff7ed; color:#4a2211; font-size:13px; line-height:1.45; }}
.table-wrap {{ max-height:760px; overflow:auto; border:1px solid var(--line); background:white; }}
table {{ width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }}
th, td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ position:sticky; top:0; z-index:1; background:var(--head); font-size:12px; text-transform:uppercase; letter-spacing:0; color:#33424c; }}
tr:nth-child(even) td {{ background:var(--alt); }}
td:nth-child(1), td:nth-child(5), td:nth-child(6), td:nth-child(7), td:nth-child(8), td:nth-child(9) {{ text-align:right; font-variant-numeric:tabular-nums; }}
td:nth-child(4) {{ font-weight:750; color:var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
<h1>Technology Stock Ranking Top 100</h1>
<p class="meta">Model date: {model_date}. Ranking uses {html_escape(PRODUCTION_SCORE_LABEL)} with a {format_dollars(PRODUCTION_MIN_DOLLAR_VOLUME)} minimum 20-day dollar-volume filter. Eligible names: {eligible_count}.</p>
<p class="meta">Production alpha blend: 30% confidence-adjusted fair value, 45% subindustry-relative mid/long momentum, 15% risk-adjusted momentum, and 10% downside quality.</p>
<div class="disclaimer"><strong>Research only.</strong> This output is not financial advice, not a recommendation to buy or sell securities, and not an automated trading system. It is a personal quantitative research screen that still requires human review, forward paper testing, and risk controls before any real allocation decision.</div>
<div class="table-wrap">
<table>
<thead><tr>{headers}</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>
<p class="meta">Generated: {html_escape(generated_at)}. Source files: latest feature history, Sharadar canonical prices, and production-default decision backtests.</p>
</div>
</body>
</html>
"""
    (output_dir / "stock-ranking-top-100-scrollable.html").write_text(html_text, encoding="utf-8")


def save_top_ranking_outputs(ranked: pd.DataFrame, output_dir: Path, generated_at: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_dir / "production-default-top-100.csv", index=False)
    write_top_table_html(ranked, output_dir, generated_at)


def svg_text(x: float, y: float, text: object, size: int = 13, fill: str = COLORS["text"], anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
        f"{html_escape(text)}</text>"
    )


def polyline(points: list[tuple[float, float]], color: str, width: float = 2.4) -> str:
    if not points:
        return ""
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{width:.1f}" stroke-linejoin="round" stroke-linecap="round"/>'


def write_line_chart_svg(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, str]],
    title: str,
    y_label: str,
    output_path: Path,
    as_percent: bool = False,
) -> None:
    if frame.empty:
        output_path.write_text("<svg></svg>\n", encoding="utf-8")
        return
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")
    values = []
    for col, _, _ in columns:
        values.extend(pd.to_numeric(data[col], errors="coerce").dropna().tolist())
    if not values:
        output_path.write_text("<svg></svg>\n", encoding="utf-8")
        return

    width, height = CHART_WIDTH, CHART_HEIGHT
    left, right, top, bottom = 76, 34, 68, 64
    x0, x1 = left, width - right
    y0, y1 = height - bottom, top
    d0, d1 = data["Date"].min(), data["Date"].max()
    t0, t1 = d0.value, d1.value
    if t0 == t1:
        t1 = t0 + 1
    vmin, vmax = min(values), max(values)
    pad = (vmax - vmin) * 0.08 if vmax != vmin else 0.1
    vmin -= pad
    vmax += pad
    if as_percent:
        vmin = min(vmin, 0.0)

    def x_scale(dt: pd.Timestamp) -> float:
        return x0 + (dt.value - t0) / (t1 - t0) * (x1 - x0)

    def y_scale(value: float) -> float:
        return y0 - (value - vmin) / (vmax - vmin) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        svg_text(left, 34, title, size=22, weight="700"),
        svg_text(left, 54, y_label, size=13, fill=COLORS["muted"]),
        f'<rect x="{x0}" y="{y1}" width="{x1-x0}" height="{y0-y1}" fill="none" stroke="{COLORS["line"]}" stroke-width="1"/>',
    ]

    for i in range(5):
        value = vmin + (vmax - vmin) * i / 4
        y = y_scale(value)
        label = f"{value * 100:.0f}%" if as_percent else f"{value:.2f}"
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{COLORS["line"]}" stroke-width="1"/>')
        parts.append(svg_text(x0 - 10, y + 4, label, size=12, fill=COLORS["muted"], anchor="end"))

    for frac in [0.0, 0.33, 0.66, 1.0]:
        dt = d0 + (d1 - d0) * frac
        x = x_scale(dt)
        parts.append(svg_text(x, y0 + 24, dt.strftime("%Y-%m"), size=12, fill=COLORS["muted"], anchor="middle"))

    legend_x = x1 - 330
    legend_y = 32
    for idx, (col, label, color) in enumerate(columns):
        lx = legend_x + idx * 110
        parts.append(f'<line x1="{lx}" x2="{lx+22}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(svg_text(lx + 28, legend_y + 4, label, size=12, fill=COLORS["text"]))

    for col, _, color in columns:
        series = pd.to_numeric(data[col], errors="coerce")
        pts = [
            (x_scale(date), y_scale(float(value)))
            for date, value in zip(data["Date"], series)
            if pd.notna(value) and np.isfinite(float(value))
        ]
        parts.append(polyline(pts, color))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def drawdown_series(equity: pd.Series) -> pd.Series:
    values = pd.to_numeric(equity, errors="coerce")
    peak = values.cummax()
    return values / peak - 1.0


def write_bar_chart_svg(
    frame: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    output_path: Path,
    value_format: str = "pct",
    color: str = COLORS["accent"],
) -> None:
    data = frame.copy().head(12)
    width, height = CHART_WIDTH, 520
    left, right, top, bottom = 220, 60, 68, 42
    chart_w = width - left - right
    row_h = (height - top - bottom) / max(len(data), 1)
    max_value = pd.to_numeric(data[value_col], errors="coerce").max()
    max_value = float(max_value) if pd.notna(max_value) and max_value > 0 else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        svg_text(42, 36, title, size=22, weight="700"),
    ]
    for idx, row in data.iterrows():
        y = top + idx * row_h
        value = float(row[value_col]) if pd.notna(row[value_col]) else 0.0
        bar_w = chart_w * max(value, 0.0) / max_value
        label = str(row[label_col])
        if value_format == "pct":
            value_text = format_pct(value)
        elif value_format == "score":
            value_text = format_number(value, 2)
        else:
            value_text = format_number(value, 2)
        parts.append(svg_text(left - 12, y + row_h * 0.62, label, size=13, anchor="end"))
        parts.append(f'<rect x="{left}" y="{y+7:.1f}" width="{bar_w:.1f}" height="{max(row_h-14, 8):.1f}" fill="{color}"/>')
        parts.append(svg_text(left + bar_w + 8, y + row_h * 0.62, value_text, size=12, fill=COLORS["muted"]))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_fallback_policy_svg(selections_path: Path, output_path: Path) -> None:
    width, height = CHART_WIDTH, 420
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        svg_text(42, 36, "XLK Fallback Trigger Periods", size=22, weight="700"),
        svg_text(42, 58, "Policy shown: dynamic fallback to XLK only on relaxed or fallback-all selector stages.", size=13, fill=COLORS["muted"]),
    ]
    if not selections_path.exists():
        parts.append(svg_text(42, 120, "No fallback selection file found.", size=15, fill=COLORS["muted"]))
        parts.append("</svg>")
        output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        return

    selections = pd.read_csv(selections_path)
    sub = selections[
        (selections["Policy_ID"] == "dynamic_xlk_relaxed_only")
        & (pd.to_numeric(selections["Liquidity_Floor"], errors="coerce") == PRODUCTION_MIN_DOLLAR_VOLUME)
    ].copy()
    if sub.empty:
        parts.append(svg_text(42, 120, "No dynamic fallback selections found for the production liquidity floor.", size=15, fill=COLORS["muted"]))
        parts.append("</svg>")
        output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        return
    sub = sub.sort_values("Walk_Forward_Round")
    clean_stages = {
        "family_stable_strict_recent_xlk_drawdown_downside",
        "family_stable_recent_defensive_downside",
        "family_stable_benchmark_drawdown_guard",
        "strict_recent_xlk_drawdown_downside",
        "recent_defensive_downside",
    }
    x0, y0 = 60, 116
    bar_w = (width - 120) / len(sub)
    for idx, (_, row) in enumerate(sub.iterrows()):
        stage = str(row.get("Selection_Guard_Stage", ""))
        triggers = stage not in clean_stages
        color = COLORS["warning"] if triggers else COLORS["accent"]
        x = x0 + idx * bar_w
        parts.append(f'<rect x="{x+8:.1f}" y="{y0}" width="{bar_w-16:.1f}" height="116" fill="{color}"/>')
        parts.append(svg_text(x + bar_w / 2, y0 + 144, row.get("Validation_Fold", ""), size=13, anchor="middle"))
        parts.append(svg_text(x + bar_w / 2, y0 + 164, "XLK fallback" if triggers else "stock picks", size=12, fill=COLORS["muted"], anchor="middle"))
        parts.append(svg_text(x + bar_w / 2, y0 + 184, stage.replace("family_stable_", ""), size=11, fill=COLORS["muted"], anchor="middle"))
    parts.append(f'<rect x="42" y="286" width="16" height="16" fill="{COLORS["accent"]}"/>')
    parts.append(svg_text(66, 299, "Clean enough to stay in selected stocks", size=13))
    parts.append(f'<rect x="42" y="314" width="16" height="16" fill="{COLORS["warning"]}"/>')
    parts.append(svg_text(66, 327, "Would rotate to XLK under the fallback rule", size=13))
    parts.append(svg_text(42, 368, "The fixed production-qualified website ranking shown here uses no active fallback; this chart documents when the dynamic fallback rule would have fired.", size=12, fill=COLORS["muted"]))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run_production_backtest(
    score_history: Path,
    price_file: Path,
    historical_universe_path: Path,
    output_dir: Path,
    min_dollar_volume: float,
    top_n: int,
    portfolio_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = alpha_signal_quality.load_scores(score_history)
    tickers = sorted(set(scores["Ticker"].astype(str).str.upper()) | {"QQQ", "XLK"})
    start_date = scores["Date"].min() - pd.Timedelta(days=260)
    end_date = None
    price_wide, volume_wide = load_price_and_volume(price_file, tickers, start_date=start_date, end_date=end_date)
    enriched = alpha_signal_quality.attach_ranked_signals(scores, price_wide)
    if PRODUCTION_SCORE_COLUMN not in enriched.columns:
        raise RuntimeError(f"{PRODUCTION_SCORE_COLUMN} was not generated for historical scores")

    params = monthly_sim.SimulationParams(min_dollar_volume=min_dollar_volume)
    adv20 = monthly_sim.rolling_dollar_volume(price_wide, volume_wide, params.adv_window)
    historical_universe = monthly_sim.load_historical_universe(
        historical_universe_path,
        enriched,
        price_wide,
    )
    config = monthly_sim.DirectScoreConfig(
        config_id=PRODUCTION_CONFIG_ID,
        score_column=PRODUCTION_SCORE_COLUMN,
        notes="Production-qualified website default from the Sharadar decision pass.",
    )
    scored = monthly_sim.score_frame_for_config(enriched, config)
    daily, rebalances = monthly_sim.simulate_variant(
        scored=scored,
        config_id=config.config_id,
        top_n=top_n,
        mode=portfolio_mode,
        price_wide=price_wide,
        adv20=adv20,
        params=params,
        historical_universe=historical_universe,
    )
    summary = monthly_sim.summarize_simulation(daily, rebalances)
    latest_holdings = monthly_sim.build_latest_holdings(
        scored=scored,
        price_wide=price_wide,
        adv20=adv20,
        params=params,
        top_n_list=[top_n],
        modes=[portfolio_mode],
        config=config,
        historical_universe=historical_universe,
    )
    daily.to_csv(output_dir / "production-default-equity-curve.csv", index=False)
    rebalances.to_csv(output_dir / "production-default-rebalance-log.csv", index=False)
    summary.to_csv(output_dir / "production-default-summary.csv", index=False)
    latest_holdings.to_csv(output_dir / "production-default-latest-holdings.csv", index=False)
    return daily, rebalances, summary, latest_holdings


def write_backtest_charts(
    daily: pd.DataFrame,
    latest_holdings: pd.DataFrame,
    decision_selections: Path,
    output_dir: Path,
) -> None:
    chart = daily.copy()
    if not chart.empty:
        write_line_chart_svg(
            frame=chart,
            columns=[
                ("Strategy_Equity", "Model", COLORS["strategy"]),
                ("QQQ_Equity", "QQQ", COLORS["qqq"]),
                ("XLK_Equity", "XLK", COLORS["xlk"]),
            ],
            title="Compounded Equity Curve",
            y_label="Growth of $1 after simulated costs and slippage",
            output_path=output_dir / "production-default-equity-curve.svg",
        )
        dd = chart[["Date", "Strategy_Equity", "QQQ_Equity", "XLK_Equity"]].copy()
        dd["Strategy_Drawdown"] = drawdown_series(dd["Strategy_Equity"])
        dd["QQQ_Drawdown"] = drawdown_series(dd["QQQ_Equity"])
        dd["XLK_Drawdown"] = drawdown_series(dd["XLK_Equity"])
        write_line_chart_svg(
            frame=dd,
            columns=[
                ("Strategy_Drawdown", "Model", COLORS["strategy"]),
                ("QQQ_Drawdown", "QQQ", COLORS["qqq"]),
                ("XLK_Drawdown", "XLK", COLORS["xlk"]),
            ],
            title="Drawdown Through Time",
            y_label="Drawdown from prior peak",
            output_path=output_dir / "production-default-drawdown.svg",
            as_percent=True,
        )

    if not latest_holdings.empty:
        holdings = latest_holdings[
            (latest_holdings["Config_ID"] == PRODUCTION_CONFIG_ID)
            & (pd.to_numeric(latest_holdings["Top_N"], errors="coerce") == PRODUCTION_TOP_N)
            & (latest_holdings["Portfolio_Mode"] == PRODUCTION_PORTFOLIO_MODE)
        ].copy()
        if not holdings.empty:
            holdings = holdings.sort_values("Target_Weight", ascending=False)
            holdings["Label"] = holdings["Ticker"] + " - " + holdings["SubIndustry"].astype(str)
            write_bar_chart_svg(
                frame=holdings,
                label_col="Label",
                value_col="Target_Weight",
                title="Latest Top Holdings Target Weights",
                output_path=output_dir / "production-default-top-holdings.svg",
                value_format="pct",
                color=COLORS["accent"],
            )
    write_fallback_policy_svg(
        selections_path=decision_selections,
        output_path=output_dir / "production-default-fallback-periods.svg",
    )


def write_dashboard_html(output_dir: Path, ranked: pd.DataFrame, summary: pd.DataFrame, generated_at: str) -> None:
    model_date = html_escape(ranked["Model_Date"].iloc[0] if not ranked.empty else "")
    top_rows = render_rank_table_rows(ranked, 30)
    headers = "".join(f"<th>{html_escape(label)}</th>" for _, label, _ in top_table_columns())
    metrics = {}
    if not summary.empty:
        metrics = summary.iloc[0].to_dict()
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Analysis Research Dashboard</title>
<style>
:root {{ color-scheme: light; --ink:#172026; --muted:#64717b; --line:#d9e0e5; --head:#eef3f7; --alt:#f8fafb; --accent:#0f6b63; --warn:#7c2d12; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; color:var(--ink); background:#f3f6f8; }}
.wrap {{ max-width:1240px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 8px; font-size:26px; }}
h2 {{ margin:28px 0 12px; font-size:18px; }}
.meta {{ margin:0 0 10px; color:var(--muted); font-size:14px; line-height:1.45; }}
.disclaimer {{ margin:16px 0; padding:12px 14px; border-left:4px solid var(--warn); background:#fff7ed; color:#4a2211; font-size:13px; line-height:1.45; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:18px 0; }}
.metric {{ background:#fff; border:1px solid var(--line); padding:12px; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; }}
.metric strong {{ display:block; margin-top:4px; font-size:20px; }}
.chart {{ background:#fff; border:1px solid var(--line); margin:14px 0; overflow:auto; }}
.chart img {{ display:block; width:100%; height:auto; }}
.table-wrap {{ max-height:520px; overflow:auto; border:1px solid var(--line); background:white; }}
table {{ width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }}
th, td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ position:sticky; top:0; z-index:1; background:var(--head); font-size:12px; text-transform:uppercase; letter-spacing:0; color:#33424c; }}
tr:nth-child(even) td {{ background:var(--alt); }}
td:nth-child(1), td:nth-child(5), td:nth-child(6), td:nth-child(7), td:nth-child(8), td:nth-child(9) {{ text-align:right; font-variant-numeric:tabular-nums; }}
td:nth-child(4) {{ font-weight:750; color:var(--accent); }}
@media (max-width: 760px) {{ .metrics {{ grid-template-columns:1fr 1fr; }} .wrap {{ padding:16px; }} }}
</style>
</head>
<body>
<div class="wrap">
<h1>Stock Analysis Research Dashboard</h1>
<p class="meta">Model date: {model_date}. Production-qualified research default: {html_escape(PRODUCTION_SCORE_COLUMN)}, top {PRODUCTION_TOP_N}, {html_escape(PRODUCTION_PORTFOLIO_MODE)}, {format_dollars(PRODUCTION_MIN_DOLLAR_VOLUME)} minimum 20-day dollar volume.</p>
<div class="disclaimer"><strong>Research only.</strong> This dashboard is not financial advice, not a recommendation to buy or sell securities, and not an automated trading system. Results are simulated and require human review, forward paper testing, and risk controls before any real allocation decision.</div>
<div class="metrics">
<div class="metric"><span>Total Return</span><strong>{format_pct(metrics.get("Total_Return"))}</strong></div>
<div class="metric"><span>Excess vs XLK</span><strong>{format_pct(metrics.get("Excess_Return_vs_XLK"))}</strong></div>
<div class="metric"><span>Sharpe</span><strong>{format_number(metrics.get("Sharpe"), 2)}</strong></div>
<div class="metric"><span>Max Drawdown</span><strong>{format_pct(metrics.get("Max_Drawdown"))}</strong></div>
</div>
<h2>Backtest Charts</h2>
<div class="chart"><img src="production-default-equity-curve.svg" alt="Compounded equity curve compared with QQQ and XLK"></div>
<div class="chart"><img src="production-default-drawdown.svg" alt="Drawdown chart compared with QQQ and XLK"></div>
<div class="chart"><img src="production-default-fallback-periods.svg" alt="Fallback policy trigger periods"></div>
<div class="chart"><img src="production-default-top-holdings.svg" alt="Latest target holdings weights"></div>
<h2>Top Ranked Stocks</h2>
<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{top_rows}</tbody></table></div>
<p class="meta">Generated: {html_escape(generated_at)}.</p>
</div>
</body>
</html>
"""
    (output_dir / "stock-analysis-research-dashboard.html").write_text(html_text, encoding="utf-8")


def write_readme(output_dir: Path, summary: pd.DataFrame, generated_at: str) -> None:
    metrics = summary.iloc[0].to_dict() if not summary.empty else {}
    text = f"""# Website Assets

Generated: {generated_at}

These files are for the Stock Analysis website/demo. They are research outputs, not financial advice, not trade recommendations, and not automated order instructions.

Current website defaults:

- Ranking score: `{PRODUCTION_SCORE_COLUMN}`
- Liquidity filter: `${PRODUCTION_MIN_DOLLAR_VOLUME:,.0f}` minimum 20-day dollar volume
- Portfolio display: top `{PRODUCTION_TOP_N}` / `{PRODUCTION_PORTFOLIO_MODE}`
- Backtest total return: `{format_pct(metrics.get("Total_Return"))}`
- Backtest excess vs XLK: `{format_pct(metrics.get("Excess_Return_vs_XLK"))}`
- Backtest Sharpe: `{format_number(metrics.get("Sharpe"), 4)}`
- Backtest max drawdown: `{format_pct(metrics.get("Max_Drawdown"))}`

Primary website files:

- `stock-ranking-top-100-scrollable.html`
- `stock-analysis-research-dashboard.html`
- `production-default-top-100.csv`
- `production-default-summary.csv`
- `production-default-latest-holdings.csv`
- `production-default-equity-curve.svg`
- `production-default-drawdown.svg`
- `production-default-fallback-periods.svg`
- `production-default-top-holdings.svg`

Use the language "research system" or "quant research demo" on the website. Do not describe this as a proven live trading system.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def generate_website_assets(
    feature_history: Path = DEFAULT_FEATURE_HISTORY,
    score_history: Path = DEFAULT_SCORE_HISTORY,
    historical_price_file: Path = DEFAULT_PRICE_FILE,
    historical_universe: Path = DEFAULT_HISTORICAL_UNIVERSE,
    decision_selections: Path = DEFAULT_DECISION_SELECTIONS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_dollar_volume: float = PRODUCTION_MIN_DOLLAR_VOLUME,
    top_n: int = PRODUCTION_TOP_N,
    portfolio_mode: str = PRODUCTION_PORTFOLIO_MODE,
    skip_backtest: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M:%S %Z")

    latest, latest_date = load_latest_feature_rows(feature_history)
    ranked, _, _, adv_date = enriched_latest_ranking(
        latest=latest,
        latest_date=latest_date,
        price_file=historical_price_file,
        min_dollar_volume=min_dollar_volume,
    )
    save_top_ranking_outputs(ranked, output_dir, generated_at)

    daily = pd.DataFrame()
    rebalances = pd.DataFrame()
    summary = pd.DataFrame()
    latest_holdings = pd.DataFrame()
    if not skip_backtest:
        daily, rebalances, summary, latest_holdings = run_production_backtest(
            score_history=score_history,
            price_file=historical_price_file,
            historical_universe_path=historical_universe,
            output_dir=output_dir,
            min_dollar_volume=min_dollar_volume,
            top_n=top_n,
            portfolio_mode=portfolio_mode,
        )
        write_backtest_charts(daily, latest_holdings, decision_selections, output_dir)
    else:
        write_fallback_policy_svg(decision_selections, output_dir / "production-default-fallback-periods.svg")

    write_dashboard_html(output_dir, ranked, summary, generated_at)
    write_readme(output_dir, summary, generated_at)

    metadata = {
        "status": "complete",
        "generated_at": generated_at,
        "feature_history": str(feature_history),
        "score_history": str(score_history),
        "historical_price_file": str(historical_price_file),
        "historical_universe": str(historical_universe),
        "decision_selections": str(decision_selections),
        "output_dir": str(output_dir),
        "model_date": latest_date.strftime("%Y-%m-%d"),
        "adv_date": format_date(adv_date),
        "production_score_column": PRODUCTION_SCORE_COLUMN,
        "min_dollar_volume": min_dollar_volume,
        "top_n": top_n,
        "portfolio_mode": portfolio_mode,
        "eligible_ranked_rows": int(len(ranked)),
        "backtest_daily_rows": int(len(daily)),
        "backtest_rebalance_rows": int(len(rebalances)),
        "summary_rows": int(len(summary)),
        "latest_holding_rows": int(len(latest_holdings)),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "website_asset_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    metadata = generate_website_assets(
        feature_history=args.feature_history,
        score_history=args.score_history,
        historical_price_file=args.historical_price_file,
        historical_universe=args.historical_universe,
        decision_selections=args.decision_selections,
        output_dir=args.output_dir,
        min_dollar_volume=args.min_dollar_volume,
        top_n=args.top_n,
        portfolio_mode=args.portfolio_mode,
        skip_backtest=args.skip_backtest,
    )
    print("[SUCCESS] Website assets generated")
    print(json.dumps({
        "output_dir": metadata["output_dir"],
        "model_date": metadata["model_date"],
        "eligible_ranked_rows": metadata["eligible_ranked_rows"],
        "backtest_daily_rows": metadata["backtest_daily_rows"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
