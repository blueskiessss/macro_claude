"""
macro_claude.py — Macro regime classification and trade idea generator.

Clean port of macro_engine.py with the following fixes applied:
  Fix 1: OECD Japan CPI — correct dataflow ID, dynamic startPeriod, CSV format
  Fix 2: LLM output — enforce JSON-structured response
  Fix 3: Deterministic JSON parsing (replaces fragile regex parsing)
  Fix 4: calc_trade_levels() uses explicit long_leg/short_leg from JSON
  Fix 5: build_plain_text_email() iterates JSON trade dicts (not regex split)
  Fix 6: main() wires JSON trade dicts into calc_trade_levels()
"""

import io
import json
import logging
import os
import re
import smtplib
import warnings
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import anthropic
from fredapi import Fred

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "Medevereux@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "Michael.devereux@schroders.com")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macro_claude_journal.json")

# FRED series: label -> (series_id, frequency)
# frequency: "monthly" or "daily"
FRED_SERIES = {
    # Growth (leading indicators, monthly)
    "US Growth (LEI)":         ("USALOLITOAASTSAM", "monthly"),
    "Eurozone Growth (LEI)":   ("DEULOLITOAASTSAM", "monthly"),
    "UK Growth (LEI)":         ("GBRLOLITOAASTSAM", "monthly"),
    "Japan Growth (LEI)":      ("JPNLOLITOAASTSAM", "monthly"),
    # Unemployment (monthly)
    "US Unemployment":         ("UNRATE",           "monthly"),
    "Germany Unemployment":    ("LRHUTTTTDEM156S",  "monthly"),
    "UK Unemployment":         ("LRHUTTTTGBM156S",  "monthly"),
    "Japan Unemployment":      ("LRUNTTTTJPM156S",  "monthly"),
    # Inflation (monthly)
    "US Inflation (PCE)":      ("PCEPI",            "monthly"),
    "Eurozone Inflation":      ("TOTNRGFOODEA20MI15XM", "monthly"),
    "UK Inflation (CPI)":      ("GBRCPICORMINMEI",  "monthly"),
    # Central bank rates
    "US CB Rate":              ("FEDFUNDS",         "monthly"),
    "Eurozone CB Rate":        ("ECBDFR",           "monthly"),
    "UK CB Rate":              ("IUDSOIA",          "daily"),    # daily, resample to monthly mean
    "Japan CB Rate":           ("IRSTCI01JPM156N",  "monthly"),
    # Yield curve
    "US Yield Curve (2s10s)":  ("T10Y2Y",          "daily"),    # use daily data directly
}

# ECB SDW: Eurozone 2s10s yield curve (daily)
ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/YC/"
    "B.U2.EUR.4F.G_N_A.SV_C_YM.SRS_10Y_2Y?format=csvdata"
)

# Yahoo Finance: label -> ticker
ASSET_TICKERS = {
    "US Equity (SPY)":       "SPY",
    "Eurozone Equity (EZU)": "EZU",
    "UK Equity (EWU)":       "EWU",
    "Japan Equity (EWJ)":    "EWJ",
    "US Bond (IEF)":         "IEF",
    "Eurozone Bond (EUNH)":  "EUNH.DE",
    "UK Bond (IGLT)":        "IGLT.L",
    "Japan Bond (236A)":     "236A.T",
    "USD Index (DXY)":       "DX-Y.NYB",
    "EUR/USD":               "EURUSD=X",
    "GBP/USD":               "GBPUSD=X",
    "USD/JPY":               "JPY=X",      # USDJPY: higher = weaker JPY
    "Gold (GC)":             "GC=F",
    "Silver (SI)":           "SI=F",
    "Copper (HG)":           "HG=F",
    "Oil WTI (CL)":          "CL=F",
}

# Commodities also get macro momentum treatment (price level)
COMMODITY_MACRO_LABELS = ["Gold (GC)", "Silver (SI)", "Copper (HG)", "Oil WTI (CL)"]

# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_fred_data() -> dict:
    """
    Fetch all FRED series. Returns dict of {label: pd.Series}.
    Daily series are kept at daily frequency.
    Monthly series are resampled to monthly last value.
    UK CB Rate (daily) is resampled to monthly mean before returning.
    On individual series failure, logs a warning and stores an empty Series.
    """
    fred_key = os.environ.get("FRED_API_KEY", "")
    if not fred_key:
        log.warning("FRED_API_KEY not set — FRED data will be empty")
        return {label: pd.Series(dtype=float) for label in FRED_SERIES}

    fred = Fred(api_key=fred_key)
    start_date = (datetime.today() - timedelta(days=365 * 6)).strftime("%Y-%m-%d")
    result = {}

    for label, (series_id, freq) in FRED_SERIES.items():
        try:
            raw = fred.get_series(series_id, observation_start=start_date)
            raw = raw.dropna()

            if freq == "monthly":
                # Monthly series: resample to month-end last observation
                series = raw.resample("ME").last().dropna()
            elif label == "UK CB Rate":
                # Daily series that we want as monthly for CB rate context
                series = raw.resample("ME").mean().dropna()
            else:
                # Keep daily (e.g. US Yield Curve 2s10s)
                series = raw

            result[label] = series
            log.info(f"FRED: fetched {label} ({series_id}), {len(series)} obs")
        except Exception as exc:
            log.warning(f"FRED: failed to fetch {label} ({series_id}): {exc}")
            result[label] = pd.Series(dtype=float)

    return result


def yoy_to_index(yoy: pd.Series, base: float = 100.0) -> pd.Series:
    """
    Reconstruct a price index from a YoY % change series.
    Sets the first 12 months to base=100, then chains forward using:
        Index(t) = Index(t-12) * (1 + YoY(t) / 100)
    This correctly respects the YoY relationship without compounding error.
    """
    yoy = yoy.sort_index().dropna()
    index = pd.Series(index=yoy.index, dtype=float)
    index.iloc[:12] = base
    for i in range(12, len(yoy)):
        index.iloc[i] = index.iloc[i - 12] * (1 + yoy.iloc[i] / 100)
    return index.dropna()


def fetch_oecd_japan_cpi() -> pd.Series:
    """
    Fetch Japan CPI ex-food and energy (COICOP 2018) from OECD SDMX API.
    Returns YoY % series converted to index level via yoy_to_index().
    Returns empty Series on failure.
    """
    start = (datetime.today() - timedelta(days=365 * 6)).strftime("%Y-%m")
    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.TPS,DSD_PRICES_COICOP2018@DF_PRICES_C2018_N_TXCP01_NRG,1.0/"
        f"JPN.M......?startPeriod={start}&format=csvfile"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
        yoy = df.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float).sort_index()
        series = yoy_to_index(yoy)
        log.info(f"OECD: fetched Japan CPI ex-food/energy (COICOP 2018), {len(series)} obs")
        return series
    except Exception as exc:
        log.warning(f"OECD: failed to fetch Japan CPI: {exc}")
        return pd.Series(dtype=float)


def fetch_ecb_yield_curve() -> pd.Series:
    """
    Fetch Eurozone 2s10s yield curve from ECB SDW API (CSV format).
    Returns daily pd.Series indexed by datetime, or empty Series on failure.
    """
    try:
        resp = requests.get(ECB_URL, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        # Expected columns: TIME_PERIOD, OBS_VALUE (plus others)
        df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
        df = df.dropna(subset=["OBS_VALUE"])
        df = df.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float).sort_index()

        log.info(f"ECB: fetched Eurozone 2s10s yield curve, {len(df)} obs")
        return df

    except Exception as exc:
        log.warning(f"ECB: failed to fetch Eurozone 2s10s yield curve: {exc}")
        return pd.Series(dtype=float)


def fetch_asset_prices() -> dict:
    """
    Fetch 5 years of daily close prices for all assets via yfinance.
    Returns dict of {label: pd.Series}. On ticker failure, stores empty Series.
    """
    result = {}
    for label, ticker in ASSET_TICKERS.items():
        try:
            raw = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
            if raw.empty:
                raise ValueError("Empty dataframe returned")
            prices = raw["Close"].squeeze().dropna()
            result[label] = prices
            log.info(f"yfinance: fetched {label} ({ticker}), {len(prices)} obs")
        except Exception as exc:
            log.warning(f"yfinance: failed to fetch {label} ({ticker}): {exc}")
            result[label] = pd.Series(dtype=float)

    return result


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

def _safe_iloc(series: pd.Series, idx: int):
    """Return series.iloc[idx] if index is valid, else NaN."""
    if len(series) > abs(idx):
        val = series.iloc[idx]
        return float(val) if pd.notna(val) else np.nan
    return np.nan


def calc_macro_momentum(series_dict: dict, is_daily: dict) -> dict:
    """
    Calculate momentum metrics for macro series.

    For monthly series (is_daily=False):
        - 3M change: index offset -1 vs -4  (latest vs 3 months ago)
        - 12M change: index offset -1 vs -13
        - 3M of 3M:  current 3M change minus 3M change computed 3 months ago
        - 3M of 12M: current 12M change minus 12M change computed 3 months ago

    For daily series (is_daily=True):
        - 3M ~ 63 trading days
        - 12M ~ 252 trading days
        Equivalent offsets applied.

    Returns dict: {label: {latest, 3m_change, 12m_change, 3m_of_3m, 3m_of_12m}}
    """
    results = {}

    for label, series in series_dict.items():
        series = series.dropna()
        if len(series) < 2:
            results[label] = {
                "latest": np.nan, "3m_change": np.nan,
                "12m_change": np.nan, "3m_of_3m": np.nan, "3m_of_12m": np.nan
            }
            continue

        daily = is_daily.get(label, False)
        # Offsets: how many observations back = 3M and 12M
        off_3m  = 63  if daily else 3
        off_12m = 252 if daily else 12

        latest    = _safe_iloc(series, -1)
        ago_3m    = _safe_iloc(series, -(off_3m + 1))
        ago_12m   = _safe_iloc(series, -(off_12m + 1))
        # 3M change 3 months ago (for acceleration):
        ago_3m_b  = _safe_iloc(series, -(off_3m * 2 + 1))
        ago_12m_b = _safe_iloc(series, -(off_12m + off_3m + 1))

        chg_3m  = latest - ago_3m   if pd.notna(ago_3m)    else np.nan
        chg_12m = latest - ago_12m  if pd.notna(ago_12m)   else np.nan

        # 3M change from 3 months ago
        chg_3m_prev  = ago_3m - ago_3m_b  if pd.notna(ago_3m) and pd.notna(ago_3m_b) else np.nan
        # 12M change from 3 months ago
        chg_12m_prev = ago_3m - ago_12m_b if pd.notna(ago_3m) and pd.notna(ago_12m_b) else np.nan

        results[label] = {
            "latest":    round(latest, 4)    if pd.notna(latest)     else np.nan,
            "3m_change": round(chg_3m, 4)    if pd.notna(chg_3m)    else np.nan,
            "12m_change":round(chg_12m, 4)   if pd.notna(chg_12m)   else np.nan,
            "3m_of_3m":  round(chg_3m - chg_3m_prev, 4)
                         if pd.notna(chg_3m) and pd.notna(chg_3m_prev) else np.nan,
            "3m_of_12m": round(chg_12m - chg_12m_prev, 4)
                         if pd.notna(chg_12m) and pd.notna(chg_12m_prev) else np.nan,
        }

    return results


def calc_asset_metrics(price_dict: dict) -> dict:
    """
    Calculate 6M Sharpe ratio and 5Y percentile for each asset.

    6M Sharpe:
        mean(daily returns, last 126 days) / std(daily returns, last 126 days) * sqrt(125)
        Risk-free rate = 0.

    5Y Percentile:
        Rolling 6M Sharpe over last 1260 trading days.
        Today's Sharpe expressed as percentile of that 5Y distribution.

    Returns dict: {label: {sharpe_6m, pct_5y}}
    """
    results = {}

    for label, prices in price_dict.items():
        prices = prices.dropna()
        if len(prices) < 130:
            results[label] = {"sharpe_6m": np.nan, "pct_5y": np.nan}
            continue

        returns = prices.pct_change().dropna()

        # Current 6M Sharpe
        ret_126 = returns.iloc[-126:]
        mu, sigma = ret_126.mean(), ret_126.std()
        sharpe_6m = (mu / sigma * np.sqrt(252)) if sigma > 0 else np.nan

        # Rolling 6M Sharpe over 5Y window
        lookback = min(len(returns), 1260)
        ret_5y = returns.iloc[-lookback:]

        rolling_sharpe = ret_5y.rolling(126).apply(
            lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan,
            raw=True
        ).dropna()

        if len(rolling_sharpe) > 0 and pd.notna(sharpe_6m):
            pct_5y = float(np.sum(rolling_sharpe <= sharpe_6m) / len(rolling_sharpe) * 100)
        else:
            pct_5y = np.nan

        results[label] = {
            "sharpe_6m": round(sharpe_6m, 3) if pd.notna(sharpe_6m) else np.nan,
            "pct_5y":    round(pct_5y, 1)    if pd.notna(pct_5y)    else np.nan,
        }

    return results


# ---------------------------------------------------------------------------
# Data Availability Check
# ---------------------------------------------------------------------------

def check_data_availability(macro_momentum: dict, asset_metrics: dict) -> None:
    """
    Raise RuntimeError if fewer than 50% of all series have valid (non-NaN) latest values.
    """
    total, valid = 0, 0

    for label, stats in macro_momentum.items():
        total += 1
        if pd.notna(stats.get("latest", np.nan)):
            valid += 1

    for label, stats in asset_metrics.items():
        total += 1
        if pd.notna(stats.get("sharpe_6m", np.nan)):
            valid += 1

    pct = valid / total * 100 if total > 0 else 0
    log.info(f"Data availability: {valid}/{total} series valid ({pct:.0f}%)")

    if pct < 50:
        raise RuntimeError(
            f"Data availability failure: only {valid}/{total} series ({pct:.0f}%) have valid data. "
            f"Check API keys and network connectivity. Aborting."
        )


# ---------------------------------------------------------------------------
# Prompt Building
# ---------------------------------------------------------------------------

def _fmt(val, decimals=3):
    """Format a float value for table display, showing 'n/a' for NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val:+.{decimals}f}" if val != 0 else f"{val:.{decimals}f}"


def _fmt_plain(val, decimals=2):
    """Format a float value without sign prefix."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val:.{decimals}f}"


def build_prompt(
    macro_momentum: dict,
    commodity_momentum: dict,
    asset_metrics: dict,
) -> str:
    """
    Build a structured 3-layer prompt for the LLM.

    Layer 1: Data tables (macro per country, commodities, asset momentum/valuation)
    Layer 2: Regime classification instructions (JSON output)
    Layer 3: Trade idea generation instructions (JSON output, up to 3 ideas)
    """

    # --- Organise macro data by country ---
    country_groups = {
        "United States": [
            "US Growth (LEI)", "US Unemployment", "US Inflation (PCE)",
            "US CB Rate", "US Yield Curve (2s10s)"
        ],
        "Eurozone": [
            "Eurozone Growth (LEI)", "Germany Unemployment", "Eurozone Inflation",
            "Eurozone CB Rate", "Eurozone 2s10s Yield Curve"
        ],
        "United Kingdom": [
            "UK Growth (LEI)", "UK Unemployment", "UK Inflation (CPI)", "UK CB Rate"
        ],
        "Japan": [
            "Japan Growth (LEI)", "Japan Unemployment", "Japan Core CPI",
            "Japan CB Rate"
        ],
    }

    def macro_table_rows(labels: list, data: dict) -> str:
        header = f"{'Series':<35} {'Latest':>10} {'3M Chg':>10} {'12M Chg':>10} {'3M/3M':>10} {'3M/12M':>10}"
        sep = "-" * len(header)
        rows = [header, sep]
        for lbl in labels:
            stats = data.get(lbl)
            if stats is None:
                continue
            rows.append(
                f"{lbl:<35} "
                f"{_fmt_plain(stats['latest']):>10} "
                f"{_fmt(stats['3m_change']):>10} "
                f"{_fmt(stats['12m_change']):>10} "
                f"{_fmt(stats['3m_of_3m']):>10} "
                f"{_fmt(stats['3m_of_12m']):>10}"
            )
        return "\n".join(rows)

    # Build per-country blocks
    country_blocks = []
    for country, labels in country_groups.items():
        block = f"## {country}\n{macro_table_rows(labels, macro_momentum)}"
        country_blocks.append(block)

    # Commodity macro table
    comm_header = f"{'Commodity':<35} {'Latest':>10} {'3M Chg':>10} {'12M Chg':>10} {'3M/3M':>10} {'3M/12M':>10}"
    comm_sep = "-" * len(comm_header)
    comm_rows = [comm_header, comm_sep]
    for lbl in COMMODITY_MACRO_LABELS:
        stats = commodity_momentum.get(lbl)
        if stats is None:
            continue
        comm_rows.append(
            f"{lbl:<35} "
            f"{_fmt_plain(stats['latest']):>10} "
            f"{_fmt(stats['3m_change']):>10} "
            f"{_fmt(stats['12m_change']):>10} "
            f"{_fmt(stats['3m_of_3m']):>10} "
            f"{_fmt(stats['3m_of_12m']):>10}"
        )

    # Asset momentum and valuation table
    asset_header = f"{'Asset':<35} {'6M Sharpe':>10} {'5Y Pctile':>10}"
    asset_sep = "-" * len(asset_header)
    asset_rows = [asset_header, asset_sep]
    for lbl, stats in asset_metrics.items():
        asset_rows.append(
            f"{lbl:<35} "
            f"{_fmt_plain(stats['sharpe_6m'], 3):>10} "
            f"{_fmt_plain(stats['pct_5y'], 1):>10}"
        )

    prompt = f"""
You are a discretionary macro portfolio manager's analytical engine. Your role is to classify macroeconomic regimes and generate actionable trade ideas from structured data.

Investment philosophy:
- Every trade compares PRICE TREND with ECONOMIC TREND.
- MOMENTUM: price following economic trend — ride it.
- MEAN REVERSION: price dislocated from economic trend — fade it (requires higher conviction bar; multiple independent confirmations needed).
- Directional trades on single assets are preferred over relative value where possible (one bet, not two; positive risk premia assets appreciate over time).
- FX trades must always be expressed as currency pairs, never single-leg directional.
- Japan FX always in USDJPY convention: higher USD/JPY = weaker JPY.
- Commodity long/short directional trades are allowed.

========================================================
LAYER 1: MACRO DATA
========================================================

{chr(10).join(country_blocks)}

## Commodities (Macro Momentum — price level)
{chr(10).join(comm_rows)}

## Asset Momentum and Valuation
{chr(10).join(asset_rows)}

Note on asset table:
- 6M Sharpe: annualised Sharpe ratio over last 126 trading days (risk-free = 0).
- 5Y Percentile: where today's 6M Sharpe sits in its 5-year rolling history (0=lowest ever, 100=highest ever).
- USD/JPY (JPY=X): higher = weaker JPY. Interpret momentum direction accordingly.

========================================================
LAYER 2: REGIME CLASSIFICATION
========================================================

For each of the four countries/regions — United States, Eurozone, United Kingdom, Japan —
classify the macro regime based on the data above.

For each country classify:
1. Growth: "Accelerating" | "Stable" | "Decelerating"
2. Inflation: "Rising" | "Stable" | "Falling"
3. Quadrant: "Goldilocks" | "Reflation" | "Stagflation" | "Recession"
4. CB Stance: "Easing" | "Neutral" | "Tightening"
5. Yield Curve: "Steepening" | "Flat" | "Inverting" | "N/A"
6. Rationale: one sentence per country summarising the key signal

========================================================
LAYER 3: TRADE IDEAS
========================================================

Based on the regime classifications and asset momentum/valuation data, generate up to 3
trade ideas. Only generate an idea where the data genuinely supports conviction.
Do not force ideas to fill a quota.

Rules:
- Directional trades (long or short a single asset) are allowed and preferred where conviction is clear.
- Relative value (long one / short another) is allowed within the same asset class only.
- Individual commodity longs or shorts are allowed.
- FX trades must be expressed as pairs (e.g. "Long EUR/USD"). Never single-leg FX.
- USD/JPY: higher = weaker JPY. Short USD/JPY = bullish JPY.

========================================================
OUTPUT FORMAT — CRITICAL
========================================================

You must respond with ONLY a single valid JSON object. No preamble, no explanation, no markdown
code fences. The JSON must exactly follow this schema:

{{
  "regimes": {{
    "United States": {{
      "growth": "Accelerating|Stable|Decelerating",
      "inflation": "Rising|Stable|Falling",
      "quadrant": "Goldilocks|Reflation|Stagflation|Recession",
      "cb_stance": "Easing|Neutral|Tightening",
      "yield_curve": "Steepening|Flat|Inverting|N/A",
      "rationale": "one sentence"
    }},
    "Eurozone": {{ "growth": "...", "inflation": "...", "quadrant": "...", "cb_stance": "...", "yield_curve": "...", "rationale": "..." }},
    "United Kingdom": {{ "growth": "...", "inflation": "...", "quadrant": "...", "cb_stance": "...", "yield_curve": "...", "rationale": "..." }},
    "Japan": {{ "growth": "...", "inflation": "...", "quadrant": "...", "cb_stance": "...", "yield_curve": "...", "rationale": "..." }}
  }},
  "trade_ideas": [
    {{
      "id": 1,
      "description": "e.g. Long SPY, Long EUR/USD, Long Gold (GC)",
      "long_leg": "exact label from ASSET_TICKERS or null",
      "short_leg": "exact label from ASSET_TICKERS or null",
      "rationale": "how macro regime supports this trade",
      "momentum_valuation": "what 6M Sharpe and 5Y percentile tell you",
      "confidence": "High|Medium|Low",
      "confidence_reasoning": "key reason for conviction level and primary risk"
    }}
  ]
}}

Use exact asset labels from the asset table (e.g. "US Equity (SPY)", "EUR/USD", "Gold (GC)").
The trade_ideas array may contain 0 to 3 items. Return an empty array if no ideas meet conviction bar.
"""

    return prompt.strip()


# ---------------------------------------------------------------------------
# LLM Call
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """
    Call Claude via the Anthropic SDK. Returns the response text.
    Uses claude-sonnet-4-20250514 as specified.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> dict:
    """
    Parse JSON response from LLM. Falls back to empty structure on failure.
    Returns dict with keys: 'regimes', 'trade_ideas', 'raw_response'.
    """
    countries = ["United States", "Eurozone", "United Kingdom", "Japan"]
    empty_regime = {
        "growth": "Unknown", "inflation": "Unknown", "quadrant": "Unknown",
        "cb_stance": "Unknown", "yield_curve": "Unknown", "rationale": ""
    }
    fallback = {
        "regimes": {c: empty_regime.copy() for c in countries},
        "trade_ideas": [],
        "raw_response": response_text,
    }

    try:
        # Strip any accidental markdown fences if present
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
        data = json.loads(clean)

        # Ensure all four countries present
        regimes = data.get("regimes", {})
        for country in countries:
            if country not in regimes:
                regimes[country] = empty_regime.copy()

        return {
            "regimes": regimes,
            "trade_ideas": data.get("trade_ideas", []),
            "raw_response": response_text,
        }

    except (json.JSONDecodeError, KeyError) as exc:
        log.warning(f"Failed to parse LLM JSON response: {exc}. Storing raw response.")
        return fallback


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def write_journal(
    parsed: dict,
    raw_snapshot: dict,
) -> None:
    """
    Append a run entry to macro_claude_journal.json.
    Loads existing file if present, appends, writes back.

    Entry schema:
    {
        "timestamp": ISO datetime string,
        "regimes": {country: {growth, inflation, quadrant, cb_stance, yield_curve, rationale}, ...},
        "trade_ideas": list of trade dicts,
        "raw_data_snapshot": {label: latest_value, ...}
    }
    """
    entry = {
        "timestamp":        datetime.utcnow().isoformat(),
        "regimes":          parsed.get("regimes", {}),
        "trade_ideas":      parsed.get("trade_ideas", []),
        "raw_data_snapshot": raw_snapshot,
    }

    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "r") as f:
            journal = json.load(f)
    else:
        journal = []

    journal.append(entry)

    with open(JOURNAL_FILE, "w") as f:
        json.dump(journal, f, indent=2, default=str)

    log.info(f"Journal updated: {JOURNAL_FILE} ({len(journal)} entries)")


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------

def calc_trade_levels(trade: dict, asset_prices: dict) -> dict:
    """
    Calculate entry, stop, and target for a trade idea dict.
    Uses explicit long_leg / short_leg fields from JSON output.
    stop   = entry * (1 - 0.5  * vol_60d)   [directional]
    target = entry * (1 + 0.75 * vol_60d)   [directional]
    For RV: spread vol-based stop/target.
    """
    na = {"entry": "N/A", "stop": "N/A", "target": "N/A"}
    _FX_LABELS = {"EUR/USD", "GBP/USD", "USD/JPY"}

    def is_fx(label):
        return label in _FX_LABELS if label else False

    def fmt(v, label):
        return f"{v:.4f}" if is_fx(label) else f"{v:.2f}"

    long_leg  = trade.get("long_leg")
    short_leg = trade.get("short_leg")

    # Directional trade
    if long_leg and not short_leg:
        prices = asset_prices.get(long_leg, pd.Series(dtype=float)).dropna()
        if len(prices) < 61:
            return na
        try:
            log_ret = np.log(prices / prices.shift(1)).dropna()
            vol_60d = float(log_ret.iloc[-60:].std() * np.sqrt(252))
            entry = float(prices.iloc[-1])
            return {
                "entry":  fmt(entry, long_leg),
                "stop":   fmt(entry * (1 - 0.5  * vol_60d), long_leg),
                "target": fmt(entry * (1 + 0.75 * vol_60d), long_leg),
            }
        except Exception:
            return na

    # Short-only directional
    if short_leg and not long_leg:
        prices = asset_prices.get(short_leg, pd.Series(dtype=float)).dropna()
        if len(prices) < 61:
            return na
        try:
            log_ret = np.log(prices / prices.shift(1)).dropna()
            vol_60d = float(log_ret.iloc[-60:].std() * np.sqrt(252))
            entry = float(prices.iloc[-1])
            return {
                "entry":  fmt(entry, short_leg),
                "stop":   fmt(entry * (1 + 0.5  * vol_60d), short_leg),
                "target": fmt(entry * (1 - 0.75 * vol_60d), short_leg),
            }
        except Exception:
            return na

    # Relative value trade
    if long_leg and short_leg:
        lp = asset_prices.get(long_leg,  pd.Series(dtype=float)).dropna()
        sp = asset_prices.get(short_leg, pd.Series(dtype=float)).dropna()
        if len(lp) < 61 or len(sp) < 61:
            return na
        try:
            combined = pd.DataFrame({"long": lp, "short": sp}).dropna()
            if len(combined) < 61:
                return na
            spread = (combined["long"] - combined["short"]).iloc[-126:]
            entry = float(spread.iloc[-1])
            spread_vol = float(spread.diff().dropna().iloc[-60:].std() * np.sqrt(252))
            both_fx = is_fx(long_leg) and is_fx(short_leg)
            f = lambda v: f"{v:.4f}" if both_fx else f"{v:.2f}"
            return {
                "entry":  f(entry),
                "stop":   f(entry - spread_vol * 0.5),
                "target": f(entry + spread_vol * 0.75),
            }
        except Exception:
            return na

    return na


def build_plain_text_email(parsed: dict, levels: list) -> str:
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    lines = []

    lines.append(f"MACRO CLAUDE \u2014 {today_str}")
    lines.append("=" * 50)
    lines.append("")

    # Section 1: Regime Classifications
    lines.append("REGIME CLASSIFICATIONS")
    lines.append("-" * 22)
    for country, fields in parsed.get("regimes", {}).items():
        lines.append(country)
        lines.append(f"  Growth:      {fields.get('growth', 'N/A')}")
        lines.append(f"  Inflation:   {fields.get('inflation', 'N/A')}")
        lines.append(f"  Quadrant:    {fields.get('quadrant', 'N/A')}")
        lines.append(f"  CB Stance:   {fields.get('cb_stance', 'N/A')}")
        lines.append(f"  Yield Curve: {fields.get('yield_curve', 'N/A')}")
        lines.append(f"  Rationale:   {fields.get('rationale', '')}")
        lines.append("")

    # Section 2: Trade Ideas
    lines.append("TRADE IDEAS")
    lines.append("-" * 11)
    trade_ideas = parsed.get("trade_ideas", [])
    if not trade_ideas:
        lines.append("No trade ideas generated this run.")
    else:
        for i, trade in enumerate(trade_ideas):
            lines.append(f"TRADE {trade.get('id', i+1)}: {trade.get('description', '')}")
            lines.append(f"  Rationale:            {trade.get('rationale', '')}")
            lines.append(f"  Momentum/Valuation:   {trade.get('momentum_valuation', '')}")
            lines.append(f"  Confidence:           {trade.get('confidence', '')} \u2014 {trade.get('confidence_reasoning', '')}")
            if i < len(levels):
                lvl = levels[i]
                lines.append(f"  Entry:  {lvl.get('entry', 'N/A')}")
                lines.append(f"  Stop:   {lvl.get('stop', 'N/A')}")
                lines.append(f"  Target: {lvl.get('target', 'N/A')}")
            lines.append("")

    lines.append("=" * 50)
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("For internal use only.")
    return "\n".join(lines)


def send_email(subject: str, body: str) -> None:
    """Send a plain text email via Gmail SMTP SSL (port 465)."""
    if not GMAIL_APP_PASSWORD:
        log.warning("GMAIL_APP_PASSWORD not set \u2014 skipping email send")
        return
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = RECIPIENT_EMAIL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        log.info("Email sent to %s", RECIPIENT_EMAIL)
    except Exception as e:
        log.error("Failed to send email: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== macro_claude.py starting ===")

    # 1. Fetch FRED data
    fred_data = fetch_fred_data()

    # 2. Fetch OECD Japan CPI
    japan_cpi = fetch_oecd_japan_cpi()
    fred_data["Japan Core CPI"] = japan_cpi

    # 3. Fetch ECB Eurozone 2s10s
    ecb_curve = fetch_ecb_yield_curve()
    fred_data["Eurozone 2s10s Yield Curve"] = ecb_curve

    # 4. Fetch asset prices
    asset_prices = fetch_asset_prices()

    # 5. Identify which series are daily vs monthly for momentum calc
    is_daily_map = {
        "US Yield Curve (2s10s)": True,
        "Eurozone 2s10s Yield Curve": True,
        # UK CB Rate was resampled to monthly already; treat as monthly
    }

    # 6. Compute macro momentum (all FRED + OECD + ECB series)
    macro_momentum = calc_macro_momentum(fred_data, is_daily_map)

    # 7. Compute commodity macro momentum from asset prices
    commodity_momentum = {}
    for lbl in COMMODITY_MACRO_LABELS:
        prices = asset_prices.get(lbl, pd.Series(dtype=float))
        if not prices.empty:
            # Daily price series — use daily offsets
            commodity_momentum[lbl] = calc_macro_momentum(
                {lbl: prices}, {lbl: True}
            )[lbl]
        else:
            commodity_momentum[lbl] = {
                "latest": np.nan, "3m_change": np.nan,
                "12m_change": np.nan, "3m_of_3m": np.nan, "3m_of_12m": np.nan
            }

    # 8. Compute asset momentum and valuation metrics
    asset_metrics = calc_asset_metrics(asset_prices)

    # 9. Check data availability — abort if <50% valid
    check_data_availability(macro_momentum, asset_metrics)

    # 10. Build prompt
    prompt = build_prompt(macro_momentum, commodity_momentum, asset_metrics)

    # 11. Call LLM
    log.info("Calling LLM...")
    response_text = call_llm(prompt)

    # 12. Print full response to console
    print("\n" + "=" * 70)
    print("MACRO CLAUDE — LLM RESPONSE")
    print("=" * 70)
    print(response_text)
    print("=" * 70 + "\n")

    # 13. Parse response
    parsed = parse_llm_response(response_text)

    # 14. Build raw snapshot (latest values only)
    raw_snapshot = {}
    # Latest value of each macro series
    for label, stats in macro_momentum.items():
        val = stats.get("latest", np.nan)
        raw_snapshot[label] = val if pd.notna(val) else None
    # Latest closing price of each asset and commodity ticker
    for label, prices in asset_prices.items():
        prices_clean = prices.dropna()
        raw_snapshot[label] = float(prices_clean.iloc[-1]) if len(prices_clean) > 0 else None

    # 15. Calculate entry/stop/target levels for each trade
    trade_ideas = parsed.get("trade_ideas", [])
    trade_levels = [calc_trade_levels(trade, asset_prices) for trade in trade_ideas]

    # 16. Build plain text email body
    email_body = build_plain_text_email(parsed, trade_levels)

    # 17. Send email
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    send_email(f"Macro Claude \u2014 {today_str}", email_body)

    # 18. Write journal
    write_journal(parsed, raw_snapshot)

    log.info("=== macro_claude.py complete ===")


if __name__ == "__main__":
    main()
