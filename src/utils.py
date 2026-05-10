"""
Utility functions: options data fetching, date helpers, realised vol computation.

Data source: yfinance (free, sufficient for SPY/SPX liquid options).
Known limitations:
  - yfinance options data may have gaps for far-dated expiries
  - bid/ask mid-prices used as proxy for market price; spreads can be wide
  - risk-free rate approximated from 3M T-bill yield (^IRX)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

def get_risk_free_rate() -> float:
    """
    Fetch the current 3-month T-bill yield as a proxy for the risk-free rate.
    Returns the annualised continuously compounded rate.
    Falls back to 0.05 if fetch fails.
    """
    try:
        tbill = yf.Ticker("^IRX")
        hist = tbill.history(period="5d")
        rate_pct = hist["Close"].dropna().iloc[-1]
        return float(rate_pct) / 100.0
    except Exception:
        print("Warning: could not fetch risk-free rate. Using 5% fallback.")
        return 0.05


# ---------------------------------------------------------------------------
# Options chain fetching
# ---------------------------------------------------------------------------

def fetch_options_chain(
    ticker: str = "SPY",
    min_dte: int = 7,
    max_dte: int = 365,
    min_open_interest: int = 100,
    min_volume: int = 10,
) -> pd.DataFrame:
    """
    Fetch and clean the full options chain for a given ticker.

    Filters applied:
      - Expiry window: [min_dte, max_dte] calendar days
      - Minimum open interest and volume (liquidity screen)
      - Zero or negative bid removed
      - Mid-price used as market price proxy

    Returns a DataFrame with columns:
      expiry, strike, option_type, bid, ask, mid, volume,
      open_interest, T (years to expiry), S (spot)
    """
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    today = datetime.today().date()

    expiries = tk.options  # tuple of expiry strings
    records = []

    for exp_str in expiries:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days

        if not (min_dte <= dte <= max_dte):
            continue

        T = dte / 365.0

        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            continue

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["option_type"] = opt_type
            df["expiry"] = exp_str
            df["T"] = T
            df["S"] = spot
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
            records.append(df)

    if not records:
        raise ValueError(f"No options data found for {ticker}")

    data = pd.concat(records, ignore_index=True)

    # Rename for consistency
    data = data.rename(columns={"strike": "strike", "openInterest": "open_interest"})

    # Liquidity filters
    data = data[data["bid"] > 0]
    data = data[data["open_interest"] >= min_open_interest]
    data = data[data["volume"] >= min_volume]

    # Moneyness filter: keep strikes within 40% of spot (avoids deep wings noise)
    data = data[
        (data["strike"] >= 0.60 * data["S"]) &
        (data["strike"] <= 1.40 * data["S"])
    ]

    cols = ["expiry", "T", "S", "strike", "option_type", "bid", "ask", "mid",
            "volume", "open_interest"]
    available = [c for c in cols if c in data.columns]
    return data[available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Realised volatility
# ---------------------------------------------------------------------------

def compute_realised_vol(
    ticker: str = "SPY",
    window: int = 21,
    period: str = "2y",
    annualise: bool = True,
) -> pd.Series:
    """
    Compute rolling realised volatility from log returns.

    Parameters
    ----------
    ticker  : equity ticker
    window  : rolling window in trading days (21 ≈ 1 month)
    period  : yfinance history period string
    annualise : if True, multiply by sqrt(252)

    Returns
    -------
    pd.Series of rolling realised vol (NaN for first `window` observations)
    """
    hist = yf.Ticker(ticker).history(period=period)["Close"]
    log_ret = np.log(hist / hist.shift(1)).dropna()
    rv = log_ret.rolling(window).std()
    if annualise:
        rv = rv * np.sqrt(252)
    rv.name = f"realised_vol_{window}d"
    return rv


def compute_forward_realised_vol(
    ticker: str = "SPY",
    horizon: int = 21,
    period: str = "2y",
) -> pd.Series:
    """
    Compute forward-looking realised vol — i.e. vol realised over the *next*
    `horizon` trading days from each date. Used as the dependent variable
    in the implied vs realised regression.
    """
    rv = compute_realised_vol(ticker=ticker, window=horizon, period=period)
    # Shift backwards so each date has the vol realised going forward
    return rv.shift(-horizon).rename(f"fwd_realised_vol_{horizon}d")


# ---------------------------------------------------------------------------
# Implied vol spread (term structure)
# ---------------------------------------------------------------------------

def compute_iv_spread(
    iv_surface: pd.DataFrame,
    short_tenor_days: int = 30,
    long_tenor_days: int = 90,
    moneyness_band: float = 0.02,
) -> pd.Series:
    """
    Compute the ATM implied vol spread between two tenors.

    For each date in the surface, find the ATM IV at approximately
    `short_tenor_days` and `long_tenor_days`, then return the spread.

    Parameters
    ----------
    iv_surface : DataFrame with columns [date, T, strike, S, iv]
    short_tenor_days : approx short tenor in calendar days
    long_tenor_days  : approx long tenor in calendar days
    moneyness_band   : |K/S - 1| < moneyness_band to be considered ATM

    Returns
    -------
    pd.Series: iv_spread indexed by date
    """
    T_short = short_tenor_days / 365.0
    T_long = long_tenor_days / 365.0
    tol = 10 / 365.0  # 10-day tolerance for matching tenor

    results = {}

    for date, grp in iv_surface.groupby("date"):
        atm = grp[abs(grp["strike"] / grp["S"] - 1) < moneyness_band]

        short = atm[abs(atm["T"] - T_short) < tol]
        long_ = atm[abs(atm["T"] - T_long) < tol]

        if short.empty or long_.empty:
            continue

        iv_short = short["iv"].mean()
        iv_long = long_["iv"].mean()
        results[date] = iv_long - iv_short  # positive = upward sloping term structure

    return pd.Series(results, name=f"iv_spread_{short_tenor_days}d_{long_tenor_days}d")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def trading_days_between(start: datetime, end: datetime) -> int:
    """Approximate number of trading days between two dates."""
    return int(np.busday_count(start.date(), end.date()))


def dte_to_years(dte: int) -> float:
    """Convert days to expiry to fraction of year."""
    return dte / 365.0
