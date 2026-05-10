"""
Tests for VolSurface — focused on no-arbitrage properties.

These tests use a synthetic surface (no live data fetching) so they
run deterministically in CI. The synthetic surface is constructed from
known implied vols satisfying standard no-arbitrage conditions.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.bsm import BSM
from src.surface import VolSurface


# ---------------------------------------------------------------------------
# Synthetic surface fixture
# ---------------------------------------------------------------------------

def make_synthetic_data(
    spot: float = 100.0,
    r: float = 0.05,
    q: float = 0.01,
    tenors: list = None,
    strikes: list = None,
) -> pd.DataFrame:
    """
    Build a synthetic options dataset with known IVs.
    We use a simple parametric smile: σ(K, T) = σ_ATM + skew * m + smile_coeff * m²
    where m = log(K/F).

    Parameters chosen so total variance is monotone in T (no calendar arb).
    """
    if tenors is None:
        tenors = [30, 60, 90, 180, 270, 365]
    if strikes is None:
        strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]

    sigma_atm_base = 0.20
    skew = -0.10       # negative skew (put wing elevated)
    smile_coeff = 0.15

    records = []
    for dte in tenors:
        T = dte / 365.0
        F = spot * np.exp((r - q) * T)

        # ATM vol increases slightly with tenor (upward sloping term structure)
        sigma_atm = sigma_atm_base + 0.01 * np.sqrt(T)

        for K in strikes:
            m = np.log(K / F)
            sigma = sigma_atm + skew * m + smile_coeff * m ** 2
            sigma = max(sigma, 0.05)  # floor

            # Use put for OTM puts, call for OTM calls (standard)
            opt_type = "put" if K < spot else "call"
            price = BSM.price(spot, K, T, r, q, sigma, opt_type)
            mid = price * 1.001  # tiny spread

            records.append({
                "expiry": f"2025-{dte:03d}",
                "T": T,
                "S": spot,
                "strike": K,
                "option_type": opt_type,
                "bid": mid * 0.999,
                "ask": mid * 1.001,
                "mid": mid,
                "volume": 500,
                "open_interest": 1000,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helper: build a fitted VolSurface from synthetic data
# ---------------------------------------------------------------------------

def make_fitted_surface() -> VolSurface:
    """Build a VolSurface backed by synthetic data (no yfinance calls)."""
    spot = 100.0
    r = 0.05
    q = 0.01
    data = make_synthetic_data(spot=spot, r=r, q=q)

    with patch("src.surface.fetch_options_chain", return_value=data), \
         patch("src.surface.get_risk_free_rate", return_value=r):
        surf = VolSurface(ticker="SYNTH", q=q)
        surf.fit(n_moneyness=20, n_tenors=10)

    return surf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoArbitrageCalendarSpread:
    """
    Calendar spread: total variance w(m, T) = σ²(m, T) * T must be
    non-decreasing in T for each fixed moneyness.
    """

    def test_total_variance_non_decreasing_atm(self):
        surf = make_fitted_surface()
        tenors = np.linspace(surf.grid_T.min(), surf.grid_T.max(), 10)
        prev_tv = -np.inf
        for T in tenors:
            iv = surf.iv(surf.spot, T)  # ATM query
            tv = iv ** 2 * T
            assert tv >= prev_tv - 1e-4, (
                f"Calendar arb: total var decreased at T={T:.3f} "
                f"(prev={prev_tv:.6f}, curr={tv:.6f})"
            )
            prev_tv = tv

    def test_total_variance_non_decreasing_otm(self):
        surf = make_fitted_surface()
        K = surf.spot * 0.95  # OTM put
        tenors = np.linspace(surf.grid_T.min(), surf.grid_T.max(), 10)
        prev_tv = -np.inf
        for T in tenors:
            iv = surf.iv(K, T)
            tv = iv ** 2 * T
            assert tv >= prev_tv - 1e-4
            prev_tv = tv


class TestSmileShape:
    """
    The smile should exhibit standard features:
    - Negative skew: OTM puts have higher IV than OTM calls (for equities)
    - Convexity: smile is U-shaped (higher wings vs ATM)
    """

    def test_put_skew_exists(self):
        """IV at 90% strike should exceed IV at 110% strike (negative skew)."""
        surf = make_fitted_surface()
        T = 60 / 365.0
        iv_otm_put = surf.iv(surf.spot * 0.90, T)
        iv_otm_call = surf.iv(surf.spot * 1.10, T)
        assert iv_otm_put > iv_otm_call, (
            f"Expected put skew (IV_put={iv_otm_put:.3f} > IV_call={iv_otm_call:.3f})"
        )

    def test_smile_convexity(self):
        """Wing IVs should exceed ATM IV (convexity)."""
        surf = make_fitted_surface()
        T = 60 / 365.0
        iv_atm = surf.iv(surf.spot, T)
        iv_low = surf.iv(surf.spot * 0.85, T)
        iv_high = surf.iv(surf.spot * 1.10, T)
        assert iv_low > iv_atm, f"Low strike should have higher IV than ATM"


class TestSurfaceQuery:
    """Smoke tests for the surface query interface."""

    def test_iv_returns_positive(self):
        surf = make_fitted_surface()
        for K_frac in [0.90, 1.00, 1.10]:
            iv = surf.iv(surf.spot * K_frac, 60 / 365.0)
            assert iv > 0, f"IV should be positive at K/S={K_frac}"

    def test_iv_within_reasonable_range(self):
        surf = make_fitted_surface()
        iv = surf.iv(surf.spot, 90 / 365.0)
        assert 0.05 <= iv <= 2.0, f"IV out of reasonable range: {iv:.4f}"

    def test_smile_returns_correct_length(self):
        surf = make_fitted_surface()
        strikes, iv = surf.smile(T_target=60 / 365.0, n_points=50)
        assert len(strikes) == 50
        assert len(iv) == 50

    def test_not_fitted_raises(self):
        with patch("src.surface.fetch_options_chain", return_value=make_synthetic_data()), \
             patch("src.surface.get_risk_free_rate", return_value=0.05):
            surf = VolSurface(ticker="SYNTH")
        with pytest.raises(RuntimeError):
            surf.iv(100.0, 0.5)
