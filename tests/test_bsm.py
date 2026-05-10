"""
Tests for BSM pricing engine.

These are not exhaustive — they serve as sanity checks that any
interviewer would expect: boundary conditions, put-call parity,
Greeks sign conventions, and implied vol round-trip.
"""

import numpy as np
import pytest
from src.bsm import BSM


# ---- Fixtures ----

PARAMS = dict(S=100.0, K=100.0, T=0.5, r=0.05, q=0.01, sigma=0.20)
PARAMS_ITM = dict(S=110.0, K=100.0, T=0.5, r=0.05, q=0.01, sigma=0.20)
PARAMS_OTM = dict(S=90.0, K=100.0, T=0.5, r=0.05, q=0.01, sigma=0.20)


# ---- Put-Call Parity ----

class TestPutCallParity:
    def test_atm_parity(self):
        assert BSM.put_call_parity_check(**PARAMS), "ATM put-call parity failed"

    def test_itm_parity(self):
        assert BSM.put_call_parity_check(**PARAMS_ITM), "ITM put-call parity failed"

    def test_otm_parity(self):
        assert BSM.put_call_parity_check(**PARAMS_OTM), "OTM put-call parity failed"

    def test_high_vol(self):
        params = dict(S=100, K=100, T=1.0, r=0.05, q=0.00, sigma=0.80)
        assert BSM.put_call_parity_check(**params), "High vol put-call parity failed"

    def test_short_dated(self):
        params = dict(S=100, K=100, T=5/365, r=0.05, q=0.01, sigma=0.20)
        assert BSM.put_call_parity_check(**params), "Short-dated put-call parity failed"


# ---- Boundary Conditions ----

class TestBoundaryConditions:
    def test_call_never_negative(self):
        for K in [80, 100, 120]:
            price = BSM.price(100, K, 0.5, 0.05, 0.01, 0.20, "call")
            assert price >= 0, f"Call price negative for K={K}"

    def test_put_never_negative(self):
        for K in [80, 100, 120]:
            price = BSM.price(100, K, 0.5, 0.05, 0.01, 0.20, "put")
            assert price >= 0, f"Put price negative for K={K}"

    def test_call_above_intrinsic(self):
        """Call price >= max(S - K*exp(-rT), 0)"""
        S, K, T, r, q, sigma = 110, 100, 0.5, 0.05, 0.01, 0.20
        price = BSM.price(S, K, T, r, q, sigma, "call")
        lower_bound = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        assert price >= lower_bound - 1e-6

    def test_call_below_spot(self):
        """Call price <= S * exp(-qT)"""
        S, K, T, r, q, sigma = 100, 80, 0.5, 0.05, 0.01, 0.20
        price = BSM.price(S, K, T, r, q, sigma, "call")
        assert price <= S * np.exp(-q * T) + 1e-6

    def test_zero_vol_call(self):
        """With zero vol, call = max(S - K*exp(-rT), 0)"""
        S, K, T, r, q = 110, 100, 0.5, 0.05, 0.0
        sigma = 1e-6
        price = BSM.price(S, K, T, r, q, sigma, "call")
        expected = max(S - K * np.exp(-r * T), 0)
        assert abs(price - expected) < 0.01

    def test_zero_time_call(self):
        """At expiry, call = max(S - K, 0)"""
        S, K = 110, 100
        price = BSM.price(S, K, T=0.0, r=0.05, q=0.01, sigma=0.20, option_type="call")
        assert abs(price - max(S - K, 0)) < 1e-6


# ---- Greeks sign conventions ----

class TestGreeks:
    def test_call_delta_positive(self):
        g = BSM.greeks(**PARAMS, option_type="call")
        assert 0 < g.delta < 1, "Call delta should be in (0, 1)"

    def test_put_delta_negative(self):
        g = BSM.greeks(**PARAMS, option_type="put")
        assert -1 < g.delta < 0, "Put delta should be in (-1, 0)"

    def test_gamma_positive(self):
        g = BSM.greeks(**PARAMS, option_type="call")
        assert g.gamma > 0, "Gamma should be positive"

    def test_vega_positive(self):
        g = BSM.greeks(**PARAMS, option_type="call")
        assert g.vega > 0, "Vega should be positive"

    def test_call_theta_negative(self):
        """Long call loses time value (theta < 0)"""
        g = BSM.greeks(**PARAMS, option_type="call")
        assert g.theta < 0, "Call theta should be negative (time decay)"

    def test_call_rho_positive(self):
        """Higher rates → higher call value"""
        g = BSM.greeks(**PARAMS, option_type="call")
        assert g.rho > 0, "Call rho should be positive"

    def test_put_rho_negative(self):
        """Higher rates → lower put value"""
        g = BSM.greeks(**PARAMS, option_type="put")
        assert g.rho < 0, "Put rho should be negative"

    def test_call_put_gamma_equal(self):
        """Gamma is identical for call and put at same (S, K, T)"""
        g_call = BSM.greeks(**PARAMS, option_type="call")
        g_put = BSM.greeks(**PARAMS, option_type="put")
        assert abs(g_call.gamma - g_put.gamma) < 1e-10

    def test_call_put_vega_equal(self):
        """Vega is identical for call and put at same (S, K, T)"""
        g_call = BSM.greeks(**PARAMS, option_type="call")
        g_put = BSM.greeks(**PARAMS, option_type="put")
        assert abs(g_call.vega - g_put.vega) < 1e-10

    def test_call_delta_put_delta_relationship(self):
        """call_delta - put_delta = exp(-qT) (from put-call parity)"""
        g_call = BSM.greeks(**PARAMS, option_type="call")
        g_put = BSM.greeks(**PARAMS, option_type="put")
        expected = np.exp(-PARAMS["q"] * PARAMS["T"])
        assert abs((g_call.delta - g_put.delta) - expected) < 1e-6


# ---- Implied Vol Round-trip ----

class TestImpliedVol:
    def test_round_trip_call(self):
        """Implied vol should recover input sigma exactly."""
        for sigma in [0.10, 0.20, 0.30, 0.50, 0.80]:
            price = BSM.price(100, 100, 0.5, 0.05, 0.01, sigma, "call")
            iv = BSM.implied_vol(price, 100, 100, 0.5, 0.05, 0.01, "call")
            assert abs(iv - sigma) < 1e-5, f"Round-trip failed for sigma={sigma}"

    def test_round_trip_put(self):
        for sigma in [0.10, 0.20, 0.50]:
            price = BSM.price(100, 110, 0.5, 0.05, 0.01, sigma, "put")
            iv = BSM.implied_vol(price, 100, 110, 0.5, 0.05, 0.01, "put")
            assert abs(iv - sigma) < 1e-5, f"Round-trip failed for sigma={sigma}"

    def test_below_intrinsic_returns_nan(self):
        """Price below intrinsic should return NaN."""
        iv = BSM.implied_vol(
            market_price=0.0, S=110, K=100, T=0.5, r=0.05, q=0.0, option_type="call"
        )
        assert np.isnan(iv), "Expected NaN for sub-intrinsic price"

    def test_zero_dte_returns_nan(self):
        iv = BSM.implied_vol(5.0, 105, 100, T=0.0, r=0.05, q=0.0, option_type="call")
        assert np.isnan(iv)
