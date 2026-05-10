"""
Black-Scholes-Merton pricing engine.

Conventions:
  S     : spot price
  K     : strike price
  T     : time to expiry in years
  r     : risk-free rate (continuously compounded)
  q     : continuous dividend yield
  sigma : implied volatility (annualised)

All Greeks follow market-standard sign conventions.
Vega and Rho are scaled per 1% move (divide by 100) for interpretability.
Theta is per calendar day.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Literal


OptionType = Literal["call", "put"]


@dataclass
class BSMResult:
    price: float
    delta: float
    gamma: float
    vega: float   # per 1% vol move
    theta: float  # per calendar day
    rho: float    # per 1% rate move

    def __repr__(self) -> str:
        return (
            f"BSMResult(\n"
            f"  price={self.price:.4f},\n"
            f"  delta={self.delta:.4f},\n"
            f"  gamma={self.gamma:.6f},\n"
            f"  vega={self.vega:.4f},\n"
            f"  theta={self.theta:.4f},\n"
            f"  rho={self.rho:.4f}\n"
            f")"
        )


class BSM:
    """
    Vectorised Black-Scholes-Merton model.
    All methods accept scalar or numpy array inputs.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _d1(S, K, T, r, q, sigma):
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def _d2(S, K, T, r, q, sigma):
        return BSM._d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)

    # ------------------------------------------------------------------
    # Price
    # ------------------------------------------------------------------

    @staticmethod
    def price(
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType = "call",
    ) -> float:
        """Compute BSM option price."""
        if np.isscalar(T) and T <= 0:
            intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
            return float(intrinsic)

        d1 = BSM._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return (
                S * np.exp(-q * T) * norm.cdf(d1)
                - K * np.exp(-r * T) * norm.cdf(d2)
            )
        else:
            return (
                K * np.exp(-r * T) * norm.cdf(-d2)
                - S * np.exp(-q * T) * norm.cdf(-d1)
            )

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    @staticmethod
    def greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType = "call",
    ) -> BSMResult:
        """Compute BSM price and all first-order Greeks."""
        d1 = BSM._d1(S, K, T, r, q, sigma)
        d2 = d1 - sigma * np.sqrt(T)

        sqrt_T = np.sqrt(T)
        exp_qT = np.exp(-q * T)
        exp_rT = np.exp(-r * T)

        # --- Delta ---
        if option_type == "call":
            delta = exp_qT * norm.cdf(d1)
        else:
            delta = -exp_qT * norm.cdf(-d1)

        # --- Gamma (call == put) ---
        gamma = exp_qT * norm.pdf(d1) / (S * sigma * sqrt_T)

        # --- Vega (call == put), per 1% vol move ---
        vega = S * exp_qT * norm.pdf(d1) * sqrt_T / 100.0

        # --- Theta, per calendar day ---
        decay = -(S * exp_qT * norm.pdf(d1) * sigma) / (2 * sqrt_T)
        if option_type == "call":
            theta = (
                decay
                - r * K * exp_rT * norm.cdf(d2)
                + q * S * exp_qT * norm.cdf(d1)
            ) / 365.0
        else:
            theta = (
                decay
                + r * K * exp_rT * norm.cdf(-d2)
                - q * S * exp_qT * norm.cdf(-d1)
            ) / 365.0

        # --- Rho, per 1% rate move ---
        if option_type == "call":
            rho = K * T * exp_rT * norm.cdf(d2) / 100.0
        else:
            rho = -K * T * exp_rT * norm.cdf(-d2) / 100.0

        price = BSM.price(S, K, T, r, q, sigma, option_type)

        return BSMResult(
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
        )

    # ------------------------------------------------------------------
    # Implied volatility
    # ------------------------------------------------------------------

    @staticmethod
    def implied_vol(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: OptionType = "call",
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> float:
        """
        Extract implied volatility from a market price via Brent's method.

        Returns np.nan if:
          - T <= 0
          - market_price is below intrinsic value
          - no root found in [0.0001, 5.0]
        """
        if T <= 0:
            return np.nan

        # Intrinsic value check (allow small numerical slack)
        intrinsic = (
            max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
            if option_type == "call"
            else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        )
        if market_price < intrinsic - 1e-4:
            return np.nan

        def objective(sigma: float) -> float:
            return BSM.price(S, K, T, r, q, sigma, option_type) - market_price

        try:
            iv = brentq(objective, 1e-4, 5.0, xtol=tol, maxiter=max_iter)
            return float(iv)
        except (ValueError, RuntimeError):
            return np.nan

    # ------------------------------------------------------------------
    # Put-call parity (validation utility)
    # ------------------------------------------------------------------

    @staticmethod
    def put_call_parity_check(
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        tol: float = 1e-6,
    ) -> bool:
        """
        Verify C - P = S*exp(-qT) - K*exp(-rT).
        Returns True if parity holds within tolerance.
        """
        C = BSM.price(S, K, T, r, q, sigma, "call")
        P = BSM.price(S, K, T, r, q, sigma, "put")
        lhs = C - P
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        return abs(lhs - rhs) < tol
