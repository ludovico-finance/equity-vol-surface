"""
Local volatility surface via Dupire's equation.

Dupire (1994) showed that given a complete set of European call prices C(K,T),
the unique diffusion consistent with market prices has local vol:

    σ²_loc(K,T) = [∂C/∂T + (r−q)K ∂C/∂K + qC] / [½ K² ∂²C/∂K²]

In practice:
  - We differentiate the *fitted implied vol surface* numerically, not raw prices.
  - This avoids noise amplification from differentiating scattered market quotes.
  - The numerator involves the Breeden-Litzenberger formula denominator implicitly.

We implement the vol-surface form of Dupire (working in (K,T) space directly),
which is numerically more stable than differentiating call prices.

Reference: Gatheral (2006), "The Volatility Surface", Chapter 1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from .surface import VolSurface
from .bsm import BSM


class LocalVolSurface:
    """
    Dupire local volatility surface derived from a fitted VolSurface.

    The local vol is computed via numerical differentiation of the
    fitted implied vol spline. We use central differences with a
    small step size calibrated to the surface scale.

    Attributes
    ----------
    iv_surface : the underlying VolSurface object
    """

    def __init__(self, iv_surface: VolSurface):
        if not iv_surface._fitted:
            raise ValueError("VolSurface must be fitted before computing local vol.")
        self.iv_surface = iv_surface
        self.spot = iv_surface.spot
        self.r = iv_surface.r
        self.q = iv_surface.q

    # ------------------------------------------------------------------
    # Core Dupire formula
    # ------------------------------------------------------------------

    def local_vol(self, K: float, T: float) -> float:
        """
        Compute local vol σ_loc(K, T) via Dupire's equation.

        Uses numerical differentiation of the implied vol spline to
        compute ∂σ_imp/∂T, ∂σ_imp/∂K, ∂²σ_imp/∂K².

        Then applies the full Dupire formula in terms of implied vol
        (Gatheral 2006, eq. 1.4):

            σ²_loc = σ²_imp * [1 + ...] / denominator

        For cleaner implementation we use the call-price differentiation
        form, computing call prices from the IV surface.

        Returns np.nan if denominator is non-positive (arbitrage region).
        """
        # Step sizes for numerical differentiation
        dK = K * 0.02     # 2% of strike
        dT = max(T * 0.02, 1 / 365.0)  # 2% of tenor, min 1 day

        iv0 = self.iv_surface.iv(K, T)
        if np.isnan(iv0):
            return np.nan

        # Call prices via BSM at perturbed points
        def call(k, t):
            sig = self.iv_surface.iv(k, t)
            if np.isnan(sig) or sig <= 0:
                return np.nan
            return BSM.price(self.spot, k, t, self.r, self.q, sig, "call")

        C0 = call(K, T)
        if np.isnan(C0):
            return np.nan

        # --- ∂C/∂T (forward difference — can't go below T=0) ---
        T_up = T + dT
        C_T_up = call(K, T_up)
        if np.isnan(C_T_up):
            dCdT = np.nan
        else:
            dCdT = (C_T_up - C0) / dT

        # --- ∂C/∂K (central difference) ---
        C_K_up = call(K + dK, T)
        C_K_dn = call(K - dK, T)
        if np.isnan(C_K_up) or np.isnan(C_K_dn):
            return np.nan
        dCdK = (C_K_up - C_K_dn) / (2 * dK)

        # --- ∂²C/∂K² (central second difference) ---
        d2CdK2 = (C_K_up - 2 * C0 + C_K_dn) / dK ** 2

        if np.isnan(dCdT):
            return np.nan

        # --- Dupire numerator and denominator ---
        numerator = dCdT + (self.r - self.q) * K * dCdK + self.q * C0
        denominator = 0.5 * K ** 2 * d2CdK2

        if denominator <= 1e-8:
            # Non-positive denominator signals arbitrage or surface edge
            return np.nan

        local_var = numerator / denominator
        if local_var <= 0:
            return np.nan

        return float(np.sqrt(local_var))

    # ------------------------------------------------------------------
    # Grid computation
    # ------------------------------------------------------------------

    def compute_grid(
        self,
        n_strikes: int = 30,
        n_tenors: int = 15,
        moneyness_range: Tuple[float, float] = (0.85, 1.10),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute local vol over a (strike, tenor) grid.

        Returns
        -------
        K_grid : (n_tenors x n_strikes) strike grid
        T_grid : (n_tenors x n_strikes) tenor grid
        LV     : (n_tenors x n_strikes) local vol grid (NaN in arbitrage regions)
        """
        iv_surf = self.iv_surface
        strikes = np.linspace(
            moneyness_range[0] * self.spot,
            moneyness_range[1] * self.spot,
            n_strikes,
        )
        tenors = np.linspace(iv_surf.grid_T.min(), iv_surf.grid_T.max(), n_tenors)

        K_grid, T_grid = np.meshgrid(strikes, tenors)
        LV = np.zeros_like(K_grid)

        for i in range(n_tenors):
            for j in range(n_strikes):
                LV[i, j] = self.local_vol(K_grid[i, j], T_grid[i, j])

        return K_grid, T_grid, LV

    # ------------------------------------------------------------------
    # Comparison utility
    # ------------------------------------------------------------------

    def compare_iv_lv(
        self,
        T_target: float,
        n_points: int = 60,
        moneyness_range: Tuple[float, float] = (0.85, 1.10),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot implied vol vs local vol at a given tenor.

        The local vol smile is typically steeper than the implied vol smile,
        consistent with the Dupire relationship: σ_loc ≈ 2σ_imp - σ_atm
        in the limit of small skew (Derman & Kani 1994 approximation).
        """
        iv_surf = self.iv_surface
        T_actual = iv_surf.grid_T[np.argmin(np.abs(iv_surf.grid_T - T_target))]

        # Sample wider, then filter to liquid moneyness range
        strikes_full, iv_full = iv_surf.smile(T_actual, n_points=n_points * 2)
        moneyness_full = strikes_full / self.spot
        mask = (moneyness_full >= moneyness_range[0]) & (moneyness_full <= moneyness_range[1])
        strikes = strikes_full[mask]
        iv_vals = iv_full[mask]
        lv_vals = np.array([self.local_vol(K, T_actual) for K in strikes])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(strikes / self.spot, iv_vals * 100, label="Implied Vol", lw=2, color="#2196F3")
        ax.plot(
            strikes / self.spot,
            lv_vals * 100,
            label="Local Vol (Dupire)",
            lw=2,
            linestyle="--",
            color="#F44336",
        )
        ax.axvline(1.0, color="grey", linestyle=":", alpha=0.6, label="ATM")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Volatility (%)")
        ax.set_title(
            f"{iv_surf.ticker} — Implied vs Local Vol  |  T ≈ {T_actual*365:.0f}d",
            fontsize=13,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def plot_local_vol_surface(
        self,
        n_strikes: int = 25,
        n_tenors: int = 12,
        save_path: Optional[str] = None,
    ) -> None:
        """3D plot of the local vol surface."""
        K_grid, T_grid, LV = self.compute_grid(n_strikes=n_strikes, n_tenors=n_tenors)
        moneyness = K_grid / self.spot

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            moneyness, T_grid, LV * 100, cmap="RdYlGn_r", alpha=0.85
        )
        ax.set_xlabel("Moneyness (K/S)", labelpad=10)
        ax.set_ylabel("Tenor (years)", labelpad=10)
        ax.set_zlabel("Local Vol (%)", labelpad=10)
        ax.set_title(
            f"{self.iv_surface.ticker} Local Volatility Surface (Dupire)", fontsize=14, pad=20
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, label="Local Vol (%)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()
