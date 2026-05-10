"""
Implied volatility surface construction.

Pipeline:
  1. Fetch live options chain (via utils.fetch_options_chain)
  2. Extract implied vols via BSM inversion
  3. Apply arbitrage filters (calendar spread, butterfly)
  4. Fit a smooth surface via 2D cubic spline interpolation
  5. Expose query interface for σ_imp(K, T) at arbitrary points

Design notes:
  - We work in (log-moneyness, T) space for better numerical conditioning.
    Log-moneyness: m = log(K/F) where F = S*exp((r-q)*T) is the forward price.
  - The surface is fit on a regular grid and queried via RectBivariateSpline.
  - Arbitrage filters are heuristic: a full static arbitrage check (Roper 2010)
    is left as a known extension.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple

from .bsm import BSM
from .utils import fetch_options_chain, get_risk_free_rate


class VolSurface:
    """
    Implied volatility surface fitted from live market options data.

    Attributes
    ----------
    ticker     : underlying ticker
    spot       : spot price at snapshot time
    r          : risk-free rate used
    q          : dividend yield used
    raw        : raw cleaned options data with implied vols
    grid_m     : log-moneyness grid nodes
    grid_T     : tenor grid nodes
    iv_grid    : implied vol grid (shape: len(grid_T) x len(grid_m))
    _spline    : fitted RectBivariateSpline object
    """

    def __init__(
        self,
        ticker: str = "SPY",
        q: float = 0.013,   # approximate SPY dividend yield
        min_dte: int = 7,
        max_dte: int = 365,
        min_open_interest: int = 100,
        min_volume: int = 10,
    ):
        self.ticker = ticker
        self.q = q
        self.r = get_risk_free_rate()
        self._fitted = False

        print(f"Fetching options chain for {ticker}...")
        raw = fetch_options_chain(
            ticker=ticker,
            min_dte=min_dte,
            max_dte=max_dte,
            min_open_interest=min_open_interest,
            min_volume=min_volume,
        )
        self.spot = raw["S"].iloc[0]
        self.raw = self._extract_iv(raw)

    # ------------------------------------------------------------------
    # Step 1: implied vol extraction
    # ------------------------------------------------------------------

    def _extract_iv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Invert BSM on each option to get implied vol."""
        ivs = []
        for _, row in data.iterrows():
            iv = BSM.implied_vol(
                market_price=row["mid"],
                S=row["S"],
                K=row["strike"],
                T=row["T"],
                r=self.r,
                q=self.q,
                option_type=row["option_type"],
            )
            ivs.append(iv)

        data = data.copy()
        data["iv"] = ivs

        n_before = len(data)
        data = data.dropna(subset=["iv"])
        data = data[data["iv"] > 0.01]   # floor: 1% vol
        data = data[data["iv"] < 2.00]   # ceiling: 200% vol
        n_after = len(data)

        print(f"IV extraction: {n_before} options → {n_after} valid ({n_before - n_after} dropped)")

        # Compute log-moneyness: log(K/F) where F = S*exp((r-q)*T)
        data["F"] = data["S"] * np.exp((self.r - self.q) * data["T"])
        data["log_moneyness"] = np.log(data["strike"] / data["F"])

        return data

    # ------------------------------------------------------------------
    # Step 2: arbitrage filters
    # ------------------------------------------------------------------

    def _apply_arbitrage_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Heuristic arbitrage filters.

        Calendar spread: for the same log-moneyness, implied total variance
        σ²(m,T)*T must be non-decreasing in T. We flag and remove violations.

        Butterfly: we do not implement a full butterfly check here; this is
        noted as a known limitation.
        """
        n_before = len(data)

        # Total variance: w = iv^2 * T  (must be non-decreasing in T)
        data = data.copy()
        data["total_var"] = data["iv"] ** 2 * data["T"]

        # Group by expiry and check monotonicity crudely:
        # For each strike bucket, sort by T and drop backwards-jumping points
        keep = []
        for strike_bucket, grp in data.groupby(pd.cut(data["log_moneyness"], bins=20)):
            grp_sorted = grp.sort_values("T")
            max_tv = -np.inf
            for idx, row in grp_sorted.iterrows():
                if row["total_var"] >= max_tv - 1e-4:
                    keep.append(idx)
                    max_tv = max(max_tv, row["total_var"])

        data = data.loc[keep]
        n_after = len(data)
        print(f"Arbitrage filter: {n_before} → {n_after} points ({n_before - n_after} removed)")
        return data

    # ------------------------------------------------------------------
    # Step 3: surface fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        n_moneyness: int = 30,
        n_tenors: int = 15,
        smoothing: float = 0.0,
    ) -> "VolSurface":
        """
        Fit a smooth surface via 2D cubic spline in (log-moneyness, T) space.

        Parameters
        ----------
        n_moneyness : number of log-moneyness grid points
        n_tenors    : number of tenor grid points
        smoothing   : spline smoothing factor (0 = interpolating)
        """
        data = self._apply_arbitrage_filters(self.raw)

        # Build a regular grid
        m_min, m_max = data["log_moneyness"].quantile(0.02), data["log_moneyness"].quantile(0.98)
        T_min, T_max = data["T"].min(), data["T"].max()

        self.grid_m = np.linspace(m_min, m_max, n_moneyness)
        self.grid_T = np.linspace(T_min, T_max, n_tenors)

        # Interpolate scattered data onto the regular grid using linear griddata
        points = data[["log_moneyness", "T"]].values
        values = data["iv"].values

        grid_M, grid_T2 = np.meshgrid(self.grid_m, self.grid_T)
        iv_grid_raw = griddata(
            points, values,
            (grid_M, grid_T2),
            method="linear",
        )

        # Fill any remaining NaNs with nearest-neighbour
        mask = np.isnan(iv_grid_raw)
        if mask.any():
            iv_grid_nn = griddata(points, values, (grid_M, grid_T2), method="nearest")
            iv_grid_raw[mask] = iv_grid_nn[mask]

        self.iv_grid = gaussian_filter(iv_grid_raw, sigma=1.0)

        # Fit spline on the regular grid
        self._spline = RectBivariateSpline(
            self.grid_T, self.grid_m, self.iv_grid, kx=3, ky=3, s=smoothing
        )

        self._fitted = True
        print(f"Surface fitted: {n_tenors} tenors × {n_moneyness} moneyness nodes")
        return self

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def iv(self, K: float, T: float) -> float:
        """
        Query implied vol at strike K and tenor T.
        K is absolute strike; internally converted to log-moneyness.
        """
        self._check_fitted()
        F = self.spot * np.exp((self.r - self.q) * T)
        m = np.log(K / F)
        return float(self._spline(T, m)[0, 0])

    def iv_grid_query(
        self,
        moneyness_range: Tuple[float, float] = (-0.2, 0.2),
        T_range: Tuple[float, float] = None,
        n_m: int = 50,
        n_T: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a dense grid of implied vols for plotting.

        Returns
        -------
        M   : log-moneyness grid (n_T x n_m)
        T   : tenor grid         (n_T x n_m)
        IV  : implied vol grid   (n_T x n_m)
        """
        self._check_fitted()
        if T_range is None:
            T_range = (self.grid_T.min(), self.grid_T.max())

        m_grid = np.linspace(*moneyness_range, n_m)
        T_grid = np.linspace(*T_range, n_T)
        M, T = np.meshgrid(m_grid, T_grid)
        IV = self._spline(T_grid, m_grid)
        return M, T, IV

    # ------------------------------------------------------------------
    # Smile slice
    # ------------------------------------------------------------------

    def smile(
        self, T_target: float, n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the vol smile (strike, IV) at a given tenor.

        T_target is matched to the nearest available tenor.
        """
        self._check_fitted()
        T_actual = self.grid_T[np.argmin(np.abs(self.grid_T - T_target))]
        m_range = np.linspace(self.grid_m.min(), self.grid_m.max(), n_points)
        iv_vals = self._spline(T_actual, m_range).flatten()

        F = self.spot * np.exp((self.r - self.q) * T_actual)
        strikes = F * np.exp(m_range)
        return strikes, iv_vals

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_surface(self, save_path: Optional[str] = None) -> None:
        """3D surface plot of implied volatility."""
        self._check_fitted()
        M, T, IV = self.iv_grid_query()

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(M, T, IV * 100, cmap="RdYlGn_r", alpha=0.85)

        ax.set_xlabel("Log-Moneyness log(K/F)", labelpad=10)
        ax.set_ylabel("Tenor (years)", labelpad=10)
        ax.set_zlabel("Implied Vol (%)", labelpad=10)
        ax.set_title(f"{self.ticker} Implied Volatility Surface", fontsize=14, pad=20)
        fig.colorbar(surf, ax=ax, shrink=0.5, label="IV (%)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    def plot_smiles(
        self,
        tenors_days: list = [30, 60, 90, 180],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Overlay smile slices at multiple tenors.
        Highlights the skew structure across the term structure.
        """
        self._check_fitted()
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(tenors_days)))

        for T_days, color in zip(tenors_days, colors):
            T = T_days / 365.0
            if T < self.grid_T.min() or T > self.grid_T.max():
                continue
            strikes, iv = self.smile(T)
            moneyness = strikes / self.spot
            ax.plot(moneyness, iv * 100, color=color, label=f"{T_days}d", linewidth=2)

        ax.axvline(1.0, color="grey", linestyle="--", alpha=0.5, label="ATM")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Volatility (%)")
        ax.set_title(f"{self.ticker} Vol Smile — Multiple Tenors")
        ax.legend(title="Tenor")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Surface not fitted yet. Call .fit() first.")
