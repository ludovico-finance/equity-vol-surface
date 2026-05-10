# Data

No raw data files are committed to this repository.

## Sources

| Data | Source | Access |
|------|--------|--------|
| Options chain (SPY) | yfinance | Free, via `yf.Ticker.option_chain()` |
| Spot prices | yfinance | Free, via `yf.Ticker.history()` |
| Risk-free rate | yfinance (^IRX, 3M T-bill) | Free |
| VIX / VIX3M | yfinance (^VIX, ^VIX3M) | Free |

## Known Limitations

- **yfinance options data**: bid/ask mid-prices are used as market price proxies.
  Spreads can be wide for illiquid strikes, introducing noise in implied vol extraction.
- **Far-dated expiries**: yfinance may have gaps or stale quotes beyond 6 months.
  The moneyness and liquidity filters in `utils.fetch_options_chain` mitigate this.
- **Historical options data**: yfinance does not provide historical options chains.
  For a proper historical backtest, a paid data source (CBOE DataShop, OptionMetrics,
  Interactive Brokers) would be required.
- **Dividend yield**: approximated as a constant `q` input. In production, a
  term structure of dividend expectations would be used.

## For Reproducibility

All figures are generated deterministically from live market data at run time.
Output may differ from the committed figures due to market moves.
To exactly reproduce the figures in the README, note the date in the figure filenames.
