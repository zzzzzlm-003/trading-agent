"""Liquidity / size filters.

Blogger's teaching targets large-cap liquid names (TSLA/AAPL/AMZN-class).
We gate signals on:
  - enough history (≥ `min_bars` days)
  - average 20-day dollar volume ≥ `min_adv_usd` (rough "large-cap liquid" proxy)

Size buckets (for reporting):
  - mega:   ADV ≥ 500M USD
  - large:  ADV ≥ 50M USD
  - mid:    ADV ≥ 5M USD
  - small:  ADV < 5M USD (or untraded bars > 10%)
"""
from __future__ import annotations

import pandas as pd


def avg_dollar_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return (df["Close"].astype(float) * df["Volume"].astype(float)
            ).rolling(window, min_periods=1).mean()


def size_bucket(df: pd.DataFrame, fx_rate_to_usd: float = 1.0) -> str:
    """Rough bucket from latest ADV. `fx_rate_to_usd` to convert local price."""
    adv = avg_dollar_volume(df).iloc[-1] * fx_rate_to_usd
    if adv >= 500e6: return "mega"
    if adv >=  50e6: return "large"
    if adv >=   5e6: return "mid"
    return "small"


def passes_liquidity(df: pd.DataFrame, min_adv_usd: float = 10e6,
                     fx_rate_to_usd: float = 1.0) -> bool:
    if len(df) < 60:
        return False
    adv = avg_dollar_volume(df).iloc[-20:].mean() * fx_rate_to_usd
    return adv >= min_adv_usd


# Crude FX lookup for bucketing local prices into USD. Keep it simple —
# we only need order-of-magnitude for size filtering.
FX_TO_USD = {
    "CNY": 0.14,   # ≈ 7 CNY / USD
    "HKD": 0.13,
    "USD": 1.0,
}


def infer_fx(symbol: str) -> float:
    s = symbol.upper()
    if s.endswith(".SS") or s.endswith(".SZ") or s.endswith(".SH") or (s.isdigit() and len(s) == 6):
        return FX_TO_USD["CNY"]
    if s.endswith(".HK") or s in ("^HSI", "^HSTECH"):
        return FX_TO_USD["HKD"]
    return FX_TO_USD["USD"]
