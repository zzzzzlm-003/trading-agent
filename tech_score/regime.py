"""Regime detector: classify each bar as 'trending' / 'ranging' / 'bear'.

Used by composite scorer to switch modes:
  - trending: skip mean-reversion rules, ride the trend (let BH run)
  - ranging:  use full composite, multi-indicator confluence shines
  - bear:     composite as defensive exit signal

Classification (all on daily close):
  1. Trend strength: ADX(14) ≥ 25 → strong trend
  2. Direction: close vs MA200 → above = bull, below = bear
  3. Rolling 60-day return > 10% → bull trending; < -10% → bear trending
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib


def classify(df: pd.DataFrame, adx_period: int = 14, ma_period: int = 200,
             mom_window: int = 60, mom_bull: float = 0.10,
             mom_bear: float = -0.10, adx_strong: float = 25) -> pd.Series:
    """Return Series of {'trending_up','trending_down','ranging'} per bar."""
    h = df["High"].astype(float).values
    l = df["Low"].astype(float).values
    c = df["Close"].astype(float).values

    adx = pd.Series(talib.ADX(h, l, c, timeperiod=adx_period), index=df.index)
    close = df["Close"].astype(float)
    ma200 = close.rolling(ma_period, min_periods=1).mean()
    mom = close.pct_change(mom_window)

    bull_trend = (adx >= adx_strong) & (close > ma200) & (mom >= mom_bull)
    bear_trend = (adx >= adx_strong) & (close < ma200) & (mom <= mom_bear)

    r = pd.Series("ranging", index=df.index)
    r[bull_trend] = "trending_up"
    r[bear_trend] = "trending_down"
    return r
