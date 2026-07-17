"""Bollinger Bands — 会员第52期.

规则 (literal mean-reversion version):
  - 收盘价触及下轨 (bb_pos < 5%) = 超跌买信号
  - 收盘价触及上轨 (bb_pos > 95%) = 超涨卖信号
  - 参数 20, 2σ
"""
from __future__ import annotations
import pandas as pd

NAME = "boll"
FAMILY = "volatility"
EPISODE = 52


def signal(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    mid = c.rolling(20, min_periods=1).mean()
    std = c.rolling(20, min_periods=1).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    pos = (c - lower) / (upper - lower + 1e-10) * 100
    s = pd.Series(0, index=df.index, dtype=int)
    s[(pos < 5) & (pos.shift(1) >= 5)] = 1
    s[(pos > 95) & (pos.shift(1) <= 95)] = -1
    return s
