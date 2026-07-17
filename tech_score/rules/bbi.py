"""BBI 多空指标 — 会员第67期.

BBI = (MA3 + MA6 + MA12 + MA24) / 4
规则 (literal):
  - 价格上穿 BBI = 买
  - 价格下穿 BBI = 卖
"""
from __future__ import annotations
import pandas as pd

NAME = "bbi"
FAMILY = "trend"
EPISODE = 67


def signal(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    bbi = (c.rolling(3).mean() + c.rolling(6).mean()
           + c.rolling(12).mean() + c.rolling(24).mean()) / 4
    up = (c > bbi) & (c.shift(1) <= bbi.shift(1))
    dn = (c < bbi) & (c.shift(1) >= bbi.shift(1))
    s = pd.Series(0, index=df.index, dtype=int)
    s[up] = 1
    s[dn] = -1
    return s
