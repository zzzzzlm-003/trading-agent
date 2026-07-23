"""KDJ — 会员第32期.

博主规则 (literal):
  - K 在 20 以下上穿 D = 低位金叉 (买点)
  - K 在 80 以上下穿 D = 高位死叉 (卖点)
  - 参数 9, 3, 3
"""
from __future__ import annotations
import pandas as pd

NAME = "kdj"
FAMILY = "momentum"
EPISODE = 32


def _kdj(df: pd.DataFrame, n=9, m1=3, m2=3):
    h, l, c = df["High"], df["Low"], df["Close"]
    ll = l.rolling(n, min_periods=1).min()
    hh = h.rolling(n, min_periods=1).max()
    rsv = (c - ll) / (hh - ll + 1e-10) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d


def signal(df: pd.DataFrame) -> pd.Series:
    k, d = _kdj(df)
    gc = (k > d) & (k.shift(1) <= d.shift(1))
    dc = (k < d) & (k.shift(1) >= d.shift(1))
    s = pd.Series(0, index=df.index, dtype=int)
    s[gc & (k < 20)] = 1
    s[dc & (k > 80)] = -1
    return s
