"""OBV — 会员第53期.

博主规则 (literal momentum):
  - OBV 20 日均线上穿 OBV 60 日均线 = 量能突破，买
  - OBV 20 日均线下穿 OBV 60 日均线 = 量能衰竭，卖
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NAME = "obv"
FAMILY = "volume"
EPISODE = 53


def signal(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    direction = np.sign(c.diff()).fillna(0)
    obv = (direction * v).cumsum()
    fast = obv.rolling(20, min_periods=1).mean()
    slow = obv.rolling(60, min_periods=1).mean()
    gc = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    dc = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    s = pd.Series(0, index=df.index, dtype=int)
    s[gc] = 1
    s[dc] = -1
    return s
