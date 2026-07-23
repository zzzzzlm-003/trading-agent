"""CCI — 会员第63期.

博主规则 (literal):
  - CCI(14) 上穿 -100 = 超卖反弹，买
  - CCI(14) 下穿 +100 = 超买回调，卖
"""
from __future__ import annotations
import pandas as pd
import talib

NAME = "cci"
FAMILY = "momentum"
EPISODE = 63


def signal(df: pd.DataFrame) -> pd.Series:
    cci = talib.CCI(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=14)
    cci = pd.Series(cci, index=df.index)
    s = pd.Series(0, index=df.index, dtype=int)
    s[(cci > -100) & (cci.shift(1) <= -100)] = 1
    s[(cci < 100) & (cci.shift(1) >= 100)] = -1
    return s
