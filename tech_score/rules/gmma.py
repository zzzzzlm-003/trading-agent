"""GMMA 顾比均线 — 会员第140期.

标准 GMMA:
  短期组 EMA: [3, 5, 8, 10, 12, 15]
  长期组 EMA: [30, 35, 40, 45, 50, 60]
用法 (literal):
  - 短组全部在长组上方 = 多头趋势（买入持有区）
  - 短组全部在长组下方 = 空头趋势（空仓/做空区）
  - 短组穿过长组中心 = 趋势转折信号
"""
from __future__ import annotations
import pandas as pd

NAME = "gmma"
FAMILY = "trend"
EPISODE = 140

_SHORT = [3, 5, 8, 10, 12, 15]
_LONG  = [30, 35, 40, 45, 50, 60]


def signal(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    short_emas = pd.concat([c.ewm(span=n, adjust=False).mean() for n in _SHORT], axis=1)
    long_emas  = pd.concat([c.ewm(span=n, adjust=False).mean() for n in _LONG],  axis=1)
    short_min = short_emas.min(axis=1)
    short_max = short_emas.max(axis=1)
    long_min  = long_emas.min(axis=1)
    long_max  = long_emas.max(axis=1)

    bull = short_min > long_max   # 短组全部 > 长组最大 = 彻底分离向上
    bear = short_max < long_min   # 短组全部 < 长组最小 = 彻底分离向下

    new_bull = bull & ~bull.shift(1).fillna(False)
    new_bear = bear & ~bear.shift(1).fillna(False)

    s = pd.Series(0, index=df.index, dtype=int)
    s[new_bull] = 1
    s[new_bear] = -1
    return s
