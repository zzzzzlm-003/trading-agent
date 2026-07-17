"""MA/EMA — 会员第22期.

规则 (literal):
  - 短期 MA5 上穿 MA20 = 买点 (多头排列)
  - 短期 MA5 下穿 MA20 = 卖点 (空头排列)
  - 价格在 MA20 之上时持仓，跌破 MA20 离场（趋势过滤）
"""
from __future__ import annotations
import pandas as pd

NAME = "ma"
FAMILY = "trend"
EPISODE = 22


def signal(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].astype(float)
    ma5 = close.rolling(5, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    gc = (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))
    dc = (ma5 < ma20) & (ma5.shift(1) >= ma20.shift(1))
    s = pd.Series(0, index=df.index, dtype=int)
    s[gc] = 1
    s[dc] = -1
    return s
