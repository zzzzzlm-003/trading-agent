"""Ichimoku — 会员第138期.

规则 (literal):
  - 价格在云之上 + 转换线上穿基准线 = 买
  - 价格在云之下 + 转换线下穿基准线 = 卖
  - 参数 9/26/52
"""
from __future__ import annotations
import pandas as pd

NAME = "ichimoku"
FAMILY = "trend"
EPISODE = 138


def signal(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    gc = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    dc = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
    above = c > cloud_top
    below = c < cloud_bot
    s = pd.Series(0, index=df.index, dtype=int)
    s[gc & above] = 1
    s[dc & below] = -1
    return s
