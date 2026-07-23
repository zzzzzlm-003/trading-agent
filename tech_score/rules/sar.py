"""Parabolic SAR — 会员第64期.

博主规则 (literal):
  - SAR 点翻转到价格下方 = 多头启动，买
  - SAR 点翻转到价格上方 = 空头启动，卖
"""
from __future__ import annotations
import pandas as pd
import talib

NAME = "sar"
FAMILY = "momentum"
EPISODE = 64


def signal(df: pd.DataFrame) -> pd.Series:
    sar = talib.SAR(df["High"].values, df["Low"].values, acceleration=0.02, maximum=0.2)
    sar = pd.Series(sar, index=df.index)
    close = df["Close"]
    bull = close > sar
    bear = close < sar
    s = pd.Series(0, index=df.index, dtype=int)
    s[bull & ~bull.shift(1).fillna(False)] = 1
    s[bear & ~bear.shift(1).fillna(False)] = -1
    return s
