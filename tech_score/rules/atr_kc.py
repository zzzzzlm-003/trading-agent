"""Keltner Channel + ATR — 会员第71期.

博主改良版 (literal):
  - 中轨 EMA20、上下轨 ±2·ATR(14)
  - 博主原话："不要机械地下轨买、上轨卖；通道是观察点"
  - Buy: 上升通道（中轨上斜）+ 价格回测中轨企稳
  - Sell: 上升通道 + 价格触上轨且 3 根 K 线内中轨破位
"""
from __future__ import annotations
import pandas as pd
import talib

NAME = "atr_kc"
FAMILY = "volatility"
EPISODE = 71


def signal(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    atr = pd.Series(talib.ATR(h, l, c, timeperiod=14), index=df.index)
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    upper = ema20 + 2 * atr
    # lower = ema20 - 2 * atr  # not used in blogger's rule
    close = df["Close"]

    # 上升通道：中轨 5 日斜率 > 0
    slope_up = ema20.diff(5) > 0

    # 触上轨日
    touch_up = close >= upper

    # 中轨破位（收盘 < 中轨）
    break_mid = close < ema20

    # Buy: 上升通道 + 当日回测中轨企稳（close 刚上穿 EMA20）
    reclaim_mid = (close > ema20) & (close.shift(1) <= ema20.shift(1))
    buy = slope_up & reclaim_mid

    # Sell: 上升通道 + 过去 3 日触过上轨 + 今天中轨破位
    touched_recent = touch_up.rolling(3).max() > 0
    sell = slope_up & touched_recent & break_mid & ~break_mid.shift(1).fillna(False)

    s = pd.Series(0, index=df.index, dtype=int)
    s[buy] = 1
    s[sell] = -1
    return s
