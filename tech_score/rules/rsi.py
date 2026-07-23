"""RSI — 会员第30期.

博主规则 (literal):
  - 使用周期 6 (快线) 为主，日线及以上
  - RSI6 < 20 = 超卖反弹机会 (弱买信号)
  - RSI6 > 80 = 超买回调风险 (弱卖信号)
  - 背离 (顶/底) 博主强调但 literal 回测仅用 20/80 阈值
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NAME = "rsi"
FAMILY = "momentum"
EPISODE = 30


def _rsi(close: pd.Series, period: int = 6) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def signal(df: pd.DataFrame) -> pd.Series:
    rsi6 = _rsi(df["Close"], 6)
    s = pd.Series(0, index=df.index, dtype=int)
    # Cross into oversold → buy; cross out of overbought → sell
    s[(rsi6 < 20) & (rsi6.shift(1) >= 20)] = 1
    s[(rsi6 > 80) & (rsi6.shift(1) <= 80)] = -1
    return s
