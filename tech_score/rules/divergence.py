"""RSI + MACD 背离 — 会员第30/31期都强调背离是关键信号.

 literal 用法:
  - 顶背离：价格新高 + 指标不新高 → 卖（回调预警）
  - 底背离：价格新低 + 指标不新低 → 买（反弹预警）
  - 窗口取 30 日内的 swing high/low 对比

实现：检测过去 30 日内的价格极值 vs 同期指标极值是否同步。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NAME = "divergence"
FAMILY = "momentum"
EPISODE = 30  # RSI 期；MACD 在 31 期也有


def _rsi(c: pd.Series, p=14) -> pd.Series:
    d = c.diff()
    g = d.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    loss = (-d).clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    rs = g / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _macd_hist(c: pd.Series) -> pd.Series:
    ef = c.ewm(span=12, adjust=False).mean()
    es = c.ewm(span=26, adjust=False).mean()
    dif = ef - es
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif - dea


def _divergence(price: pd.Series, ind: pd.Series, window: int = 30):
    # Price new extreme in window but indicator not matching
    price_hi = price.rolling(window).max() == price
    price_lo = price.rolling(window).min() == price
    ind_hi = ind.rolling(window).max() == ind
    ind_lo = ind.rolling(window).min() == ind
    top_div = price_hi & ~ind_hi
    bot_div = price_lo & ~ind_lo
    return top_div, bot_div


def signal(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    rsi = _rsi(c)
    hist = _macd_hist(c)

    rsi_top, rsi_bot = _divergence(c, rsi)
    macd_top, macd_bot = _divergence(c, hist)

    # Need BOTH RSI and MACD to confirm divergence (reduce noise)
    both_top = rsi_top & macd_top
    both_bot = rsi_bot & macd_bot

    s = pd.Series(0, index=df.index, dtype=int)
    s[both_bot] = 1
    s[both_top] = -1
    return s
