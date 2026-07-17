"""Fibonacci 回撤 + 扩展 — 会员第12 & 113期.

用法 (literal):
  - 取最近 60 日 swing high/low 作为 Fib 基准
  - 上升趋势中价格回撤到 0.382 / 0.5 / 0.618 企稳 → 买（每档都是潜在支撑）
  - 上升趋势价格突破 1.272 / 1.618 扩展位 → 卖（获利了结区）
  - 禁区: 横盘/震荡行情 Fib 无效，需有明确趋势
"""
from __future__ import annotations
import pandas as pd

NAME = "fib"
FAMILY = "sr"
EPISODE = 12


def signal(df: pd.DataFrame, window: int = 60) -> pd.Series:
    c = df["Close"].astype(float)
    swing_high = c.rolling(window).max()
    swing_low = c.rolling(window).min()
    rng = swing_high - swing_low
    pos = (c - swing_low) / rng.replace(0, pd.NA)   # 0 = bottom, 1 = top

    # Trend filter: 20 日 MA 斜率
    ma20 = c.rolling(20).mean()
    up = ma20.diff(10) > 0
    dn = ma20.diff(10) < 0

    # 回撤到支撑企稳：连续 2 根 K 线 pos 落在 [0.382, 0.618] 后企稳上行
    in_support = (pos >= 0.382) & (pos <= 0.618)
    reclaim = (c > c.shift(1)) & in_support.shift(1).fillna(False) & up
    buy = reclaim

    # 扩展位回落：触及 pos >= 0.95 (接近 swing high) + 今天下跌
    near_top = pos >= 0.95
    sell = near_top.shift(1).fillna(False) & (c < c.shift(1)) & up

    # 下跌趋势中对称处理
    near_bot = pos <= 0.05
    sell_dn = near_bot.shift(1).fillna(False) & (c < c.shift(1)) & dn
    buy_dn = (pos >= 0.382) & (pos <= 0.618) & dn & (c > c.shift(1)) & reclaim.shift(1).fillna(False)

    s = pd.Series(0, index=df.index, dtype=int)
    s[buy | buy_dn] = 1
    s[sell | sell_dn] = -1
    return s
