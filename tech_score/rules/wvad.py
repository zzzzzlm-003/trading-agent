"""WVAD 威廉变异离散量 — 会员第147期.

改良版 (literal):
  WVAD[i] = (Close - Open) / (High - Low) × Volume
  WVAD_sum = rolling(24) WVAD    （常用周期）
  - 下降趋势中: WVAD 由负转正 + K 线见底形态 → 买
  - 上升趋势中: WVAD 由正转负 + K 线见顶形态 → 卖
  - 禁区: 横盘震荡中零轴附近上下穿越 → 空信号（原规则说："小震不动作"）
"""
from __future__ import annotations
import pandas as pd

NAME = "wvad"
FAMILY = "volume"
EPISODE = 147


def signal(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    v = df["Volume"].astype(float)
    denom = (h - l).replace(0, pd.NA)
    wvad_bar = ((c - o) / denom).fillna(0) * v
    wvad = wvad_bar.rolling(24, min_periods=24).sum()

    # 趋势过滤：20 日收盘斜率定方向
    ma20 = c.rolling(20, min_periods=1).mean()
    trend_down = c < ma20.shift(20).fillna(c.iloc[0])
    trend_up   = c > ma20.shift(20).fillna(c.iloc[0])

    cross_up = (wvad > 0) & (wvad.shift(1) <= 0)
    cross_dn = (wvad < 0) & (wvad.shift(1) >= 0)

    # 只在明确趋势中取信号
    s = pd.Series(0, index=df.index, dtype=int)
    s[cross_up & trend_down] = 1   # 下跌中 WVAD 由负转正 = 止跌
    s[cross_dn & trend_up] = -1    # 上涨中 WVAD 由正转负 = 见顶
    return s
