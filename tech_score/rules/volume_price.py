"""量价关系 — 会员第10 & 161期.

核心口诀 (literal):
  - 量增价涨 → 多头延续（买）
  - 量缩价跌 → 空头衰竭（反弹机会，弱买）
  - 量增价跌 → 空头延续（卖）
  - 量缩价涨 → 多头衰竭（警惕顶，弱卖）

实现：量 = 成交量 / MA20，价 = close 日变化率
"""
from __future__ import annotations
import pandas as pd

NAME = "volume_price"
FAMILY = "volume"
EPISODE = 10


def signal(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    vol_ratio = v / v.rolling(20, min_periods=1).mean().replace(0, pd.NA)
    ret = c.pct_change()

    big_vol = vol_ratio > 1.5   # 放量
    low_vol = vol_ratio < 0.7   # 缩量
    up = ret > 0.01             # 涨 > 1%
    dn = ret < -0.01            # 跌 > 1%

    s = pd.Series(0, index=df.index, dtype=int)
    s[big_vol & up] = 1    # 量增价涨
    s[low_vol & dn] = 1    # 量缩价跌 = 空头衰竭
    s[big_vol & dn] = -1   # 量增价跌
    s[low_vol & up] = -1   # 量缩价涨 = 多头衰竭
    return s
