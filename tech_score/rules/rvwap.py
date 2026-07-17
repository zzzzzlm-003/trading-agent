"""Rolling VWAP (5 日) — 会员第270期.

用法 (literal):
  - 5 日 rolling VWAP 作为机构近期成本线
  - Buy: 价格从 VWAP 下方上穿 VWAP + VWAP 当日斜率 > 0
  - Sell: 价格从 VWAP 上方跌破 VWAP + 连续 3 日保持下方
  - 注意："VWAP 是滞后线，不能作为先行信号"
"""
from __future__ import annotations
import pandas as pd

NAME = "rvwap"
FAMILY = "volume"
EPISODE = 270


def signal(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    v = df["Volume"].astype(float)
    num = (tp * v).rolling(5).sum()
    den = v.rolling(5).sum().replace(0, pd.NA)
    vwap = (num / den).astype(float)
    slope_up = vwap.diff() > 0
    close = df["Close"]

    up_cross = (close > vwap) & (close.shift(1) <= vwap.shift(1))
    dn_cross = (close < vwap) & (close.shift(1) >= vwap.shift(1))
    below_3 = (close < vwap).rolling(3).sum() >= 3

    s = pd.Series(0, index=df.index, dtype=int)
    s[up_cross & slope_up] = 1
    s[dn_cross & below_3] = -1
    return s
