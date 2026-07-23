"""DeMark 9-13 — 会员第127期.

博主改良版 (literal):
  - Setup 9: 连续 9 根 K 线 close < close[-4]（下跌 setup）或 > close[-4]（上涨 setup）
  - 博主原话："到 9 就可以进场，不用等 13" + K 线翻转形态确认
  - 为避免 look-ahead，我们在 Setup 9 完成当天发信号
  - Buy: 下跌 setup 完成到 9 (超卖反转)
  - Sell: 上涨 setup 完成到 9 (超买反转)
  - 博主禁区：30 min 以下级别，博主不用 13 做止损（会被打掉）
"""
from __future__ import annotations
import pandas as pd

NAME = "demark"
FAMILY = "momentum"
EPISODE = 127


def signal(df: pd.DataFrame) -> pd.Series:
    c = df["Close"].astype(float)
    c4 = c.shift(4)
    down_bar = (c < c4).astype(int)  # bearish setup bar
    up_bar   = (c > c4).astype(int)  # bullish setup bar

    # Count consecutive runs of down_bar / up_bar
    def _run(x: pd.Series) -> pd.Series:
        grp = (x != x.shift()).cumsum()
        return x.groupby(grp).cumsum() * x  # resets when condition breaks

    down_run = _run(down_bar)
    up_run = _run(up_bar)

    s = pd.Series(0, index=df.index, dtype=int)
    s[down_run == 9] = 1   # 下跌 setup 完成 → 买（超卖反转）
    s[up_run == 9] = -1    # 上涨 setup 完成 → 卖（超买反转）
    return s
