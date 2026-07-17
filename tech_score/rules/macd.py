"""MACD — 规则抽取自会员第31期.

规则要点：
  1. 只用日线及以上级别（本模块不提供 intraday）
  2. 先看 DIF / DEA 相对零轴的位置 → 定"大趋势"
     - 都 > 0  → 多头市场 (bull)，进攻思维
     - 都 < 0  → 空头市场 (bear)，防御思维；金叉只能当反弹
     - 一正一负 → 趋势未定 (mixed)，观望不参考
  3. 再看 DIF / DEA 交叉 → 定"短趋势"
     - 金叉 (DIF 上穿 DEA)：零轴上方=正信号；零轴下方=弱反弹信号
     - 死叉 (DIF 下穿 DEA)：零轴上方=高位死叉，不能买；零轴下方=强空信号
  4. 柱子仅作动能参考（可以不看）
  5. 背离另算（本模块未实现，留给 macd_divergence.py）
  6. 参数 12/26/9 默认，不调

signal(df) 返回 {-1, 0, +1} 的 pd.Series（每根 K 线一个信号）：
  +1 = 规则下"可买"
  -1 = 规则下"明确空 / 卖"
   0 = 观望

NOTE: signal 是基于当日收盘后的状态（计算完指标再发信号），回测引擎需要在
      次日开盘执行以避免 look-ahead bias。vectorbt 的 from_signals 默认下一根 K 线成交。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "macd"
FAMILY = "momentum"
EPISODE = 31


def compute(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_p: int = 9):
    """Return DIF, DEA, HIST series (用默认 12/26/9)."""
    close = df["Close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal_p, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def regime(dif: pd.Series, dea: pd.Series) -> pd.Series:
    """'bull' / 'bear' / 'mixed'."""
    bull = (dif > 0) & (dea > 0)
    bear = (dif < 0) & (dea < 0)
    r = pd.Series(index=dif.index, dtype=object)
    r[:] = "mixed"
    r[bull] = "bull"
    r[bear] = "bear"
    return r


def signal(df: pd.DataFrame) -> pd.Series:
    """ MACD 规则 → {-1, 0, +1} per bar."""
    dif, dea, hist = compute(df)
    reg = regime(dif, dea)

    # crosses (today's state vs yesterday's)
    golden = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    death  = (dif < dea) & (dif.shift(1) >= dea.shift(1))

    s = pd.Series(0, index=df.index, dtype=int)

    # bull + golden → strong buy
    s[(reg == "bull") & golden] = 1
    # bull + death (高位死叉) → 不能买，但原规则说"该持的持"，不强制卖
    # 保持 0；如果用户偏保守可改为 -1
    # bear + death → 强确认空
    s[(reg == "bear") & death] = -1
    # bear + golden → 原规则说只能当反弹，这里记 +1 但幅度弱（由 weighting 决定）
    # 为了避免重仓空头市场反弹，记 0（观望）
    # mixed → 0

    return s


def signal_with_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Return signal + intermediate state for inspection/plotting."""
    dif, dea, hist = compute(df)
    reg = regime(dif, dea)
    golden = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    death  = (dif < dea) & (dif.shift(1) >= dea.shift(1))
    out = pd.DataFrame({
        "dif": dif, "dea": dea, "hist": hist,
        "regime": reg, "golden": golden, "death": death,
        "signal": signal(df),
    })
    return out
