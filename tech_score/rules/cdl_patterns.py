"""K 线形态 - TA-Lib 61 个 CDL_* 全部打包 (family=pattern).

博主会员合集里讲过大量单期形态 (岛形反转/钻石顶/头肩/W底/M顶/旗形/杯柄 等)，
都属于 K 线组合。TA-Lib 已经覆盖了这些经典形态的识别算法 (每个返回 -100/0/+100)。

我们把 61 个形态作为一个"形态族":
  - 当日所有看多形态数 - 当日所有看空形态数 → 净看多度
  - 净 ≥ 2 → buy (多个形态同向确认)
  - 净 ≤ -2 → sell
  - 否则 0
这样避免单个形态噪声，取 consensus。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

NAME = "cdl_patterns"
FAMILY = "pattern"
EPISODE = 0  # 覆盖博主的多期 K 线形态专题 (110/112/40/47/18/41/150/151 等)

# 所有 CDL_* 函数
_CDL_FUNCS = [fn for fn in talib.get_function_groups()["Pattern Recognition"]]


def signal(df: pd.DataFrame) -> pd.Series:
    o = df["Open"].astype(float).values
    h = df["High"].astype(float).values
    l = df["Low"].astype(float).values
    c = df["Close"].astype(float).values

    net = np.zeros(len(df), dtype=int)
    for name in _CDL_FUNCS:
        try:
            fn = getattr(talib, name)
            out = fn(o, h, l, c)
            # TA-Lib CDL returns +100 / -100 / 0
            net += np.sign(out).astype(int)
        except Exception:
            continue

    s = pd.Series(0, index=df.index, dtype=int)
    s[net >= 2] = 1
    s[net <= -2] = -1
    return s
