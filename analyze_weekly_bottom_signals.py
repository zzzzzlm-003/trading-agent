"""周线底部信号 — 在A股大盘指数上跑三个信号，看历史触发点 + 触发后走势。

信号1：周线 MACD/RSI 双确认底背离  —— 直接复用 tech_score/rules/divergence.py
信号2：神奇九转 / DeMark Setup 9   —— 直接复用 tech_score/rules/demark.py
信号3：颜色序列"红绿红绿绿绿绿红"（8根K线，最后一根=当周）—— 本脚本新写，一次性验证用

用法：
    python analyze_weekly_bottom_signals.py [SYMBOL]

SYMBOL 默认 000001（上证指数）。如果你能连 MerQube 的数据库，把下面
`load_index_ohlc()` 里标 TODO 的那段换成你自己的查询，返回的 DataFrame
只要满足：DatetimeIndex（日线或已经是周线都行）+ 列名 Open/High/Low/Close
（Volume 可选，仅用于展示，不参与信号计算），下面的逻辑不用改。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent  # 如果脚本跟 tech_score 不在同一层，改成实际仓库路径
sys.path.insert(0, str(REPO_ROOT))

from tech_score.rules import divergence, demark  # noqa: E402


# ────────────────────────────────────────────────────────────────
# 1. 数据加载 —— 换成你自己的 MerQube 查询就行
# ────────────────────────────────────────────────────────────────
def load_index_ohlc(symbol: str = "000001") -> pd.DataFrame:
    """返回日线 OHLC，DatetimeIndex，列名 Open/High/Low/Close(/Volume)。"""
    # ---- TODO: 换成 MerQube 数据库查询，示例（伪代码，按你实际的表结构改）----
    # import your_merqube_client as mq
    # df = mq.query(f"SELECT date, open, high, low, close FROM index_daily WHERE symbol='{symbol}'")
    # df = df.rename(columns={"date": "Date", "open": "Open", "high": "High",
    #                          "low": "Low", "close": "Close"})
    # return df.set_index("Date").sort_index()

    # ---- 默认走仓库原有逻辑（akshare/yfinance），在能连外网的环境里直接能用 ----
    from tech_score.data import fetch
    return fetch(symbol, period="max")


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    w = df.resample("W-FRI").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    return w


# ────────────────────────────────────────────────────────────────
# 2. 信号3：颜色序列检测（新写，本脚本独有，不进 tech_score/rules）
# ────────────────────────────────────────────────────────────────
def color_sequence_signal(df: pd.DataFrame, sequence: str = "RGRGGGGR") -> pd.Series:
    """sequence 里 R=红(跌)/G=绿(涨)，最后一个字符对应"当周"。

    默认 "RGRGGGGR" 对应用户说的"红绿红绿绿绿绿红"（8周）。
    """
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    color = np.where(close >= open_, "G", "R")  # G=涨(绿) R=跌(红)
    color = pd.Series(color, index=df.index)

    n = len(sequence)
    hit = pd.Series(False, index=df.index)
    for i in range(n - 1, len(color)):
        window = "".join(color.iloc[i - n + 1: i + 1])
        if window == sequence:
            hit.iloc[i] = True
    return hit


# ────────────────────────────────────────────────────────────────
# 3. 触发后走势统计
# ────────────────────────────────────────────────────────────────
FORWARD_WEEKS = [4, 12, 26, 52]


def forward_returns(close: pd.Series, trigger_dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for d in trigger_dates:
        i = close.index.get_loc(d)
        row = {"date": d.date()}
        for w in FORWARD_WEEKS:
            j = i + w
            if j < len(close):
                row[f"+{w}w_%"] = round((close.iloc[j] / close.iloc[i] - 1) * 100, 1)
            else:
                row[f"+{w}w_%"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def report(name: str, close: pd.Series, sig: pd.Series):
    dates = sig[sig != 0].index if sig.dtype != bool else sig[sig].index
    print(f"\n{'='*60}\n{name} — 共触发 {len(dates)} 次\n{'='*60}")
    if len(dates) == 0:
        print("（历史上没有触发过）")
        return
    tbl = forward_returns(close, dates)
    print(tbl.to_string(index=False))


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "000001"
    print(f"加载 {symbol} 日线数据...")
    daily = load_index_ohlc(symbol)
    print(f"日线 {len(daily)} 条，{daily.index.min().date()} ~ {daily.index.max().date()}")

    weekly = to_weekly(daily)
    print(f"重采样为周线 {len(weekly)} 条")

    close = weekly["Close"].astype(float)

    sig_div = divergence.signal(weekly)          # +1 = 底背离
    sig_demark = demark.signal(weekly)            # +1 = 下跌九转完成
    sig_colorseq = color_sequence_signal(weekly)  # True = 命中颜色序列

    report("信号1：周线 MACD/RSI 底背离", close, sig_div[sig_div == 1])
    report("信号2：神奇九转（周线，下跌 setup 完成）", close, sig_demark[sig_demark == 1])
    report("信号3：颜色序列 红绿红绿绿绿绿红 (RGRGGGGR)", close, sig_colorseq)


if __name__ == "__main__":
    main()
