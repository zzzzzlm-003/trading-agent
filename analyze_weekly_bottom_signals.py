"""底部信号 — 在A股大盘指数上跑三个信号，支持日/周/月三个周期，看历史触发点 + 触发后走势。

信号1：MACD/RSI 双确认底背离  —— 直接复用 tech_score/rules/divergence.py
信号2：神奇九转 / DeMark Setup 9 —— 直接复用 tech_score/rules/demark.py
信号3：阳/阴 K线序列（本脚本新写，一次性验证用）

用法：
    python analyze_weekly_bottom_signals.py [SYMBOL] [--timeframe D|W|M] [--sequence YNYNNNY]

不传 --timeframe 时，默认把 D/W/M 三个周期各跑一遍。
--sequence 用 Y(阳/涨，Close>=Open) / N(阴/跌，Close<Open) 描述一段K线序列，
最后一个字符对应"当期"。默认 "YNYNNNY" = 阳阴阳阴阴阴阳。

SYMBOL 默认 000001（上证指数）。如果你能连 MerQube 的数据库，把下面
`load_index_ohlc()` 里标 TODO 的那段换成你自己的查询，返回的 DataFrame
只要满足：DatetimeIndex（日线） + 列名 Open/High/Low/Close
（Volume 可选，仅用于展示，不参与信号计算），下面的逻辑不用改。
"""
from __future__ import annotations

import argparse
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


# ────────────────────────────────────────────────────────────────
# 2. 周期重采样：日 / 周 / 月
# ────────────────────────────────────────────────────────────────
_RESAMPLE_RULE = {"D": None, "W": "W-FRI", "M": "ME"}


def to_period(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = _RESAMPLE_RULE[timeframe]
    if rule is None:
        return df.sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    return df.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])


# ────────────────────────────────────────────────────────────────
# 3. 信号3：阳/阴 K线序列检测（新写，本脚本独有，不进 tech_score/rules）
# ────────────────────────────────────────────────────────────────
def sequence_signal(df: pd.DataFrame, sequence: str) -> pd.Series:
    """sequence 用 Y(阳/涨) / N(阴/跌) 描述，最后一个字符对应"当期"。"""
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    label = pd.Series(np.where(close >= open_, "Y", "N"), index=df.index)  # Y=阳 N=阴

    n = len(sequence)
    hit = pd.Series(False, index=df.index)
    for i in range(n - 1, len(label)):
        window = "".join(label.iloc[i - n + 1: i + 1])
        if window == sequence:
            hit.iloc[i] = True
    return hit


# ────────────────────────────────────────────────────────────────
# 4. 触发后走势统计（按周期调整往后看多少根K线）
# ────────────────────────────────────────────────────────────────
FORWARD_PERIODS = {
    "D": [20, 60, 120, 250],   # 约等于 4/12/26/52 周的交易日数
    "W": [4, 12, 26, 52],
    "M": [1, 3, 6, 12],
}


def forward_returns(close: pd.Series, trigger_dates: pd.DatetimeIndex, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for d in trigger_dates:
        i = close.index.get_loc(d)
        row = {"date": d.date()}
        for h in horizons:
            j = i + h
            if j < len(close):
                row[f"+{h}_%"] = round((close.iloc[j] / close.iloc[i] - 1) * 100, 1)
            else:
                row[f"+{h}_%"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def report(name: str, close: pd.Series, sig: pd.Series, horizons: list[int]):
    dates = sig[sig != 0].index if sig.dtype != bool else sig[sig].index
    print(f"\n{'='*60}\n{name} — 共触发 {len(dates)} 次\n{'='*60}")
    if len(dates) == 0:
        print("（历史上没有触发过）")
        return
    tbl = forward_returns(close, dates, horizons)
    print(tbl.to_string(index=False))


# ────────────────────────────────────────────────────────────────
# 5. 单个周期跑三个信号
# ────────────────────────────────────────────────────────────────
TIMEFRAME_NAME = {"D": "日线", "W": "周线", "M": "月线"}


def run_timeframe(daily: pd.DataFrame, timeframe: str, sequence: str):
    period_df = to_period(daily, timeframe)
    name = TIMEFRAME_NAME[timeframe]
    print(f"\n\n{'#'*70}\n# {name}（{timeframe}） — {len(period_df)} 条\n{'#'*70}")

    close = period_df["Close"].astype(float)
    horizons = FORWARD_PERIODS[timeframe]

    sig_div = divergence.signal(period_df)
    sig_demark = demark.signal(period_df)
    sig_seq = sequence_signal(period_df, sequence)

    report(f"信号1：{name} MACD/RSI 底背离", close, sig_div[sig_div == 1], horizons)
    report(f"信号2：神奇九转（{name}，下跌 setup 完成）", close, sig_demark[sig_demark == 1], horizons)
    report(f"信号3：{name} K线序列 {sequence}", close, sig_seq, horizons)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("symbol", nargs="?", default="000001")
    p.add_argument("--timeframe", choices=["D", "W", "M"], default=None,
                   help="只跑指定周期；不传则 D/W/M 三个都跑")
    p.add_argument("--sequence", default="YNYNNNY",
                   help="阳(Y)/阴(N) K线序列，最后一个字符=当期，默认 YNYNNNY（阳阴阳阴阴阴阳）")
    args = p.parse_args()

    print(f"加载 {args.symbol} 日线数据...")
    daily = load_index_ohlc(args.symbol)
    print(f"日线 {len(daily)} 条，{daily.index.min().date()} ~ {daily.index.max().date()}")

    timeframes = [args.timeframe] if args.timeframe else ["D", "W", "M"]
    for tf in timeframes:
        run_timeframe(daily, tf, args.sequence)


if __name__ == "__main__":
    main()
