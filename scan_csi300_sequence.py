"""在沪深300全部成分股上扫描 K线阳/阴序列信号（信号3，复用 analyze_weekly_bottom_signals.py 的逻辑）。

周线：只看触发后下1周（+1根周K线）的收益。
月线：只看触发后下1个月（+1根月K线）的收益。

用法：
    python scan_csi300_sequence.py [--sequence YNYNNNY]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from analyze_weekly_bottom_signals import to_period, sequence_signal  # noqa: E402


def get_csi300_constituents() -> pd.DataFrame:
    df = ak.index_stock_cons_csindex(symbol="000300")
    df = df[["成分券代码", "成分券名称", "交易所"]].rename(
        columns={"成分券代码": "code", "成分券名称": "name", "交易所": "exchange"}
    )
    df["ticker"] = df.apply(
        lambda r: str(r["code"]).zfill(6) + (".SS" if "上海" in r["exchange"] else ".SZ"), axis=1
    )
    return df


def batch_download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, period="max", progress=False, auto_adjust=True,
                       group_by="ticker", threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t][["Open", "High", "Low", "Close"]].dropna()
            if len(df) >= 60:
                out[t] = df
        except KeyError:
            continue
    return out


HORIZONS = [1, 2, 3]


def scan_one(df_daily: pd.DataFrame, timeframe: str, sequence: str, dd_threshold: float | None = None) -> list[dict]:
    """dd_threshold: 若给定（如 0.5），只保留触发时「较历史最高点回撤 >= dd_threshold」的点。"""
    period_df = to_period(df_daily, timeframe)
    close = period_df["Close"].astype(float)
    hit = sequence_signal(period_df, sequence)
    if dd_threshold is not None:
        drawdown = close / close.cummax() - 1  # <=0，-0.5 代表回撤50%
        hit = hit & (drawdown <= -dd_threshold)
    dates = hit[hit].index
    rows = []
    for d in dates:
        i = close.index.get_loc(d)
        row = {"date": d.date()}
        for h in HORIZONS:
            j = i + h
            row[f"+{h}_%"] = round((close.iloc[j] / close.iloc[i] - 1) * 100, 2) if j < len(close) else np.nan
        row["is_latest_bar"] = (i + 1) >= len(close)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", default="YNYNNNY")
    args = p.parse_args()
    seq = args.sequence

    t0 = time.time()
    print("拉取沪深300最新成分股列表...")
    cons = get_csi300_constituents()
    print(f"共 {len(cons)} 只，开始批量下载历史行情...")

    tickers = cons["ticker"].tolist()
    data = batch_download(tickers)
    print(f"成功下载 {len(data)}/{len(tickers)} 只，耗时 {time.time()-t0:.1f}s")

    name_map = dict(zip(cons["ticker"], cons["name"]))

    daily_rows, weekly_rows, monthly_rows = [], [], []
    for t, df in data.items():
        for r in scan_one(df, "D", seq):
            daily_rows.append({"ticker": t, "name": name_map[t], **r})
        for r in scan_one(df, "W", seq):
            weekly_rows.append({"ticker": t, "name": name_map[t], **r})
        for r in scan_one(df, "M", seq):
            monthly_rows.append({"ticker": t, "name": name_map[t], **r})

    daily_df = pd.DataFrame(daily_rows)
    weekly_df = pd.DataFrame(weekly_rows)
    monthly_df = pd.DataFrame(monthly_rows)

    unit = {"日线": "天", "周线": "周", "月线": "个月"}
    for label, tbl in [("日线", daily_df), ("周线", weekly_df), ("月线", monthly_df)]:
        u = unit[label]
        print(f"\n{'='*70}\n{label} K线序列 {seq} — 全市场共触发 {len(tbl)} 次（{len(tbl['ticker'].unique()) if len(tbl) else 0} 只股票）\n{'='*70}")
        if len(tbl) == 0:
            print("（没有触发）")
            continue
        for h in HORIZONS:
            col = f"+{h}_%"
            done = tbl.dropna(subset=[col])
            if len(done):
                win_rate = (done[col] > 0).mean() * 100
                print(f"+{h}{u}：{len(done)} 次有结果，平均收益 {done[col].mean():.2f}%，"
                      f"中位数 {done[col].median():.2f}%，胜率 {win_rate:.1f}%")
        latest = tbl[tbl["is_latest_bar"]]
        if len(latest):
            print(f"\n*** 当前（最新一根{label}）正好触发此序列的股票（{len(latest)} 只，后续结果尚未走完）：")
            print(latest[["ticker", "name", "date"]].to_string(index=False))

    # 叠加条件：触发时较历史最高点回撤 >= 50%（只看周线/月线）
    dd_weekly_rows, dd_monthly_rows = [], []
    for t, df in data.items():
        for r in scan_one(df, "W", seq, dd_threshold=0.5):
            dd_weekly_rows.append({"ticker": t, "name": name_map[t], **r})
        for r in scan_one(df, "M", seq, dd_threshold=0.5):
            dd_monthly_rows.append({"ticker": t, "name": name_map[t], **r})
    dd_weekly_df = pd.DataFrame(dd_weekly_rows)
    dd_monthly_df = pd.DataFrame(dd_monthly_rows)

    print(f"\n\n{'#'*70}\n# 叠加条件：触发时较历史最高点回撤 >= 50%\n{'#'*70}")
    for label, tbl in [("周线", dd_weekly_df), ("月线", dd_monthly_df)]:
        u = unit[label]
        print(f"\n{'='*70}\n{label} K线序列 {seq} + 回撤>=50% — 全市场共触发 {len(tbl)} 次（{len(tbl['ticker'].unique()) if len(tbl) else 0} 只股票）\n{'='*70}")
        if len(tbl) == 0:
            print("（没有触发）")
            continue
        for h in HORIZONS:
            col = f"+{h}_%"
            done = tbl.dropna(subset=[col])
            if len(done):
                win_rate = (done[col] > 0).mean() * 100
                print(f"+{h}{u}：{len(done)} 次有结果，平均收益 {done[col].mean():.2f}%，"
                      f"中位数 {done[col].median():.2f}%，胜率 {win_rate:.1f}%")
        latest = tbl[tbl["is_latest_bar"]]
        if len(latest):
            print(f"\n*** 当前（最新一根{label}）正好同时满足序列+回撤>=50%的股票（{len(latest)} 只）：")
            print(latest[["ticker", "name", "date"]].to_string(index=False))

    out_dir = Path("/private/tmp/claude-501/-Users-luomengzhou/bbc32f4d-7c70-4bdc-aadb-b603203f2443/scratchpad")
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_dir / "csi300_daily_sequence_hits.csv", index=False)
    weekly_df.to_csv(out_dir / "csi300_weekly_sequence_hits.csv", index=False)
    monthly_df.to_csv(out_dir / "csi300_monthly_sequence_hits.csv", index=False)
    dd_weekly_df.to_csv(out_dir / "csi300_weekly_sequence_dd50_hits.csv", index=False)
    dd_monthly_df.to_csv(out_dir / "csi300_monthly_sequence_dd50_hits.csv", index=False)
    print(f"\n明细已存到 {out_dir}/csi300_{{daily,weekly,monthly}}_sequence_hits.csv "
          f"和 csi300_{{weekly,monthly}}_sequence_dd50_hits.csv")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
