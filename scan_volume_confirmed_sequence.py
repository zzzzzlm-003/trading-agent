"""在沪深300/中证500/中证1000三档样本股上，扫描 K线阳/阴序列 + 量能确认 的联合信号。

量能确认规则（逐根比较相邻两根K线）：
  阳(Y) -> 阴(N)：要求成交额比上一根缩量（更小）
  阴(N) -> 阳(Y)：要求成交额比上一根放量（更大）
  同色相邻（如 N->N）：不做要求

成交额用 Close * Volume 近似（yfinance 不直接提供真实成交额）。

用法：
    python scan_volume_confirmed_sequence.py [--sequence YNYNNNY]
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

from analyze_weekly_bottom_signals import to_period  # noqa: E402

UNIVERSES = [("000300", "沪深300(大盘)"), ("000905", "中证500(中盘)"), ("000852", "中证1000(中小盘)")]
HORIZONS = [1, 2, 3]


def get_constituents(code: str) -> pd.DataFrame:
    df = ak.index_stock_cons_csindex(symbol=code)
    df = df[["成分券代码", "成分券名称", "交易所"]].rename(
        columns={"成分券代码": "code", "成分券名称": "name", "交易所": "exchange"}
    )
    df["ticker"] = df.apply(
        lambda r: str(r["code"]).zfill(6) + (".SS" if "上海" in r["exchange"] else ".SZ"), axis=1
    )
    return df


def _download_chunk(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, period="max", progress=False, auto_adjust=True,
                       group_by="ticker", threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t][["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= 60:
                out[t] = df
        except KeyError:
            continue
    return out


def batch_download(tickers: list[str], chunk_size: int = 150, retries: int = 4) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    pending = list(tickers)
    for attempt in range(retries):
        if not pending:
            break
        if attempt > 0:
            wait = 10 * attempt
            print(f"  重试第 {attempt} 次，剩余 {len(pending)} 只，先等 {wait}s 避开限流...")
            time.sleep(wait)
        for i in range(0, len(pending), chunk_size):
            chunk = pending[i:i + chunk_size]
            out.update(_download_chunk(chunk))
        pending = [t for t in tickers if t not in out]
    return out


def sequence_and_volume_hits(df_daily: pd.DataFrame, timeframe: str, sequence: str) -> list[dict]:
    period_df = to_period(df_daily, timeframe)
    close = period_df["Close"].astype(float)
    open_ = period_df["Open"].astype(float)
    amount = (close * period_df["Volume"].astype(float))  # 成交额近似
    label = pd.Series(np.where(close >= open_, "Y", "N"), index=period_df.index)

    n = len(sequence)
    rows = []
    for i in range(n - 1, len(label)):
        window = "".join(label.iloc[i - n + 1: i + 1])
        if window != sequence:
            continue
        ok = True
        for k in range(i - n + 2, i + 1):
            prev, cur = label.iloc[k - 1], label.iloc[k]
            if prev == "Y" and cur == "N" and not (amount.iloc[k] < amount.iloc[k - 1]):
                ok = False
                break
            if prev == "N" and cur == "Y" and not (amount.iloc[k] > amount.iloc[k - 1]):
                ok = False
                break
        if not ok:
            continue
        d = label.index[i]
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
    print("拉取沪深300/中证500/中证1000成分股列表...")
    universe_map: dict[str, list[str]] = {}
    name_map: dict[str, str] = {}
    all_tickers: set[str] = set()
    for code, label in UNIVERSES:
        cons = get_constituents(code)
        universe_map[label] = cons["ticker"].tolist()
        name_map.update(dict(zip(cons["ticker"], cons["name"])))
        all_tickers.update(cons["ticker"].tolist())
    print(f"三个指数去重后共 {len(all_tickers)} 只，开始批量下载历史行情...")

    data = batch_download(sorted(all_tickers))
    print(f"成功下载 {len(data)}/{len(all_tickers)} 只，耗时 {time.time()-t0:.1f}s")

    unit = {"W": "周", "M": "个月"}
    tf_name = {"W": "周线", "M": "月线"}

    out_dir = Path("/private/tmp/claude-501/-Users-luomengzhou/bbc32f4d-7c70-4bdc-aadb-b603203f2443/scratchpad")
    out_dir.mkdir(parents=True, exist_ok=True)

    for universe_label, tickers in universe_map.items():
        print(f"\n\n{'#'*70}\n# {universe_label} — 样本股 {len(tickers)} 只\n{'#'*70}")
        for tf in ["W", "M"]:
            rows = []
            for t in tickers:
                if t not in data:
                    continue
                for r in sequence_and_volume_hits(data[t], tf, seq):
                    rows.append({"ticker": t, "name": name_map[t], **r})
            tbl = pd.DataFrame(rows)
            n_stocks = tbl["ticker"].nunique() if len(tbl) else 0
            print(f"\n{'='*60}\n{tf_name[tf]} K线序列 {seq} + 量能确认 — 共触发 {len(tbl)} 次（{n_stocks} 只股票）\n{'='*60}")
            if len(tbl) == 0:
                print("（没有样本）")
                continue
            u = unit[tf]
            for h in HORIZONS:
                col = f"+{h}_%"
                done = tbl.dropna(subset=[col])
                if len(done):
                    win_rate = (done[col] > 0).mean() * 100
                    print(f"+{h}{u}：{len(done)} 次有结果，平均收益 {done[col].mean():.2f}%，"
                          f"中位数 {done[col].median():.2f}%，胜率 {win_rate:.1f}%")
            latest = tbl[tbl["is_latest_bar"]]
            if len(latest):
                print(f"当前（最新一根{tf_name[tf]}）同时满足序列+量能确认的股票（{len(latest)} 只）：")
                print(latest[["ticker", "name", "date"]].to_string(index=False))

            fname = f"{universe_label.split('(')[0]}_{tf_name[tf]}_sequence_volume_hits.csv"
            tbl.to_csv(out_dir / fname, index=False)

    print(f"\n明细 CSV 已存到 {out_dir}")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
