"""Price data loader — handles US / HK / A-share stocks AND A-share indices.

Upgrades over signal_generator._fetch_ohlcv:
  1. `period` is honored for A-shares (maps to days)
  2. A-share indices (000300/399006/etc.) try akshare's index API first
  3. HK index ^HSTECH falls back to ETF proxy 513180.SS
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UNIVERSE_PATH = Path(__file__).resolve().parent / "universe.yml"

_PERIOD_DAYS = {
    "1y": 400, "2y": 800, "3y": 1200, "5y": 1900, "10y": 3800, "max": 9000,
}

_HK_INDEX_FALLBACK = {
    "^HSTECH": "513180.SS",  # 恒生科技 ETF 作为代理
}

# 已知的 A 股指数代码（akshare 需要走 index API）
_A_INDEX_CODES = {
    "000001", "000016", "000300", "000852", "000905", "000906",
    "399001", "399006", "399300", "399905", "399106",
}


def load_universe() -> dict:
    with open(UNIVERSE_PATH) as f:
        return yaml.safe_load(f)


def _period_days(period: str) -> int:
    return _PERIOD_DAYS.get(period.lower(), 400)


def _with_retry(fn, *args, retries: int = 3, sleep: float = 1.5, **kw):
    import time
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kw)
        except Exception as e:
            last = e
            time.sleep(sleep * (i + 1))
    raise last


def _normalize_ak(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "日期": "Date", "开盘": "Open", "最高": "High",
        "最低": "Low", "收盘": "Close", "成交量": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def _fetch_a_index(code: str, period: str) -> pd.DataFrame:
    import akshare as ak
    days = _period_days(period)
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")
    df = _with_retry(ak.index_zh_a_hist, symbol=code, period="daily",
                     start_date=start, end_date=end)
    return _normalize_ak(df)


def _fetch_a_stock(code: str, period: str) -> pd.DataFrame:
    import akshare as ak
    days = _period_days(period)
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")
    df = _with_retry(ak.stock_zh_a_hist, symbol=code, period="daily",
                     start_date=start, end_date=end, adjust="qfq")
    return _normalize_ak(df)


def _fetch_yfinance(symbol: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned empty for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fetch(symbol: str, period: str = "10y") -> pd.DataFrame:
    s = symbol.upper().strip()

    # HK fallback
    if s in _HK_INDEX_FALLBACK:
        try:
            return _fetch_yfinance(s, period)
        except Exception:
            return _fetch_yfinance(_HK_INDEX_FALLBACK[s], period)

    # A-share: 6 digits or .SS/.SZ/.SH suffix
    core = s.replace(".SS", "").replace(".SZ", "").replace(".SH", "")
    is_a = (s.endswith(".SS") or s.endswith(".SZ") or s.endswith(".SH") or
            (s.isdigit() and len(s) == 6))
    if is_a:
        code = core.zfill(6)
        # Try yfinance first — handles A-share ETFs cleanly and avoids akshare flakiness.
        # Only fall back to akshare when yfinance truly has no data (rare for ETFs,
        # common for raw index codes like 000300).
        if s.endswith(".SS") or s.endswith(".SZ"):
            try:
                df = _fetch_yfinance(s, period)
                if len(df) >= 50:
                    return df
            except Exception:
                pass
        # akshare fallback: index first if in known list, else stock
        if code in _A_INDEX_CODES:
            try:
                return _fetch_a_index(code, period)
            except Exception:
                pass
        try:
            return _fetch_a_stock(code, period)
        except Exception as e1:
            try:
                return _fetch_a_index(code, period)
            except Exception as e2:
                raise RuntimeError(f"A-share fetch failed for {code}: stock={e1}; index={e2}")

    # Default: yfinance
    return _fetch_yfinance(s, period)


def fetch_many(symbols: Iterable[str], period: str = "10y") -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        try:
            out[s] = fetch(s, period=period)
        except Exception as e:
            print(f"  ✗ {s}: {e}")
    return out
