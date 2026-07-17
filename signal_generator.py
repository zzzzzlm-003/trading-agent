"""
Trading Signal Generator
========================
多时间框架交易信号生成器（日线 + 周线共振）

功能：
- 综合多项技术指标生成 BUY / SELL / HOLD 信号
- 信号强度评分 0-100 分
- 每个信号附带中文理由说明
- 支持批量扫描多只股票

依赖：yfinance, pandas, numpy（无需 TA-Lib）
安装：pip install yfinance pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
import sys

# ─────────────────────────────────────────────────────────────
# 基础技术指标计算（纯 pandas / numpy 实现，无需 ta-lib）
# ─────────────────────────────────────────────────────────────

def calc_sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=window, min_periods=1).mean()

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI 相对强弱指数"""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9) -> tuple:
    """MACD 指标，返回 (macd线, 信号线, 柱状图)"""
    ema_fast   = calc_ema(series, fast)
    ema_slow   = calc_ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bbands(series: pd.Series, window=20, num_std=2.0) -> tuple:
    """布林带，返回 (上轨, 中轨, 下轨)"""
    middle = calc_sma(series, window)
    std    = series.rolling(window=window, min_periods=1).std()
    upper  = middle + num_std * std
    lower  = middle - num_std * std
    return upper, middle, lower

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    """ATR 平均真实波幅"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
             n=9, m1=3, m2=3) -> tuple:
    """KDJ 随机指标，返回 (K, D, J)"""
    low_min  = low.rolling(n, min_periods=1).min()
    high_max = high.rolling(n, min_periods=1).max()
    rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
    K = rsv.ewm(com=m1 - 1, adjust=False).mean()
    D = K.ewm(com=m2 - 1, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J

def calc_volume_ratio(volume: pd.Series, window=20) -> pd.Series:
    """成交量比（当日量 / MA20量）"""
    vol_ma = volume.rolling(window=window, min_periods=1).mean()
    return volume / vol_ma.replace(0, np.nan)

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV 能量潮"""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

# ─────────────────────────────────────────────────────────────
# 数据获取
# ─────────────────────────────────────────────────────────────

def _fetch_ohlcv(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    获取 OHLCV 数据，支持美股、A股（.SH/.SZ 后缀）、港股（.HK）
    A股通过 akshare 获取，其余通过 yfinance
    """
    s = symbol.upper().strip()

    # A股：纯6位数字或带 .SH/.SZ 后缀
    is_a_share = (s.isdigit() and len(s) == 6) or s.endswith(".SH") or s.endswith(".SZ")

    if is_a_share:
        try:
            import akshare as ak
            code = s.replace(".SH", "").replace(".SZ", "").zfill(6)
            start = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
            end   = datetime.now().strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust="qfq"
            )
            df = df.rename(columns={
                "日期": "Date", "开盘": "Open", "最高": "High",
                "最低": "Low",  "收盘": "Close","成交量": "Volume"
            })
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        except ImportError:
            raise RuntimeError("A股需要 akshare：pip install akshare")
        except Exception as e:
            raise RuntimeError(f"A股数据获取失败: {e}")

    # 美股 / 港股：yfinance
    df = yf.download(s, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"无法获取 {symbol} 数据，请检查代码是否正确")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """将日线数据聚合为周线"""
    weekly = df.resample("W").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum"
    }).dropna()
    return weekly

# ─────────────────────────────────────────────────────────────
# 单时间框架信号分析
# ─────────────────────────────────────────────────────────────

def _analyze_tf(df: pd.DataFrame, label: str = "日线") -> dict:
    """
    对一个时间框架的 DataFrame 计算所有指标，返回信号字典
    返回结构：{
        "score": 0-100,
        "signals": [(分值, 方向, 描述), ...],
        "indicators": {...原始指标值...}
    }
    """
    if len(df) < 30:
        return {"score": 50, "signals": [], "indicators": {}, "label": label}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── 计算指标 ──
    ma5   = calc_sma(close, 5)
    ma10  = calc_sma(close, 10)
    ma20  = calc_sma(close, 20)
    ma60  = calc_sma(close, 60) if len(close) >= 60 else None
    ema20 = calc_ema(close, 20)

    rsi14 = calc_rsi(close, 14)
    rsi6  = calc_rsi(close, 6)

    macd_line, macd_sig, macd_hist = calc_macd(close)

    bb_upper, bb_mid, bb_lower = calc_bbands(close, 20)

    k, d, j = calc_kdj(high, low, close)

    atr14     = calc_atr(high, low, close, 14)
    vol_ratio = calc_volume_ratio(volume, 20)

    # 取最新值
    cur   = close.iloc[-1]
    prev  = close.iloc[-2] if len(close) > 1 else cur

    def last(s): return s.iloc[-1] if isinstance(s, pd.Series) else float('nan')
    def prev2(s): return s.iloc[-2] if isinstance(s, pd.Series) and len(s) > 1 else float('nan')

    indicators = {
        "price":        cur,
        "price_chg%":   (cur / prev - 1) * 100,
        "ma5":          last(ma5),
        "ma10":         last(ma10),
        "ma20":         last(ma20),
        "ma60":         last(ma60) if ma60 is not None else None,
        "ema20":        last(ema20),
        "rsi14":        last(rsi14),
        "rsi6":         last(rsi6),
        "macd":         last(macd_line),
        "macd_signal":  last(macd_sig),
        "macd_hist":    last(macd_hist),
        "macd_hist_prev": prev2(macd_hist),
        "bb_upper":     last(bb_upper),
        "bb_mid":       last(bb_mid),
        "bb_lower":     last(bb_lower),
        "bb_pos%":      (cur - last(bb_lower)) / (last(bb_upper) - last(bb_lower) + 1e-10) * 100,
        "bb_width%":    (last(bb_upper) - last(bb_lower)) / (last(bb_mid) + 1e-10) * 100,
        "kdj_k":        last(k),
        "kdj_d":        last(d),
        "kdj_j":        last(j),
        "atr14":        last(atr14),
        "vol_ratio":    last(vol_ratio),
        "recent_high20": high.iloc[-20:].max(),
        "recent_low20":  low.iloc[-20:].min(),
    }

    # ── 信号评分系统 ──
    # 每条信号：(加分值[-30~+30], 方向emoji, 描述文字)
    signals = []

    # 1. 均线多空排列（权重：20分）
    m5, m20 = indicators["ma5"], indicators["ma20"]
    m60 = indicators["ma60"]
    if m5 > m20:
        if m60 is None or m20 > m60:
            signals.append((+20, "✅", f"[{label}] 均线多头排列 MA5>{m5:.2f} > MA20={m20:.2f}"))
        else:
            signals.append((+10, "✅", f"[{label}] MA5>MA20，短期偏多"))
    else:
        if m60 is None or m20 < m60:
            signals.append((-20, "🔴", f"[{label}] 均线空头排列 MA5={m5:.2f} < MA20={m20:.2f}"))
        else:
            signals.append((-10, "🔴", f"[{label}] MA5<MA20，短期偏空"))

    # 2. MACD（权重：20分）
    mh   = indicators["macd_hist"]
    mh_p = indicators["macd_hist_prev"]
    if mh > 0 and mh_p <= 0:
        signals.append((+20, "✅", f"[{label}] MACD金叉，柱状图由负转正={mh:.4f}"))
    elif mh < 0 and mh_p >= 0:
        signals.append((-20, "🔴", f"[{label}] MACD死叉，柱状图由正转负={mh:.4f}"))
    elif mh > 0 and mh > mh_p:
        signals.append((+12, "✅", f"[{label}] MACD柱状图为正且扩张，多头动能增强"))
    elif mh > 0 and mh < mh_p:
        signals.append((+5, "⚠️",  f"[{label}] MACD柱状图为正但收缩，多头动能减弱"))
    elif mh < 0 and mh < mh_p:
        signals.append((-12, "🔴", f"[{label}] MACD柱状图为负且扩张，空头动能增强"))
    else:
        signals.append((-5, "⚠️",  f"[{label}] MACD柱状图为负但收缩，空头动能减弱"))

    # 3. RSI（权重：15分）
    rsi = indicators["rsi14"]
    if rsi > 70:
        signals.append((-10, "⚠️", f"[{label}] RSI={rsi:.1f} 超买区间（>70），注意回调风险"))
    elif rsi > 60:
        signals.append((+10, "✅", f"[{label}] RSI={rsi:.1f} 强势区间（60-70），上涨动能充足"))
    elif rsi > 50:
        signals.append((+5,  "✅", f"[{label}] RSI={rsi:.1f} 偏多区间（50-60）"))
    elif rsi > 40:
        signals.append((-5,  "⚠️", f"[{label}] RSI={rsi:.1f} 偏空区间（40-50）"))
    elif rsi > 30:
        signals.append((-10, "🔴", f"[{label}] RSI={rsi:.1f} 弱势区间（30-40）"))
    else:
        signals.append((+5,  "⚠️", f"[{label}] RSI={rsi:.1f} 超卖区间（<30），可能存在反弹机会"))

    # 4. KDJ（权重：10分）
    k_val, d_val, j_val = indicators["kdj_k"], indicators["kdj_d"], indicators["kdj_j"]
    if k_val > d_val and k_val > 50:
        signals.append((+8, "✅", f"[{label}] KDJ: K={k_val:.1f} > D={d_val:.1f}，偏多"))
    elif k_val < d_val and k_val < 50:
        signals.append((-8, "🔴", f"[{label}] KDJ: K={k_val:.1f} < D={d_val:.1f}，偏空"))
    elif j_val > 80:
        signals.append((-5, "⚠️", f"[{label}] KDJ J值={j_val:.1f} 超买"))
    elif j_val < 20:
        signals.append((+5, "⚠️", f"[{label}] KDJ J值={j_val:.1f} 超卖，可能反弹"))

    # 5. 布林带（权重：10分）
    bb_pos = indicators["bb_pos%"]
    bb_w   = indicators["bb_width%"]
    if bb_pos > 90:
        signals.append((-8, "⚠️", f"[{label}] 价格接近布林带上轨（位置{bb_pos:.0f}%），注意阻力"))
    elif bb_pos > 50:
        signals.append((+5, "✅", f"[{label}] 价格在布林带中上方（位置{bb_pos:.0f}%），偏强"))
    elif bb_pos < 10:
        signals.append((+8, "⚠️", f"[{label}] 价格接近布林带下轨（位置{bb_pos:.0f}%），可能超跌反弹"))
    else:
        signals.append((-5, "🔴", f"[{label}] 价格在布林带中下方（位置{bb_pos:.0f}%），偏弱"))

    if bb_w < 5:
        signals.append((0, "⚠️", f"[{label}] 布林带极度收窄（带宽{bb_w:.1f}%），大行情蓄势中"))

    # 6. 成交量（权重：10分）
    vr = indicators["vol_ratio"]
    price_up = cur > prev
    if not np.isnan(vr):
        if vr > 2.0 and price_up:
            signals.append((+10, "✅", f"[{label}] 量价齐升，成交量{vr:.1f}倍均量，上涨确认度高"))
        elif vr > 1.5 and price_up:
            signals.append((+6, "✅", f"[{label}] 放量上涨，成交量{vr:.1f}倍均量"))
        elif vr > 2.0 and not price_up:
            signals.append((-10, "🔴", f"[{label}] 放量下跌，成交量{vr:.1f}倍均量，卖压沉重"))
        elif vr > 1.5 and not price_up:
            signals.append((-6, "🔴", f"[{label}] 放量下跌，成交量{vr:.1f}倍均量"))
        elif vr < 0.5:
            signals.append((0, "⚠️", f"[{label}] 成交量极度萎缩（{vr:.1f}倍），方向不明"))

    # 7. 均线金叉/死叉（MA5 vs MA20 状态变化，权重：5分）
    if len(ma5) > 3 and len(ma20) > 3:
        prev_ma5  = ma5.iloc[-2]
        prev_ma20 = ma20.iloc[-2]
        if prev_ma5 <= prev_ma20 and last(ma5) > last(ma20):
            signals.append((+15, "✅", f"[{label}] MA5刚上穿MA20，均线金叉！"))
        elif prev_ma5 >= prev_ma20 and last(ma5) < last(ma20):
            signals.append((-15, "🔴", f"[{label}] MA5刚下穿MA20，均线死叉！"))

    # ── 汇总评分 ──
    raw_score = sum(s[0] for s in signals)
    # 映射到 0-100 区间（-100 → 0, 0 → 50, +100 → 100）
    score = min(100, max(0, 50 + raw_score * 0.5))

    return {
        "label":      label,
        "score":      round(score, 1),
        "raw_score":  raw_score,
        "signals":    signals,
        "indicators": indicators
    }

# ─────────────────────────────────────────────────────────────
# 多时间框架合并（日线 + 周线）
# ─────────────────────────────────────────────────────────────

def _combine_signals(daily: dict, weekly: dict) -> dict:
    """
    将日线和周线信号合并，判断是否共振
    日线权重 60%，周线权重 40%
    """
    composite = daily["score"] * 0.60 + weekly["score"] * 0.40

    # 判断方向是否共振
    d_bull = daily["score"]  >= 55
    d_bear = daily["score"]  <= 45
    w_bull = weekly["score"] >= 55
    w_bear = weekly["score"] <= 45

    if d_bull and w_bull:
        resonance = "✅ 日线+周线双向看多共振"
        resonance_bonus = +5
    elif d_bear and w_bear:
        resonance = "🔴 日线+周线双向看空共振"
        resonance_bonus = -5
    elif d_bull and w_bear:
        resonance = "⚠️ 日线偏多、周线偏空（背离，谨慎）"
        resonance_bonus = -3
    elif d_bear and w_bull:
        resonance = "⚠️ 日线偏空、周线偏多（可能调整中）"
        resonance_bonus = -3
    else:
        resonance = "⚪ 信号中性，方向尚不明朗"
        resonance_bonus = 0

    final_score = min(100, max(0, composite + resonance_bonus))

    return {
        "daily_score":    daily["score"],
        "weekly_score":   weekly["score"],
        "final_score":    round(final_score, 1),
        "resonance":      resonance,
        "resonance_bonus": resonance_bonus,
    }

# ─────────────────────────────────────────────────────────────
# 主信号生成函数
# ─────────────────────────────────────────────────────────────

def generate_signal(symbol: str, verbose: bool = True) -> dict:
    """
    为单只股票生成综合交易信号

    参数：
        symbol  股票代码，如 "NVDA"、"600519" (A股茅台)、"0700.HK" (腾讯)
        verbose 是否打印详细分析报告

    返回：
        {
            "symbol":       股票代码,
            "signal":       "BUY" / "SELL" / "HOLD",
            "score":        0-100 综合评分,
            "daily_score":  日线评分,
            "weekly_score": 周线评分,
            "price":        当前价格,
            "atr":          ATR14（波动性参考）,
            "reasons":      [信号理由列表],
            "resonance":    多时间框架共振描述,
            "daily_signals":  日线信号列表,
            "weekly_signals": 周线信号列表,
            "indicators":   日线指标字典
        }
    """
    result = {
        "symbol":  symbol,
        "signal":  "HOLD",
        "score":   50,
        "error":   None
    }

    try:
        # 获取日线数据（1年+）
        daily_df = _fetch_ohlcv(symbol, period="1y")

        # 生成周线数据
        weekly_df = _resample_weekly(daily_df)

        # 日线分析
        daily_analysis  = _analyze_tf(daily_df,  label="日线")

        # 周线分析
        weekly_analysis = _analyze_tf(weekly_df, label="周线")

        # 多时间框架合并
        combined = _combine_signals(daily_analysis, weekly_analysis)

        final_score = combined["final_score"]

        # 信号判定阈值
        if final_score >= 65:
            signal = "BUY"
        elif final_score <= 35:
            signal = "SELL"
        else:
            signal = "HOLD"

        # 汇总理由（只取每个方向最强的几条）
        all_signals = daily_analysis["signals"] + weekly_analysis["signals"]
        reasons = [desc for (_, _, desc) in
                   sorted(all_signals, key=lambda x: abs(x[0]), reverse=True)[:8]]

        result.update({
            "signal":         signal,
            "score":          final_score,
            "daily_score":    combined["daily_score"],
            "weekly_score":   combined["weekly_score"],
            "price":          daily_analysis["indicators"].get("price"),
            "atr":            daily_analysis["indicators"].get("atr14"),
            "reasons":        reasons,
            "resonance":      combined["resonance"],
            "daily_signals":  daily_analysis["signals"],
            "weekly_signals": weekly_analysis["signals"],
            "indicators":     daily_analysis["indicators"],
            "error":          None
        })

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"❌ {symbol} 分析失败: {e}")
        return result

    if verbose:
        _print_report(result)

    return result


def _print_report(r: dict):
    """打印单只股票的详细信号报告"""
    w = 56
    sep = "─" * w

    signal_emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️ "}.get(r["signal"], "?")
    signal_cn    = {"BUY": "买入", "SELL": "卖出", "HOLD": "观望"}.get(r["signal"], r["signal"])

    print(f"\n{'='*w}")
    print(f"  📊 {r['symbol']}  综合交易信号报告")
    print(f"{'='*w}")

    price = r.get("price")
    if price:
        print(f"  当前价格: {price:.2f}")
    print(f"  日线评分: {r['daily_score']:.1f}/100")
    print(f"  周线评分: {r['weekly_score']:.1f}/100")
    print(f"  综合评分: {r['score']:.1f}/100")
    print(f"  共振情况: {r['resonance']}")
    print(f"\n  {signal_emoji} 信号方向: 【{signal_cn}】（{r['score']:.0f}分）")

    # 评分进度条
    bar_filled = int(r["score"] / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"  [{bar}]")

    print(f"\n{sep}")
    print(f"  📋 主要理由（按影响力排序）")
    print(sep)
    for i, reason in enumerate(r.get("reasons", []), 1):
        print(f"  {i}. {reason}")

    # 关键指标
    ind = r.get("indicators", {})
    if ind:
        print(f"\n{sep}")
        print(f"  📈 关键指标快照（日线）")
        print(sep)
        ma60_str = f"MA60={ind['ma60']:.2f}" if ind.get('ma60') else ''
        print(f"  MA5={ind.get('ma5',0):.2f}  MA20={ind.get('ma20',0):.2f}  {ma60_str}")
        print(f"  RSI14={ind.get('rsi14',0):.1f}  "
              f"MACD柱={ind.get('macd_hist',0):.4f}  "
              f"ATR14={ind.get('atr14',0):.2f}")
        print(f"  KDJ: K={ind.get('kdj_k',0):.1f}  D={ind.get('kdj_d',0):.1f}  "
              f"J={ind.get('kdj_j',0):.1f}")
        bb_pos = ind.get("bb_pos%", 50)
        print(f"  布林带位置={bb_pos:.0f}%  成交量比={ind.get('vol_ratio',1):.2f}x")
        if ind.get("atr14"):
            print(f"  参考止损（1.5×ATR）: {ind['price'] - 1.5 * ind['atr14']:.2f}  "
                  f"参考止盈（2×ATR）: {ind['price'] + 2.0 * ind['atr14']:.2f}")

    print(f"{'='*w}")
    print("  ⚠️  以上分析仅供参考，不构成投资建议")
    print(f"{'='*w}\n")


# ─────────────────────────────────────────────────────────────
# 批量扫描
# ─────────────────────────────────────────────────────────────

def scan_stocks(symbols: list, verbose: bool = False) -> pd.DataFrame:
    """
    批量扫描多只股票，返回信号汇总 DataFrame

    参数：
        symbols  股票代码列表，如 ["AAPL", "NVDA", "MSFT"]
        verbose  是否打印每只股票的详细报告

    返回：
        DataFrame，按综合评分排序，包含：
        symbol, signal, score, daily_score, weekly_score, price, resonance
    """
    records = []
    total = len(symbols)
    print(f"\n🔍 开始扫描 {total} 只股票...\n")

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{total}] 分析 {sym}...", end=" ", flush=True)
        r = generate_signal(sym, verbose=verbose)
        if r.get("error"):
            print(f"❌ 失败: {r['error']}")
        else:
            sig_emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️ "}.get(r["signal"], "")
            print(f"{sig_emoji} {r['signal']} ({r['score']:.0f}分)")
        records.append({
            "代码":      r["symbol"],
            "信号":      r.get("signal", "ERROR"),
            "综合评分":  r.get("score", 0),
            "日线评分":  r.get("daily_score", 0),
            "周线评分":  r.get("weekly_score", 0),
            "当前价格":  r.get("price"),
            "共振状态":  r.get("resonance", ""),
            "错误信息":  r.get("error", ""),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("综合评分", ascending=False).reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"  扫描结果汇总（共 {total} 只，按综合评分降序）")
    print(f"{'='*70}")

    # 按信号分组显示
    for signal in ["BUY", "HOLD", "SELL"]:
        label = {"BUY": "📈 BUY 买入", "SELL": "📉 SELL 卖出", "HOLD": "⏸️  HOLD 观望"}[signal]
        subset = df[df["信号"] == signal]
        if not subset.empty:
            print(f"\n  {label}（{len(subset)}只）")
            print(f"  {'─'*60}")
            for _, row in subset.iterrows():
                print(f"  {row['代码']:10s}  评分={row['综合评分']:.0f}  "
                      f"日={row['日线评分']:.0f}/周={row['周线评分']:.0f}  "
                      f"价格={row['当前价格']:.2f}" if row['当前价格'] else
                      f"  {row['代码']:10s}  评分={row['综合评分']:.0f}")

    errors = df[df["错误信息"] != ""]
    if not errors.empty:
        print(f"\n  ❌ 获取失败 ({len(errors)}只): {', '.join(errors['代码'].tolist())}")

    print(f"\n{'='*70}")
    print("  ⚠️  以上分析仅供参考，不构成投资建议")
    print(f"{'='*70}\n")

    return df


# ─────────────────────────────────────────────────────────────
# 获取信号历史序列（供回测使用）
# ─────────────────────────────────────────────────────────────

def generate_signal_series(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    在一个完整的 DataFrame 上滚动生成历史信号序列（供回测使用）

    参数：
        df        OHLCV DataFrame（日线）
        lookback  最小计算窗口（交易日数）

    返回：
        pd.Series，index 对齐 df，值为 1（买入）/ -1（卖出）/ 0（观望）
    """
    signals = pd.Series(0, index=df.index)

    for i in range(lookback, len(df)):
        window = df.iloc[max(0, i - lookback): i + 1]
        try:
            analysis = _analyze_tf(window, label="")
            score = analysis["score"]
            if score >= 65:
                signals.iloc[i] = 1
            elif score <= 35:
                signals.iloc[i] = -1
            else:
                signals.iloc[i] = 0
        except Exception:
            signals.iloc[i] = 0

    return signals


# ─────────────────────────────────────────────────────────────
# 主程序示例
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        # 默认：单只股票详细分析
        print("用法:")
        print("  python signal_generator.py NVDA              # 单只股票详细分析")
        print("  python signal_generator.py NVDA AAPL MSFT    # 批量扫描")
        print("")
        print("运行默认示例（NVDA）...")
        generate_signal("NVDA", verbose=True)

    elif len(args) == 1:
        # 单只股票详细分析
        generate_signal(args[0], verbose=True)

    else:
        # 多只股票批量扫描
        result_df = scan_stocks(args, verbose=False)
        print("\n（如需查看某只股票详细报告，单独运行：python signal_generator.py <代码>）")
