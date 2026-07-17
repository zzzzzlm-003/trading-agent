"""
Trading Analysis Agent
=====================
支持市场：美股 (含期权) | A股 (含资金流) | 恒生科技指数
依赖库：talib, yfinance, akshare, pandas, numpy
安装：pip install ta-lib yfinance akshare pandas numpy mplfinance
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
import talib
from datetime import datetime, timedelta
from typing import Optional

# ─────────────────────────────────────────────
# 蜡烛图形态中文对照表（日本蜡烛图技术）
# ─────────────────────────────────────────────
CDL_NAMES_ZH = {
    "CDL2CROWS":           ("两只乌鸦",         "看跌"),
    "CDL3BLACKCROWS":      ("三只黑乌鸦",        "看跌"),
    "CDL3INSIDE":          ("三内部上涨/下跌",    "双向"),
    "CDL3LINESTRIKE":      ("三线打击",          "双向"),
    "CDL3OUTSIDE":         ("三外部上涨/下跌",    "双向"),
    "CDL3STARSINSOUTH":    ("南方三星",          "看涨"),
    "CDL3WHITESOLDIERS":   ("三白兵",            "看涨"),
    "CDLABANDONEDBABY":    ("弃婴",              "双向"),
    "CDLADVANCEBLOCK":     ("推进阻塞",          "看跌"),
    "CDLBELTHOLD":         ("捉腰带线",          "双向"),
    "CDLBREAKAWAY":        ("脱离",              "双向"),
    "CDLCLOSINGMARUBOZU":  ("收盘秃头线",        "双向"),
    "CDLCONCEALBABYSWALL": ("藏婴吞没",          "看涨"),
    "CDLCOUNTERATTACK":    ("反击线",            "双向"),
    "CDLDARKCLOUDCOVER":   ("乌云盖顶",          "看跌"),
    "CDLDOJI":             ("十字星",            "中性"),
    "CDLDOJISTAR":         ("十字星星",          "双向"),
    "CDLDRAGONFLYDOJI":    ("蜻蜓十字",          "看涨"),
    "CDLENGULFING":        ("吞没形态",          "双向"),
    "CDLEVENINGDOJISTAR":  ("十字黄昏星",        "看跌"),
    "CDLEVENINGSTAR":      ("黄昏星",            "看跌"),
    "CDLGAPSIDESIDEWHITE": ("向上/下跳空并列阳线", "双向"),
    "CDLGRAVESTONEDOJI":   ("墓碑十字",          "看跌"),
    "CDLHAMMER":           ("锤子线",            "看涨"),
    "CDLHANGINGMAN":       ("上吊线",            "看跌"),
    "CDLHARAMI":           ("孕线",              "双向"),
    "CDLHARAMICROSS":      ("十字孕线",          "双向"),
    "CDLHIGHWAVE":         ("高浪线",            "中性"),
    "CDLHIKKAKE":          ("陷阱形态",          "双向"),
    "CDLHIKKAKEMOD":       ("改良陷阱形态",       "双向"),
    "CDLHOMINGPIGEON":     ("归巢鸽",            "看涨"),
    "CDLIDENTICAL3CROWS":  ("相同三乌鸦",        "看跌"),
    "CDLINNECK":           ("颈内线",            "看跌"),
    "CDLINVERTEDHAMMER":   ("倒锤线",            "看涨"),
    "CDLKICKING":          ("反冲形态",          "双向"),
    "CDLKICKINGBYLENGTH":  ("由较长缺口决定的反冲", "双向"),
    "CDLLADDERBOTTOM":     ("梯底",              "看涨"),
    "CDLLONGLEGGEDDOJI":   ("长腿十字",          "中性"),
    "CDLLONGLINE":         ("长实体线",          "双向"),
    "CDLMARUBOZU":         ("光头光脚线/秃头线",  "双向"),
    "CDLMATCHINGLOW":      ("相匹配的低价",       "看涨"),
    "CDLMATHOLD":          ("持仓形态",          "看涨"),
    "CDLMORNINGDOJISTAR":  ("十字晨星",          "看涨"),
    "CDLMORNINGSTAR":      ("晨星",              "看涨"),
    "CDLONNECK":           ("颈上线",            "看跌"),
    "CDLPIERCING":         ("刺透形态",          "看涨"),
    "CDLRICKSHAWMAN":      ("黄包车夫",          "中性"),
    "CDLRISEFALL3METHODS": ("上升/下降三法",      "双向"),
    "CDLSEPARATINGLINES":  ("分离线",            "双向"),
    "CDLSHOOTINGSTAR":     ("流星线",            "看跌"),
    "CDLSHORTLINE":        ("短实体线",          "中性"),
    "CDLSPINNINGTOP":      ("纺锤线",            "中性"),
    "CDLSTALLEDPATTERN":   ("停顿形态",          "看跌"),
    "CDLSTICKSANDWICH":    ("条形三明治",        "看涨"),
    "CDLTAKURI":           ("探水竿",            "看涨"),
    "CDLTASUKIGAP":        ("跳空并列阴阳线",    "双向"),
    "CDLTHRUSTING":        ("推入线",            "看跌"),
    "CDLTRISTAR":          ("三星",              "双向"),
    "CDLUNIQUE3RIVER":     ("独特三河底",        "看涨"),
    "CDLUPSIDEGAP2CROWS":  ("向上跳空两乌鸦",   "看跌"),
    "CDLXSIDEGAP3METHODS": ("跳空三法",          "双向"),
}

# ─────────────────────────────────────────────
# 市场检测
# ─────────────────────────────────────────────
def detect_market(symbol: str) -> str:
    """自动判断市场类型"""
    s = symbol.upper().strip()
    if s in ["HSTECH", "3033.HK", "3067.HK", "2836.HK"]:
        return "HSI_TECH"
    if s.endswith(".HK"):
        return "HK"
    # A股：纯数字6位
    if s.isdigit() and len(s) == 6:
        return "A_SHARE"
    # 带交易所后缀
    if s.endswith(".SH") or s.endswith(".SZ"):
        return "A_SHARE"
    return "US"

# ─────────────────────────────────────────────
# 数据获取层
# ─────────────────────────────────────────────
def get_price_data(symbol: str, market: str, period: str = "6mo") -> pd.DataFrame:
    """获取OHLCV数据"""
    if market == "A_SHARE":
        # 标准化代码
        code = symbol.replace(".SH", "").replace(".SZ", "").zfill(6)
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=(datetime.now() - timedelta(days=180)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq"  # 前复权
            )
            df = df.rename(columns={
                "日期": "Date", "开盘": "Open", "最高": "High",
                "最低": "Low", "收盘": "Close", "成交量": "Volume"
            })
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            raise ValueError(f"A股数据获取失败: {e}")

    elif market == "HSI_TECH":
        # 用3033.HK作为恒生科技指数代理（CSOP ETF）
        yf_symbol = "3033.HK"
        df = yf.download(yf_symbol, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("恒生科技数据获取失败，请检查网络")
        return df[["Open", "High", "Low", "Close", "Volume"]]

    else:  # US
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"美股数据获取失败: {symbol}")
        # yfinance有时返回MultiIndex列
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]]


# ─────────────────────────────────────────────
# 技术分析层
# ─────────────────────────────────────────────
def analyze_technical(df: pd.DataFrame) -> dict:
    """运行全套技术指标 + 蜡烛图形态识别"""
    open_  = df["Open"].astype(float).values
    high   = df["High"].astype(float).values
    low    = df["Low"].astype(float).values
    close  = df["Close"].astype(float).values
    volume = df["Volume"].astype(float).values

    result = {}

    # ── 趋势指标 ──
    result["MA5"]   = talib.SMA(close, 5)[-1]
    result["MA10"]  = talib.SMA(close, 10)[-1]
    result["MA20"]  = talib.SMA(close, 20)[-1]
    result["MA60"]  = talib.SMA(close, 60)[-1] if len(close) >= 60 else None
    result["EMA20"] = talib.EMA(close, 20)[-1]

    macd, signal, hist = talib.MACD(close)
    result["MACD"]        = macd[-1]
    result["MACD_Signal"] = signal[-1]
    result["MACD_Hist"]   = hist[-1]
    result["MACD_Cross"]  = "金叉" if hist[-1] > 0 and hist[-2] <= 0 else (
                            "死叉" if hist[-1] < 0 and hist[-2] >= 0 else "无")

    # ── 震荡指标 ──
    result["RSI14"] = talib.RSI(close, 14)[-1]
    result["RSI6"]  = talib.RSI(close, 6)[-1]

    stoch_k, stoch_d = talib.STOCH(high, low, close)
    result["KDJ_K"] = stoch_k[-1]
    result["KDJ_D"] = stoch_d[-1]
    result["KDJ_J"] = 3 * stoch_k[-1] - 2 * stoch_d[-1]

    # ── 波动率 ──
    upper, middle, lower = talib.BBANDS(close, 20)
    result["BB_Upper"]  = upper[-1]
    result["BB_Middle"] = middle[-1]
    result["BB_Lower"]  = lower[-1]
    bw = (upper[-1] - lower[-1]) / middle[-1] * 100
    result["BB_Width%"] = bw
    # 当前价在布林带中的位置 0-100%
    result["BB_Position%"] = (close[-1] - lower[-1]) / (upper[-1] - lower[-1]) * 100

    result["ATR14"] = talib.ATR(high, low, close, 14)[-1]

    # ── 成交量 ──
    result["Volume_MA20"] = talib.SMA(volume, 20)[-1]
    result["Volume_Ratio"] = volume[-1] / result["Volume_MA20"] if result["Volume_MA20"] > 0 else None

    # ── 支撑/压力（近期高低点）──
    result["Recent_High20"] = max(high[-20:])
    result["Recent_Low20"]  = min(low[-20:])

    # ── 蜡烛图形态（扫描过去5根K线）──
    patterns_found = []
    for pat_name in CDL_NAMES_ZH:
        func = getattr(talib, pat_name, None)
        if func is None:
            continue
        signals = func(open_, high, low, close)
        # 检查最近5根K线
        for i in range(-5, 0):
            val = signals[i]
            if val != 0:
                zh_name, direction = CDL_NAMES_ZH[pat_name]
                candle_dir = "看涨" if val > 0 else "看跌"
                days_ago = abs(i)
                patterns_found.append({
                    "pattern":   pat_name,
                    "name_zh":   zh_name,
                    "direction": candle_dir,
                    "strength":  abs(val),
                    "days_ago":  days_ago - 1  # 0=今天
                })

    result["Candle_Patterns"] = patterns_found
    result["Current_Price"]   = close[-1]
    result["Price_Change%"]   = (close[-1] / close[-2] - 1) * 100 if len(close) >= 2 else 0

    return result


# ─────────────────────────────────────────────
# A股资金流向
# ─────────────────────────────────────────────
def get_a_share_fund_flow(symbol: str) -> dict:
    """获取A股个股资金流向（东方财富数据）"""
    code = symbol.replace(".SH", "").replace(".SZ", "").zfill(6)
    market = "sh" if code.startswith("6") else "sz"

    try:
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        latest = df.iloc[-1]
        result = {
            "date":             str(latest["日期"]),
            "main_net_inflow":  float(latest["主力净流入-净额"]),       # 主力净流入（元）
            "main_net_ratio":   float(latest["主力净流入-净占比"]),      # 占比%
            "super_large_net":  float(latest["超大单净流入-净额"]),      # 超大单（机构）
            "large_net":        float(latest["大单净流入-净额"]),        # 大单
            "medium_net":       float(latest["中单净流入-净额"]),        # 中单（散户）
            "small_net":        float(latest["小单净流入-净额"]),        # 小单
        }

        # 3日/5日趋势
        if len(df) >= 5:
            result["main_net_3d"] = df["主力净流入-净额"].tail(3).sum()
            result["main_net_5d"] = df["主力净流入-净额"].tail(5).sum()

        return result
    except Exception as e:
        return {"error": str(e)}


def get_northbound_flow() -> dict:
    """北向资金（沪深港通）今日情况"""
    try:
        df = ak.stock_hsgt_fund_flow_summary_em()
        # 只取北向（沪股通+深股通）
        north = df[df["资金方向"] == "北向"]
        total_net = north["成交净买额"].sum()
        return {
            "date":      str(df["交易日"].iloc[0]),
            "total_net": total_net,   # 亿元
            "detail":    north[["板块", "成交净买额", "资金净流入"]].to_dict("records")
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# 美股期权分析
# ─────────────────────────────────────────────
def analyze_options(symbol: str, price: float) -> dict:
    """
    获取美股期权链，分析：
    - IV Rank（用最近期权隐含波动率均值估算）
    - 最大痛点（Max Pain）
    - ATM期权建议（Covered Call / Cash Secured Put / Long Call 等）
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return {"error": "无法获取期权数据"}

        # 取最近3个到期日
        results = {}
        for exp in expirations[:3]:
            chain = ticker.option_chain(exp)
            calls = chain.calls.copy()
            puts  = chain.puts.copy()

            # ATM 期权（±5% 区间）
            atm_calls = calls[(calls["strike"] >= price * 0.97) &
                              (calls["strike"] <= price * 1.05)].copy()
            atm_puts  = puts[(puts["strike"] >= price * 0.95) &
                             (puts["strike"] <= price * 1.03)].copy()

            # 计算 IV Rank 近似：用 ATM IV 的位置
            avg_iv = pd.concat([
                atm_calls["impliedVolatility"],
                atm_puts["impliedVolatility"]
            ]).mean() * 100  # 转为百分比

            # 最大痛点（open interest 加权）
            all_strikes = pd.concat([
                calls[["strike", "openInterest"]],
                puts[["strike", "openInterest"]]
            ]).groupby("strike")["openInterest"].sum()
            if not all_strikes.empty:
                max_pain = all_strikes.idxmax()
            else:
                max_pain = None

            # 成交量最大的 call/put
            top_call = calls.nlargest(1, "volume")[["strike","lastPrice","impliedVolatility","volume","openInterest"]].to_dict("records")
            top_put  = puts.nlargest(1, "volume")[["strike","lastPrice","impliedVolatility","volume","openInterest"]].to_dict("records")

            results[exp] = {
                "avg_atm_iv":  round(avg_iv, 1),
                "max_pain":    max_pain,
                "top_call":    top_call,
                "top_put":     top_put,
                "atm_calls":   atm_calls[["strike","bid","ask","impliedVolatility","volume","openInterest"]].to_dict("records"),
                "atm_puts":    atm_puts[["strike","bid","ask","impliedVolatility","volume","openInterest"]].to_dict("records"),
            }

        return results

    except Exception as e:
        return {"error": str(e)}


def suggest_option_strategy(tech: dict, options: dict) -> list:
    """根据技术面信号和期权数据给出期权策略建议"""
    suggestions = []
    rsi = tech.get("RSI14", 50)
    macd_hist = tech.get("MACD_Hist", 0)
    bb_pos = tech.get("BB_Position%", 50)
    patterns = tech.get("Candle_Patterns", [])

    bullish_patterns = [p for p in patterns if p["direction"] == "看涨"]
    bearish_patterns = [p for p in patterns if p["direction"] == "看跌"]

    # 判断多空偏向
    bull_score = len(bullish_patterns)
    if macd_hist > 0:  bull_score += 1
    if rsi > 50:       bull_score += 1
    if bb_pos > 50:    bull_score += 1

    bear_score = len(bearish_patterns)
    if macd_hist < 0:  bear_score += 1
    if rsi < 50:       bear_score += 1
    if bb_pos < 50:    bear_score += 1

    # 期权 IV 水平（用第一个到期日）
    first_exp = list(options.keys())[0] if options else None
    iv = options[first_exp]["avg_atm_iv"] if first_exp and "error" not in options else None

    if bull_score >= 3:
        if iv and iv > 40:
            suggestions.append({
                "strategy":    "Bull Put Spread",
                "reason":      f"技术面看涨(多/空={bull_score}/{bear_score})，IV偏高({iv:.0f}%)适合卖方策略",
                "risk":        "有限亏损（价差宽度）",
                "max_profit":  "收取权利金"
            })
        else:
            suggestions.append({
                "strategy":    "Long Call / Call Debit Spread",
                "reason":      f"技术面看涨(多/空={bull_score}/{bear_score})，IV正常适合买方",
                "risk":        "最大亏损为权利金",
                "max_profit":  "理论无限（Long Call）"
            })

    elif bear_score >= 3:
        if iv and iv > 40:
            suggestions.append({
                "strategy":    "Bear Call Spread",
                "reason":      f"技术面看跌(多/空={bull_score}/{bear_score})，IV偏高适合卖方",
                "risk":        "有限亏损（价差宽度）",
                "max_profit":  "收取权利金"
            })
        else:
            suggestions.append({
                "strategy":    "Long Put / Put Debit Spread",
                "reason":      f"技术面看跌(多/空={bull_score}/{bear_score})",
                "risk":        "最大亏损为权利金",
                "max_profit":  "理论最大为标的价格"
            })

    else:
        if iv and iv > 45:
            suggestions.append({
                "strategy":    "Short Strangle / Iron Condor",
                "reason":      f"方向不明确，IV高({iv:.0f}%)，适合震荡策略",
                "risk":        "需设止损，理论无限亏损（Strangle）",
                "max_profit":  "收取两腿权利金"
            })
        else:
            suggestions.append({
                "strategy":    "观望 / 等待方向明确",
                "reason":      f"多空信号均衡(多={bull_score}/空={bear_score})，IV正常，入场性价比低",
                "risk":        "-",
                "max_profit":  "-"
            })

    return suggestions


# ─────────────────────────────────────────────
# 主入口：综合分析
# ─────────────────────────────────────────────
def analyze(symbol: str, period: str = "6mo") -> None:
    """
    一键分析：输入股票代码，输出完整报告

    示例:
        analyze("NVDA")          # 美股
        analyze("600519")        # A股茅台
        analyze("AAPL")          # 美股苹果
        analyze("HSTECH")        # 恒生科技指数
    """
    print(f"\n{'='*60}")
    print(f"  交易分析报告：{symbol.upper()}")
    print(f"  生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # 市场识别
    market = detect_market(symbol)
    market_labels = {
        "US": "美股", "A_SHARE": "A股", "HSI_TECH": "恒生科技ETF", "HK": "港股"
    }
    print(f"\n📍 市场：{market_labels.get(market, market)}")

    # 获取价格数据
    print("\n⏳ 获取价格数据...")
    try:
        df = get_price_data(symbol, market, period)
        print(f"   数据范围：{df.index[0].date()} → {df.index[-1].date()}（{len(df)}个交易日）")
    except Exception as e:
        print(f"❌ 数据获取失败：{e}")
        return

    # 技术分析
    print("\n⏳ 运行技术分析...")
    tech = analyze_technical(df)

    current = tech["Current_Price"]
    change  = tech["Price_Change%"]
    arrow   = "▲" if change > 0 else "▼"

    print(f"\n{'─'*40}")
    print(f"📊 价格动态")
    print(f"{'─'*40}")
    print(f"  当前价：{current:.2f}   {arrow} {abs(change):.2f}%")
    ma60_str = f"{tech['MA60']:.2f}" if tech['MA60'] else 'N/A'
    print(f"  均线：MA5={tech['MA5']:.2f}  MA20={tech['MA20']:.2f}  MA60={ma60_str}")

    # 均线多头/空头排列
    if tech["MA5"] > tech["MA20"] and (tech["MA60"] is None or tech["MA20"] > tech["MA60"]):
        print(f"  均线排列：✅ 多头排列")
    elif tech["MA5"] < tech["MA20"] and (tech["MA60"] is None or tech["MA20"] < tech["MA60"]):
        print(f"  均线排列：🔴 空头排列")
    else:
        print(f"  均线排列：⚠️  交错中")

    print(f"\n{'─'*40}")
    print(f"📈 技术指标")
    print(f"{'─'*40}")

    # RSI 解读
    rsi = tech["RSI14"]
    rsi_note = "超买区间" if rsi > 70 else ("超卖区间" if rsi < 30 else "正常区间")
    print(f"  RSI(14): {rsi:.1f}  [{rsi_note}]")
    print(f"  RSI(6):  {tech['RSI6']:.1f}")

    # MACD 解读
    macd_note = f"柱状图{tech['MACD_Hist']:.3f} | {tech['MACD_Cross']}"
    print(f"  MACD: {tech['MACD']:.3f}  Signal: {tech['MACD_Signal']:.3f}  {macd_note}")

    # KDJ
    print(f"  KDJ: K={tech['KDJ_K']:.1f}  D={tech['KDJ_D']:.1f}  J={tech['KDJ_J']:.1f}")

    # 布林带
    bb_pos = tech["BB_Position%"]
    bb_note = "接近上轨" if bb_pos > 80 else ("接近下轨" if bb_pos < 20 else "中轨附近")
    print(f"  布林带: 上={tech['BB_Upper']:.2f}  中={tech['BB_Middle']:.2f}  下={tech['BB_Lower']:.2f}")
    print(f"  布林带位置: {bb_pos:.0f}%  [{bb_note}]  带宽={tech['BB_Width%']:.1f}%")

    # 成交量
    vol_ratio = tech.get("Volume_Ratio")
    if vol_ratio:
        vol_note = "放量" if vol_ratio > 1.5 else ("缩量" if vol_ratio < 0.7 else "正常")
        print(f"  成交量比: {vol_ratio:.2f}x  [{vol_note}]")

    print(f"  ATR(14): {tech['ATR14']:.2f}  (波动性参考)")
    print(f"  近20日高: {tech['Recent_High20']:.2f}  近20日低: {tech['Recent_Low20']:.2f}")

    # 蜡烛图形态
    print(f"\n{'─'*40}")
    print(f"🕯️  蜡烛图形态（近5日）")
    print(f"{'─'*40}")
    patterns = tech["Candle_Patterns"]
    if patterns:
        for p in sorted(patterns, key=lambda x: x["days_ago"]):
            day_label = "今日" if p["days_ago"] == 0 else f"{p['days_ago']}日前"
            direction_emoji = "🟢" if p["direction"] == "看涨" else ("🔴" if p["direction"] == "看跌" else "⚪")
            print(f"  {direction_emoji} [{day_label}] {p['name_zh']} ({p['pattern']})  →  {p['direction']}")
    else:
        print("  本周期内无显著蜡烛图形态")

    # A股专属：资金流向
    if market == "A_SHARE":
        print(f"\n{'─'*40}")
        print(f"💰 资金流向（A股）")
        print(f"{'─'*40}")
        flow = get_a_share_fund_flow(symbol)
        if "error" not in flow:
            main = flow["main_net_inflow"] / 1e8
            main_sign = "净流入" if main > 0 else "净流出"
            print(f"  今日主力: {main_sign} {abs(main):.2f}亿  占比{flow['main_net_ratio']:.1f}%")
            super_large = flow["super_large_net"] / 1e8
            print(f"  超大单（机构）: {'净流入' if super_large > 0 else '净流出'} {abs(super_large):.2f}亿")
            large = flow["large_net"] / 1e8
            print(f"  大单: {'净流入' if large > 0 else '净流出'} {abs(large):.2f}亿")
            if "main_net_3d" in flow:
                d3 = flow["main_net_3d"] / 1e8
                d5 = flow["main_net_5d"] / 1e8
                print(f"  近3日主力合计: {'净流入' if d3 > 0 else '净流出'} {abs(d3):.2f}亿")
                print(f"  近5日主力合计: {'净流入' if d5 > 0 else '净流出'} {abs(d5):.2f}亿")
        else:
            print(f"  资金流向获取失败: {flow['error']}")

        # 北向资金
        print(f"\n{'─'*40}")
        print(f"🌐 北向资金（今日市场层面）")
        print(f"{'─'*40}")
        north = get_northbound_flow()
        if "error" not in north:
            total = north["total_net"]
            print(f"  北向合计: {'净流入' if total > 0 else '净流出'} {abs(total):.2f}亿")
            for d in north.get("detail", []):
                print(f"  {d['板块']}: 净买额 {d['成交净买额']:.2f}亿")
        else:
            print(f"  北向数据: {north['error']}")

    # 美股专属：期权分析
    if market == "US":
        print(f"\n{'─'*40}")
        print(f"📋 期权链分析（美股）")
        print(f"{'─'*40}")
        print("  ⏳ 获取期权数据...")
        options = analyze_options(symbol, current)
        if "error" not in options:
            for exp, data in list(options.items())[:2]:  # 只显示前2个到期日
                print(f"\n  到期日 {exp}:")
                print(f"    ATM IV均值: {data['avg_atm_iv']:.1f}%")
                print(f"    最大痛点: {data['max_pain']}")
                if data["top_call"]:
                    tc = data["top_call"][0]
                    print(f"    成交最活跃 Call: {tc['strike']} strike  | 价格{tc['lastPrice']:.2f}  | IV{tc['impliedVolatility']*100:.0f}%  | 成交量{tc['volume']:,}")
                if data["top_put"]:
                    tp = data["top_put"][0]
                    print(f"    成交最活跃 Put:  {tp['strike']} strike  | 价格{tp['lastPrice']:.2f}  | IV{tp['impliedVolatility']*100:.0f}%  | 成交量{tp['volume']:,}")

            # 期权策略建议
            print(f"\n{'─'*40}")
            print(f"💡 期权策略建议")
            print(f"{'─'*40}")
            strategies = suggest_option_strategy(tech, options)
            for s in strategies:
                print(f"  📌 {s['strategy']}")
                print(f"     理由：{s['reason']}")
                print(f"     风险：{s['risk']}")
        else:
            print(f"  期权数据获取失败: {options['error']}")

    # 综合信号总结
    print(f"\n{'─'*40}")
    print(f"🎯 综合信号总结")
    print(f"{'─'*40}")

    signals = []
    # 均线
    if tech["MA5"] > tech["MA20"]:
        signals.append(("✅", "短期均线在长期均线上方，趋势偏多"))
    else:
        signals.append(("🔴", "短期均线在长期均线下方，趋势偏空"))
    # MACD
    if tech["MACD_Hist"] > 0:
        signals.append(("✅", f"MACD柱状图为正 {tech['MACD_Cross']}"))
    else:
        signals.append(("🔴", f"MACD柱状图为负 {tech['MACD_Cross']}"))
    # RSI
    if 40 < tech["RSI14"] < 70:
        signals.append(("✅", f"RSI={tech['RSI14']:.0f} 健康区间"))
    elif tech["RSI14"] >= 70:
        signals.append(("⚠️", f"RSI={tech['RSI14']:.0f} 超买，注意回调风险"))
    else:
        signals.append(("⚠️", f"RSI={tech['RSI14']:.0f} 超卖，可能企稳"))
    # 蜡烛图
    bullish = [p for p in patterns if p["direction"] == "看涨" and p["days_ago"] <= 2]
    bearish = [p for p in patterns if p["direction"] == "看跌" and p["days_ago"] <= 2]
    if bullish:
        names = "、".join([p["name_zh"] for p in bullish[:3]])
        signals.append(("✅", f"近2日出现看涨形态：{names}"))
    if bearish:
        names = "、".join([p["name_zh"] for p in bearish[:3]])
        signals.append(("🔴", f"近2日出现看跌形态：{names}"))

    bull_count = sum(1 for s in signals if s[0] == "✅")
    bear_count = sum(1 for s in signals if s[0] == "🔴")

    for emoji, text in signals:
        print(f"  {emoji} {text}")

    print(f"\n  多空信号比：{bull_count} 看涨 / {bear_count} 看跌", end="")
    if bull_count > bear_count + 1:
        print("  →  📈 整体偏多，可考虑逢低布局")
    elif bear_count > bull_count + 1:
        print("  →  📉 整体偏空，注意控制仓位")
    else:
        print("  →  ⚖️  信号混杂，建议等待更明确方向")

    print(f"\n{'='*60}")
    print("  ⚠️  以上分析仅供参考，不构成投资建议")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# 快速运行示例
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["NVDA"]
    for sym in symbols:
        analyze(sym)
