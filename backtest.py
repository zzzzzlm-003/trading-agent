"""
Backtesting Module
==================
基于 signal_generator.py 信号的历史回测引擎

功能：
- 基于滚动技术指标信号（BUY/SELL/HOLD）驱动的策略回测
- 关键指标：总收益、年化收益、最大回撤、夏普比率、胜率、盈亏比
- 可视化权益曲线（输出 PNG 图片）
- 对比基准 Buy & Hold 策略

依赖：yfinance, pandas, numpy, matplotlib（无需 TA-Lib）
安装：pip install yfinance pandas numpy matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")   # 无 GUI 模式，避免 display 问题
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os
import sys
from datetime import datetime
from typing import Optional

# 从同目录导入信号生成函数
try:
    from signal_generator import (
        _fetch_ohlcv,
        _analyze_tf,
        calc_sma, calc_ema, calc_rsi,
        calc_macd, calc_bbands, calc_atr,
        calc_kdj, calc_volume_ratio
    )
    _HAS_SIG_GEN = True
except ImportError:
    _HAS_SIG_GEN = False
    # 内联基础指标（兜底）
    def calc_sma(s, w): return s.rolling(w, min_periods=1).mean()
    def calc_ema(s, w): return s.ewm(span=w, adjust=False).mean()
    def calc_rsi(s, period=14):
        delta = s.diff(); g = delta.clip(lower=0); l = (-delta).clip(lower=0)
        ag = g.ewm(com=period-1, min_periods=period).mean()
        al = l.ewm(com=period-1, min_periods=period).mean()
        return 100 - 100/(1 + ag/al.replace(0, np.nan))
    def calc_macd(s, f=12, sl=26, sig=9):
        ml = calc_ema(s,f)-calc_ema(s,sl); sl_ = calc_ema(ml,sig); return ml, sl_, ml-sl_
    def calc_atr(h,l,c,p=14):
        tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        return tr.ewm(com=p-1, min_periods=p).mean()
    def _fetch_ohlcv(symbol, period="2y"):
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[["Open","High","Low","Close","Volume"]].astype(float)


# ─────────────────────────────────────────────────────────────
# 策略信号生成（轻量版，不需要网络二次请求）
# ─────────────────────────────────────────────────────────────

def _fast_signal(window: pd.DataFrame, lookback: int = 60) -> int:
    """
    在给定窗口内快速生成交易信号
    返回：1（买入）/ -1（卖出）/ 0（观望）
    """
    if len(window) < lookback:
        return 0

    close  = window["Close"]
    high   = window["High"]
    low    = window["Low"]
    volume = window["Volume"]

    score = 0  # 累计分，正=偏多，负=偏空

    # 均线
    ma5  = calc_sma(close, 5)
    ma20 = calc_sma(close, 20)
    if ma5.iloc[-1] > ma20.iloc[-1]:
        score += 20
    else:
        score -= 20

    # 均线交叉
    if len(ma5) > 2:
        if ma5.iloc[-2] <= ma20.iloc[-2] and ma5.iloc[-1] > ma20.iloc[-1]:
            score += 15  # 金叉
        elif ma5.iloc[-2] >= ma20.iloc[-2] and ma5.iloc[-1] < ma20.iloc[-1]:
            score -= 15  # 死叉

    # MACD
    _, _, hist = calc_macd(close)
    if hist.iloc[-1] > 0:
        score += 15 if hist.iloc[-1] > hist.iloc[-2] else 5
    else:
        score -= 15 if hist.iloc[-1] < hist.iloc[-2] else 5

    # RSI
    rsi = calc_rsi(close, 14).iloc[-1]
    if rsi > 70: score -= 10
    elif rsi > 55: score += 10
    elif rsi < 30: score += 5
    elif rsi < 45: score -= 10

    # 成交量确认
    vol_ma = volume.rolling(20).mean()
    vr = volume.iloc[-1] / (vol_ma.iloc[-1] + 1e-10)
    price_up = close.iloc[-1] > close.iloc[-2]
    if vr > 1.5 and price_up: score += 8
    elif vr > 1.5 and not price_up: score -= 8

    # 映射到 -100~100 → 信号
    if score >= 30:
        return 1
    elif score <= -30:
        return -1
    return 0


# ─────────────────────────────────────────────────────────────
# 回测引擎
# ─────────────────────────────────────────────────────────────

class BacktestResult:
    """回测结果容器"""
    def __init__(self):
        self.symbol          = ""
        self.start_date      = None
        self.end_date        = None
        self.initial_capital = 0.0
        self.final_equity    = 0.0
        self.total_return    = 0.0
        self.annual_return   = 0.0
        self.max_drawdown    = 0.0
        self.sharpe_ratio    = 0.0
        self.sortino_ratio   = 0.0
        self.win_rate        = 0.0
        self.profit_factor   = 0.0
        self.total_trades    = 0
        self.winning_trades  = 0
        self.losing_trades   = 0
        self.avg_win         = 0.0
        self.avg_loss        = 0.0
        self.calmar_ratio    = 0.0
        self.equity_curve    = pd.Series(dtype=float)
        self.benchmark_curve = pd.Series(dtype=float)
        self.trades          = []       # [(entry_date, exit_date, entry_price, exit_price, pnl%)]
        self.drawdown_series = pd.Series(dtype=float)

    def summary(self) -> str:
        years = max((self.end_date - self.start_date).days / 365.25, 0.01)
        lines = [
            f"\n{'='*56}",
            f"  📊 回测结果：{self.symbol}",
            f"  时间范围：{self.start_date.date()} → {self.end_date.date()}（{years:.1f}年）",
            f"{'='*56}",
            f"  初始资金：${self.initial_capital:,.0f}",
            f"  期末资金：${self.final_equity:,.0f}",
            f"  总收益率：{self.total_return:+.2f}%",
            f"  年化收益：{self.annual_return:+.2f}%",
            f"  最大回撤：{self.max_drawdown:.2f}%",
            f"  夏普比率：{self.sharpe_ratio:.3f}",
            f"  索提诺比率：{self.sortino_ratio:.3f}",
            f"  卡玛比率：{self.calmar_ratio:.3f}",
            f"  {'─'*50}",
            f"  总交易次数：{self.total_trades}",
            f"  盈利次数：{self.winning_trades}  亏损次数：{self.losing_trades}",
            f"  胜率：{self.win_rate:.1f}%",
            f"  平均盈利：{self.avg_win:.2f}%  平均亏损：{self.avg_loss:.2f}%",
            f"  盈亏比（Profit Factor）：{self.profit_factor:.2f}",
            f"  {'─'*50}",
            f"  基准（Buy&Hold）回报：{self._bh_return():.2f}%",
            f"  超额收益（Alpha）：{self.total_return - self._bh_return():.2f}%",
            f"{'='*56}",
        ]
        return "\n".join(lines)

    def _bh_return(self) -> float:
        if self.benchmark_curve.empty: return 0.0
        return (self.benchmark_curve.iloc[-1] / self.benchmark_curve.iloc[0] - 1) * 100


def run_backtest(
    symbol: str,
    period: str = "3y",
    initial_capital: float = 100_000.0,
    commission: float = 0.001,      # 单边手续费 0.1%
    signal_lookback: int = 60,      # 信号计算回望期（交易日）
    stop_loss_atr: float = 2.0,     # ATR 止损倍数（0 = 不止损）
    take_profit_atr: float = 4.0,   # ATR 止盈倍数（0 = 不止盈）
    output_dir: str = ".",          # 图表输出目录
    plot: bool = True,
    verbose: bool = True,
) -> BacktestResult:
    """
    运行历史回测

    参数：
        symbol          股票代码
        period          回测历史数据长度（yfinance 格式，如 "3y"）
        initial_capital 初始资金
        commission      单边手续费（小数，如 0.001 = 0.1%）
        signal_lookback 滚动信号计算的最小历史窗口
        stop_loss_atr   ATR 止损倍数（0 = 关闭）
        take_profit_atr ATR 止盈倍数（0 = 关闭）
        output_dir      输出 PNG 目录
        plot            是否生成图表
        verbose         是否打印结果

    返回：
        BacktestResult 对象
    """
    res = BacktestResult()
    res.symbol          = symbol
    res.initial_capital = initial_capital

    # ── 1. 获取数据 ──
    if verbose:
        print(f"\n⏳ 获取 {symbol} 历史数据（{period}）...")
    df = _fetch_ohlcv(symbol, period=period)
    df = df.dropna()

    if len(df) < signal_lookback + 10:
        raise ValueError(f"历史数据不足，需要至少 {signal_lookback + 10} 个交易日")

    res.start_date = df.index[signal_lookback]
    res.end_date   = df.index[-1]

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ── 2. 预计算 ATR（用于止损止盈） ──
    atr_series = calc_atr(high, low, close, 14)

    # ── 3. 滚动生成信号序列 ──
    if verbose:
        print(f"⏳ 生成历史信号序列（约需 {len(df) - signal_lookback} 次计算）...")

    raw_signals = []
    for i in range(signal_lookback, len(df)):
        window = df.iloc[max(0, i - signal_lookback): i + 1]
        raw_signals.append((df.index[i], _fast_signal(window, signal_lookback)))

    sig_df = pd.DataFrame(raw_signals, columns=["date", "signal"]).set_index("date")
    sig_series = sig_df["signal"]

    # ── 4. 模拟交易（状态机） ──
    equity    = initial_capital
    position  = 0.0       # 持仓量（股数）
    entry_price = 0.0
    entry_date  = None
    stop_price  = None
    take_price  = None

    equity_curve = {}
    trades       = []

    for i in range(signal_lookback, len(df)):
        date  = df.index[i]
        price = close.iloc[i]
        atr   = atr_series.iloc[i]
        sig   = sig_series.get(date, 0)

        # 记录当日权益
        equity_curve[date] = equity + position * price

        # ── 止损 / 止盈检查（盘中用 High/Low 触发） ──
        if position > 0:
            day_low  = low.iloc[i]
            day_high = high.iloc[i]

            triggered = False
            exit_reason = ""
            exit_price_ = price

            if stop_loss_atr > 0 and stop_price and day_low <= stop_price:
                triggered   = True
                exit_reason = "止损"
                exit_price_ = stop_price   # 按止损价成交

            elif take_profit_atr > 0 and take_price and day_high >= take_price:
                triggered   = True
                exit_reason = "止盈"
                exit_price_ = take_price

            if triggered:
                pnl = (exit_price_ / entry_price - 1) * 100
                proceeds = position * exit_price_ * (1 - commission)
                equity   = equity + proceeds
                trades.append({
                    "entry_date":  entry_date,
                    "exit_date":   date,
                    "entry_price": entry_price,
                    "exit_price":  exit_price_,
                    "pnl%":        pnl,
                    "reason":      exit_reason,
                })
                position    = 0
                entry_price = 0
                stop_price  = None
                take_price  = None
                equity_curve[date] = equity  # 更新
                continue

        # ── 信号执行 ──
        if sig == 1 and position == 0:
            # 买入：全仓
            buy_price = price * (1 + commission)
            position  = equity / buy_price
            equity    = 0.0
            entry_price = buy_price
            entry_date  = date
            # 设置止损止盈
            stop_price = (buy_price - stop_loss_atr * atr) if stop_loss_atr > 0 else None
            take_price = (buy_price + take_profit_atr * atr) if take_profit_atr > 0 else None

        elif sig == -1 and position > 0:
            # 卖出：平仓
            sell_price = price * (1 - commission)
            pnl        = (sell_price / entry_price - 1) * 100
            proceeds   = position * sell_price
            equity     = equity + proceeds
            trades.append({
                "entry_date":  entry_date,
                "exit_date":   date,
                "entry_price": entry_price,
                "exit_price":  sell_price,
                "pnl%":        pnl,
                "reason":      "信号卖出",
            })
            position    = 0
            entry_price = 0
            stop_price  = None
            take_price  = None
            equity_curve[date] = equity  # 更新

    # 最终若仍持仓，按最后收盘价平仓
    if position > 0:
        last_price = close.iloc[-1] * (1 - commission)
        pnl = (last_price / entry_price - 1) * 100
        proceeds = position * last_price
        equity   = equity + proceeds
        trades.append({
            "entry_date":  entry_date,
            "exit_date":   df.index[-1],
            "entry_price": entry_price,
            "exit_price":  last_price,
            "pnl%":        pnl,
            "reason":      "强制平仓（期末）",
        })
        equity_curve[df.index[-1]] = equity

    # ── 5. 计算统计指标 ──
    eq_series = pd.Series(equity_curve).sort_index()
    res.equity_curve    = eq_series
    res.benchmark_curve = close.loc[eq_series.index]

    res.final_equity  = eq_series.iloc[-1]
    res.total_return  = (res.final_equity / initial_capital - 1) * 100

    years = max((res.end_date - res.start_date).days / 365.25, 0.01)
    res.annual_return = ((res.final_equity / initial_capital) ** (1 / years) - 1) * 100

    # 最大回撤
    rolling_max  = eq_series.cummax()
    drawdown     = (eq_series - rolling_max) / rolling_max * 100
    res.drawdown_series = drawdown
    res.max_drawdown    = abs(drawdown.min())

    # 夏普比率（日收益率年化）
    daily_ret = eq_series.pct_change().dropna()
    if daily_ret.std() > 0:
        res.sharpe_ratio = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    else:
        res.sharpe_ratio = 0.0

    # 索提诺比率
    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and downside.std() > 0:
        res.sortino_ratio = daily_ret.mean() / downside.std() * np.sqrt(252)
    else:
        res.sortino_ratio = 0.0

    # 卡玛比率
    if res.max_drawdown > 0:
        res.calmar_ratio = res.annual_return / res.max_drawdown
    else:
        res.calmar_ratio = float('inf')

    # 交易统计
    res.trades        = trades
    res.total_trades  = len(trades)
    winning = [t for t in trades if t["pnl%"] > 0]
    losing  = [t for t in trades if t["pnl%"] <= 0]
    res.winning_trades = len(winning)
    res.losing_trades  = len(losing)
    res.win_rate       = len(winning) / max(len(trades), 1) * 100
    res.avg_win        = np.mean([t["pnl%"] for t in winning]) if winning else 0.0
    res.avg_loss       = np.mean([t["pnl%"] for t in losing])  if losing  else 0.0
    gross_profit = sum(t["pnl%"] for t in winning) if winning else 0
    gross_loss   = abs(sum(t["pnl%"] for t in losing)) if losing else 0
    res.profit_factor  = gross_profit / max(gross_loss, 0.01)

    if verbose:
        print(res.summary())

    # ── 6. 绘图 ──
    if plot:
        out_path = _plot_results(res, output_dir)
        if verbose and out_path:
            print(f"\n📊 权益曲线图已保存至: {out_path}\n")

    return res


# ─────────────────────────────────────────────────────────────
# 可视化：权益曲线 + 回撤 + 信号
# ─────────────────────────────────────────────────────────────

def _plot_results(res: BacktestResult, output_dir: str = ".") -> Optional[str]:
    """生成回测可视化图表（PNG）"""
    try:
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor("#0d0d0d")
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)

        ax1 = fig.add_subplot(gs[0])  # 权益曲线
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # 回撤
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # 收益分布

        _style_ax(ax1); _style_ax(ax2); _style_ax(ax3)

        eq  = res.equity_curve
        bm  = res.benchmark_curve
        dd  = res.drawdown_series

        # ── 权益曲线 ──
        # 标准化到初始资金=100
        eq_norm = eq / eq.iloc[0] * 100
        bm_norm = bm / bm.iloc[0] * 100

        ax1.plot(eq_norm.index, eq_norm.values, color="#00d4ff", linewidth=1.8,
                 label=f"策略权益曲线 (+{res.total_return:.1f}%)", zorder=3)
        ax1.plot(bm_norm.index, bm_norm.values, color="#888888", linewidth=1.2,
                 linestyle="--", label=f"Buy & Hold (+{res._bh_return():.1f}%)", zorder=2)
        ax1.fill_between(eq_norm.index, eq_norm.values, 100,
                         where=eq_norm.values >= 100,
                         color="#00d4ff", alpha=0.08)
        ax1.fill_between(eq_norm.index, eq_norm.values, 100,
                         where=eq_norm.values < 100,
                         color="#ff4444", alpha=0.08)
        ax1.axhline(100, color="#555555", linewidth=0.8, linestyle=":")

        # 标记入场/出场点
        for t in res.trades:
            ed = t["entry_date"]
            xd = t["exit_date"]
            if ed in eq_norm.index:
                ax1.axvline(ed, color="#22ff77", alpha=0.25, linewidth=0.7)
            if xd in eq_norm.index:
                color = "#ff4444" if t["pnl%"] < 0 else "#22ff77"
                ax1.axvline(xd, color=color, alpha=0.25, linewidth=0.7)

        # 统计信息文本框
        stats_text = (
            f"总收益: {res.total_return:+.1f}%  年化: {res.annual_return:+.1f}%\n"
            f"最大回撤: -{res.max_drawdown:.1f}%  夏普: {res.sharpe_ratio:.2f}\n"
            f"胜率: {res.win_rate:.0f}%  盈亏比: {res.profit_factor:.2f}  "
            f"交易次数: {res.total_trades}"
        )
        ax1.text(0.01, 0.97, stats_text, transform=ax1.transAxes,
                 color="#cccccc", fontsize=9, verticalalignment="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                           edgecolor="#333355", alpha=0.85))

        ax1.set_ylabel("净值（初始=100）", color="#aaaaaa", fontsize=9)
        ax1.set_title(f"{res.symbol}  策略回测  {res.start_date.date()} → {res.end_date.date()}",
                      color="#ffffff", fontsize=12, fontweight="bold", pad=10)
        ax1.legend(loc="lower right", facecolor="#1a1a1a", edgecolor="#444444",
                   labelcolor="#cccccc", fontsize=9)
        ax1.set_ylim(bottom=min(eq_norm.min(), bm_norm.min()) * 0.97)

        # ── 最大回撤 ──
        ax2.fill_between(dd.index, dd.values, 0, color="#ff4444", alpha=0.5)
        ax2.plot(dd.index, dd.values, color="#ff6666", linewidth=0.8)
        ax2.axhline(0, color="#555555", linewidth=0.5)
        ax2.set_ylabel("回撤 %", color="#aaaaaa", fontsize=9)
        max_dd_date = dd.idxmin()
        ax2.annotate(f"  最大回撤\n  -{res.max_drawdown:.1f}%",
                     xy=(max_dd_date, dd.min()),
                     color="#ff8888", fontsize=8,
                     xytext=(max_dd_date, dd.min() * 0.6),
                     arrowprops=dict(arrowstyle="->", color="#ff6666", lw=0.8))

        # ── 每笔交易盈亏条形图 ──
        if res.trades:
            pnls  = [t["pnl%"] for t in res.trades]
            dates = [t["exit_date"] for t in res.trades]
            colors = ["#22cc66" if p > 0 else "#cc3333" for p in pnls]
            ax3.bar(dates, pnls, color=colors, width=3, alpha=0.8)
            ax3.axhline(0, color="#555555", linewidth=0.5)
            avg_win_line  = res.avg_win
            avg_loss_line = res.avg_loss
            ax3.axhline(avg_win_line,  color="#22cc66", linewidth=0.8, linestyle="--", alpha=0.6)
            ax3.axhline(avg_loss_line, color="#cc3333", linewidth=0.8, linestyle="--", alpha=0.6)
            ax3.set_ylabel("单笔盈亏 %", color="#aaaaaa", fontsize=9)

        # 统一 x 轴格式
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(colors="#888888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        fig.text(0.5, 0.01,
                 "⚠️ 以上回测结果仅供参考，历史表现不代表未来收益，不构成投资建议",
                 ha="center", color="#666666", fontsize=8)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        filename = f"backtest_{res.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_path

    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")
        return None


def _style_ax(ax):
    ax.set_facecolor("#111111")
    ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


# ─────────────────────────────────────────────────────────────
# 多策略参数对比（网格搜索）
# ─────────────────────────────────────────────────────────────

def parameter_sweep(
    symbol: str,
    period: str = "3y",
    stop_loss_atrs: list = [1.5, 2.0, 3.0],
    take_profit_atrs: list = [3.0, 4.0, 6.0],
    initial_capital: float = 100_000.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    对不同止损止盈参数进行网格搜索，找出最优组合

    参数：
        symbol           股票代码
        period           历史数据长度
        stop_loss_atrs   止损 ATR 倍数列表
        take_profit_atrs 止盈 ATR 倍数列表
        initial_capital  初始资金
        verbose          是否打印结果

    返回：
        DataFrame，包含每种参数组合的回测指标
    """
    records = []
    configs = [(sl, tp) for sl in stop_loss_atrs for tp in take_profit_atrs]
    total   = len(configs)

    if verbose:
        print(f"\n🔍 {symbol} 参数网格搜索（{total} 种组合）...\n")

    for i, (sl, tp) in enumerate(configs, 1):
        if verbose:
            print(f"  [{i}/{total}] 止损={sl}x ATR  止盈={tp}x ATR ...", end=" ", flush=True)
        try:
            r = run_backtest(symbol, period=period, initial_capital=initial_capital,
                             stop_loss_atr=sl, take_profit_atr=tp,
                             plot=False, verbose=False)
            records.append({
                "止损ATR倍数":   sl,
                "止盈ATR倍数":   tp,
                "总收益%":       round(r.total_return, 2),
                "年化收益%":     round(r.annual_return, 2),
                "最大回撤%":     round(r.max_drawdown, 2),
                "夏普比率":      round(r.sharpe_ratio, 3),
                "胜率%":         round(r.win_rate, 1),
                "盈亏比":        round(r.profit_factor, 2),
                "卡玛比率":      round(r.calmar_ratio, 3),
                "交易次数":      r.total_trades,
            })
            if verbose:
                print(f"年化={r.annual_return:+.1f}%  回撤={r.max_drawdown:.1f}%  "
                      f"夏普={r.sharpe_ratio:.2f}")
        except Exception as e:
            if verbose:
                print(f"❌ {e}")

    df = pd.DataFrame(records).sort_values("夏普比率", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  参数搜索结果（按夏普比率排序）")
        print(f"{'='*70}")
        print(df.to_string(index=True))
        print(f"{'='*70}")
        print(f"\n  最优组合（夏普最高）：止损={df.iloc[0]['止损ATR倍数']}x  "
              f"止盈={df.iloc[0]['止盈ATR倍数']}x")

    return df


# ─────────────────────────────────────────────────────────────
# 主程序示例
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    # 默认参数
    symbol  = args[0] if args else "NVDA"
    period  = args[1] if len(args) > 1 else "3y"

    print(f"📊 回测：{symbol}  历史数据：{period}")
    print(f"   初始资金：$100,000  手续费：0.1%  止损：2x ATR  止盈：4x ATR")
    print(f"   信号：多时间框架技术指标综合评分（日线滚动60期）\n")

    # 获取脚本所在目录作为输出目录
    output_dir = os.path.dirname(os.path.abspath(__file__))

    result = run_backtest(
        symbol         = symbol,
        period         = period,
        initial_capital= 100_000.0,
        commission     = 0.001,
        signal_lookback= 60,
        stop_loss_atr  = 2.0,
        take_profit_atr= 4.0,
        output_dir     = output_dir,
        plot           = True,
        verbose        = True,
    )

    # 打印每笔交易明细
    if result.trades:
        print(f"\n📋 最近5笔交易明细：")
        print(f"  {'入场日期':12s} {'出场日期':12s} {'入场价':>8s} {'出场价':>8s} {'盈亏%':>8s} {'原因':8s}")
        print(f"  {'─'*64}")
        for t in result.trades[-5:]:
            pnl_str = f"{t['pnl%']:+.2f}%"
            pnl_clr = "✅" if t["pnl%"] > 0 else "🔴"
            print(f"  {str(t['entry_date'].date()):12s} {str(t['exit_date'].date()):12s} "
                  f"{t['entry_price']:>8.2f} {t['exit_price']:>8.2f} "
                  f"{pnl_str:>8s} {pnl_clr} {t['reason']}")
