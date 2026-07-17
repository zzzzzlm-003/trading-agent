"""
Risk Management Module
======================
投资组合风险控制模块

功能：
- ATR 动态止损（1.5x / 2x / 3x ATR，自动追踪移动止损）
- Kelly Criterion 头寸定规（全 Kelly / 半 Kelly）
- 投资组合层面风险敞口控制（单股上限、相关性控制、VaR）
- 风险报告生成

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

# 从同目录导入
try:
    from signal_generator import _fetch_ohlcv, calc_atr, calc_sma, calc_ema
    _HAS_SIG_GEN = True
except ImportError:
    _HAS_SIG_GEN = False
    def calc_atr(h, l, c, p=14):
        tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        return tr.ewm(com=p-1, min_periods=p).mean()
    def calc_sma(s, w): return s.rolling(w, min_periods=1).mean()
    def _fetch_ohlcv(symbol, period="1y"):
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[["Open","High","Low","Close","Volume"]].astype(float)


# ─────────────────────────────────────────────────────────────
# ATR 动态止损计算
# ─────────────────────────────────────────────────────────────

class ATRStopLoss:
    """
    基于 ATR 的动态止损计算器

    支持：
    - 固定 ATR 倍数止损（初始止损）
    - 追踪止损（随价格上涨自动上移）
    - 时间止损（持仓超过 N 天自动提示）
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,    # 止损 ATR 倍数
        trailing: bool = True,           # 是否启用追踪止损
        trailing_multiplier: float = 1.5, # 追踪止损 ATR 倍数（通常比初始止损紧）
        max_hold_days: int = 30,         # 最大持仓天数（时间止损）
    ):
        self.atr_mult      = atr_multiplier
        self.trailing      = trailing
        self.trail_mult    = trailing_multiplier
        self.max_hold_days = max_hold_days

    def compute_initial_stop(self, entry_price: float, atr: float) -> dict:
        """
        计算初始止损价格

        参数：
            entry_price  入场价格
            atr          当前 ATR14

        返回：止损信息字典
        """
        stop_price    = entry_price - self.atr_mult * atr
        stop_pct      = (entry_price - stop_price) / entry_price * 100
        risk_per_unit = entry_price - stop_price

        return {
            "entry_price":    entry_price,
            "atr":            atr,
            "stop_price":     round(stop_price, 4),
            "stop_pct%":      round(stop_pct, 2),
            "risk_per_unit":  round(risk_per_unit, 4),
            "atr_multiplier": self.atr_mult,
        }

    def update_trailing_stop(
        self,
        current_price: float,
        current_atr: float,
        current_stop: float,
        highest_since_entry: float,
    ) -> dict:
        """
        更新追踪止损（只上移，不下移）

        参数：
            current_price        当前价格
            current_atr          当前 ATR
            current_stop         当前止损价
            highest_since_entry  入场以来最高价

        返回：更新后的止损信息
        """
        new_trail_stop = highest_since_entry - self.trail_mult * current_atr
        new_stop = max(current_stop, new_trail_stop)  # 只上移

        profit_locked = (new_stop - current_stop) > 0
        profit_pct    = (current_price - new_stop) / current_price * 100

        return {
            "current_price":      current_price,
            "highest_since_entry": highest_since_entry,
            "trail_stop":         round(new_trail_stop, 4),
            "effective_stop":     round(new_stop, 4),
            "stop_updated":       profit_locked,
            "remaining_upside_to_stop%": round(profit_pct, 2),
        }

    def simulate_trailing_stop(
        self,
        symbol: str,
        entry_date: str,
        entry_price: Optional[float] = None,
        period: str = "6mo",
    ) -> dict:
        """
        对历史数据模拟追踪止损，展示何时会被触发

        参数：
            symbol      股票代码
            entry_date  入场日期（格式：YYYY-MM-DD）
            entry_price 入场价（None 则用入场日收盘价）
            period      获取历史数据长度

        返回：模拟结果字典
        """
        df = _fetch_ohlcv(symbol, period=period)
        df = df.loc[entry_date:]

        if df.empty:
            return {"error": f"{entry_date} 之后无数据"}

        atr_series = calc_atr(df["High"], df["Low"], df["Close"], 14)

        if entry_price is None:
            entry_price = float(df["Close"].iloc[0])

        initial_atr   = float(atr_series.iloc[0])
        initial_stop  = entry_price - self.atr_mult * initial_atr
        current_stop  = initial_stop
        highest       = entry_price

        trail_history = []
        stop_triggered = None

        for i in range(len(df)):
            row     = df.iloc[i]
            date    = df.index[i]
            price   = float(row["Close"])
            high    = float(row["High"])
            low     = float(row["Low"])
            atr_val = float(atr_series.iloc[i])

            # 更新最高价
            if high > highest:
                highest = high

            # 更新追踪止损
            if self.trailing:
                new_trail = highest - self.trail_mult * atr_val
                current_stop = max(current_stop, new_trail)

            # 检查是否触发
            if low <= current_stop:
                stop_triggered = {
                    "date":        date,
                    "trigger_price": current_stop,
                    "close":       price,
                    "days_held":   i,
                    "pnl%":        (current_stop / entry_price - 1) * 100,
                }
                break

            # 时间止损
            if i >= self.max_hold_days:
                stop_triggered = {
                    "date":        date,
                    "trigger_price": price,
                    "close":       price,
                    "days_held":   i,
                    "pnl%":        (price / entry_price - 1) * 100,
                    "reason":      "时间止损",
                }
                break

            trail_history.append({
                "date":        date,
                "close":       price,
                "stop":        round(current_stop, 4),
                "highest":     round(highest, 4),
                "pnl%":        round((price / entry_price - 1) * 100, 2),
            })

        result = {
            "symbol":         symbol,
            "entry_date":     entry_date,
            "entry_price":    entry_price,
            "initial_stop":   round(initial_stop, 4),
            "initial_atr":    round(initial_atr, 4),
            "atr_multiplier": self.atr_mult,
            "trail_multiplier": self.trail_mult,
            "stop_triggered": stop_triggered,
            "still_open":     stop_triggered is None,
            "current_stop":   round(current_stop, 4),
            "highest":        round(highest, 4),
            "trail_history":  trail_history,
        }

        if stop_triggered is None and trail_history:
            last = trail_history[-1]
            result["current_pnl%"] = last["pnl%"]

        return result

    def print_stop_report(self, symbol: str, entry_date: str,
                          entry_price: Optional[float] = None):
        """打印追踪止损模拟报告"""
        r = self.simulate_trailing_stop(symbol, entry_date, entry_price)

        w = 56
        print(f"\n{'='*w}")
        print(f"  🛡️  {r['symbol']} ATR 追踪止损模拟报告")
        print(f"{'='*w}")
        print(f"  入场日期：{r['entry_date']}")
        print(f"  入场价格：{r['entry_price']:.2f}")
        print(f"  初始 ATR(14)：{r['initial_atr']:.2f}")
        print(f"  初始止损价：{r['initial_stop']:.2f}（{r['atr_multiplier']}x ATR）")
        print(f"  追踪止损倍数：{r['trail_multiplier']}x ATR")
        print(f"  {'─'*50}")

        if r.get("stop_triggered"):
            t = r["stop_triggered"]
            emoji = "✅" if t["pnl%"] > 0 else "🔴"
            print(f"  🔔 止损触发：{t['date'].date()}（持仓{t['days_held']}天）")
            print(f"  触发价格：{t['trigger_price']:.2f}")
            print(f"  盈亏：{emoji} {t['pnl%']:+.2f}%")
        else:
            print(f"  📊 持仓中（截至最新数据）")
            print(f"  当前追踪止损：{r['current_stop']:.2f}")
            print(f"  入场以来最高：{r['highest']:.2f}")
            print(f"  当前盈亏：{r.get('current_pnl%', 0):+.2f}%")

        print(f"{'='*w}\n")


# ─────────────────────────────────────────────────────────────
# Kelly Criterion 头寸定规
# ─────────────────────────────────────────────────────────────

class KellyCriterion:
    """
    Kelly 公式头寸定规

    公式：f* = (p * b - q) / b
    其中：
        p = 胜率
        q = 败率 = 1 - p
        b = 平均盈利 / 平均亏损（盈亏比）
    """

    def __init__(self, fraction: float = 0.5):
        """
        参数：
            fraction  Kelly 分数（1.0 = 全 Kelly，0.5 = 半 Kelly 推荐）
        """
        self.fraction = fraction

    def calculate(
        self,
        win_rate: float,         # 胜率（0~1）
        avg_win_pct: float,      # 平均单笔盈利%（正数）
        avg_loss_pct: float,     # 平均单笔亏损%（正数）
        capital: float = 100_000,
        entry_price: float = 100,
        verbose: bool = True,
    ) -> dict:
        """
        计算最优头寸大小

        参数：
            win_rate     胜率（0~1，如 0.55 = 55%）
            avg_win_pct  平均盈利百分比（如 8.0 = 8%）
            avg_loss_pct 平均亏损百分比（正数，如 3.0 = 3%）
            capital      总资金
            entry_price  入场股价
            verbose      是否打印

        返回：
            {
                "kelly_pct":      Kelly 比例（%）,
                "adjusted_pct":   调整后比例（fraction * kelly，推荐使用）,
                "dollar_size":    建议投入金额,
                "share_size":     建议买入股数,
                "max_loss_dollar": 最大单笔亏损金额,
                "edge":           期望值（正=有优势）,
            }
        """
        if win_rate <= 0 or win_rate >= 1:
            raise ValueError("胜率必须在 0~1 之间（不含）")
        if avg_win_pct <= 0 or avg_loss_pct <= 0:
            raise ValueError("盈亏百分比必须为正数")

        lose_rate = 1 - win_rate
        b = avg_win_pct / avg_loss_pct  # 盈亏比

        kelly_pct   = (win_rate * b - lose_rate) / b * 100
        kelly_pct   = max(0, kelly_pct)   # 不做空（Kelly为负则不交易）
        adj_pct     = kelly_pct * self.fraction  # 调整后 Kelly

        dollar_size = capital * adj_pct / 100
        share_size  = int(dollar_size / entry_price) if entry_price > 0 else 0
        max_loss    = dollar_size * avg_loss_pct / 100
        edge        = win_rate * avg_win_pct - lose_rate * avg_loss_pct  # 期望盈利%

        result = {
            "win_rate":           win_rate,
            "lose_rate":          lose_rate,
            "avg_win_pct":        avg_win_pct,
            "avg_loss_pct":       avg_loss_pct,
            "profit_factor":      b,
            "edge_pct":           round(edge, 3),
            "kelly_pct":          round(kelly_pct, 2),
            "kelly_fraction":     self.fraction,
            "adjusted_pct":       round(adj_pct, 2),
            "dollar_size":        round(dollar_size, 2),
            "share_size":         share_size,
            "max_loss_dollar":    round(max_loss, 2),
            "max_loss_pct_cap":   round(adj_pct * avg_loss_pct / 100, 2),
        }

        if verbose:
            self._print_report(result, capital, entry_price)
        return result

    def _print_report(self, r: dict, capital: float, entry_price: float):
        w = 56
        print(f"\n{'='*w}")
        print(f"  🎯  Kelly Criterion 头寸定规")
        print(f"{'='*w}")
        print(f"  总资金：${capital:,.0f}  入场价：${entry_price:.2f}")
        print(f"  {'─'*50}")
        print(f"  胜率：{r['win_rate']*100:.1f}%  败率：{r['lose_rate']*100:.1f}%")
        print(f"  平均盈利：{r['avg_win_pct']:.2f}%  平均亏损：{r['avg_loss_pct']:.2f}%")
        print(f"  盈亏比（b）：{r['profit_factor']:.2f}")
        print(f"  期望值（Edge）：{r['edge_pct']:+.2f}%/笔",
              end="" if r["edge_pct"] > 0 else "\n")
        if r["edge_pct"] > 0:
            print("  ✅ 正期望，可以交易")
        else:
            print("  ❌ 负期望，建议不交易")
        print(f"  {'─'*50}")
        print(f"  全 Kelly 仓位：{r['kelly_pct']:.2f}% 资金")
        print(f"  调整后 Kelly（×{r['kelly_fraction']}）：{r['adjusted_pct']:.2f}% 资金")
        print(f"  {'─'*50}")
        print(f"  建议投入金额：${r['dollar_size']:,.2f}")
        print(f"  建议股数：{r['share_size']:,} 股")
        print(f"  最大单笔亏损：${r['max_loss_dollar']:,.2f}（占总资金 {r['max_loss_pct_cap']:.2f}%）")
        print(f"{'='*w}\n")

    def calculate_from_backtest(
        self,
        trades: list,
        capital: float = 100_000,
        entry_price: float = 100,
        verbose: bool = True,
    ) -> dict:
        """
        直接从 backtest.py 的 trades 列表计算 Kelly

        参数：
            trades  BacktestResult.trades 列表
        """
        if not trades:
            raise ValueError("无交易记录")

        pnls = [t["pnl%"] for t in trades]
        wins = [p for p in pnls if p > 0]
        loss = [abs(p) for p in pnls if p <= 0]

        win_rate    = len(wins) / len(pnls)
        avg_win     = np.mean(wins)  if wins else 0.01
        avg_loss    = np.mean(loss)  if loss else 0.01

        return self.calculate(win_rate, avg_win, avg_loss,
                              capital, entry_price, verbose)


# ─────────────────────────────────────────────────────────────
# 投资组合风险管理
# ─────────────────────────────────────────────────────────────

class PortfolioRiskManager:
    """
    投资组合层面风险控制

    功能：
    - 单股最大仓位上限
    - 组合 VaR（历史模拟法）
    - 相关性分析（避免持仓高度相关的股票）
    - 整体风险敞口报告
    """

    def __init__(
        self,
        total_capital: float = 100_000.0,
        max_single_pos_pct: float = 20.0,   # 单只股票最大仓位 %
        max_sector_pct: float = 40.0,        # 单一板块最大仓位 %
        max_corr_threshold: float = 0.75,    # 相关性警戒线（>0.75 提示）
        var_confidence: float = 0.95,        # VaR 置信度
        var_horizon_days: int = 1,           # VaR 时间范围（天）
    ):
        self.capital         = total_capital
        self.max_single_pct  = max_single_pos_pct
        self.max_sector_pct  = max_sector_pct
        self.corr_threshold  = max_corr_threshold
        self.var_confidence  = var_confidence
        self.var_horizon     = var_horizon_days

    def check_position_size(
        self,
        symbol: str,
        proposed_value: float,
        portfolio: Optional[dict] = None,
    ) -> dict:
        """
        检查新头寸是否符合风险限制

        参数：
            symbol          股票代码
            proposed_value  拟投入金额
            portfolio       现有投资组合 {symbol: value}

        返回：{passed, warnings, proposed_pct, allowed_value}
        """
        existing_total = sum((portfolio or {}).values())
        new_total      = existing_total + proposed_value
        proposed_pct   = proposed_value / self.capital * 100

        warnings_list = []
        passed = True

        # 单股上限
        if proposed_pct > self.max_single_pct:
            warnings_list.append(
                f"❌ {symbol} 拟投 {proposed_pct:.1f}% > 单股上限 {self.max_single_pct}%"
            )
            passed = False

        # 总仓位
        total_pct = new_total / self.capital * 100
        if total_pct > 100:
            warnings_list.append(f"❌ 总仓位 {total_pct:.1f}% 超出 100%")
            passed = False

        # 允许的最大投入
        allowed_value = self.capital * self.max_single_pct / 100

        return {
            "symbol":         symbol,
            "passed":         passed,
            "proposed_pct":   round(proposed_pct, 2),
            "allowed_pct":    self.max_single_pct,
            "allowed_value":  round(allowed_value, 2),
            "total_pct_after": round(total_pct, 2),
            "warnings":       warnings_list,
        }

    def compute_portfolio_var(
        self,
        portfolio: dict,   # {symbol: position_value}
        period: str = "1y",
        verbose: bool = True,
    ) -> dict:
        """
        用历史模拟法计算投资组合 VaR

        参数：
            portfolio  {股票代码: 持仓市值} 字典
            period     历史数据长度

        返回：VaR 分析结果
        """
        if not portfolio:
            return {"error": "组合为空"}

        symbols = list(portfolio.keys())
        values  = list(portfolio.values())
        total   = sum(values)
        weights = [v / total for v in values]

        # 获取各股票收益率
        returns_data = {}
        failed = []
        for sym in symbols:
            try:
                df = _fetch_ohlcv(sym, period=period)
                returns_data[sym] = df["Close"].pct_change().dropna()
            except Exception as e:
                failed.append(sym)

        if not returns_data:
            return {"error": "无法获取任何股票数据"}

        # 对齐时间序列
        ret_df = pd.DataFrame(returns_data).dropna()
        w = np.array([weights[symbols.index(s)] for s in ret_df.columns])
        w = w / w.sum()  # 重新归一化

        # 组合日收益率
        port_daily_ret = ret_df.values @ w

        # VaR（历史模拟法）
        var_pct = np.percentile(port_daily_ret, (1 - self.var_confidence) * 100)
        var_dollar = abs(var_pct) * total

        # CVaR（条件 VaR / 期望亏损）
        cvar_pct    = np.mean(port_daily_ret[port_daily_ret <= var_pct])
        cvar_dollar = abs(cvar_pct) * total

        # 扩展到 N 天
        var_n_pct    = var_pct * np.sqrt(self.var_horizon)
        var_n_dollar = abs(var_n_pct) * total

        # 波动率
        port_vol_annual = port_daily_ret.std() * np.sqrt(252) * 100

        # 年化收益
        port_ret_annual = port_daily_ret.mean() * 252 * 100

        # 相关性矩阵
        corr_matrix = ret_df.corr()

        # 高相关对（警告）
        high_corr_pairs = []
        cols = list(ret_df.columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                c = corr_matrix.iloc[i, j]
                if abs(c) > self.corr_threshold:
                    high_corr_pairs.append({
                        "pair": f"{cols[i]} / {cols[j]}",
                        "correlation": round(c, 3),
                        "warning": "高度正相关，分散效果弱" if c > 0 else "高度负相关"
                    })

        result = {
            "portfolio":        portfolio,
            "total_value":      round(total, 2),
            "num_positions":    len(symbols),
            "failed_symbols":   failed,
            "var_confidence":   self.var_confidence,
            "var_horizon_days": self.var_horizon,
            "var_1d_pct":       round(var_pct * 100, 3),
            "var_1d_dollar":    round(var_dollar, 2),
            "var_nd_pct":       round(var_n_pct * 100, 3),
            "var_nd_dollar":    round(var_n_dollar, 2),
            "cvar_1d_pct":      round(cvar_pct * 100, 3),
            "cvar_1d_dollar":   round(cvar_dollar, 2),
            "annual_vol_pct":   round(port_vol_annual, 2),
            "annual_ret_pct":   round(port_ret_annual, 2),
            "sharpe_ratio":     round(port_ret_annual / port_vol_annual, 3) if port_vol_annual > 0 else 0,
            "high_corr_pairs":  high_corr_pairs,
            "correlation_matrix": corr_matrix,
        }

        if verbose:
            self._print_var_report(result)
        return result

    def _print_var_report(self, r: dict):
        w = 60
        conf_pct = r["var_confidence"] * 100
        print(f"\n{'='*w}")
        print(f"  📊  投资组合风险报告（VaR 分析）")
        print(f"{'='*w}")
        print(f"  总持仓市值：${r['total_value']:,.2f}  持仓只数：{r['num_positions']}")
        print(f"  {'─'*56}")
        print(f"  📉 VaR（历史模拟法，{conf_pct:.0f}%置信度）")
        print(f"  1日 VaR：  {r['var_1d_pct']:+.2f}%  即 -${r['var_1d_dollar']:,.2f}")
        if r["var_horizon_days"] > 1:
            print(f"  {r['var_horizon_days']}日 VaR：  {r['var_nd_pct']:+.2f}%  即 -${r['var_nd_dollar']:,.2f}")
        print(f"  CVaR（尾部均损）：{r['cvar_1d_pct']:+.2f}%  即 -${r['cvar_1d_dollar']:,.2f}")
        print(f"  {'─'*56}")
        print(f"  📈 组合绩效估算（历史数据）")
        print(f"  年化波动率：{r['annual_vol_pct']:.2f}%")
        print(f"  年化收益（历史均值）：{r['annual_ret_pct']:+.2f}%")
        print(f"  夏普比率：{r['sharpe_ratio']:.3f}")

        if r.get("high_corr_pairs"):
            print(f"  {'─'*56}")
            print(f"  ⚠️  高相关性预警（>{self.corr_threshold}）")
            for p in r["high_corr_pairs"]:
                print(f"  {p['pair']}  ρ={p['correlation']}  → {p['warning']}")

        if r.get("failed_symbols"):
            print(f"  ❌ 以下代码数据获取失败：{r['failed_symbols']}")

        print(f"{'='*w}")
        print(f"  ⚠️  VaR 为统计估计，极端市场环境下可能严重低估实际风险")
        print(f"{'='*w}\n")

    def full_risk_report(
        self,
        portfolio: dict,
        kelly_trades: Optional[list] = None,
        period: str = "1y",
    ):
        """
        完整风险报告：仓位检查 + VaR + Kelly

        参数：
            portfolio    {股票代码: 持仓市值}
            kelly_trades 历史交易记录（用于 Kelly 计算，可选）
            period       历史数据长度
        """
        total = sum(portfolio.values())

        print(f"\n{'='*60}")
        print(f"  🏦  全面风险报告")
        print(f"{'='*60}")
        print(f"  总资金：${self.capital:,.0f}  已投入：${total:,.0f}（{total/self.capital*100:.1f}%）")
        print(f"  {'─'*56}")
        print(f"  📋 持仓明细")
        for sym, val in sorted(portfolio.items(), key=lambda x: -x[1]):
            pct = val / self.capital * 100
            bar = "█" * int(pct / 2) + "░" * max(0, 10 - int(pct / 2))
            limit_flag = " ⚠️ 超限" if pct > self.max_single_pct else ""
            print(f"    {sym:10s}  ${val:>10,.2f}  {pct:5.1f}%  [{bar}]{limit_flag}")

        # 仓位检查
        print(f"\n  {'─'*56}")
        print(f"  🔍 仓位合规检查（单股上限 {self.max_single_pct}%）")
        all_ok = True
        for sym, val in portfolio.items():
            chk = self.check_position_size(sym, val, {k:v for k,v in portfolio.items() if k!=sym})
            if not chk["passed"]:
                all_ok = False
                for w_msg in chk["warnings"]:
                    print(f"  {w_msg}")
        if all_ok:
            print(f"  ✅ 所有持仓符合单股仓位限制")

        # VaR
        self.compute_portfolio_var(portfolio, period=period, verbose=True)

        # Kelly（如有历史交易）
        if kelly_trades:
            print(f"\n  {'─'*56}")
            print(f"  🎯  Kelly 头寸建议（基于历史回测数据）")
            kc = KellyCriterion(fraction=0.5)
            kc.calculate_from_backtest(
                kelly_trades,
                capital=self.capital,
                entry_price=100,  # 用 100 计算百分比
                verbose=True
            )


# ─────────────────────────────────────────────────────────────
# 快捷函数：一键计算止损、头寸
# ─────────────────────────────────────────────────────────────

def quick_stop_size(
    symbol: str,
    entry_price: Optional[float] = None,
    capital: float = 100_000,
    risk_pct: float = 1.0,          # 每笔交易最大亏损占总资金 %
    atr_multiplier: float = 2.0,
    kelly_win_rate: Optional[float] = None,   # 若提供则一并计算 Kelly
    kelly_avg_win:  Optional[float] = None,
    kelly_avg_loss: Optional[float] = None,
) -> dict:
    """
    快捷计算：给定股票代码，一键输出止损价 + 建议买入股数

    参数：
        symbol         股票代码
        entry_price    入场价（None 则自动获取最新收盘价）
        capital        总资金
        risk_pct       单笔最大风险（占总资金 %，如 1.0 = 1%）
        atr_multiplier 止损 ATR 倍数
        kelly_*        若提供历史统计，则额外计算 Kelly 建议

    返回：综合建议字典
    """
    print(f"\n⏳ 获取 {symbol} 最新数据...")
    df = _fetch_ohlcv(symbol, period="3mo")
    atr_val = calc_atr(df["High"], df["Low"], df["Close"], 14).iloc[-1]

    if entry_price is None:
        entry_price = float(df["Close"].iloc[-1])

    stop_price = entry_price - atr_multiplier * atr_val
    stop_pct   = (entry_price - stop_price) / entry_price * 100
    risk_dollar = capital * risk_pct / 100
    share_count = int(risk_dollar / (entry_price - stop_price)) if entry_price > stop_price else 0
    position_value = share_count * entry_price
    pos_pct     = position_value / capital * 100

    result = {
        "symbol":          symbol,
        "entry_price":     round(entry_price, 4),
        "atr14":           round(float(atr_val), 4),
        "stop_price":      round(stop_price, 4),
        "stop_pct":        round(stop_pct, 2),
        "atr_multiplier":  atr_multiplier,
        "risk_dollar":     round(risk_dollar, 2),
        "risk_pct":        risk_pct,
        "share_count":     share_count,
        "position_value":  round(position_value, 2),
        "position_pct":    round(pos_pct, 2),
    }

    w = 56
    print(f"\n{'='*w}")
    print(f"  🔢  {symbol} 一键止损 + 头寸计算")
    print(f"{'='*w}")
    print(f"  入场价：{entry_price:.2f}")
    print(f"  ATR(14)：{float(atr_val):.4f}")
    print(f"  止损价（{atr_multiplier}x ATR）：{stop_price:.4f}  ({stop_pct:.2f}% 距离)")
    print(f"  {'─'*50}")
    print(f"  总资金：${capital:,.0f}  单笔风险上限：{risk_pct}% = ${risk_dollar:,.2f}")
    print(f"  建议买入股数：{share_count:,} 股")
    print(f"  对应持仓市值：${position_value:,.2f}（占总资金 {pos_pct:.1f}%）")
    print(f"  最大亏损（若触及止损）：-${risk_dollar:,.2f}（-{risk_pct}%）")

    if kelly_win_rate and kelly_avg_win and kelly_avg_loss:
        print(f"  {'─'*50}")
        kc = KellyCriterion(fraction=0.5)
        kr = kc.calculate(
            kelly_win_rate, kelly_avg_win, kelly_avg_loss,
            capital, entry_price, verbose=False
        )
        print(f"  Kelly 建议仓位：{kr['adjusted_pct']:.2f}%（半Kelly）= ${kr['dollar_size']:,.2f}")
        result["kelly"] = kr

    print(f"{'='*w}\n")
    return result


# ─────────────────────────────────────────────────────────────
# 主程序示例
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    symbol = args[0] if args else "NVDA"

    print(f"\n{'='*60}")
    print(f"  风险管理模块演示  ──  {symbol}")
    print(f"{'='*60}")

    # ── 示例1：一键止损 + 头寸计算 ──
    print("\n【示例1】一键止损 + 头寸计算")
    quick_stop_size(
        symbol          = symbol,
        capital         = 100_000,
        risk_pct        = 1.0,           # 每笔不超过总资金 1%
        atr_multiplier  = 2.0,
        kelly_win_rate  = 0.55,          # 假设历史胜率 55%
        kelly_avg_win   = 6.5,           # 平均盈利 6.5%
        kelly_avg_loss  = 3.2,           # 平均亏损 3.2%
    )

    # ── 示例2：ATR 追踪止损模拟 ──
    print("\n【示例2】ATR 追踪止损模拟（近6个月）")
    atr_stop = ATRStopLoss(atr_multiplier=2.0, trailing=True, trailing_multiplier=1.5)

    # 取6个月前的日期作为假设入场日
    entry_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    atr_stop.print_stop_report(symbol, entry_date)

    # ── 示例3：Kelly 头寸计算（直接输入统计数据）──
    print("\n【示例3】Kelly Criterion 头寸定规")
    kc = KellyCriterion(fraction=0.5)
    kc.calculate(
        win_rate     = 0.55,    # 55% 胜率
        avg_win_pct  = 8.0,     # 平均盈利 8%
        avg_loss_pct = 4.0,     # 平均亏损 4%
        capital      = 100_000,
        entry_price  = 200.0,   # 假设股价 $200
    )

    # ── 示例4：组合 VaR 分析 ──
    print("\n【示例4】投资组合 VaR 分析（假设组合）")
    portfolio_example = {
        symbol:  30_000,   # 30%
        "AAPL":  25_000,   # 25%
        "MSFT":  20_000,   # 20%
        "GOOGL": 15_000,   # 15%
        "SPY":   10_000,   # 10%
    }

    prm = PortfolioRiskManager(
        total_capital       = 100_000,
        max_single_pos_pct  = 30,
        var_confidence      = 0.95,
        var_horizon_days    = 1,
    )
    prm.full_risk_report(portfolio_example, period="1y")
