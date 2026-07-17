"""
L1 交易闸门（Trade Gate）
========================
任何开仓/加仓动作必须先过这里。

核心入口：
    check_entry(account, symbol, entry_price, atr, signal_score, today_change_pct)

返回结构：
    {
      "allowed":       True / False,
      "reason":        "通过" 或 "拦截原因",
      "blockers":      [硬规则阻止, ...],       # allowed 必为 False 时非空
      "warnings":      [软警告, ...],
      "size_tier":     "full" / "half" / "quarter" / "none",
      "suggested_shares": int,
      "suggested_value":  float,   # 账户本币
      "risk_dollar":   float,      # 单笔最大亏损金额
      "stop_price":    float,
      "take_profit_1R": float,
      "take_profit_2R": float,
      "oco_order":     {"entry": ..., "stop": ..., "target_2R": ...},
    }

设计理念：
- 能不开仓就不开仓——所有规则默认"疑罪从有"
- 硬规则（blockers）一条命中就拒绝
- 软警告（warnings）照单全收，用户看清楚后决定
"""

import math
from datetime import date
from typing import Optional

from config import (
    ACCOUNTS, BEHAVIORAL_RULES, STOP_LOSS,
    get_account, get_effective_capital,
)
import trade_state as ts


# ─────────────────────────────────────────────────────────────
# 主入口：开仓检查
# ─────────────────────────────────────────────────────────────

def check_entry(
    account: str,
    symbol: str,
    entry_price: float,
    atr: float,
    signal_score: float,
    today_change_pct: float = 0.0,
    *,
    strategy_label: str = "swing",
) -> dict:
    """
    入场闸门——是否允许、允许多少。

    参数：
        account            "A_SHARES" / "US_STOCKS"
        symbol             标的代码
        entry_price        拟入场价
        atr                当前 ATR14（来自 signal_generator.indicators["atr14"]）
        signal_score       L2 综合评分 0-100
        today_change_pct   今日涨幅百分比（来自 indicators["price_chg%"]）
        strategy_label     策略标签，用于复盘归类
    """
    acc = get_account(account)
    blockers: list[str] = []
    warnings: list[str] = []

    # ─── 硬规则 1：账户锁仓（熔断/连亏冷却） ───
    locked, lock_reason = ts.is_locked(account)
    if locked:
        blockers.append(f"🔒 {lock_reason}")

    # ─── 硬规则 2：账户回撤熔断 ───
    drawdown = ts.get_account_drawdown(account)
    if drawdown is not None and drawdown <= -acc["drawdown_circuit_breaker_pct"]:
        blockers.append(
            f"🚨 熔断：账户回撤 {drawdown:.2f}% 已触及 "
            f"-{acc['drawdown_circuit_breaker_pct']}%，必须减仓并停手 "
            f"{acc['circuit_breaker_cooldown_days']} 天"
        )
        # 自动锁仓
        ts.lock_account(account,
                        days=acc["circuit_breaker_cooldown_days"],
                        reason=f"账户回撤熔断 {drawdown:.2f}%")

    # ─── 硬规则 3：今日交易次数上限 ───
    today_count = ts.get_daily_trade_count(account)
    if today_count >= acc["max_daily_trades"]:
        blockers.append(
            f"🛑 今日已交易 {today_count} 笔（上限 {acc['max_daily_trades']}），"
            f"频繁操作拦截。明日再说。"
        )

    # ─── 硬规则 4：连亏锁仓 ───
    if ts.consecutive_losses(account, n=BEHAVIORAL_RULES["consecutive_loss_lockout_count"]):
        blockers.append(
            f"🛑 连亏 {BEHAVIORAL_RULES['consecutive_loss_lockout_count']} 笔，"
            f"当日锁仓。写完复盘再来。"
        )
        ts.lock_account(account,
                        days=BEHAVIORAL_RULES["lockout_days"],
                        reason=f"连亏 {BEHAVIORAL_RULES['consecutive_loss_lockout_count']} 笔")

    # ─── 硬规则 5：追涨拦截 ───
    if today_change_pct > acc["chase_rally_threshold_pct"]:
        blockers.append(
            f"🚫 追涨拦截：{symbol} 今日涨 {today_change_pct:.2f}% "
            f"> 阈值 {acc['chase_rally_threshold_pct']}%，当日禁追。"
            f"等 1 根回调 K 线再看。"
        )

    # ─── 硬规则 6：信号分数门槛 ───
    if signal_score < acc["min_signal_score"]:
        blockers.append(
            f"📉 信号分 {signal_score:.0f} < 门槛 {acc['min_signal_score']}，不入场。"
        )

    # ─── 硬规则 7：单股上限 / 重复持仓 ───
    positions = ts.get_positions(account)
    existing = next((p for p in positions if p["symbol"] == symbol.upper()), None)
    if existing:
        warnings.append(
            f"⚠️ 已持有 {symbol}（{existing['shares']} 股 @ {existing['entry_price']}），"
            f"这将是加仓。请先确认加仓理由："
        )

    if len(positions) >= acc["max_positions"] and not existing:
        blockers.append(
            f"🛑 持仓数 {len(positions)} 已达上限 {acc['max_positions']}，"
            f"先关一笔再开新仓。"
        )

    # ─── 仓位计算（即使有 blocker 也算出来给用户看） ───
    size = _calc_position_size(
        account_key=account,
        entry_price=entry_price,
        atr=atr,
        signal_score=signal_score,
    )

    # 单股上限检查（软->硬）
    if size["suggested_value"] > 0:
        pos_pct = size["suggested_value"] / get_effective_capital(account) * 100
        if pos_pct > acc["max_single_pos_pct"]:
            warnings.append(
                f"⚠️ 建议仓位 {pos_pct:.1f}% > 单股上限 {acc['max_single_pos_pct']}%，"
                f"已自动压缩到上限"
            )
            # 压缩到上限（并重算风险金）
            capped_value = get_effective_capital(account) * acc["max_single_pos_pct"] / 100
            size["suggested_shares"] = _floor_shares(capped_value / entry_price, account)
            size["suggested_value"]  = round(size["suggested_shares"] * entry_price, 2)
            if size["stop_price"]:
                risk_per_share = entry_price - size["stop_price"]
                size["risk_dollar"] = round(size["suggested_shares"] * risk_per_share, 2)
                size["risk_pct"] = size["risk_dollar"] / get_effective_capital(account) * 100

    # ─── 硬规则 8：可交易股数 = 0（A股 100 股取整归零 / 价格过高） ───
    if size["suggested_shares"] == 0 and size["size_tier"] != "none":
        currency = acc.get("currency", "USD")
        # A股 1 手价值
        one_lot_value = 100 * entry_price if currency == "CNY" else entry_price
        blockers.append(
            f"🛑 无法下单：以当前风险预算 + 信号档位，建议股数归零。"
            f"（{'A股 1 手需 ' + f'{one_lot_value:,.0f} CNY' if currency == 'CNY' else '单股 ' + f'${entry_price:.2f}'}，"
            f"可能是标的单价过高或单笔风险太紧。）换标的或等更好信号。"
        )

    # 美股单笔心理红线检查（最终值）
    if acc.get("risk_per_trade_hard_cap") and size["risk_dollar"] > acc["risk_per_trade_hard_cap"]:
        blockers.append(
            f"🚨 最终风险金 {size['risk_dollar']:.2f} "
            f"> 心理红线 {acc['risk_per_trade_hard_cap']}，拒绝"
        )

    # ─── OCO 挂单（强制要求） ───
    oco = None
    if BEHAVIORAL_RULES["require_oco_on_entry"] and size["stop_price"]:
        oco = {
            "entry":     round(entry_price, 4),
            "stop":      round(size["stop_price"], 4),
            "target_1R": round(size["take_profit_1R"], 4),
            "target_2R": round(size["take_profit_2R"], 4),
            "note":      "入场后立即挂 OCO（止损 + 目标），未挂不允许离开屏幕",
        }

    allowed = len(blockers) == 0 and size["suggested_shares"] > 0

    # 信号分档的文字说明
    size_note = ""
    if signal_score >= acc["full_size_signal_score"]:
        size_note = f"信号 {signal_score:.0f} ≥ {acc['full_size_signal_score']}：满档"
    elif signal_score >= acc["half_size_signal_score"]:
        size_note = f"信号 {signal_score:.0f} in [{acc['half_size_signal_score']}, {acc['full_size_signal_score']})：3/4 档"
    elif signal_score >= acc["min_signal_score"]:
        size_note = f"信号 {signal_score:.0f} in [{acc['min_signal_score']}, {acc['half_size_signal_score']})：半档"
    if size_note:
        warnings.append(f"🎯 {size_note}")

    return {
        "allowed":          allowed,
        "reason":           "✅ 通过" if allowed else "❌ 拦截：" + "；".join(blockers),
        "account":          account,
        "symbol":           symbol.upper(),
        "entry_price":      round(entry_price, 4),
        "signal_score":     round(signal_score, 1),
        "blockers":         blockers,
        "warnings":         warnings,
        "size_tier":        size["size_tier"],
        "suggested_shares": size["suggested_shares"],
        "suggested_value":  size["suggested_value"],
        "position_pct":     round(size["suggested_value"] / get_effective_capital(account) * 100, 2)
                            if size["suggested_value"] else 0,
        "risk_dollar":      size["risk_dollar"],
        "risk_pct":         round(size["risk_pct"], 2),
        "stop_price":       size["stop_price"],
        "take_profit_1R":   size["take_profit_1R"],
        "take_profit_2R":   size["take_profit_2R"],
        "atr":              round(atr, 4),
        "oco_order":        oco,
        "strategy":         strategy_label,
    }


# ─────────────────────────────────────────────────────────────
# 仓位公式
# ─────────────────────────────────────────────────────────────

def _calc_position_size(
    account_key: str,
    entry_price: float,
    atr: float,
    signal_score: float,
) -> dict:
    """
    基于 ATR 止损 + 单笔风险% + 信号分档三个约束取最小值。
    """
    acc = get_account(account_key)
    capital = get_effective_capital(account_key)

    # 1. ATR 止损
    stop_price = entry_price - STOP_LOSS["atr_multiplier_initial"] * atr
    if stop_price <= 0 or entry_price <= stop_price:
        return _empty_size()
    risk_per_share = entry_price - stop_price

    # 2. 单笔风险金（账面规则）
    risk_budget = capital * acc["risk_per_trade_pct"] / 100

    # 3. 心理红线硬上限
    if acc.get("risk_per_trade_hard_cap"):
        risk_budget = min(risk_budget, acc["risk_per_trade_hard_cap"])

    # 4. 原始股数（满档）
    full_shares = risk_budget / risk_per_share

    # 5. 信号分档缩放
    if signal_score >= acc["full_size_signal_score"]:
        size_tier, mult = "full", 1.0
    elif signal_score >= acc["half_size_signal_score"]:
        size_tier, mult = "three_quarter", 0.75
    elif signal_score >= acc["min_signal_score"]:
        size_tier, mult = "half", 0.5
    else:
        size_tier, mult = "none", 0.0

    scaled_shares = _floor_shares(full_shares * mult, account_key)
    position_value = scaled_shares * entry_price
    actual_risk = scaled_shares * risk_per_share

    return {
        "size_tier":       size_tier,
        "suggested_shares": int(scaled_shares),
        "suggested_value":  round(position_value, 2),
        "risk_dollar":      round(actual_risk, 2),
        "risk_pct":         actual_risk / capital * 100 if capital else 0,
        "stop_price":       round(stop_price, 4),
        "take_profit_1R":   round(entry_price + risk_per_share, 4),      # 1R = 风险等额
        "take_profit_2R":   round(entry_price + 2 * risk_per_share, 4),
    }


def _empty_size() -> dict:
    return {
        "size_tier": "none", "suggested_shares": 0, "suggested_value": 0,
        "risk_dollar": 0, "risk_pct": 0,
        "stop_price": None, "take_profit_1R": None, "take_profit_2R": None,
    }


def _floor_shares(raw: float, account_key: str) -> int:
    """A股必须 100 股一手；美股按 1 股。"""
    acc = get_account(account_key)
    if acc.get("currency") == "CNY":
        return int(math.floor(raw / 100.0) * 100)
    return int(math.floor(raw))


# ─────────────────────────────────────────────────────────────
# 出场闸门（减仓/止盈）
# ─────────────────────────────────────────────────────────────

def check_exit_partial(account: str, symbol: str, current_price: float) -> dict:
    """
    出场前先过这里。主要防止"过早止盈"——盈利未到 1R 不允许减仓。
    """
    positions = ts.get_positions(account)
    pos = next((p for p in positions if p["symbol"] == symbol.upper()), None)
    if pos is None:
        return {"allowed": True, "note": "无持仓，不拦截（可能是手动录入前的平仓）"}

    entry = pos["entry_price"]
    stop  = pos["initial_stop"]
    r_unit = entry - stop
    if r_unit <= 0:
        return {"allowed": True, "note": "入场止损异常，放行"}

    r_multiple = (current_price - entry) / r_unit

    blockers, warnings = [], []
    if BEHAVIORAL_RULES["lock_profit_until_1R"] and r_multiple < 1.0:
        blockers.append(
            f"🔒 过早止盈拦截：当前盈利 {r_multiple:.2f}R < 1R，禁止减仓。"
            f"要么全平，要么继续持有。"
        )

    if r_multiple >= 2.0 and BEHAVIORAL_RULES["trailing_stop_above_2R"]:
        warnings.append(
            f"📈 已达 {r_multiple:.2f}R，应切换跟踪止损（1.5×ATR 回撤保护）"
        )

    return {
        "allowed":     len(blockers) == 0,
        "r_multiple":  round(r_multiple, 3),
        "entry_price": entry,
        "current_price": current_price,
        "initial_stop": stop,
        "blockers":    blockers,
        "warnings":    warnings,
    }


# ─────────────────────────────────────────────────────────────
# 漂亮打印
# ─────────────────────────────────────────────────────────────

def print_entry_decision(d: dict) -> None:
    w = 62
    print(f"\n{'═' * w}")
    print(f"  🚦  {d['symbol']}  @ {d['account']}  入场决策")
    print(f"{'═' * w}")
    print(f"  {d['reason']}")
    print(f"  {'─' * (w - 4)}")
    print(f"  入场价：{d['entry_price']}  ATR(14)：{d['atr']}")
    print(f"  信号分：{d['signal_score']}/100  档位：{d['size_tier']}")

    if d["suggested_shares"]:
        cur = "CNY" if d["account"] == "A_SHARES" else "USD"
        print(f"  建议股数：{d['suggested_shares']:,} 股")
        print(f"  持仓市值：{d['suggested_value']:,.2f} {cur}（占账户 {d['position_pct']}%）")
        print(f"  风险金额：{d['risk_dollar']:,.2f} {cur}（占账户 {d['risk_pct']:.2f}%）")
        print(f"  止损价：  {d['stop_price']}")
        print(f"  止盈 1R： {d['take_profit_1R']}")
        print(f"  止盈 2R： {d['take_profit_2R']}")

    if d["oco_order"]:
        oco = d["oco_order"]
        print(f"\n  💡 OCO 挂单（入场立即挂）：")
        print(f"     入场 Entry  = {oco['entry']}")
        print(f"     止损 Stop   = {oco['stop']}")
        print(f"     目标 Target = {oco['target_2R']}  (2R)")

    if d["warnings"]:
        print(f"\n  ⚠️ 警告：")
        for w_msg in d["warnings"]:
            print(f"     {w_msg}")

    if d["blockers"]:
        print(f"\n  🛑 拦截理由：")
        for b in d["blockers"]:
            print(f"     {b}")

    print(f"{'═' * w}\n")


# ─────────────────────────────────────────────────────────────
# 与 signal_generator 的集成入口
# ─────────────────────────────────────────────────────────────

def gate_with_signal(account: str, symbol: str, *, verbose: bool = True) -> dict:
    """
    一键调用：跑信号 + 过闸门 + 打印决策。
    """
    from signal_generator import generate_signal
    sig = generate_signal(symbol, verbose=False)
    if sig.get("error"):
        return {"allowed": False, "reason": f"信号失败: {sig['error']}"}

    ind = sig.get("indicators", {})
    decision = check_entry(
        account=account,
        symbol=symbol,
        entry_price=float(ind.get("price", 0)),
        atr=float(ind.get("atr14", 0)),
        signal_score=float(sig["score"]),
        today_change_pct=float(ind.get("price_chg%", 0)),
    )
    if verbose:
        print_entry_decision(decision)
    return decision


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) < 2:
        print("用法: python trade_gate.py <ACCOUNT> <SYMBOL>")
        print("示例: python trade_gate.py US_STOCKS NVDA")
        print("      python trade_gate.py A_SHARES 600519")
        sys.exit(0)

    gate_with_signal(args[0], args[1], verbose=True)
