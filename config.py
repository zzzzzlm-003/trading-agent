"""
交易系统个性化配置
==================
本文件是整个 L1 风控的真理源。调参只改这里，不动其他模块。

账户结构（2026-04 基准）：
- A_SHARES      A股券商账户  25w RMB
- US_STOCKS     美股账户     ~13k USD 可用（扣$7k学费后）
- US_OPTIONS    美股期权子账户（共享美股资金，单独规则）
- GAMBLING      赌狗预算     $100/月（物理隔离投机欲）
"""

from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 账户参数
# ─────────────────────────────────────────────────────────────

ACCOUNTS = {
    "A_SHARES": {
        "name":                           "A股券商账户",
        "currency":                       "CNY",
        "capital":                        250_000,     # 可用资金（不含基金6w）
        "risk_per_trade_pct":             1.0,         # 单笔最大亏损 1% = 2500元
        "risk_per_trade_hard_cap":        None,        # A股无心理红线
        "max_positions":                  5,           # 同时最多持仓数
        "max_single_pos_pct":             20.0,        # 单只最大 20%
        "max_daily_trades":               3,           # 单日最多交易次数
        "drawdown_circuit_breaker_pct":   8.0,         # 账户从峰值回撤 8% 熔断
        "circuit_breaker_reduce_pct":     50.0,        # 熔断时减仓 50%
        "circuit_breaker_cooldown_days":  3,           # 熔断后停手天数
        "chase_rally_threshold_pct":      4.0,         # 单日涨幅 >4% 当日禁追
        "min_signal_score":               60,          # L2 信号低于 60 不交易
        "half_size_signal_score":         75,          # 60-75 半档仓位
        "full_size_signal_score":         85,          # >85 满档
    },

    "US_STOCKS": {
        "name":                           "美股账户",
        "currency":                       "USD",
        "capital":                        13_000,      # 扣除 $7k 学费后可用
        "reserved":                       7_000,       # 学费储备，不可动
        "risk_per_trade_pct":             1.5,         # 账面硬规则 $200
        "risk_per_trade_hard_cap":        300,         # 心理红线：单笔绝不越 $300
        "max_positions":                  5,
        "max_single_pos_pct":             25.0,
        "max_daily_trades":               3,
        "drawdown_circuit_breaker_pct":   8.0,
        "circuit_breaker_reduce_pct":     50.0,
        "circuit_breaker_cooldown_days":  3,
        "chase_rally_threshold_pct":      5.0,         # 美股单日 >5% 禁追
        "min_signal_score":               60,
        "half_size_signal_score":         75,
        "full_size_signal_score":         85,
    },

    "US_OPTIONS": {
        "name":                           "美股期权（子账户）",
        "currency":                       "USD",
        "capital_source":                 "US_STOCKS", # 共享美股资金池
        "max_total_max_loss_pct":         10.0,        # 所有期权 max_loss 之和 ≤ 美股账户 10%
        "max_single_max_loss_usd":        200,         # 单笔 max_loss ≤ $200
        "forbid_naked_short":             True,        # 禁止裸卖方（理论无限亏损）
        "max_daily_trades":               2,
        "allowed_strategies": [
            "long_call", "long_put",
            "bull_put_spread", "bear_call_spread",
            "iron_condor", "iron_butterfly",
            "covered_call", "cash_secured_put",
        ],
        "forbidden_strategies": [
            "naked_call", "naked_put", "short_strangle_uncovered",
        ],
    },

    "GAMBLING": {
        "name":                           "赌狗预算",
        "currency":                       "USD",
        "monthly_budget":                 100,         # $100/月
        "reset_day":                      1,           # 每月 1 号重置
        "allow_rollover":                 False,       # 盈利不累积进下月
        "allow_topup_from_main":          False,       # 亏完不允许从主账户补仓
        "typical_instrument":             "末日期权 / 小额投机",
    },
}

# ─────────────────────────────────────────────────────────────
# 行为拦截规则（对抗她自认的 4 个犯错点）
# ─────────────────────────────────────────────────────────────

BEHAVIORAL_RULES = {
    "require_oco_on_entry":            True,   # 入场必须同时挂止损
    "lock_profit_until_1R":            True,   # 盈利未到 1R 禁止减仓
    "allow_partial_exit_1R_to_2R":     True,   # 1R-2R 允许分批
    "trailing_stop_above_2R":          True,   # >2R 切跟踪止损
    "consecutive_loss_lockout_count":  2,      # 连亏 2 笔当日锁仓
    "lockout_days":                    1,      # 锁仓天数
    "revenge_trade_window_hours":      2,      # 亏损后 2 小时内的新仓判定为报复性
}

# ─────────────────────────────────────────────────────────────
# 止损参数（沿用 risk_manager.py 的 ATR 止损）
# ─────────────────────────────────────────────────────────────

STOP_LOSS = {
    "atr_multiplier_initial":          2.0,    # 初始止损 2×ATR
    "atr_multiplier_trailing":         1.5,    # 跟踪止损 1.5×ATR（上移到 2R 后启用）
    "max_hold_days":                   30,     # 波段策略默认最长持仓
}

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def get_account(account_key: str) -> dict:
    """获取账户配置，未找到则抛错。"""
    if account_key not in ACCOUNTS:
        raise KeyError(f"未知账户: {account_key}，可用: {list(ACCOUNTS.keys())}")
    return ACCOUNTS[account_key]


def get_effective_capital(account_key: str) -> float:
    """获取账户的'可用于交易'资金（扣除储备金）。"""
    acc = get_account(account_key)
    if "capital_source" in acc:
        return get_effective_capital(acc["capital_source"])
    return acc["capital"]


if __name__ == "__main__":
    print("─" * 50)
    print("  交易系统配置概览")
    print("─" * 50)
    for key, acc in ACCOUNTS.items():
        if "capital_source" in acc:
            cap = get_effective_capital(key)
            print(f"  [{key}] {acc['name']}  共享自 {acc['capital_source']}（{cap} {acc['currency']}）")
        elif "monthly_budget" in acc:
            print(f"  [{key}] {acc['name']}  {acc['monthly_budget']} {acc['currency']}/月")
        else:
            reserve = f" (储备 {acc.get('reserved', 0)})" if acc.get("reserved") else ""
            print(f"  [{key}] {acc['name']}  {acc['capital']} {acc['currency']}{reserve}")
    print("─" * 50)
