"""
L1 演示脚本
===========
展示闸门在 5 个典型场景下的行为，用假数据跑，不触网不花钱。

运行：python demo_l1.py
"""

from config import ACCOUNTS, get_effective_capital
from trade_gate import check_entry, check_exit_partial, print_entry_decision
import trade_state as ts
import discipline_ledger as gl


def section(title):
    print(f"\n\n{'█' * 64}")
    print(f"  {title}")
    print(f"{'█' * 64}")


def main():
    # ── 准备：清零演示用状态 ──
    import os
    for f in [ts.STATE_FILE, gl.LEDGER_FILE]:
        if os.path.exists(f):
            os.remove(f)

    section("场景 1：正常通过（美股高分信号）")
    d1 = check_entry(
        account="US_STOCKS",
        symbol="NVDA",
        entry_price=180.0,
        atr=4.5,
        signal_score=82,
        today_change_pct=1.2,
    )
    print_entry_decision(d1)

    # 登记持仓，模拟已开仓
    if d1["allowed"]:
        ts.open_position("US_STOCKS", "NVDA",
                         entry_price=d1["entry_price"],
                         shares=d1["suggested_shares"],
                         stop_price=d1["stop_price"],
                         atr=d1["atr"])
        ts.increment_daily_trade_count("US_STOCKS")

    section("场景 2：追涨拦截（同一只涨 7% 还想买）")
    d2 = check_entry(
        account="US_STOCKS", symbol="AAPL",
        entry_price=250.0, atr=3.0,
        signal_score=78,
        today_change_pct=7.5,
    )
    print_entry_decision(d2)

    section("场景 3：信号分不足（刚过 60 门槛→半档）")
    d3 = check_entry(
        account="US_STOCKS", symbol="MSFT",
        entry_price=420.0, atr=6.0,
        signal_score=62,
        today_change_pct=0.5,
    )
    print_entry_decision(d3)

    section("场景 4：频繁操作拦截（把今日次数刷满）")
    # 先把美股账户的今日次数刷到上限
    for _ in range(ACCOUNTS["US_STOCKS"]["max_daily_trades"] - 1):
        ts.increment_daily_trade_count("US_STOCKS")
    d4 = check_entry(
        account="US_STOCKS", symbol="TSLA",
        entry_price=350.0, atr=8.0,
        signal_score=80,
        today_change_pct=1.0,
    )
    print_entry_decision(d4)

    section("场景 5：过早止盈拦截（NVDA 只涨了 0.5R 就想减仓）")
    # NVDA 已在场景 1 开仓
    e1 = check_exit_partial(
        account="US_STOCKS", symbol="NVDA",
        current_price=d1["entry_price"] + 0.5 * (d1["entry_price"] - d1["stop_price"]),
    )
    print(f"\n  当前 R 倍数: {e1['r_multiple']:.2f}R")
    print(f"  允许减仓：{e1['allowed']}")
    for b in e1["blockers"]:
        print(f"    {b}")

    section("场景 6：A股中等信号（600519 茅台，半档）")
    d6 = check_entry(
        account="A_SHARES", symbol="600519",
        entry_price=1560.0, atr=28.0,
        signal_score=70,
        today_change_pct=0.8,
    )
    print_entry_decision(d6)

    section("场景 7：账户回撤熔断（模拟 -10% 回撤）")
    ts.update_account_value("US_STOCKS", 10_000)   # 峰值
    ts.update_account_value("US_STOCKS", 9_000)    # -10%
    d7 = check_entry(
        account="US_STOCKS", symbol="GOOGL",
        entry_price=180.0, atr=3.0,
        signal_score=85,
        today_change_pct=0.5,
    )
    print_entry_decision(d7)

    section("场景 8：投机账户——买 $5 末日期权")
    r = gl.check_discipline_trade(5.0, "NVDA 0DTE call")
    print(f"  允许：{r['allowed']}  剩余预算：${r['remaining']}")
    if r["allowed"]:
        gl.log_discipline_entry(5.0, "NVDA 0DTE call", "演示")
    gl.print_status()

    section("场景 9：投机预算已耗尽想再买 $50")
    # 把剩余预算先花掉
    for _ in range(19):
        check = gl.check_discipline_trade(5.0)
        if check["allowed"]:
            gl.log_discipline_entry(5.0, "demo burn")
    r2 = gl.check_discipline_trade(50.0, "SPY 0DTE")
    print(f"  允许：{r2['allowed']}")
    for b in r2["blockers"]:
        print(f"  {b}")
    gl.print_status()

    print("\n\n✅ 演示完成。trade_state.json 和 discipline_ledger.json 已生成在当前目录。")
    print("   运行  python trade_state.py   查看完整状态")
    print("   运行  python discipline_ledger.py   查看投机账本")


if __name__ == "__main__":
    main()
