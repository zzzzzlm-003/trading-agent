"""
赌狗预算账本
============
物理隔离投机欲，保护主账户纪律。

规则（见 config.GAMBLING）：
- 月初自动重置到 $100
- 当月亏完即停，不得从主账户补仓
- 盈利不累积进下月（月末结转主账户）
- 主账户纪律不受此账户盈亏影响

本模块独立于 trade_gate.py：赌狗账户不走信号打分、不走风控闸门，
只有一条硬规则——本月预算还没花完就能买，花完就不能买。

存储：./gambling_ledger.json
"""

import json
from datetime import date, datetime
from pathlib import Path

from config import ACCOUNTS

LEDGER_FILE = Path(__file__).parent / "gambling_ledger.json"


def _current_month() -> str:
    return date.today().strftime("%Y-%m")


def _default_ledger() -> dict:
    return {
        "months": {},        # "2026-04" -> {"budget": 100, "used": 0, "trades": [...]}
        "last_updated": None,
    }


def _load() -> dict:
    if not LEDGER_FILE.exists():
        return _default_ledger()
    try:
        with open(LEDGER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return _default_ledger()


def _save(ledger: dict) -> None:
    ledger["last_updated"] = datetime.now().isoformat(timespec="seconds")
    with open(LEDGER_FILE, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False, default=str)


def _ensure_month(ledger: dict, month: str) -> dict:
    """保证当月条目存在（月初自动开新账）。"""
    if month not in ledger["months"]:
        budget = ACCOUNTS["GAMBLING"]["monthly_budget"]
        ledger["months"][month] = {
            "budget":      budget,
            "used":        0.0,
            "realized_pnl": 0.0,
            "trades":      [],
            "status":      "active",
            "opened_at":   date.today().isoformat(),
        }
    return ledger["months"][month]


# ─────────────────────────────────────────────────────────────
# 开仓检查
# ─────────────────────────────────────────────────────────────

def check_gambling_trade(cost_usd: float, description: str = "") -> dict:
    """
    赌狗账户开仓前检查。只有一条规则：本月预算够不够。
    """
    ledger = _load()
    month = _current_month()
    m = _ensure_month(ledger, month)
    remaining = m["budget"] - m["used"]

    blockers = []
    if cost_usd > remaining:
        blockers.append(
            f"🛑 {month} 预算剩余 ${remaining:.2f}，本次需 ${cost_usd:.2f}，不够。"
            f"不得从主账户补仓。等下月 1 号。"
        )

    if cost_usd > m["budget"]:
        blockers.append(
            f"🛑 单笔 ${cost_usd:.2f} > 月预算 ${m['budget']}，违反隔离原则"
        )

    allowed = len(blockers) == 0
    _save(ledger)

    return {
        "allowed":      allowed,
        "month":        month,
        "budget":       m["budget"],
        "used_before":  round(m["used"], 2),
        "remaining":    round(remaining, 2),
        "cost":         round(cost_usd, 2),
        "blockers":     blockers,
        "reminder":     (
            "⚠️ 赌狗账户只消化你的投机欲，不影响主账户判断。"
            "主账户的信号、纪律、仓位——照旧执行。"
        ),
    }


def log_gambling_entry(cost_usd: float, instrument: str, note: str = "") -> dict:
    """记录赌狗开仓（扣减预算）。"""
    ledger = _load()
    month = _current_month()
    m = _ensure_month(ledger, month)

    trade = {
        "trade_id":    f"g_{len(m['trades'])+1}",
        "opened_at":   date.today().isoformat(),
        "instrument":  instrument,
        "cost":        round(cost_usd, 2),
        "note":        note,
        "closed":      False,
        "pnl":         None,
    }
    m["trades"].append(trade)
    m["used"] = round(m["used"] + cost_usd, 2)

    if m["used"] >= m["budget"] - 0.01:
        m["status"] = "exhausted"

    _save(ledger)
    return trade


def log_gambling_exit(trade_id: str, pnl_usd: float) -> dict:
    """记录赌狗关仓（盈利不回流预算，归入 realized_pnl 供月末结转）。"""
    ledger = _load()
    month = _current_month()
    m = _ensure_month(ledger, month)

    trade = next((t for t in m["trades"] if t["trade_id"] == trade_id), None)
    if trade is None:
        return {"error": f"未找到 trade_id={trade_id}"}
    trade["closed"] = True
    trade["closed_at"] = date.today().isoformat()
    trade["pnl"] = round(pnl_usd, 2)
    m["realized_pnl"] = round(m["realized_pnl"] + pnl_usd, 2)
    _save(ledger)
    return trade


# ─────────────────────────────────────────────────────────────
# 月度状态与结转
# ─────────────────────────────────────────────────────────────

def get_month_status(month: str | None = None) -> dict:
    """查某个月（默认当月）的赌狗账户状态。"""
    ledger = _load()
    month = month or _current_month()
    m = _ensure_month(ledger, month)
    _save(ledger)
    return {
        "month":         month,
        "budget":        m["budget"],
        "used":          round(m["used"], 2),
        "remaining":     round(m["budget"] - m["used"], 2),
        "status":        m["status"],
        "realized_pnl":  round(m["realized_pnl"], 2),
        "num_trades":    len(m["trades"]),
        "num_closed":    sum(1 for t in m["trades"] if t["closed"]),
    }


def close_month_and_transfer(month: str | None = None) -> dict:
    """
    月末结转：把当月 realized_pnl 归入主账户（逻辑归零），
    下月自动开新账户预算 $100。
    """
    ledger = _load()
    month = month or _current_month()
    if month not in ledger["months"]:
        return {"error": f"{month} 无记录"}
    m = ledger["months"][month]
    transferred = m["realized_pnl"]
    m["status"] = "closed_and_transferred"
    m["realized_pnl"] = 0.0
    _save(ledger)
    return {
        "month":                  month,
        "transferred_to_main":    round(transferred, 2),
        "note":                   "盈利归主账户，不累积进下月预算",
    }


def print_status(month: str | None = None) -> None:
    s = get_month_status(month)
    rules = ACCOUNTS["GAMBLING"]
    w = 52
    print(f"\n{'─' * w}")
    print(f"  🎰 赌狗账户  {s['month']}")
    print(f"{'─' * w}")
    print(f"  月预算：        ${s['budget']:.2f}")
    print(f"  已使用：        ${s['used']:.2f}")
    print(f"  剩余可用：      ${s['remaining']:.2f}")
    print(f"  状态：          {s['status']}")
    print(f"  已实现盈亏：    ${s['realized_pnl']:+.2f}")
    print(f"  交易数：        {s['num_trades']}（已平 {s['num_closed']}）")
    print(f"{'─' * w}")
    print(f"  规则：{rules['rule'] if 'rule' in rules else '亏完当月停，盈利归主账户'}")
    print(f"{'─' * w}\n")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    if not args or args[0] == "status":
        print_status()

    elif args[0] == "check" and len(args) >= 2:
        cost = float(args[1])
        r = check_gambling_trade(cost)
        print(json.dumps(r, indent=2, ensure_ascii=False))

    elif args[0] == "demo":
        print("演示：$100 预算，买 $5 末日 call")
        r1 = check_gambling_trade(5, "NVDA 0DTE call")
        print(f"  check: {r1['allowed']}  remaining={r1['remaining']}")
        if r1["allowed"]:
            t = log_gambling_entry(5, "NVDA 0DTE call", "demo")
            print(f"  开仓记录：{t['trade_id']}")
        print_status()
