"""
交易状态持久化
==============
把跨会话需要记住的东西存到 JSON 文件，让 trade_gate.py 能判断：
- 今日每个账户交易了几笔？（频繁操作拦截）
- 最近几笔交易的盈亏？（连亏锁仓判断）
- 各账户资金峰值？（熔断触发判断）
- 当前持仓？（单股上限、重复开仓判断）
- 锁仓到什么时候？（熔断/连亏冷却期）

状态文件路径：./trade_state.json
所有金额数字无货币单位，按账户本币存。
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

STATE_FILE = Path(__file__).parent / "trade_state.json"


def _default_state() -> dict:
    return {
        "positions": {},         # "US_STOCKS:NVDA" -> {...}
        "daily_trades": {},      # "2026-04-17" -> {"US_STOCKS": 2, "A_SHARES": 0, ...}
        "recent_trades": [],     # list of last 10 closed trades: {account, symbol, pnl_pct, closed_at}
        "account_peaks": {},     # account_key -> peak_value
        "account_current": {},   # account_key -> last_known_value（用户手动同步）
        "locked_until": {},      # account_key -> ISO date string
        "last_updated": None,
    }


# ─────────────────────────────────────────────────────────────
# 读写
# ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    """读取状态，文件不存在则返回默认值。"""
    if not STATE_FILE.exists():
        return _default_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        # 补齐旧版本缺失的键
        default = _default_state()
        for k, v in default.items():
            state.setdefault(k, v)
        return state
    except (json.JSONDecodeError, OSError):
        return _default_state()


def save_state(state: dict) -> None:
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────────
# 持仓管理
# ─────────────────────────────────────────────────────────────

def _pos_key(account: str, symbol: str) -> str:
    return f"{account}:{symbol.upper()}"


def open_position(account: str, symbol: str, entry_price: float, shares: float,
                  stop_price: float, atr: float, strategy: str = "swing") -> None:
    """登记新持仓。"""
    state = load_state()
    key = _pos_key(account, symbol)
    state["positions"][key] = {
        "account":     account,
        "symbol":      symbol.upper(),
        "entry_price": entry_price,
        "shares":      shares,
        "initial_stop": stop_price,
        "current_stop": stop_price,
        "atr_at_entry": atr,
        "highest_since_entry": entry_price,
        "entry_date":  date.today().isoformat(),
        "strategy":    strategy,
    }
    save_state(state)


def close_position(account: str, symbol: str, exit_price: float) -> dict:
    """关仓并记录到 recent_trades（用于连亏锁仓判断）。"""
    state = load_state()
    key = _pos_key(account, symbol)
    pos = state["positions"].pop(key, None)
    if pos is None:
        return {"error": f"无持仓: {key}"}

    pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
    pnl_abs = (exit_price - pos["entry_price"]) * pos["shares"]

    trade_log = {
        "account":     account,
        "symbol":      symbol.upper(),
        "entry_price": pos["entry_price"],
        "exit_price":  exit_price,
        "shares":      pos["shares"],
        "pnl_pct":     round(pnl_pct, 3),
        "pnl_abs":     round(pnl_abs, 2),
        "opened_at":   pos["entry_date"],
        "closed_at":   date.today().isoformat(),
    }

    state["recent_trades"].append(trade_log)
    # 只保留最近 50 笔
    state["recent_trades"] = state["recent_trades"][-50:]

    save_state(state)
    return trade_log


def get_positions(account: str | None = None) -> list[dict]:
    state = load_state()
    pos_list = list(state["positions"].values())
    if account:
        pos_list = [p for p in pos_list if p["account"] == account]
    return pos_list


def update_position_trailing(account: str, symbol: str,
                             new_highest: float | None = None,
                             new_stop: float | None = None) -> None:
    """盘中更新持仓的最高价与跟踪止损。"""
    state = load_state()
    key = _pos_key(account, symbol)
    if key not in state["positions"]:
        return
    if new_highest is not None:
        state["positions"][key]["highest_since_entry"] = max(
            state["positions"][key]["highest_since_entry"], new_highest
        )
    if new_stop is not None:
        # 止损只上移不下移
        state["positions"][key]["current_stop"] = max(
            state["positions"][key]["current_stop"], new_stop
        )
    save_state(state)


# ─────────────────────────────────────────────────────────────
# 交易计数（当日）
# ─────────────────────────────────────────────────────────────

def today_key() -> str:
    return date.today().isoformat()


def get_daily_trade_count(account: str) -> int:
    state = load_state()
    return state["daily_trades"].get(today_key(), {}).get(account, 0)


def increment_daily_trade_count(account: str) -> int:
    state = load_state()
    today = today_key()
    state["daily_trades"].setdefault(today, {})
    state["daily_trades"][today][account] = state["daily_trades"][today].get(account, 0) + 1
    # 清理 30 天前的记录
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    state["daily_trades"] = {
        d: v for d, v in state["daily_trades"].items() if d >= cutoff
    }
    save_state(state)
    return state["daily_trades"][today][account]


# ─────────────────────────────────────────────────────────────
# 连亏检测
# ─────────────────────────────────────────────────────────────

def consecutive_losses(account: str, n: int = 2) -> bool:
    """True = 最近 n 笔都是亏损（今日的）。"""
    state = load_state()
    today = today_key()
    today_trades = [
        t for t in state["recent_trades"]
        if t["account"] == account and t["closed_at"] == today
    ]
    if len(today_trades) < n:
        return False
    last_n = today_trades[-n:]
    return all(t["pnl_pct"] < 0 for t in last_n)


# ─────────────────────────────────────────────────────────────
# 账户峰值与回撤
# ─────────────────────────────────────────────────────────────

def update_account_value(account: str, current_value: float) -> dict:
    """
    更新账户当前市值。自动维护峰值。
    返回：{current, peak, drawdown_pct}
    """
    state = load_state()
    peak = state["account_peaks"].get(account, current_value)
    peak = max(peak, current_value)
    state["account_peaks"][account] = peak
    state["account_current"][account] = current_value
    drawdown_pct = (current_value / peak - 1) * 100 if peak > 0 else 0
    save_state(state)
    return {
        "account":       account,
        "current":       current_value,
        "peak":          peak,
        "drawdown_pct":  round(drawdown_pct, 3),
    }


def get_account_drawdown(account: str) -> float | None:
    """返回当前从峰值的回撤百分比（负数），未登记返回 None。"""
    state = load_state()
    peak = state["account_peaks"].get(account)
    cur = state["account_current"].get(account)
    if peak is None or cur is None or peak <= 0:
        return None
    return (cur / peak - 1) * 100


# ─────────────────────────────────────────────────────────────
# 锁仓
# ─────────────────────────────────────────────────────────────

def lock_account(account: str, days: int = 1, reason: str = "") -> None:
    state = load_state()
    until = (date.today() + timedelta(days=days)).isoformat()
    state["locked_until"][account] = {"until": until, "reason": reason}
    save_state(state)


def is_locked(account: str) -> tuple[bool, str]:
    """返回 (is_locked, reason)。"""
    state = load_state()
    entry = state["locked_until"].get(account)
    if not entry:
        return False, ""
    # 兼容两种格式
    until = entry["until"] if isinstance(entry, dict) else entry
    reason = entry.get("reason", "") if isinstance(entry, dict) else ""
    if date.today().isoformat() <= until:
        return True, f"锁仓至 {until}（{reason}）"
    # 过期自动解锁
    state["locked_until"].pop(account, None)
    save_state(state)
    return False, ""


# ─────────────────────────────────────────────────────────────
# CLI 查看
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = load_state()
    print("─" * 50)
    print("  交易状态快照")
    print("─" * 50)
    print(f"  最后更新: {s.get('last_updated')}")
    print(f"  当前持仓: {len(s['positions'])} 个")
    for k, p in s["positions"].items():
        print(f"    {k}  entry={p['entry_price']}  stop={p['current_stop']}  "
              f"shares={p['shares']}  since={p['entry_date']}")
    print(f"  今日交易次数: {s['daily_trades'].get(today_key(), {})}")
    print(f"  账户峰值: {s['account_peaks']}")
    print(f"  账户当前: {s['account_current']}")
    print(f"  锁仓状态: {s['locked_until']}")
    print(f"  最近关仓: {len(s['recent_trades'])} 笔")
    for t in s["recent_trades"][-5:]:
        emoji = "✅" if t["pnl_pct"] > 0 else "🔴"
        print(f"    {emoji} {t['symbol']} ({t['account']})  "
              f"{t['pnl_pct']:+.2f}%  closed={t['closed_at']}")
    print("─" * 50)
