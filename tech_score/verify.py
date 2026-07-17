"""Rule verification harness.

For each symbol in verification universe, run the rule, build a long-only strategy:
  - Enter on signal=+1, exit on signal=-1 (or on opposite signal), hold otherwise.
Compare to Buy&Hold: total return, Sharpe, max drawdown, win rate, # trades.

A rule is considered "validated" if across the verification universe it shows:
  - Median Sharpe ≥ 0.3
  - Positive expectancy in > 60% of tickers
  - Max drawdown ≤ Buy&Hold max drawdown on the same ticker
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from . import rules as _rules_pkg
from .data import fetch, load_universe
from .filters import avg_dollar_volume, infer_fx, passes_liquidity, size_bucket


def discover_rules() -> list[str]:
    """Return list of rule module names under tech_score.rules."""
    return [m.name for m in pkgutil.iter_modules(_rules_pkg.__path__)]


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def verification_tickers() -> list[str]:
    u = load_universe()["verification"]
    return [t for group in u.values() for t in group]


def backtest_one(df: pd.DataFrame, sig: pd.Series, init_cash: float = 10_000.0):
    """Long-only: entries where sig==+1, exits where sig==-1. Fees 0.05%."""
    close = df["Close"].astype(float)
    entries = sig == 1
    exits = sig == -1
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=0.0005,
        freq="1D",
    )
    bh = vbt.Portfolio.from_holding(close=close, init_cash=init_cash, freq="1D")
    return pf, bh


def _stats_row(symbol: str, pf, bh, df=None) -> dict:
    def _safe(x, default=np.nan):
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    row = {
        "symbol": symbol,
        "rule_return_%": _safe(pf.total_return() * 100),
        "bh_return_%": _safe(bh.total_return() * 100),
        "alpha_%": _safe((pf.total_return() - bh.total_return()) * 100),
        "rule_sharpe": _safe(pf.sharpe_ratio()),
        "bh_sharpe": _safe(bh.sharpe_ratio()),
        "rule_maxdd_%": _safe(pf.max_drawdown() * 100),
        "bh_maxdd_%": _safe(bh.max_drawdown() * 100),
        "n_trades": int(pf.trades.count()),
        "win_rate_%": _safe(pf.trades.win_rate() * 100),
        "expectancy_%": _safe(pf.trades.expectancy()),
    }
    if df is not None:
        fx = infer_fx(symbol)
        row["size"] = size_bucket(df, fx_rate_to_usd=fx)
        row["adv_usd_m"] = round(float(avg_dollar_volume(df).iloc[-20:].mean() * fx) / 1e6, 1)
    return row


def verify(rule_name: str, period: str = "10y", universe: str = "verification") -> pd.DataFrame:
    """Run rule on every symbol in the chosen universe group, return per-symbol stats."""
    mod = importlib.import_module(f"tech_score.rules.{rule_name}")
    print(f"\n== Verifying `{rule_name}` (family={mod.FAMILY}, 期={mod.EPISODE}) ==")

    u = load_universe()[universe]
    symbols = [t for group in u.values() for t in group] if isinstance(u, dict) else list(u)

    rows = []
    for sym in symbols:
        try:
            df = fetch(sym, period=period)
            if len(df) < 200:
                print(f"  ⏭️  {sym}: only {len(df)} bars, skip")
                continue
            sig = mod.signal(df)
            pf, bh = backtest_one(df, sig)
            row = _stats_row(sym, pf, bh, df=df)
            rows.append(row)
            print(f"  ✓ {sym}: rule {row['rule_return_%']:+.1f}% / BH {row['bh_return_%']:+.1f}% "
                  f"(α {row['alpha_%']:+.1f}%, Sharpe {row['rule_sharpe']:.2f} vs {row['bh_sharpe']:.2f}, "
                  f"DD {row['rule_maxdd_%']:.1f}% vs {row['bh_maxdd_%']:.1f}%, "
                  f"n={row['n_trades']}, win={row['win_rate_%']:.0f}%)")
        except Exception as e:
            print(f"  ✗ {sym}: {type(e).__name__}: {e}")

    df_out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"{rule_name}_{universe}.csv"
    df_out.to_csv(out_path, index=False)

    # Verdict
    if len(df_out) > 0:
        med_sharpe = df_out["rule_sharpe"].median()
        pct_positive_alpha = (df_out["alpha_%"] > 0).mean() * 100
        med_dd_ratio = (df_out["rule_maxdd_%"] / df_out["bh_maxdd_%"]).median()
        verdict = (med_sharpe >= 0.3 and pct_positive_alpha >= 50 and med_dd_ratio >= 1.0)
        # Note: maxdd is negative, "rule DD / BH DD >= 1" means rule DD shallower than BH DD
        print(f"\n-- Verdict for `{rule_name}` --")
        print(f"  N symbols: {len(df_out)}")
        print(f"  Median rule Sharpe: {med_sharpe:.2f}")
        print(f"  % symbols with positive alpha vs BH: {pct_positive_alpha:.0f}%")
        print(f"  Median DD ratio (rule/BH, ≥1 means shallower): {med_dd_ratio:.2f}")
        print(f"  >>> {'VALIDATED ✅' if verdict else 'NOT validated ❌ (reconsider or refine rule)'}")
        print(f"  Saved: {out_path}")

    return df_out


def verify_all(period: str = "10y", universe: str = "verification") -> pd.DataFrame:
    """Run every rule module under tech_score.rules, summarize into a leaderboard."""
    rule_names = discover_rules()
    print(f"Found {len(rule_names)} rules: {rule_names}")
    rows = []
    for name in rule_names:
        try:
            df = verify(name, period=period, universe=universe)
            if df is None or len(df) == 0:
                continue
            rows.append({
                "rule": name,
                "family": importlib.import_module(f"tech_score.rules.{name}").FAMILY,
                "episode": importlib.import_module(f"tech_score.rules.{name}").EPISODE,
                "n_symbols": len(df),
                "median_sharpe": df["rule_sharpe"].median(),
                "median_bh_sharpe": df["bh_sharpe"].median(),
                "median_alpha_%": df["alpha_%"].median(),
                "pct_beat_bh": (df["alpha_%"] > 0).mean() * 100,
                "median_win_rate_%": df["win_rate_%"].median(),
                "median_dd_ratio": (df["rule_maxdd_%"] / df["bh_maxdd_%"]).median(),
            })
        except Exception as e:
            print(f"  ✗ verify_all({name}): {e}")
    lb = pd.DataFrame(rows).sort_values("median_alpha_%", ascending=False)
    lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    print("\n=== LEADERBOARD (sorted by median α vs B&H) ===")
    print(lb.to_string(index=False))
    return lb


def verify_composite(period: str = "10y", universe: str = "verification",
                     hi: float = 65, lo: float = 35, min_share: float = 0.6,
                     min_adv_usd: float = 10e6,
                     regime_filter: bool = True) -> pd.DataFrame:
    """Backtest the composite-score signal (all rules combined) on the universe."""
    from .composite import composite_signal

    u = load_universe()[universe]
    symbols = [t for group in u.values() for t in group] if isinstance(u, dict) else list(u)

    print(f"\n== Composite verify (hi={hi}, lo={lo}, min_share={min_share}, "
          f"min ADV ≥ ${min_adv_usd/1e6:.0f}M) ==")
    rows = []
    for sym in symbols:
        try:
            df = fetch(sym, period=period)
            if len(df) < 200:
                print(f"  ⏭️  {sym}: only {len(df)} bars, skip")
                continue
            fx = infer_fx(sym)
            if not passes_liquidity(df, min_adv_usd=min_adv_usd, fx_rate_to_usd=fx):
                print(f"  ⏭️  {sym}: ADV too low (bucket={size_bucket(df, fx_rate_to_usd=fx)}), skip")
                continue
            sig = composite_signal(df, hi=hi, lo=lo, min_share=min_share,
                                   regime_filter=regime_filter)
            pf, bh = backtest_one(df, sig)
            rows.append(_stats_row(sym, pf, bh, df=df))
            r = rows[-1]
            print(f"  ✓ {sym} [{r['size']} ${r['adv_usd_m']:.0f}M]: "
                  f"rule {r['rule_return_%']:+.1f}% / BH {r['bh_return_%']:+.1f}% "
                  f"(α {r['alpha_%']:+.1f}%, Sharpe {r['rule_sharpe']:.2f} vs {r['bh_sharpe']:.2f}, "
                  f"n={r['n_trades']}, win={r['win_rate_%']:.0f}%)")
        except Exception as e:
            print(f"  ✗ {sym}: {type(e).__name__}: {e}")

    df_out = pd.DataFrame(rows)
    suffix = "regime" if regime_filter else "norgm"
    out_path = RESULTS_DIR / f"composite_{universe}_{suffix}.csv"
    df_out.to_csv(out_path, index=False)

    if len(df_out) > 0:
        print("\n-- Overall --")
        print(f"  N symbols: {len(df_out)}")
        print(f"  Median rule Sharpe: {df_out['rule_sharpe'].median():.2f} "
              f"(BH {df_out['bh_sharpe'].median():.2f})")
        print(f"  % beat BH: {(df_out['alpha_%'] > 0).mean() * 100:.0f}%")
        print(f"  Median α: {df_out['alpha_%'].median():+.1f}%")
        if "size" in df_out.columns:
            print("\n-- By size bucket --")
            for bkt, sub in df_out.groupby("size"):
                print(f"  {bkt:5s}  n={len(sub):2d}  α {sub['alpha_%'].median():+.1f}%  "
                      f"beat BH {(sub['alpha_%']>0).mean()*100:.0f}%  "
                      f"Sharpe {sub['rule_sharpe'].median():.2f}")
        print(f"\n  Saved: {out_path}")

    return df_out
