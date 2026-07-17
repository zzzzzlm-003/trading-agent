"""Composite scorer.

Aggregates all rule modules in tech_score.rules into a single 0–100 score
with per-family sub-scores and a buy/sell/hold direction.

Logic:
  1. Each rule outputs signal ∈ {-1, 0, +1} per bar.
  2. Rules are grouped by FAMILY (trend / momentum / volatility / volume / pattern / sr).
  3. Family score = mean(signal) inside family, scaled to 0–100 (50 = neutral).
  4. Composite = weighted sum of family scores using FAMILY_WEIGHTS.
  5. Vote count: how many indicators currently say buy vs sell (ignoring 0).
  6. Direction:
       composite ≥ 65 AND bullish vote share ≥ 60% → BUY
       composite ≤ 35 AND bearish vote share ≥ 60% → SELL
       else → HOLD.
  7. Liquidity gate: signals for symbols failing `passes_liquidity` are returned
     with `direction="SKIP_ILLIQUID"` and the raw score for inspection.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List

import numpy as np
import pandas as pd

from . import rules as _rules_pkg
from .data import fetch
from .filters import avg_dollar_volume, infer_fx, passes_liquidity, size_bucket


# 用户定的族间手动比例（sums to 1.0 ideally；会自动归一化）
FAMILY_WEIGHTS = {
    "trend":      0.30,
    "momentum":   0.25,
    "volume":     0.15,
    "volatility": 0.15,
    "pattern":    0.10,
    "sr":         0.05,
}


def _load_rules() -> list:
    mods = []
    for m in pkgutil.iter_modules(_rules_pkg.__path__):
        mods.append(importlib.import_module(f"tech_score.rules.{m.name}"))
    return mods


def signal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of per-bar event signals (only fire on cross day)."""
    mods = _load_rules()
    sigs = {}
    for mod in mods:
        try:
            sigs[mod.NAME] = mod.signal(df).astype(int)
        except Exception as e:
            print(f"  ⚠️  {mod.NAME}: {e}")
    out = pd.DataFrame(sigs, index=df.index).fillna(0).astype(int)
    return out


def position_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill last non-zero event → persistent 'current state' per rule.

    Used for scoring today: if MACD fired a golden cross 5 days ago and no death
    cross since, MACD's current state is +1.
    """
    sig_mat = signal_matrix(df)
    pos = sig_mat.replace(0, np.nan).ffill().fillna(0).astype(int)
    return pos


def _family_of(rule_name: str) -> str:
    mod = importlib.import_module(f"tech_score.rules.{rule_name}")
    return mod.FAMILY


def family_scores(sig_mat: pd.DataFrame) -> pd.DataFrame:
    """For each bar, mean(signal) per family → scaled to 0-100 (50 neutral)."""
    fam_map = {c: _family_of(c) for c in sig_mat.columns}
    fam_cols = {}
    for fam in set(fam_map.values()):
        cols = [c for c, f in fam_map.items() if f == fam]
        fam_cols[fam] = sig_mat[cols].mean(axis=1)
    fam_df = pd.DataFrame(fam_cols)
    # scale: signal ∈ [-1, +1] → score ∈ [0, 100]
    return 50 + fam_df * 50


def composite_score(fam_df: pd.DataFrame) -> pd.Series:
    """Weighted average of family scores; normalizes weights to families present."""
    cols = [c for c in FAMILY_WEIGHTS if c in fam_df.columns]
    w = np.array([FAMILY_WEIGHTS[c] for c in cols])
    w = w / w.sum()
    return (fam_df[cols] * w).sum(axis=1)


def _vote_share(sig_mat: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-bar bull/bear vote counts and bull share (of active votes)."""
    bull = (sig_mat > 0).sum(axis=1)
    bear = (sig_mat < 0).sum(axis=1)
    active = bull + bear
    share = (bull / active.replace(0, np.nan)).fillna(0.5)
    return bull, bear, share


def direction(composite: float, bull_share: float,
              hi: float = 65, lo: float = 35, min_share: float = 0.60) -> str:
    if composite >= hi and bull_share >= min_share:
        return "BUY"
    if composite <= lo and (1 - bull_share) >= min_share:
        return "SELL"
    return "HOLD"


def score(symbol: str, period: str = "3y", min_adv_usd: float = 10e6) -> dict:
    """High-level call: fetch + compute + return dict for latest bar."""
    df = fetch(symbol, period=period)
    fx = infer_fx(symbol)
    if not passes_liquidity(df, min_adv_usd=min_adv_usd, fx_rate_to_usd=fx):
        # still compute for visibility, but flag
        bucket = size_bucket(df, fx_rate_to_usd=fx)
    else:
        bucket = size_bucket(df, fx_rate_to_usd=fx)

    # for scoring today use persistent positions, not just cross events
    pos_mat = position_matrix(df)
    fam = family_scores(pos_mat)
    comp = composite_score(fam)
    bull, bear, share = _vote_share(pos_mat)

    last = df.index[-1]
    comp_v = float(comp.iloc[-1])
    share_v = float(share.iloc[-1])
    dir_ = direction(comp_v, share_v)
    if not passes_liquidity(df, min_adv_usd=min_adv_usd, fx_rate_to_usd=fx):
        dir_ = f"{dir_}·LOW_LIQ"

    # latest per-rule persistent state (+1 bullish / -1 bearish / 0 flat)
    detail = {c: int(pos_mat[c].iloc[-1]) for c in pos_mat.columns}

    return {
        "symbol":   symbol,
        "as_of":    str(last.date()),
        "size":     bucket,
        "adv_usd":  float(avg_dollar_volume(df).iloc[-1] * fx),
        "composite": round(comp_v, 1),
        "direction": dir_,
        "vote":     f"{int(bull.iloc[-1])}↑ / {int(bear.iloc[-1])}↓",
        "bull_share_%": round(share_v * 100, 0),
        "by_family": {f: round(float(fam[f].iloc[-1]), 1) for f in fam.columns},
        "rules":    detail,
    }


# ──────────────────────────────────────────────────────────────
# composite backtest: treat composite score as an entry/exit signal
# ──────────────────────────────────────────────────────────────
def composite_signal(df: pd.DataFrame, hi: float = 65, lo: float = 35,
                     min_share: float = 0.60,
                     regime_filter: bool = True) -> pd.Series:
    """Backtest signal with optional regime filter.

    When `regime_filter=True`:
      - trending_up  → stay long (emit +1 once, no exits) — ride the trend
      - ranging      → use composite confluence (enter on comp≥hi, exit on comp≤lo)
      - trending_down → exit on any composite sell signal, no new entries

    When `regime_filter=False`: original composite-only behavior (pure confluence).
    """
    pos_mat = position_matrix(df)
    fam = family_scores(pos_mat)
    comp = composite_score(fam)
    _, _, share = _vote_share(pos_mat)

    enter = (comp >= hi) & (share >= min_share)
    exit_ = (comp <= lo) & ((1 - share) >= min_share)

    if regime_filter:
        from .regime import classify
        reg = classify(df)
        # trending_up: enter as soon as regime turns up, stay in
        trend_up_start = (reg == "trending_up") & (reg.shift(1) != "trending_up")
        # trending_down: exit on any bear confluence within down regime
        down_exit = exit_ & (reg == "trending_down")
        # ranging: use composite confluence both directions
        rng = reg == "ranging"
        rng_enter = enter & rng
        rng_exit = exit_ & rng

        s = pd.Series(0, index=df.index, dtype=int)
        s[(trend_up_start | rng_enter) & ~(trend_up_start | rng_enter).shift(1).fillna(False)] = 1
        s[(rng_exit | down_exit) & ~(rng_exit | down_exit).shift(1).fillna(False)] = -1
        return s

    # no-regime mode
    s = pd.Series(0, index=df.index, dtype=int)
    s[enter & ~enter.shift(1).fillna(False)] = 1
    s[exit_ & ~exit_.shift(1).fillna(False)] = -1
    return s
