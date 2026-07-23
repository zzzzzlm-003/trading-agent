"""Cross-reference 犀牛哥的 tech_score vs 阿云的实际呼叫.

Input: yuncun/results/signal_forward_returns.csv (102 actions, with 1m/3m/6m fwd returns)
Output: for each action, compute tech_score composite on the action date using only
        history BEFORE that date (no look-ahead), then analyze:
  - Distribution of composite scores on 阿云 action days
  - Does aligned direction (她买 + composite>50) beat misaligned?
  - Hit rate split by composite bucket
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .composite import composite_score, family_scores, position_matrix, _vote_share
from .data import fetch

YUNCUN_DIR = Path(__file__).resolve().parent.parent / "yuncun"
ACTIONS_CSV = YUNCUN_DIR / "results" / "signal_forward_returns.csv"
TICKERS_YML = YUNCUN_DIR / "code" / "asset_tickers.yml"


def _load_asset_map() -> dict:
    """Load asset→ticker map. Supports both old schema (`primary`) and new
    schema (`etf_primary`/`stock_primary`). Prefers ETF for alignment since
    that's what 阿云 most often calls."""
    with open(TICKERS_YML) as f:
        y = yaml.safe_load(f)
    out = {}
    for asset, spec in y.items():
        if not isinstance(spec, dict):
            continue
        # new schema
        if "etf_primary" in spec:
            out[asset] = spec["etf_primary"]
        elif "stock_primary" in spec:
            out[asset] = spec["stock_primary"]
        # old schema
        elif "primary" in spec:
            out[asset] = spec["primary"]
    return out


def score_at(df_full: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """Composite score using only bars up to and including as_of."""
    df = df_full.loc[:as_of]
    if len(df) < 100:
        return {"composite": np.nan, "bull_share": np.nan}
    pos = position_matrix(df)
    fam = family_scores(pos)
    comp = composite_score(fam)
    _, _, share = _vote_share(pos)
    return {
        "composite": float(comp.iloc[-1]),
        "bull_share": float(share.iloc[-1]),
    }


def align() -> pd.DataFrame:
    actions = pd.read_csv(ACTIONS_CSV)
    actions = actions[actions.scenario == "blogger"].copy()
    actions["date"] = pd.to_datetime(actions["date"])
    amap = _load_asset_map()
    print(f"{len(actions)} actions across {actions.asset.nunique()} assets")

    price_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, a in actions.iterrows():
        ticker = amap.get(a.asset)
        if not ticker:
            continue
        if ticker not in price_cache:
            try:
                price_cache[ticker] = fetch(ticker, period="5y")
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")
                price_cache[ticker] = None
        df = price_cache[ticker]
        if df is None:
            continue
        s = score_at(df, a.date)
        rows.append({
            "date": a.date.date(),
            "asset": a.asset,
            "ticker": ticker,
            "reason": a.reason,
            "composite": round(s["composite"], 1) if np.isfinite(s.get("composite", np.nan)) else np.nan,
            "bull_share_%": round(s["bull_share"] * 100, 0) if np.isfinite(s.get("bull_share", np.nan)) else np.nan,
            "fwd_1m": a["1m"],
            "fwd_3m": a["3m"],
            "fwd_6m": a["6m"],
        })
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame) -> None:
    print("\n== Tech score distribution on 阿云 action days (all buy/add) ==")
    print(df["composite"].describe().round(1).to_string())

    print("\n== Forward returns split by composite bucket ==")
    df = df.dropna(subset=["composite"])
    df["bkt"] = pd.cut(df["composite"], bins=[0, 40, 50, 60, 100],
                       labels=["<40 (bear)", "40-50", "50-60", ">60 (bull)"])
    g = df.groupby("bkt", observed=False).agg(
        n=("composite", "size"),
        mean_1m=("fwd_1m", "mean"),
        win_1m=("fwd_1m", lambda x: (x > 0).mean()),
        mean_3m=("fwd_3m", "mean"),
        win_3m=("fwd_3m", lambda x: (x > 0).mean()),
        mean_6m=("fwd_6m", "mean"),
        win_6m=("fwd_6m", lambda x: (x > 0).mean()),
    )
    print(g.round(3).to_string())

    # Alignment hit
    print("\n== Alignment hit rate ==")
    aligned = df[df["composite"] >= 50]
    miss = df[df["composite"] < 50]
    for horizon in ["fwd_1m", "fwd_3m", "fwd_6m"]:
        a_mean, a_wr = aligned[horizon].mean(), (aligned[horizon] > 0).mean()
        m_mean, m_wr = miss[horizon].mean(), (miss[horizon] > 0).mean()
        print(f"  {horizon}: aligned (composite≥50, n={len(aligned)}) mean={a_mean:+.1%} win={a_wr:.0%} "
              f"| misaligned (<50, n={len(miss)}) mean={m_mean:+.1%} win={m_wr:.0%} "
              f"| Δ mean {a_mean - m_mean:+.1%}")


def main():
    df = align()
    out = YUNCUN_DIR / "results" / "yuncun_vs_techscore.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    analyze(df)


if __name__ == "__main__":
    main()
