"""CLI: python -m tech_score verify macd [--period 10y] [--universe verification]"""
from __future__ import annotations

import argparse

from .verify import verify, verify_all, verify_composite
from .composite import score as composite_score_fn


def main():
    p = argparse.ArgumentParser(prog="tech_score")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify", help="Backtest a blogger rule on the verification universe")
    pv.add_argument("rule", help="rule module name, e.g. macd")
    pv.add_argument("--period", default="10y")
    pv.add_argument("--universe", default="verification",
                    choices=["verification", "personal"])

    pa = sub.add_parser("verify-all", help="Run every rule and print a leaderboard")
    pa.add_argument("--period", default="10y")
    pa.add_argument("--universe", default="verification",
                    choices=["verification", "personal"])

    pc = sub.add_parser("verify-composite", help="Backtest the composite-score signal")
    pc.add_argument("--period", default="10y")
    pc.add_argument("--universe", default="verification",
                    choices=["verification", "personal"])
    pc.add_argument("--hi", type=float, default=65)
    pc.add_argument("--lo", type=float, default=35)
    pc.add_argument("--min-share", type=float, default=0.60)
    pc.add_argument("--min-adv-m", type=float, default=10.0,
                    help="Min 20d avg-dollar-volume in USD millions (default 10)")
    pc.add_argument("--no-regime", action="store_true",
                    help="Disable regime filter (use pure composite confluence)")

    ps = sub.add_parser("score", help="Print latest composite score for one symbol")
    ps.add_argument("symbol")
    ps.add_argument("--period", default="3y")
    ps.add_argument("--min-adv-m", type=float, default=10.0)

    args = p.parse_args()
    if args.cmd == "verify":
        verify(args.rule, period=args.period, universe=args.universe)
    elif args.cmd == "verify-all":
        verify_all(period=args.period, universe=args.universe)
    elif args.cmd == "verify-composite":
        verify_composite(period=args.period, universe=args.universe,
                         hi=args.hi, lo=args.lo, min_share=args.min_share,
                         min_adv_usd=args.min_adv_m * 1e6,
                         regime_filter=not args.no_regime)
    elif args.cmd == "score":
        import json
        out = composite_score_fn(args.symbol, period=args.period,
                                 min_adv_usd=args.min_adv_m * 1e6)
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
