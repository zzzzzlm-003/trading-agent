# Trading Agent — Technical-Signal Research Pipeline

Modular research pipeline for rule-based technical signals: 19 indicator rule modules, per-rule backtest verification against buy-and-hold, regime-aware composite scoring, and an LLM methodology agent built on top of transcribed research notes.

<!-- TODO: add results chart -->
![Results](docs/results.png)

## Headline results (per-rule verification, 26-symbol universe)

Every rule is backtested per symbol and compared against buy-and-hold (B&H) on the same window. Excerpt from `tech_score/results/leaderboard.csv`:

| Rule | Family | Median Sharpe | Median B&H Sharpe | % beat B&H |
|---|---|---|---|---|
| rvwap | volume | 0.72 | 0.73 | 36% |
| atr_kc | volatility | 0.68 | 0.72 | 38% |
| boll | volatility | 0.61 | 0.72 | 4% |
| rsi | momentum | 0.61 | 0.72 | 27% |
| sar | momentum | 0.57 | 0.72 | 12% |

The honest takeaway: **most literal single-indicator rules do not beat buy-and-hold** — which is exactly why the pipeline verifies every rule individually before anything reaches the composite, and why the composite is regime-aware rather than a naive vote.

## Architecture

```
tech_score/
├── data.py        # OHLCV fetch + caching (yfinance / AkShare)
├── rules/         # 19 rule modules, one indicator each, uniform signal(df) API
├── verify.py      # Per-rule event-study backtest vs buy-and-hold
├── regime.py      # Trend/range regime classification
├── composite.py   # Regime-aware weighted composite score
├── filters.py     # Liquidity / data-quality filters
└── results/       # Verification CSVs + leaderboard
```

Around the scoring core:

| File | Role |
|---|---|
| `analyzer.py` | Single-ticker technical snapshot: indicators + candlestick patterns + (A-share) money flow + (US) options context |
| `signal_generator.py` / `trade_gate.py` / `risk_manager.py` | L1 signal → gate → position-sizing chain |
| `backtest.py` | Strategy-level backtesting |
| `transcript_strategy_agent.py` | Aggregates transcribed research notes into a queryable methodology agent (Claude API) |
| `get_youtube_transcript.py` | Whisper-based transcription utility — for audio/video you have rights to |

## Rule modules — design rationale

One line per module in `tech_score/rules/` (each exports a uniform `signal(df)`):

- **ma** — MA5/MA20 crossover as the baseline trend-following signal.
- **macd** — MACD line/signal cross on daily+ bars; momentum confirmation.
- **rsi** — fast RSI(6) for overbought/oversold timing.
- **kdj** — K/D golden cross in the low zone (<20) as a bottom-fishing entry.
- **boll** — touch of the lower Bollinger band as a mean-reversion buy.
- **atr_kc** — Keltner-style channel (EMA20 ± 2·ATR14) for volatility breakouts.
- **bbi** — average of MA3/6/12/24 as a single bull/bear dividing line.
- **cci** — CCI(14) crossing up through −100 as an oversold rebound.
- **cdl_patterns** — aggregate vote over TA-Lib's 61 candlestick patterns.
- **demark** — TD Setup-9 count as a trend-exhaustion counter.
- **divergence** — price/indicator divergence as an early reversal warning.
- **fib** — retracement zones off the 60-day swing high/low.
- **gmma** — alignment of short vs long EMA groups for trend strength.
- **ichimoku** — price-vs-cloud position plus Tenkan/Kijun cross.
- **obv** — OBV MA20 crossing MA60 as a volume-led breakout.
- **rvwap** — 5-day rolling VWAP as an institutional cost line.
- **sar** — parabolic SAR flip for stop-and-reverse trend entries.
- **volume_price** — classic volume-price relation heuristics (rising volume + rising price = continuation).
- **wvad** — intrabar buying-pressure accumulation ((C−O)/(H−L)·V).

## Usage

```bash
pip install ta-lib yfinance akshare pandas numpy mplfinance openai-whisper anthropic

# Verify every rule on the universe and rebuild the leaderboard
python -m tech_score

# Single-ticker snapshot (US / A-share / HK index proxy)
python analyzer.py NVDA
python analyzer.py 600519
```

## Methodology agent (optional)

`transcript_strategy_agent.py` builds a strategy profile and a system prompt from transcribed research notes in `transcripts/` (not tracked in git), then answers questions with retrieved evidence:

```bash
python transcript_strategy_agent.py --build
python transcript_strategy_agent.py --ask "Does this system weight MACD or RSI more?"
```

Transcription: use `get_youtube_transcript.py` to transcribe audio/video you have rights to; outputs go to `transcripts/`.

## Data sources

| Market | Prices | Flow / options |
|---|---|---|
| US | yfinance | yfinance options chain |
| A-share | AkShare (Eastmoney) | AkShare money flow |
| HK tech | yfinance (3033.HK proxy) | — |

---

*For research/education only, not investment advice.*
