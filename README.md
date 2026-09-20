# transfer-market-signals

[Algorithm guide: pseudocode, time complexity, and memory](docs/ALGORITHM_GUIDE.md).

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![NewsAPI](https://img.shields.io/badge/API-NewsAPI-red)
![API-Football](https://img.shields.io/badge/API-API--Football-blue)

> A quant-style football (soccer) transfer market intelligence engine. Predicts likely transfer targets, computes fair valuations, and ranks opportunities using a multi-signal model.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                          │
│                                                                              │
│   ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────────┐ │
│   │  API-Football   │   │    NewsAPI        │   │   Hardcoded Datasets     │ │
│   │  (RapidAPI)     │   │   (Free Tier)     │   │  (FFP / Contract Data)  │ │
│   └────────┬────────┘   └────────┬──────────┘   └────────────┬─────────────┘ │
└────────────┼────────────────────┼──────────────────────────┼───────────────┘
             │                    │                          │
             ▼                    ▼                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE PIPELINE                                      │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐  ┌───────────┐  │
│  │  AgeValueCurve   │  │  Performance     │  │  FFP       │  │ Contract  │  │
│  │  (career arc,    │  │  Trajectory      │  │  Headroom  │  │ Signals   │  │
│  │   depreciation)  │  │  (polyfit trend) │  │  Analyzer  │  │ (urgency) │  │
│  └────────┬─────────┘  └───────┬──────────┘  └─────┬──────┘  └─────┬─────┘  │
│           │                   │                   │               │        │
│           └───────────────────┴──────┬────────────┘───────────────┘        │
│                                      │                                      │
│           ┌──────────────────────────▼──────────────────┐                   │
│           │         News Sentiment Scorer                │                   │
│           │  (NewsAPI articles → keyword scoring → hype) │                   │
│           └──────────────────────────┬──────────────────┘                   │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        VALUATION MODEL                                       │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  TransferValuator                                                   │   │
│   │  · fair_value()       ← age curve + performance + contract discount  │   │
│   │  · transfer_probability()  ← FFP + sentiment + urgency + need fit   │   │
│   └──────────────────────────────────┬──────────────────────────────────┘   │
└─────────────────────────────────────┼─────────────────────────────────────-─┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   TRANSFER PROBABILITY SCORER                                │
│                                                                              │
│   budget filter → position filter → rank_transfer_targets() → scored table  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         RANKED OUTPUT                                        │
│                                                                              │
│  Rank │ Player              │ Pos │ Age │ Fair Value │ Mkt Val │ Prob │ Score │
│  ───────────────────────────────────────────────────────────────────────     │
│   1   │ Leny Yoro           │ CB  │ 18  │ €85.0m     │ €70.0m  │ 0.82 │ 91.2  │
│   2   │ Manuel Ugarte       │ CM  │ 23  │ €62.0m     │ €55.0m  │ 0.75 │ 87.4  │
│   3   │ Joao Neves          │ CM  │ 20  │ €75.0m     │ €70.0m  │ 0.71 │ 83.6  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Age-Value Curves** — Position-specific career arc modeling (GK peaks 27–31, FW peaks 24–28). Computes undervaluation scores and 5-year value projections.
- **Performance Trajectory** — Linear regression (numpy polyfit) over 3 seasons of stats. Labels players as emerging / prime / declining / veteran.
- **FFP Headroom Analyzer** — 2025/26 wage bill, revenue, and amortized debt data for 20 top clubs. Ranks clubs by available budget and identifies FFP-stressed sellers.
- **Contract Signal Analysis** — 30 real players with contract expiry, release clause, and reported interest data. Urgency scoring and free-agent value uplift.
- **News Sentiment Scoring** — Real NewsAPI integration. Keyword-based transfer sentiment (-1 to +1) with coverage volume weighting.
- **Transfer Valuator** — Master multi-factor model combining all signals into fair value and transfer probability scores.
- **Ranked Target Lists** — Filter by budget and position, ranked by composite opportunity score.

---

## Quickstart

```bash
# Clone and create an isolated environment (macOS/Linux)
git clone https://github.com/fatehaszaman/transfer-market-signals.git
cd transfer-market-signals
python3 -m venv .venv
source .venv/bin/activate

# Offline example: standard library only, no API keys required
python -c "from signals.age_value_curve import AgeValueCurve; print(AgeValueCurve().expected_value_at_age(50.0, 24, 'CM'))"
```

This runs one implemented heuristic module, not the full ranking pipeline.
The repository does not currently ship `requirements.txt`, `.env.example`,
or `examples/run_transfer_analysis.py`; those are not setup prerequisites.

---

## API Keys

| Service | Usage | Free Tier |
|---------|-------|-----------|
| [NewsAPI](https://newsapi.org) | Player news sentiment | 100 req/day |
| [API-Football (RapidAPI)](https://rapidapi.com/api-sports/api/api-football) | Player stats, transfers, contracts | 100 req/day |

Set in `.env`:
```
NEWS_API_KEY=your_newsapi_key_here
RAPIDAPI_KEY=your_rapidapi_key_here
```

---

## Sample Output

Illustrative report mockup for a €150m budget targeting CB and CM positions.
This is not output from a shipped full-pipeline runner or a current forecast:

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║            TRANSFER MARKET SIGNALS — TARGET ANALYSIS REPORT                             ║
║            Budget: €150m  |  Positions: CB, CM  |  Window: Summer 2025                  ║
╠════╦═══════════════════════╦═════╦═════╦═══════════════╦════════════╦═══════╦═══════════╣
║ #  ║ Player                ║ Pos ║ Age ║ Fair Value    ║ Mkt Value  ║ Prob  ║ Score     ║
╠════╬═══════════════════════╬═════╬═════╬═══════════════╬════════════╬═══════╬═══════════╣
║  1 ║ Leny Yoro             ║ CB  ║ 18  ║ €85.0m        ║ €70.0m     ║ 0.82  ║ 91.2      ║
║  2 ║ Manuel Ugarte         ║ CM  ║ 23  ║ €62.0m        ║ €55.0m     ║ 0.75  ║ 87.4      ║
║  3 ║ Joao Neves            ║ CM  ║ 20  ║ €75.0m        ║ €70.0m     ║ 0.71  ║ 83.6      ║
║  4 ║ Dean Huijsen          ║ CB  ║ 19  ║ €55.0m        ║ €45.0m     ║ 0.68  ║ 79.1      ║
║  5 ║ Manu Kone             ║ CM  ║ 23  ║ €48.0m        ║ €40.0m     ║ 0.64  ║ 74.8      ║
╚════╩═══════════════════════╩═════╩═════╩═══════════════╩════════════╩═══════╩═══════════╝

Top Buy Recommendation: Leny Yoro
  Age-value curve: ASCENDING (+12.5% upside vs market)
  Performance trajectory: emerging (slope: +0.31/season)
  Contract urgency: HIGH (18 months remaining)
  FFP fit: 6 potential suitors with sufficient headroom
  Transfer probability: 82%
```

---

## Module Reference

```
transfer-market-signals/
├── signals/
│   ├── age_value_curve.py        # Career arc + depreciation + undervaluation scoring
│   ├── performance_trajectory.py # Linear regression over seasonal stats
│   ├── ffp_headroom.py           # FFP / PSR spending headroom for top clubs
│   ├── contract_signal.py        # Contract expiry urgency + release clause plays
│   └── sentiment_scorer.py       # NewsAPI sentiment scoring
├── valuation/
│   └── transfer_valuator.py      # Master valuation + probability + ranking model
├── data/
│   ├── schemas.py               # Data models
│   └── cache.py                 # Cache utilities
├── tests/
│   └── regression/              # Scaffolding; no executable tests yet
├── config.py
└── README.md
```

---

## Installation

The offline age-curve example above needs only Python 3.10 or newer. Other
modules have separate third-party dependencies; a consolidated installation
manifest is not currently provided.

## Testing status

The test directories are scaffolding, not an executable test suite. No passing
test status or coverage percentage is claimed. The offline example is only a
smoke check; the heuristic projections have not been validated by this cleanup.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
