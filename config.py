"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
Central configuration for the transfer-market-signals engine. All tuneable
parameters — weighting constants, API settings, cache paths, thresholds —
live here so that analysts can adjust the model without touching signal code.

Key design decision — the PSYCH_WEIGHT / PHYSICAL_WEIGHT ratio:
A technically elite player with low motivation still commands high transfer
value and performs above average. Psychological state adjusts at the margin.
The 1.5:1.0 physical-to-psychological weighting reflects this floor effect:
a deeply unhappy player (contentment score 30/100) still produces at ~80%
of their peak physical ability.
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (no-op if missing)
load_dotenv()

# ============================================================================ #
# Project paths                                                                  #
# ============================================================================ #

PROJECT_ROOT = Path(__file__).parent
CACHE_DB_PATH = PROJECT_ROOT / ".cache" / "etl_cache.db"
SAMPLE_PLAYERS_PATH = PROJECT_ROOT / "data" / "sample_players.json"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================================ #
# API credentials (loaded from .env)                                             #
# ============================================================================ #

NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST: str = "api-football-v1.p.rapidapi.com"

NEWS_API_BASE_URL     = "https://newsapi.org/v2"
API_FOOTBALL_BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

# ============================================================================ #
# Performance / Motivation Weighting                                             #
# ============================================================================ #
#
# Weighting: physical/technical (1.5) outweighs psychological/motivation (1.0).
# Rationale: a technically elite player with low motivation still commands high
# transfer value and performs above average. Psychological state adjusts at the
# margin — a deeply unhappy player (contentment score 30/100) still produces
# at ~80% of peak physical ability. The 1.5:1.0 ratio reflects this floor effect.

PSYCH_WEIGHT: float = 1.0
PHYSICAL_WEIGHT: float = 1.5
TOTAL_WEIGHT: float = PSYCH_WEIGHT + PHYSICAL_WEIGHT  # 2.5

# ============================================================================ #
# Valuation model weights (used in TransferValuator.batch_score)                #
# ============================================================================ #
# Each sub-signal's contribution to the composite score (must sum to 1.0)

VALUATION_WEIGHTS = {
    "age_curve":     0.20,   # Age-value curve undervaluation signal
    "performance":   0.25,   # Trajectory slope (ascending = high weight)
    "contract":      0.20,   # Urgency of contract situation
    "ffp_fit":       0.15,   # Number of clubs that can afford + need the player
    "sentiment":     0.10,   # News coverage hype score
    "motivation":    0.10,   # Contentment / psychological signal
}

assert abs(sum(VALUATION_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"VALUATION_WEIGHTS must sum to 1.0, got {sum(VALUATION_WEIGHTS.values())}"
)

# ============================================================================ #
# Transfer probability parameters                                                #
# ============================================================================ #

# Sigmoid steepness for transfer probability conversion
PROB_SIGMOID_STEEPNESS: float = 6.0

# Maximum uplift that motivation can add to base transfer probability
MOTIVATION_MAX_UPLIFT: float = 0.25

# Minimum transfer probability floor (even the happiest contract player
# has some non-zero baseline chance of a shock transfer)
TRANSFER_PROB_FLOOR: float = 0.03

# Maximum transfer probability ceiling (no transfer is certain)
TRANSFER_PROB_CEILING: float = 0.97

# ============================================================================ #
# Contentment score thresholds                                                   #
# ============================================================================ #

CONTENTMENT_VERY_HAPPY     = 80   # Strong stay signal
CONTENTMENT_CONTENT        = 60   # Neutral — would listen but not push
CONTENTMENT_RESTLESS       = 45   # Starting to look around
CONTENTMENT_UNHAPPY        = 30   # Actively seeking exit
CONTENTMENT_TOXIC          = 15   # Burned bridges, exit inevitable

# ============================================================================ #
# API rate limiting                                                               #
# ============================================================================ #

# NewsAPI free tier: 100 requests/day
NEWSAPI_RATE_LIMIT_PER_DAY: int = 100
NEWSAPI_REQUESTS_PER_SECOND: float = 0.5   # Conservative

# API-Football free tier: 100 requests/day
API_FOOTBALL_RATE_LIMIT_PER_DAY: int = 100
API_FOOTBALL_REQUESTS_PER_SECOND: float = 0.5

# ThreadPoolExecutor: max concurrent sentiment fetch threads
SENTIMENT_FETCH_MAX_WORKERS: int = 5

# ============================================================================ #
# Batch processing memory optimisation                                           #
# ============================================================================ #

# Chunk size for large player DataFrame processing (players per chunk)
BATCH_CHUNK_SIZE: int = 250

# Explicit dtypes for pandas DataFrames to minimise memory footprint.
# int8/int16 for bounded integer columns; float32 for signals (precision sufficient).
PLAYER_DF_DTYPES = {
    "age":                   "int8",
    "contract_months":       "int16",
    "current_value_eur_m":   "float32",
    "rating":                "float32",
    "trajectory_slope":      "float32",
    "urgency_score":         "float32",
    "injury_risk":           "float32",
    "sentiment_hype":        "float32",
    "contentment_score":     "float32",
    "fair_value_eur_m":      "float32",
    "transfer_probability":  "float32",
    "composite_score":       "float32",
}

# ============================================================================ #
# League difficulty baseline                                                     #
# ============================================================================ #

# Premier League = 1.0 reference. All other leagues relative.
LEAGUE_DIFFICULTY_BASELINE: str = "Premier League"
LEAGUE_DIFFICULTY_BASELINE_FACTOR: float = 1.0

# ============================================================================ #
# Backtest / validation                                                          #
# ============================================================================ #

# Historical transfer evaluation window (transfers considered for backtest)
BACKTEST_WINDOW_START: str = "2022-01-01"
BACKTEST_WINDOW_END:   str = "2025-01-01"

# Destination accuracy round scoring
DESTINATION_ACCURACY_POINTS = {1: 3, 2: 2, 3: 1}   # rank → points
DESTINATION_MAX_SCORE: int = 15                       # 5 transfers × 3 pts max

# ============================================================================ #
# FFP headroom thresholds                                                        #
# ============================================================================ #

FFP_MIN_HEADROOM_TO_BID_EUR_M: float = 20.0   # Clubs below this can't realistically bid
FFP_STRESSED_THRESHOLD_EUR_M:  float = 30.0   # Below this = likely seller

# ============================================================================ #
# Sentiment scoring keyword lists                                                 #
# ============================================================================ #

SENTIMENT_POSITIVE_KEYWORDS = [
    "agrees personal terms",
    "medical scheduled",
    "deal agreed",
    "signs contract",
    "completed signing",
    "done deal",
    "officially unveiled",
    "transfer confirmed",
    "joins",
    "secures signing",
    "wants to sign",
    "in advanced talks",
    "bid accepted",
    "passes medical",
]

SENTIMENT_NEGATIVE_KEYWORDS = [
    "breaks down",
    "denies interest",
    "rules out transfer",
    "not for sale",
    "rejected bid",
    "pulls out",
    "no deal",
    "contract extension",
    "staying",
    "committed to",
    "turns down",
    "no intention of leaving",
    "new deal signed",
]
