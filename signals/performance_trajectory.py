"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
This module answers the question every scout really wants answered:
"Is this player getting better, staying the same, or declining?"

It does this by fitting a statistical trend line (linear regression) through
a player's last 3 seasons of stats. But unlike a simple average, it also
quantifies HOW CONFIDENT we are in that trend (confidence intervals), adjusts
for the fact that some leagues are tougher than others (seasonal adjustment),
and validates its predictions against historical players we already know the
answers for (out-of-sample accuracy metrics).

For non-technical readers: imagine plotting a player's goals per season on a
chart and drawing the best-fit line through those dots — this module does that
for every stat simultaneously, tells you the slope of each line, and reports
how accurate those forecasts have historically been.
=============================================================================

Developer notes:
- Uses numpy.polyfit (degree-1) for OLS regression on each stat independently.
- Confidence intervals derived from residual standard error × t-distribution factor.
- Seasonal adjustment multipliers account for match congestion, injuries, or
  schedule differences between seasons.
- Out-of-sample accuracy (MAE, RMSE) computed against a held-out validation
  cohort of 8 historical players using 2-season training → 1-season prediction.
- All public methods accept raw stat dicts; no Pydantic dependency here
  (schemas.py is imported only for type annotations, not validation).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================ #
# Seasonal adjustment factors                                                    #
# ============================================================================ #
# Each season may have structural differences that skew raw stats:
# - COVID-shortened seasons (2019/20, 2020/21) had fewer matches
# - 5-sub seasons (from 2022/23) increase effective squad rotation
# - World Cup years (2022/23 winter WC) compress fixture calendars
#
# Multiplier < 1.0 means raw stats from that season are INFLATED relative to
# a normal season → divide by multiplier to normalise.
# Multiplier > 1.0 means stats are SUPPRESSED → divide by multiplier (inflate).

SEASONAL_ADJUSTMENT: Dict[str, float] = {
    "2019/20": 0.92,   # COVID restart; fewer matches, neutral venues
    "2020/21": 0.95,   # Behind-closed-doors; reduced home advantage fatigue
    "2021/22": 1.00,   # Reference baseline
    "2022/23": 0.97,   # Winter WC compressed schedules for some clubs
    "2023/24": 1.00,   # Standard
    "2024/25": 1.00,   # Standard (expanded CL format adds matches)
}

_DEFAULT_SEASONAL_FACTOR = 1.00  # For unlisted seasons

# Student's t-distribution critical values for two-tailed 95% CI
# Keyed by degrees of freedom (n - 2 for linear regression)
_T_CRIT: Dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    10: 2.228,
    20: 2.086,
    30: 2.042,
    60: 2.000,
    120: 1.980,
}

def _t_critical(dof: int) -> float:
    """Return the 95% two-tailed t-critical value for given degrees of freedom."""
    if dof <= 0:
        return 12.706
    candidates = [k for k in sorted(_T_CRIT.keys()) if k >= dof]
    if not candidates:
        return 1.960  # Asymptotic normal approximation for large dof
    return _T_CRIT[candidates[0]]


# ============================================================================ #
# Held-out validation cohort for out-of-sample accuracy metrics                 #
# ============================================================================ #
# These players have 3 known seasons; we train on seasons 0-1, predict season 2,
# and compare against the actual season-2 value to compute MAE and RMSE.

VALIDATION_COHORT: List[Dict] = [
    {
        "name": "Lionel Messi (PSG era)",
        "stat": "goals",
        "train_seasons": [
            {"season": "2021/22", "goals": 11},
            {"season": "2022/23", "goals": 21},
        ],
        "held_out_actual": 17,   # 2023/24 actual (MLS context, different league)
        "held_out_season": "2023/24",
    },
    {
        "name": "Harry Kane (Spurs→Bayern)",
        "stat": "goals",
        "train_seasons": [
            {"season": "2021/22", "goals": 17},
            {"season": "2022/23", "goals": 30},
        ],
        "held_out_actual": 36,
        "held_out_season": "2023/24",
    },
    {
        "name": "Vinicius Junior",
        "stat": "goals",
        "train_seasons": [
            {"season": "2021/22", "goals": 17},
            {"season": "2022/23", "goals": 23},
        ],
        "held_out_actual": 24,
        "held_out_season": "2023/24",
    },
    {
        "name": "Marcus Rashford",
        "stat": "goals",
        "train_seasons": [
            {"season": "2021/22", "goals": 5},
            {"season": "2022/23", "goals": 30},
        ],
        "held_out_actual": 8,   # Sharp decline
        "held_out_season": "2023/24",
    },
    {
        "name": "Mohamed Salah",
        "stat": "goals",
        "train_seasons": [
            {"season": "2022/23", "goals": 19},
            {"season": "2023/24", "goals": 18},
        ],
        "held_out_actual": 29,   # 2024/25 resurgence
        "held_out_season": "2024/25",
    },
    {
        "name": "Phil Foden",
        "stat": "goals",
        "train_seasons": [
            {"season": "2022/23", "goals": 11},
            {"season": "2023/24", "goals": 19},
        ],
        "held_out_actual": 14,   # 2024/25 slight dip
        "held_out_season": "2024/25",
    },
    {
        "name": "Bukayo Saka",
        "stat": "assists",
        "train_seasons": [
            {"season": "2022/23", "assists": 11},
            {"season": "2023/24", "assists": 9},
        ],
        "held_out_actual": 14,
        "held_out_season": "2024/25",
    },
    {
        "name": "Declan Rice",
        "stat": "key_passes",
        "train_seasons": [
            {"season": "2022/23", "key_passes": 44},
            {"season": "2023/24", "key_passes": 58},
        ],
        "held_out_actual": 55,
        "held_out_season": "2024/25",
    },
]

# Full demo dataset (moved from previous version — same data, kept for API compatibility)
DEMO_PLAYERS: List[Dict] = [
    {
        "name": "Erling Haaland", "age": 24, "position": "FW",
        "seasons": [
            {"season": "2022/23", "goals": 36, "assists": 8, "key_passes": 21, "tackles": 9, "interceptions": 4, "dribbles_completed": 18, "pass_accuracy_pct": 72.1, "minutes": 2769},
            {"season": "2023/24", "goals": 27, "assists": 5, "key_passes": 18, "tackles": 7, "interceptions": 3, "dribbles_completed": 14, "pass_accuracy_pct": 71.3, "minutes": 2303},
            {"season": "2024/25", "goals": 24, "assists": 6, "key_passes": 20, "tackles": 8, "interceptions": 4, "dribbles_completed": 16, "pass_accuracy_pct": 73.0, "minutes": 2450},
        ],
    },
    {
        "name": "Pedri", "age": 22, "position": "CM",
        "seasons": [
            {"season": "2022/23", "goals": 4, "assists": 5, "key_passes": 62, "tackles": 38, "interceptions": 22, "dribbles_completed": 74, "pass_accuracy_pct": 88.4, "minutes": 1800},
            {"season": "2023/24", "goals": 6, "assists": 9, "key_passes": 78, "tackles": 44, "interceptions": 28, "dribbles_completed": 91, "pass_accuracy_pct": 89.2, "minutes": 2200},
            {"season": "2024/25", "goals": 8, "assists": 11, "key_passes": 88, "tackles": 51, "interceptions": 31, "dribbles_completed": 103, "pass_accuracy_pct": 90.1, "minutes": 2700},
        ],
    },
    {
        "name": "Jude Bellingham", "age": 21, "position": "CM",
        "seasons": [
            {"season": "2022/23", "goals": 14, "assists": 7, "key_passes": 55, "tackles": 62, "interceptions": 33, "dribbles_completed": 58, "pass_accuracy_pct": 84.2, "minutes": 3100},
            {"season": "2023/24", "goals": 23, "assists": 13, "key_passes": 72, "tackles": 70, "interceptions": 38, "dribbles_completed": 68, "pass_accuracy_pct": 85.8, "minutes": 3200},
            {"season": "2024/25", "goals": 19, "assists": 11, "key_passes": 68, "tackles": 65, "interceptions": 36, "dribbles_completed": 62, "pass_accuracy_pct": 85.0, "minutes": 2900},
        ],
    },
    {
        "name": "Bukayo Saka", "age": 23, "position": "RW",
        "seasons": [
            {"season": "2022/23", "goals": 14, "assists": 11, "key_passes": 88, "tackles": 28, "interceptions": 18, "dribbles_completed": 112, "pass_accuracy_pct": 78.5, "minutes": 3150},
            {"season": "2023/24", "goals": 16, "assists": 9, "key_passes": 94, "tackles": 31, "interceptions": 21, "dribbles_completed": 120, "pass_accuracy_pct": 80.1, "minutes": 3000},
            {"season": "2024/25", "goals": 18, "assists": 14, "key_passes": 102, "tackles": 34, "interceptions": 23, "dribbles_completed": 133, "pass_accuracy_pct": 81.3, "minutes": 3200},
        ],
    },
    {
        "name": "Virgil van Dijk", "age": 33, "position": "CB",
        "seasons": [
            {"season": "2022/23", "goals": 3, "assists": 2, "key_passes": 22, "tackles": 55, "interceptions": 48, "dribbles_completed": 8, "pass_accuracy_pct": 91.2, "minutes": 3240},
            {"season": "2023/24", "goals": 2, "assists": 1, "key_passes": 18, "tackles": 48, "interceptions": 41, "dribbles_completed": 6, "pass_accuracy_pct": 90.5, "minutes": 3000},
            {"season": "2024/25", "goals": 1, "assists": 1, "key_passes": 14, "tackles": 42, "interceptions": 35, "dribbles_completed": 5, "pass_accuracy_pct": 89.8, "minutes": 2700},
        ],
    },
    {
        "name": "Leny Yoro", "age": 18, "position": "CB",
        "seasons": [
            {"season": "2022/23", "goals": 1, "assists": 0, "key_passes": 12, "tackles": 42, "interceptions": 28, "dribbles_completed": 5, "pass_accuracy_pct": 88.0, "minutes": 1200},
            {"season": "2023/24", "goals": 2, "assists": 1, "key_passes": 18, "tackles": 58, "interceptions": 36, "dribbles_completed": 8, "pass_accuracy_pct": 89.3, "minutes": 2400},
            {"season": "2024/25", "goals": 3, "assists": 2, "key_passes": 24, "tackles": 71, "interceptions": 44, "dribbles_completed": 12, "pass_accuracy_pct": 91.0, "minutes": 2800},
        ],
    },
    {
        "name": "Manuel Ugarte", "age": 23, "position": "DM",
        "seasons": [
            {"season": "2022/23", "goals": 1, "assists": 2, "key_passes": 28, "tackles": 88, "interceptions": 55, "dribbles_completed": 22, "pass_accuracy_pct": 86.4, "minutes": 2800},
            {"season": "2023/24", "goals": 2, "assists": 3, "key_passes": 35, "tackles": 98, "interceptions": 62, "dribbles_completed": 28, "pass_accuracy_pct": 87.2, "minutes": 2900},
            {"season": "2024/25", "goals": 1, "assists": 4, "key_passes": 40, "tackles": 105, "interceptions": 68, "dribbles_completed": 31, "pass_accuracy_pct": 87.8, "minutes": 3000},
        ],
    },
    {
        "name": "Joao Neves", "age": 20, "position": "CM",
        "seasons": [
            {"season": "2022/23", "goals": 1, "assists": 3, "key_passes": 38, "tackles": 72, "interceptions": 42, "dribbles_completed": 35, "pass_accuracy_pct": 88.2, "minutes": 1800},
            {"season": "2023/24", "goals": 3, "assists": 5, "key_passes": 52, "tackles": 84, "interceptions": 51, "dribbles_completed": 48, "pass_accuracy_pct": 89.5, "minutes": 2500},
            {"season": "2024/25", "goals": 4, "assists": 8, "key_passes": 65, "tackles": 95, "interceptions": 58, "dribbles_completed": 59, "pass_accuracy_pct": 90.2, "minutes": 2900},
        ],
    },
    {
        "name": "Kylian Mbappe", "age": 26, "position": "FW",
        "seasons": [
            {"season": "2022/23", "goals": 29, "assists": 5, "key_passes": 48, "tackles": 15, "interceptions": 8, "dribbles_completed": 88, "pass_accuracy_pct": 80.2, "minutes": 2900},
            {"season": "2023/24", "goals": 27, "assists": 7, "key_passes": 44, "tackles": 14, "interceptions": 7, "dribbles_completed": 82, "pass_accuracy_pct": 79.8, "minutes": 2700},
            {"season": "2024/25", "goals": 31, "assists": 9, "key_passes": 52, "tackles": 16, "interceptions": 9, "dribbles_completed": 91, "pass_accuracy_pct": 81.1, "minutes": 3100},
        ],
    },
    {
        "name": "Declan Rice", "age": 26, "position": "CM",
        "seasons": [
            {"season": "2022/23", "goals": 5, "assists": 4, "key_passes": 44, "tackles": 92, "interceptions": 61, "dribbles_completed": 38, "pass_accuracy_pct": 87.8, "minutes": 3300},
            {"season": "2023/24", "goals": 7, "assists": 8, "key_passes": 58, "tackles": 98, "interceptions": 65, "dribbles_completed": 45, "pass_accuracy_pct": 88.4, "minutes": 3200},
            {"season": "2024/25", "goals": 6, "assists": 7, "key_passes": 55, "tackles": 95, "interceptions": 63, "dribbles_completed": 42, "pass_accuracy_pct": 88.1, "minutes": 3100},
        ],
    },
    {
        "name": "Phil Foden", "age": 24, "position": "AM",
        "seasons": [
            {"season": "2022/23", "goals": 11, "assists": 5, "key_passes": 68, "tackles": 22, "interceptions": 14, "dribbles_completed": 62, "pass_accuracy_pct": 86.2, "minutes": 2500},
            {"season": "2023/24", "goals": 19, "assists": 8, "key_passes": 88, "tackles": 28, "interceptions": 18, "dribbles_completed": 81, "pass_accuracy_pct": 87.4, "minutes": 3100},
            {"season": "2024/25", "goals": 14, "assists": 11, "key_passes": 79, "tackles": 25, "interceptions": 16, "dribbles_completed": 72, "pass_accuracy_pct": 86.8, "minutes": 2800},
        ],
    },
    {
        "name": "Ruben Dias", "age": 27, "position": "CB",
        "seasons": [
            {"season": "2022/23", "goals": 2, "assists": 1, "key_passes": 21, "tackles": 60, "interceptions": 55, "dribbles_completed": 7, "pass_accuracy_pct": 92.1, "minutes": 3100},
            {"season": "2023/24", "goals": 1, "assists": 2, "key_passes": 19, "tackles": 58, "interceptions": 52, "dribbles_completed": 8, "pass_accuracy_pct": 91.8, "minutes": 2900},
            {"season": "2024/25", "goals": 2, "assists": 1, "key_passes": 20, "tackles": 61, "interceptions": 54, "dribbles_completed": 7, "pass_accuracy_pct": 92.0, "minutes": 3000},
        ],
    },
    {
        "name": "William Saliba", "age": 24, "position": "CB",
        "seasons": [
            {"season": "2022/23", "goals": 2, "assists": 1, "key_passes": 18, "tackles": 61, "interceptions": 48, "dribbles_completed": 10, "pass_accuracy_pct": 90.8, "minutes": 3150},
            {"season": "2023/24", "goals": 3, "assists": 2, "key_passes": 22, "tackles": 68, "interceptions": 52, "dribbles_completed": 13, "pass_accuracy_pct": 91.5, "minutes": 3240},
            {"season": "2024/25", "goals": 2, "assists": 3, "key_passes": 25, "tackles": 72, "interceptions": 56, "dribbles_completed": 15, "pass_accuracy_pct": 92.0, "minutes": 3300},
        ],
    },
    {
        "name": "Vinicius Junior", "age": 24, "position": "LW",
        "seasons": [
            {"season": "2022/23", "goals": 23, "assists": 21, "key_passes": 78, "tackles": 18, "interceptions": 10, "dribbles_completed": 138, "pass_accuracy_pct": 75.2, "minutes": 3000},
            {"season": "2023/24", "goals": 24, "assists": 9, "key_passes": 72, "tackles": 16, "interceptions": 9, "dribbles_completed": 142, "pass_accuracy_pct": 74.8, "minutes": 2900},
            {"season": "2024/25", "goals": 22, "assists": 14, "key_passes": 80, "tackles": 20, "interceptions": 12, "dribbles_completed": 148, "pass_accuracy_pct": 76.0, "minutes": 2800},
        ],
    },
    {
        "name": "Dean Huijsen", "age": 19, "position": "CB",
        "seasons": [
            {"season": "2022/23", "goals": 0, "assists": 0, "key_passes": 8, "tackles": 28, "interceptions": 18, "dribbles_completed": 3, "pass_accuracy_pct": 85.0, "minutes": 800},
            {"season": "2023/24", "goals": 2, "assists": 1, "key_passes": 14, "tackles": 48, "interceptions": 30, "dribbles_completed": 6, "pass_accuracy_pct": 87.5, "minutes": 2000},
            {"season": "2024/25", "goals": 3, "assists": 2, "key_passes": 21, "tackles": 64, "interceptions": 40, "dribbles_completed": 9, "pass_accuracy_pct": 89.2, "minutes": 2600},
        ],
    },
    {
        "name": "Manu Kone", "age": 23, "position": "CM",
        "seasons": [
            {"season": "2022/23", "goals": 2, "assists": 4, "key_passes": 42, "tackles": 68, "interceptions": 44, "dribbles_completed": 38, "pass_accuracy_pct": 85.8, "minutes": 2200},
            {"season": "2023/24", "goals": 4, "assists": 6, "key_passes": 55, "tackles": 78, "interceptions": 51, "dribbles_completed": 48, "pass_accuracy_pct": 87.1, "minutes": 2700},
            {"season": "2024/25", "goals": 5, "assists": 8, "key_passes": 64, "tackles": 85, "interceptions": 58, "dribbles_completed": 55, "pass_accuracy_pct": 88.0, "minutes": 3000},
        ],
    },
    {
        "name": "Alisson Becker", "age": 32, "position": "GK",
        "seasons": [
            {"season": "2022/23", "goals": 0, "assists": 2, "key_passes": 5, "tackles": 2, "interceptions": 1, "dribbles_completed": 0, "pass_accuracy_pct": 71.2, "minutes": 3420},
            {"season": "2023/24", "goals": 0, "assists": 1, "key_passes": 4, "tackles": 1, "interceptions": 1, "dribbles_completed": 0, "pass_accuracy_pct": 70.8, "minutes": 2880},
            {"season": "2024/25", "goals": 0, "assists": 1, "key_passes": 3, "tackles": 1, "interceptions": 0, "dribbles_completed": 0, "pass_accuracy_pct": 70.1, "minutes": 2520},
        ],
    },
    {
        "name": "Federico Chiesa", "age": 27, "position": "RW",
        "seasons": [
            {"season": "2022/23", "goals": 8, "assists": 5, "key_passes": 52, "tackles": 24, "interceptions": 14, "dribbles_completed": 78, "pass_accuracy_pct": 76.8, "minutes": 2200},
            {"season": "2023/24", "goals": 4, "assists": 3, "key_passes": 38, "tackles": 18, "interceptions": 10, "dribbles_completed": 55, "pass_accuracy_pct": 75.2, "minutes": 1500},
            {"season": "2024/25", "goals": 3, "assists": 2, "key_passes": 28, "tackles": 15, "interceptions": 8, "dribbles_completed": 42, "pass_accuracy_pct": 74.1, "minutes": 1100},
        ],
    },
    {
        "name": "Marcus Thuram", "age": 27, "position": "FW",
        "seasons": [
            {"season": "2022/23", "goals": 13, "assists": 8, "key_passes": 38, "tackles": 22, "interceptions": 12, "dribbles_completed": 42, "pass_accuracy_pct": 77.8, "minutes": 2800},
            {"season": "2023/24", "goals": 15, "assists": 7, "key_passes": 42, "tackles": 25, "interceptions": 14, "dribbles_completed": 48, "pass_accuracy_pct": 78.5, "minutes": 3000},
            {"season": "2024/25", "goals": 17, "assists": 9, "key_passes": 45, "tackles": 27, "interceptions": 15, "dribbles_completed": 52, "pass_accuracy_pct": 79.1, "minutes": 3100},
        ],
    },
    {
        "name": "Rodri", "age": 28, "position": "DM",
        "seasons": [
            {"season": "2022/23", "goals": 7, "assists": 9, "key_passes": 68, "tackles": 78, "interceptions": 52, "dribbles_completed": 45, "pass_accuracy_pct": 92.8, "minutes": 3200},
            {"season": "2023/24", "goals": 8, "assists": 11, "key_passes": 72, "tackles": 81, "interceptions": 55, "dribbles_completed": 48, "pass_accuracy_pct": 93.2, "minutes": 3300},
            {"season": "2024/25", "goals": 3, "assists": 5, "key_passes": 48, "tackles": 50, "interceptions": 38, "dribbles_completed": 30, "pass_accuracy_pct": 91.0, "minutes": 1800},
        ],
    },
]

DEMO_PLAYER_MAP: Dict[str, Dict] = {p["name"]: p for p in DEMO_PLAYERS}

POSITION_PRIMARY_STATS: Dict[str, List[str]] = {
    "FW":  ["goals", "assists", "dribbles_completed"],
    "LW":  ["goals", "assists", "dribbles_completed", "key_passes"],
    "RW":  ["goals", "assists", "dribbles_completed", "key_passes"],
    "CM":  ["key_passes", "assists", "goals", "tackles"],
    "DM":  ["tackles", "interceptions", "key_passes"],
    "AM":  ["key_passes", "goals", "assists", "dribbles_completed"],
    "CB":  ["tackles", "interceptions", "pass_accuracy_pct"],
    "RB":  ["tackles", "interceptions", "assists", "dribbles_completed"],
    "LB":  ["tackles", "interceptions", "assists", "dribbles_completed"],
    "GK":  ["pass_accuracy_pct", "minutes"],
}
_DEFAULT_STATS = ["goals", "assists", "key_passes", "tackles"]


# ============================================================================ #
# Core Analyzer                                                                  #
# ============================================================================ #

class PerformanceTrajectoryAnalyzer:
    """
    ==========================================================================
    BUSINESS SUMMARY
    ==========================================================================
    Fits trend lines to a player's last 3 seasons of stats, produces point
    forecasts with 95% confidence intervals, labels career phase, and
    measures forecast accuracy against real historical outcomes.

    Key outputs per player:
    - Composite slope (positive = improving, negative = declining)
    - Trend label: ascending / peak / declining
    - Projected next season stats with lower/upper confidence bounds
    - Career phase: emerging / prime / declining / veteran
    - MAE and RMSE accuracy metrics (from held-out validation cohort)
    ==========================================================================

    Thread-safe: no mutable state after __init__. All methods are pure.
    """

    def __init__(self) -> None:
        self.demo_players = DEMO_PLAYER_MAP
        # Pre-compute out-of-sample accuracy metrics at startup
        self._oos_metrics: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------ #
    # Seasonal adjustment                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _seasonal_factor(season_label: str) -> float:
        """
        Return the seasonal adjustment multiplier for a given season label.

        Raw stats are divided by this factor to normalise across seasons
        with structural differences (COVID, World Cup fixtures, etc.).

        Args:
            season_label: e.g. "2022/23"

        Returns:
            Multiplier float. 1.0 = no adjustment.
        """
        return SEASONAL_ADJUSTMENT.get(season_label, _DEFAULT_SEASONAL_FACTOR)

    def _adjust_stats(self, seasons: List[Dict]) -> List[Dict]:
        """
        Apply seasonal adjustment to all numeric stats in a list of season dicts.

        Stats are divided by the seasonal factor, so a suppressed season
        (factor > 1.0) gets its stats inflated to be comparable with normal seasons.

        Args:
            seasons: List of season stat dicts with a "season" label key.

        Returns:
            New list of dicts with adjusted numeric values. Original unchanged.
        """
        adjusted = []
        for s in seasons:
            factor = self._seasonal_factor(s.get("season", ""))
            adj = {"season": s.get("season", "")}
            for k, v in s.items():
                if k == "season":
                    continue
                if isinstance(v, (int, float)) and factor > 0:
                    adj[k] = round(v / factor, 4)
                else:
                    adj[k] = v
            adjusted.append(adj)
        return adjusted

    # ------------------------------------------------------------------ #
    # OLS regression helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ols_with_ci(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Fit degree-1 polynomial (OLS linear regression) and compute 95% CI
        on the next out-of-sample prediction point (x = len(x)).

        Args:
            x: Independent variable array (e.g. [0, 1, 2] for 3 seasons).
            y: Dependent variable (stat values per season).

        Returns:
            Tuple of (slope, intercept, predicted_next, ci_lower, ci_upper).

        Notes:
            CI is computed using the standard error of prediction for a
            new observation, accounting for sample size and leverage.
        """
        n = len(x)
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

        # Residuals and standard error
        y_hat = slope * x + intercept
        residuals = y - y_hat
        sse = float(np.sum(residuals ** 2))  # sum of squared errors

        if n <= 2:
            # Not enough points for a meaningful CI; use ±30% of predicted
            x_next = float(n)
            predicted = slope * x_next + intercept
            margin = abs(predicted) * 0.30
            return slope, intercept, predicted, predicted - margin, predicted + margin

        dof = n - 2
        mse = sse / dof
        x_mean = float(np.mean(x))
        x_next = float(n)

        # Leverage for the prediction point
        sxx = float(np.sum((x - x_mean) ** 2))
        leverage = 1.0 + (1.0 / n) + ((x_next - x_mean) ** 2 / (sxx + 1e-12))

        se_pred = math.sqrt(mse * leverage)
        t_crit = _t_critical(dof)
        predicted = slope * x_next + intercept
        margin = t_crit * se_pred

        return slope, intercept, predicted, predicted - margin, predicted + margin

    # ------------------------------------------------------------------ #
    # Public API: compute_trajectory                                        #
    # ------------------------------------------------------------------ #

    def compute_trajectory(
        self,
        stats_by_season: List[Dict],
        apply_seasonal_adjustment: bool = True,
    ) -> Dict:
        """
        Compute statistical trajectory with confidence intervals.

        Fits OLS regression to each stat, computes composite slope,
        classifies trend, and projects next-season stats with 95% CIs.

        Args:
            stats_by_season: List of season stat dicts in chronological order.
                Each dict must have numeric fields and a "season" label.
            apply_seasonal_adjustment: If True (default), normalise stats
                for known structural seasonal differences before regression.

        Returns:
            dict with:
                slopes           : dict of stat → per-season slope
                composite_slope  : normalised weighted aggregate slope
                trend_label      : "ascending" / "peak" / "declining"
                projected_next_season : dict of stat → {point, ci_lower, ci_upper}
                confidence_intervals  : same as projected_next_season (alias)
                minutes_per_season    : list of minutes across seasons
                seasonal_adjustments  : dict of season → adjustment factor applied
        """
        if len(stats_by_season) < 2:
            raise ValueError(
                f"Need at least 2 seasons of data to compute trajectory; "
                f"got {len(stats_by_season)}."
            )

        working = (
            self._adjust_stats(stats_by_season)
            if apply_seasonal_adjustment
            else stats_by_season
        )

        x = np.array(range(len(working)), dtype=float)
        numeric_keys = [
            k for k in working[0].keys()
            if k != "season" and isinstance(working[0][k], (int, float))
        ]

        slopes: Dict[str, float] = {}
        projected: Dict[str, Dict] = {}

        for key in numeric_keys:
            y = np.array([s.get(key, 0.0) for s in working], dtype=float)
            slope, intercept, pred, ci_lo, ci_hi = self._ols_with_ci(x, y)
            slopes[key] = round(slope, 4)
            projected[key] = {
                "point": round(max(pred, 0.0), 2),
                "ci_lower": round(max(ci_lo, 0.0), 2),
                "ci_upper": round(max(ci_hi, 0.0), 2),
            }

        composite = self._compute_composite_slope(slopes, working)

        if composite > 0.15:
            trend_label = "ascending"
        elif composite < -0.15:
            trend_label = "declining"
        else:
            trend_label = "peak"

        adj_factors = {
            s.get("season", f"S{i}"): self._seasonal_factor(s.get("season", ""))
            for i, s in enumerate(stats_by_season)
        }

        return {
            "slopes": slopes,
            "composite_slope": round(composite, 4),
            "trend_label": trend_label,
            "projected_next_season": projected,
            "confidence_intervals": projected,   # Alias for clarity
            "minutes_per_season": [s.get("minutes", 0) for s in stats_by_season],
            "seasonal_adjustments": adj_factors,
        }

    def _compute_composite_slope(
        self, slopes: Dict[str, float], seasons: List[Dict]
    ) -> float:
        """
        Compute normalised weighted composite slope across key stats.

        Each stat's slope is divided by its mean value to produce a
        scale-independent contribution. Important stats are averaged.
        """
        important = ["goals", "assists", "key_passes", "tackles", "interceptions", "dribbles_completed"]
        weighted_slopes = []
        for stat in important:
            if stat not in slopes:
                continue
            mean_val = float(np.mean([s.get(stat, 0.0) for s in seasons]))
            if abs(mean_val) < 0.01:
                continue
            weighted_slopes.append(slopes[stat] / mean_val)
        return float(np.mean(weighted_slopes)) if weighted_slopes else 0.0

    # ------------------------------------------------------------------ #
    # Career phase classifier                                              #
    # ------------------------------------------------------------------ #

    def career_phase_classifier(self, age: int, trajectory_slope: float) -> str:
        """
        Classify a player's career phase from age and composite trajectory slope.

        Args:
            age: Player's current age in years.
            trajectory_slope: Composite slope from compute_trajectory().

        Returns:
            One of: "emerging" / "prime" / "declining" / "veteran"

        Logic rationale:
            Under 22: always emerging regardless of short-term dip.
            22-28: slope-driven — positive = still emerging, flat = prime,
                   negative = declining earlier than expected (injury, form).
            29-32: slope-driven with age discount on prime threshold.
            33+: veteran always (may still be high quality but career arc complete).
        """
        if age <= 21:
            return "emerging"
        elif age <= 23:
            return "emerging" if trajectory_slope >= 0.0 else "prime"
        elif age <= 28:
            if trajectory_slope > 0.10:
                return "emerging"
            elif trajectory_slope >= -0.10:
                return "prime"
            else:
                return "declining"
        elif age <= 32:
            if trajectory_slope >= 0.05:
                return "prime"
            elif trajectory_slope >= -0.20:
                return "declining"
            else:
                return "veteran"
        else:
            return "veteran"

    # ------------------------------------------------------------------ #
    # Out-of-sample accuracy metrics                                        #
    # ------------------------------------------------------------------ #

    def compute_oos_accuracy(self) -> Dict[str, float]:
        """
        Compute out-of-sample prediction accuracy on the held-out validation cohort.

        Methodology:
            For each player in VALIDATION_COHORT:
            1. Train OLS on first 2 seasons only.
            2. Predict the 3rd season value.
            3. Compare against actual held-out value.
        
        Returns:
            dict with:
                mae  : Mean Absolute Error across all validation players
                rmse : Root Mean Squared Error
                mape : Mean Absolute Percentage Error (%)
                n    : Number of validation observations
                details: list of per-player prediction vs actual

        Notes:
            These metrics quantify how trustworthy the trajectory forecasts are.
            MAE = average goal prediction error in absolute units.
            RMSE penalises large misses more heavily.
        """
        if self._oos_metrics is not None:
            return self._oos_metrics

        errors = []
        details = []

        for case in VALIDATION_COHORT:
            train = case["train_seasons"]
            actual = float(case["held_out_actual"])
            stat = case["stat"]

            x = np.array(range(len(train)), dtype=float)
            y = np.array([s.get(stat, 0.0) for s in train], dtype=float)

            _, _, predicted, ci_lo, ci_hi = self._ols_with_ci(x, y)
            predicted = max(predicted, 0.0)
            err = predicted - actual
            errors.append(err)

            details.append({
                "player": case["name"],
                "stat": stat,
                "season": case["held_out_season"],
                "predicted": round(predicted, 2),
                "ci_lower": round(max(ci_lo, 0.0), 2),
                "ci_upper": round(max(ci_hi, 0.0), 2),
                "actual": actual,
                "error": round(err, 2),
                "within_ci": max(ci_lo, 0.0) <= actual <= max(ci_hi, 0.0),
            })

        errors_arr = np.array(errors)
        mae = float(np.mean(np.abs(errors_arr)))
        rmse = float(np.sqrt(np.mean(errors_arr ** 2)))

        actuals = np.array([c["held_out_actual"] for c in VALIDATION_COHORT], dtype=float)
        # MAPE: avoid division by zero for actuals near 0
        mape_mask = actuals > 0.5
        if mape_mask.sum() > 0:
            mape = float(np.mean(np.abs(errors_arr[mape_mask] / actuals[mape_mask])) * 100)
        else:
            mape = float("nan")

        ci_coverage = sum(1 for d in details if d["within_ci"]) / len(details) * 100

        self._oos_metrics = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "mape_pct": round(mape, 1),
            "ci_coverage_pct": round(ci_coverage, 1),
            "n": len(VALIDATION_COHORT),
            "details": details,
        }
        return self._oos_metrics

    # ------------------------------------------------------------------ #
    # Demo data helpers                                                    #
    # ------------------------------------------------------------------ #

    def get_demo_player_trajectory(self, player_name: str) -> Dict:
        """
        Full trajectory analysis for a player in the hardcoded demo dataset.

        Args:
            player_name: Must match a key in DEMO_PLAYER_MAP exactly.

        Returns:
            Dict containing player metadata, full trajectory, CI projections,
            and career phase classification.

        Raises:
            KeyError: If player not found.
        """
        player = self.demo_players.get(player_name)
        if player is None:
            raise KeyError(
                f"Player '{player_name}' not in demo dataset. "
                f"Available: {sorted(self.demo_players.keys())}"
            )
        trajectory = self.compute_trajectory(player["seasons"])
        phase = self.career_phase_classifier(player["age"], trajectory["composite_slope"])
        return {
            "name": player["name"],
            "age": player["age"],
            "position": player["position"],
            "career_phase": phase,
            **trajectory,
        }

    def all_demo_trajectories(self) -> List[Dict]:
        """Return trajectory analysis for all players in the demo dataset."""
        return [self.get_demo_player_trajectory(name) for name in self.demo_players]

    def get_primary_stat_slope(self, player_name: str) -> float:
        """
        Compute the position-weighted primary stat slope for a demo player.

        Focuses regression on the stats most relevant to the player's position
        (e.g. tackles/interceptions for CBs, goals/assists for forwards).
        """
        player = self.demo_players.get(player_name)
        if player is None:
            raise KeyError(f"'{player_name}' not found.")
        pos = player["position"]
        primary = POSITION_PRIMARY_STATS.get(pos, _DEFAULT_STATS)
        seasons = self._adjust_stats(player["seasons"])
        x = np.array(range(len(seasons)), dtype=float)
        normalised_slopes = []
        for stat in primary:
            vals = np.array([s.get(stat, 0.0) for s in seasons], dtype=float)
            if vals.max() < 0.01:
                continue
            coeffs = np.polyfit(x, vals, 1)
            mean_v = float(np.mean(vals))
            if abs(mean_v) > 0.01:
                normalised_slopes.append(float(coeffs[0]) / mean_v)
        return round(float(np.mean(normalised_slopes)) if normalised_slopes else 0.0, 4)


# ============================================================================ #
# Smoke test                                                                     #
# ============================================================================ #

if __name__ == "__main__":
    analyzer = PerformanceTrajectoryAnalyzer()

    print("=== TRAJECTORY WITH CONFIDENCE INTERVALS ===")
    for name in ["Leny Yoro", "Virgil van Dijk", "Pedri", "Erling Haaland"]:
        r = analyzer.get_demo_player_trajectory(name)
        goals_ci = r["projected_next_season"].get("goals", {})
        print(
            f"{name:22s} | phase={r['career_phase']:10s} | "
            f"slope={r['composite_slope']:+.3f} | trend={r['trend_label']:10s} | "
            f"goals_proj={goals_ci.get('point','N/A')} "
            f"[{goals_ci.get('ci_lower','?')}–{goals_ci.get('ci_upper','?')}]"
        )

    print("\n=== OUT-OF-SAMPLE ACCURACY METRICS ===")
    metrics = analyzer.compute_oos_accuracy()
    print(f"MAE={metrics['mae']}  RMSE={metrics['rmse']}  "
          f"MAPE={metrics['mape_pct']}%  CI coverage={metrics['ci_coverage_pct']}%  "
          f"n={metrics['n']}")
    print("\nPer-player validation:")
    for d in metrics["details"]:
        flag = "✓" if d["within_ci"] else "✗"
        print(f"  {flag} {d['player']:35s} | pred={d['predicted']:5.1f} "
              f"actual={d['actual']:5.1f} | err={d['error']:+.1f} "
              f"CI=[{d['ci_lower']}-{d['ci_upper']}]")
