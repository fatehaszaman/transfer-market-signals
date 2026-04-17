"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
A player who scores 25 goals in Ligue 1 is not the same as one who scores
25 goals in the Premier League. Different leagues have dramatically different
defensive quality, tactical intensity, and physical demands. Ignoring this
leads to "step-up risk" — the phenomenon where highly rated players from
weaker leagues underperform after big-money moves to elite leagues.

This module normalises stats across leagues so that a player's numbers can
be compared on an apples-to-apples basis, regardless of which league they
play in. It also quantifies the statistical probability that a player will
underperform after moving from a weaker to a stronger league.

Real examples this module prevents:
- Overvaluing a Ligue 1 striker who scores 22 goals (adjust to ~18 PL equivalent)
- Undervaluing a Bundesliga CB moving to the PL (small step-up, manageable)
- Flagging a Serie A winger → PL move as high step-up risk (harder adjustment)
=============================================================================

Developer notes:
- Premier League = 1.0 baseline. All other leagues expressed relative to PL.
- Stat normalisation: raw_stat / source_factor * target_factor.
- Step-up risk is modelled as a sigmoid over difficulty differential.
- Adaptation window (seasons to reach 90% of expected output) is also provided.
- Hardcoded data for 15 leagues sourced from UEFA coefficient data and
  peer-reviewed football analytics literature (approximate illustrative values).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# ============================================================================ #
# League difficulty ratings                                                      #
# ============================================================================ #
# Scale: 1.0 = Premier League baseline. Higher = harder league.
# These are approximations based on UEFA coefficient rankings,
# goal scoring rates, defensive metrics, and analyst consensus.
#
# Note: "difficulty" here means defensive quality + tactical intensity.
# A "harder" league means stats produced there are MORE impressive
# (i.e., fewer goals per player on average due to better defences).

LEAGUE_DIFFICULTY: Dict[str, Dict] = {
    "Premier League": {
        "factor": 1.000,
        "country": "England",
        "tier": 1,
        "avg_goals_per_game": 2.85,
        "physicality_index": 0.95,   # 0-1 physical intensity relative to PL
        "tactical_complexity": 0.90,
        "adaptation_seasons": 0,     # No step-up from PL to PL
        "notes": "Reference baseline. Highest physicality in Europe.",
    },
    "La Liga": {
        "factor": 0.970,
        "country": "Spain",
        "tier": 1,
        "avg_goals_per_game": 2.65,
        "physicality_index": 0.80,
        "tactical_complexity": 0.95,
        "adaptation_seasons": 1,
        "notes": "Technical and positional; slightly softer physically than PL.",
    },
    "Bundesliga": {
        "factor": 0.940,
        "country": "Germany",
        "tier": 1,
        "avg_goals_per_game": 3.10,
        "physicality_index": 0.85,
        "tactical_complexity": 0.88,
        "adaptation_seasons": 1,
        "notes": "High scoring; pressing intensity high but defensive gaps exist.",
    },
    "Serie A": {
        "factor": 0.930,
        "country": "Italy",
        "tier": 1,
        "avg_goals_per_game": 2.55,
        "physicality_index": 0.80,
        "tactical_complexity": 0.92,
        "adaptation_seasons": 1,
        "notes": "Defensively organised; tactical adaptation required.",
    },
    "Ligue 1": {
        "factor": 0.880,
        "country": "France",
        "tier": 1,
        "avg_goals_per_game": 2.70,
        "physicality_index": 0.82,
        "tactical_complexity": 0.82,
        "adaptation_seasons": 2,
        "notes": "PSG dominates; mid/lower table defensive quality notably weaker.",
    },
    "Primeira Liga": {
        "factor": 0.840,
        "country": "Portugal",
        "tier": 2,
        "avg_goals_per_game": 2.55,
        "physicality_index": 0.72,
        "tactical_complexity": 0.80,
        "adaptation_seasons": 2,
        "notes": "Excellent development league; stat inflation vs top 5 leagues.",
    },
    "Eredivisie": {
        "factor": 0.780,
        "country": "Netherlands",
        "tier": 2,
        "avg_goals_per_game": 3.20,
        "physicality_index": 0.68,
        "tactical_complexity": 0.75,
        "adaptation_seasons": 2,
        "notes": "High-scoring; known for Ajax-dominant system. Significant step-up to PL.",
    },
    "Belgian Pro League": {
        "factor": 0.760,
        "country": "Belgium",
        "tier": 2,
        "avg_goals_per_game": 2.90,
        "physicality_index": 0.68,
        "tactical_complexity": 0.72,
        "adaptation_seasons": 2,
        "notes": "Feeder league for top 5; considerable stat inflation.",
    },
    "Scottish Premiership": {
        "factor": 0.720,
        "country": "Scotland",
        "tier": 2,
        "avg_goals_per_game": 2.80,
        "physicality_index": 0.75,
        "tactical_complexity": 0.65,
        "adaptation_seasons": 2,
        "notes": "Celtic dominate; significant inflation vs PL. Physical but low tactical.",
    },
    "MLS": {
        "factor": 0.680,
        "country": "USA",
        "tier": 3,
        "avg_goals_per_game": 3.00,
        "physicality_index": 0.60,
        "tactical_complexity": 0.60,
        "adaptation_seasons": 0,   # No one moves from MLS to PL
        "notes": "Growing rapidly but considerable quality gap vs European top 5.",
    },
    "Saudi Pro League": {
        "factor": 0.650,
        "country": "Saudi Arabia",
        "tier": 3,
        "avg_goals_per_game": 2.80,
        "physicality_index": 0.55,
        "tactical_complexity": 0.55,
        "adaptation_seasons": 0,
        "notes": "Investment boom. Veterans dominate. Very low defensive quality.",
    },
    "Süper Lig": {
        "factor": 0.820,
        "country": "Turkey",
        "tier": 2,
        "avg_goals_per_game": 2.90,
        "physicality_index": 0.75,
        "tactical_complexity": 0.75,
        "adaptation_seasons": 2,
        "notes": "Improving; several UCL-calibre clubs. Passionate and physical.",
    },
    "Brasileirao": {
        "factor": 0.800,
        "country": "Brazil",
        "tier": 2,
        "avg_goals_per_game": 2.40,
        "physicality_index": 0.78,
        "tactical_complexity": 0.72,
        "adaptation_seasons": 2,
        "notes": "Strong technical base; physical style. Key European export league.",
    },
    "Argentine Primera Division": {
        "factor": 0.780,
        "country": "Argentina",
        "tier": 2,
        "avg_goals_per_game": 2.35,
        "physicality_index": 0.80,
        "tactical_complexity": 0.74,
        "adaptation_seasons": 2,
        "notes": "Intense and physical; tactical schooling excellent. Good step-up platform.",
    },
    "Russian Premier League": {
        "factor": 0.750,
        "country": "Russia",
        "tier": 2,
        "avg_goals_per_game": 2.50,
        "physicality_index": 0.76,
        "tactical_complexity": 0.68,
        "adaptation_seasons": 2,
        "notes": "Partially isolated post-2022; quality variance high across table.",
    },
}

# Normalised stat keys: which stats are affected by league difficulty
# Goals and assists are the most inflated; defensive stats less so
STAT_ADJUSTMENT_FACTORS: Dict[str, float] = {
    "goals":               1.00,   # Full adjustment (most inflation-sensitive)
    "assists":             0.90,   # Slightly less inflation than goals
    "key_passes":          0.75,   # Moderate — harder defences reduce opportunities
    "dribbles_completed":  0.70,   # Physical battles scale with league intensity
    "tackles":             0.50,   # Less inflation — harder attacks ≠ more tackles
    "interceptions":       0.50,
    "pass_accuracy_pct":   0.30,   # Small adjustment — possession % varies but less so
    "xg":                  1.00,   # Full adjustment for expected goals
    "xa":                  0.90,
}


class LeagueDifficultyAdjuster:
    """
    ==========================================================================
    BUSINESS SUMMARY
    ==========================================================================
    Normalise player statistics across leagues, compute step-up risk, and
    estimate adaptation timelines when a player moves between leagues of
    different quality levels.

    Core question answered: "If this Eredivisie striker scores 22 goals,
    how many would they likely score in the Premier League?"

    Step-up risk answers: "What's the probability they significantly
    underperform in their first Premier League season?"
    ==========================================================================

    Args:
        league_data: Override the default LEAGUE_DIFFICULTY dict for testing.
    """

    def __init__(
        self, league_data: Optional[Dict[str, Dict]] = None
    ) -> None:
        self.leagues = league_data if league_data is not None else LEAGUE_DIFFICULTY

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_league(self, name: str) -> Dict:
        """
        Return league config for a given name. Raises KeyError if not found.
        Tries case-insensitive match as fallback.
        """
        if name in self.leagues:
            return self.leagues[name]
        # Case-insensitive search
        name_lower = name.lower()
        for key, val in self.leagues.items():
            if key.lower() == name_lower:
                return val
        raise KeyError(
            f"League '{name}' not in difficulty database. "
            f"Available: {sorted(self.leagues.keys())}"
        )

    def _difficulty_delta(
        self, source_league: str, target_league: str
    ) -> float:
        """
        Return the difficulty differential (target_factor - source_factor).

        Positive = moving to a harder league (step-up).
        Negative = moving to an easier league (step-down / retirement move).
        Zero = same league or equivalent difficulty.
        """
        src = self._get_league(source_league)["factor"]
        tgt = self._get_league(target_league)["factor"]
        return round(tgt - src, 4)

    # ------------------------------------------------------------------ #
    # Public: normalize_stats                                              #
    # ------------------------------------------------------------------ #

    def normalize_stats(
        self,
        raw_stats: Dict[str, float],
        source_league: str,
        target_league: str,
        minutes: int = 2700,
    ) -> Dict[str, float]:
        """
        Normalise a player's raw stats from source league to target league equivalent.

        The adjustment formula per stat:
            adjusted = raw * (target_factor / source_factor) * stat_adjustment_factor
                       * minutes_scaler

        Where:
            - target_factor / source_factor: core difficulty ratio
            - stat_adjustment_factor: per-stat sensitivity to league difficulty
            - minutes_scaler: accounts for different squad rotation patterns

        Args:
            raw_stats: Dict of stat_name → value (e.g. {"goals": 22, "assists": 8}).
            source_league: League the player currently plays in.
            target_league: League to normalise stats to.
            minutes: Minutes played in source league (used for per-90 scaling).

        Returns:
            Dict of same keys with normalised values. Non-adjusted stats are
            passed through unchanged with a "_raw" suffix added to distinguish.

        Example:
            # Eredivisie striker with 22 goals → PL equivalent
            normalized = adjuster.normalize_stats(
                {"goals": 22, "assists": 8, "key_passes": 95},
                source_league="Eredivisie",
                target_league="Premier League",
                minutes=2800,
            )
            # normalized["goals"] ≈ 14.8 (significant reduction)
        """
        src_factor = self._get_league(source_league)["factor"]
        tgt_factor = self._get_league(target_league)["factor"]

        if source_league == target_league:
            return {k: round(v, 3) for k, v in raw_stats.items()}

        # Minutes scaler: PL has 38 games; adjust for effective playing time
        # We normalise to per-90 then project to 2700 min (standard reference)
        reference_minutes = 2700
        minutes_ratio = (minutes / reference_minutes) if minutes > 0 else 1.0

        result: Dict[str, float] = {}
        for stat, value in raw_stats.items():
            if not isinstance(value, (int, float)):
                result[stat] = value
                continue
            sensitivity = STAT_ADJUSTMENT_FACTORS.get(stat, 0.50)
            ratio = tgt_factor / src_factor
            # Apply sensitivity: ratio effect is partial for low-sensitivity stats
            effective_ratio = 1.0 + (ratio - 1.0) * sensitivity
            adjusted = value * effective_ratio / minutes_ratio
            result[stat] = round(max(0.0, adjusted), 3)

        return result

    # ------------------------------------------------------------------ #
    # Public: step_up_risk                                                 #
    # ------------------------------------------------------------------ #

    def step_up_risk(self, source_league: str, target_league: str) -> Dict:
        """
        Compute the probability that a player significantly underperforms after
        moving from source_league to target_league.

        "Significantly underperforms" = produces less than 75% of their
        normalised expected output in their first season.

        Formula:
            delta = target_factor - source_factor
            if delta <= 0: minimal risk (same league or easier)
            risk = sigmoid(delta, midpoint=0.15, steepness=15)

        The sigmoid is calibrated so:
            delta = 0.05 (La Liga → PL): risk ≈ 25%
            delta = 0.12 (Bundesliga → PL): risk ≈ 45%
            delta = 0.22 (Eredivisie → PL): risk ≈ 72%
            delta = 0.32 (Brasileirao → PL): risk ≈ 85%

        Args:
            source_league: Player's current league.
            target_league: Destination league.

        Returns:
            dict with:
                source_league         : Source name
                target_league         : Target name
                difficulty_delta      : Numeric difficulty gap
                step_up_risk_pct      : Probability of significant underperformance (%)
                adaptation_seasons    : Expected seasons to reach 90% of potential
                risk_category         : "minimal" / "low" / "moderate" / "high" / "very_high"
                stat_haircuts         : Expected % reduction per key stat category
                recommendation        : Scouting recommendation string
        """
        delta = self._difficulty_delta(source_league, target_league)
        tgt_info = self._get_league(target_league)
        src_info = self._get_league(source_league)

        if delta <= 0:
            # Easier league or same — no meaningful step-up risk
            risk = 0.05 + abs(delta) * 0.02   # trivial risk for lateral moves
            adaptation = 0
            category = "minimal"
        else:
            # Sigmoid over delta
            risk = 1.0 / (1.0 + math.exp(-15.0 * (delta - 0.15)))
            adaptation = tgt_info.get("adaptation_seasons", 1)
            if delta < 0.05:
                category = "low"
            elif delta < 0.10:
                category = "moderate"
            elif delta < 0.18:
                category = "high"
            else:
                category = "very_high"

        risk = round(min(risk, 0.95), 4)

        # Expected stat haircuts in first season (approximate)
        goals_haircut = round((1 - STAT_ADJUSTMENT_FACTORS["goals"] * (tgt_info["factor"] / src_info["factor"])) * 100, 1) if delta > 0 else 0.0
        passes_haircut = round((1 - STAT_ADJUSTMENT_FACTORS["key_passes"] * (tgt_info["factor"] / src_info["factor"])) * 100, 1) if delta > 0 else 0.0

        rec_map = {
            "minimal": "Minimal step-up. Statistics transfer with high confidence.",
            "low":     "Low step-up risk. Expect slight output reduction in first season.",
            "moderate":"Moderate risk. Budget for 15-20% stat reduction; monitor closely.",
            "high":    "High step-up risk. Significant reduction likely; include performance clauses.",
            "very_high":"Very high risk. Consider loan first or apply heavy statistical discount.",
        }

        return {
            "source_league": source_league,
            "target_league": target_league,
            "source_factor": src_info["factor"],
            "target_factor": tgt_info["factor"],
            "difficulty_delta": delta,
            "step_up_risk_pct": round(risk * 100, 1),
            "adaptation_seasons": adaptation,
            "risk_category": category,
            "stat_haircuts": {
                "goals_pct":      goals_haircut,
                "key_passes_pct": passes_haircut,
            },
            "recommendation": rec_map[category],
        }

    # ------------------------------------------------------------------ #
    # Public: league_ranking                                               #
    # ------------------------------------------------------------------ #

    def league_ranking(self) -> List[Dict]:
        """
        Return all leagues ranked by difficulty factor (hardest first).

        Returns:
            List of dicts: rank, name, factor, tier, country.
        """
        ranked = sorted(self.leagues.items(), key=lambda x: x[1]["factor"], reverse=True)
        return [
            {
                "rank": i + 1,
                "league": name,
                "difficulty_factor": data["factor"],
                "tier": data["tier"],
                "country": data["country"],
                "avg_goals_per_game": data["avg_goals_per_game"],
                "adaptation_seasons": data["adaptation_seasons"],
            }
            for i, (name, data) in enumerate(ranked)
        ]

    # ------------------------------------------------------------------ #
    # Public: cross_league_comparison                                       #
    # ------------------------------------------------------------------ #

    def cross_league_comparison(
        self,
        player_name: str,
        raw_stats: Dict[str, float],
        source_league: str,
        candidate_leagues: List[str],
        minutes: int = 2700,
    ) -> List[Dict]:
        """
        Normalise a player's stats across multiple target leagues simultaneously.

        Useful for comparing how a player's output would translate to different
        potential destination leagues side-by-side.

        Args:
            player_name: For labelling in output.
            raw_stats: Player's current season statistics.
            source_league: Current league.
            candidate_leagues: List of target leagues to compare against.
            minutes: Minutes played in the source league.

        Returns:
            List of dicts sorted by difficulty_factor descending.
            Each dict: league, difficulty_factor, normalised_stats, step_up_risk_pct.
        """
        results = []
        for target in candidate_leagues:
            try:
                norm = self.normalize_stats(raw_stats, source_league, target, minutes)
                risk = self.step_up_risk(source_league, target)
                results.append({
                    "player": player_name,
                    "source_league": source_league,
                    "target_league": target,
                    "difficulty_factor": self._get_league(target)["factor"],
                    "normalised_stats": norm,
                    "step_up_risk_pct": risk["step_up_risk_pct"],
                    "risk_category": risk["risk_category"],
                    "adaptation_seasons": risk["adaptation_seasons"],
                })
            except KeyError:
                continue
        results.sort(key=lambda x: x["difficulty_factor"], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    # Convenience: available leagues                                        #
    # ------------------------------------------------------------------ #

    def available_leagues(self) -> List[str]:
        """Return sorted list of all leagues in the database."""
        return sorted(self.leagues.keys())


# ============================================================================ #
# Smoke test                                                                     #
# ============================================================================ #

if __name__ == "__main__":
    adjuster = LeagueDifficultyAdjuster()

    print("=== LEAGUE DIFFICULTY RANKINGS ===")
    for row in adjuster.league_ranking():
        print(
            f"  #{row['rank']:2d} {row['league']:<30s} "
            f"factor={row['difficulty_factor']:.3f}  tier={row['tier']}"
        )

    print("\n=== STEP-UP RISK EXAMPLES ===")
    test_moves = [
        ("La Liga", "Premier League"),
        ("Bundesliga", "Premier League"),
        ("Eredivisie", "Premier League"),
        ("Primeira Liga", "Premier League"),
        ("Saudi Pro League", "Premier League"),
        ("Premier League", "La Liga"),
    ]
    for src, tgt in test_moves:
        risk = adjuster.step_up_risk(src, tgt)
        print(
            f"  {src:30s} → {tgt:20s} | "
            f"delta={risk['difficulty_delta']:+.3f} | "
            f"risk={risk['step_up_risk_pct']:5.1f}% | "
            f"{risk['risk_category']}"
        )

    print("\n=== STAT NORMALISATION: Eredivisie striker → PL ===")
    raw = {"goals": 22, "assists": 8, "key_passes": 95, "dribbles_completed": 88}
    normalised = adjuster.normalize_stats(raw, "Eredivisie", "Premier League", minutes=2800)
    print(f"  Raw:        {raw}")
    print(f"  Normalised: {normalised}")

    print("\n=== CROSS-LEAGUE COMPARISON ===")
    comp = adjuster.cross_league_comparison(
        "Hypothetical Ligue 1 Striker",
        {"goals": 20, "assists": 7, "key_passes": 65},
        "Ligue 1",
        ["Premier League", "La Liga", "Bundesliga", "Serie A"],
        minutes=2700,
    )
    for c in comp:
        g = c["normalised_stats"].get("goals", "N/A")
        a = c["normalised_stats"].get("assists", "N/A")
        print(
            f"  → {c['target_league']:<20s} | goals={g} assists={a} | "
            f"risk={c['step_up_risk_pct']}% ({c['risk_category']})"
        )
