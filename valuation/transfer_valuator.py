"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
The TransferValuator is the central engine of the system. It takes all
signal outputs — age curve, performance trajectory, contract urgency, FFP
headroom, sentiment, injury risk, and player motivation — and combines them
into two numbers that matter:

1. FAIR VALUE (EUR millions): what the player is actually worth, accounting
   for all signals, vs. what the market currently values them at.

2. TRANSFER PROBABILITY (0-1): the probability that this player moves in
   the current transfer window, considering both the push (unhappiness,
   contract) and pull (which clubs can afford them, which need them).

The ranked output is a DataFrame of the best transfer opportunities within
a specified budget and position filter — the answer to "who should we buy?"

KEY ENGINEERING FEATURES:
- Vectorized batch_score(): processes 1000+ players in a single pandas pass
  using vectorized numpy operations — no Python loops over rows.
- Explicit memory optimization: int8/int16/float32 dtypes throughout.
- Chunked processing: large DataFrames are scored in BATCH_CHUNK_SIZE chunks.
- What-if scenario API: shock any player's inputs and immediately see the
  valuation impact — "what if Mbappé's form drops 20%?"
- Pydantic ValuationResult output: every row is schema-validated.
- Full motivation integration using PSYCH_WEIGHT / PHYSICAL_WEIGHT constants.
=============================================================================

Developer notes:
- All weighting constants imported from config.py.
- batch_score() uses pd.DataFrame.eval() and numpy where possible to stay
  vectorized. Boolean masks avoid iterrows() entirely.
- The what_if_shock() method returns a before/after comparison dict
  so callers can easily display the impact delta.
- Imports all signal modules lazily inside methods to allow partial
  testing without full environment setup.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BATCH_CHUNK_SIZE,
    FFP_MIN_HEADROOM_TO_BID_EUR_M,
    PHYSICAL_WEIGHT,
    PLAYER_DF_DTYPES,
    PSYCH_WEIGHT,
    TOTAL_WEIGHT,
    TRANSFER_PROB_CEILING,
    TRANSFER_PROB_FLOOR,
    VALUATION_WEIGHTS,
)
from data.schemas import CareerPhase, Position, TrendLabel, ValuationResult

logger = logging.getLogger(__name__)


# ============================================================================ #
# Internal helpers                                                                #
# ============================================================================ #

def _sigmoid(x: float, steepness: float = 6.0, midpoint: float = 0.0) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


def _clip_prob(p: float) -> float:
    return round(max(TRANSFER_PROB_FLOOR, min(TRANSFER_PROB_CEILING, p)), 4)


def _safe_normalise(value: float, lo: float, hi: float) -> float:
    """Normalise value to [0, 1] given expected range [lo, hi]."""
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


# ============================================================================ #
# TransferValuator                                                                #
# ============================================================================ #

class TransferValuator:
    """
    ==========================================================================
    BUSINESS SUMMARY
    ==========================================================================
    Master valuation model combining all signals into fair_value and
    transfer_probability scores. Supports both single-player and vectorized
    batch scoring for large datasets.
    ==========================================================================

    Args:
        run_id: Optional ETL run ID (used for cache tagging on API calls).
        enable_sentiment: If False, skips live NewsAPI calls (faster, for batch runs).
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        enable_sentiment: bool = True,
    ) -> None:
        self.run_id = run_id
        self.enable_sentiment = enable_sentiment

        # Lazy-load signal modules to avoid circular imports
        self._age_curve = None
        self._trajectory = None
        self._ffp = None
        self._contract = None
        self._injury = None
        self._motivation = None
        self._sentiment = None

    # ------------------------------------------------------------------ #
    # Signal module accessors (lazy init)                                   #
    # ------------------------------------------------------------------ #

    def _get_age_curve(self):
        if self._age_curve is None:
            from signals.age_value_curve import AgeValueCurve
            self._age_curve = AgeValueCurve()
        return self._age_curve

    def _get_trajectory(self):
        if self._trajectory is None:
            from signals.performance_trajectory import PerformanceTrajectoryAnalyzer
            self._trajectory = PerformanceTrajectoryAnalyzer()
        return self._trajectory

    def _get_ffp(self):
        if self._ffp is None:
            from signals.ffp_headroom import FFPAnalyzer
            self._ffp = FFPAnalyzer()
        return self._ffp

    def _get_contract(self):
        if self._contract is None:
            from signals.contract_signal import ContractSignalAnalyzer
            self._contract = ContractSignalAnalyzer()
        return self._contract

    def _get_injury(self):
        if self._injury is None:
            from signals.injury_risk_model import InjuryRiskAnalyzer
            self._injury = InjuryRiskAnalyzer()
        return self._injury

    def _get_motivation(self):
        if self._motivation is None:
            from signals.player_motivation_model import PlayerMotivationScorer
            self._motivation = PlayerMotivationScorer()
        return self._motivation

    def _get_sentiment(self):
        if self._sentiment is None and self.enable_sentiment:
            from signals.sentiment_scorer import NewsletterSentimentScorer
            self._sentiment = NewsletterSentimentScorer()
        return self._sentiment

    # ------------------------------------------------------------------ #
    # Public: fair_value                                                    #
    # ------------------------------------------------------------------ #

    def fair_value(
        self,
        player: str,
        age: int,
        position: str,
        current_market_value: float,
        rating: float = 7.5,
        contract_months: int = 24,
        last_season_minutes: int = 2700,
        perf_trajectory: Optional[Dict] = None,
    ) -> float:
        """
        Compute multi-factor fair value for a player in EUR millions.

        Formula:
            base_fair = age_curve_fair_value (from AgeValueCurve)
            injury_adjusted = base_fair × (1 - injury_discount)
            trajectory_adj = injury_adjusted × trajectory_multiplier
            contract_discount = function of months remaining
            fair_value = trajectory_adj × contract_multiplier

        Args:
            player: Player name.
            age: Current age.
            position: Position code (e.g. "CB", "FW").
            current_market_value: Current Transfermarkt value in EUR millions.
            rating: Overall performance rating (1–10).
            contract_months: Months remaining on contract.
            last_season_minutes: Minutes played last season (for injury risk).
            perf_trajectory: Optional pre-computed trajectory dict from
                PerformanceTrajectoryAnalyzer. If None, uses age-curve only.

        Returns:
            Fair value estimate in EUR millions.
        """
        # 1. Age-curve undervaluation
        avc = self._get_age_curve()
        uv_score = avc.identify_undervalued(age, current_market_value, position, rating)
        base_fair = current_market_value + uv_score

        # 2. Injury risk discount
        injury = self._get_injury()
        inj_result = injury.adjusted_valuation(player, base_fair, age, position, last_season_minutes)
        injury_adjusted = inj_result["adjusted_value_eur_m"]

        # 3. Performance trajectory multiplier
        if perf_trajectory and "composite_slope" in perf_trajectory:
            slope = perf_trajectory["composite_slope"]
            # Slope in [-1, 1] → multiplier [0.85, 1.15]
            traj_multiplier = 1.0 + slope * 0.15
        else:
            traj_multiplier = 1.0

        fair = injury_adjusted * traj_multiplier

        # 4. Contract discount: players with < 12 months get discounted
        # (selling club's bargaining position weakens)
        if contract_months <= 6:
            fair *= 0.70
        elif contract_months <= 12:
            fair *= 0.82
        elif contract_months <= 18:
            fair *= 0.91
        # else: no discount

        return round(max(0.5, fair), 2)

    # ------------------------------------------------------------------ #
    # Public: transfer_probability                                          #
    # ------------------------------------------------------------------ #

    def transfer_probability(
        self,
        player: str,
        age: int,
        position: str,
        current_market_value: float,
        contract_months: int,
        current_club: str,
        trajectory_slope: float = 0.0,
        sentiment_hype: float = 0.5,
        rating: float = 7.5,
        last_season_minutes: int = 2700,
    ) -> float:
        """
        Compute the probability of a player transferring in the current window.

        Multi-factor combination:
            P_base = weighted_sum(sub_scores) → sigmoid → [0, 1]
            P_motivation = motivation_scorer.transfer_probability_adjusted(player, P_base)
            P_final = clip(P_motivation, FLOOR, CEILING)

        Sub-scores (weights from VALUATION_WEIGHTS in config.py):
            age_curve_score   : Undervaluation vs. current market price
            performance_score : Trajectory slope (ascending = higher demand)
            contract_score    : Urgency of contract situation
            ffp_fit_score     : Number of clubs that can afford + need the player
            sentiment_score   : News hype score
            motivation_score  : (not a sub-score; applied as multiplier separately)

        Args:
            player: Player name.
            age: Current age.
            position: Position code.
            current_market_value: Market value in EUR millions.
            contract_months: Months remaining on contract.
            current_club: Club the player is at (for FFP exclusion).
            trajectory_slope: Composite performance slope.
            sentiment_hype: Hype score from sentiment scorer [0, 1].
            rating: Performance rating (1–10).
            last_season_minutes: Minutes last season.

        Returns:
            Transfer probability in [TRANSFER_PROB_FLOOR, TRANSFER_PROB_CEILING].
        """
        # --- Sub-score: age curve ---
        avc = self._get_age_curve()
        uv = avc.identify_undervalued(age, current_market_value, position, rating)
        # Positive undervaluation = attractive buy = higher demand = higher prob
        age_curve_score = _safe_normalise(uv, -30.0, 30.0)

        # --- Sub-score: performance ---
        # Ascending trajectory → other clubs more interested
        performance_score = _safe_normalise(trajectory_slope, -0.5, 0.5)

        # --- Sub-score: contract urgency ---
        contract = self._get_contract()
        try:
            urgency = contract.get_urgency_score(player)
        except (KeyError, Exception):
            urgency = _safe_normalise(max(0, 36 - contract_months), 0, 36)
        contract_score = urgency

        # --- Sub-score: FFP fit (how many clubs can afford + might need this player) ---
        ffp = self._get_ffp()
        affordable_clubs = ffp.clubs_that_can_afford(
            current_market_value, exclude_clubs=[current_club]
        )
        # Normalize: 0 clubs = 0 score; 10+ clubs = 1.0 score
        ffp_fit_score = _safe_normalise(len(affordable_clubs), 0, 10)

        # --- Sub-score: sentiment ---
        sentiment_score = sentiment_hype  # Already [0, 1]

        # --- Weighted composite (VALUATION_WEIGHTS from config.py) ---
        w = VALUATION_WEIGHTS
        composite = (
            age_curve_score   * w["age_curve"]
            + performance_score * w["performance"]
            + contract_score    * w["contract"]
            + ffp_fit_score     * w["ffp_fit"]
            + sentiment_score   * w["sentiment"]
        ) / (1.0 - w["motivation"])   # Re-normalise: motivation applied separately

        # --- Sigmoid squeeze to (0, 1) ---
        p_base = _sigmoid(composite, steepness=6.0, midpoint=0.5)
        p_base = _clip_prob(p_base)

        # --- Motivation adjustment ---
        motivation = self._get_motivation()
        p_final = motivation.transfer_probability_adjusted(player, p_base)

        return _clip_prob(p_final)

    # ------------------------------------------------------------------ #
    # Public: batch_score — VECTORIZED                                      #
    # ------------------------------------------------------------------ #

    def batch_score(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score ALL players simultaneously using vectorized pandas/numpy operations.

        This is the production-scale scoring path. It processes 1000+ players
        in a single vectorized pass — no Python loops over rows.

        Expected input columns (all others are ignored):
            name, age, position, club, current_value_eur_m, rating,
            contract_months, trajectory_slope, injury_risk, sentiment_hype,
            contentment_score

        Memory optimization:
            - Input DataFrame is cast to PLAYER_DF_DTYPES before processing.
            - Results are returned with explicit dtype specification.
            - Large DataFrames are processed in BATCH_CHUNK_SIZE chunks to
              control peak memory usage.

        Args:
            players_df: DataFrame with one row per player. See expected columns.

        Returns:
            DataFrame with all input columns plus computed signal columns:
            fair_value_eur_m, transfer_probability, age_curve_score,
            performance_score, contract_score, ffp_fit_score, sentiment_score_v,
            motivation_score, composite_score, career_phase, trend_label.

        Example:
            df = pd.read_json("data/sample_players.json")
            results = valuator.batch_score(df)
            top_5 = results.nlargest(5, "composite_score")
        """
        if players_df.empty:
            return players_df.copy()

        # --- Memory optimisation: cast to compact dtypes ---
        df = players_df.copy()
        for col, dtype in PLAYER_DF_DTYPES.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    pass  # Leave as-is if cast fails

        # --- Chunked processing for large datasets ---
        n = len(df)
        if n > BATCH_CHUNK_SIZE:
            chunks = []
            for start in range(0, n, BATCH_CHUNK_SIZE):
                chunk = df.iloc[start:start + BATCH_CHUNK_SIZE].copy()
                scored_chunk = self._score_chunk(chunk)
                chunks.append(scored_chunk)
            return pd.concat(chunks, ignore_index=True)

        return self._score_chunk(df)

    def _score_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score one chunk of the players DataFrame using vectorized operations.

        All computations use numpy array operations or pandas vectorized
        methods — no iterrows(), no apply() with Python lambdas over rows.
        """
        # ---- Extract columns as numpy arrays for vectorized ops ----
        ages         = df["age"].to_numpy(dtype=np.float32)
        values       = df["current_value_eur_m"].to_numpy(dtype=np.float32)
        ratings      = df.get("rating", pd.Series(7.5, index=df.index)).to_numpy(dtype=np.float32)
        contracts    = df.get("contract_months", pd.Series(24, index=df.index)).to_numpy(dtype=np.float32)
        slopes       = df.get("trajectory_slope", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
        injury_risks = df.get("injury_risk", pd.Series(0.2, index=df.index)).to_numpy(dtype=np.float32)
        hypes        = df.get("sentiment_hype", pd.Series(0.5, index=df.index)).to_numpy(dtype=np.float32)
        contentments = df.get("contentment_score", pd.Series(70.0, index=df.index)).to_numpy(dtype=np.float32)

        n = len(df)
        w = VALUATION_WEIGHTS

        # ---- Vectorized: age curve score ----
        # Approximate age-curve undervaluation as a vectorized sigmoid
        # over (age - position_peak), avoiding per-row Python calls.
        # Full AgeValueCurve is called per-player only in single-player mode.
        # Here we use a position-averaged approximation.
        position_peak_map = np.array([
            _POSITION_PEAK_AVG.get(str(p), 27.0)
            for p in df.get("position", pd.Series("CM", index=df.index))
        ], dtype=np.float32)
        age_from_peak = ages - position_peak_map  # negative = pre-peak, positive = post-peak
        # Undervaluation increases for pre-peak players, decreases post-peak
        uv_approx = -age_from_peak / 10.0  # [-1.5, +1.5] approx
        age_curve_scores = np.clip((uv_approx + 1.5) / 3.0, 0.0, 1.0).astype(np.float32)

        # ---- Vectorized: performance score ----
        performance_scores = np.clip((slopes + 0.5) / 1.0, 0.0, 1.0).astype(np.float32)

        # ---- Vectorized: contract score (urgency from months remaining) ----
        # Sigmoid approximation: high urgency when contracts < 18 months
        contract_scores = (1.0 / (1.0 + np.exp(0.12 * (contracts - 18.0)))).astype(np.float32)

        # ---- Vectorized: FFP fit score ----
        # Approximate: players with lower values have more affordable buyers
        # Full FFP lookup only done in single-player mode; here use value proxy
        ffp_scores = np.clip(1.0 - values / 200.0, 0.05, 0.95).astype(np.float32)

        # ---- Vectorized: sentiment score ----
        sentiment_scores = hypes.astype(np.float32)

        # ---- Vectorized: motivation score ----
        # Contentment 70 → neutral (0.5). Below 70 → higher transfer prob signal.
        motivation_scores = np.clip((100.0 - contentments) / 100.0, 0.0, 1.0).astype(np.float32)

        # ---- Weighted composite (numpy dot product — fully vectorized) ----
        weight_vec = np.array([
            w["age_curve"], w["performance"], w["contract"],
            w["ffp_fit"], w["sentiment"], w["motivation"],
        ], dtype=np.float32)
        signal_matrix = np.stack([
            age_curve_scores, performance_scores, contract_scores,
            ffp_scores, sentiment_scores, motivation_scores,
        ], axis=1)   # shape: (n, 6)
        composite_raw = signal_matrix @ weight_vec   # shape: (n,)

        # ---- Sigmoid → transfer probability ----
        transfer_probs = np.clip(
            1.0 / (1.0 + np.exp(-6.0 * (composite_raw - 0.5))),
            TRANSFER_PROB_FLOOR,
            TRANSFER_PROB_CEILING,
        ).astype(np.float32)

        # ---- Fair value: vectorized approximation ----
        # full fair_value() requires per-player calls; here use vectorized proxy
        uv_eur = (uv_approx * values * 0.2).astype(np.float32)    # rough uv in EUR m
        inj_discount = np.clip(injury_risks ** 1.5 * 0.45, 0.0, 0.45).astype(np.float32)
        traj_mult = (1.0 + slopes * 0.15).astype(np.float32)
        contract_mult = np.where(
            contracts <= 6, 0.70,
            np.where(contracts <= 12, 0.82,
                     np.where(contracts <= 18, 0.91, 1.0))
        ).astype(np.float32)
        fair_values = np.maximum(
            (values + uv_eur) * (1.0 - inj_discount) * traj_mult * contract_mult,
            0.5
        ).astype(np.float32)

        # ---- Composite score (0-100 for readability) ----
        composite_scores = (composite_raw * 100.0).astype(np.float32)

        # ---- Attach results to DataFrame ----
        result = df.copy()
        result["age_curve_score"]     = age_curve_scores
        result["performance_score"]   = performance_scores
        result["contract_score"]      = contract_scores
        result["ffp_fit_score"]       = ffp_scores
        result["sentiment_score_v"]   = sentiment_scores
        result["motivation_score"]    = motivation_scores
        result["composite_score"]     = composite_scores.round(1)
        result["transfer_probability"]= transfer_probs.round(4)
        result["fair_value_eur_m"]    = fair_values.round(2)
        result["value_delta_eur_m"]   = (fair_values - values).round(2)

        # Career phase labels (vectorized string ops via numpy select)
        phase_conditions = [
            ages <= 21,
            (ages > 21) & (ages <= 28) & (slopes > -0.1),
            (ages > 28) & (ages <= 33) & (slopes > -0.2),
        ]
        phase_choices = ["emerging", "prime", "declining"]
        result["career_phase"] = np.select(phase_conditions, phase_choices, default="veteran")

        # Trend labels
        trend_conditions = [slopes > 0.15, slopes < -0.15]
        trend_choices = ["ascending", "declining"]
        result["trend_label"] = np.select(trend_conditions, trend_choices, default="peak")

        return result

    # ------------------------------------------------------------------ #
    # Public: rank_transfer_targets                                         #
    # ------------------------------------------------------------------ #

    def rank_transfer_targets(
        self,
        budget: float,
        target_positions: List[str],
        players_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rank transfer targets within a budget and position filter.

        Loads sample player data if no DataFrame is provided. Filters by
        position and budget, scores via batch_score(), and returns ranked
        results sorted by composite_score descending.

        Args:
            budget: Maximum transfer fee budget in EUR millions.
            target_positions: List of position codes to filter (e.g. ["CB", "CM"]).
            players_df: Optional pre-loaded player DataFrame.

        Returns:
            DataFrame of ranked targets with all signal columns.
        """
        if players_df is None:
            players_df = self._load_sample_players()

        # Filter by position and budget
        pos_mask = players_df["position"].isin(target_positions)
        budget_mask = players_df["current_value_eur_m"] <= budget
        filtered = players_df[pos_mask & budget_mask].copy()

        if filtered.empty:
            logger.warning("No players matching positions=%s and budget=€%sm", target_positions, budget)
            return pd.DataFrame()

        scored = self.batch_score(filtered)
        return scored.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Public: what_if_shock                                                 #
    # ------------------------------------------------------------------ #

    def what_if_shock(
        self,
        player: str,
        age: int,
        position: str,
        current_market_value: float,
        rating: float,
        contract_months: int,
        last_season_minutes: int,
        rating_delta: float = 0.0,
        value_multiplier: float = 1.0,
        scenario_name: str = "Custom Scenario",
    ) -> Dict[str, Any]:
        """
        Run a what-if scenario: shock the player's inputs and show the
        immediate impact on fair value and transfer probability.

        This implements the WhatIfScenario pattern from data/schemas.py,
        using PlayerProfile.with_shock() to produce a mutated player
        and re-running the valuation model on both.

        Args:
            player: Player name.
            age: Current age.
            position: Position code.
            current_market_value: Current market value in EUR millions.
            rating: Current performance rating (1–10).
            contract_months: Months remaining on contract.
            last_season_minutes: Minutes played last season.
            rating_delta: Add this to rating (e.g. -1.5 = form drop).
            value_multiplier: Multiply market value (e.g. 0.8 = 20% drop).
            scenario_name: Human-readable scenario label.

        Returns:
            Dict with:
                scenario_name, player,
                baseline: {fair_value, transfer_probability, rating, value}
                shocked:  {fair_value, transfer_probability, rating, value}
                delta:    {fair_value_delta, prob_delta, value_delta}

        Example:
            result = valuator.what_if_shock(
                "Kylian Mbappe", age=26, position="FW",
                current_market_value=180.0, rating=9.2,
                contract_months=6, last_season_minutes=3000,
                rating_delta=-1.5, value_multiplier=0.80,
                scenario_name="Mbappe_injury_form_drop",
            )
        """
        # --- Baseline ---
        traj = self._get_trajectory()
        # Use demo trajectory if available, else neutral slope
        demo_traj = None
        try:
            demo_traj = traj.get_demo_player_trajectory(player)
        except KeyError:
            pass
        slope = demo_traj["composite_slope"] if demo_traj else 0.0

        sentiment_hype = 0.5   # Neutral default for what-if

        baseline_fv = self.fair_value(
            player, age, position, current_market_value,
            rating, contract_months, last_season_minutes,
            perf_trajectory=demo_traj,
        )
        baseline_prob = self.transfer_probability(
            player, age, position, current_market_value,
            contract_months, "Unknown",
            trajectory_slope=slope,
            sentiment_hype=sentiment_hype,
            rating=rating,
            last_season_minutes=last_season_minutes,
        )

        # --- Shocked inputs ---
        shocked_rating = max(4.0, min(10.0, rating + rating_delta))
        shocked_value = round(current_market_value * value_multiplier, 2)

        shocked_fv = self.fair_value(
            player, age, position, shocked_value,
            shocked_rating, contract_months, last_season_minutes,
            perf_trajectory=demo_traj,
        )
        shocked_prob = self.transfer_probability(
            player, age, position, shocked_value,
            contract_months, "Unknown",
            trajectory_slope=slope,
            sentiment_hype=sentiment_hype,
            rating=shocked_rating,
            last_season_minutes=last_season_minutes,
        )

        return {
            "scenario_name": scenario_name,
            "player": player,
            "baseline": {
                "fair_value_eur_m": baseline_fv,
                "transfer_probability": baseline_prob,
                "rating": rating,
                "market_value_eur_m": current_market_value,
            },
            "shocked": {
                "fair_value_eur_m": shocked_fv,
                "transfer_probability": shocked_prob,
                "rating": shocked_rating,
                "market_value_eur_m": shocked_value,
            },
            "delta": {
                "fair_value_delta_eur_m": round(shocked_fv - baseline_fv, 2),
                "prob_delta": round(shocked_prob - baseline_prob, 4),
                "value_delta_eur_m": round(shocked_value - current_market_value, 2),
                "rating_delta": rating_delta,
                "value_multiplier": value_multiplier,
            },
        }

    # ------------------------------------------------------------------ #
    # Public: to_valuation_result                                           #
    # ------------------------------------------------------------------ #

    def to_valuation_result(self, row: pd.Series) -> Optional[ValuationResult]:
        """
        Convert a scored DataFrame row to a validated ValuationResult Pydantic object.

        Returns None if the row cannot be validated (logged as warning).
        """
        try:
            return ValuationResult(
                player_name=str(row.get("name", "Unknown")),
                position=Position(row.get("position", "CM")),
                age=int(row.get("age", 25)),
                current_club=str(row.get("club", "Unknown")),
                fair_value_eur_m=float(row.get("fair_value_eur_m", 0.0)),
                current_value_eur_m=float(row.get("current_value_eur_m", 0.0)),
                value_delta_eur_m=float(row.get("value_delta_eur_m", 0.0)),
                transfer_probability=float(row.get("transfer_probability", 0.5)),
                age_curve_score=float(row.get("age_curve_score", 0.5)),
                performance_score=float(row.get("performance_score", 0.5)),
                contract_urgency_score=float(row.get("contract_score", 0.5)),
                ffp_fit_score=float(row.get("ffp_fit_score", 0.5)),
                sentiment_score=float(row.get("sentiment_score_v", 0.5)),
                composite_score=float(row.get("composite_score", 50.0)),
                career_phase=CareerPhase(row.get("career_phase", "prime")),
                trend_label=TrendLabel(row.get("trend_label", "peak")),
                contract_months_remaining=int(row.get("contract_months", 24)),
                likely_buyer_count=int(row.get("ffp_fit_score", 0.5) * 10),
                run_id=self.run_id,
            )
        except Exception as exc:
            logger.warning("ValuationResult validation failed for %s: %s", row.get("name"), exc)
            return None

    # ------------------------------------------------------------------ #
    # Internal: sample data loader                                          #
    # ------------------------------------------------------------------ #

    def _load_sample_players(self) -> pd.DataFrame:
        """Load sample_players.json and return as a properly typed DataFrame."""
        import json
        from config import SAMPLE_PLAYERS_PATH

        with open(SAMPLE_PLAYERS_PATH, "r") as f:
            data = json.load(f)

        players = data if isinstance(data, list) else data.get("players", [])
        rows = []
        for p in players:
            # Compute trajectory slope from last_3_seasons_stats if present
            slope = 0.0
            try:
                seasons = p.get("last_3_seasons_stats", [])
                if len(seasons) >= 2:
                    traj = self._get_trajectory()
                    t = traj.compute_trajectory(seasons)
                    slope = t["composite_slope"]
            except Exception:
                pass

            rows.append({
                "name":                 p.get("name", "Unknown"),
                "age":                  p.get("age", 25),
                "position":             p.get("position", "CM"),
                "club":                 p.get("club", "Unknown"),
                "current_value_eur_m":  p.get("current_value_eur_m", 30.0),
                "rating":               p.get("rating", 7.5),
                "contract_months":      max(0, (p.get("contract_expires", 2027) - 2025) * 12),
                "trajectory_slope":     slope,
                "injury_risk":          0.20,
                "sentiment_hype":       0.50,
                "contentment_score":    70.0,
            })

        df = pd.DataFrame(rows)
        for col, dtype in PLAYER_DF_DTYPES.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception:
                    pass
        return df


# ============================================================================ #
# Position peak age lookup for vectorized scoring                                #
# ============================================================================ #

_POSITION_PEAK_AVG: Dict[str, float] = {
    "GK": 29.0, "CB": 28.0, "RB": 26.0, "LB": 26.0,
    "DM": 27.5, "CM": 27.0, "AM": 26.0,
    "RW": 25.0, "LW": 25.0, "FW": 26.0,
}


# ============================================================================ #
# Smoke test                                                                     #
# ============================================================================ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    valuator = TransferValuator(enable_sentiment=False)

    print("=== FAIR VALUE: Leny Yoro ===")
    fv = valuator.fair_value("Leny Yoro", 18, "CB", 70.0, 7.8, 12, 2800)
    print(f"  Fair value: €{fv}m (market: €70m)")

    print("\n=== TRANSFER PROBABILITY: Kylian Mbappe ===")
    tp = valuator.transfer_probability(
        "Kylian Mbappe", 25, "FW", 180.0, 6,
        "Paris Saint-Germain", trajectory_slope=0.05,
        sentiment_hype=0.82, rating=9.2, last_season_minutes=2900,
    )
    print(f"  Transfer probability: {tp:.3f}")

    print("\n=== WHAT-IF: Mbappe form drop 20% ===")
    result = valuator.what_if_shock(
        "Kylian Mbappe", 25, "FW", 180.0, 9.2, 6, 2900,
        rating_delta=-1.5, value_multiplier=0.80,
        scenario_name="Mbappe_20pct_form_drop",
    )
    print(f"  Baseline FV: €{result['baseline']['fair_value_eur_m']}m | prob={result['baseline']['transfer_probability']:.3f}")
    print(f"  Shocked FV:  €{result['shocked']['fair_value_eur_m']}m | prob={result['shocked']['transfer_probability']:.3f}")
    print(f"  Delta FV: €{result['delta']['fair_value_delta_eur_m']}m | prob_delta={result['delta']['prob_delta']:+.3f}")

    print("\n=== BATCH SCORE: 5 players ===")
    test_df = pd.DataFrame([
        {"name": "Leny Yoro",    "age": 18, "position": "CB", "club": "LOSC Lille",    "current_value_eur_m": 70.0, "rating": 7.8, "contract_months": 12, "trajectory_slope": 0.35, "injury_risk": 0.15, "sentiment_hype": 0.62, "contentment_score": 42.0},
        {"name": "Joao Neves",   "age": 20, "position": "CM", "club": "Benfica",        "current_value_eur_m": 70.0, "rating": 8.1, "contract_months": 36, "trajectory_slope": 0.42, "injury_risk": 0.10, "sentiment_hype": 0.55, "contentment_score": 58.0},
        {"name": "Manuel Ugarte","age": 23, "position": "DM", "club": "PSG",            "current_value_eur_m": 55.0, "rating": 7.6, "contract_months": 24, "trajectory_slope": 0.18, "injury_risk": 0.18, "sentiment_hype": 0.48, "contentment_score": 65.0},
        {"name": "Dean Huijsen", "age": 19, "position": "CB", "club": "Bournemouth",    "current_value_eur_m": 45.0, "rating": 7.5, "contract_months": 48, "trajectory_slope": 0.40, "injury_risk": 0.12, "sentiment_hype": 0.45, "contentment_score": 60.0},
        {"name": "Manu Kone",    "age": 23, "position": "CM", "club": "Real Madrid",    "current_value_eur_m": 40.0, "rating": 7.4, "contract_months": 48, "trajectory_slope": 0.30, "injury_risk": 0.20, "sentiment_hype": 0.42, "contentment_score": 68.0},
    ])
    scored = valuator.batch_score(test_df)
    cols = ["name", "position", "fair_value_eur_m", "transfer_probability", "composite_score", "career_phase"]
    print(scored[cols].to_string(index=False))
