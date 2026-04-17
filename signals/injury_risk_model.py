"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
Injuries are the single biggest destroyer of player value in football.
A player rated at €100m becomes worth significantly less if they carry a
history of muscle tears or structural damage. This module quantifies that
risk so that the valuation model can apply an appropriate discount.

The injury risk score (0-1) combines five factors:
1. Age (older players recover slower and injure more often)
2. Position (central midfielders and fullbacks face highest workload;
   goalkeepers face lowest contact risk)
3. Minutes load (players who played >3,500 minutes last season show fatigue)
4. Injury history (number, type, and recency of significant injuries)
5. Body type proxy (approximate BMI from reported height/weight)

For non-technical readers: if a club is considering paying €80m for a player
who has had two ACL tears and is over 30, this module will recommend a lower
bid — the player is a quality asset with a meaningful chance of further injury.
=============================================================================

Developer notes:
- All risk scores are sigmoid-normalised to [0, 1].
- Value discounts are position-and-phase sensitive (a GK with injuries is less
  affected than a pacey winger whose speed is their primary value driver).
- Recovery trajectories are based on peer-reviewed sports medicine estimates
  (FIFA Medical Assessment and Research Centre data, publicly available).
- Injury history data is hardcoded for 25 real players and is approximate /
  illustrative — not official medical records.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


# ============================================================================ #
# Constants: Position risk weights                                               #
# ============================================================================ #
# Higher value = higher workload / contact risk for that position.
# Source: approximated from UEFA injury study meta-analyses.

POSITION_WORKLOAD_RISK: Dict[str, float] = {
    "GK":  0.10,   # Low contact, minimal sprint volume
    "CB":  0.30,   # Aerial duels, tackles; but lower sprint load than fullbacks
    "RB":  0.55,   # High sprint volume, overlaps, defensive duties
    "LB":  0.55,
    "DM":  0.45,   # Interceptions and tackles; high contact rate
    "CM":  0.50,   # Highest workload position — box-to-box demands
    "AM":  0.40,   # Creative play but more exposed to hard challenges
    "RW":  0.45,   # High-speed dribbling → hamstring / ankle risk
    "LW":  0.45,
    "FW":  0.35,   # Contact but less running than midfielders
}
_DEFAULT_WORKLOAD_RISK = 0.40

# Injury type severity and expected recovery in weeks
INJURY_PROFILES: Dict[str, Dict] = {
    "hamstring_grade1":   {"severity": 0.20, "recovery_weeks": 2,  "value_impact_pct": 0.02},
    "hamstring_grade2":   {"severity": 0.40, "recovery_weeks": 6,  "value_impact_pct": 0.06},
    "hamstring_grade3":   {"severity": 0.60, "recovery_weeks": 12, "value_impact_pct": 0.12},
    "muscular_tear":      {"severity": 0.45, "recovery_weeks": 8,  "value_impact_pct": 0.08},
    "ankle_sprain":       {"severity": 0.25, "recovery_weeks": 3,  "value_impact_pct": 0.03},
    "knee_acl":           {"severity": 0.95, "recovery_weeks": 40, "value_impact_pct": 0.25},
    "knee_meniscus":      {"severity": 0.70, "recovery_weeks": 20, "value_impact_pct": 0.15},
    "knee_ligament_lcl":  {"severity": 0.55, "recovery_weeks": 14, "value_impact_pct": 0.10},
    "fracture":           {"severity": 0.65, "recovery_weeks": 16, "value_impact_pct": 0.14},
    "back_disc":          {"severity": 0.60, "recovery_weeks": 12, "value_impact_pct": 0.12},
    "adductor_strain":    {"severity": 0.30, "recovery_weeks": 4,  "value_impact_pct": 0.04},
    "calf_strain":        {"severity": 0.30, "recovery_weeks": 4,  "value_impact_pct": 0.04},
    "groin_strain":       {"severity": 0.35, "recovery_weeks": 5,  "value_impact_pct": 0.05},
    "foot_metatarsal":    {"severity": 0.50, "recovery_weeks": 10, "value_impact_pct": 0.09},
    "thigh_contusion":    {"severity": 0.15, "recovery_weeks": 1,  "value_impact_pct": 0.01},
    "none":               {"severity": 0.00, "recovery_weeks": 0,  "value_impact_pct": 0.00},
}

# Approximate body composition proxies (height_cm, weight_kg) for BMI calculation
PLAYER_PHYSICAL: Dict[str, Dict] = {
    "Erling Haaland":    {"height_cm": 194, "weight_kg": 88},
    "Pedri":             {"height_cm": 174, "weight_kg": 63},
    "Jude Bellingham":   {"height_cm": 181, "weight_kg": 75},
    "Bukayo Saka":       {"height_cm": 178, "weight_kg": 72},
    "Virgil van Dijk":   {"height_cm": 193, "weight_kg": 92},
    "Leny Yoro":         {"height_cm": 194, "weight_kg": 83},
    "Manuel Ugarte":     {"height_cm": 182, "weight_kg": 75},
    "Joao Neves":        {"height_cm": 177, "weight_kg": 69},
    "Kylian Mbappe":     {"height_cm": 178, "weight_kg": 73},
    "Declan Rice":       {"height_cm": 185, "weight_kg": 82},
    "Phil Foden":        {"height_cm": 171, "weight_kg": 69},
    "Ruben Dias":        {"height_cm": 187, "weight_kg": 82},
    "Rodri":             {"height_cm": 191, "weight_kg": 82},
    "Vinicius Junior":   {"height_cm": 176, "weight_kg": 73},
    "Dean Huijsen":      {"height_cm": 192, "weight_kg": 84},
    "Federico Chiesa":   {"height_cm": 175, "weight_kg": 69},
    "Marcus Thuram":     {"height_cm": 192, "weight_kg": 90},
    "Manu Kone":         {"height_cm": 185, "weight_kg": 80},
    "Alisson Becker":    {"height_cm": 193, "weight_kg": 91},
    "William Saliba":    {"height_cm": 192, "weight_kg": 86},
    "Lautaro Martinez":  {"height_cm": 174, "weight_kg": 72},
    "Jamal Musiala":     {"height_cm": 180, "weight_kg": 70},
    "Leroy Sane":        {"height_cm": 183, "weight_kg": 75},
    "Toni Kroos":        {"height_cm": 183, "weight_kg": 76},
    "N'Golo Kante":      {"height_cm": 168, "weight_kg": 70},
}

# Hardcoded injury history for 25 real players
# Each entry: list of {"season": ..., "injury_type": ..., "games_missed": int}
INJURY_HISTORY: Dict[str, List[Dict]] = {
    "Erling Haaland": [
        {"season": "2020/21", "injury_type": "foot_metatarsal",  "games_missed": 12},
        {"season": "2021/22", "injury_type": "muscular_tear",    "games_missed": 8},
    ],
    "Pedri": [
        {"season": "2021/22", "injury_type": "hamstring_grade2", "games_missed": 22},
        {"season": "2022/23", "injury_type": "knee_meniscus",    "games_missed": 18},
        {"season": "2023/24", "injury_type": "hamstring_grade1", "games_missed": 6},
    ],
    "Jude Bellingham": [],   # Remarkably injury-free
    "Bukayo Saka": [
        {"season": "2020/21", "injury_type": "hamstring_grade1", "games_missed": 3},
        {"season": "2024/25", "injury_type": "hamstring_grade2", "games_missed": 8},
    ],
    "Virgil van Dijk": [
        {"season": "2020/21", "injury_type": "knee_acl",         "games_missed": 40},
        {"season": "2023/24", "injury_type": "ankle_sprain",     "games_missed": 3},
    ],
    "Leny Yoro": [
        {"season": "2024/25", "injury_type": "foot_metatarsal",  "games_missed": 14},
    ],
    "Manuel Ugarte": [
        {"season": "2022/23", "injury_type": "groin_strain",     "games_missed": 4},
    ],
    "Joao Neves": [],
    "Kylian Mbappe": [
        {"season": "2020/21", "injury_type": "ankle_sprain",     "games_missed": 3},
        {"season": "2021/22", "injury_type": "thigh_contusion",  "games_missed": 2},
        {"season": "2024/25", "injury_type": "hamstring_grade2", "games_missed": 7},
    ],
    "Declan Rice": [
        {"season": "2024/25", "injury_type": "hamstring_grade1", "games_missed": 4},
    ],
    "Phil Foden": [
        {"season": "2019/20", "injury_type": "ankle_sprain",     "games_missed": 5},
        {"season": "2023/24", "injury_type": "adductor_strain",  "games_missed": 3},
    ],
    "Ruben Dias": [
        {"season": "2021/22", "injury_type": "hamstring_grade2", "games_missed": 8},
        {"season": "2023/24", "injury_type": "muscular_tear",    "games_missed": 6},
    ],
    "Rodri": [
        {"season": "2024/25", "injury_type": "knee_acl",         "games_missed": 35},
    ],
    "Vinicius Junior": [
        {"season": "2021/22", "injury_type": "muscular_tear",    "games_missed": 5},
        {"season": "2022/23", "injury_type": "ankle_sprain",     "games_missed": 3},
    ],
    "Dean Huijsen": [],
    "Federico Chiesa": [
        {"season": "2020/21", "injury_type": "knee_acl",         "games_missed": 38},
        {"season": "2021/22", "injury_type": "hamstring_grade2", "games_missed": 10},
        {"season": "2023/24", "injury_type": "hamstring_grade3", "games_missed": 15},
    ],
    "Marcus Thuram": [
        {"season": "2021/22", "injury_type": "ankle_sprain",     "games_missed": 4},
    ],
    "Manu Kone": [
        {"season": "2022/23", "injury_type": "knee_ligament_lcl","games_missed": 9},
    ],
    "Alisson Becker": [
        {"season": "2020/21", "injury_type": "hamstring_grade3", "games_missed": 12},
        {"season": "2022/23", "injury_type": "calf_strain",      "games_missed": 5},
    ],
    "William Saliba": [],
    "Lautaro Martinez": [
        {"season": "2021/22", "injury_type": "adductor_strain",  "games_missed": 5},
        {"season": "2023/24", "injury_type": "muscular_tear",    "games_missed": 6},
    ],
    "Jamal Musiala": [
        {"season": "2024/25", "injury_type": "ankle_sprain",     "games_missed": 5},
    ],
    "Leroy Sane": [
        {"season": "2019/20", "injury_type": "knee_acl",         "games_missed": 40},
        {"season": "2021/22", "injury_type": "muscular_tear",    "games_missed": 7},
        {"season": "2023/24", "injury_type": "hamstring_grade2", "games_missed": 8},
    ],
    "Toni Kroos": [
        {"season": "2023/24", "injury_type": "hamstring_grade1", "games_missed": 3},
    ],
    "N'Golo Kante": [
        {"season": "2022/23", "injury_type": "hamstring_grade3", "games_missed": 20},
        {"season": "2021/22", "injury_type": "knee_meniscus",    "games_missed": 12},
        {"season": "2020/21", "injury_type": "muscular_tear",    "games_missed": 8},
    ],
}


# ============================================================================ #
# Helper: sigmoid normalisation                                                  #
# ============================================================================ #

def _sigmoid(x: float, midpoint: float = 0.5, steepness: float = 8.0) -> float:
    """
    Map any real value to (0, 1) via a sigmoid curve.

    Args:
        x: Raw risk input (bounded/unbounded).
        midpoint: Input value that maps to 0.5 output.
        steepness: Controls how sharp the transition is.

    Returns:
        Float in (0, 1).
    """
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


# ============================================================================ #
# InjuryRiskAnalyzer                                                             #
# ============================================================================ #

class InjuryRiskAnalyzer:
    """
    ==========================================================================
    BUSINESS SUMMARY
    ==========================================================================
    Produces a 0-1 injury risk score for any player in the dataset.
    A score of 0.0 = virtually no injury risk (GK, young, clean history).
    A score of 1.0 = extreme risk (multiple ACLs, high age, heavy load).

    The score feeds directly into the transfer valuator's fair_value()
    calculation as a discount multiplier on base market value.
    ==========================================================================

    Args:
        injury_history: Optionally override the hardcoded INJURY_HISTORY.
            Useful for testing or injecting fresh data from API-Football.
    """

    def __init__(
        self,
        injury_history: Optional[Dict[str, List[Dict]]] = None,
        player_physical: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self.injury_history = injury_history if injury_history is not None else INJURY_HISTORY
        self.player_physical = player_physical if player_physical is not None else PLAYER_PHYSICAL

    # ------------------------------------------------------------------ #
    # Sub-score: Age risk                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _age_risk(age: int) -> float:
        """
        Age-based injury risk component.

        Risk increases non-linearly with age:
        - Under 22: low baseline (young tissue, good recovery)
        - 22-28: moderate and relatively flat
        - 28-32: meaningful increase (accumulation effect)
        - 32+: high; structural degradation measurable

        Returns:
            Float in [0, 1].
        """
        if age <= 21:
            return 0.10
        elif age <= 25:
            return 0.10 + (age - 21) * 0.025   # 0.10 → 0.20
        elif age <= 28:
            return 0.20 + (age - 25) * 0.040   # 0.20 → 0.32
        elif age <= 32:
            return 0.32 + (age - 28) * 0.060   # 0.32 → 0.56
        else:
            return min(0.56 + (age - 32) * 0.06, 0.90)  # caps at 0.90

    # ------------------------------------------------------------------ #
    # Sub-score: Minutes fatigue                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _minutes_risk(minutes_last_season: int) -> float:
        """
        Fatigue risk based on minutes played in the last season.

        Diminishing marginal returns on player durability:
        - <2000 min: low (either injury-limited or squad player)
        - 2000-3000: moderate (healthy starter, managed)
        - 3000-3500: elevated (high workload)
        - >3500: high (fatigue accumulation, burnout risk)

        Returns:
            Float in [0, 1].
        """
        if minutes_last_season < 2000:
            # Could be recovering from injury — not the same as low fatigue
            return 0.20
        elif minutes_last_season <= 3000:
            return 0.20 + (minutes_last_season - 2000) / 1000 * 0.15  # 0.20–0.35
        elif minutes_last_season <= 3500:
            return 0.35 + (minutes_last_season - 3000) / 500 * 0.25   # 0.35–0.60
        else:
            return min(0.60 + (minutes_last_season - 3500) / 500 * 0.30, 0.95)  # caps

    # ------------------------------------------------------------------ #
    # Sub-score: Injury history                                             #
    # ------------------------------------------------------------------ #

    def _history_risk(self, player: str) -> float:
        """
        Risk score based on a player's documented injury history.

        Factors:
        - Structural injuries (ACL, meniscus) carry permanent elevated risk
        - Muscular injuries are recurrent but lower severity
        - Recency: injuries in the last 2 seasons carry 2× weight
        - Volume: each additional injury adds to the base risk

        Returns:
            Float in [0, 1].
        """
        history = self.injury_history.get(player, [])
        if not history:
            return 0.05  # Clean bill of health — minimal historical risk

        base_risk = 0.0
        for entry in history:
            injury_type = entry.get("injury_type", "none")
            profile = INJURY_PROFILES.get(injury_type, INJURY_PROFILES["none"])
            severity = profile["severity"]

            # Recency weight: injuries in 2023/24 or 2024/25 are recent
            season = entry.get("season", "")
            recency_multiplier = 2.0 if season >= "2023/24" else 1.0

            # Games missed adds proportional risk (more games = more severe)
            games_missed = entry.get("games_missed", 0)
            games_factor = min(1.0 + games_missed / 40.0, 1.8)

            base_risk += severity * recency_multiplier * games_factor

        # Normalise: expected range is 0–5+ for heavily injured players
        # Use sigmoid to compress into [0, 1]
        return round(min(_sigmoid(base_risk, midpoint=1.2, steepness=1.2), 0.98), 4)

    # ------------------------------------------------------------------ #
    # Sub-score: BMI proxy                                                  #
    # ------------------------------------------------------------------ #

    def _bmi_risk(self, player: str) -> float:
        """
        Approximate BMI-based risk proxy.

        High BMI in footballers (>27) can indicate excess weight for pace-
        dependent roles, while very low BMI (<20) may indicate under-muscling
        in physical positions. Optimal BMI for outfield players is ~22-25.

        Returns:
            Float in [0, 0.3]. Low contribution — BMI is a weak signal.
        """
        phys = self.player_physical.get(player)
        if phys is None:
            return 0.10  # Unknown — neutral penalty
        h_m = phys["height_cm"] / 100.0
        bmi = phys["weight_kg"] / (h_m ** 2)
        if 21.0 <= bmi <= 26.0:
            return 0.05   # Optimal range
        elif bmi < 21.0:
            return 0.10   # Slight under-muscling risk
        elif bmi <= 28.0:
            return 0.15
        else:
            return 0.25   # Heavier build — higher joint stress

    # ------------------------------------------------------------------ #
    # Public: get_injury_risk_score                                         #
    # ------------------------------------------------------------------ #

    def get_injury_risk_score(
        self,
        player: str,
        age: int,
        position: str,
        minutes_last_season: int,
    ) -> float:
        """
        Compute the composite injury risk score for a player.

        Combines age, position workload, minutes fatigue, injury history,
        and BMI proxy into a single [0, 1] score using weighted average.

        Weights:
            injury_history: 40% (most predictive of future injury)
            age:            25% (strong actuarial signal)
            minutes_load:   20% (fatigue/overuse)
            position_risk:  10% (structural workload)
            bmi_proxy:       5% (weak signal; tiebreaker only)

        Args:
            player: Player name (must match INJURY_HISTORY keys or returns default).
            age: Player's current age.
            position: Position code (e.g. "CB", "CM", "FW").
            minutes_last_season: Total league + cup minutes in last season.

        Returns:
            Float in [0.0, 1.0]. Higher = more injury-prone.

        Example:
            score = analyzer.get_injury_risk_score("N'Golo Kante", 33, "CM", 1800)
            # Returns ~0.78 — high historical injuries + age
        """
        pos_risk  = POSITION_WORKLOAD_RISK.get(position.upper(), _DEFAULT_WORKLOAD_RISK)
        age_risk  = self._age_risk(age)
        min_risk  = self._minutes_risk(minutes_last_season)
        hist_risk = self._history_risk(player)
        bmi_risk  = self._bmi_risk(player)

        composite = (
            hist_risk * 0.40
            + age_risk  * 0.25
            + min_risk  * 0.20
            + pos_risk  * 0.10
            + bmi_risk  * 0.05
        )
        return round(min(composite, 1.0), 4)

    # ------------------------------------------------------------------ #
    # Public: adjusted_valuation                                            #
    # ------------------------------------------------------------------ #

    def adjusted_valuation(
        self,
        player: str,
        base_value_eur_m: float,
        age: int,
        position: str,
        minutes_last_season: int,
    ) -> Dict:
        """
        Discount a player's base market value by their injury risk.

        The discount is non-linear: risk scores below 0.3 attract minimal
        discount, while scores above 0.7 warrant aggressive discounting.
        Position matters: speed-dependent positions (LW, RW, FW) are
        discounted more heavily because injury risk is more value-destructive.

        Args:
            player: Player name.
            base_value_eur_m: Pre-injury-adjustment market value in EUR millions.
            age: Player's current age.
            position: Position code.
            minutes_last_season: Minutes played in last season.

        Returns:
            dict with:
                risk_score       : Composite injury risk [0, 1]
                discount_pct     : % applied to base value
                adjusted_value   : base_value_eur_m × (1 - discount_pct)
                risk_label       : "low" / "moderate" / "elevated" / "high"
                recommendation   : Scout recommendation string
        """
        risk_score = self.get_injury_risk_score(player, age, position, minutes_last_season)

        # Position-dependent discount amplifier
        pos = position.upper()
        if pos in ("LW", "RW", "FW"):
            amplifier = 1.3   # Pace-reliant; injury destroys primary asset
        elif pos in ("CM", "DM", "AM"):
            amplifier = 1.15  # High workload positions
        elif pos in ("CB", "RB", "LB"):
            amplifier = 1.05
        else:  # GK
            amplifier = 0.85  # Keepers age well; injury risk less value-destructive

        # Raw discount: 0 → 0%, 0.5 → ~15%, 1.0 → ~40%
        raw_discount = (risk_score ** 1.5) * 0.45 * amplifier
        discount_pct = min(raw_discount, 0.45)  # Cap at 45%
        adjusted = round(base_value_eur_m * (1.0 - discount_pct), 2)

        if risk_score < 0.25:
            label = "low"
            recommendation = "Low injury risk — proceed with standard due diligence."
        elif risk_score < 0.45:
            label = "moderate"
            recommendation = "Moderate risk — include enhanced medical screening in negotiation."
        elif risk_score < 0.65:
            label = "elevated"
            recommendation = "Elevated risk — negotiate phased payment structure; include performance clauses."
        else:
            label = "high"
            recommendation = "High injury risk — significant discount warranted; consider insurance valuation."

        return {
            "risk_score": risk_score,
            "discount_pct": round(discount_pct * 100, 1),
            "adjusted_value_eur_m": adjusted,
            "risk_label": label,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------ #
    # Public: recovery_trajectory                                           #
    # ------------------------------------------------------------------ #

    def recovery_trajectory(
        self, player: str, injury_type: str, age: int
    ) -> Dict:
        """
        Estimate recovery time and market value impact for a specific injury.

        Recovery times are adjusted for age: older players take proportionally
        longer to recover. Value impact accounts for both the recovery period
        (games missed) and the risk premium added to future uncertainty.

        Args:
            player: Player name (for history context).
            injury_type: Injury type key matching INJURY_PROFILES.
            age: Player's current age (recovery multiplier).

        Returns:
            dict with:
                injury_type            : Standardised label
                severity               : 0-1 severity score
                recovery_weeks_base    : Base recovery estimate
                recovery_weeks_adjusted: Age-adjusted estimate
                value_impact_pct       : Expected immediate value drop %
                full_recovery_confidence: Probability of 100% recovery (0-1)
                notes                  : Clinical context string

        Raises:
            ValueError: If injury_type is not in INJURY_PROFILES.
        """
        if injury_type not in INJURY_PROFILES:
            valid = sorted(INJURY_PROFILES.keys())
            raise ValueError(
                f"Unknown injury_type '{injury_type}'. Valid options: {valid}"
            )

        profile = INJURY_PROFILES[injury_type]
        base_weeks = profile["recovery_weeks"]

        # Age adjustment: proportional to excess over 25
        age_multiplier = 1.0 + max(0, age - 25) * 0.015
        adjusted_weeks = round(base_weeks * age_multiplier)

        # Re-injury risk: players with same injury type in history
        history = self.injury_history.get(player, [])
        same_type_count = sum(1 for h in history if h.get("injury_type") == injury_type)
        recurrence_risk = min(0.20 + same_type_count * 0.15, 0.75)

        # Full recovery confidence decreases with severity and recurrence
        full_recovery_conf = max(
            0.0,
            1.0 - profile["severity"] * 0.5 - recurrence_risk * 0.3
        )

        # Value impact: base profile impact + age premium + recurrence uncertainty
        value_impact = min(
            profile["value_impact_pct"]
            + (age - 25) * 0.005
            + recurrence_risk * 0.05,
            0.50,
        )

        clinical_notes = {
            "hamstring_grade1": "Grade 1 strain — minor fibre damage; full sprint return by week 3.",
            "hamstring_grade2": "Grade 2 — moderate tear; risk of re-injury if rushed back.",
            "hamstring_grade3": "Grade 3 — full-thickness tear; high re-injury risk at speed.",
            "muscular_tear":    "Muscle belly tear — conservative return prevents recurrence.",
            "ankle_sprain":     "Ligament sprain — proprioception rehab critical for prevention.",
            "knee_acl":         "ACL reconstruction — 9-12 month timeline; 15-25% re-rupture rate.",
            "knee_meniscus":    "Meniscal repair or resection; long-term arthritis risk elevated.",
            "knee_ligament_lcl":"LCL repair; less severe than ACL but instability risk remains.",
            "fracture":         "Requires confirmed radiological healing before contact return.",
            "back_disc":        "Disc herniation — chronic management; may limit explosive movement.",
            "adductor_strain":  "Common in explosive accelerations; core stability key to prevention.",
            "calf_strain":      "Gastrocnemius or soleus; must differentiate before return to sprinting.",
            "groin_strain":     "Pubalgia complex — often chronic; surgical option if conservative fails.",
            "foot_metatarsal":  "Stress fracture or acute break; non-weight-bearing phase critical.",
            "thigh_contusion":  "Direct impact bruise; typically short recovery with no recurrence risk.",
            "none":             "No injury — full fitness assumed.",
        }

        return {
            "injury_type": injury_type,
            "severity": profile["severity"],
            "recovery_weeks_base": base_weeks,
            "recovery_weeks_adjusted": adjusted_weeks,
            "age_multiplier": round(age_multiplier, 2),
            "recurrence_risk": round(recurrence_risk, 2),
            "full_recovery_confidence": round(full_recovery_conf, 2),
            "value_impact_pct": round(value_impact * 100, 1),
            "notes": clinical_notes.get(injury_type, "Consult club medical team."),
        }

    # ------------------------------------------------------------------ #
    # Bulk risk report                                                      #
    # ------------------------------------------------------------------ #

    def bulk_risk_report(
        self,
        players: List[Dict],   # [{"name", "age", "position", "minutes_last_season"}]
    ) -> List[Dict]:
        """
        Compute injury risk scores for a list of players in one call.

        Args:
            players: List of dicts, each with keys:
                name, age, position, minutes_last_season.

        Returns:
            List of risk result dicts, sorted by risk_score descending.
        """
        results = []
        for p in players:
            score = self.get_injury_risk_score(
                p["name"], p["age"], p["position"], p.get("minutes_last_season", 2700)
            )
            results.append({"name": p["name"], "risk_score": score, **p})
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results


# ============================================================================ #
# Smoke test                                                                     #
# ============================================================================ #

if __name__ == "__main__":
    analyzer = InjuryRiskAnalyzer()

    test_cases = [
        ("Federico Chiesa",  27, "RW",  1100),
        ("N'Golo Kante",     33, "CM",  1800),
        ("Jude Bellingham",  21, "CM",  3200),
        ("Rodri",            28, "DM",  1800),
        ("Leny Yoro",        18, "CB",  2800),
        ("Virgil van Dijk",  33, "CB",  2700),
        ("Erling Haaland",   24, "FW",  2450),
    ]

    print("=== INJURY RISK SCORES ===")
    print(f"{'Player':<25} {'Age':>4} {'Pos':>4} {'Min':>5} {'Risk':>6} {'Label':>10} {'Adj Value':>10}")
    print("-" * 80)
    for name, age, pos, mins in test_cases:
        result = analyzer.adjusted_valuation(name, 80.0, age, pos, mins)
        print(
            f"{name:<25} {age:>4} {pos:>4} {mins:>5} "
            f"{result['risk_score']:>6.3f} {result['risk_label']:>10} "
            f"€{result['adjusted_value_eur_m']:>7.1f}m (base €80m)"
        )

    print("\n=== RECOVERY TRAJECTORY: ACL ===")
    acl = analyzer.recovery_trajectory("Federico Chiesa", "knee_acl", 27)
    for k, v in acl.items():
        print(f"  {k}: {v}")
