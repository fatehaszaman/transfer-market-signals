"""
Age-Value Curve Modeling for Football Players.

Models the typical career value arc by position, computes depreciation rates,
and identifies undervalued players relative to their age/position/rating peers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# Position peak and curve constants                                            #
# --------------------------------------------------------------------------- #
# Each position has (peak_start, peak_end, pre_peak_growth_rate, post_peak_drop_rate)
# Rates are annual percentage changes as decimals (0.15 = 15% growth per year)
POSITION_CURVES: Dict[str, Dict] = {
    "GK": {
        "peak_start": 27,
        "peak_end": 31,
        "pre_peak_growth": 0.12,   # Value grows ~12% per year before peak
        "post_peak_drop": 0.08,    # Value drops ~8% per year after peak
        "cliff_age": 35,           # Steeper depreciation beyond this age
        "cliff_drop": 0.18,
    },
    "CB": {
        "peak_start": 26,
        "peak_end": 30,
        "pre_peak_growth": 0.14,
        "post_peak_drop": 0.10,
        "cliff_age": 34,
        "cliff_drop": 0.20,
    },
    "RB": {
        "peak_start": 24,
        "peak_end": 28,
        "pre_peak_growth": 0.15,
        "post_peak_drop": 0.11,
        "cliff_age": 32,
        "cliff_drop": 0.22,
    },
    "LB": {
        "peak_start": 24,
        "peak_end": 28,
        "pre_peak_growth": 0.15,
        "post_peak_drop": 0.11,
        "cliff_age": 32,
        "cliff_drop": 0.22,
    },
    "CM": {
        "peak_start": 25,
        "peak_end": 29,
        "pre_peak_growth": 0.16,
        "post_peak_drop": 0.12,
        "cliff_age": 33,
        "cliff_drop": 0.24,
    },
    "DM": {
        "peak_start": 25,
        "peak_end": 30,
        "pre_peak_growth": 0.13,
        "post_peak_drop": 0.09,
        "cliff_age": 34,
        "cliff_drop": 0.18,
    },
    "AM": {
        "peak_start": 24,
        "peak_end": 28,
        "pre_peak_growth": 0.17,
        "post_peak_drop": 0.13,
        "cliff_age": 32,
        "cliff_drop": 0.25,
    },
    "FW": {
        "peak_start": 24,
        "peak_end": 28,
        "pre_peak_growth": 0.18,
        "post_peak_drop": 0.14,
        "cliff_age": 31,
        "cliff_drop": 0.27,
    },
    "LW": {
        "peak_start": 23,
        "peak_end": 27,
        "pre_peak_growth": 0.18,
        "post_peak_drop": 0.15,
        "cliff_age": 31,
        "cliff_drop": 0.28,
    },
    "RW": {
        "peak_start": 23,
        "peak_end": 27,
        "pre_peak_growth": 0.18,
        "post_peak_drop": 0.15,
        "cliff_age": 31,
        "cliff_drop": 0.28,
    },
}

# Fallback for unknown positions
_DEFAULT_CURVE = {
    "peak_start": 25,
    "peak_end": 29,
    "pre_peak_growth": 0.15,
    "post_peak_drop": 0.12,
    "cliff_age": 33,
    "cliff_drop": 0.24,
}

# Rating multipliers: a highly rated player depreciates more slowly
# and grows faster because elite demand sustains value.
RATING_MULTIPLIERS: List[Tuple[float, float]] = [
    # (min_rating, multiplier on growth/slower_drop)
    (9.0, 1.35),
    (8.0, 1.20),
    (7.5, 1.10),
    (7.0, 1.00),
    (6.5, 0.90),
    (0.0, 0.80),
]


def _get_curve(position: str) -> Dict:
    """Return the curve config for a given position string, case-insensitive."""
    return POSITION_CURVES.get(position.upper(), _DEFAULT_CURVE)


def _rating_multiplier(rating: float) -> float:
    """Return the value multiplier for a player based on their overall rating."""
    for min_r, mult in RATING_MULTIPLIERS:
        if rating >= min_r:
            return mult
    return 0.80


class AgeValueCurve:
    """
    Models the career value arc for football players by position.

    The career arc has three phases:
    - Pre-peak: value grows as the player develops
    - Peak window: value is relatively stable (small gains/losses)
    - Post-peak: value depreciates, accelerating past a cliff age

    All monetary values are in EUR millions.
    """

    def __init__(self) -> None:
        self._curves = POSITION_CURVES

    # ------------------------------------------------------------------ #
    # Core single-year rate                                                #
    # ------------------------------------------------------------------ #
    def annual_growth_rate(self, age: int, position: str, rating: float = 7.5) -> float:
        """
        Return the expected annual value change rate for a player at a given age.

        Positive = value growth, Negative = value depreciation.
        Rate is a decimal (0.12 = +12%, -0.10 = -10%).
        """
        curve = _get_curve(position)
        mult = _rating_multiplier(rating)

        if age < curve["peak_start"]:
            # Pre-peak: growing phase
            # The further from peak the faster the growth (young players develop quickly)
            years_to_peak = curve["peak_start"] - age
            boost = min(0.05, years_to_peak * 0.01)  # extra boost for very young players
            return (curve["pre_peak_growth"] + boost) * mult

        elif curve["peak_start"] <= age <= curve["peak_end"]:
            # Peak window: minimal movement (slight appreciation or flat)
            # In mid-peak there's a tiny plateau; at the edges there's slight drag
            midpoint = (curve["peak_start"] + curve["peak_end"]) / 2
            distance_from_mid = abs(age - midpoint)
            flat_rate = 0.02 - distance_from_mid * 0.005
            return flat_rate * mult

        else:
            # Post-peak: depreciation phase
            years_past_peak = age - curve["peak_end"]
            if age >= curve["cliff_age"]:
                base_drop = curve["cliff_drop"]
            else:
                # Accelerating depreciation: each year past peak adds 0.5%
                base_drop = curve["post_peak_drop"] + (years_past_peak * 0.005)
            # High-rated players lose value more slowly (elite demand)
            return -base_drop / mult  # negative = depreciation

    # ------------------------------------------------------------------ #
    # Depreciation rate (public convenience method)                        #
    # ------------------------------------------------------------------ #
    def depreciation_rate(self, age: int, position: str) -> float:
        """
        Annual percentage value drop post-peak for a player.

        Returns a positive float representing the % drop (e.g. 0.12 = 12% drop).
        Returns 0.0 if the player is not yet in their post-peak phase.
        """
        curve = _get_curve(position)
        if age <= curve["peak_end"]:
            return 0.0
        years_past_peak = age - curve["peak_end"]
        if age >= curve["cliff_age"]:
            return curve["cliff_drop"]
        return curve["post_peak_drop"] + (years_past_peak * 0.005)

    # ------------------------------------------------------------------ #
    # 5-year trajectory                                                    #
    # ------------------------------------------------------------------ #
    def expected_value_at_age(
        self,
        current_value: float,
        current_age: int,
        position: str,
        rating: float = 7.5,
    ) -> Dict:
        """
        Project the player's market value over the next 5 years.

        Args:
            current_value: Current market value in EUR millions.
            current_age: Player's current age.
            position: Player's position code (e.g. "CB", "FW", "CM").
            rating: Overall performance rating (1-10 scale, default 7.5).

        Returns:
            Dictionary with keys:
                - "trajectory": list of dicts [{age, value, rate}] for next 5 years
                - "peak_value": estimated peak value in EUR millions
                - "phase": current career phase string
                - "total_5yr_return": % change from now to year 5
        """
        curve = _get_curve(position)
        trajectory = []
        value = current_value

        for offset in range(1, 6):
            age = current_age + offset
            rate = self.annual_growth_rate(age - 1, position, rating)  # rate entering this year
            value = max(value * (1 + rate), 0.5)  # floor at €0.5m
            trajectory.append(
                {
                    "age": age,
                    "value_eur_m": round(value, 2),
                    "annual_rate_pct": round(rate * 100, 1),
                }
            )

        # Peak value: simulate forward from current age to find maximum
        sim_value = current_value
        peak_value = current_value
        for a in range(current_age, current_age + 15):
            r = self.annual_growth_rate(a, position, rating)
            sim_value = max(sim_value * (1 + r), 0.5)
            if sim_value > peak_value:
                peak_value = sim_value
            elif a > curve["peak_end"]:
                break  # Past peak and declining; stop looking

        # Phase
        if current_age < curve["peak_start"]:
            phase = "pre-peak"
        elif current_age <= curve["peak_end"]:
            phase = "peak"
        elif current_age < curve["cliff_age"]:
            phase = "post-peak"
        else:
            phase = "veteran"

        total_5yr_return = (trajectory[-1]["value_eur_m"] - current_value) / current_value * 100

        return {
            "trajectory": trajectory,
            "peak_value": round(peak_value, 2),
            "phase": phase,
            "total_5yr_return_pct": round(total_5yr_return, 1),
        }

    # ------------------------------------------------------------------ #
    # Undervaluation scorer                                                #
    # ------------------------------------------------------------------ #
    def identify_undervalued(
        self,
        player_age: int,
        current_value_eur_m: float,
        position: str,
        rating: float,
    ) -> float:
        """
        Compute an undervaluation score for a player.

        The score represents how much cheaper (or more expensive) the player is
        compared to a theoretical fair market price based on age, position, and rating.

        A positive score means the player is undervalued (good buy).
        A negative score means the player is overvalued.

        The reference fair value is computed by:
        1. Starting with a baseline of €50m for a 7.5-rated prime player.
        2. Adjusting for position, age (career phase), and rating.

        Args:
            player_age: Player's current age.
            current_value_eur_m: Current market valuation in EUR millions.
            position: Position code string.
            rating: Performance rating (1-10).

        Returns:
            Undervaluation score in EUR millions (positive = undervalued).
        """
        curve = _get_curve(position)

        # Compute a rating-adjusted baseline fair value
        # The model assumes a 7.5-rated player at their peak is worth ~€50m
        # (baseline for established top-5-league player)
        baseline_peak_value = 50.0
        rating_factor = (_rating_multiplier(rating) ** 2)  # quadratic boost for elites
        adjusted_peak = baseline_peak_value * rating_factor

        # Discount adjusted_peak based on career phase
        if player_age < curve["peak_start"]:
            years_to_peak = curve["peak_start"] - player_age
            # Pre-peak player: discount for development risk, but upside multiplies
            phase_discount = 0.60 + (0.08 * (5 - min(years_to_peak, 5)))
            fair_value = adjusted_peak * phase_discount
        elif player_age <= curve["peak_end"]:
            # Prime: near full value
            midpoint = (curve["peak_start"] + curve["peak_end"]) / 2
            distance = abs(player_age - midpoint) / (curve["peak_end"] - curve["peak_start"])
            fair_value = adjusted_peak * (1.0 - distance * 0.08)
        else:
            # Post-peak: discount for years past peak
            years_past = player_age - curve["peak_end"]
            annual_disc = self.depreciation_rate(player_age, position)
            total_disc = (1 - annual_disc) ** years_past
            fair_value = adjusted_peak * total_disc

        undervaluation = fair_value - current_value_eur_m
        return round(undervaluation, 2)

    # ------------------------------------------------------------------ #
    # Helper: position normalisation                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalise_position(raw: str) -> str:
        """Map common position strings to standard codes used in this module."""
        mapping = {
            "goalkeeper": "GK",
            "centre-back": "CB",
            "center-back": "CB",
            "centreback": "CB",
            "right-back": "RB",
            "left-back": "LB",
            "defensive midfield": "DM",
            "central midfield": "CM",
            "attacking midfield": "AM",
            "right winger": "RW",
            "left winger": "LW",
            "centre-forward": "FW",
            "center-forward": "FW",
            "striker": "FW",
        }
        normalised = mapping.get(raw.lower().strip(), raw.upper().strip())
        return normalised if normalised in POSITION_CURVES else raw.upper()


# --------------------------------------------------------------------------- #
# Quick smoke-test when run directly                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    curve_model = AgeValueCurve()

    players = [
        ("Leny Yoro", 18, "CB", 60.0, 7.8),
        ("Virgil van Dijk", 33, "CB", 35.0, 8.2),
        ("Pedri", 22, "CM", 80.0, 8.5),
        ("Erling Haaland", 24, "FW", 180.0, 9.2),
        ("Alisson Becker", 32, "GK", 45.0, 8.3),
    ]

    for name, age, pos, value, rating in players:
        result = curve_model.expected_value_at_age(value, age, pos, rating)
        uv = curve_model.identify_undervalued(age, value, pos, rating)
        print(f"\n{name} ({pos}, age {age}, €{value}m, rating {rating})")
        print(f"  Phase: {result['phase']} | Peak: €{result['peak_value']}m")
        print(f"  5-yr return: {result['total_5yr_return_pct']}%")
        print(f"  Undervaluation score: €{uv}m")
        print(f"  Trajectory: {[f\"{t['age']}:{t['value_eur_m']}\" for t in result['trajectory']]}")
