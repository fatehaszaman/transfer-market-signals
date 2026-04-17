"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
A player with 6 months left on their contract is worth fundamentally less
as a transfer asset — because any interested club knows the player will be
free in January. This module quantifies that urgency and translates it into
actionable signals: urgency scores, free-agent value uplifts, and release
clause opportunity identification.

For non-technical readers: this is the "clock" component of the model. A
player's contract situation creates a ticking countdown that affects both
the selling club's negotiating power and the buying club's opportunity window.
=============================================================================

Developer notes:
- Urgency score: 0 = years remaining, 1 = weeks away from free agency.
- Free agent value: player's negotiating power increases when they can offer
  zero transfer fee (PSG paid zero for Mbappe; Liverpool zero for Thiago).
- Release clause plays: hard-coded clauses are filtered by budget.
- Pydantic integration: ContractSignal schema from data/schemas.py used.
- All dates relative to 2025/26 season start (August 2025).
"""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional

from data.schemas import ContractSignal, TransferValidationError

# Reference date for urgency calculations (start of 2025/26 season)
_REFERENCE_DATE = date(2025, 8, 1)
_MONTHS_IN_YEAR = 12


# ============================================================================ #
# Hardcoded contract dataset — 30 real players                                   #
# ============================================================================ #
# Fields: contract_expires (year), release_clause_eur_m (or None),
#         reported_interest (list of clubs)

CONTRACT_DATA: Dict[str, Dict] = {
    "Kylian Mbappe": {
        "club": "Paris Saint-Germain",
        "contract_expires": 2024,
        "release_clause_eur_m": None,
        "reported_interest": ["Real Madrid"],
        "contract_type": "superstar",
    },
    "Jude Bellingham": {
        "club": "Borussia Dortmund",
        "contract_expires": 2025,
        "release_clause_eur_m": None,
        "reported_interest": ["Real Madrid", "Liverpool", "Manchester City"],
        "contract_type": "standard",
    },
    "Erling Haaland": {
        "club": "Borussia Dortmund",
        "contract_expires": 2024,
        "release_clause_eur_m": 60.0,
        "reported_interest": ["Manchester City", "Real Madrid", "Barcelona"],
        "contract_type": "release_clause",
    },
    "Declan Rice": {
        "club": "West Ham United",
        "contract_expires": 2024,
        "release_clause_eur_m": None,
        "reported_interest": ["Arsenal", "Manchester City", "Chelsea"],
        "contract_type": "standard",
    },
    "Lautaro Martinez": {
        "club": "Inter Milan",
        "contract_expires": 2026,
        "release_clause_eur_m": None,
        "reported_interest": ["Barcelona"],
        "contract_type": "standard",
    },
    "Pedri": {
        "club": "Barcelona",
        "contract_expires": 2026,
        "release_clause_eur_m": 1000.0,  # €1bn buyout — effectively not a clause
        "reported_interest": [],
        "contract_type": "buyout",
    },
    "Leny Yoro": {
        "club": "LOSC Lille",
        "contract_expires": 2025,
        "release_clause_eur_m": None,
        "reported_interest": ["Real Madrid", "Manchester United", "Liverpool"],
        "contract_type": "standard",
    },
    "Joao Neves": {
        "club": "Benfica",
        "contract_expires": 2028,
        "release_clause_eur_m": 120.0,
        "reported_interest": ["Paris Saint-Germain", "Manchester City"],
        "contract_type": "release_clause",
    },
    "Bukayo Saka": {
        "club": "Arsenal",
        "contract_expires": 2027,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Virgil van Dijk": {
        "club": "Liverpool",
        "contract_expires": 2026,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "William Saliba": {
        "club": "Arsenal",
        "contract_expires": 2027,
        "release_clause_eur_m": None,
        "reported_interest": ["Real Madrid"],
        "contract_type": "standard",
    },
    "Ruben Dias": {
        "club": "Manchester City",
        "contract_expires": 2027,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Rodri": {
        "club": "Manchester City",
        "contract_expires": 2027,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Vinicius Junior": {
        "club": "Real Madrid",
        "contract_expires": 2027,
        "release_clause_eur_m": 1000.0,
        "reported_interest": [],
        "contract_type": "buyout",
    },
    "Dean Huijsen": {
        "club": "AFC Bournemouth",
        "contract_expires": 2029,
        "release_clause_eur_m": 50.0,
        "reported_interest": ["Real Madrid", "Arsenal", "Liverpool"],
        "contract_type": "release_clause",
    },
    "Federico Chiesa": {
        "club": "Liverpool",
        "contract_expires": 2028,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Marcus Thuram": {
        "club": "Inter Milan",
        "contract_expires": 2028,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Manu Kone": {
        "club": "Real Madrid",
        "contract_expires": 2029,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Alisson Becker": {
        "club": "Liverpool",
        "contract_expires": 2027,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Manuel Ugarte": {
        "club": "Manchester United",
        "contract_expires": 2029,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Phil Foden": {
        "club": "Manchester City",
        "contract_expires": 2027,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Joao Cancelo": {
        "club": "Manchester City",
        "contract_expires": 2027,
        "release_clause_eur_m": None,
        "reported_interest": ["Barcelona", "Bayern Munich"],
        "contract_type": "standard",
    },
    "Bernardo Silva": {
        "club": "Manchester City",
        "contract_expires": 2026,
        "release_clause_eur_m": None,
        "reported_interest": ["Barcelona", "Paris Saint-Germain"],
        "contract_type": "standard",
    },
    "Jamal Musiala": {
        "club": "Bayern Munich",
        "contract_expires": 2026,
        "release_clause_eur_m": None,
        "reported_interest": ["Real Madrid", "Manchester City", "Liverpool"],
        "contract_type": "standard",
    },
    "N'Golo Kante": {
        "club": "Chelsea",
        "contract_expires": 2024,
        "release_clause_eur_m": None,
        "reported_interest": ["Al Ittihad"],
        "contract_type": "standard",
    },
    "Toni Kroos": {
        "club": "Real Madrid",
        "contract_expires": 2024,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Antoine Griezmann": {
        "club": "Atletico Madrid",
        "contract_expires": 2026,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Lucas Hernandez": {
        "club": "Paris Saint-Germain",
        "contract_expires": 2027,
        "release_clause_eur_m": None,
        "reported_interest": [],
        "contract_type": "standard",
    },
    "Leroy Sane": {
        "club": "Bayern Munich",
        "contract_expires": 2025,
        "release_clause_eur_m": None,
        "reported_interest": ["Manchester City", "Arsenal"],
        "contract_type": "standard",
    },
    "Gavi": {
        "club": "Barcelona",
        "contract_expires": 2026,
        "release_clause_eur_m": 1000.0,
        "reported_interest": [],
        "contract_type": "buyout",
    },
}


# ============================================================================ #
# Validation helpers                                                             #
# ============================================================================ #

_VALID_CONTRACT_TYPES = {"standard", "release_clause", "buyout", "superstar"}


def _validate_player_name(player: str) -> None:
    """
    Pre-processing validation for player name inputs.

    Raises:
        TransferValidationError: If name is empty, too long, or contains
            illegal characters.
    """
    if not isinstance(player, str) or not player.strip():
        raise TransferValidationError(
            field="player_name",
            value=player,
            reason="Player name must be a non-empty string.",
        )
    if len(player) > 80:
        raise TransferValidationError(
            field="player_name",
            value=player,
            reason=f"Player name exceeds 80 characters (got {len(player)}).",
        )


def _validate_budget(budget: float) -> None:
    """Validate a budget value is a positive finite float."""
    if not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise TransferValidationError(
            field="max_budget_eur_m",
            value=budget,
            reason="Budget must be a non-negative finite number in EUR millions.",
        )


# ============================================================================ #
# ContractSignalAnalyzer                                                         #
# ============================================================================ #

class ContractSignalAnalyzer:
    """
    ==========================================================================
    BUSINESS SUMMARY
    ==========================================================================
    Computes contract urgency signals for football players. The urgency score
    drives valuation discounts and transfer probability — a player in the last
    6 months of their deal is significantly more likely to move than one who
    has 4 years remaining.

    Outputs:
    - urgency_score [0, 1]: higher = more likely to transfer due to contract
    - free_agent_value: adjusted market value accounting for zero-fee negotiating
    - release_clause_plays: players with affordable activated clauses
    - ContractSignal schema objects for integration with the valuation model
    ==========================================================================

    Args:
        contract_data: Override the hardcoded CONTRACT_DATA for testing.
        reference_date: Override the reference date (useful for backtests).
    """

    def __init__(
        self,
        contract_data: Optional[Dict[str, Dict]] = None,
        reference_date: Optional[date] = None,
    ) -> None:
        self.contracts = contract_data if contract_data is not None else CONTRACT_DATA
        self.ref_date = reference_date if reference_date is not None else _REFERENCE_DATE

    # ------------------------------------------------------------------ #
    # Internal: months remaining                                            #
    # ------------------------------------------------------------------ #

    def _months_remaining(self, contract_expires_year: int) -> int:
        """
        Compute calendar months remaining until contract expiry.

        Contracts are assumed to expire at June 30 of the given year.
        Returns max(0, computed_months) — negative values are floored.

        Args:
            contract_expires_year: Year the contract expires.

        Returns:
            Integer months remaining (0 = already expired).
        """
        expiry = date(contract_expires_year, 6, 30)
        delta_days = (expiry - self.ref_date).days
        months = int(delta_days / 30.44)   # Average days per month
        return max(0, months)

    # ------------------------------------------------------------------ #
    # Public: get_urgency_score                                             #
    # ------------------------------------------------------------------ #

    def get_urgency_score(self, player: str) -> float:
        """
        Compute contract urgency score for a player.

        Urgency represents how much the contract situation is pushing toward
        a transfer this window:
          - 0–6 months: very high urgency (0.85–1.0) — sell now or lose free
          - 6–12 months: high urgency (0.60–0.85) — final window at value
          - 12–18 months: moderate (0.35–0.60) — clubs circling
          - 18–30 months: low urgency (0.15–0.35) — not a pressing concern
          - 30+ months: minimal (0.0–0.15) — long-term security

        Formula: urgency = sigmoid(-months_remaining, midpoint=-18, steepness=0.10)
        This produces a decreasing S-curve where 0 months → near 1.0
        and 36+ months → near 0.0.

        Args:
            player: Player name. Must be in contract dataset or raises TransferValidationError.

        Returns:
            Float in [0.0, 1.0].

        Raises:
            TransferValidationError: If player name is invalid.
            KeyError: If player not found in contract dataset.
        """
        _validate_player_name(player)

        if player not in self.contracts:
            raise KeyError(
                f"Player '{player}' not in contract dataset. "
                f"Available: {sorted(self.contracts.keys())}"
            )

        data = self.contracts[player]
        months = self._months_remaining(data["contract_expires"])

        # Sigmoid: centred at 18 months → 0.5 urgency
        # Steepness 0.12: 6 months → ~0.90, 36 months → ~0.10
        urgency = 1.0 / (1.0 + math.exp(0.12 * (months - 18)))
        return round(min(1.0, max(0.0, urgency)), 4)

    # ------------------------------------------------------------------ #
    # Public: get_free_agent_value                                          #
    # ------------------------------------------------------------------ #

    def get_free_agent_value(self, player: str, current_market_value_eur_m: float) -> float:
        """
        Compute adjusted market value for a player approaching free agency.

        When a player can move for free, two opposing forces are at play:
        1. The receiving club pays zero transfer fee → higher wage budget available
           → player commands a higher wage (estimated 15–30% of saved fee)
        2. The selling club loses all transfer value → pressured to sell early or accept free
        3. Player's negotiating leverage increases substantially when months < 12

        This method returns the PLAYER's effective negotiating value (what they
        can extract in wages + signing bonus from a free transfer), expressed as
        a market-value-equivalent in EUR millions.

        Args:
            player: Player name.
            current_market_value_eur_m: Current market valuation in EUR millions.

        Returns:
            Adjusted value in EUR millions. Will be >= current_market_value_eur_m
            when approaching free agency (player captures some of the fee saving).

        Raises:
            TransferValidationError: On invalid inputs.
            KeyError: If player not in dataset.
        """
        _validate_player_name(player)

        if player not in self.contracts:
            raise KeyError(f"Player '{player}' not in contract dataset.")

        months = self._months_remaining(self.contracts[player]["contract_expires"])

        if months > 18:
            # No free-agent premium — normal market value
            return round(current_market_value_eur_m, 2)

        # Free agent premium: % of current value the player captures as wage uplift
        # Modelled as linearly increasing as months decrease to zero
        # At 0 months: player captures ~25% of market value as wage premium
        # At 18 months: no premium (negotiations just starting)
        uplift_rate = max(0.0, (18 - months) / 18.0) * 0.25
        adjusted = current_market_value_eur_m * (1.0 + uplift_rate)
        return round(adjusted, 2)

    # ------------------------------------------------------------------ #
    # Public: get_release_clause_plays                                      #
    # ------------------------------------------------------------------ #

    def get_release_clause_plays(self, max_budget_eur_m: float) -> List[Dict]:
        """
        Return players whose release clauses are within the specified budget.

        Excludes nominal buyout clauses (> €500m — these are not real triggers).

        Args:
            max_budget_eur_m: Maximum transfer budget in EUR millions.

        Returns:
            List of dicts sorted by clause value ascending (cheapest first).
            Each dict: player, club, release_clause_eur_m, contract_expires,
            months_remaining, urgency_score, reported_interest.

        Raises:
            TransferValidationError: If budget is invalid.
        """
        _validate_budget(max_budget_eur_m)

        plays = []
        for player_name, data in self.contracts.items():
            clause = data.get("release_clause_eur_m")
            if clause is None or clause > 500.0:  # Exclude nominal buyouts
                continue
            if clause <= max_budget_eur_m:
                months = self._months_remaining(data["contract_expires"])
                try:
                    urgency = self.get_urgency_score(player_name)
                except KeyError:
                    urgency = 0.5

                plays.append({
                    "player": player_name,
                    "club": data["club"],
                    "release_clause_eur_m": clause,
                    "contract_expires": data["contract_expires"],
                    "months_remaining": months,
                    "urgency_score": urgency,
                    "reported_interest": data["reported_interest"],
                    "contract_type": data.get("contract_type", "unknown"),
                })

        plays.sort(key=lambda x: x["release_clause_eur_m"])
        return plays

    # ------------------------------------------------------------------ #
    # Public: get_contract_signal (Pydantic output)                         #
    # ------------------------------------------------------------------ #

    def get_contract_signal(
        self,
        player: str,
        current_market_value_eur_m: float,
    ) -> ContractSignal:
        """
        Build a validated ContractSignal Pydantic object for a player.

        Integrates urgency score, free-agent value uplift, release clause data,
        and reported interest clubs into the typed schema used by TransferValuator.

        Args:
            player: Player name.
            current_market_value_eur_m: Current market value in EUR millions.

        Returns:
            ContractSignal Pydantic model instance.

        Raises:
            TransferValidationError: If inputs are invalid.
            KeyError: If player not found.
        """
        _validate_player_name(player)

        if player not in self.contracts:
            raise KeyError(f"Player '{player}' not in contract dataset.")

        data = self.contracts[player]
        months = self._months_remaining(data["contract_expires"])
        urgency = self.get_urgency_score(player)
        fa_value = self.get_free_agent_value(player, current_market_value_eur_m)
        uplift_pct = (fa_value - current_market_value_eur_m) / current_market_value_eur_m if current_market_value_eur_m > 0 else 0.0
        clause = data.get("release_clause_eur_m")
        real_clause = clause if (clause is not None and clause <= 500.0) else None

        return ContractSignal(
            player_name=player,
            contract_expires_year=data["contract_expires"],
            months_remaining=months,
            urgency_score=urgency,
            has_release_clause=real_clause is not None,
            release_clause_eur_m=real_clause,
            free_agent_value_uplift_pct=round(uplift_pct, 4),
            reported_interest_clubs=data.get("reported_interest", []),
        )

    # ------------------------------------------------------------------ #
    # Public: batch_urgency_scores                                          #
    # ------------------------------------------------------------------ #

    def batch_urgency_scores(self) -> List[Dict]:
        """
        Compute urgency scores for all players in the dataset.

        Returns:
            List of dicts sorted by urgency_score descending (most urgent first).
        """
        results = []
        for player_name, data in self.contracts.items():
            months = self._months_remaining(data["contract_expires"])
            urgency = self.get_urgency_score(player_name)
            results.append({
                "player": player_name,
                "club": data["club"],
                "contract_expires": data["contract_expires"],
                "months_remaining": months,
                "urgency_score": urgency,
                "has_release_clause": (
                    data.get("release_clause_eur_m") is not None
                    and data["release_clause_eur_m"] <= 500.0
                ),
                "release_clause_eur_m": (
                    data["release_clause_eur_m"]
                    if data.get("release_clause_eur_m") and data["release_clause_eur_m"] <= 500.0
                    else None
                ),
            })
        results.sort(key=lambda x: x["urgency_score"], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    # Public: players_by_window                                             #
    # ------------------------------------------------------------------ #

    def players_by_window(self, window: str = "summer_2025") -> List[Dict]:
        """
        Return players most likely to be in transfer play for a specific window.

        Filters to players with > 0.4 urgency score and sorts by urgency.

        Args:
            window: Transfer window identifier (informational label).

        Returns:
            List of dicts with urgency and club info.
        """
        all_scores = self.batch_urgency_scores()
        return [p for p in all_scores if p["urgency_score"] >= 0.40]


# ============================================================================ #
# Smoke test                                                                     #
# ============================================================================ #

if __name__ == "__main__":
    analyzer = ContractSignalAnalyzer()

    print("=== CONTRACT URGENCY SCORES (All Players) ===")
    for row in analyzer.batch_urgency_scores()[:10]:
        print(
            f"  {row['player']:<25} | expires={row['contract_expires']} | "
            f"months={row['months_remaining']:>3} | urgency={row['urgency_score']:.3f}"
            + (f" | clause=€{row['release_clause_eur_m']}m" if row["has_release_clause"] else "")
        )

    print("\n=== RELEASE CLAUSE PLAYS (€120m budget) ===")
    for p in analyzer.get_release_clause_plays(120.0):
        print(f"  {p['player']:<22} | clause=€{p['release_clause_eur_m']}m | urgency={p['urgency_score']:.3f}")

    print("\n=== FREE AGENT VALUE: Erling Haaland (historical) ===")
    val = analyzer.get_free_agent_value("Erling Haaland", 150.0)
    print(f"  Adjusted value: €{val}m (base €150m)")

    print("\n=== PYDANTIC CONTRACT SIGNAL: Leny Yoro ===")
    sig = analyzer.get_contract_signal("Leny Yoro", 60.0)
    print(sig.model_dump())
