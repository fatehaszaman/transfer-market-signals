"""
Financial Fair Play (FFP) / Profit & Sustainability Rules (PSR) Headroom Analyzer.

Models the spending headroom of the top 20 European clubs for the 2025/26 season.
Uses realistic estimates of wage bill, revenue, amortized transfer debt, and
UEFA/domestic allowable loss thresholds.

All monetary values are in EUR millions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# 2025/26 Season Club Financial Data                                           #
# --------------------------------------------------------------------------- #
# Fields per club:
#   revenue_eur_m       - Total revenue (matchday + broadcast + commercial)
#   wage_bill_eur_m     - Annual gross wage bill
#   amortised_debt_eur_m - Amortised portion of transfer fee debt (spread over contract length)
#   operating_costs_eur_m - Non-wage operating costs
#   allowable_loss_eur_m  - UEFA FFPR / domestic PSR allowable loss threshold
#   ffp_regime          - "UEFA" | "PL_PSR" | "DNCG" etc.
#   net_spend_3yr_eur_m  - Net transfer spend over rolling 3-year PSR window
#   estimated_headroom_note - Human-readable context note
# --------------------------------------------------------------------------- #
CLUB_FINANCIALS: Dict[str, Dict] = {
    "Manchester City": {
        "revenue_eur_m": 890.0,
        "wage_bill_eur_m": 410.0,
        "amortised_debt_eur_m": 180.0,
        "operating_costs_eur_m": 95.0,
        "allowable_loss_eur_m": 105.0,  # UEFA new FFPR cycle
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 320.0,
        "estimated_headroom_note": "Strong revenue but high wage bill; UEFA charges pending.",
    },
    "Real Madrid": {
        "revenue_eur_m": 1080.0,
        "wage_bill_eur_m": 520.0,
        "amortised_debt_eur_m": 130.0,
        "operating_costs_eur_m": 110.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 180.0,
        "estimated_headroom_note": "Bernabeu renovation debt offset by record revenues.",
    },
    "Barcelona": {
        "revenue_eur_m": 950.0,
        "wage_bill_eur_m": 580.0,
        "amortised_debt_eur_m": 280.0,
        "operating_costs_eur_m": 105.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 410.0,
        "estimated_headroom_note": "La Liga salary cap severely restricted; leveraged heavily.",
    },
    "Paris Saint-Germain": {
        "revenue_eur_m": 780.0,
        "wage_bill_eur_m": 600.0,
        "amortised_debt_eur_m": 150.0,
        "operating_costs_eur_m": 90.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 560.0,
        "estimated_headroom_note": "Post-Mbappe/Neymar exit wage savings partially offset by new signings.",
    },
    "Bayern Munich": {
        "revenue_eur_m": 860.0,
        "wage_bill_eur_m": 420.0,
        "amortised_debt_eur_m": 95.0,
        "operating_costs_eur_m": 88.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 200.0,
        "estimated_headroom_note": "Conservative spend, strong fundamentals, comfortable position.",
    },
    "Liverpool": {
        "revenue_eur_m": 710.0,
        "wage_bill_eur_m": 350.0,
        "amortised_debt_eur_m": 110.0,
        "operating_costs_eur_m": 78.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "PL_PSR",
        "net_spend_3yr_eur_m": 240.0,
        "estimated_headroom_note": "Healthy PSR position; Klopp-era contracts winding down frees wages.",
    },
    "Arsenal": {
        "revenue_eur_m": 640.0,
        "wage_bill_eur_m": 320.0,
        "amortised_debt_eur_m": 105.0,
        "operating_costs_eur_m": 72.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "PL_PSR",
        "net_spend_3yr_eur_m": 280.0,
        "estimated_headroom_note": "Moderate headroom; UEFA qualification improving revenue base.",
    },
    "Chelsea": {
        "revenue_eur_m": 620.0,
        "wage_bill_eur_m": 480.0,
        "amortised_debt_eur_m": 350.0,
        "operating_costs_eur_m": 88.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "PL_PSR",
        "net_spend_3yr_eur_m": 1100.0,
        "estimated_headroom_note": "Extreme PSR pressure from Boehly-era spending; must sell to buy.",
    },
    "Manchester United": {
        "revenue_eur_m": 660.0,
        "wage_bill_eur_m": 420.0,
        "amortised_debt_eur_m": 190.0,
        "operating_costs_eur_m": 82.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "PL_PSR",
        "net_spend_3yr_eur_m": 480.0,
        "estimated_headroom_note": "PSR constrained; INEOS prioritising player sales before buying.",
    },
    "Tottenham Hotspur": {
        "revenue_eur_m": 545.0,
        "wage_bill_eur_m": 290.0,
        "amortised_debt_eur_m": 140.0,
        "operating_costs_eur_m": 68.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "PL_PSR",
        "net_spend_3yr_eur_m": 180.0,
        "estimated_headroom_note": "One of better PSR positions in PL; focused spend expected.",
    },
    "Atletico Madrid": {
        "revenue_eur_m": 510.0,
        "wage_bill_eur_m": 310.0,
        "amortised_debt_eur_m": 120.0,
        "operating_costs_eur_m": 65.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 220.0,
        "estimated_headroom_note": "Disciplined under Cerezo; strategic sales fund buys.",
    },
    "Juventus": {
        "revenue_eur_m": 440.0,
        "wage_bill_eur_m": 280.0,
        "amortised_debt_eur_m": 170.0,
        "operating_costs_eur_m": 62.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 340.0,
        "estimated_headroom_note": "Under UEFA monitoring; must balance books before spending.",
    },
    "AC Milan": {
        "revenue_eur_m": 420.0,
        "wage_bill_eur_m": 230.0,
        "amortised_debt_eur_m": 95.0,
        "operating_costs_eur_m": 55.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 190.0,
        "estimated_headroom_note": "RedBird ownership pushing profitability; modest headroom available.",
    },
    "Inter Milan": {
        "revenue_eur_m": 400.0,
        "wage_bill_eur_m": 220.0,
        "amortised_debt_eur_m": 115.0,
        "operating_costs_eur_m": 58.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 200.0,
        "estimated_headroom_note": "Oaktree ownership stabilising; controlled spending expected.",
    },
    "Borussia Dortmund": {
        "revenue_eur_m": 480.0,
        "wage_bill_eur_m": 240.0,
        "amortised_debt_eur_m": 80.0,
        "operating_costs_eur_m": 60.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 120.0,
        "estimated_headroom_note": "Sell-to-buy model; Bellingham/Sancho sales created headroom.",
    },
    "Newcastle United": {
        "revenue_eur_m": 390.0,
        "wage_bill_eur_m": 230.0,
        "amortised_debt_eur_m": 85.0,
        "operating_costs_eur_m": 55.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "PL_PSR",
        "net_spend_3yr_eur_m": 310.0,
        "estimated_headroom_note": "Saudi PIF backing but PSR limits remain; balanced budget approach.",
    },
    "Aston Villa": {
        "revenue_eur_m": 340.0,
        "wage_bill_eur_m": 200.0,
        "amortised_debt_eur_m": 110.0,
        "operating_costs_eur_m": 48.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "PL_PSR",
        "net_spend_3yr_eur_m": 260.0,
        "estimated_headroom_note": "UCL revenue helping; Purslow managing costs carefully.",
    },
    "RB Leipzig": {
        "revenue_eur_m": 360.0,
        "wage_bill_eur_m": 195.0,
        "amortised_debt_eur_m": 70.0,
        "operating_costs_eur_m": 48.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 140.0,
        "estimated_headroom_note": "Red Bull structured approach; sell well, reinvest efficiently.",
    },
    "Napoli": {
        "revenue_eur_m": 330.0,
        "wage_bill_eur_m": 165.0,
        "amortised_debt_eur_m": 65.0,
        "operating_costs_eur_m": 45.0,
        "allowable_loss_eur_m": 105.0,
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": 130.0,
        "estimated_headroom_note": "De Laurentiis model: sell stars at peak, reinvest below market.",
    },
    "Porto": {
        "revenue_eur_m": 210.0,
        "wage_bill_eur_m": 100.0,
        "amortised_debt_eur_m": 55.0,
        "operating_costs_eur_m": 35.0,
        "allowable_loss_eur_m": 60.0,  # Smaller club, lower UEFA threshold
        "ffp_regime": "UEFA",
        "net_spend_3yr_eur_m": -80.0,  # Net seller historically
        "estimated_headroom_note": "Consistent net seller; develops and exports talent.",
    },
}

# Clubs under significant FFP/PSR financial stress (likely sellers)
FFP_STRESSED_CLUBS: Dict[str, List[str]] = {
    "wage_pressure": ["Barcelona", "Paris Saint-Germain", "Chelsea", "Manchester United"],
    "amortisation_overload": ["Chelsea", "Barcelona", "Manchester United", "Juventus"],
    "sell_to_comply": ["Chelsea", "Barcelona", "Juventus", "Aston Villa"],
    "net_seller_model": ["Porto", "RB Leipzig", "Borussia Dortmund", "Napoli"],
}


class FFPAnalyzer:
    """
    Financial Fair Play / PSR headroom analyzer for top European clubs.

    Computes available transfer spending headroom based on:
    - Revenue minus operating costs and wages = operating margin
    - Comparison of operating margin vs allowable loss threshold
    - Remaining net spend capacity within rolling 3-year PSR window
    - Headroom = min(operating_margin + allowable_loss, 3yr_window_remaining)
    """

    def __init__(self) -> None:
        self.clubs = CLUB_FINANCIALS

    # ------------------------------------------------------------------ #
    # Headroom calculation                                                 #
    # ------------------------------------------------------------------ #
    def _compute_headroom(self, club_data: Dict) -> float:
        """
        Compute available spending headroom in EUR millions.

        Headroom = (revenue - wage_bill - amortised_debt - operating_costs)
                   + allowable_loss_threshold
        Then capped by how much 3-year net spend window has remaining.
        Floored at 0 (negative headroom means they need to sell first).
        """
        operating_profit = (
            club_data["revenue_eur_m"]
            - club_data["wage_bill_eur_m"]
            - club_data["amortised_debt_eur_m"]
            - club_data["operating_costs_eur_m"]
        )
        # Allowable headroom = operating profit + regulatory allowable loss
        regulatory_headroom = operating_profit + club_data["allowable_loss_eur_m"]

        # 3-year window: most regimes allow ~€105m loss over 3 years
        # Remaining window = 3 * allowable_loss - net_spend_3yr
        # (simplified: we treat 3yr net spend as a direct constraint)
        window_remaining = (
            3 * club_data["allowable_loss_eur_m"] - club_data["net_spend_3yr_eur_m"]
        )
        window_remaining = max(window_remaining, 0.0)

        headroom = min(regulatory_headroom, window_remaining)
        return max(round(headroom, 1), 0.0)

    def get_headroom(self, club: str) -> float:
        """
        Return EUR millions available to spend for a specific club.

        Args:
            club: Club name matching a key in CLUB_FINANCIALS.

        Returns:
            Available spending headroom in EUR millions.

        Raises:
            KeyError if club not found in dataset.
        """
        if club not in self.clubs:
            available = list(self.clubs.keys())
            raise KeyError(
                f"Club '{club}' not in FFP dataset. Available: {available}"
            )
        return self._compute_headroom(self.clubs[club])

    def get_headroom_with_context(self, club: str) -> Dict:
        """
        Return headroom plus contextual detail for a club.

        Args:
            club: Club name.

        Returns:
            Dict with headroom, regime, note, and financial breakdown.
        """
        if club not in self.clubs:
            raise KeyError(f"Club '{club}' not in FFP dataset.")
        data = self.clubs[club]
        headroom = self._compute_headroom(data)
        operating_profit = (
            data["revenue_eur_m"]
            - data["wage_bill_eur_m"]
            - data["amortised_debt_eur_m"]
            - data["operating_costs_eur_m"]
        )
        return {
            "club": club,
            "headroom_eur_m": headroom,
            "ffp_regime": data["ffp_regime"],
            "operating_profit_eur_m": round(operating_profit, 1),
            "allowable_loss_eur_m": data["allowable_loss_eur_m"],
            "net_spend_3yr_eur_m": data["net_spend_3yr_eur_m"],
            "note": data["estimated_headroom_note"],
        }

    # ------------------------------------------------------------------ #
    # Likely sellers                                                       #
    # ------------------------------------------------------------------ #
    def get_likely_sellers(self, reason: str = "all") -> List[str]:
        """
        Return clubs likely to sell players due to FFP/financial pressure.

        Args:
            reason: Filter by reason type:
                - "wage_pressure"      - wage bill too high relative to revenue
                - "amortisation_overload" - heavy transfer debt repayments
                - "sell_to_comply"     - must sell to meet PSR/FFP deadline
                - "net_seller_model"   - structural sell-to-buy clubs
                - "all"               - union of all categories

        Returns:
            List of club name strings.
        """
        if reason == "all":
            sellers: set = set()
            for clubs in FFP_STRESSED_CLUBS.values():
                sellers.update(clubs)
            # Also include any clubs with computed headroom < 30m
            for club_name, data in self.clubs.items():
                if self._compute_headroom(data) < 30:
                    sellers.add(club_name)
            return sorted(sellers)

        if reason not in FFP_STRESSED_CLUBS:
            valid = list(FFP_STRESSED_CLUBS.keys()) + ["all"]
            raise ValueError(f"Unknown reason '{reason}'. Valid options: {valid}")

        return FFP_STRESSED_CLUBS[reason]

    # ------------------------------------------------------------------ #
    # Buying power ranking                                                 #
    # ------------------------------------------------------------------ #
    def buying_power_rank(self) -> pd.DataFrame:
        """
        Rank all clubs in the dataset by available spending headroom.

        Returns:
            pandas DataFrame with columns:
            rank, club, headroom_eur_m, ffp_regime, operating_profit_eur_m, note
        """
        rows = []
        for club_name, data in self.clubs.items():
            headroom = self._compute_headroom(data)
            op_profit = (
                data["revenue_eur_m"]
                - data["wage_bill_eur_m"]
                - data["amortised_debt_eur_m"]
                - data["operating_costs_eur_m"]
            )
            rows.append(
                {
                    "club": club_name,
                    "headroom_eur_m": headroom,
                    "ffp_regime": data["ffp_regime"],
                    "revenue_eur_m": data["revenue_eur_m"],
                    "wage_bill_eur_m": data["wage_bill_eur_m"],
                    "operating_profit_eur_m": round(op_profit, 1),
                    "note": data["estimated_headroom_note"],
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("headroom_eur_m", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    # ------------------------------------------------------------------ #
    # Clubs that can afford a given fee                                    #
    # ------------------------------------------------------------------ #
    def clubs_that_can_afford(
        self,
        transfer_fee_eur_m: float,
        exclude_clubs: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Return list of clubs with sufficient FFP headroom to spend a given fee.

        Args:
            transfer_fee_eur_m: Required transfer fee in EUR millions.
            exclude_clubs: Optional list of club names to exclude (e.g. the selling club).

        Returns:
            List of club names sorted by headroom (highest first).
        """
        exclude = set(exclude_clubs or [])
        eligible = []
        for club_name, data in self.clubs.items():
            if club_name in exclude:
                continue
            headroom = self._compute_headroom(data)
            if headroom >= transfer_fee_eur_m:
                eligible.append((club_name, headroom))
        eligible.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in eligible]

    # ------------------------------------------------------------------ #
    # Club data accessors                                                  #
    # ------------------------------------------------------------------ #
    def all_clubs(self) -> List[str]:
        """Return sorted list of all club names in the dataset."""
        return sorted(self.clubs.keys())

    def get_revenue(self, club: str) -> float:
        """Return club's total revenue in EUR millions."""
        return self.clubs[club]["revenue_eur_m"]

    def get_wage_bill(self, club: str) -> float:
        """Return club's annual wage bill in EUR millions."""
        return self.clubs[club]["wage_bill_eur_m"]


# --------------------------------------------------------------------------- #
# Quick smoke-test when run directly                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ffp = FFPAnalyzer()

    print("=== TOP 10 BUYING POWER RANKING ===")
    df = ffp.buying_power_rank()
    print(df[["rank", "club", "headroom_eur_m", "ffp_regime"]].head(10).to_string(index=False))

    print("\n=== CLUBS LIKELY TO SELL ===")
    print(ffp.get_likely_sellers("sell_to_comply"))

    print("\n=== CLUBS THAT CAN AFFORD €80m ===")
    print(ffp.clubs_that_can_afford(80.0))

    print("\n=== CHELSEA DETAILED ===")
    print(ffp.get_headroom_with_context("Chelsea"))
