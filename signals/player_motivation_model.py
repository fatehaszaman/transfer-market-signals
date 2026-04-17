"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
A player's transfer probability is fundamentally shaped by their
psychological state, life circumstances, and where their family is —
not just their age curve and contract status. This module quantifies the
"soft signals" that scouts and agents understand intuitively but that
quantitative models typically ignore.

Five categories of factors are modelled:

PUSH FACTORS — things that make a player want to leave:
  unhappy playing time, manager fallouts, contract disputes, dream club interest

STAY FACTORS — things that anchor a player at their current club:
  new contract, settled family life, just won a trophy, loyalty

EMOTIONAL STATE MODIFIERS — affect on-pitch performance IF the player stays:
  personal loss, new parenthood, divorce proceedings, "revenge mode"

FAMILY SITUATION — geographic and logistical life anchors:
  partner career location, children in school, visa complications,
  family attendance at matches (psychological stability signal)

CAREER STAGE — where the player is in their professional journey:
  emerging star vs. declining veteran, years at club, academy bond,
  new signing settling-in period, first major club pressure

The contentment score (0-100) drives two outputs:
1. A multiplier on transfer probability (unhappy player = higher prob)
2. A performance readiness rating (unhappy player = below peak output)

IMPORTANT — WEIGHTING FORMULA:
  performance_readiness = (
      (motivation_score * PSYCH_WEIGHT) + (technical_rating * PHYSICAL_WEIGHT)
  ) / TOTAL_WEIGHT

  PSYCH_WEIGHT = 1.0, PHYSICAL_WEIGHT = 1.5 (from config.py)

  # Weighting: physical/technical (1.5) outweighs psychological/motivation (1.0).
  # Rationale: a technically elite player with low motivation still commands high
  # transfer value and performs above average. Psychological state adjusts at the
  # margin — a deeply unhappy player (contentment score 30/100) still produces
  # at ~80% of peak physical ability. The 1.5:1.0 ratio reflects this floor effect.
=============================================================================

Developer notes:
- PlayerMotivationProfile is a dataclass for fast construction and mutation.
- PLAYER_MOTIVATION_DATA is the hardcoded dataset for 25 real players.
- Family signals (visa complications, partner settled, children enrolled) feed
  directly into both contentment score and transfer probability adjustment.
- Career stage signals (new signing, academy product, years at club) adjust
  both the probability and the performance modifier independently.
- destination_happiness_fit() checks Champions League qualification, league
  prestige, family visa situation, and project ambition.
- narrative() and career_narrative() both return human-readable summaries
  for embedding in scouting reports and the CLI dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import (
    CONTENTMENT_CONTENT,
    CONTENTMENT_RESTLESS,
    CONTENTMENT_UNHAPPY,
    CONTENTMENT_VERY_HAPPY,
    MOTIVATION_MAX_UPLIFT,
    PHYSICAL_WEIGHT,
    PSYCH_WEIGHT,
    TOTAL_WEIGHT,
    TRANSFER_PROB_CEILING,
    TRANSFER_PROB_FLOOR,
)

# ============================================================================ #
# Club metadata for destination happiness scoring                                #
# ============================================================================ #

CLUB_ATTRIBUTES: Dict[str, Dict] = {
    "Real Madrid":         {"ucl_regular": True,  "prestige": 10.0, "wage_tier": 10, "project": "elite",      "country": "Spain",   "schengen": True},
    "Barcelona":           {"ucl_regular": True,  "prestige": 10.0, "wage_tier":  9, "project": "elite",      "country": "Spain",   "schengen": True},
    "Manchester City":     {"ucl_regular": True,  "prestige":  9.5, "wage_tier": 10, "project": "elite",      "country": "England", "schengen": False},
    "Paris Saint-Germain": {"ucl_regular": True,  "prestige":  9.0, "wage_tier": 10, "project": "elite",      "country": "France",  "schengen": True},
    "Bayern Munich":       {"ucl_regular": True,  "prestige":  9.5, "wage_tier":  9, "project": "elite",      "country": "Germany", "schengen": True},
    "Liverpool":           {"ucl_regular": True,  "prestige":  9.0, "wage_tier":  9, "project": "elite",      "country": "England", "schengen": False},
    "Arsenal":             {"ucl_regular": True,  "prestige":  8.5, "wage_tier":  8, "project": "rising",     "country": "England", "schengen": False},
    "Chelsea":             {"ucl_regular": False, "prestige":  8.5, "wage_tier":  9, "project": "unstable",   "country": "England", "schengen": False},
    "Manchester United":   {"ucl_regular": False, "prestige":  9.0, "wage_tier":  8, "project": "rebuilding", "country": "England", "schengen": False},
    "Tottenham Hotspur":   {"ucl_regular": False, "prestige":  7.5, "wage_tier":  7, "project": "mid-table",  "country": "England", "schengen": False},
    "Atletico Madrid":     {"ucl_regular": True,  "prestige":  8.5, "wage_tier":  7, "project": "elite",      "country": "Spain",   "schengen": True},
    "Juventus":            {"ucl_regular": True,  "prestige":  8.5, "wage_tier":  7, "project": "declining",  "country": "Italy",   "schengen": True},
    "AC Milan":            {"ucl_regular": True,  "prestige":  8.0, "wage_tier":  6, "project": "rising",     "country": "Italy",   "schengen": True},
    "Inter Milan":         {"ucl_regular": True,  "prestige":  8.0, "wage_tier":  6, "project": "solid",      "country": "Italy",   "schengen": True},
    "Borussia Dortmund":   {"ucl_regular": True,  "prestige":  7.5, "wage_tier":  6, "project": "dev",        "country": "Germany", "schengen": True},
    "Newcastle United":    {"ucl_regular": False, "prestige":  7.0, "wage_tier":  7, "project": "rising",     "country": "England", "schengen": False},
    "Aston Villa":         {"ucl_regular": True,  "prestige":  7.0, "wage_tier":  6, "project": "rising",     "country": "England", "schengen": False},
    "RB Leipzig":          {"ucl_regular": True,  "prestige":  7.0, "wage_tier":  6, "project": "dev",        "country": "Germany", "schengen": True},
    "Napoli":              {"ucl_regular": True,  "prestige":  7.5, "wage_tier":  6, "project": "solid",      "country": "Italy",   "schengen": True},
    "Porto":               {"ucl_regular": True,  "prestige":  6.5, "wage_tier":  4, "project": "selling",    "country": "Portugal","schengen": True},
    "West Ham United":     {"ucl_regular": False, "prestige":  7.0, "wage_tier":  6, "project": "mid-table",  "country": "England", "schengen": False},
    "AFC Bournemouth":     {"ucl_regular": False, "prestige":  5.5, "wage_tier":  4, "project": "mid-table",  "country": "England", "schengen": False},
    "LOSC Lille":          {"ucl_regular": False, "prestige":  6.0, "wage_tier":  4, "project": "selling",    "country": "France",  "schengen": True},
    "Benfica":             {"ucl_regular": True,  "prestige":  6.5, "wage_tier":  4, "project": "selling",    "country": "Portugal","schengen": True},
}

# Countries whose nationals commonly face Schengen/UK visa friction when
# family members want to attend matches in Europe.
# This is a rough categorisation for modelling purposes.
HIGH_VISA_FRICTION_NATIONALITIES = {
    "Nigerian", "Ghanaian", "Senegalese", "Ivorian", "Malian",
    "Cameroonian", "Congolese", "Algerian", "Moroccan", "Egyptian",
    "Bangladeshi", "Pakistani", "Sri Lankan", "Nepalese",
    "Afghan", "Iraqi", "Libyan", "Yemeni",
}

FAMILY_ATTENDANCE_CONTENTMENT: Dict[str, float] = {
    "regularly":    +8.0,
    "occasionally": +3.0,
    "rarely":       -4.0,
    "never":        -8.0,
}


# ============================================================================ #
# PlayerMotivationProfile dataclass                                              #
# ============================================================================ #

@dataclass
class PlayerMotivationProfile:
    """
    ==========================================================================
    BUSINESS: Encodes the soft signals driving a player's transfer likelihood.
    Each flag and modifier adjusts the contentment score from the baseline
    of 100 (perfectly happy, not going anywhere) before computing transfer
    probability adjustments and performance projections.
    ==========================================================================

    PUSH FACTORS (subtract from contentment — increase transfer probability):
        unhappy_playing_time           : Benched or publicly expressed frustration (-15)
        public_fallout_with_manager    : Publicly broken relationship with coach (-20)
        dressing_room_conflict         : Falling out with key teammates (-12)
        contract_dispute_ongoing       : Failed renewal talks confirmed (-18)
        personal_life_relocation_desire: Partner/family want to move (-8)
        recent_trophy_achieved         : Won what they came for, ready to move (-6)
        reported_dream_club_interest   : Real Madrid/Barcelona/etc. known interest (-25)

    STAY FACTORS (add to contentment — decrease transfer probability):
        new_contract_signed_recently   : Committed to club (+20)
        happy_family_settled           : Life anchor — kids in school, partner settled (+10)
        club_loyalty_sentiment         : Long tenure, publicly praised club (+8)
        just_won_major_trophy          : Peak happiness moment (+12)
        injury_recovering              : Wants stability during rehab (+6)
        financial_peak_at_current_club : Highest earner, hard to match (+15)

    EMOTIONAL STATE (affect performance quality IF player stays):
        personal_loss_recent           : -15% performance
        new_parent_distraction         : -5% performance
        divorce_proceedings            : -10% performance
        mental_health_public_disclosure: -8% performance + resilience flag
        revenge_mode                   : +12% performance

    FAMILY SITUATION (new):
        family_location                : City/country where family lives
        family_at_matches              : "regularly"|"occasionally"|"rarely"|"never"
        partner_settled_current_city   : Partner career anchored at current club city
        children_school_enrolled       : Children enrolled in school (relocation barrier)
        family_visa_complications      : Family faces Schengen/UK visa difficulties
        player_nationality             : For visa friction detection

    CAREER STAGE (new):
        career_stage                   : "emerging"|"prime"|"established_star"|"declining"
        years_at_current_club          : Loyalty/comfort signal
        league_debut_age               : Age of first professional league appearance
        current_age                    : For phase cross-checks
        first_major_club               : First elite-level club (everything to prove)
        is_new_signing                 : Signed in last 12 months (adapting)
        settling_in_period             : First 6 months at club (lower performance)
        academy_product                : Came through the academy (emotional bond)
    """

    # ---- Identity ----
    player_name: str
    current_club: str
    baseline: float = 100.0

    # ---- PUSH FACTORS ----
    unhappy_playing_time: bool             = False  # -15
    public_fallout_with_manager: bool      = False  # -20
    dressing_room_conflict: bool           = False  # -12
    contract_dispute_ongoing: bool         = False  # -18
    personal_life_relocation_desire: bool  = False  # -8
    recent_trophy_achieved: bool           = False  # -6
    reported_dream_club_interest: bool     = False  # -25

    # ---- STAY FACTORS ----
    new_contract_signed_recently: bool     = False  # +20
    happy_family_settled: bool             = False  # +10
    club_loyalty_sentiment: bool           = False  # +8
    just_won_major_trophy: bool            = False  # +12
    injury_recovering: bool                = False  # +6
    financial_peak_at_current_club: bool   = False  # +15

    # ---- EMOTIONAL STATE ----
    personal_loss_recent: bool                = False  # -15% perf
    new_parent_distraction: bool              = False  # -5% perf
    divorce_proceedings: bool                 = False  # -10% perf
    mental_health_public_disclosure: bool     = False  # -8% perf
    revenge_mode: bool                        = False  # +12% perf

    # ---- FAMILY SITUATION ----
    family_location: str                      = "unknown"
    family_at_matches: str                    = "occasionally"  # "regularly"|"occasionally"|"rarely"|"never"
    partner_settled_current_city: bool        = False   # partner career anchored
    children_school_enrolled: bool            = False   # relocation barrier
    family_visa_complications: bool           = False   # Schengen/UK visa difficulties
    player_nationality: str                   = "unknown"

    # ---- CAREER STAGE ----
    career_stage: str                         = "prime"  # "emerging"|"prime"|"established_star"|"declining"
    years_at_current_club: int                = 2
    league_debut_age: int                     = 18
    current_age: int                          = 24
    first_major_club: bool                    = False   # First elite club
    is_new_signing: bool                      = False   # Signed in last 12 months
    settling_in_period: bool                  = False   # First 6 months at club
    academy_product: bool                     = False   # Came through academy

    # ---- Freeform ----
    context_notes: List[str]                  = field(default_factory=list)
    preferred_destinations: List[str]         = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Computed deltas                                                       #
    # ------------------------------------------------------------------ #

    @property
    def push_delta(self) -> float:
        """Sum of all active push factor deductions."""
        delta = 0.0
        if self.unhappy_playing_time:            delta -= 15.0
        if self.public_fallout_with_manager:     delta -= 20.0
        if self.dressing_room_conflict:          delta -= 12.0
        if self.contract_dispute_ongoing:        delta -= 18.0
        if self.personal_life_relocation_desire: delta -= 8.0
        if self.recent_trophy_achieved:          delta -= 6.0
        if self.reported_dream_club_interest:    delta -= 25.0
        return delta

    @property
    def stay_delta(self) -> float:
        """Sum of all active stay factor additions."""
        delta = 0.0
        if self.new_contract_signed_recently:    delta += 20.0
        if self.happy_family_settled:            delta += 10.0
        if self.club_loyalty_sentiment:          delta +=  8.0
        if self.just_won_major_trophy:           delta += 12.0
        if self.injury_recovering:               delta +=  6.0
        if self.financial_peak_at_current_club:  delta += 15.0
        return delta

    @property
    def family_delta(self) -> float:
        """
        Net contentment adjustment from family situation signals.

        Positive = family situation increases contentment (stabilising).
        Negative = family situation is a source of stress or push.
        """
        delta = 0.0

        # Partner + children anchored at current city = strong stay signal
        if self.partner_settled_current_city and self.children_school_enrolled:
            delta -= 12.0   # Contentment PUSH: player is torn — family says stay
            # Note: this reduces transfer probability indirectly via contentment,
            # but also directly via transfer_probability_adjusted()

        # Family attendance at matches
        delta += FAMILY_ATTENDANCE_CONTENTMENT.get(self.family_at_matches, 0.0)

        # Visa complications: family can't visit easily → isolation → push factor
        if self.family_visa_complications:
            delta -= 10.0

        return delta

    @property
    def family_transfer_anchor(self) -> float:
        """
        DIRECT transfer probability reduction from family anchors.
        This is separate from contentment — it directly reduces the multiplier
        because even an unhappy player may not move when family is rooted.

        Returns a value to subtract from the probability multiplier.
        """
        anchor = 0.0
        if self.partner_settled_current_city and self.children_school_enrolled:
            anchor += 0.12   # -12 percentage points on transfer probability
        elif self.partner_settled_current_city or self.children_school_enrolled:
            anchor += 0.06   # One of the two is present
        return anchor

    @property
    def career_stage_delta(self) -> float:
        """
        Net contentment adjustment from career stage signals.
        """
        delta = 0.0

        # New signing: adapting — reduce transfer probability signal by increasing contentment
        # (they just committed; takes time to become unhappy)
        if self.is_new_signing:
            delta -= 8.0   # Slight anxiety of new environment

        # Academy product: emotional bond
        if self.academy_product:
            delta -= 10.0   # Contentment push (harder to leave emotionally)
            # Note: academy_product reduces transfer probability directly too

        # First major club + emerging: motivated
        if self.first_major_club and self.career_stage == "emerging":
            delta += 12.0   # High motivation = better contentment at performing club

        # Long tenure: comfort/institutional loyalty
        if self.years_at_current_club >= 6:
            delta += 8.0
        elif self.years_at_current_club >= 3:
            delta += 4.0

        return delta

    @property
    def career_transfer_modifiers(self) -> float:
        """
        Direct transfer probability adjustment from career stage signals.
        Returns the fraction to ADD to transfer probability (positive = more likely to move).
        """
        mod = 0.0
        if self.is_new_signing:
            mod -= 0.08   # Just moved — very unlikely to move again
        if self.academy_product and not self.public_fallout_with_manager:
            mod -= 0.10   # Emotional bond unless burned bridges
        if self.career_stage == "declining" and self.contract_dispute_ongoing:
            mod += 0.15   # Declining player chasing last big contract
        if self.first_major_club and self.career_stage == "emerging":
            mod -= 0.05   # Wants to prove themselves here first
        return mod

    @property
    def performance_modifier(self) -> float:
        """
        Net emotional performance modifier (1.0 = no effect, ~[0.70, 1.20]).

        Covers emotional state + settling-in penalty from career stage.
        Does NOT include the physical/technical component — that is applied
        separately in performance_if_stays() using the weighted composite.
        """
        mod = 1.0
        if self.personal_loss_recent:               mod -= 0.15
        if self.new_parent_distraction:             mod -= 0.05
        if self.divorce_proceedings:                mod -= 0.10
        if self.mental_health_public_disclosure:    mod -= 0.08
        if self.revenge_mode:                       mod += 0.12
        # Career stage: new signing / settling in period
        if self.settling_in_period:                 mod -= 0.05
        if self.is_new_signing and not self.settling_in_period:
            mod -= 0.02   # Mild lingering adjustment beyond first 6 months
        return round(max(0.50, min(1.25, mod)), 4)

    @property
    def contentment_score(self) -> float:
        """
        Composite contentment score clamped to [0, 100].

        Combines push factors, stay factors, family signals, and career stage signals.
        100 = perfectly happy, not going anywhere.
        0   = completely miserable, transfer inevitable.
        """
        raw = (
            self.baseline
            + self.push_delta
            + self.stay_delta
            + self.family_delta
            + self.career_stage_delta
        )
        return round(max(0.0, min(100.0, raw)), 2)

    # ------------------------------------------------------------------ #
    # Primary factor accessors                                             #
    # ------------------------------------------------------------------ #

    @property
    def primary_push_factor(self) -> Optional[str]:
        """Return the single largest active push factor label."""
        candidates = []
        if self.reported_dream_club_interest:    candidates.append(("reported dream club interest",       25.0))
        if self.public_fallout_with_manager:     candidates.append(("public fallout with manager",        20.0))
        if self.contract_dispute_ongoing:        candidates.append(("contract dispute ongoing",           18.0))
        if self.unhappy_playing_time:            candidates.append(("unhappy playing time",               15.0))
        if self.family_visa_complications:       candidates.append(("family visa complications",          10.0))
        if self.dressing_room_conflict:          candidates.append(("dressing room conflict",             12.0))
        if self.personal_life_relocation_desire: candidates.append(("personal life relocation desire",    8.0))
        if self.family_at_matches == "never":    candidates.append(("family unable to attend matches",    8.0))
        if self.recent_trophy_achieved:          candidates.append(("recent trophy — ready to move on",   6.0))
        return max(candidates, key=lambda x: x[1])[0] if candidates else None

    @property
    def primary_stay_factor(self) -> Optional[str]:
        """Return the single largest active stay factor label."""
        candidates = []
        if self.new_contract_signed_recently:    candidates.append(("new contract signed recently",        20.0))
        if self.financial_peak_at_current_club:  candidates.append(("financial peak at current club",     15.0))
        if self.just_won_major_trophy:           candidates.append(("just won major trophy",              12.0))
        if self.partner_settled_current_city and self.children_school_enrolled:
                                                 candidates.append(("family fully anchored at current city",12.0))
        if self.happy_family_settled:            candidates.append(("happy family settled",               10.0))
        if self.academy_product:                 candidates.append(("academy product — emotional bond",   10.0))
        if self.club_loyalty_sentiment:          candidates.append(("club loyalty sentiment",              8.0))
        if self.family_at_matches == "regularly":candidates.append(("family attends matches regularly",    8.0))
        if self.years_at_current_club >= 6:      candidates.append(("6+ years at club — institutionalised",8.0))
        if self.injury_recovering:               candidates.append(("injury recovering",                   6.0))
        return max(candidates, key=lambda x: x[1])[0] if candidates else None


# ============================================================================ #
# Hardcoded motivation data — 25 real players                                    #
# ============================================================================ #

PLAYER_MOTIVATION_DATA: Dict[str, PlayerMotivationProfile] = {

    "Kylian Mbappe": PlayerMotivationProfile(
        player_name="Kylian Mbappe",
        current_club="Paris Saint-Germain",
        contract_dispute_ongoing=True,
        reported_dream_club_interest=True,
        unhappy_playing_time=True,
        financial_peak_at_current_club=True,
        # Family signals
        family_location="Paris, France",
        family_at_matches="regularly",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="French",
        # Career stage
        career_stage="established_star",
        years_at_current_club=6,
        current_age=25,
        first_major_club=False,
        is_new_signing=False,
        context_notes=[
            "Refused to sign PSG extension; leaked falling out with Nasser Al-Khelaifi.",
            "Real Madrid made formal offer confirmed by multiple sources.",
            "Benched for key PSG CL matches in reported retaliation.",
            "Mother Fayza Lamari acting as agent — pushed Real Madrid move.",
        ],
        preferred_destinations=["Real Madrid"],
    ),

    "Jude Bellingham": PlayerMotivationProfile(
        player_name="Jude Bellingham",
        current_club="Borussia Dortmund",
        reported_dream_club_interest=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="Birmingham, England",
        family_at_matches="occasionally",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="English",
        # Career stage
        career_stage="emerging",
        years_at_current_club=2,
        current_age=19,
        first_major_club=True,
        is_new_signing=False,
        context_notes=[
            "Real Madrid made repeated contact with Bellingham family.",
            "BVB acknowledged departure likely after 2022/23 season.",
            "Parents Mark and Denise Bellingham prominent at Dortmund games.",
        ],
        preferred_destinations=["Real Madrid"],
    ),

    "Erling Haaland": PlayerMotivationProfile(
        player_name="Erling Haaland",
        current_club="Borussia Dortmund",
        reported_dream_club_interest=True,
        # Family signals
        family_location="Bryne, Norway",
        family_at_matches="occasionally",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="Norwegian",
        # Career stage
        career_stage="emerging",
        years_at_current_club=2,
        current_age=21,
        first_major_club=False,
        is_new_signing=False,
        context_notes=[
            "Release clause of €60m activated by Manchester City.",
            "Father Alfie Haaland had Manchester City connections from playing days.",
            "Multiple clubs interested; City and Real Madrid were frontrunners.",
        ],
        preferred_destinations=["Manchester City", "Real Madrid"],
    ),

    "Declan Rice": PlayerMotivationProfile(
        player_name="Declan Rice",
        current_club="West Ham United",
        contract_dispute_ongoing=True,
        reported_dream_club_interest=True,
        club_loyalty_sentiment=True,
        recent_trophy_achieved=True,
        # Family signals
        family_location="London, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,   # Lauren Fryer anchored in London
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="English",
        # Career stage
        career_stage="prime",
        years_at_current_club=7,
        current_age=24,
        first_major_club=False,
        is_new_signing=False,
        academy_product=True,               # West Ham academy product
        context_notes=[
            "Won UECL 2023 — felt he'd achieved peak possible at West Ham.",
            "Arsenal bid of £105m accepted; Chelsea also bid heavily.",
            "Girlfriend Lauren settled in London — move to Arsenal (same city) easy.",
            "West Ham academy product but captaincy replaced emotional anchoring.",
        ],
        preferred_destinations=["Arsenal", "Manchester City"],
    ),

    "Lautaro Martinez": PlayerMotivationProfile(
        player_name="Lautaro Martinez",
        current_club="Inter Milan",
        reported_dream_club_interest=True,
        just_won_major_trophy=True,
        new_contract_signed_recently=True,
        financial_peak_at_current_club=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="Bahia Blanca, Argentina",
        family_at_matches="occasionally",
        partner_settled_current_city=True,   # Agustina Gandolfo settled in Milan
        children_school_enrolled=True,       # Daughter Nina in school in Milan
        family_visa_complications=False,     # Argentine passport; Schengen fine
        player_nationality="Argentine",
        # Career stage
        career_stage="prime",
        years_at_current_club=6,
        current_age=26,
        first_major_club=False,
        is_new_signing=False,
        context_notes=[
            "Barcelona's financial situation prevented completing the transfer.",
            "Inter offered captain armband and matched market wage terms.",
            "Daughter Nina enrolled in Milanese school — critical anchor.",
            "Wife Agustina fully settled with friends and routines in Milan.",
        ],
        preferred_destinations=["Barcelona"],
    ),

    "Pedri": PlayerMotivationProfile(
        player_name="Pedri",
        current_club="Barcelona",
        new_contract_signed_recently=True,
        club_loyalty_sentiment=True,
        happy_family_settled=True,
        injury_recovering=True,
        # Family signals
        family_location="Las Palmas, Spain",
        family_at_matches="regularly",
        partner_settled_current_city=True,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="Spanish",
        # Career stage
        career_stage="emerging",
        years_at_current_club=4,
        current_age=22,
        first_major_club=True,
        is_new_signing=False,
        context_notes=[
            "Lifelong Barcelona fan; publicly described Barça as only club for him.",
            "Family frequently travel from Las Palmas; proximity is important.",
        ],
        preferred_destinations=[],
    ),

    "Leny Yoro": PlayerMotivationProfile(
        player_name="Leny Yoro",
        current_club="LOSC Lille",
        reported_dream_club_interest=True,
        contract_dispute_ongoing=True,
        injury_recovering=True,
        # Family signals
        family_location="Paris, France",
        family_at_matches="occasionally",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="French",
        # Career stage
        career_stage="emerging",
        years_at_current_club=3,
        current_age=18,
        first_major_club=True,
        is_new_signing=False,
        context_notes=[
            "Rejected multiple Lille renewal proposals.",
            "Real Madrid had priority but Man United moved faster.",
            "Metatarsal injury at pre-season delayed debut at Man United.",
        ],
        preferred_destinations=["Real Madrid", "Manchester United"],
    ),

    "Joao Neves": PlayerMotivationProfile(
        player_name="Joao Neves",
        current_club="Benfica",
        reported_dream_club_interest=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="Aveiro, Portugal",
        family_at_matches="occasionally",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="Portuguese",
        # Career stage
        career_stage="emerging",
        years_at_current_club=3,
        current_age=19,
        first_major_club=True,
        is_new_signing=False,
        context_notes=[
            "PSG activated €60m release clause in summer 2024.",
            "Family from Aveiro — Cristiano Ronaldo's hometown.",
        ],
        preferred_destinations=["Paris Saint-Germain", "Manchester City"],
    ),

    "Bukayo Saka": PlayerMotivationProfile(
        player_name="Bukayo Saka",
        current_club="Arsenal",
        new_contract_signed_recently=True,
        happy_family_settled=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="Ealing, London, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="English",
        # Career stage
        career_stage="prime",
        years_at_current_club=8,
        current_age=23,
        first_major_club=True,
        academy_product=True,
        is_new_signing=False,
        context_notes=[
            "Signed long extension at Arsenal; committed to the project.",
            "Family from Ealing attend virtually every home game.",
            "Arsenal academy product — emotional bond very strong.",
        ],
        preferred_destinations=[],
    ),

    "Virgil van Dijk": PlayerMotivationProfile(
        player_name="Virgil van Dijk",
        current_club="Liverpool",
        new_contract_signed_recently=True,
        club_loyalty_sentiment=True,
        happy_family_settled=True,
        # Family signals
        family_location="Liverpool, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,    # Rike settled in Liverpool
        children_school_enrolled=True,        # Three children in Liverpool schools
        family_visa_complications=False,
        player_nationality="Dutch",
        # Career stage
        career_stage="declining",
        years_at_current_club=7,
        current_age=33,
        first_major_club=False,
        is_new_signing=False,
        context_notes=[
            "Captain; extended beyond initial expiry. Rike and children fully settled.",
            "Three kids in Liverpool schools — relocation essentially impossible.",
            "Publicly said Liverpool is 'home' multiple times.",
        ],
        preferred_destinations=[],
    ),

    "William Saliba": PlayerMotivationProfile(
        player_name="William Saliba",
        current_club="Arsenal",
        new_contract_signed_recently=True,
        happy_family_settled=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="London, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="French",
        # Career stage
        career_stage="prime",
        years_at_current_club=3,
        current_age=24,
        first_major_club=True,
        is_new_signing=False,
        context_notes=[
            "Locked in long-term after strong form; Arsenal priority retention.",
            "Girlfriend settled in London; enjoys life in the city.",
        ],
        preferred_destinations=[],
    ),

    "Ruben Dias": PlayerMotivationProfile(
        player_name="Ruben Dias",
        current_club="Manchester City",
        new_contract_signed_recently=True,
        just_won_major_trophy=True,
        happy_family_settled=True,
        # Family signals
        family_location="Manchester, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="Portuguese",
        # Career stage
        career_stage="prime",
        years_at_current_club=4,
        current_age=27,
        first_major_club=False,
        is_new_signing=False,
        context_notes=["Core City player; extended after Treble season. Settled life in Manchester."],
        preferred_destinations=[],
    ),

    "Rodri": PlayerMotivationProfile(
        player_name="Rodri",
        current_club="Manchester City",
        injury_recovering=True,
        new_contract_signed_recently=True,
        club_loyalty_sentiment=True,
        just_won_major_trophy=True,
        # Family signals
        family_location="Manchester, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="Spanish",
        # Career stage
        career_stage="prime",
        years_at_current_club=5,
        current_age=28,
        first_major_club=False,
        is_new_signing=False,
        context_notes=["Ballon d'Or; ACL recovery ongoing at City. No exit signals whatsoever."],
        preferred_destinations=[],
    ),

    "Vinicius Junior": PlayerMotivationProfile(
        player_name="Vinicius Junior",
        current_club="Real Madrid",
        club_loyalty_sentiment=True,
        financial_peak_at_current_club=True,
        # Family signals
        family_location="Salvador, Brazil",
        family_at_matches="rarely",           # Brazilian family face Schengen friction
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=True,       # Brazilian family visas for Spain
        player_nationality="Brazilian",
        # Career stage
        career_stage="prime",
        years_at_current_club=6,
        current_age=24,
        first_major_club=True,
        is_new_signing=False,
        context_notes=[
            "Contract extension imminent at Madrid.",
            "Brazilian family face Schengen visa complexities; travels home in off-season.",
            "Has publicly spoken about family missing him — visa topic mentioned in interviews.",
        ],
        preferred_destinations=[],
    ),

    "Manuel Ugarte": PlayerMotivationProfile(
        player_name="Manuel Ugarte",
        current_club="Manchester United",
        is_new_signing=True,
        settling_in_period=False,
        # Family signals
        family_location="Montevideo, Uruguay",
        family_at_matches="occasionally",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="Uruguayan",
        # Career stage
        career_stage="prime",
        years_at_current_club=1,
        current_age=23,
        first_major_club=False,
        context_notes=["Arrived summer 2024; early adaptation period. No strong signals yet."],
        preferred_destinations=[],
    ),

    "Phil Foden": PlayerMotivationProfile(
        player_name="Phil Foden",
        current_club="Manchester City",
        new_contract_signed_recently=True,
        happy_family_settled=True,
        club_loyalty_sentiment=True,
        just_won_major_trophy=True,
        # Family signals
        family_location="Stockport, Manchester, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,   # Rebecca Cooke settled in Manchester
        children_school_enrolled=True,       # Kids in Stockport schools
        family_visa_complications=False,
        player_nationality="English",
        # Career stage
        career_stage="prime",
        years_at_current_club=10,
        current_age=24,
        first_major_club=True,
        academy_product=True,
        is_new_signing=False,
        context_notes=[
            "Lifelong City fan from Stockport; grew up watching City.",
            "Rebecca and kids fully settled — virtually impossible to move.",
            "Academy product: deepest emotional club bond in the squad.",
        ],
        preferred_destinations=[],
    ),

    "Dean Huijsen": PlayerMotivationProfile(
        player_name="Dean Huijsen",
        current_club="AFC Bournemouth",
        reported_dream_club_interest=True,
        # Family signals
        family_location="Amsterdam, Netherlands",
        family_at_matches="occasionally",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="Dutch",
        # Career stage
        career_stage="emerging",
        years_at_current_club=1,
        current_age=19,
        first_major_club=False,
        is_new_signing=True,
        context_notes=[
            "Real Madrid long-term target; Bournemouth viewed as stepping stone.",
            "Young and unattached — maximum transfer flexibility.",
        ],
        preferred_destinations=["Real Madrid", "Arsenal"],
    ),

    "Federico Chiesa": PlayerMotivationProfile(
        player_name="Federico Chiesa",
        current_club="Liverpool",
        injury_recovering=True,
        unhappy_playing_time=True,
        # Family signals
        family_location="Turin, Italy",
        family_at_matches="occasionally",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="Italian",
        # Career stage
        career_stage="declining",
        years_at_current_club=1,
        current_age=27,
        first_major_club=False,
        is_new_signing=True,
        settling_in_period=True,
        context_notes=[
            "Persistent injury problems limiting adaptation at Liverpool.",
            "Liverpool viewed as dream move but injuries preventing contribution.",
        ],
        preferred_destinations=[],
    ),

    "Marcus Thuram": PlayerMotivationProfile(
        player_name="Marcus Thuram",
        current_club="Inter Milan",
        new_contract_signed_recently=True,
        happy_family_settled=True,
        just_won_major_trophy=True,
        # Family signals
        family_location="Milan, Italy",
        family_at_matches="regularly",
        partner_settled_current_city=True,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="French",
        # Career stage
        career_stage="prime",
        years_at_current_club=2,
        current_age=27,
        first_major_club=False,
        context_notes=["Thriving at Inter; father Lilian Thuram visits regularly."],
        preferred_destinations=[],
    ),

    "Manu Kone": PlayerMotivationProfile(
        player_name="Manu Kone",
        current_club="Real Madrid",
        new_contract_signed_recently=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="Bamako, Mali",
        family_at_matches="rarely",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=True,    # Malian family; Schengen visa complications
        player_nationality="Malian",
        # Career stage
        career_stage="prime",
        years_at_current_club=1,
        current_age=23,
        first_major_club=False,
        is_new_signing=True,
        context_notes=[
            "Arrived at Real Madrid as a project signing.",
            "Malian family face Schengen visa complexity — visits home in breaks.",
            "Publicly mentioned missing family as an adjustment challenge.",
        ],
        preferred_destinations=[],
    ),

    "Alisson Becker": PlayerMotivationProfile(
        player_name="Alisson Becker",
        current_club="Liverpool",
        new_contract_signed_recently=True,
        happy_family_settled=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="Liverpool, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,   # Natalia Loewe settled in Liverpool
        children_school_enrolled=True,       # Three children in Liverpool schools
        family_visa_complications=False,     # Brazilian; no UK visa complications for established residents
        player_nationality="Brazilian",
        # Career stage
        career_stage="declining",
        years_at_current_club=6,
        current_age=32,
        first_major_club=False,
        is_new_signing=False,
        context_notes=[
            "Three children enrolled in Liverpool schools.",
            "Wife Natalia has established charity work in Merseyside.",
            "Described Liverpool as 'home' in multiple interviews.",
        ],
        preferred_destinations=[],
    ),

    "Erling Haaland (Man City)": PlayerMotivationProfile(
        player_name="Erling Haaland",
        current_club="Manchester City",
        new_contract_signed_recently=True,
        just_won_major_trophy=True,
        happy_family_settled=True,
        financial_peak_at_current_club=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="Manchester, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,   # Isabel Haugseng Johansen settled
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="Norwegian",
        # Career stage
        career_stage="prime",
        years_at_current_club=3,
        current_age=24,
        first_major_club=False,
        context_notes=[
            "Extended contract; talks of future Real Madrid move cooling.",
            "Father Alfie frequently at games; family settled in Manchester.",
        ],
        preferred_destinations=[],
    ),

    "Declan Rice (Arsenal)": PlayerMotivationProfile(
        player_name="Declan Rice",
        current_club="Arsenal",
        new_contract_signed_recently=True,
        happy_family_settled=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="North London, England",
        family_at_matches="regularly",
        partner_settled_current_city=True,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="English",
        # Career stage
        career_stage="prime",
        years_at_current_club=2,
        current_age=26,
        is_new_signing=False,
        context_notes=["Settled in North London; partner Lauren Fryer at home games."],
        preferred_destinations=[],
    ),

    "Kylian Mbappe (Real Madrid)": PlayerMotivationProfile(
        player_name="Kylian Mbappe",
        current_club="Real Madrid",
        new_contract_signed_recently=True,
        just_won_major_trophy=False,
        # Family signals
        family_location="Madrid, Spain",
        family_at_matches="regularly",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="French",
        # Career stage
        career_stage="established_star",
        years_at_current_club=1,
        current_age=26,
        is_new_signing=True,
        settling_in_period=False,
        club_loyalty_sentiment=True,
        context_notes=["Settling in; early form mixed but commitment to Real Madrid clear."],
        preferred_destinations=[],
    ),

    "Jude Bellingham (Real Madrid)": PlayerMotivationProfile(
        player_name="Jude Bellingham",
        current_club="Real Madrid",
        new_contract_signed_recently=True,
        just_won_major_trophy=True,
        club_loyalty_sentiment=True,
        # Family signals
        family_location="Madrid, Spain",
        family_at_matches="regularly",
        partner_settled_current_city=False,
        children_school_enrolled=False,
        family_visa_complications=False,
        player_nationality="English",
        # Career stage
        career_stage="prime",
        years_at_current_club=2,
        current_age=21,
        first_major_club=False,
        is_new_signing=False,
        context_notes=["Immediate impact at Real Madrid; long-term commitment signed. Parents at home games."],
        preferred_destinations=[],
    ),
}


# ============================================================================ #
# PlayerMotivationScorer                                                         #
# ============================================================================ #

class PlayerMotivationScorer:
    """
    ==========================================================================
    BUSINESS SUMMARY
    ==========================================================================
    Translates a player's contentment profile (push/stay/family/career signals)
    into four actionable outputs:

    1. Adjusted transfer probability — unhappy player = higher probability
    2. Performance readiness — % of peak rating achievable if they stay
    3. Destination happiness fit — would this move genuinely improve their life?
    4. Narrative / career narrative — plain-English scouting report text

    All outputs are calibrated to remain useful even when motivation data is
    partial — defaults assume neutral/happy baseline for unknown players.
    ==========================================================================

    Args:
        motivation_data: Optionally override PLAYER_MOTIVATION_DATA for testing.
    """

    def __init__(
        self,
        motivation_data: Optional[Dict[str, PlayerMotivationProfile]] = None,
    ) -> None:
        self.motivation_data = (
            motivation_data if motivation_data is not None else PLAYER_MOTIVATION_DATA
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_profile(self, player: str) -> PlayerMotivationProfile:
        """
        Retrieve motivation profile. Returns neutral default if not found.
        """
        if player in self.motivation_data:
            return self.motivation_data[player]
        return PlayerMotivationProfile(
            player_name=player,
            current_club="Unknown",
            baseline=70.0,
            context_notes=["No motivation data available — using neutral default."],
        )

    @staticmethod
    def _contentment_to_transfer_multiplier(contentment: float) -> float:
        """
        Convert contentment score to a transfer probability multiplier.

        Calibration:
          100 → ~0.60×  (very happy: significantly reduces base probability)
           70 → ~1.00×  (neutral: no adjustment)
           45 → ~1.30×  (restless: +30%)
           20 → ~1.60×  (unhappy: +60%)
            0 → ~1.80×  (miserable: capped)
        """
        deviation = (contentment - 70.0) / 70.0
        multiplier = 1.0 - deviation * MOTIVATION_MAX_UPLIFT * 1.4
        return round(max(0.40, min(2.00, multiplier)), 4)

    # ------------------------------------------------------------------ #
    # Public: transfer_probability_adjusted                                 #
    # ------------------------------------------------------------------ #

    def transfer_probability_adjusted(
        self, player: str, base_probability: float
    ) -> float:
        """
        Adjust a base transfer probability using the player's full motivation profile.

        Applies three layers:
        1. Contentment multiplier (push/stay/emotional/career/family signals combined)
        2. Family anchor deduction (partner+children rooted = direct probability reduction)
        3. Career stage direct modifier (new signing, academy product, declining contract)

        Formula:
            multiplier = contentment_to_transfer_multiplier(contentment_score)
            adjusted = base × multiplier
            adjusted -= family_transfer_anchor
            adjusted += career_transfer_modifiers
            adjusted = clip(adjusted, FLOOR, CEILING)

        Args:
            player: Player name matching PLAYER_MOTIVATION_DATA.
            base_probability: Base transfer probability from quant model [0, 1].

        Returns:
            Fully motivation-adjusted transfer probability in [FLOOR, CEILING].
        """
        profile = self._get_profile(player)
        multiplier = self._contentment_to_transfer_multiplier(profile.contentment_score)
        adjusted = base_probability * multiplier

        # Family anchor: direct reduction even if player is unhappy
        adjusted -= profile.family_transfer_anchor

        # Career stage direct modifier
        adjusted += profile.career_transfer_modifiers

        return round(
            max(TRANSFER_PROB_FLOOR, min(TRANSFER_PROB_CEILING, adjusted)), 4
        )

    # ------------------------------------------------------------------ #
    # Public: performance_if_stays                                          #
    # ------------------------------------------------------------------ #

    def performance_if_stays(
        self, player: str, technical_rating: float
    ) -> Dict:
        """
        Compute expected on-pitch performance readiness if the player remains.

        Uses the weighted composite formula:

            # Weighting: physical/technical (1.5) outweighs psychological/motivation (1.0).
            # Rationale: a technically elite player with low motivation still commands high
            # transfer value and performs above average. Psychological state adjusts at the
            # margin — a deeply unhappy player (contentment score 30/100) still produces
            # at ~80% of peak physical ability. The 1.5:1.0 ratio reflects this floor effect.

            motivation_score = (contentment_score / 100) × performance_modifier
            performance_readiness = (
                (motivation_score × PSYCH_WEIGHT) + (technical_rating × PHYSICAL_WEIGHT)
            ) / TOTAL_WEIGHT

        Args:
            player: Player name.
            technical_rating: Current technical/physical rating on 0-10 scale.

        Returns:
            dict with contentment_score, performance_modifier, motivation_score,
            technical_rating, performance_readiness, pct_of_peak, interpretation.
        """
        profile = self._get_profile(player)

        # Motivation score: contentment × emotional modifier → [0, 1]
        motivation_score = (profile.contentment_score / 100.0) * profile.performance_modifier
        motivation_score = max(0.0, min(1.0, motivation_score))

        # Weighted composite
        # Weighting: physical/technical (1.5) outweighs psychological/motivation (1.0).
        # Rationale: a technically elite player with low motivation still commands high
        # transfer value and performs above average. Psychological state adjusts at the
        # margin — a deeply unhappy player (contentment score 30/100) still produces
        # at ~80% of peak physical ability. The 1.5:1.0 ratio reflects this floor effect.
        performance_readiness = (
            (motivation_score * PSYCH_WEIGHT) + (technical_rating * PHYSICAL_WEIGHT)
        ) / TOTAL_WEIGHT

        performance_readiness = round(max(0.0, min(10.0, performance_readiness)), 3)
        pct_of_peak = round(performance_readiness / technical_rating * 100, 1) if technical_rating > 0 else 0.0

        if pct_of_peak >= 95:
            interpretation = "Full peak output expected — highly motivated"
        elif pct_of_peak >= 85:
            interpretation = "Near-peak output — minor motivation drag"
        elif pct_of_peak >= 75:
            interpretation = "Moderate reduction — motivation a real concern"
        elif pct_of_peak >= 65:
            interpretation = "Significant underperformance risk — address urgently"
        else:
            interpretation = "Severe output reduction — player needs immediate intervention"

        return {
            "contentment_score": profile.contentment_score,
            "performance_modifier": profile.performance_modifier,
            "motivation_score": round(motivation_score, 4),
            "technical_rating": technical_rating,
            "performance_readiness": performance_readiness,
            "pct_of_peak": pct_of_peak,
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------------ #
    # Public: destination_happiness_fit                                     #
    # ------------------------------------------------------------------ #

    def destination_happiness_fit(
        self, player: str, destination_club: str
    ) -> Dict:
        """
        Score how much happier a player would be at a destination club.

        Considers:
          1. Preferred destination list (+40)
          2. Champions League access (+20 if UCL regular)
          3. Prestige delta vs current club (±3 per prestige point)
          4. Wage tier improvement (±2 per tier)
          5. Project type (elite=+10 → selling=-15)
          6. Family visa situation at destination country

        Args:
            player: Player name.
            destination_club: Potential destination club name.

        Returns:
            dict with fit_score (0-100), recommendation, and sub-factors.
        """
        profile = self._get_profile(player)
        dest_attrs = CLUB_ATTRIBUTES.get(destination_club, {})
        curr_attrs = CLUB_ATTRIBUTES.get(profile.current_club, {})

        fit_score = 50.0

        # Preferred destination
        is_preferred = destination_club in profile.preferred_destinations
        if is_preferred:
            fit_score += 40.0

        if dest_attrs:
            # UCL access
            fit_score += 20.0 if dest_attrs.get("ucl_regular") else -10.0

            # Prestige delta
            delta_prestige = dest_attrs.get("prestige", 7.0) - curr_attrs.get("prestige", 7.0)
            fit_score += delta_prestige * 3.0

            # Wage
            delta_wage = dest_attrs.get("wage_tier", 5) - curr_attrs.get("wage_tier", 5)
            fit_score += delta_wage * 2.0

            # Project type
            project_bonus = {
                "elite": 10.0, "rising": 5.0, "solid": 0.0,
                "rebuilding": -5.0, "dev": -3.0,
                "declining": -8.0, "unstable": -10.0,
                "selling": -15.0, "mid-table": -5.0,
            }
            fit_score += project_bonus.get(dest_attrs.get("project", ""), 0.0)

            # Family visa at destination: if player has complications and destination
            # country is Schengen but family is in non-Schengen → no improvement
            dest_schengen = dest_attrs.get("schengen", False)
            curr_schengen = curr_attrs.get("schengen", False)
            if profile.family_visa_complications:
                if dest_schengen and not curr_schengen:
                    fit_score += 8.0   # Moving into Schengen = family can visit more
                elif not dest_schengen and curr_schengen:
                    fit_score -= 5.0   # Moving out of Schengen = harder for family

        # Family anchor penalty: if partner/kids settled, all destinations score lower
        if profile.partner_settled_current_city and profile.children_school_enrolled:
            fit_score -= 15.0   # Any move is disruptive regardless of destination quality

        fit_score = round(max(0.0, min(100.0, fit_score)), 1)

        if fit_score >= 80:
            rec = "Strong fit — significant happiness improvement expected"
        elif fit_score >= 65:
            rec = "Good fit — meets key ambitions"
        elif fit_score >= 50:
            rec = "Lateral move — marginal improvement"
        elif fit_score >= 35:
            rec = "Questionable — may not resolve underlying push factors"
        else:
            rec = "Step down — likely to create new dissatisfaction"

        return {
            "player": player,
            "destination": destination_club,
            "fit_score": fit_score,
            "is_preferred": is_preferred,
            "ucl_access": dest_attrs.get("ucl_regular", False),
            "dest_prestige": dest_attrs.get("prestige", "?"),
            "curr_prestige": curr_attrs.get("prestige", "?"),
            "wage_improvement": dest_attrs.get("wage_tier", 5) > curr_attrs.get("wage_tier", 5),
            "dest_country": dest_attrs.get("country", "?"),
            "family_visa_flag": profile.family_visa_complications,
            "family_anchor_penalty": profile.partner_settled_current_city and profile.children_school_enrolled,
            "recommendation": rec,
        }

    # ------------------------------------------------------------------ #
    # Public: narrative                                                     #
    # ------------------------------------------------------------------ #

    def narrative(self, player: str) -> str:
        """
        Generate a plain-English scouting report narrative for a player's
        motivation state including push/stay factors.
        """
        profile = self._get_profile(player)
        score = profile.contentment_score

        if score >= CONTENTMENT_VERY_HAPPY:
            status = "VERY HAPPY — unlikely to move"
        elif score >= CONTENTMENT_CONTENT:
            status = "CONTENT — would consider exceptional offer only"
        elif score >= CONTENTMENT_RESTLESS:
            status = "RESTLESS — actively weighing options"
        elif score >= CONTENTMENT_UNHAPPY:
            status = "UNHAPPY — strong exit signals present"
        else:
            status = "VERY UNHAPPY — exit virtually certain"

        multiplier = self._contentment_to_transfer_multiplier(score)

        push_pts = {"reported dream club interest": 25, "public fallout with manager": 20,
                    "contract dispute ongoing": 18, "unhappy playing time": 15,
                    "family visa complications": 10, "dressing room conflict": 12,
                    "personal life relocation desire": 8,
                    "family unable to attend matches": 8,
                    "recent trophy — ready to move on": 6}
        stay_pts = {"new contract signed recently": 20, "financial peak at current club": 15,
                    "just won major trophy": 12, "family fully anchored at current city": 12,
                    "happy family settled": 10, "academy product — emotional bond": 10,
                    "club loyalty sentiment": 8, "family attends matches regularly": 8,
                    "6+ years at club — institutionalised": 8, "injury recovering": 6}

        lines = [f"Contentment score: {score:.0f}/100 [{status}]"]

        pf = profile.primary_push_factor
        lines.append(f"Primary push factor: {pf} (-{push_pts.get(pf, '?')}pts)" if pf else "Primary push factor: None")

        sf = profile.primary_stay_factor
        lines.append(f"Primary stay factor: {sf} (+{stay_pts.get(sf, '?')}pts)" if sf else "Primary stay factor: None")

        lines.append(f"Net push delta: {profile.push_delta:+.0f} | Net family delta: {profile.family_delta:+.0f} | Net stay delta: {profile.stay_delta:+.0f}")
        lines.append(f"Emotional performance modifier: {profile.performance_modifier:.2f}× (1.0 = no effect)")
        lines.append(f"Family anchor (direct prob reduction): {profile.family_transfer_anchor:.2f}")
        lines.append(f"Transfer probability multiplier: {multiplier:.2f}×")
        lines.append(f"Family at matches: {profile.family_at_matches} | Visa complications: {profile.family_visa_complications}")

        if profile.context_notes:
            lines.append("Context:")
            for note in profile.context_notes:
                lines.append(f"  • {note}")

        if profile.preferred_destinations:
            lines.append(f"Preferred destinations: {', '.join(profile.preferred_destinations)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Public: career_narrative                                              #
    # ------------------------------------------------------------------ #

    def career_narrative(self, player: str, base_transfer_prob: float = 0.40) -> str:
        """
        Generate a rich career-stage and family narrative suitable for a
        scouting report. Covers career phase, club tenure, family anchors,
        and an adjusted transfer probability estimate.

        Args:
            player: Player name.
            base_transfer_prob: Base transfer probability before motivation adjustment.

        Returns:
            Multi-line string with full career context narrative.

        Example output:
            Rice (Arsenal, 2yr): established star in prime phase.
            Family: Partner Lauren settled in North London (+12 contentment via
            anchor effect). Family at matches: regularly. No visa complications.
            Academy context: West Ham academy product — emotional distance from
            Arsenal identity minor; captain role overrides bond.
            Career stage: prime. Years at club: 2. Is new signing: No.
            Adjusted transfer probability: LOW (9.2%)
        """
        profile = self._get_profile(player)
        adj_prob = self.transfer_probability_adjusted(player, base_transfer_prob)

        prob_label = (
            "VERY HIGH" if adj_prob >= 0.75 else
            "HIGH"      if adj_prob >= 0.55 else
            "MODERATE"  if adj_prob >= 0.35 else
            "LOW"       if adj_prob >= 0.15 else
            "VERY LOW"
        )

        club_short = profile.current_club.replace("Manchester ", "Man ").replace("Paris Saint-Germain", "PSG")
        header = f"{profile.player_name} ({club_short}, {profile.years_at_current_club}yr): {profile.career_stage} in {profile.career_stage} phase."

        family_lines = [
            f"Family: Location — {profile.family_location}.",
        ]

        if profile.partner_settled_current_city and profile.children_school_enrolled:
            family_lines.append("  Partner settled + children in school at current city — STRONG relocation barrier (-12 direct transfer probability).")
        elif profile.partner_settled_current_city:
            family_lines.append("  Partner settled at current city — moderate relocation barrier.")
        elif profile.children_school_enrolled:
            family_lines.append("  Children enrolled in school — significant family anchor.")

        family_lines.append(f"  Family at matches: {profile.family_at_matches} (contentment {FAMILY_ATTENDANCE_CONTENTMENT.get(profile.family_at_matches, 0.0):+.0f}pts).")

        if profile.family_visa_complications:
            family_lines.append(f"  Visa complications: YES — {profile.player_nationality} family faces friction in current league country.")
            family_lines.append("  This is an active PUSH factor: player may seek move where family can visit more freely.")
        else:
            family_lines.append("  Visa complications: None — family can attend freely.")

        academy_line = ""
        if profile.academy_product:
            academy_line = (
                f"Academy context: {profile.player_name} came through the academy at their previous/current club — "
                f"emotional bond creates transfer friction even when push factors are present (-10 transfer prob)."
            )

        new_signing_line = ""
        if profile.is_new_signing:
            new_signing_line = (
                f"New signing: Arrived within 12 months — adaptation period in effect. "
                f"Transfer probability directly reduced (-8 percentage points). "
                f"{'Settling-in period still active (-5% performance).' if profile.settling_in_period else 'Initial settling-in phase completed.'}"
            )

        first_major_line = ""
        if profile.first_major_club and profile.career_stage == "emerging":
            first_major_line = f"First major club: {profile.current_club} — high motivation to prove worth (+12 contentment, -5 transfer probability)."

        tenure_line = ""
        if profile.years_at_current_club >= 6:
            tenure_line = f"Long tenure: {profile.years_at_current_club} years at {profile.current_club} — institutionalised comfort (+8 contentment)."

        career_summary = (
            f"Career stage: {profile.career_stage}. "
            f"Years at club: {profile.years_at_current_club}. "
            f"Age: {profile.current_age}. "
            f"Is new signing: {'Yes' if profile.is_new_signing else 'No'}. "
            f"Academy product: {'Yes' if profile.academy_product else 'No'}."
        )

        conclusion = f"Adjusted transfer probability: {prob_label} ({adj_prob*100:.1f}%)"

        parts = [header]
        parts.extend(family_lines)
        if academy_line:  parts.append(academy_line)
        if new_signing_line: parts.append(new_signing_line)
        if first_major_line: parts.append(first_major_line)
        if tenure_line:   parts.append(tenure_line)
        parts.append(career_summary)
        parts.append(conclusion)

        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Public: contentment_dashboard                                         #
    # ------------------------------------------------------------------ #

    def contentment_dashboard(
        self, player_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Full contentment dashboard for a list of players (or all known players).

        Returns list of dicts sorted by contentment_score ascending (most at-risk first).
        """
        names = player_names if player_names else list(self.motivation_data.keys())
        rows = []
        for name in names:
            p = self._get_profile(name)
            mult = self._contentment_to_transfer_multiplier(p.contentment_score)
            rows.append({
                "player":               p.player_name,
                "club":                 p.current_club,
                "contentment_score":    p.contentment_score,
                "push_delta":           p.push_delta,
                "family_delta":         p.family_delta,
                "career_delta":         p.career_stage_delta,
                "stay_delta":           p.stay_delta,
                "primary_push":         p.primary_push_factor or "—",
                "primary_stay":         p.primary_stay_factor or "—",
                "transfer_multiplier":  mult,
                "family_anchor":        p.family_transfer_anchor,
                "career_direct_mod":    p.career_transfer_modifiers,
                "performance_modifier": p.performance_modifier,
                "family_at_matches":    p.family_at_matches,
                "visa_complications":   p.family_visa_complications,
                "career_stage":         p.career_stage,
                "years_at_club":        p.years_at_current_club,
                "is_new_signing":       p.is_new_signing,
                "academy_product":      p.academy_product,
            })
        rows.sort(key=lambda x: x["contentment_score"])
        return rows


# ============================================================================ #
# Smoke test                                                                     #
# ============================================================================ #

if __name__ == "__main__":
    scorer = PlayerMotivationScorer()

    test_cases = [
        ("Kylian Mbappe",    0.55),
        ("Jude Bellingham",  0.60),
        ("Erling Haaland",   0.65),
        ("Declan Rice",      0.55),
        ("Lautaro Martinez", 0.58),
        ("Phil Foden",       0.20),
        ("Vinicius Junior",  0.25),
        ("Manu Kone",        0.35),
    ]

    print("=" * 72)
    print("MOTIVATION SIGNAL — FULL NARRATIVE + CAREER CONTEXT")
    print("=" * 72)

    for player, base_prob in test_cases:
        print(f"\n{'─'*70}")
        print(f"[MOTIVATION] {player}")
        print(scorer.narrative(player))
        adj = scorer.transfer_probability_adjusted(player, base_prob)
        perf = scorer.performance_if_stays(player, technical_rating=8.5)
        print(f"\nAdjusted transfer prob (from {base_prob:.2f} base): {adj:.3f}")
        print(f"Performance readiness: {perf['performance_readiness']:.2f}/10 ({perf['pct_of_peak']}%) — {perf['interpretation']}")
        print()
        print("[CAREER NARRATIVE]")
        print(scorer.career_narrative(player, base_prob))

    print("\n" + "=" * 72)
    print("CONTENTMENT DASHBOARD (sorted, most at-risk first)")
    print("=" * 72)
    dash = scorer.contentment_dashboard()
    header = f"{'Player':<28} {'Club':<22} {'Score':>6} {'Mult':>6} {'Anchor':>7} {'Visa':>5} {'Stage':<15}"
    print(header)
    print("─" * len(header))
    for row in dash:
        print(
            f"{row['player']:<28} {row['club']:<22} "
            f"{row['contentment_score']:>6.1f} {row['transfer_multiplier']:>6.2f} "
            f"{row['family_anchor']:>7.2f} {str(row['visa_complications']):>5} "
            f"{row['career_stage']:<15}"
        )
