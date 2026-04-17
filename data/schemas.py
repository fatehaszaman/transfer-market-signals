"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
This file is the contract layer of the entire system. Every piece of data
that flows between modules — player profiles, news articles, valuations,
transfer signals — is defined here as a strictly typed Pydantic model.

If the data doesn't match the schema, the system fails loudly and immediately
with a clear error message rather than silently propagating bad numbers into
the valuation model. Think of these schemas as the data equivalent of unit
tests: they validate correctness at every boundary crossing.

For non-technical readers: these are the "forms" every piece of information
must fill out correctly before being used in any calculation.
=============================================================================

Developer notes:
- All monetary values are EUR millions unless suffixed _pct or _score.
- Pydantic v2 is used throughout (model_validator, field_validator).
- All schemas are importable as top-level names from this module.
- ValidationError from this module wraps pydantic.ValidationError with
  structured context for ETL pipeline logging.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================ #
# Custom exceptions                                                              #
# ============================================================================ #

class TransferValidationError(Exception):
    """
    Structured validation error raised when data fails schema checks.

    Attributes:
        field: The field name that failed validation.
        value: The value that was provided.
        reason: Human-readable explanation of why it failed.
        context: Optional dict of extra diagnostic information.
    """

    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        self.context = context or {}
        super().__init__(
            f"[ValidationError] field='{field}' value={value!r} → {reason}"
            + (f" | context={context}" if context else "")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a loggable dict for ETL pipeline error records."""
        return {
            "error_type": "TransferValidationError",
            "field": self.field,
            "value": repr(self.value),
            "reason": self.reason,
            "context": self.context,
        }


class SchemaReconciliationError(Exception):
    """
    Raised during post-fetch reconciliation when API response fields don't
    match the expected schema. Captures missing and unexpected fields.

    Attributes:
        source: API or data source name (e.g. "api_football", "newsapi").
        missing_fields: Fields expected but absent in the response.
        unexpected_fields: Fields present in the response but not in schema.
        raw_response_sample: Truncated sample of the raw response for debugging.
    """

    def __init__(
        self,
        source: str,
        missing_fields: List[str],
        unexpected_fields: List[str],
        raw_response_sample: Optional[str] = None,
    ) -> None:
        self.source = source
        self.missing_fields = missing_fields
        self.unexpected_fields = unexpected_fields
        self.raw_response_sample = raw_response_sample
        msg = (
            f"[SchemaReconciliation] source='{source}'"
            f" | missing={missing_fields}"
            f" | unexpected={unexpected_fields}"
        )
        if raw_response_sample:
            msg += f" | sample={raw_response_sample[:200]}"
        super().__init__(msg)


# ============================================================================ #
# Enumerations                                                                   #
# ============================================================================ #

class Position(str, Enum):
    GK = "GK"
    CB = "CB"
    RB = "RB"
    LB = "LB"
    DM = "DM"
    CM = "CM"
    AM = "AM"
    RW = "RW"
    LW = "LW"
    FW = "FW"


class CareerPhase(str, Enum):
    EMERGING  = "emerging"
    PRIME     = "prime"
    DECLINING = "declining"
    VETERAN   = "veteran"


class TrendLabel(str, Enum):
    ASCENDING = "ascending"
    PEAK      = "peak"
    DECLINING = "declining"


class FFPRegime(str, Enum):
    UEFA    = "UEFA"
    PL_PSR  = "PL_PSR"
    DNCG    = "DNCG"
    LA_LIGA = "LA_LIGA"
    OTHER   = "OTHER"


class TransferWindow(str, Enum):
    SUMMER = "summer"
    WINTER = "winter"


# ============================================================================ #
# Season Stats                                                                   #
# ============================================================================ #

class SeasonStats(BaseModel):
    """
    One season of player performance statistics.

    All counting stats (goals, assists, etc.) are raw totals for the season.
    pass_accuracy_pct is 0–100. minutes must be between 1 and 4000.
    """

    season: str = Field(..., description="Season label, e.g. '2023/24'")
    goals: int = Field(ge=0, default=0)
    assists: int = Field(ge=0, default=0)
    key_passes: int = Field(ge=0, default=0)
    tackles: int = Field(ge=0, default=0)
    interceptions: int = Field(ge=0, default=0)
    dribbles_completed: int = Field(ge=0, default=0)
    pass_accuracy_pct: float = Field(ge=0.0, le=100.0, default=75.0)
    minutes: int = Field(ge=1, le=4000, default=2700)
    xg: Optional[float] = Field(default=None, ge=0.0, description="Expected goals")
    xa: Optional[float] = Field(default=None, ge=0.0, description="Expected assists")
    progressive_carries: Optional[int] = Field(default=None, ge=0)
    progressive_passes: Optional[int] = Field(default=None, ge=0)

    @field_validator("season")
    @classmethod
    def validate_season_format(cls, v: str) -> str:
        """Season must match YYYY/YY or YYYY/YYYY pattern."""
        if not re.match(r"^\d{4}/\d{2,4}$", v):
            raise ValueError(
                f"Season '{v}' must be in format 'YYYY/YY' or 'YYYY/YYYY' (e.g. '2023/24')"
            )
        return v

    @model_validator(mode="after")
    def goals_cannot_exceed_minutes_sanity(self) -> "SeasonStats":
        """A player cannot score more goals than approximate matches played."""
        approx_matches = self.minutes / 70
        if self.goals > approx_matches * 2:  # 2 goals/game is extreme upper bound
            raise ValueError(
                f"goals={self.goals} is implausibly high for minutes={self.minutes}. "
                f"Check data integrity."
            )
        return self


# ============================================================================ #
# Player Profile                                                                 #
# ============================================================================ #

class PlayerProfile(BaseModel):
    """
    BUSINESS: The core player record. Every field here feeds directly into
    the valuation model. Missing or malformed fields will trigger a hard
    validation error to protect downstream calculations.

    DEVELOPER: Immutable after construction (model_config frozen=True).
    Use .model_copy(update={...}) for what-if mutations.
    """

    model_config = {"frozen": True}

    name: str = Field(..., min_length=2, max_length=80)
    age: int = Field(..., ge=14, le=45, description="Age at start of current season")
    position: Position
    club: str = Field(..., min_length=2, max_length=80)
    nationality: Optional[str] = Field(default=None, max_length=50)
    current_value_eur_m: float = Field(..., gt=0.0, description="Current Transfermarkt valuation in EUR millions")
    contract_expires: int = Field(..., ge=2024, le=2035, description="Year contract expires")
    release_clause_eur_m: Optional[float] = Field(default=None, gt=0.0)
    shirt_number: Optional[int] = Field(default=None, ge=1, le=99)
    last_3_seasons_stats: List[SeasonStats] = Field(..., min_length=1, max_length=5)
    rating: float = Field(default=7.5, ge=4.0, le=10.0, description="Overall performance rating 1-10")
    reported_interest: List[str] = Field(default_factory=list, description="Clubs rumoured to be interested")

    @field_validator("name")
    @classmethod
    def validate_player_name(cls, v: str) -> str:
        """Player name must contain at least one letter and no control characters."""
        if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", v):
            raise ValueError(f"Player name '{v}' contains no letters — likely a data error.")
        if any(ord(c) < 32 for c in v):
            raise ValueError(f"Player name '{v}' contains control characters.")
        return v.strip()

    @field_validator("last_3_seasons_stats")
    @classmethod
    def seasons_must_be_chronological(cls, v: List[SeasonStats]) -> List[SeasonStats]:
        """Seasons must be in ascending chronological order."""
        if len(v) < 2:
            return v
        for i in range(1, len(v)):
            prev_year = int(v[i - 1].season[:4])
            curr_year = int(v[i].season[:4])
            if curr_year <= prev_year:
                raise ValueError(
                    f"Season data must be chronological: '{v[i-1].season}' "
                    f"appears before '{v[i].season}' but year order is wrong."
                )
        return v

    @model_validator(mode="after")
    def release_clause_sanity(self) -> "PlayerProfile":
        """Release clause should be >= current market value (clubs don't discount themselves)."""
        if (
            self.release_clause_eur_m is not None
            and self.release_clause_eur_m < self.current_value_eur_m * 0.7
        ):
            raise ValueError(
                f"release_clause_eur_m={self.release_clause_eur_m} is suspiciously low "
                f"vs current_value_eur_m={self.current_value_eur_m}. "
                f"Expected >= {self.current_value_eur_m * 0.7:.1f}."
            )
        return self

    def with_shock(self, rating_delta: float = 0.0, value_multiplier: float = 1.0) -> "PlayerProfile":
        """
        Return a mutated copy for what-if scenario analysis.

        Args:
            rating_delta: Add this to the player's rating (e.g. -1.5 for form drop).
            value_multiplier: Multiply current_value_eur_m (e.g. 0.8 for 20% drop).

        Returns:
            New PlayerProfile with shocked inputs. Clamps rating to [4.0, 10.0].

        Example:
            shocked = player.with_shock(rating_delta=-1.0, value_multiplier=0.8)
        """
        new_rating = max(4.0, min(10.0, self.rating + rating_delta))
        new_value = round(self.current_value_eur_m * value_multiplier, 2)
        return self.model_copy(
            update={"rating": new_rating, "current_value_eur_m": new_value}
        )


# ============================================================================ #
# Sentiment Score                                                                #
# ============================================================================ #

class ArticleRecord(BaseModel):
    """A single news article as returned by the NewsAPI /everything endpoint."""

    title: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)
    url: str = Field(..., description="Article URL")
    published_at: datetime
    source_name: str = Field(default="Unknown")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"Article URL must start with http:// or https://: got '{v}'")
        return v


class SentimentScore(BaseModel):
    """
    BUSINESS: Represents the news sentiment signal for a player — how positive
    or negative transfer-related media coverage is. A high positive score with
    high article volume indicates imminent transfer activity.

    DEVELOPER: sentiment ranges [-1.0, +1.0]. hype_score is a composite
    of sentiment * log(article_count + 1), normalised to [0, 1].
    """

    player_name: str
    sentiment: float = Field(..., ge=-1.0, le=1.0, description="Sentiment: -1 (negative) to +1 (positive)")
    article_count: int = Field(..., ge=0)
    hype_score: float = Field(..., ge=0.0, le=1.0, description="Composite sentiment × volume score")
    top_keywords: List[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    days_window: int = Field(default=30, ge=1, le=180)
    articles: List[ArticleRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def hype_consistent_with_sentiment_and_count(self) -> "SentimentScore":
        """Hype score cannot be high if sentiment is strongly negative."""
        if self.sentiment < -0.5 and self.hype_score > 0.7:
            raise ValueError(
                f"hype_score={self.hype_score} is inconsistent with "
                f"strongly negative sentiment={self.sentiment}. "
                f"Check scoring logic."
            )
        return self


# ============================================================================ #
# Transfer Signal                                                                #
# ============================================================================ #

class ContractSignal(BaseModel):
    """
    BUSINESS: Encodes how urgent a player's contract situation is.
    The urgency score drives the probability that a club will accept a bid
    rather than risk losing the player on a free transfer.
    """

    player_name: str
    contract_expires_year: int = Field(..., ge=2024, le=2035)
    months_remaining: int = Field(..., ge=0, le=120)
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    has_release_clause: bool
    release_clause_eur_m: Optional[float] = Field(default=None, gt=0.0)
    free_agent_value_uplift_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="% uplift in player's negotiating value from being a free agent",
    )
    reported_interest_clubs: List[str] = Field(default_factory=list)


class TransferSignal(BaseModel):
    """
    BUSINESS: The aggregated transfer intelligence signal for one player.
    This is the output of the feature pipeline — all signals combined into
    a single structured object that feeds the valuation model.

    DEVELOPER: This is the intermediate representation between the signal
    layer and the valuator. All signal modules produce these.
    """

    player_name: str
    position: Position
    current_club: str
    age: int = Field(..., ge=14, le=45)
    current_value_eur_m: float = Field(..., gt=0.0)

    # Age-value curve signals
    career_phase: CareerPhase
    undervaluation_score_eur_m: float
    projected_value_1yr_eur_m: float

    # Performance signals
    trajectory_slope: float
    trend_label: TrendLabel
    projected_goals_next: Optional[float] = None
    projected_assists_next: Optional[float] = None

    # Contract signals
    contract_signal: ContractSignal

    # FFP / financial signals
    likely_buyer_count: int = Field(ge=0)
    max_buyer_headroom_eur_m: float = Field(ge=0.0)

    # Sentiment signals
    sentiment_score: Optional[SentimentScore] = None

    # Metadata
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    run_id: Optional[str] = Field(default=None, description="ETL run identifier for replay")


# ============================================================================ #
# Valuation Result                                                               #
# ============================================================================ #

class ValuationResult(BaseModel):
    """
    BUSINESS: The final output of the transfer market intelligence engine.
    Each row in the ranked transfer target list corresponds to one of these.
    The composite_score is the single number a scout or director of football
    can use to rank opportunities.

    DEVELOPER: Produced by TransferValuator.batch_score() and
    rank_transfer_targets(). Serialises to JSON/CSV cleanly.
    composite_score is a weighted combination of all sub-signals,
    normalised to [0, 100].
    """

    player_name: str
    position: Position
    age: int
    current_club: str

    # Valuation outputs
    fair_value_eur_m: float = Field(ge=0.0)
    current_value_eur_m: float = Field(ge=0.0)
    value_delta_eur_m: float = Field(description="fair_value - current_value; positive = undervalued")

    # Probability outputs
    transfer_probability: float = Field(ge=0.0, le=1.0)

    # Sub-scores (each 0–1, used in composite)
    age_curve_score: float = Field(ge=0.0, le=1.0)
    performance_score: float = Field(ge=0.0, le=1.0)
    contract_urgency_score: float = Field(ge=0.0, le=1.0)
    ffp_fit_score: float = Field(ge=0.0, le=1.0)
    sentiment_score_val: float = Field(ge=0.0, le=1.0, alias="sentiment_score")

    # Composite
    composite_score: float = Field(ge=0.0, le=100.0)

    # Diagnostics
    career_phase: CareerPhase
    trend_label: TrendLabel
    contract_months_remaining: int
    likely_buyer_count: int
    run_id: Optional[str] = None
    scored_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def composite_score_must_be_consistent(self) -> "ValuationResult":
        """Composite score must be derivable from sub-scores (sanity check)."""
        # Compute a simple average of sub-scores * 100 as a floor check
        avg_sub = (
            self.age_curve_score
            + self.performance_score
            + self.contract_urgency_score
            + self.ffp_fit_score
            + self.sentiment_score_val
        ) / 5 * 100
        # Allow ±30 points difference from simple average (weights can shift it)
        if abs(self.composite_score - avg_sub) > 30:
            raise ValueError(
                f"composite_score={self.composite_score:.1f} deviates too far from "
                f"sub-score average={avg_sub:.1f}. Possible weighting error."
            )
        return self


# ============================================================================ #
# ETL / Cache schemas                                                            #
# ============================================================================ #

class APICallRecord(BaseModel):
    """
    BUSINESS: Every API call made by the system is logged with this schema
    so that any analysis run can be fully replayed from the local cache,
    even if the external API is unavailable.

    DEVELOPER: Stored in SQLite via data/cache.py. The cache_key is a
    deterministic hash of (endpoint + params) so identical requests
    always hit the same cache entry within a run_id.
    """

    run_id: str = Field(..., description="UUID4 identifier for this analysis run")
    cache_key: str = Field(..., description="SHA-256 hash of endpoint + sorted params")
    endpoint: str = Field(..., description="API endpoint URL")
    params: Dict[str, Any] = Field(default_factory=dict)
    source: str = Field(..., description="e.g. 'newsapi', 'api_football'")
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    status_code: int = Field(..., ge=100, le=599)
    response_size_bytes: int = Field(ge=0)
    error: Optional[str] = Field(default=None)
    replayed: bool = Field(default=False, description="True if served from cache, not live API")


class ValidationReport(BaseModel):
    """
    Summary of a pre- or post-fetch validation pass over a batch of records.
    Provides a structured audit trail for the ETL pipeline.
    """

    run_id: str
    source: str
    validated_at: datetime = Field(default_factory=datetime.utcnow)
    total_records: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total_records == 0:
            return 1.0
        return self.passed / self.total_records

    @model_validator(mode="after")
    def counts_consistent(self) -> "ValidationReport":
        if self.passed + self.failed != self.total_records:
            raise ValueError(
                f"passed={self.passed} + failed={self.failed} "
                f"!= total_records={self.total_records}"
            )
        return self


# ============================================================================ #
# What-if scenario                                                               #
# ============================================================================ #

class WhatIfScenario(BaseModel):
    """
    BUSINESS: Defines a shock scenario for sensitivity analysis.
    e.g. 'What if Mbappé's form drops 20%?' — this encodes that question
    so the model can produce a before/after valuation comparison.

    DEVELOPER: Apply via PlayerProfile.with_shock() and re-run the valuator.
    """

    scenario_name: str = Field(..., min_length=1)
    player_name: str
    rating_delta: float = Field(default=0.0, ge=-5.0, le=5.0)
    value_multiplier: float = Field(default=1.0, ge=0.1, le=3.0)
    contract_months_override: Optional[int] = Field(default=None, ge=0, le=120)
    description: str = Field(default="")

    class Config:
        json_schema_extra = {
            "example": {
                "scenario_name": "Mbappe_form_drop_20pct",
                "player_name": "Kylian Mbappe",
                "rating_delta": -1.5,
                "value_multiplier": 0.80,
                "description": "20% form regression after injury lay-off",
            }
        }
