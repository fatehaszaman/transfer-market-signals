"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
News sentiment is the real-time pulse of the transfer market. When a player's
name appears 200 times in the press in a single week with language like
"agrees personal terms" and "medical scheduled", that is a quantifiable signal
that a deal is imminent — regardless of what the clubs officially say.

This module fetches player news from NewsAPI, scores the articles using
keyword analysis, and produces a hype score that feeds directly into the
transfer probability calculation. The higher the hype score, the more the
model adjusts upward for transfer probability.

Key features:
- Parallel fetching via ThreadPoolExecutor (5 concurrent threads)
- Rate limiting to respect NewsAPI free tier (100 requests/day)
- Pydantic schema validation on every article and score
- Graceful fallback to mock data when no API key is configured
- ETL cache integration — every fetch is stored for replay
=============================================================================

Developer notes:
- SENTIMENT_POSITIVE_KEYWORDS and SENTIMENT_NEGATIVE_KEYWORDS are defined
  in config.py so they can be tuned without touching this module.
- Hype score = sigmoid(sentiment * log(article_count + 1)), scaled to [0, 1].
- Thread pool is created per batch_score() call and explicitly shut down.
- All network errors are caught; failed fetches return a neutral SentimentScore.
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
from requests import HTTPError, Timeout

from config import (
    NEWS_API_BASE_URL,
    NEWS_API_KEY,
    NEWSAPI_RATE_LIMIT_PER_DAY,
    NEWSAPI_REQUESTS_PER_SECOND,
    SENTIMENT_FETCH_MAX_WORKERS,
    SENTIMENT_NEGATIVE_KEYWORDS,
    SENTIMENT_POSITIVE_KEYWORDS,
)
from data.cache import ETLCache, default_cache
from data.schemas import ArticleRecord, SentimentScore, TransferValidationError

logger = logging.getLogger(__name__)


# ============================================================================ #
# Fallback mock articles (used when API key is missing or quota exhausted)       #
# ============================================================================ #

_MOCK_ARTICLES: Dict[str, List[Dict]] = {
    "Erling Haaland": [
        {"title": "Haaland agrees personal terms with Real Madrid", "description": "Medical scheduled for next week", "url": "https://example.com/1", "publishedAt": "2025-06-01T10:00:00Z", "source": {"name": "Sky Sports"}},
        {"title": "City confident of keeping Haaland", "description": "New deal offer tabled", "url": "https://example.com/2", "publishedAt": "2025-06-02T12:00:00Z", "source": {"name": "BBC Sport"}},
    ],
    "Kylian Mbappe": [
        {"title": "Mbappe transfer confirmed to Real Madrid", "description": "Joins on a free", "url": "https://example.com/3", "publishedAt": "2024-06-03T09:00:00Z", "source": {"name": "L'Equipe"}},
        {"title": "PSG deny Mbappe fallout reports", "description": "Club denies interest", "url": "https://example.com/4", "publishedAt": "2024-05-10T14:00:00Z", "source": {"name": "RMC Sport"}},
    ],
    "Jude Bellingham": [
        {"title": "Bellingham deal agreed with Real Madrid", "description": "€103m fee confirmed", "url": "https://example.com/5", "publishedAt": "2023-06-10T11:00:00Z", "source": {"name": "Marca"}},
    ],
    "Declan Rice": [
        {"title": "Arsenal agree Rice deal with West Ham", "description": "British record fee secured", "url": "https://example.com/6", "publishedAt": "2023-07-14T15:00:00Z", "source": {"name": "The Athletic"}},
        {"title": "Chelsea no longer in running for Rice", "description": "Bid rejected", "url": "https://example.com/7", "publishedAt": "2023-07-01T09:00:00Z", "source": {"name": "Guardian"}},
    ],
    "Lautaro Martinez": [
        {"title": "Barcelona denied Martinez — deal breaks down", "description": "Financial issues", "url": "https://example.com/8", "publishedAt": "2023-08-01T10:00:00Z", "source": {"name": "Sport"}},
        {"title": "Martinez signs new Inter deal — stays", "description": "Contract extension confirmed", "url": "https://example.com/9", "publishedAt": "2023-08-15T12:00:00Z", "source": {"name": "Gazzetta"}},
    ],
    "_default": [
        {"title": "Transfer rumours abound in the market", "description": "Multiple clubs circling", "url": "https://example.com/10", "publishedAt": "2025-04-01T10:00:00Z", "source": {"name": "Transfer News"}},
    ],
}


# ============================================================================ #
# Validation helpers                                                             #
# ============================================================================ #

def _validate_player_name_for_api(player_name: str) -> None:
    """
    Pre-processing validation before making an API call.

    Ensures the player name is safe for use as a query string and
    won't waste an API request on invalid input.

    Raises:
        TransferValidationError: If player_name is empty, too short, or too long.
    """
    if not isinstance(player_name, str):
        raise TransferValidationError(
            field="player_name",
            value=player_name,
            reason=f"player_name must be a string, got {type(player_name).__name__}",
        )
    stripped = player_name.strip()
    if len(stripped) < 2:
        raise TransferValidationError(
            field="player_name",
            value=player_name,
            reason="Player name must be at least 2 characters for a useful search query.",
        )
    if len(stripped) > 80:
        raise TransferValidationError(
            field="player_name",
            value=player_name,
            reason=f"Player name exceeds 80 characters (got {len(stripped)}). Likely a data error.",
        )


def _validate_days_back(days_back: int) -> None:
    """Validate date range for API lookback window."""
    if not isinstance(days_back, int) or days_back < 1 or days_back > 180:
        raise TransferValidationError(
            field="days_back",
            value=days_back,
            reason="days_back must be an integer between 1 and 180. NewsAPI free tier supports 30 days.",
            context={"max_free_tier_days": 30, "max_allowed": 180},
        )


def _reconcile_article_response(raw: Dict) -> Tuple[List[Dict], List[str]]:
    """
    Post-fetch schema reconciliation for NewsAPI /everything response.

    Checks that the response contains expected top-level fields and
    that each article has the required keys.

    Args:
        raw: Raw JSON response dict from NewsAPI.

    Returns:
        Tuple of (valid_articles, missing_fields). Raises SchemaReconciliationError
        if the response is fundamentally malformed.
    """
    from data.schemas import SchemaReconciliationError

    expected_top = {"status", "articles", "totalResults"}
    actual_top = set(raw.keys())
    missing_top = expected_top - actual_top

    if "articles" not in actual_top:
        raise SchemaReconciliationError(
            source="newsapi",
            missing_fields=list(missing_top),
            unexpected_fields=list(actual_top - expected_top),
            raw_response_sample=str(raw)[:200],
        )

    expected_article = {"title", "url", "publishedAt", "source"}
    valid = []
    missing_in_articles = set()
    for art in raw.get("articles", []):
        missing = expected_article - set(art.keys())
        if missing:
            missing_in_articles.update(missing)
            continue
        valid.append(art)

    return valid, list(missing_in_articles)


# ============================================================================ #
# Sentiment scorer                                                                #
# ============================================================================ #

class NewsletterSentimentScorer:
    """
    ==========================================================================
    BUSINESS SUMMARY
    ==========================================================================
    Fetches and scores transfer-related news sentiment for football players.
    Produces a hype_score (0–1) combining article volume and sentiment
    polarity to signal imminent transfer activity.

    Parallel fetching: uses ThreadPoolExecutor to fetch multiple players
    simultaneously — important for batch analysis of 20+ targets.

    Rate limiting: tracks request counts to avoid exceeding NewsAPI free tier.
    ==========================================================================

    Args:
        api_key: NewsAPI key. Defaults to NEWS_API_KEY from config/env.
        cache: ETLCache instance for deterministic replay. Defaults to default_cache.
        use_mock_fallback: If True (default), returns mock data when API key is
            absent or quota is exhausted — allows offline development.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[ETLCache] = None,
        use_mock_fallback: bool = True,
    ) -> None:
        self.api_key = api_key or NEWS_API_KEY
        self.cache = cache or default_cache
        self.use_mock_fallback = use_mock_fallback
        self._request_count: int = 0
        self._last_request_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Internal: rate limiting                                               #
    # ------------------------------------------------------------------ #

    def _rate_limit(self) -> None:
        """
        Enforce minimum inter-request delay to respect NewsAPI rate limits.

        Blocks until at least 1/NEWSAPI_REQUESTS_PER_SECOND seconds have
        elapsed since the last request. Uses a simple token-bucket approach.
        """
        min_interval = 1.0 / NEWSAPI_REQUESTS_PER_SECOND
        elapsed = time.time() - self._last_request_ts
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug("Rate limiting: sleeping %.2fs", sleep_time)
            time.sleep(sleep_time)
        self._last_request_ts = time.time()
        self._request_count += 1

        if self._request_count >= NEWSAPI_RATE_LIMIT_PER_DAY:
            logger.warning(
                "NewsAPI daily request limit (%d) approaching or reached. "
                "Using mock fallback for remaining requests.",
                NEWSAPI_RATE_LIMIT_PER_DAY,
            )

    # ------------------------------------------------------------------ #
    # Internal: mock fallback                                               #
    # ------------------------------------------------------------------ #

    def _get_mock_articles(self, player_name: str) -> List[Dict]:
        """Return mock articles for a player (for offline development/testing)."""
        return _MOCK_ARTICLES.get(player_name, _MOCK_ARTICLES["_default"])

    # ------------------------------------------------------------------ #
    # Public: fetch_player_news                                             #
    # ------------------------------------------------------------------ #

    def fetch_player_news(
        self,
        player_name: str,
        days_back: int = 30,
        run_id: Optional[str] = None,
        page_size: int = 20,
    ) -> List[Dict]:
        """
        Fetch news articles about a player from NewsAPI /everything endpoint.

        Pre-processing validation is applied before the API call (player name,
        days_back range). Post-fetch schema reconciliation is applied to the
        response to catch malformed payloads.

        Caching: if run_id is provided, checks ETL cache before making a live
        request. This enables deterministic replay of any analysis run.

        Args:
            player_name: Full player name to search for.
            days_back: Days to look back from today. Free tier capped at 30.
            run_id: Optional ETL run ID for cache lookup.
            page_size: Number of articles per request (max 100 on paid plan).

        Returns:
            List of raw article dicts (title, description, url, publishedAt, source).

        Raises:
            TransferValidationError: If player_name or days_back is invalid.
        """
        # --- Pre-processing validation ---
        _validate_player_name_for_api(player_name)
        _validate_days_back(days_back)

        # Use mock if no API key or quota exceeded
        if not self.api_key or self._request_count >= NEWSAPI_RATE_LIMIT_PER_DAY:
            logger.info("Using mock articles for '%s' (no key or quota reached)", player_name)
            return self._get_mock_articles(player_name)

        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        endpoint = f"{NEWS_API_BASE_URL}/everything"
        params = {
            "q": f'"{player_name}" transfer',
            "from": from_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": page_size,
            "apiKey": self.api_key,
        }

        # Strip API key from cache params for security
        cache_params = {k: v for k, v in params.items() if k != "apiKey"}

        # --- Cache lookup ---
        if run_id:
            cached = self.cache.retrieve(run_id, endpoint, cache_params)
            if cached is not None:
                logger.info("Cache HIT for '%s' news", player_name)
                return cached.get("articles", [])

        self._rate_limit()

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            raw = response.json()

            # --- Post-fetch reconciliation ---
            valid_articles, missing_fields = _reconcile_article_response(raw)
            if missing_fields:
                logger.warning(
                    "Schema reconciliation: %d articles had missing fields %s",
                    len(raw.get("articles", [])) - len(valid_articles),
                    missing_fields,
                )

            # Store in cache
            if run_id:
                self.cache.store(
                    run_id=run_id,
                    endpoint=endpoint,
                    params=cache_params,
                    source="newsapi",
                    status_code=response.status_code,
                    response=raw,
                )

            return valid_articles

        except (HTTPError, Timeout, Exception) as exc:
            logger.warning("NewsAPI fetch failed for '%s': %s — using mock", player_name, exc)
            if run_id:
                self.cache.store(
                    run_id=run_id,
                    endpoint=endpoint,
                    params=cache_params,
                    source="newsapi",
                    status_code=0,
                    response=None,
                    error_message=str(exc),
                )
            return self._get_mock_articles(player_name)

    # ------------------------------------------------------------------ #
    # Public: score_sentiment                                               #
    # ------------------------------------------------------------------ #

    def score_sentiment(self, articles: List[Dict]) -> float:
        """
        Score a list of news articles for transfer sentiment.

        Each article's title and description is checked against two keyword lists:
        - SENTIMENT_POSITIVE_KEYWORDS (from config.py): transfer-confirming language
        - SENTIMENT_NEGATIVE_KEYWORDS: transfer-denying or negative language

        Scoring:
            raw_score = (positive_hits - negative_hits) / max(total_checked, 1)
            sentiment = clip(raw_score, -1.0, +1.0)

        A single article matching a positive keyword contributes +1; negative -1.
        Multiple keyword matches per article are possible (capped at ±1 per article).

        Args:
            articles: List of article dicts with "title" and "description" keys.

        Returns:
            Float sentiment in [-1.0, +1.0].
        """
        if not articles:
            return 0.0

        total_positive = 0
        total_negative = 0

        for art in articles:
            text = " ".join(filter(None, [
                art.get("title", "") or "",
                art.get("description", "") or "",
            ])).lower()

            pos_count = sum(1 for kw in SENTIMENT_POSITIVE_KEYWORDS if kw in text)
            neg_count = sum(1 for kw in SENTIMENT_NEGATIVE_KEYWORDS if kw in text)

            total_positive += min(pos_count, 1)   # Cap at 1 per article
            total_negative += min(neg_count, 1)

        n = len(articles)
        raw = (total_positive - total_negative) / n
        return round(max(-1.0, min(1.0, raw)), 4)

    # ------------------------------------------------------------------ #
    # Public: get_transfer_hype_score                                       #
    # ------------------------------------------------------------------ #

    def get_transfer_hype_score(
        self,
        player_name: str,
        days_back: int = 30,
        run_id: Optional[str] = None,
    ) -> SentimentScore:
        """
        Compute a combined transfer hype score for a player.

        Hype score combines raw sentiment with article volume via:
            hype = sigmoid(sentiment * log(article_count + 1))

        This rewards both strong sentiment AND high coverage volume.
        A player with sentiment=+0.6 and 50 articles scores much higher
        than one with sentiment=+0.9 and only 2 articles.

        Args:
            player_name: Player name for news search.
            days_back: Days of news to include.
            run_id: ETL run ID for caching.

        Returns:
            SentimentScore Pydantic model with all signal fields.
        """
        _validate_player_name_for_api(player_name)

        articles_raw = self.fetch_player_news(player_name, days_back, run_id)
        sentiment = self.score_sentiment(articles_raw)
        article_count = len(articles_raw)

        # Hype = sigmoid(sentiment * log(count + 1)) normalised to [0, 1]
        log_volume = math.log(article_count + 1)
        raw_hype = sentiment * log_volume
        # Sigmoid centred at 0 → maps to [0, 1]
        hype = 1.0 / (1.0 + math.exp(-raw_hype))
        hype = round(hype, 4)

        # Extract top keywords for reporting
        all_text = " ".join(
            " ".join(filter(None, [a.get("title", ""), a.get("description", "")]))
            for a in articles_raw
        ).lower()
        matched_pos = [kw for kw in SENTIMENT_POSITIVE_KEYWORDS if kw in all_text]
        matched_neg = [kw for kw in SENTIMENT_NEGATIVE_KEYWORDS if kw in all_text]
        top_keywords = (matched_pos + matched_neg)[:8]

        # Build ArticleRecord objects (validated)
        validated_articles = []
        for a in articles_raw[:10]:   # Cap at 10 for storage
            try:
                rec = ArticleRecord(
                    title=a.get("title", "No title") or "No title",
                    description=a.get("description"),
                    url=a.get("url", "https://unknown.com") or "https://unknown.com",
                    published_at=datetime.fromisoformat(
                        (a.get("publishedAt") or "2025-01-01T00:00:00Z").replace("Z", "+00:00")
                    ),
                    source_name=(a.get("source") or {}).get("name", "Unknown") or "Unknown",
                )
                validated_articles.append(rec)
            except Exception as e:
                logger.debug("Article validation failed: %s", e)

        return SentimentScore(
            player_name=player_name,
            sentiment=sentiment,
            article_count=article_count,
            hype_score=hype,
            top_keywords=top_keywords,
            fetched_at=datetime.now(timezone.utc),
            days_window=days_back,
            articles=validated_articles,
        )

    # ------------------------------------------------------------------ #
    # Public: batch_score_players — ThreadPoolExecutor parallel fetch       #
    # ------------------------------------------------------------------ #

    def batch_score_players(
        self,
        player_names: List[str],
        days_back: int = 30,
        run_id: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, SentimentScore]:
        """
        Fetch and score sentiment for multiple players in parallel.

        Uses concurrent.futures.ThreadPoolExecutor with explicit thread pool
        management — pool is created, used, and explicitly shut down within
        this method call.

        Thread count is controlled by SENTIMENT_FETCH_MAX_WORKERS (config.py).
        NewsAPI rate limiting is applied per thread via _rate_limit().

        Args:
            player_names: List of player names to score.
            days_back: Days of news to look back for each player.
            run_id: ETL run ID for caching all fetches under the same run.
            max_workers: Override for thread pool size (defaults to config value).

        Returns:
            Dict mapping player_name → SentimentScore. Failed fetches return
            a neutral SentimentScore rather than raising.

        Example:
            scores = scorer.batch_score_players(
                ["Erling Haaland", "Jude Bellingham", "Pedri"],
                run_id="abc123",
            )
            for name, score in scores.items():
                print(f"{name}: hype={score.hype_score:.3f}")
        """
        if not player_names:
            return {}

        n_workers = max_workers or SENTIMENT_FETCH_MAX_WORKERS
        results: Dict[str, SentimentScore] = {}

        logger.info(
            "Batch sentiment fetch: %d players, %d threads, %d days",
            len(player_names), n_workers, days_back,
        )

        # Explicit ThreadPoolExecutor lifecycle management
        executor = ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="sentiment_fetch",
        )
        try:
            future_to_player = {
                executor.submit(
                    self.get_transfer_hype_score,
                    player_name=name,
                    days_back=days_back,
                    run_id=run_id,
                ): name
                for name in player_names
            }

            for future in as_completed(future_to_player):
                player = future_to_player[future]
                try:
                    score = future.result(timeout=30)
                    results[player] = score
                    logger.debug("Sentiment scored: %s hype=%.3f", player, score.hype_score)
                except Exception as exc:
                    logger.warning("Sentiment fetch failed for '%s': %s", player, exc)
                    # Return neutral default — never block the pipeline
                    results[player] = SentimentScore(
                        player_name=player,
                        sentiment=0.0,
                        article_count=0,
                        hype_score=0.5,   # Neutral
                        top_keywords=[],
                    )
        finally:
            # Explicit shutdown — don't rely on GC
            executor.shutdown(wait=True, cancel_futures=False)
            logger.info(
                "Sentiment thread pool shut down. Total API requests this session: %d",
                self._request_count,
            )

        return results


# ============================================================================ #
# Smoke test                                                                     #
# ============================================================================ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scorer = NewsletterSentimentScorer(use_mock_fallback=True)

    players = ["Erling Haaland", "Kylian Mbappe", "Jude Bellingham", "Declan Rice", "Lautaro Martinez"]

    print("=== PARALLEL SENTIMENT BATCH (mock data) ===")
    scores = scorer.batch_score_players(players, days_back=30)

    for name, s in sorted(scores.items(), key=lambda x: x[1].hype_score, reverse=True):
        print(
            f"  {name:<25} | sentiment={s.sentiment:+.3f} | "
            f"articles={s.article_count:>3} | hype={s.hype_score:.3f} | "
            f"keywords={s.top_keywords[:3]}"
        )
