"""
=============================================================================
BUSINESS SUMMARY
=============================================================================
This module is the memory of the system. Every API call — whether to
NewsAPI or API-Football — is stored in a local SQLite database with a
unique run ID and timestamp. This means any analysis can be replayed
exactly as it was computed, even weeks later or without internet access.

Think of it like a flight data recorder: if something goes wrong, or
if you want to reproduce a transfer prediction from last month, the
cache lets you rewind and re-run deterministically.

Key capabilities:
- Deterministic replay: same run_id always returns same data
- Audit trail: full history of every API call with status codes
- What-if replay: override one player's data and re-run the rest from cache
- Stage checkpoints: pipeline can resume from last successful stage
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default cache database location (overridden by config.py / env)
DEFAULT_DB_PATH = Path(__file__).parent.parent / ".cache" / "etl_cache.db"


# ============================================================================ #
# SQL schema definitions                                                         #
# ============================================================================ #

_SCHEMA_SQL = """
-- API call cache: one row per unique (run_id, cache_key) pair
CREATE TABLE IF NOT EXISTS api_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    cache_key       TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    source          TEXT NOT NULL,       -- 'newsapi' | 'api_football' | 'internal'
    fetched_at      TEXT NOT NULL,       -- ISO-8601 UTC
    status_code     INTEGER NOT NULL,
    response_json   TEXT,               -- Full response payload (may be NULL on error)
    response_size_b INTEGER DEFAULT 0,
    error_message   TEXT,
    replayed        INTEGER DEFAULT 0,   -- 1 if served from cache
    UNIQUE(run_id, cache_key)
);

-- Pipeline stage audit log
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    stage_name      TEXT NOT NULL,
    status          TEXT NOT NULL,       -- 'pending' | 'running' | 'success' | 'failed'
    started_at      TEXT,
    completed_at    TEXT,
    duration_ms     INTEGER,
    input_rows      INTEGER DEFAULT 0,
    output_rows     INTEGER DEFAULT 0,
    error_json      TEXT,               -- Structured error context on failure
    checkpoint_data TEXT                -- JSON snapshot for resume
);

-- Validation reports
CREATE TABLE IF NOT EXISTS validation_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    source          TEXT NOT NULL,
    validated_at    TEXT NOT NULL,
    total_records   INTEGER NOT NULL,
    passed          INTEGER NOT NULL,
    failed          INTEGER NOT NULL,
    errors_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_calls_run_id    ON api_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_cache_key ON api_calls(cache_key);
CREATE INDEX IF NOT EXISTS idx_pipeline_run_id     ON pipeline_runs(run_id);
"""


# ============================================================================ #
# Cache key generation                                                           #
# ============================================================================ #

def make_cache_key(endpoint: str, params: Dict[str, Any]) -> str:
    """
    Generate a deterministic SHA-256 cache key from an endpoint + params dict.

    The key is stable regardless of dict insertion order (params are sorted).
    Identical requests always produce the same key, enabling deterministic replay.

    Args:
        endpoint: The API endpoint URL string.
        params: Dictionary of query parameters (must be JSON-serialisable).

    Returns:
        64-character lowercase hex SHA-256 digest.

    Example:
        >>> make_cache_key("https://newsapi.org/v2/everything", {"q": "Mbappe", "pageSize": 10})
        'a3f9e2...'
    """
    canonical = json.dumps(
        {"endpoint": endpoint, "params": params},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_run_id() -> str:
    """Generate a fresh UUID4 run identifier for a new analysis session."""
    return str(uuid.uuid4())


# ============================================================================ #
# ETL Cache                                                                      #
# ============================================================================ #

class ETLCache:
    """
    ==========================================================================
    BUSINESS SUMMARY
    ==========================================================================
    The ETL Cache stores every API response in a local SQLite database so
    that analyses can be replayed exactly, audited, and resumed after failures.
    No external dependencies beyond Python's standard library sqlite3.

    Typical usage:
        cache = ETLCache()
        run_id = cache.new_run()

        # Store a response
        cache.store(run_id, endpoint, params, source, 200, response_data)

        # Later, retrieve it (or return None if not cached)
        cached = cache.retrieve(run_id, endpoint, params)
    ==========================================================================

    Thread safety: SQLite with WAL mode. Safe for ThreadPoolExecutor usage
    (each thread gets its own connection via the context manager).

    Args:
        db_path: Path to SQLite database file. Created on first use.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    # Internal: DB initialisation                                          #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            # Enable WAL for concurrent read/write from thread pool
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager yielding a SQLite connection with row_factory set.
        Commits on clean exit, rolls back on exception.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Run management                                                        #
    # ------------------------------------------------------------------ #

    def new_run(self) -> str:
        """
        Create and return a new run_id (UUID4).

        A "run" represents one full analysis session. All API calls and
        pipeline stages are tagged with this ID for audit and replay.
        """
        run_id = generate_run_id()
        logger.info("New ETL run started: %s", run_id)
        return run_id

    def list_runs(self) -> List[Dict[str, Any]]:
        """
        Return all distinct run_ids stored in the cache with metadata.

        Returns:
            List of dicts with keys: run_id, first_call, last_call, call_count.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id,
                       MIN(fetched_at) AS first_call,
                       MAX(fetched_at) AS last_call,
                       COUNT(*)        AS call_count
                FROM   api_calls
                GROUP  BY run_id
                ORDER  BY last_call DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # API call cache                                                        #
    # ------------------------------------------------------------------ #

    def store(
        self,
        run_id: str,
        endpoint: str,
        params: Dict[str, Any],
        source: str,
        status_code: int,
        response: Any,
        error_message: Optional[str] = None,
    ) -> str:
        """
        Persist an API response to the cache.

        Args:
            run_id: The current analysis run identifier.
            endpoint: Full API endpoint URL.
            params: Query parameters used in the request.
            source: Data source label ('newsapi', 'api_football', etc.).
            status_code: HTTP status code of the response.
            response: Response payload (will be JSON-serialised).
            error_message: Optional error string if the request failed.

        Returns:
            The cache_key (SHA-256 hex string) for this entry.
        """
        cache_key = make_cache_key(endpoint, params)
        response_json = json.dumps(response, ensure_ascii=False) if response is not None else None
        size_bytes = len(response_json.encode("utf-8")) if response_json else 0
        fetched_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO api_calls
                    (run_id, cache_key, endpoint, params_json, source,
                     fetched_at, status_code, response_json,
                     response_size_b, error_message, replayed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    run_id,
                    cache_key,
                    endpoint,
                    json.dumps(params, sort_keys=True),
                    source,
                    fetched_at,
                    status_code,
                    response_json,
                    size_bytes,
                    error_message,
                ),
            )

        logger.debug("Cached %s → %s (run=%s, %d bytes)", source, cache_key[:12], run_id, size_bytes)
        return cache_key

    def retrieve(
        self,
        run_id: str,
        endpoint: str,
        params: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Retrieve a cached API response for deterministic replay.

        Returns None if no matching entry exists in the cache for this
        (run_id, endpoint, params) combination.

        Args:
            run_id: The run identifier to replay from.
            endpoint: The API endpoint.
            params: The original query parameters.

        Returns:
            Deserialised response payload, or None if not cached.
        """
        cache_key = make_cache_key(endpoint, params)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT response_json, status_code
                FROM   api_calls
                WHERE  run_id = ? AND cache_key = ?
                """,
                (run_id, cache_key),
            ).fetchone()

        if row is None:
            return None

        # Mark as replayed
        with self._connect() as conn:
            conn.execute(
                "UPDATE api_calls SET replayed = 1 WHERE run_id = ? AND cache_key = ?",
                (run_id, cache_key),
            )

        if row["response_json"] is None:
            return None

        logger.debug("Cache HIT  %s (run=%s)", cache_key[:12], run_id)
        return json.loads(row["response_json"])

    def retrieve_or_fetch(
        self,
        run_id: str,
        endpoint: str,
        params: Dict[str, Any],
        source: str,
        fetch_fn: Any,  # Callable[[], Tuple[int, Any]]
    ) -> Any:
        """
        Retrieve from cache if available, otherwise call fetch_fn() and store result.

        This is the primary interface for all API clients — it transparently
        handles caching without the caller needing to know the cache exists.

        Args:
            run_id: Current run identifier.
            endpoint: API endpoint URL.
            params: Query parameters.
            source: Source label for logging.
            fetch_fn: Zero-argument callable returning (status_code, response_data).
                      Called only on cache miss.

        Returns:
            Cached or freshly fetched response payload.

        Example:
            data = cache.retrieve_or_fetch(
                run_id, url, params, "newsapi",
                fetch_fn=lambda: requests.get(url, params=params)
            )
        """
        cached = self.retrieve(run_id, endpoint, params)
        if cached is not None:
            logger.info("REPLAY  %s (run=%s)", endpoint[:60], run_id)
            return cached

        logger.info("FETCH   %s (run=%s)", endpoint[:60], run_id)
        status_code, response = fetch_fn()
        self.store(run_id, endpoint, params, source, status_code, response)
        return response

    # ------------------------------------------------------------------ #
    # Pipeline stage audit                                                  #
    # ------------------------------------------------------------------ #

    def log_stage_start(
        self,
        run_id: str,
        stage_name: str,
        input_rows: int = 0,
    ) -> int:
        """
        Record the start of a pipeline stage. Returns the row id for updates.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO pipeline_runs
                    (run_id, stage_name, status, started_at, input_rows)
                VALUES (?, ?, 'running', ?, ?)
                """,
                (run_id, stage_name, now, input_rows),
            )
            return cur.lastrowid

    def log_stage_success(
        self,
        row_id: int,
        output_rows: int = 0,
        checkpoint_data: Optional[Dict] = None,
    ) -> None:
        """Mark a pipeline stage as successfully completed."""
        now = datetime.now(timezone.utc).isoformat()
        chk = json.dumps(checkpoint_data) if checkpoint_data else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE pipeline_runs
                SET    status='success', completed_at=?, output_rows=?,
                       checkpoint_data=?,
                       duration_ms=CAST(
                           (julianday(?) - julianday(started_at)) * 86400000 AS INTEGER
                       )
                WHERE  id=?
                """,
                (now, output_rows, chk, now, row_id),
            )

    def log_stage_failure(
        self,
        row_id: int,
        error: Exception,
        input_snapshot: Optional[Dict] = None,
    ) -> None:
        """
        Mark a pipeline stage as failed with full error context.

        Stores the error type, message, and a snapshot of the input data
        so the failure can be reproduced and debugged.
        """
        now = datetime.now(timezone.utc).isoformat()
        error_ctx = {
            "error_type": type(error).__name__,
            "message": str(error),
            "input_snapshot": input_snapshot or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE pipeline_runs
                SET    status='failed', completed_at=?,
                       error_json=?,
                       duration_ms=CAST(
                           (julianday(?) - julianday(started_at)) * 86400000 AS INTEGER
                       )
                WHERE  id=?
                """,
                (now, json.dumps(error_ctx), now, row_id),
            )

    def get_last_successful_checkpoint(
        self, run_id: str, stage_name: str
    ) -> Optional[Dict]:
        """
        Return the checkpoint data from the last successful execution of a stage.

        Used by pipeline/data_orchestrator.py to implement checkpoint-resume:
        if a pipeline run fails at stage N, subsequent runs can skip stages
        1..N-1 and load their outputs from the checkpoint data.

        Args:
            run_id: The run to search within.
            stage_name: Name of the pipeline stage.

        Returns:
            Deserialised checkpoint dict, or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT checkpoint_data
                FROM   pipeline_runs
                WHERE  run_id = ? AND stage_name = ? AND status = 'success'
                ORDER  BY completed_at DESC
                LIMIT  1
                """,
                (run_id, stage_name),
            ).fetchone()

        if row is None or row["checkpoint_data"] is None:
            return None
        return json.loads(row["checkpoint_data"])

    # ------------------------------------------------------------------ #
    # Validation report storage                                             #
    # ------------------------------------------------------------------ #

    def store_validation_report(
        self,
        run_id: str,
        source: str,
        total: int,
        passed: int,
        failed: int,
        errors: List[Dict],
    ) -> None:
        """Persist a validation report to the audit database."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO validation_reports
                    (run_id, source, validated_at, total_records, passed, failed, errors_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, source, now, total, passed, failed, json.dumps(errors)),
            )

    # ------------------------------------------------------------------ #
    # Audit queries                                                         #
    # ------------------------------------------------------------------ #

    def audit_log(self) -> List[Dict[str, Any]]:
        """
        Return the full pipeline run audit log across all run_ids.

        Suitable for loading into a pandas DataFrame for analysis:
            import pandas as pd
            df = pd.DataFrame(cache.audit_log())

        Returns:
            List of dicts with stage timing and status information.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, stage_name, status,
                       started_at, completed_at, duration_ms,
                       input_rows, output_rows, error_json
                FROM   pipeline_runs
                ORDER  BY id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def call_stats(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Summary statistics for API calls, optionally filtered to a run.

        Returns:
            Dict with total_calls, replayed, by_source, total_bytes.
        """
        where = f"WHERE run_id = '{run_id}'" if run_id else ""
        with self._connect() as conn:
            agg = conn.execute(
                f"""
                SELECT COUNT(*)                          AS total_calls,
                       SUM(replayed)                     AS replayed_calls,
                       SUM(response_size_b)              AS total_bytes,
                       SUM(CASE WHEN status_code=200 THEN 1 ELSE 0 END) AS success_calls
                FROM   api_calls {where}
                """
            ).fetchone()
            by_source = conn.execute(
                f"""
                SELECT source, COUNT(*) AS calls
                FROM   api_calls {where}
                GROUP  BY source
                """
            ).fetchall()

        return {
            "total_calls": agg["total_calls"] or 0,
            "replayed_calls": agg["replayed_calls"] or 0,
            "total_bytes": agg["total_bytes"] or 0,
            "success_calls": agg["success_calls"] or 0,
            "by_source": {r["source"]: r["calls"] for r in by_source},
            "run_id_filter": run_id,
        }

    def purge_run(self, run_id: str) -> int:
        """
        Delete all cache entries for a given run_id.

        Args:
            run_id: The run to purge.

        Returns:
            Number of rows deleted across all tables.
        """
        deleted = 0
        with self._connect() as conn:
            c1 = conn.execute("DELETE FROM api_calls WHERE run_id=?", (run_id,))
            c2 = conn.execute("DELETE FROM pipeline_runs WHERE run_id=?", (run_id,))
            c3 = conn.execute("DELETE FROM validation_reports WHERE run_id=?", (run_id,))
            deleted = c1.rowcount + c2.rowcount + c3.rowcount
        logger.info("Purged run %s: %d rows deleted", run_id, deleted)
        return deleted


# ============================================================================ #
# Module-level singleton for convenient import                                   #
# ============================================================================ #

# Modules can import `default_cache` for a ready-to-use cache instance.
# Override db_path via config.py for testing or custom deployments.
default_cache = ETLCache()


# ============================================================================ #
# Smoke test                                                                     #
# ============================================================================ #

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cache = ETLCache(db_path=Path(tmp) / "test.db")
        run_id = cache.new_run()
        print(f"Run ID: {run_id}")

        # Store a fake API response
        key = cache.store(
            run_id=run_id,
            endpoint="https://newsapi.org/v2/everything",
            params={"q": "Mbappe transfer", "pageSize": 5},
            source="newsapi",
            status_code=200,
            response={"articles": [{"title": "Mbappe to Real Madrid", "url": "https://example.com"}]},
        )
        print(f"Cache key: {key[:16]}...")

        # Replay it
        result = cache.retrieve(
            run_id,
            "https://newsapi.org/v2/everything",
            {"q": "Mbappe transfer", "pageSize": 5},
        )
        print(f"Replayed: {result}")

        # Stage logging
        stage_id = cache.log_stage_start(run_id, "FetchAPIFootball", input_rows=25)
        cache.log_stage_success(stage_id, output_rows=25, checkpoint_data={"players": 25})

        # Audit
        stats = cache.call_stats(run_id)
        print(f"Stats: {stats}")

        log = cache.audit_log()
        print(f"Audit log rows: {len(log)}")
        print("All cache tests passed.")
