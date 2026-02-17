"""DuckDB storage layer for demand data."""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "energy.duckdb"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS demand (
    timestamp TIMESTAMPTZ NOT NULL,
    region VARCHAR NOT NULL,
    demand_mwh DOUBLE NOT NULL,
    PRIMARY KEY (timestamp, region)
)
"""


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection, creating the database and table if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    conn.execute(CREATE_TABLE_SQL)
    return conn


def upsert_demand(df: pd.DataFrame) -> int:
    """Insert demand data, skipping rows that already exist (idempotent).

    Args:
        df: DataFrame with columns: timestamp, region, demand_mwh.

    Returns:
        Number of new rows inserted.
    """
    conn = get_connection()

    # Register the DataFrame as a temporary view
    conn.register("incoming", df)

    # Count existing rows before insert
    before = conn.execute("SELECT COUNT(*) FROM demand").fetchone()[0]

    # Insert only rows that don't already exist
    conn.execute("""
        INSERT INTO demand (timestamp, region, demand_mwh)
        SELECT i.timestamp, i.region, i.demand_mwh
        FROM incoming i
        WHERE NOT EXISTS (
            SELECT 1 FROM demand d
            WHERE d.timestamp = i.timestamp AND d.region = i.region
        )
    """)

    after = conn.execute("SELECT COUNT(*) FROM demand").fetchone()[0]
    inserted = after - before

    conn.unregister("incoming")
    conn.close()

    logger.info("Upserted: %d new rows (%d total)", inserted, after)
    return inserted


def load_demand(
    region: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load demand data from DuckDB with optional filters.

    Args:
        region: Filter by region code.
        start: Filter start datetime (inclusive).
        end: Filter end datetime (inclusive).

    Returns:
        DataFrame with columns: timestamp, region, demand_mwh.
    """
    conn = get_connection()

    query = "SELECT timestamp, region, demand_mwh FROM demand WHERE 1=1"
    params: list = []

    if region:
        query += " AND region = ?"
        params.append(region)
    if start:
        query += " AND timestamp >= ?"
        params.append(start)
    if end:
        query += " AND timestamp <= ?"
        params.append(end)

    query += " ORDER BY timestamp"

    df = conn.execute(query, params).fetchdf()
    conn.close()

    logger.info("Loaded %d rows from DuckDB", len(df))
    return df
