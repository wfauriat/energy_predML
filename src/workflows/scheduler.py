"""Scheduled orchestration: run ingestion daily and monitoring weekly.

Uses Prefect's serve() API — no Prefect server required, just a long-lived
process. Keep this running in a dedicated terminal (or background process)
for fully automated pipeline execution.

Usage:
    make local-scheduler          # via Makefile
    .venv/bin/python -m src.workflows.scheduler   # directly

Schedules:
    data-ingestion   — daily at 06:00 UTC
    drift-monitoring — every Monday at 06:00 UTC
"""

import logging

from prefect import serve

from src.workflows.ingest import ingest_flow
from src.workflows.monitor import run_monitoring

logger = logging.getLogger(__name__)


def main() -> None:
    """Start both scheduled flows and block until interrupted."""
    logger.info("Starting scheduler — press Ctrl+C to stop.")
    logger.info("  data-ingestion  : daily at 06:00 UTC")
    logger.info("  drift-monitoring: every Monday at 06:00 UTC")

    ingest_deployment = ingest_flow.to_deployment(
        name="daily-ingest",
        cron="0 6 * * *",
        description="Fetch latest EIA demand data and store in DuckDB.",
    )

    monitor_deployment = run_monitoring.to_deployment(
        name="weekly-monitor",
        cron="0 6 * * 1",
        description="Run drift detection; auto-retrain if drift threshold exceeded.",
    )

    serve(ingest_deployment, monitor_deployment)


if __name__ == "__main__":
    main()
