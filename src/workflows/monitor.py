"""Prefect flow for drift monitoring: compare reference vs current data."""

import logging

import pandas as pd

from src.config import settings
from src.features.pipeline import build_features
from src.models.xgboost_model import NON_FEATURE_COLS
from src.monitoring.drift import (
    generate_drift_report,
    generate_regression_report,
    save_report,
)

logger = logging.getLogger(__name__)

# Fraction of drifted features that triggers a retraining alert
DRIFT_THRESHOLD = 0.3


def run_monitoring(
    train_ratio: float = 0.8,
    region: str | None = None,
) -> dict:
    """Run drift monitoring by comparing training data (reference) vs recent data (current).

    Args:
        train_ratio: Fraction of data used for training (reference).
        region: Grid region code.

    Returns:
        Dictionary with drift detection results and report paths.
    """
    region = region or settings.DEFAULT_REGION

    # Build features from all available data
    df = build_features(region=region)

    split_idx = int(len(df) * train_ratio)
    reference_df = df.iloc[:split_idx].copy()
    current_df = df.iloc[split_idx:].copy()

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # 1. Data drift report
    logger.info("--- Generating data drift report ---")
    drift_report, drift_metrics = generate_drift_report(
        reference_df, current_df, feature_cols
    )
    drift_html = save_report(drift_report, "data_drift")

    # 2. Regression performance report (needs target + prediction columns)
    # Simulate predictions using lag_24h as a simple proxy
    logger.info("--- Generating regression report ---")
    ref_regression = pd.DataFrame({
        "target": reference_df["demand_mwh"].values,
        "prediction": reference_df["demand_lag_24h"].values,
    })
    cur_regression = pd.DataFrame({
        "target": current_df["demand_mwh"].values,
        "prediction": current_df["demand_lag_24h"].values,
    })
    regression_snapshot = generate_regression_report(
        ref_regression, cur_regression
    )
    regression_html = save_report(regression_snapshot, "regression_perf")

    # 3. Check if retraining is needed
    drift_share = drift_metrics.get("drift_share", 0)
    needs_retraining = drift_share > DRIFT_THRESHOLD

    result = {
        "drift_metrics": drift_metrics,
        "drift_report_path": str(drift_html),
        "regression_report_path": str(regression_html),
        "needs_retraining": needs_retraining,
    }

    if needs_retraining:
        logger.warning(
            "DRIFT DETECTED: %.0f%% of features drifted (threshold: %.0f%%). "
            "Retraining recommended.",
            drift_share * 100,
            DRIFT_THRESHOLD * 100,
        )
    else:
        logger.info(
            "No significant drift: %.0f%% of features drifted (threshold: %.0f%%)",
            drift_share * 100,
            DRIFT_THRESHOLD * 100,
        )

    return result


if __name__ == "__main__":
    result = run_monitoring()
    print(f"\nDrift share: {result['drift_metrics'].get('drift_share', 0):.1%}")
    print(f"Needs retraining: {result['needs_retraining']}")
    print(f"Drift report: {result['drift_report_path']}")
    print(f"Regression report: {result['regression_report_path']}")
