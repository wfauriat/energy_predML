"""FastAPI app for serving energy demand predictions."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

import mlflow
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.config import settings
from src.data.store import log_prediction
from src.serving.predict import predict_demand

logger = logging.getLogger(__name__)

# Global model reference
_model = None


def load_model():
    """Load the latest registered model from MLflow."""
    global _model

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # 1. Try latest version from the model registry
    try:
        client = MlflowClient()
        versions = client.search_model_versions("name='energy_demand_xgboost'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            _model = mlflow.xgboost.load_model(
                f"models:/energy_demand_xgboost/{latest.version}"
            )
            logger.info(
                "Loaded model version %s from MLflow registry", latest.version
            )
            return
    except Exception as e:
        logger.warning("Registry load failed: %s — trying latest run", e)

    # 2. Fallback: most recent run tagged xgboost_tuned
    try:
        experiment = mlflow.get_experiment_by_name("energy_grid_forecast")
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.model_type = 'xgboost_tuned'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                _model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
                logger.info("Loaded model from run %s", run_id)
                return
    except Exception as e2:
        logger.warning("Run fallback failed: %s", e2)

    raise RuntimeError("No trained model found. Run: make train")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield


app = FastAPI(
    title="Energy Grid Load Forecast API",
    description="Predicts hourly electricity demand (MWh) for US grid regions.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request/Response schemas ---

class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    timestamp: datetime = Field(
        ...,
        description="Target timestamp for prediction (ISO 8601 format)",
        examples=["2026-02-15T14:00:00"],
    )
    region: str = Field(
        default="CISO",
        description="Grid region code (e.g., CISO, ERCO)",
    )


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    timestamp: str
    region: str
    demand_mwh: float
    confidence_interval: list[float]


class HealthResponse(BaseModel):
    """Response schema for health endpoint."""
    status: str
    model_loaded: bool
    region: str


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        region=settings.DEFAULT_REGION,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Generate a demand prediction for a given timestamp.

    The model requires historical demand data in DuckDB to compute lag features.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predict_demand(
            model=_model,
            target_ts=request.timestamp,
            region=request.region,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    try:
        log_prediction(
            timestamp=request.timestamp,
            region=request.region,
            demand_mwh=result["demand_mwh"],
        )
    except Exception as log_err:
        logger.warning("Failed to log prediction: %s", log_err)

    return PredictResponse(**result)
