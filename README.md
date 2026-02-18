# Energy Grid Load Forecasting

A production-grade ML pipeline that forecasts hourly electricity demand (MWh) for US grid regions. The system fetches live data from the EIA API, engineers features, trains XGBoost models with hyperparameter tuning, serves predictions via REST API, and monitors for data drift.

## Architecture

```
EIA API  ──>  Data Ingestion  ──>  Feature Engineering  ──>  Model Training
                (Prefect)            (Pandas/DuckDB)        (XGBoost/Optuna/MLflow)
                    │                                              │
                    ▼                                              ▼
              DuckDB Storage                              FastAPI Serving
                    │                                     (POST /predict)
                    ▼                                              │
            Drift Monitoring  <────────────────────────────────────┘
              (Evidently AI)
                    │
                    ▼
          Streamlit Dashboard
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Data Source | EIA Open Data API |
| Storage | DuckDB + Parquet |
| Feature Engineering | Pandas |
| Model Training | XGBoost + Optuna |
| Experiment Tracking | MLflow |
| Serving | FastAPI |
| Monitoring | Evidently AI |
| Dashboard | Streamlit |
| Orchestration | Prefect |
| Containerization | Docker Compose |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (required — MLflow runs as a container for all workflows)
- Free EIA API key from https://www.eia.gov/opendata/register.php

### Setup

```bash
make setup
# Edit .env with your EIA API key
```

### Run the Pipeline

```bash
# 1. Fetch data from EIA API (last 30 days of CISO demand)
make fetch-data

# 2. Train models (auto-starts the MLflow container, then runs training)
make train

# 3. Start the API locally (auto-starts MLflow if not already running)
make serve

# 4. Start the dashboard locally (auto-starts MLflow if not already running)
make dashboard

# 5. Run drift monitoring
make monitor
```

> **Note:** `make train`, `make serve`, and `make dashboard` all automatically start the MLflow Docker container (`docker compose up -d --wait mlflow`) before running. MLflow is always containerized; it is never run locally.

### Docker Deployment

Runs the full stack in containers — MLflow, API, and Streamlit dashboard.

```bash
# Train first to populate the model registry, then start all services
make train
make serve-docker   # or: docker compose up --build
```

Services:
- **MLflow UI** — `http://localhost:5000`
- **FastAPI** — `http://localhost:8000` (docs at `/docs`)
- **Streamlit** — `http://localhost:8501`

Persistent data is stored in `./mlflow_data/` (SQLite DB + model artifacts) and `./data/` (raw + processed datasets). These directories are bind-mounted into the containers and survive restarts.

## Project Structure

```
energy_predML/
├── src/
│   ├── config.py                 # Pydantic settings, path constants
│   ├── data/
│   │   ├── fetch.py              # EIA API client with pagination
│   │   ├── validate.py           # Schema validation
│   │   └── store.py              # DuckDB storage with idempotent upserts
│   ├── features/
│   │   ├── temporal.py           # Hour, day, month, weekend, holiday, cyclical
│   │   ├── lag.py                # Lags (1h/24h/48h/168h), rolling, diffs
│   │   └── pipeline.py           # Feature orchestration + versioned Parquet
│   ├── models/
│   │   ├── baseline.py           # Naive baseline (lag_24h)
│   │   ├── xgboost_model.py      # XGBoost training + Optuna tuning
│   │   └── evaluate.py           # RMSE, MAE, R2, MAPE metrics
│   ├── serving/
│   │   ├── api.py                # FastAPI app (GET /health, POST /predict)
│   │   └── predict.py            # Feature generation + inference logic
│   ├── monitoring/
│   │   └── drift.py              # Evidently data drift + regression reports
│   └── workflows/
│       ├── ingest.py             # Prefect flow: fetch -> validate -> store
│       ├── train.py              # Prefect flow: train baseline + XGBoost
│       └── monitor.py            # Prefect flow: drift detection
├── streamlit_app/
│   └── app.py                    # Dashboard: forecast, accuracy, features, drift
├── scripts/
│   └── compare_models.py         # 1-step vs 24h recursive forecast comparison
├── tests/                        # 25 unit tests (features, data, models)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── .env.example
```

## API

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true, "region": "CISO"}
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2026-02-15T14:00:00", "region": "CISO"}'
```

```json
{
  "timestamp": "2026-02-15T14:00:00",
  "region": "CISO",
  "demand_mwh": 19574.3,
  "confidence_interval": [18791.3, 20357.2]
}
```

Interactive API docs available at `http://localhost:8000/docs`.

## Features

### Temporal Features (11)
- `hour_of_day`, `day_of_week`, `day_of_month`, `month`, `week_of_year`
- `is_weekend`, `is_holiday` (US federal holidays)
- Cyclical encoding: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

### Lag & Rolling Features (10)
- Lags: 1h, 24h, 48h, 168h (1 week)
- Rolling: 24h mean/std, 7-day mean/std (shifted to prevent leakage)
- Differences: 1h and 24h rate of change (shifted to prevent leakage)

## Model Performance

Evaluated on temporal validation split (last 20% of data):

| Model | RMSE (MWh) | MAE (MWh) | R2 | MAPE |
|-------|-----------|----------|------|------|
| Baseline (lag_24h) | 1927 | 1382 | 0.31 | 5.83% |
| XGBoost (default) | 564 | 360 | 0.94 | 1.51% |
| XGBoost (tuned) | 501 | 324 | 0.95 | 1.35% |

## Testing

```bash
make test
```

Runs 25 tests covering:
- EIA API client (mocked HTTP responses, pagination)
- Data validation (schema, null handling, sorting)
- Temporal features (ranges, weekend detection, cyclical bounds)
- Lag features (correctness, NaN warm-up, no data leakage)
- Evaluation metrics (perfect predictions, known errors, MAPE)

## Key Design Decisions

- **Temporal splits only** — never shuffle time-series data. Train on past, validate on future.
- **No data leakage** — all lag/diff features use `shift()` to ensure only past data is used.
- **Idempotent ingestion** — DuckDB upserts skip existing rows, safe to re-run.
- **MLflow always containerized** — all workflows (training, serving, dashboard) connect to `http://localhost:5000`. The `mlflow-server` Makefile target starts the container automatically. Fail-fast behaviour: services exit with a clear error rather than silently falling back to a local filesystem.
- **Versioned features** — Parquet files are timestamped for reproducibility.
