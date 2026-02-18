.PHONY: setup fetch-data train serve test monitor clean

export UID := $(shell id -u)
export GID := $(shell id -g)

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "Setup complete. Edit .env with your EIA API key."

fetch-data:
	.venv/bin/python -m src.workflows.ingest

mlflow-server:
	mkdir -p mlflow_data
	docker compose up -d --wait mlflow

train: mlflow-server
	.venv/bin/python -m src.workflows.train

monitor:
	.venv/bin/python -m src.workflows.monitor

serve: mlflow-server
	.venv/bin/uvicorn src.serving.api:app --port 8000

dashboard: mlflow-server
	.venv/bin/streamlit run streamlit_app/app.py

serve-docker:
	docker compose up --build

test:
	.venv/bin/python -m pytest tests/ -v

clean:
	rm -rf data/ mlruns/ mlflow_data/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
