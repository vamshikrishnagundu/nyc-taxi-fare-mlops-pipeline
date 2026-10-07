# NYC Taxi Fare Prediction — End-to-End MLOps Pipeline

A production-style Machine Learning system that predicts NYC taxi fares with a Random Forest model and runs a **fully automated MLOps loop**: continuous data ingestion with quality validation, batch prediction, **drift-triggered retraining with an MLflow model registry (champion/challenger)**, live monitoring in Grafana, and Microsoft Teams alerting — all orchestrated with **Apache Airflow 3** and Docker Compose.

## Group Members

- Amruth Ganta
- Amaan Syed
- Nithya Niharika Kotcherla
- Bhanu Prakash Kuncham
- Vedant Rajput

---

## Architecture Overview

```
                 ┌───────────────────────── Apache Airflow 3 ─────────────────────────┐
 data/raw/  ───► │ ingest_taxi_data ──► predict_taxi_fares ──► retrain_model_dag        │
 (new files)     │  (Great Expectations   (batch inference      (drift check → train →  │
                 │   validation)           via the API)          evaluate → promote)    │
                 └───────┬───────────────────────┬────────────────────────┬─────────────┘
                         │ good_data / bad_data   │ predictions            │ champion model
                         ▼                        ▼                        ▼
                   PostgreSQL  ◄───────────  model_service (FastAPI)  ◄── MLflow Registry
                        │                          ▲    │  auto-reload      (TaxiFareModel
                        │                          │    │  /reload-model      @champion)
                        ▼                          │    ▼
                    Grafana  ──► alert rules ──► webhook-forwarder ──► MS Teams
                  (dashboards)                   (Data Quality + ML alert channels)
```

The **model service always serves the current MLflow `@champion`**. When the retraining DAG promotes a better model, it calls the API's `/reload-model` endpoint so the new champion is hot-swapped in with no restart.

---

## Project Structure

```text
nyc-taxi-fare-mlops-pipeline/
├── data/                        # Datasets & pipeline working dirs (gitignored data)
│   ├── train.csv                # Base training dataset
│   ├── raw/                     # Incoming CSV chunks the ingestion DAG consumes
│   ├── good_data/               # Rows that passed Great Expectations validation
│   ├── bad_data/                # Rows that failed validation
│   ├── archived_data/           # good_data already consumed by retraining
│   └── gx/                      # Great Expectations context + HTML data-docs
│
├── dags/                        # Apache Airflow 3 DAGs
│   ├── ingestion_dag_dsp.py     # ingest_taxi_data  — GX validation + Teams alerts
│   ├── predict_taxi_fares.py    # predict_taxi_fares — batch inference via the API
│   └── training_dag_dsp.py      # retrain_model_dag  — drift → train → promote champion
│
├── model/                       # One-off model training
│   ├── train.py                 # Trains the initial model from data/train.csv
│   └── saved_model/             # taxi_model.pkl + features.pkl (local fallback)
│
├── model_service/               # FastAPI backend (serves the MLflow champion)
│   ├── main.py  ├── database.py  ├── requirements.txt  └── Dockerfile
│
├── webapp/                      # Streamlit frontend
│   ├── app.py  ├── requirements.txt  └── Dockerfile
│
├── tools/webhook_forwarder/     # Flask relay: alerts → Teams Adaptive Cards
│   ├── app.py  ├── requirements.txt  └── Dockerfile
│
├── grafana/provisioning/        # Auto-provisioned dashboards, datasource & alerting
│   ├── datasources/postgres.yaml
│   ├── dashboards/{ingested_data,drift_prediction}.json
│   └── alerting/{rules,alerting}.yaml
│
├── mlflow-data/                 # MLflow backend store (sqlite) — gitignored
├── mlflow-artifacts/            # MLflow artifact store (shared volume) — gitignored
│
├── db_init/init.sql             # Bootstraps the taxifare + airflow databases
├── scripts/                     # Data utilities
│   ├── split_dataset.py         # Splits train.csv into the raw_data_*.csv chunks
│   ├── data_error_injection.py  # Seeds the 7 error categories into the raw chunks
│   └── inject_demo_errors.py    # Pushes every Nth chunk to ~20% invalid (demo alerts)
│
├── tests/                       # Pytest unit tests (run in CI)
│   ├── conftest.py              # Puts model_service/ + webhook_forwarder/ on sys.path
│   ├── test_model_features.py   # haversine + feature engineering
│   └── test_webhook_forwarder.py# alert payload / Adaptive Card helpers
│
├── .github/workflows/ci.yml     # CI: flake8 + pytest, auto GitHub Release on main
├── .env.example                 # Teams webhook variables (copy to .env)
└── docker-compose.yml
```

---

## The Automated Pipeline (3 DAGs)

1. **`ingest_taxi_data`** — picks up a new file from `data/raw/`, validates it with **Great Expectations v1** (missing values, negative fares, bad passenger counts, out-of-NYC / ocean coordinates, future dates). Valid rows go to `data/good_data/`, invalid rows to `data/bad_data/`, stats are written to PostgreSQL, an HTML report is published (served by nginx), and a **🔴 Data Quality Teams alert** fires when a file is ≥10% invalid (Medium/High criticality).
2. **`predict_taxi_fares`** — detects new files in `data/good_data/`, batch-predicts fares via the FastAPI `/predict` endpoint, and stores results in PostgreSQL.
3. **`retrain_model_dag`** — checks for accumulated data and **data drift** vs. the training baseline (mean trip distance shifted by **≥ 1.0 km**). If drift is detected it trains a challenger `RandomForestRegressor` (`n_estimators=50`, `max_depth=8`), logs params, MAE/RMSE/R² and inference latency to the **MLflow** `TaxiFareRetraining` experiment, then applies two promotion gates: **per-row inference latency < 50 ms** and **candidate R² ≥ champion R²**. A winner is **promoted to `@champion`** in the `TaxiFareModel` registry, written to the DB, hot-swapped into the API via `/reload-model`, and its input data is archived. Rejected candidates are tagged `evaluation_status=failed` in MLflow and raise a **🟠 ML Teams alert**.

---

## Services

| Service | Image / Build | Container | Port | Description |
|---|---|---|---|---|
| PostgreSQL | `postgres:15` | `taxi_db` | `5432` | Predictions, ingestion/training stats, Airflow metadata |
| FastAPI backend | build `./model_service` (`python:3.12-slim`) | `taxi_api` | `8000` | Inference API; serves the MLflow champion |
| Streamlit frontend | build `./webapp` (`python:3.12-slim`) | `taxi_frontend` | `8501` | Fare prediction UI + past predictions |
| MLflow | `ghcr.io/mlflow/mlflow:v2.11.3` | `mlflow_server` | `5001`→5000 | Model registry & experiment tracking (SQLite backend store) |
| Grafana | `grafana/grafana:10.2.2` | `taxi_grafana` | `3000` | Monitoring dashboards + alert rules |
| Webhook forwarder | build `./tools/webhook_forwarder` (`python:3.11-slim`) | `webhook_forwarder` | `8085` | Relays alerts to MS Teams as Adaptive Cards |
| Nginx | `nginx:alpine` | `taxi_nginx` | `8082`→80 | Serves Great Expectations data-quality reports |
| Airflow API server | `apache/airflow:3.0.0-python3.12` | `airflow_api_server` | `8080` | Airflow 3 web UI |
| Airflow scheduler | `apache/airflow:3.0.0-python3.12` | `airflow_scheduler` | – | Runs DAGs (LocalExecutor) |
| Airflow DAG processor | `apache/airflow:3.0.0-python3.12` | `airflow_dag_processor` | – | Parses DAG files |
| Airflow init | `apache/airflow:3.0.0-python3.12` | `airflow_init` | – | One-shot DB init/migration, then exits |

Named volumes: `postgres_data`, `grafana_data`. Airflow gets `pandas`, `numpy`,
`mlflow==2.11.3`, `great-expectations>=1.1.3` and `psycopg2-binary` via
`_PIP_ADDITIONAL_REQUIREMENTS`, and reaches the app database through the
pre-wired `AIRFLOW_CONN_TAXI_DB` connection.

---

## How to Run

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and **running**. This is a large stack — allocate **≥ 5 GB** to Docker (Settings → Resources → Memory).
- Python 3.12+ (only needed to train the initial model / run tests locally)

### Step 1 — Configure Teams webhooks
```bash
cp .env.example .env
# edit .env and set REAL_TEAMS_WEBHOOK (Data Quality channel) and,
# optionally, REAL_TEAMS_WEBHOOK_ML (ML channel). If ML is left blank it
# falls back to the Data Quality webhook.
```

### Step 2 — Train the initial model (once)
```bash
cd model && python train.py && cd ..
```
Outputs `model/saved_model/taxi_model.pkl` + `features.pkl` — the local fallback the API uses until MLflow has a champion.

### Step 3 — Launch the stack
```bash
docker compose up -d
docker compose ps        # wait until services are healthy
```

### Step 4 — Open the apps

| Application | URL | Notes |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Predict fares, browse past predictions |
| Airflow UI | http://localhost:8080 | Trigger / monitor the 3 DAGs (auto-login as admin) |
| FastAPI docs | http://localhost:8000/docs | Swagger for the API |
| MLflow | http://localhost:5001 | Registry: `TaxiFareModel` + `@champion` |
| Grafana | http://localhost:3000 | Dashboards & alerts (login `admin` / `admin`) |
| GX reports | http://localhost:8082 | Data-quality validation reports |

To stop everything (keeps your data & volumes):
```bash
docker compose down
```

---

## API Endpoints (`model_service`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` , `/health` | Health checks |
| POST | `/predict` | Predict fares for a batch of rides (stores results in DB) |
| GET | `/past-predictions` | Query stored predictions (date/source filters) |
| POST | `/reload-model` | Reload the champion model from the MLflow registry |
| POST | `/mock-teams/*` | Local mock endpoints for testing alerts without Teams |

---

## Monitoring & Alerting

- **Grafana** auto-provisions a Postgres datasource, two dashboards (ingested-data quality and drift/prediction monitoring), and alert rules that route to two contact points.
- Both contact points POST to the **webhook forwarder**, which wraps the message in a Teams **Adaptive Card** with a colour-coded header — **🔴 Data Quality Alert** or **🟠 ML Alert** — so the alert type is clear at a glance, then forwards it to the configured MS Teams channel(s).

---

## Testing & CI/CD

- **Tests** (`tests/`): `pytest` unit tests for the model feature helpers (haversine distance, feature engineering) and the webhook forwarder's payload/Adaptive-Card helpers.
  ```bash
  pip install -r requirements.txt && pytest tests/ -v
  ```
- **GitHub Actions** (`.github/workflows/ci.yml`): on push / PR to `main`, runs **flake8** + **pytest**; on push to `main` it also creates an automated **GitHub Release**.

---

## Tools & Services Used

Every tool, library, image and external service this project depends on.

### Languages & runtimes
| Tool | Version | Where / why |
|---|---|---|
| Python | 3.12 (forwarder image: 3.11) | All application, DAG and script code; CI runs 3.12 |
| SQL | – | `db_init/init.sql`, Grafana dashboard panels, DAG stat queries |
| Bash | – | CI steps, local run commands |

### Machine learning
| Tool | Version | Where / why |
|---|---|---|
| scikit-learn | 1.3.2 | `RandomForestRegressor`, `train_test_split`, MAE / RMSE / R² metrics |
| pandas | 2.1.3 | All dataframe I/O across the API, DAGs, scripts and webapp |
| NumPy | 1.26.2 | Feature math (haversine), error injection, array ops |
| joblib | 1.3.2 | Model serialisation support |
| pickle | stdlib | Persists `taxi_model.pkl` / `features.pkl` (the local fallback model) |

### MLOps — tracking, registry, orchestration
| Tool | Version | Where / why |
|---|---|---|
| MLflow | 2.11.3 (`ghcr.io/mlflow/mlflow`) | Experiment `TaxiFareRetraining`, registry `TaxiFareModel`, `@champion` alias, `mlflow.sklearn` log/load |
| SQLite | bundled | MLflow backend store (`mlflow-data/mlflow.db`) |
| Apache Airflow | 3.0.0 (`apache/airflow:3.0.0-python3.12`) | 3 DAGs on the **LocalExecutor**; `airflow.sdk` `@dag`/`@task` TaskFlow API |
| Airflow Postgres provider | via `_PIP_ADDITIONAL_REQUIREMENTS` | `PostgresHook` on the `AIRFLOW_CONN_TAXI_DB` connection |
| Pendulum | via Airflow | DAG schedules and timezone-aware `start_date`s |

### Data quality
| Tool | Version | Where / why |
|---|---|---|
| Great Expectations | ≥ 1.1.3 (v1 API) | Ephemeral context, `ExpectationSuite` `taxi_suite`, `ValidationDefinition`, `Checkpoint`, `UpdateDataDocsAction` |
| GX Data Docs | – | HTML validation reports written to `data/gx/uncommitted/data_docs/local_site`, served by nginx |

### Backend API
| Tool | Version | Where / why |
|---|---|---|
| FastAPI | 0.104.1 | `model_service` — `/predict`, `/past-predictions`, `/reload-model`, `/health`, `/mock-teams/*` |
| Uvicorn | 0.24.0 | ASGI server for the API container |
| Pydantic | 2.5.0 | Request/response schemas (`TaxiRide`, `PredictResponse`, …) |
| SQLAlchemy | 2.0.23 | ORM models: `Prediction`, `IngestionStats`, `TrainingStats` |
| psycopg2-binary | 2.9.9 | PostgreSQL driver (API, DAGs, Airflow metadata DB) |
| Starlette CORS middleware | via FastAPI | Cross-origin access for the Streamlit UI |
| python-multipart | 0.0.6 | Form/file payload parsing |

### Frontend
| Tool | Version | Where / why |
|---|---|---|
| Streamlit | 1.28.2 | `webapp/app.py` — single-ride + CSV batch prediction, past-predictions browser |
| requests | 2.31.0 | HTTP calls: webapp → API, DAGs → API, forwarder → Teams |

### Database
| Tool | Version | Where / why |
|---|---|---|
| PostgreSQL | 15 | `taxifare` DB (predictions, ingestion & training stats) + `airflow` metadata DB |

### Monitoring & alerting
| Tool | Version | Where / why |
|---|---|---|
| Grafana | 10.2.2 | Provisioned Postgres datasource, `ingested_data` + `drift_prediction` dashboards, alert rules |
| Grafana unified alerting | – | Contact points *Data Quality Alerts Channel* / *ML Alerts Channel*, routed by the `alert_channel` label |
| Flask | 2.2.5 | `tools/webhook_forwarder` — the alert relay service |
| Microsoft Teams | external | Destination for all alerts |
| Power Automate | external | "When a Teams webhook request is received" flows behind `REAL_TEAMS_WEBHOOK` / `REAL_TEAMS_WEBHOOK_ML` |
| Adaptive Cards | schema 1.4 | Colour-coded 🔴 Data Quality / 🟠 ML alert cards posted into Teams |

### Infrastructure
| Tool | Version | Where / why |
|---|---|---|
| Docker Engine + Docker Compose | – | Defines and runs all 11 containers |
| Docker Desktop | – | Local host runtime (≥ 5 GB memory recommended) |
| Nginx | `nginx:alpine` | Static server for the Great Expectations data-docs site |
| Docker named volumes | – | `postgres_data`, `grafana_data`; bind mounts share `mlflow-artifacts/` between MLflow, Airflow and the API |

### Testing, linting & CI/CD
| Tool | Version | Where / why |
|---|---|---|
| pytest | 8.0.0 | `tests/` unit tests; `conftest.py` puts the app dirs on `sys.path` |
| flake8 | 7.0.0 | Lint gate in CI (hard fail on `E9,F63,F7,F82`, then a non-blocking full pass) |
| GitHub Actions | `actions/checkout@v3`, `actions/setup-python@v4` | `.github/workflows/ci.yml` — lint + test on push/PR to `main` |
| GitHub Releases + `gh` CLI | – | Auto-tagged release (`vYYYY.MM.DD-HHMMSS`) on every push to `main` |
| setuptools | `<81` | Pinned so `mlflow` can still `import pkg_resources` on Python 3.12 |
| Git / GitHub | – | Trunk `main`, integration branch `develop`, `feature/*` branches |

### Data utilities
| Tool | Where / why |
|---|---|
| `scripts/split_dataset.py` | Splits `data/train.csv` into the `raw_data_*.csv` ingestion chunks |
| `scripts/data_error_injection.py` | Seeds the 7 error categories (nulls, negative fares, bad passenger counts, wrong city, wrong dtype, future dates, ocean dropoffs) |
| `scripts/inject_demo_errors.py` | Pushes every Nth chunk to ~20% invalid so the demo fires Medium-criticality alerts |
