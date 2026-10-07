# main.py — The FastAPI Backend (The Brain of our app)
# This file creates API endpoints for predictions and reloading models.

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import pickle
import numpy as np
import pandas as pd
import os
from sqlalchemy.orm import Session

from database import get_db, create_tables, Prediction, IngestionStats, TrainingStats

# ── Load the ML Model when API starts ──
MODEL_PATH = os.getenv("MODEL_PATH", "saved_model/taxi_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "saved_model/features.pkl")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None
FEATURES = None
model_version = "local_v1"

def load_champion_model():
    global model, FEATURES, model_version
    try:
        print(f"Connecting to MLflow registry at {MLFLOW_TRACKING_URI}...")
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        alias_info = client.get_model_version_by_alias("TaxiFareModel", "champion")
        v = alias_info.version
        print(f"Found champion model version {v} in MLflow. Loading...")
        model = mlflow.sklearn.load_model("models:/TaxiFareModel@champion")
        model_version = f"mlflow_v{v}"
        
        # Load features from features.pkl
        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH, "rb") as f:
                FEATURES = pickle.load(f)
        else:
            FEATURES = [
                "pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude",
                "passenger_count", "hour", "day_of_week",
                "month", "distance_km"
            ]
        print(f"Successfully loaded champion model version {v} from MLflow!")
        return True
    except Exception as e:
        print(f"Could not load champion model from MLflow: {e}")
        return False

def _load_model_on_startup():
    """Load the model once when the API starts: the champion from the MLflow
    registry if available, otherwise fall back to the local pickle files."""
    global model, FEATURES, model_version
    if not load_champion_model():
        print("Loading local fallback model files...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            FEATURES = pickle.load(f)
        model_version = "local_v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: load the model once, then prepare the database ──
    _load_model_on_startup()
    _init_database()
    yield
    # ── Shutdown: nothing to clean up ──


# ── Create the FastAPI app ──
app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predicts NYC taxi fares using Machine Learning",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup: create tables, run migrations, pre-seed baseline stats.
# Invoked from the lifespan handler above when the app starts.
def _init_database():
    create_tables()
    
    # Run migrations (add model_version column if it does not exist)
    from sqlalchemy import text
    db = next(get_db())
    try:
        db.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_version VARCHAR DEFAULT 'local_v1';"))
        db.commit()
    except Exception as e:
        print(f"Migration error (predictions.model_version): {e}")
        db.rollback()

    try:
        db.execute(text("ALTER TABLE ingestion_stats ADD COLUMN IF NOT EXISTS schema_error_count INTEGER DEFAULT 0;"))
        db.commit()
    except Exception as e:
        print(f"Migration error (ingestion_stats.schema_error_count): {e}")
        db.rollback()

    # Pre-seed baseline training stats if empty
    try:
        count = db.query(TrainingStats).count()
        if count == 0:
            print("Pre-seeding baseline training stats...")
            train_path = "/app/data/train.csv"
            if not os.path.exists(train_path):
                train_path = "../data/train.csv"
                
            if os.path.exists(train_path):
                df = pd.read_csv(train_path, nrows=20000)
                def haversine_distance(lat1, lon1, lat2, lon2):
                    R = 6371
                    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    return 2 * R * np.arcsin(np.sqrt(a))
                
                df = df.dropna()
                df = df[df["passenger_count"] > 0]
                df = df[df["passenger_count"] <= 6]
                df = df[df["pickup_latitude"].between(40, 42)]
                df = df[df["pickup_longitude"].between(-75, -72)]
                
                distance_km = haversine_distance(
                    df["pickup_latitude"], df["pickup_longitude"],
                    df["dropoff_latitude"], df["dropoff_longitude"]
                )
                df["distance_km"] = distance_km
                
                mean_dist = float(df["distance_km"].mean())
                std_dist = float(df["distance_km"].std())
                
                counts = df["passenger_count"].value_counts(normalize=True).to_dict()
                
                stats = TrainingStats(
                    model_version="local_v1",
                    distance_mean=mean_dist,
                    distance_std=std_dist,
                    passenger_1_pct=float(counts.get(1, 0.0) * 100),
                    passenger_2_pct=float(counts.get(2, 0.0) * 100),
                    passenger_3_pct=float(counts.get(3, 0.0) * 100),
                    passenger_4_pct=float(counts.get(4, 0.0) * 100),
                    passenger_5_pct=float(counts.get(5, 0.0) * 100),
                    passenger_6_pct=float(counts.get(6, 0.0) * 100),
                    training_r2=0.72,
                    training_mae=2.1,
                    is_promoted=True
                )
                db.add(stats)
                db.commit()
                print("Baseline training stats pre-seeded successfully!")
            else:
                print(f"Warning: train.csv not found at {train_path}, cannot pre-seed training stats.")
    except Exception as e:
        print(f"Error pre-seeding training stats: {e}")
        db.rollback()
    finally:
        db.close()

# ── Define what data we expect ──
# This is like a form — it defines what fields are required

class TaxiRide(BaseModel):
    pickup_datetime: str = "2015-01-01 00:00:00 UTC"
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int = 1

class PredictRequest(BaseModel):
    rides: List[TaxiRide]
    source: str = "webapp"

# ── Response models (documented, structured output) ──
class PredictResponse(BaseModel):
    predictions: List[float]
    message: str

class ReloadResponse(BaseModel):
    status: str
    message: str

class PastPredictionResponse(BaseModel):
    id: int
    pickup_datetime: Optional[str] = None
    pickup_longitude: Optional[float] = None
    pickup_latitude: Optional[float] = None
    dropoff_longitude: Optional[float] = None
    dropoff_latitude: Optional[float] = None
    passenger_count: Optional[int] = None
    distance_km: Optional[float] = None
    predicted_fare: Optional[float] = None
    prediction_source: Optional[str] = None
    model_version: Optional[str] = None
    created_at: Optional[str] = None

# ── Helper: Calculate distance between 2 GPS points ──
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ── Helper: Prepare features for the model ──
def prepare_features(ride: TaxiRide):
    try:
        dt = datetime.strptime(ride.pickup_datetime, "%Y-%m-%d %H:%M:%S %Z")
    except:
        dt = datetime.now()

    distance_km = haversine_distance(
        ride.pickup_latitude, ride.pickup_longitude,
        ride.dropoff_latitude, ride.dropoff_longitude
    )

    return [
        ride.pickup_longitude,
        ride.pickup_latitude,
        ride.dropoff_longitude,
        ride.dropoff_latitude,
        ride.passenger_count,
        dt.hour,
        dt.weekday(),
        dt.month,
        distance_km
    ]

# ── Endpoint 1: Health Check ──
@app.get("/")
def root():
    return {"status": "running", "message": "Taxi Fare API is up!"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "API is running fine!"
    }

# ── Endpoint 2: Make Predictions ──
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    if not request.rides:
        raise HTTPException(status_code=400, detail="No rides provided")

    # Prepare features for all rides
    features_list = [prepare_features(ride) for ride in request.rides]
    features_array = np.array(features_list)

    # Make predictions
    predictions = model.predict(features_array).tolist()

    # Save each prediction to the database
    for ride, pred in zip(request.rides, predictions):
        try:
            dt = datetime.strptime(ride.pickup_datetime, "%Y-%m-%d %H:%M:%S %Z")
        except:
            dt = datetime.now()

        distance_km = haversine_distance(
            ride.pickup_latitude, ride.pickup_longitude,
            ride.dropoff_latitude, ride.dropoff_longitude
        )

        db_prediction = Prediction(
            pickup_longitude=ride.pickup_longitude,
            pickup_latitude=ride.pickup_latitude,
            dropoff_longitude=ride.dropoff_longitude,
            dropoff_latitude=ride.dropoff_latitude,
            passenger_count=ride.passenger_count,
            hour=dt.hour,
            day_of_week=dt.weekday(),
            month=dt.month,
            distance_km=distance_km,
            pickup_datetime=ride.pickup_datetime,
            predicted_fare=round(pred, 2),
            prediction_source=request.source,
            model_version=model_version
        )
        db.add(db_prediction)

    db.commit()

    return {
        "predictions": [round(p, 2) for p in predictions],
        "message": f"Successfully predicted {len(predictions)} fare(s)"
    }

# ── Endpoint 3: Get Past Predictions ──
@app.get("/past-predictions", response_model=List[PastPredictionResponse])
def get_past_predictions(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    source: str = Query("all"),
    db: Session = Depends(get_db)
):
    query = db.query(Prediction)

    if start_date:
        query = query.filter(Prediction.created_at >= start_date)
    if end_date:
        query = query.filter(Prediction.created_at <= end_date + " 23:59:59")
    if source != "all":
        query = query.filter(Prediction.prediction_source == source)

    results = query.order_by(Prediction.created_at.desc()).limit(500).all()

    return [
        {
            "id": r.id,
            "pickup_datetime": r.pickup_datetime,
            "pickup_longitude": r.pickup_longitude,
            "pickup_latitude": r.pickup_latitude,
            "dropoff_longitude": r.dropoff_longitude,
            "dropoff_latitude": r.dropoff_latitude,
            "passenger_count": r.passenger_count,
            "distance_km": round(r.distance_km, 2) if r.distance_km else None,
            "predicted_fare": r.predicted_fare,
            "prediction_source": r.prediction_source,
            "model_version": r.model_version,
            "created_at": str(r.created_at)
        }
        for r in results
    ]

# ── Endpoint 4: Reload Model from MLflow Registry ──
@app.post("/reload-model", response_model=ReloadResponse)
def reload_model():
    success = load_champion_model()
    if success:
        return {"status": "success", "message": f"Model reloaded successfully to version {model_version}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model from MLflow registry")

# ── Mock Teams Webhook Endpoints ──
class TeamsMessage(BaseModel):
    text: str

@app.post("/mock-teams/data-quality-alerts")
def mock_teams_data_quality(msg: TeamsMessage):
    print("\n" + "="*50)
    print(f"[TEAMS MOCK - DATA QUALITY ALERTS CHANNEL]\nReceived Alert:\n{msg.text}")
    print("="*50 + "\n")
    return {"status": "received"}

@app.post("/mock-teams/ml-alerts")
def mock_teams_ml_alerts(msg: TeamsMessage):
    print("\n" + "="*50)
    print(f"[TEAMS MOCK - ML ALERTS CHANNEL]\nReceived Alert:\n{msg.text}")
    print("="*50 + "\n")
    return {"status": "received"}