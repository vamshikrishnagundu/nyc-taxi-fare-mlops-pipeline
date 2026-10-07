import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import os

# ── 1. Load the dataset ──
print("Loading dataset...")
df = pd.read_csv("../data/train.csv", nrows=100_000)
print(f"Dataset loaded: {df.shape[0]} rows")

# ── 2. Clean the data ──
print("Cleaning data...")
df = df.dropna()
df = df[df["fare_amount"] > 0]
df = df[df["fare_amount"] < 500]
df = df[df["passenger_count"] > 0]
df = df[df["passenger_count"] <= 6]
df = df[df["pickup_longitude"].between(-75, -72)]
df = df[df["pickup_latitude"].between(40, 42)]
df = df[df["dropoff_longitude"].between(-75, -72)]
df = df[df["dropoff_latitude"].between(40, 42)]
print(f"After cleaning: {df.shape[0]} rows")

# ── 3. Create extra features ──
print("Creating features...")
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
df["month"] = df["pickup_datetime"].dt.month

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine_distance(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)
df = df[df["distance_km"] > 0.1]

# ── 4. Define features and target ──
FEATURES = [
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "hour", "day_of_week",
    "month", "distance_km"
]
TARGET = "fare_amount"

X = df[FEATURES]
y = df[TARGET]

# ── 5. Split data ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} rows")

# ── 6. Train the model ──
print("Training model... please wait 1-2 minutes...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete!")

# ── 7. Test the model ──
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Error: ${mae:.2f} average")

# ── 8. Save the model ──
os.makedirs("saved_model", exist_ok=True)
with open("saved_model/taxi_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("saved_model/features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print("Model saved to saved_model/taxi_model.pkl")
print("Done!")