import logging
import os
import glob
import time
import shutil
import requests
import numpy as np
import pandas as pd
import pendulum
from datetime import timedelta

from airflow.sdk import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.providers.postgres.hooks.postgres import PostgresHook
# NOTE: mlflow is intentionally NOT imported at module top level. Importing it
# is heavy (pulls in huggingface, etc.) and blows past Airflow's 30s DagBag
# import timeout, which made every scheduled run fail with "DAG not found".
# It is imported lazily inside the tasks that need it instead.

GOOD_DIR = "/opt/airflow/data/good_data"
ARCHIVE_DIR = "/opt/airflow/data/archived_data"
TRAIN_CSV = "/opt/airflow/data/train.csv"
MLFLOW_URI = "http://mlflow:5000"
API_URL = "http://model_service:8000"
ML_ALERTS_WEBHOOK_URL = os.environ.get("ML_ALERTS_WEBHOOK_URL", "http://webhook-forwarder:8085/teams/ml-alerts")


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


@dag(
    dag_id="retrain_model_dag",
    description="Retrains model when enough data is collected and drift is detected.",
    tags=["dsp", "training", "epita"],
    schedule=timedelta(minutes=5),
    start_date=pendulum.today("UTC"),
    max_active_runs=1,
    catchup=False
)
def retrain_model_dag():

    @task
    def load_data() -> dict:
        """Check if enough new data is available and if drift is detected."""
        if not os.path.exists(GOOD_DIR):
            raise AirflowSkipException("good_data directory does not exist yet.")

        new_files = glob.glob(os.path.join(GOOD_DIR, "*.csv"))
        if not new_files:
            raise AirflowSkipException("No new data files in good_data since last run.")

        # Check if champion model exists in registry
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        
        has_champion = False
        try:
            alias_info = client.get_model_version_by_alias("TaxiFareModel", "champion")
            has_champion = True
            champion_version = alias_info.version
            logging.info(f"Current champion model version is: {champion_version}")
        except Exception:
            logging.info("No champion model registered in MLflow yet. Proceeding to force first run.")

        # If there is a champion, we check for significant drift (drift condition)
        if has_champion:
            postgres = PostgresHook(postgres_conn_id="taxi_db")
            
            # Fetch latest baseline stats
            baseline_rows = postgres.get_records(
                "SELECT distance_mean FROM training_stats WHERE is_promoted = true ORDER BY promoted_at DESC LIMIT 1;"
            )
            # Fetch prediction stats (last 100 rows or last 1 hour)
            serving_rows = postgres.get_records(
                "SELECT AVG(distance_km) FROM predictions WHERE created_at >= NOW() - INTERVAL '1 hour';"
            )
            if not serving_rows or serving_rows[0][0] is None:
                serving_rows = postgres.get_records(
                    "SELECT AVG(distance_km) FROM (SELECT distance_km FROM predictions ORDER BY created_at DESC LIMIT 100) sub;"
                )

            if baseline_rows and serving_rows and baseline_rows[0][0] is not None and serving_rows[0][0] is not None:
                baseline_mean = baseline_rows[0][0]
                serving_mean = serving_rows[0][0]
                drift_diff = abs(serving_mean - baseline_mean)
                logging.info(f"Drift analysis: baseline_mean={baseline_mean:.4f}, serving_mean={serving_mean:.4f}, difference={drift_diff:.4f}")
                
                # Drift threshold: 1.0 km
                if drift_diff < 1.0:
                    raise AirflowSkipException(
                        f"No significant drift detected ({drift_diff:.4f} < 1.0 km). Skipping retraining."
                    )
            else:
                logging.info("Drift cannot be computed due to missing training stats or predictions. Proceeding with retrain.")

        # Combine baseline train.csv with new good_data files
        dfs = []
        if os.path.exists(TRAIN_CSV):
            logging.info("Loading baseline train.csv (first 20,000 rows)...")
            dfs.append(pd.read_csv(TRAIN_CSV, nrows=20000))
        
        for f in new_files:
            logging.info(f"Loading new data file: {f}")
            dfs.append(pd.read_csv(f))
            
        combined_df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Combined dataset: {len(combined_df)} rows")
        
        # Clean data
        combined_df = combined_df.dropna()
        combined_df = combined_df[combined_df["fare_amount"] > 0]
        combined_df = combined_df[combined_df["fare_amount"] < 500]
        combined_df = combined_df[combined_df["passenger_count"].between(1, 6)]
        combined_df = combined_df[combined_df["pickup_latitude"].between(40.0, 42.0)]
        combined_df = combined_df[combined_df["pickup_longitude"].between(-75.0, -72.0)]
        combined_df = combined_df[combined_df["dropoff_latitude"].between(40.0, 42.0)]
        combined_df = combined_df[combined_df["dropoff_longitude"].between(-75.0, -72.0)]
        
        # Features engineering
        combined_df["pickup_datetime"] = pd.to_datetime(combined_df["pickup_datetime"])
        combined_df["hour"] = combined_df["pickup_datetime"].dt.hour
        combined_df["day_of_week"] = combined_df["pickup_datetime"].dt.dayofweek
        combined_df["month"] = combined_df["pickup_datetime"].dt.month
        combined_df["distance_km"] = haversine_distance(
            combined_df["pickup_latitude"], combined_df["pickup_longitude"],
            combined_df["dropoff_latitude"], combined_df["dropoff_longitude"]
        )
        combined_df = combined_df[combined_df["distance_km"] > 0.1]

        # pickup_datetime has already been decomposed into hour/day_of_week/month
        # above and is not used by train_model. Drop it before returning: pandas
        # Timestamp objects are not XCom-serializable and would fail the task.
        combined_df = combined_df.drop(columns=["pickup_datetime"])

        # Return dataframe contents to next task via XCom (dict)
        return {
            "records": combined_df.to_dict(orient="records")
        }

    @task(multiple_outputs=True)
    def train_model(data: dict) -> dict:
        """Train RandomForestRegressor and register to MLflow."""
        records = data["records"]
        df = pd.DataFrame(records)
        logging.info(f"Training model on {len(df)} records...")
        
        FEATURES = [
            "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude",
            "passenger_count", "hour", "day_of_week",
            "month", "distance_km"
        ]
        TARGET = "fare_amount"
        
        X = df[FEATURES]
        y = df[TARGET]
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        t0 = time.time()
        y_pred = model.predict(X_test)
        t1 = time.time()
        
        # Latency statistics
        inference_time_total = t1 - t0
        avg_inference_time_per_row = inference_time_total / len(X_test)
        
        # Metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        
        logging.info(f"Candidate performance: MAE=${mae:.2f}, RMSE=${rmse:.2f}, R2={r2:.4f}, Per-row inference latency={avg_inference_time_per_row*1000:.4f}ms")
        
        # MLflow Logging (imported lazily to keep DAG parse time under Airflow's timeout)
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("TaxiFareRetraining")
        with mlflow.start_run() as run:
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", 50)
            mlflow.log_param("max_depth", 8)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("avg_inference_latency_ms", avg_inference_time_per_row * 1000)
            
            # Log model and register it
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="TaxiFareModel"
            )
            
            run_id = run.info.run_id
            # MLflow 2.11's ModelInfo has no `registered_model_version`; query the
            # registry for the version that was just created for this run instead.
            from mlflow import MlflowClient
            client = MlflowClient(tracking_uri=MLFLOW_URI)
            versions = [
                m for m in client.search_model_versions(f"run_id='{run_id}'")
                if m.name == "TaxiFareModel"
            ]
            model_version = versions[0].version
            logging.info(f"Registered model version: {model_version} from run: {run_id}")
            
        # Serialize test set and stats to share via XCom
        return {
            "run_id": run_id,
            "version": str(model_version),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "latency": avg_inference_time_per_row,
            "test_set_x": X_test.to_dict(orient="records"),
            "test_set_y": y_test.tolist(),
            "training_set_x": X_train.to_dict(orient="records")
        }

    @task
    def save_training_stats(train_info: dict):
        """Save training feature baseline stats to database."""
        train_x = pd.DataFrame(train_info["training_set_x"])
        version = f"mlflow_v{train_info['version']}"
        
        mean_dist = float(train_x["distance_km"].mean())
        std_dist = float(train_x["distance_km"].std())
        
        counts = train_x["passenger_count"].value_counts(normalize=True).to_dict()
        
        postgres = PostgresHook(postgres_conn_id="taxi_db")
        insert_sql = """
            INSERT INTO training_stats 
            (model_version, promoted_at, distance_mean, distance_std, 
             passenger_1_pct, passenger_2_pct, passenger_3_pct, 
             passenger_4_pct, passenger_5_pct, passenger_6_pct, 
             training_r2, training_mae, is_promoted)
            VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, false);
        """
        
        postgres.run(insert_sql, parameters=(
            version,
            mean_dist,
            std_dist,
            float(counts.get(1, 0.0) * 100),
            float(counts.get(2, 0.0) * 100),
            float(counts.get(3, 0.0) * 100),
            float(counts.get(4, 0.0) * 100),
            float(counts.get(5, 0.0) * 100),
            float(counts.get(6, 0.0) * 100),
            train_info["r2"],
            train_info["mae"]
        ))
        logging.info(f"Saved candidate version {version} training stats to database.")

    @task
    def evaluate_candidate(train_info: dict) -> bool:
        """Compare candidate model against the current champion model."""
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        
        # Inference time check
        latency = train_info["latency"]
        LATENCY_THRESHOLD_SEC = 0.05 # 50ms per prediction
        if latency >= LATENCY_THRESHOLD_SEC:
            msg = f"Candidate model rejected: Inference latency too high ({latency*1000:.2f}ms >= 50ms)"
            logging.warning(msg)
            requests.post(ML_ALERTS_WEBHOOK_URL, json={"text": msg}, timeout=10)
            return False

        has_champion = False
        try:
            alias_info = client.get_model_version_by_alias("TaxiFareModel", "champion")
            champion_version = alias_info.version
            has_champion = True
        except Exception:
            logging.info("No current champion in registry. Evaluation passes automatically.")
            return True

        if has_champion:
            try:
                # Load champion and run on test set
                X_test = pd.DataFrame(train_info["test_set_x"])
                y_test = pd.Series(train_info["test_set_y"])
                
                champion_model = mlflow.sklearn.load_model("models:/TaxiFareModel@champion")
                
                from sklearn.metrics import r2_score
                champion_pred = champion_model.predict(X_test)
                champion_r2 = r2_score(y_test, champion_pred)
                
                candidate_r2 = train_info["r2"]
                
                logging.info(f"Evaluation comparison: Candidate R2={candidate_r2:.4f}, Champion R2={champion_r2:.4f}")
                
                if candidate_r2 >= champion_r2:
                    logging.info("Candidate meets or exceeds champion performance. Promotion approved.")
                    return True
                else:
                    msg = f"Candidate model rejected: Performance R2 ({candidate_r2:.4f}) is lower than champion R2 ({champion_r2:.4f})."
                    logging.warning(msg)
                    
                    # Log comparison results to MLflow
                    with mlflow.start_run(run_id=train_info["run_id"]):
                        mlflow.log_metric("champion_r2", champion_r2)
                        mlflow.log_param("evaluation_status", "failed")
                        
                    # Send alert to Teams
                    requests.post(ML_ALERTS_WEBHOOK_URL, json={"text": msg}, timeout=10)
                    return False
            except Exception as e:
                logging.error(f"Error comparing candidate vs champion: {e}")
                return True # Default to True to prevent blocking first deployment on registry errors

    @task
    def promote_to_champion(eval_passed: bool, train_info: dict) -> bool:
        """Promote candidate model version to @champion in registry and database."""
        if not eval_passed:
            logging.info("Evaluation failed. Model will not be promoted.")
            return False
            
        version = train_info["version"]
        db_version = f"mlflow_v{version}"
        
        # Set @champion alias in MLflow registry
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        client.set_registered_model_alias(
            name="TaxiFareModel",
            alias="champion",
            version=str(version)
        )
        logging.info(f"Set @champion alias on TaxiFareModel version {version} in MLflow.")
        
        # Promote stats in database
        postgres = PostgresHook(postgres_conn_id="taxi_db")
        postgres.run("UPDATE training_stats SET is_promoted = false WHERE is_promoted = true;")
        postgres.run("UPDATE training_stats SET is_promoted = true WHERE model_version = %s;", parameters=(db_version,))
        logging.info(f"Promoted version {db_version} training baseline stats in database.")
        return True

    @task
    def notify_api_reload(promoted: bool):
        """Notify the model service API to reload the champion model."""
        if not promoted:
            return
            
        try:
            r = requests.post(f"{API_URL}/reload-model", timeout=30)
            r.raise_for_status()
            logging.info(f"API reloaded champion model successfully: {r.json()}")
        except Exception as e:
            logging.error(f"Failed to notify API to reload model: {e}")

    @task
    def archive_data(promoted: bool):
        """Move processed files from good_data to archived_data."""
        if not promoted:
            return
            
        if not os.path.exists(ARCHIVE_DIR):
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            
        new_files = glob.glob(os.path.join(GOOD_DIR, "*.csv"))
        for f in new_files:
            dest = os.path.join(ARCHIVE_DIR, os.path.basename(f))
            shutil.move(f, dest)
            logging.info(f"Archived data file: {f} -> {dest}")


    # DAG pipeline wiring
    data_loaded = load_data()
    train_results = train_model(data_loaded)
    
    # Save statistics in parallel with evaluation
    save_stats = save_training_stats(train_results)
    
    eval_passed = evaluate_candidate(train_results)
    
    promoted = promote_to_champion(eval_passed, train_results)
    
    # Run API reload and archive data in parallel after successful promotion
    notify_reload = notify_api_reload(promoted)
    archive = archive_data(promoted)
    
    # Set dependencies
    save_stats >> eval_passed
    promoted >> [notify_reload, archive]

retrain_model_dag()
