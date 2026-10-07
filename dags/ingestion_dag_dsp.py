import logging
from datetime import datetime, timedelta
import os
import glob
import random
import shutil
import json
import requests

import pandas as pd
import pendulum
from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sqlalchemy import create_engine

# -- Directories --
RAW_DIR = "/opt/airflow/data/raw"
GOOD_DIR = "/opt/airflow/data/good_data"
BAD_DIR = "/opt/airflow/data/bad_data"

TEAMS_WEBHOOK_URL = os.environ.get("TEAMS_WEBHOOK_URL", "https://default3534b3d7316c4bc99ede605c860f49.d2.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/b244a880825b4b09a77e299acff4dfd7/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=5tB_YeJKKA0Aeo-xkJFYJmc0RrgFu6nd_DEhAuvzVHU")

@dag(
    dag_id='ingest_taxi_data',
    description='Ingest and validate NYC Taxi data with Great Expectations',
    tags=['dsp', 'data_ingestion', 'epita'],
    schedule=timedelta(minutes=1),
    start_date=pendulum.today("UTC"), 
    max_active_runs=1, 
    catchup=False
)
def ingest_taxi_data():
    
    @task
    def read_data() -> dict:
        if not os.path.exists(RAW_DIR):
            os.makedirs(RAW_DIR, exist_ok=True)
            
        all_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
        if not all_files:
            logging.info("No files available in raw_data")
            return {"filepath": None}
            
        chosen_file = sorted(all_files)[0]
        logging.info(f"Picked file: {chosen_file}")
        
        # Read the file to pass to the next task
        input_data_df = pd.read_csv(chosen_file)
        
        # Remove the file to avoid reprocessing (Data split task will place it to proper dir later)
        # Actually, let's keep the file and either delete it or move it at the end 
        os.remove(chosen_file)
        
        return {
            "filepath": chosen_file, 
            "content": input_data_df.to_dict(orient="records")
        }

    @task(multiple_outputs=True)
    def validate_data(data_info: dict) -> dict:
        if not data_info.get("filepath"):
            return {"skip": True}
            
        filepath = data_info["filepath"]
        content = data_info["content"]
        df = pd.DataFrame(content)
        total_rows = len(df)
        
        # Avoid GX error if no rows
        if total_rows == 0:
            return {
                "skip": False,
                "filename": os.path.basename(filepath),
                "total": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "missing_values": 0,
                "negative_fares": 0,
                "wrong_passengers": 0,
                "wrong_coordinates": 0,
                "string_in_number": 0,
                "future_dates": 0,
                "zero_coordinates": 0,
                "schema_errors": 0,
                "criticality": "None",
                "docs_url": "No Docs",
                "valid_rows": [],
                "invalid_rows": []
            }

        # A single non-numeric value (e.g. "unknown") in fare_amount makes pandas read the
        # WHOLE column as strings for this file. That crashes GX's between() check (comparing
        # str to float) and makes the type check flag every row instead of just the bad ones.
        # Coerce up front and capture the true "string in number" rows before we do.
        raw_fare = df["fare_amount"] if "fare_amount" in df.columns else pd.Series(dtype=object, index=df.index)
        fare_numeric = pd.to_numeric(raw_fare, errors="coerce")
        string_in_number_mask = fare_numeric.isna() & raw_fare.notna()
        df["fare_amount"] = fare_numeric
        if "passenger_count" in df.columns:
            df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")

        import great_expectations as gx
        context = gx.get_context(mode="ephemeral")
        
        # Using GX Core 1.x API
        data_source = context.data_sources.add_pandas("pandas_source")
        data_asset = data_source.add_dataframe_asset(name="taxi_data")
        batch_definition = data_asset.add_batch_definition_whole_dataframe("taxi_batch")
        
        suite = gx.ExpectationSuite(name="taxi_suite")
        
        # Expectations
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column="passenger_count"))
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column="fare_amount"))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="passenger_count"))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="fare_amount", min_value=0.0))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="passenger_count", min_value=1, max_value=6))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="pickup_latitude", min_value=40.0, max_value=42.0))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="pickup_longitude", min_value=-75.0, max_value=-72.0))
        # Ensure 'fare_amount' can be converted to numeric
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="fare_amount", type_="float"))

        # Time expectations
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotMatchRegex(column="pickup_datetime", regex="^2099"))
        # Ocean coordinate (0,0)
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeInSet(column="dropoff_latitude", value_set=[0.0]))
        
        suite = context.suites.add(suite)
        
        validation_def = context.validation_definitions.add(
            gx.ValidationDefinition(name="taxi_val", data=batch_definition, suite=suite)
        )
        
        action_list = [gx.checkpoint.actions.UpdateDataDocsAction(name="update_data_docs")]
        
        checkpoint = context.checkpoints.add(
            gx.Checkpoint(name="taxi_cp", validation_definitions=[validation_def], actions=action_list)
        )
        
        batch_parameters = {"dataframe": df}
        result = checkpoint.run(batch_parameters=batch_parameters)
        val_result = list(result.run_results.values())[0]
        eval_results = [r.to_json_dict() for r in val_result.results]
        
        # Tally metrics from GX outputs. unexpected_count is always present on a
        # successfully-evaluated map expectation (unlike unexpected_index_list, which
        # requires result_format="COMPLETE" and isn't populated here), so we rely on it
        # for every category except "string in number", which we compute ourselves above
        # since coercing fare_amount already discarded the original text values.
        missing_values_count = 0
        negative_fare_count = 0
        wrong_pass_count = 0
        wrong_coord_count = 0
        future_count = 0
        ocean_count = 0

        missing_col = False

        for res in eval_results:
            exp_type = res["expectation_config"]["type"]
            success = res["success"]
            col = res.get("expectation_config", {}).get("kwargs", {}).get("column")

            if not success:
                if exp_type == "expect_column_to_exist":
                    missing_col = True
                else:
                    count = res.get("result", {}).get("unexpected_count", 0) or 0
                    if exp_type == "expect_column_values_to_not_be_null":
                        missing_values_count = count
                    elif exp_type == "expect_column_values_to_be_between":
                        if col == "fare_amount":
                            negative_fare_count = count
                        elif col == "passenger_count":
                            wrong_pass_count = count
                        elif "latitude" in col or "longitude" in col:
                            wrong_coord_count = max(wrong_coord_count, count)
                    elif exp_type == "expect_column_values_to_not_match_regex":
                        future_count = count
                    elif exp_type == "expect_column_values_to_not_be_in_set":
                        ocean_count = count

        string_number_count = int(string_in_number_mask.sum())

        # Fallback split implementation in pandas — this is also the source of truth for
        # valid/invalid counts and criticality, since GX's default result_format doesn't
        # expose per-row unexpected indices (only unexpected_count, a scalar).
        valid_mask = pd.Series(True, index=df.index)
        if not missing_col:
            valid_mask &= df['passenger_count'].notna()
            valid_mask &= df['fare_amount'].between(0.01, 1000.0)
            valid_mask &= df['passenger_count'].between(1, 6)
            valid_mask &= pd.to_numeric(df['pickup_latitude'], errors='coerce').between(40.0, 42.0)
            valid_mask &= pd.to_numeric(df['pickup_longitude'], errors='coerce').between(-75.0, -72.0)
            valid_mask &= (df['dropoff_latitude'] != 0.0)
        else:
            valid_mask = pd.Series(False, index=df.index)

        valid_rows = df[valid_mask].to_dict(orient="records")
        invalid_rows = df[~valid_mask].to_dict(orient="records")

        invalid_count = len(invalid_rows)
        invalid_pct = invalid_count / total_rows if total_rows > 0 else 0

        # Criticality tiers per spec:
        #   High   : missing column OR > 50% invalid rows
        #   Medium : 10% - 50% invalid rows
        #   Low    : > 0% and < 10% invalid rows  (no alert - avoids alert fatigue)
        criticality = "None"
        if missing_col or invalid_pct > 0.5:
            criticality = "High"
        elif invalid_pct >= 0.10:
            criticality = "Medium"
        elif invalid_count > 0:
            criticality = "Low"

        # Get docs URL and copy to persistent volume served by Nginx
        site_urls = context.build_data_docs()
        docs_url = "No Docs URL"
        if site_urls:
            local_site_url = list(site_urls.values())[0]
            local_site_path = local_site_url.replace("file://", "")
            local_site_dir = os.path.dirname(local_site_path)
            
            target_dir = "/opt/airflow/data/gx/uncommitted/data_docs/local_site"
            os.makedirs(target_dir, exist_ok=True)
            
            if os.path.exists(local_site_dir):
                import subprocess
                subprocess.run(["cp", "-r", f"{local_site_dir}/.", target_dir], check=False)
                
            # Copy validation report to a friendly filename
            html_files = glob.glob(os.path.join(target_dir, "validations/**/*.html"), recursive=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            report_filename = f"report_{timestamp}.html"
            report_path = os.path.join(target_dir, report_filename)
            
            if html_files:
                latest_html = max(html_files, key=os.path.getmtime)
                shutil.copy2(latest_html, report_path)
                docs_url = f"http://localhost:8082/{report_filename}"
            else:
                docs_url = f"http://localhost:8082/index.html"
        
        return {
            "skip": False,
            "filename": os.path.basename(filepath),
            "total": total_rows,
            "valid_count": len(valid_rows),
            "invalid_count": len(invalid_rows),
            "missing_values": missing_values_count,
            "negative_fares": negative_fare_count,
            "wrong_passengers": wrong_pass_count,
            "wrong_coordinates": wrong_coord_count,
            "string_in_number": string_number_count,
            "future_dates": future_count,
            "zero_coordinates": ocean_count,
            "schema_errors": total_rows if missing_col else 0,
            "criticality": criticality,
            "docs_url": docs_url,
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows
        }

    @task
    def save_statistics(val_result: dict):
        if val_result.get("skip", True):
            return
            
        hook = PostgresHook(postgres_conn_id="taxi_db")
        
        insert_sql = """
            INSERT INTO ingestion_stats
            (filename, ingested_at, total_rows, valid_rows, invalid_rows,
             missing_values_count, negative_fare_count, wrong_passengers_count,
             wrong_coordinates_count, string_in_number_count, future_date_count,
             zero_coordinates_count, schema_error_count, status)
            VALUES
            (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        hook.run(insert_sql, parameters=(
            val_result["filename"],
            val_result["total"],
            val_result["valid_count"],
            val_result["invalid_count"],
            val_result["missing_values"],
            val_result["negative_fares"],
            val_result["wrong_passengers"],
            val_result["wrong_coordinates"],
            val_result["string_in_number"],
            val_result["future_dates"],
            val_result["zero_coordinates"],
            val_result["schema_errors"],
            val_result["criticality"]
        ))
        logging.info("Saved ingestion stats to database successfully.")

    @task
    def send_alerts(val_result: dict):
        if val_result.get("skip", True):
            return
            
        if val_result["criticality"] not in ["High", "Medium"]:
            logging.info("Criticality is Low or None. No alert sent.")
            return
            
        invalid_pct = (val_result["invalid_count"] / val_result["total"] * 100) if val_result["total"] else 0.0
        msg = f"Data Quality Alert! Criticality: {val_result['criticality']}\n"
        msg += f"File: {val_result['filename']}\n"
        msg += f"Total Rows: {val_result['total']}\n"
        msg += f"Invalid Rows: {val_result['invalid_count']} ({invalid_pct:.1f}%)\n"
        msg += f"Missing Values: {val_result['missing_values']}\n"
        msg += f"Negative Fares: {val_result['negative_fares']}\n"
        msg += f"Wrong Passengers: {val_result['wrong_passengers']}\n"
        msg += f"Wrong Coordinates: {val_result['wrong_coordinates']}\n"
        msg += f"String-in-number: {val_result['string_in_number']}\n"
        msg += f"Future Dates: {val_result['future_dates']}\n"
        msg += f"Zero Coordinates: {val_result['zero_coordinates']}\n"
        msg += f"Schema Errors: {val_result['schema_errors']}\n"
        msg += f"Report Available At: {val_result['docs_url']}"
        
        logging.info("Sending Teams Alert: " + msg)
        try:
            resp = requests.post(
                TEAMS_WEBHOOK_URL,
                json={"text": msg},
                timeout=10
            )
            logging.info(f"Teams webhook response status={resp.status_code}")
            logging.info(f"Teams webhook response body={resp.text[:1000]}")
            if resp.status_code >= 400:
                logging.error(f"Teams webhook returned error status {resp.status_code}")
        except Exception as e:
            logging.error(f"Failed to post to webhook: {e}")

    @task
    def split_and_save_data(val_result: dict):
        if val_result.get("skip", True):
            return
            
        os.makedirs(GOOD_DIR, exist_ok=True)
        os.makedirs(BAD_DIR, exist_ok=True)
        
        filename = val_result["filename"]
        valid_rows = val_result["valid_rows"]
        invalid_rows = val_result["invalid_rows"]
        
        if valid_rows:
            good_file = os.path.join(GOOD_DIR, filename)
            pd.DataFrame(valid_rows).to_csv(good_file, index=False)
            logging.info(f"Saved {len(valid_rows)} good rows to {good_file}")
            
        if invalid_rows:
            # Append _bad to filename
            name, ext = os.path.splitext(filename)
            bad_file = os.path.join(BAD_DIR, f"{name}_bad{ext}")
            pd.DataFrame(invalid_rows).to_csv(bad_file, index=False)
            logging.info(f"Saved {len(invalid_rows)} bad rows to {bad_file}")

    # Wire tasks
    raw_info = read_data()
    validation = validate_data(raw_info)
    
    # These three run in parallel after validation
    save_statistics(validation)
    send_alerts(validation)
    split_and_save_data(validation)

ingest_taxi_data()
