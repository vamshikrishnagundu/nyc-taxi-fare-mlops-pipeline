import streamlit as st
import pandas as pd
import requests
from datetime import date

API_URL = "http://model_service:8000"

st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    layout="wide"
)

st.sidebar.title("Taxi Fare Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to:",
    ["Make Predictions", "Past Predictions"]
)

def call_predict_api(rides, source="webapp"):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"rides": rides, "source": source},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure Docker is running!")
        return None

def call_past_predictions_api(start_date=None, end_date=None, source="all"):
    try:
        params = {"source": source}
        if start_date:
            params["start_date"] = str(start_date)
        if end_date:
            params["end_date"] = str(end_date)

        response = requests.get(
            f"{API_URL}/past-predictions",
            params=params,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return []
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure Docker is running!")
        return []


if page == "Make Predictions":
    st.title("NYC Taxi Fare Prediction")
    st.markdown("Fill in the trip details below to predict the fare.")
    st.markdown("---")

    mode = st.radio(
        "Choose prediction type:",
        ["Single Prediction", "Multiple Predictions (CSV)"],
        horizontal=True
    )

    if mode == "Single Prediction":
        st.subheader("Enter Trip Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pickup Location**")
            pickup_datetime = st.text_input(
                "Pickup Date and Time",
                value="2015-06-15 17:00:00 UTC",
                help="Format: YYYY-MM-DD HH:MM:SS UTC"
            )
            pickup_lat = st.number_input(
                "Pickup Latitude",
                value=40.7484,
                min_value=40.0,
                max_value=42.0,
                format="%.4f"
            )
            pickup_lon = st.number_input(
                "Pickup Longitude",
                value=-73.9857,
                min_value=-75.0,
                max_value=-72.0,
                format="%.4f"
            )

        with col2:
            st.markdown("**Dropoff Location**")
            st.markdown("")
            st.markdown("")
            dropoff_lat = st.number_input(
                "Dropoff Latitude",
                value=40.7527,
                min_value=40.0,
                max_value=42.0,
                format="%.4f"
            )
            dropoff_lon = st.number_input(
                "Dropoff Longitude",
                value=-73.9772,
                min_value=-75.0,
                max_value=-72.0,
                format="%.4f"
            )

        passenger_count = st.slider("Passengers", min_value=1, max_value=6, value=1)

        if st.button("Predict Fare", type="primary"):
            ride = {
                "pickup_datetime": pickup_datetime,
                "pickup_longitude": pickup_lon,
                "pickup_latitude": pickup_lat,
                "dropoff_longitude": dropoff_lon,
                "dropoff_latitude": dropoff_lat,
                "passenger_count": passenger_count
            }

            with st.spinner("Predicting..."):
                result = call_predict_api([ride])

            if result:
                fare = result["predictions"][0]
                st.success(f"Predicted Fare: ${fare:.2f}")

                st.markdown("#### Input Details:")
                st.dataframe(pd.DataFrame([{
                    "Pickup DateTime": pickup_datetime,
                    "Pickup Lat": pickup_lat,
                    "Pickup Lon": pickup_lon,
                    "Dropoff Lat": dropoff_lat,
                    "Dropoff Lon": dropoff_lon,
                    "Passengers": passenger_count,
                    "Predicted Fare ($)": fare
                }]), use_container_width=True)

    else:
        st.subheader("Upload CSV File")
        st.markdown("""
        Your CSV must have these columns:
        pickup_datetime, pickup_longitude, pickup_latitude,
        dropoff_longitude, dropoff_latitude, passenger_count
        """)

        sample = pd.DataFrame([{
            "pickup_datetime": "2015-06-15 17:00:00 UTC",
            "pickup_longitude": -73.9857,
            "pickup_latitude": 40.7484,
            "dropoff_longitude": -73.9772,
            "dropoff_latitude": 40.7527,
            "passenger_count": 1
        }])
        st.download_button(
            "Download Sample CSV",
            data=sample.to_csv(index=False),
            file_name="sample.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            required_cols = [
                "pickup_datetime", "pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude", "passenger_count"
            ]
            missing = [c for c in required_cols if c not in df.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if st.button("Predict All Fares", type="primary"):
                    rides = df[required_cols].to_dict(orient="records")

                    with st.spinner(f"Predicting {len(rides)} fares..."):
                        result = call_predict_api(rides)

                    if result:
                        df["predicted_fare ($)"] = result["predictions"]
                        st.success("Done!")
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "Download Results",
                            data=df.to_csv(index=False),
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

else:
    st.title("Past Predictions")
    st.markdown("View all previously made predictions.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date", value=date(2024, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=date.today())
    with col3:
        source = st.selectbox(
            "Prediction Source",
            ["all", "webapp", "scheduled"]
        )

    if st.button("Fetch Predictions", type="primary"):
        with st.spinner("Loading..."):
            data = call_past_predictions_api(start_date, end_date, source)

        if data:
            df = pd.DataFrame(data)
            st.success(f"Found {len(df)} predictions")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(df))
            col2.metric("Avg Fare", f"${df['predicted_fare'].mean():.2f}")
            col3.metric("Max Fare", f"${df['predicted_fare'].max():.2f}")

            st.dataframe(df, use_container_width=True)

        else:
            st.info("No predictions found.")