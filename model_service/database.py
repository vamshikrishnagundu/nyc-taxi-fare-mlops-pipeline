from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
#it brdges the gap between python and postgresql
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/taxifare"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

#kind of postgres schema but in python
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    pickup_longitude = Column(Float)
    pickup_latitude = Column(Float)
    dropoff_longitude = Column(Float)
    dropoff_latitude = Column(Float)
    passenger_count = Column(Integer)
    hour = Column(Integer)
    day_of_week = Column(Integer)
    month = Column(Integer)
    distance_km = Column(Float)
    pickup_datetime = Column(String)
    predicted_fare = Column(Float)
    prediction_source = Column(String, default="webapp")
    model_version = Column(String, default="local_v1")
    created_at = Column(DateTime, default=datetime.utcnow)


class IngestionStats(Base):
    __tablename__ = "ingestion_stats"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    total_rows = Column(Integer, default=0)
    valid_rows = Column(Integer, default=0)
    invalid_rows = Column(Integer, default=0)
    missing_values_count = Column(Integer, default=0)
    negative_fare_count = Column(Integer, default=0)
    wrong_passengers_count = Column(Integer, default=0)
    wrong_coordinates_count = Column(Integer, default=0)
    string_in_number_count = Column(Integer, default=0)
    future_date_count = Column(Integer, default=0)
    zero_coordinates_count = Column(Integer, default=0)
    schema_error_count = Column(Integer, default=0)
    status = Column(String, default="good")


class TrainingStats(Base):
    __tablename__ = "training_stats"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String, nullable=False)
    promoted_at = Column(DateTime, default=datetime.utcnow)
    distance_mean = Column(Float)
    distance_std = Column(Float)
    passenger_1_pct = Column(Float)
    passenger_2_pct = Column(Float)
    passenger_3_pct = Column(Float)
    passenger_4_pct = Column(Float)
    passenger_5_pct = Column(Float)
    passenger_6_pct = Column(Float)
    training_r2 = Column(Float)
    training_mae = Column(Float)
    is_promoted = Column(Boolean, default=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Database tables created!")