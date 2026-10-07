"""Unit tests for the model_service feature-engineering helpers.

These cover the pure logic (distance maths + feature vector construction) that
every prediction depends on, without needing a loaded model or a database.
"""
import pytest

from main import haversine_distance, prepare_features, TaxiRide


def test_haversine_zero_distance():
    """Identical pickup and dropoff points must be zero km apart."""
    assert haversine_distance(40.75, -73.98, 40.75, -73.98) == pytest.approx(0.0, abs=1e-9)


def test_haversine_one_degree_latitude():
    """One degree of latitude is ~111 km anywhere on Earth."""
    d = haversine_distance(40.0, -73.0, 41.0, -73.0)
    assert d == pytest.approx(111.195, abs=0.5)


def test_haversine_is_symmetric():
    """Distance from A to B equals distance from B to A."""
    a = haversine_distance(40.0, -73.0, 40.7, -74.0)
    b = haversine_distance(40.7, -74.0, 40.0, -73.0)
    assert a == pytest.approx(b, abs=1e-9)


def _sample_ride(passenger_count=1):
    # 2015-01-01 is a Thursday (weekday() == 3)
    return TaxiRide(
        pickup_datetime="2015-01-01 00:00:00 UTC",
        pickup_longitude=-73.98,
        pickup_latitude=40.75,
        dropoff_longitude=-73.96,
        dropoff_latitude=40.77,
        passenger_count=passenger_count,
    )


def test_prepare_features_returns_nine_values():
    """The model expects exactly nine features per ride."""
    assert len(prepare_features(_sample_ride())) == 9


def test_prepare_features_extracts_datetime_parts():
    """hour / weekday / month are derived from pickup_datetime."""
    feats = prepare_features(_sample_ride(passenger_count=2))
    # order: [lon, lat, dlon, dlat, passengers, hour, weekday, month, distance]
    _, _, _, _, passengers, hour, weekday, month, distance = feats
    assert passengers == 2
    assert hour == 0
    assert weekday == 3       # Thursday
    assert month == 1
    assert distance > 0       # pickup != dropoff


def test_prepare_features_handles_bad_datetime():
    """An unparseable datetime must fall back gracefully, not raise."""
    ride = _sample_ride()
    ride.pickup_datetime = "not-a-date"
    feats = prepare_features(ride)          # falls back to datetime.now()
    assert len(feats) == 9
