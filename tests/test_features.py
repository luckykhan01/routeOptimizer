from __future__ import annotations

import pandas as pd

from src.features.engineering import build_features


def test_build_features_no_nan_and_positive_target() -> None:
    trips = pd.DataFrame(
        {
            "pickup_lat": [40.75, 40.76, 40.74],
            "pickup_lon": [-73.99, -73.98, -73.97],
            "dropoff_lat": [40.77, 40.75, 40.73],
            "dropoff_lon": [-73.95, -73.96, -73.98],
            "pickup_ts": [
                "2025-01-01T08:00:00Z",
                "2025-01-01T09:00:00Z",
                "2025-01-01T10:00:00Z",
            ],
            "distance_m": [1200.0, 2200.0, 1600.0],
            "travel_time_sec": [310.0, 530.0, 400.0],
        }
    )
    features = build_features(trips)
    assert len(features) == 3
    assert not features.isna().any().any()
    assert (features["target_eta_sec"] > 0).all()
