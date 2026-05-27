from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def haversine(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    radius_m = 6_371_000.0
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return pd.Series(radius_m * c, index=lat1.index, name="haversine_m")


def encode_time_cyclical(hours: pd.Series) -> pd.DataFrame:
    """encode hour-of-day with cyclical sin/cos projection"""
    normalized = (hours.astype(float) % 24.0) / 24.0
    radians = normalized * (2.0 * math.pi)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(radians),
            "hour_cos": np.cos(radians),
        },
        index=hours.index,
    )


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_feature_frame(features: pd.DataFrame, target_col: str) -> None:
    if len(features) == 0:
        raise ValueError("Feature frame is empty")
    if features.isna().any().any():
        raise ValueError("Feature frame contains NaN values")
    if (features[target_col] <= 0).any():
        raise ValueError(f"Target column {target_col} contains non-positive values")
    LOGGER.info("Validated features: rows=%s, cols=%s, no NaN", features.shape[0], features.shape[1])


def build_features(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Build ETA-ready features from raw trips data"""
    required_columns = (
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
        "pickup_ts",
        "distance_m",
        "travel_time_sec",
    )
    _validate_columns(trips_df, required_columns)

    frame = trips_df.copy()
    frame["pickup_ts"] = pd.to_datetime(frame["pickup_ts"], utc=True, errors="raise")
    frame["pickup_hour"] = frame["pickup_ts"].dt.hour
    frame["pickup_weekday"] = frame["pickup_ts"].dt.dayofweek
    frame["is_weekend"] = (frame["pickup_weekday"] >= 5).astype(int)

    frame["haversine_m"] = haversine(
        frame["pickup_lat"],
        frame["pickup_lon"],
        frame["dropoff_lat"],
        frame["dropoff_lon"],
    )
    cyc = encode_time_cyclical(frame["pickup_hour"])

    features = pd.concat(
        [
            frame[
                [
                    "distance_m",
                    "haversine_m",
                    "pickup_hour",
                    "pickup_weekday",
                    "is_weekend",
                ]
            ].astype(float),
            cyc.astype(float),
            frame[["travel_time_sec"]].rename(columns={"travel_time_sec": "target_eta_sec"}).astype(float),
        ],
        axis=1,
    )
    _validate_feature_frame(features, target_col="target_eta_sec")
    return features
