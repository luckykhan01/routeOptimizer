"""tests for ETA model training pipeline"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.models.train_eta import load_config, prepare_dataset, train_pipeline


def _create_synthetic_trips(path: Path, rows: int = 360, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp("2025-01-01T00:00:00Z")

    pickup_lat = 40.70 + rng.uniform(0.0, 0.12, size=rows)
    pickup_lon = -74.02 + rng.uniform(0.0, 0.10, size=rows)
    dropoff_lat = 40.70 + rng.uniform(0.0, 0.12, size=rows)
    dropoff_lon = -74.02 + rng.uniform(0.0, 0.10, size=rows)
    pickup_hour = rng.integers(0, 24, size=rows)
    distance_m = rng.uniform(900.0, 12500.0, size=rows)
    noise = rng.normal(0.0, 45.0, size=rows)

    hour_penalty = np.where(np.isin(pickup_hour, [7, 8, 9, 17, 18, 19]), 220.0, 0.0)
    travel_time_sec = (distance_m / 8.5) + hour_penalty + noise + 120.0
    travel_time_sec = np.maximum(travel_time_sec, 60.0)

    trips = pd.DataFrame(
        {
            "trip_id": np.arange(rows),
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "dropoff_lat": dropoff_lat,
            "dropoff_lon": dropoff_lon,
            "pickup_ts": [
                (start_ts + pd.to_timedelta(int(i * 10), unit="m")).isoformat() for i in range(rows)
            ],
            "distance_m": distance_m,
            "travel_time_sec": travel_time_sec,
        }
    )
    trips.to_csv(path, index=False)


def _create_test_config(tmp_path: Path, trips_path: Path) -> Path:
    config_path = tmp_path / "train.yaml"
    content = f"""
seed: 42
n_jobs: 6
paths:
  raw_trips: {trips_path.as_posix()}
  raw_orders: {tmp_path.as_posix()}/orders.csv
  processed_dir: {tmp_path.as_posix()}/processed
  model_dir: {tmp_path.as_posix()}/models
  model_path: {tmp_path.as_posix()}/models/eta_lgbm.pkl
  feature_config_path: {tmp_path.as_posix()}/models/feature_config.json
  mlflow_tracking_uri: {tmp_path.as_posix()}/mlruns
  mlflow_experiment: routeOptimizer-test
lightgbm:
  objective: regression
  metric: mae
  boosting_type: gbdt
  n_estimators: 120
  learning_rate: 0.08
  num_leaves: 31
  max_depth: -1
  min_child_samples: 15
  subsample: 0.9
  colsample_bytree: 0.9
  reg_alpha: 0.0
  reg_lambda: 0.0
  random_state: 42
  n_jobs: 6
  device: cpu
"""
    config_path.write_text(content.strip() + "\n", encoding="utf-8")
    return config_path


def test_train_pipeline_and_metrics_vs_baseline(tmp_path: Path) -> None:
    trips_path = tmp_path / "trips.csv"
    _create_synthetic_trips(path=trips_path)
    cfg_path = _create_test_config(tmp_path=tmp_path, trips_path=trips_path)

    result = train_pipeline(config_path=cfg_path)

    assert Path(result["model_path"]).exists()
    assert Path(result["feature_config_path"]).exists()
    assert result["metrics"]["mae_cv_mean"] > 0

    feature_meta = json.loads(Path(result["feature_config_path"]).read_text(encoding="utf-8"))
    assert feature_meta["n_features"] == len(feature_meta["feature_names"])

    cfg = load_config(cfg_path)
    x, y = prepare_dataset(Path(cfg["paths"]["raw_trips"]))
    tscv = TimeSeriesSplit(n_splits=3)
    _, test_idx = list(tscv.split(x))[-1]
    x_test = x.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model = joblib.load(result["model_path"])
    pred = model.predict(x_test)

    assert pred.shape[0] == x_test.shape[0]
    assert np.all(pred > 0)

    baseline_pred = x_test["distance_m"].to_numpy() / 7.0
    model_mae = mean_absolute_error(y_test, pred)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    assert model_mae < baseline_mae
