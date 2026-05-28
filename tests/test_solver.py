from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features.engineering import encode_time_cyclical, haversine
from src.optimization.solver import build_time_matrix, solve_vrptw


def _build_orders() -> pd.DataFrame:
    rows = [
        {"order_id": 0, "lat": 40.7580, "lon": -73.9855, "ready_time_min": 0, "due_time_min": 500, "demand": 0, "service_time_min": 0, "is_depot": True, "created_ts": "2025-01-31T08:00:00Z"},
        {"order_id": 1, "lat": 40.7612, "lon": -73.9776, "ready_time_min": 10, "due_time_min": 220, "demand": 1, "service_time_min": 5, "created_ts": "2025-01-31T08:05:00Z"},
        {"order_id": 2, "lat": 40.7498, "lon": -73.9876, "ready_time_min": 15, "due_time_min": 240, "demand": 1, "service_time_min": 5, "created_ts": "2025-01-31T08:15:00Z"},
        {"order_id": 3, "lat": 40.7440, "lon": -73.9920, "ready_time_min": 25, "due_time_min": 260, "demand": 1, "service_time_min": 6, "created_ts": "2025-01-31T08:20:00Z"},
        {"order_id": 4, "lat": 40.7337, "lon": -73.9903, "ready_time_min": 35, "due_time_min": 320, "demand": 1, "service_time_min": 5, "created_ts": "2025-01-31T08:30:00Z"},
        {"order_id": 5, "lat": 40.7411, "lon": -73.9785, "ready_time_min": 40, "due_time_min": 340, "demand": 1, "service_time_min": 4, "created_ts": "2025-01-31T08:35:00Z"},
    ]
    return pd.DataFrame(rows)


def _train_dummy_eta_model(model_path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 300
    lat1 = pd.Series(40.70 + rng.uniform(0.0, 0.1, size=n))
    lon1 = pd.Series(-74.02 + rng.uniform(0.0, 0.1, size=n))
    lat2 = pd.Series(40.70 + rng.uniform(0.0, 0.1, size=n))
    lon2 = pd.Series(-74.02 + rng.uniform(0.0, 0.1, size=n))
    distance_m = haversine(lat1, lon1, lat2, lon2).to_numpy()
    hour = rng.integers(0, 24, size=n)
    weekday = rng.integers(0, 7, size=n)
    cyc = encode_time_cyclical(pd.Series(hour))

    x = pd.DataFrame(
        {
            "distance_m": distance_m,
            "haversine_m": distance_m,
            "pickup_hour": hour.astype(float),
            "pickup_weekday": weekday.astype(float),
            "is_weekend": (weekday >= 5).astype(float),
            "hour_sin": cyc["hour_sin"].to_numpy(),
            "hour_cos": cyc["hour_cos"].to_numpy(),
        }
    )
    y = np.maximum(30.0, distance_m / 10.0 + 20.0 + rng.normal(0, 8, size=n))

    model = LGBMRegressor(
        objective="regression",
        n_estimators=80,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        n_jobs=6,
        device="cpu",
    )
    model.fit(x, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    meta = {"feature_names": list(x.columns), "target_name": "target_eta_sec", "n_features": x.shape[1]}
    (model_path.parent / "feature_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def test_solver_solution_time_windows_and_improvement(tmp_path: Path) -> None:
    orders_df = _build_orders()
    model_path = tmp_path / "eta_lgbm.pkl"
    _train_dummy_eta_model(model_path)

    constant_matrix = build_time_matrix(orders_df=orders_df, model_path=model_path, baseline_type="constant")
    ml_matrix = build_time_matrix(orders_df=orders_df, model_path=model_path, baseline_type="ml")

    baseline_result = solve_vrptw(constant_matrix, orders_df=orders_df, num_vehicles=2)
    ml_result = solve_vrptw(ml_matrix, orders_df=orders_df, num_vehicles=2)

    assert ml_result["status"] == "solved"
    assert len(ml_result["routes"]) > 0

    for route in ml_result["routes"]:
        for stop in route["stops"]:
            assert stop["arrival_sec"] >= stop["ready_sec"]
            assert stop["arrival_sec"] <= stop["due_sec"]

    assert ml_result["metrics"]["total_time_sec"] < baseline_result["metrics"]["total_time_sec"]
