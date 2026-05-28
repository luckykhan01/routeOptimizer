from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_optimize_json_constant_baseline() -> None:
    payload = {
        "baseline_type": "constant",
        "num_vehicles": 2,
        "orders": [
            {
                "order_id": 0,
                "lat": 40.7580,
                "lon": -73.9855,
                "ready_time_min": 0,
                "due_time_min": 500,
                "demand": 0,
                "service_time_min": 0,
                "is_depot": True,
                "created_ts": "2025-01-31T08:00:00Z",
            },
            {
                "order_id": 1,
                "lat": 40.7612,
                "lon": -73.9776,
                "ready_time_min": 10,
                "due_time_min": 300,
                "demand": 1,
                "service_time_min": 5,
                "created_ts": "2025-01-31T08:05:00Z",
            },
            {
                "order_id": 2,
                "lat": 40.7498,
                "lon": -73.9876,
                "ready_time_min": 20,
                "due_time_min": 320,
                "demand": 1,
                "service_time_min": 6,
                "created_ts": "2025-01-31T08:15:00Z",
            },
        ],
    }
    with TestClient(app) as client:
        response = client.post("/optimize", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"solved", "no_solution"}
    assert "metrics" in body
