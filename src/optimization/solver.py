from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.features.engineering import encode_time_cyclical, haversine


def _ensure_depot(orders_df: pd.DataFrame) -> pd.DataFrame:
    required = {"lat", "lon"}
    missing = required.difference(orders_df.columns)
    if missing:
        raise ValueError(f"orders_df is missing required columns: {sorted(missing)}")

    frame = orders_df.copy().reset_index(drop=True)
    if "is_depot" in frame.columns and frame["is_depot"].astype(bool).any():
        depot_idx = int(frame.index[frame["is_depot"].astype(bool)][0])
        if depot_idx != 0:
            first = frame.iloc[[depot_idx]].copy()
            rest = frame.drop(index=depot_idx)
            frame = pd.concat([first, rest], ignore_index=True)
        return frame

    max_due = float(frame["due_time_min"].max()) if "due_time_min" in frame.columns else 24 * 60.0
    depot_row = {
        "order_id": 0,
        "lat": float(frame["lat"].mean()),
        "lon": float(frame["lon"].mean()),
        "demand": 0,
        "service_time_min": 0,
        "ready_time_min": 0,
        "due_time_min": max_due + 120.0,
        "is_depot": True,
    }
    frame = pd.concat([pd.DataFrame([depot_row]), frame], ignore_index=True)
    if "is_depot" not in frame.columns:
        frame["is_depot"] = False
        frame.loc[0, "is_depot"] = True
    return frame


def _extract_temporal_context(frame: pd.DataFrame) -> tuple[int, int]:
    if "created_ts" in frame.columns:
        ts = pd.to_datetime(frame["created_ts"], utc=True, errors="coerce")
        if ts.notna().any():
            hour = int(ts.dt.hour.mode().iloc[0])
            weekday = int(ts.dt.dayofweek.mode().iloc[0])
            return hour, weekday
    return 12, 2


def _pair_features(frame: pd.DataFrame, i: int, j: int, hour: int, weekday: int) -> dict[str, float]:
    p1 = frame.iloc[i]
    p2 = frame.iloc[j]
    dist = haversine(
        pd.Series([p1["lat"]]),
        pd.Series([p1["lon"]]),
        pd.Series([p2["lat"]]),
        pd.Series([p2["lon"]]),
    ).iloc[0]
    cyc = encode_time_cyclical(pd.Series([hour]))
    return {
        "distance_m": float(dist),
        "haversine_m": float(dist),
        "pickup_hour": float(hour),
        "pickup_weekday": float(weekday),
        "is_weekend": float(1 if weekday >= 5 else 0),
        "hour_sin": float(cyc["hour_sin"].iloc[0]),
        "hour_cos": float(cyc["hour_cos"].iloc[0]),
    }


def build_time_matrix(
    orders_df: pd.DataFrame,
    model_path: str | Path,
    baseline_type: str,
) -> np.ndarray:
    """build pairwise travel-time matrix in seconds

    baseline_type supports: "constant", "median", "ml"
    """
    frame = _ensure_depot(orders_df)
    n = len(frame)
    matrix = np.zeros((n, n), dtype=np.int64)
    hour, weekday = _extract_temporal_context(frame)

    model = None
    feature_names: list[str] | None = None
    if baseline_type == "ml":
        model_file = Path(model_path)
        model = joblib.load(model_file)
        feature_cfg = model_file.parent / "feature_config.json"
        if feature_cfg.exists():
            payload = json.loads(feature_cfg.read_text(encoding="utf-8"))
            feature_names = list(payload.get("feature_names", []))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            feats = _pair_features(frame, i, j, hour=hour, weekday=weekday)

            if baseline_type == "constant":
                eta = feats["distance_m"] / 7.0
            elif baseline_type == "median":
                eta = feats["distance_m"] / 6.0
            elif baseline_type == "ml":
                x = pd.DataFrame([feats])
                if feature_names:
                    x = x.reindex(columns=feature_names, fill_value=0.0)
                eta = float(model.predict(x)[0])
            else:
                raise ValueError("baseline_type must be one of: constant, median, ml")

            eta = max(1.0, eta)
            matrix[i, j] = int(round(eta))
    return matrix


def solve_vrptw(
    time_matrix: np.ndarray,
    orders_df: pd.DataFrame,
    num_vehicles: int = 3,
) -> dict[str, Any]:
    """solve VRPTW and return parsed routes with metrics"""
    frame = _ensure_depot(orders_df)
    if time_matrix.shape != (len(frame), len(frame)):
        raise ValueError("time_matrix shape does not match orders_df size (after depot handling).")

    ready = frame.get("ready_time_min", pd.Series([0] * len(frame))).fillna(0).astype(int).to_numpy()
    due = (
        frame.get("due_time_min", pd.Series([24 * 60] * len(frame)))
        .fillna(24 * 60)
        .astype(int)
        .to_numpy()
    )
    service = frame.get("service_time_min", pd.Series([0] * len(frame))).fillna(0).astype(int).to_numpy()
    demand = frame.get("demand", pd.Series([0] * len(frame))).fillna(0).astype(int).to_numpy()
    due[0] = max(due[0], int(due.max()))

    solve_mode = "strict"
    solution = None
    manager: pywrapcp.RoutingIndexManager | None = None
    routing: pywrapcp.RoutingModel | None = None
    time_dimension = None

    for window_buffer_min in (0, 120, 360):
        due_adj = np.maximum(due, ready + 1) + window_buffer_min
        due_adj[0] = max(due_adj[0], int(due_adj.max()))

        manager = pywrapcp.RoutingIndexManager(len(frame), num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def time_cb(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(time_matrix[from_node, to_node] + service[from_node] * 60)

        transit_callback_idx = routing.RegisterTransitCallback(time_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

        def demand_cb(from_index: int) -> int:
            return int(demand[manager.IndexToNode(from_index)])

        demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        capacity = max(10, int(np.ceil(demand.sum() / max(1, num_vehicles) * 1.25)))
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_idx,
            0,
            [capacity] * num_vehicles,
            True,
            "Capacity",
        )

        horizon = int(max(due_adj.max(), 24 * 60) * 60)
        routing.AddDimension(
            transit_callback_idx,
            60 * 60,
            horizon,
            False,
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        for node in range(len(frame)):
            idx = manager.NodeToIndex(node)
            time_dimension.CumulVar(idx).SetRange(int(ready[node] * 60), int(due_adj[node] * 60))

        for vehicle_id in range(num_vehicles):
            start_idx = routing.Start(vehicle_id)
            end_idx = routing.End(vehicle_id)
            time_dimension.CumulVar(start_idx).SetRange(int(ready[0] * 60), int(due_adj[0] * 60))
            time_dimension.CumulVar(end_idx).SetRange(0, horizon)

        search = pywrapcp.DefaultRoutingSearchParameters()
        search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search.time_limit.seconds = 10

        solution = routing.SolveWithParameters(search)
        if solution is not None:
            solve_mode = "strict" if window_buffer_min == 0 else f"relaxed_plus_{window_buffer_min}m"
            break

    if solution is None or manager is None or routing is None or time_dimension is None:
        return {"status": "no_solution", "routes": [], "metrics": {}}

    routes: list[dict[str, Any]] = []
    total_time = 0
    total_load = 0
    served_orders = 0

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        stops: list[dict[str, Any]] = []
        route_time = 0
        route_load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            arrival_sec = solution.Value(time_dimension.CumulVar(index))
            route_load += int(demand[node])
            if node != 0:
                served_orders += 1
            stops.append(
                {
                    "node": int(node),
                    "order_id": int(frame.iloc[node].get("order_id", node)),
                    "arrival_sec": int(arrival_sec),
                    "ready_sec": int(ready[node] * 60),
                    "due_sec": int(due[node] * 60),
                    "demand": int(demand[node]),
                }
            )
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            route_time += int(time_matrix[node, next_node])
            index = next_index

        arrival_end = solution.Value(time_dimension.CumulVar(index))
        stops.append(
            {
                "node": 0,
                "order_id": int(frame.iloc[0].get("order_id", 0)),
                "arrival_sec": int(arrival_end),
                "ready_sec": int(ready[0] * 60),
                "due_sec": int(due[0] * 60),
                "demand": 0,
            }
        )
        total_time += route_time
        total_load += route_load
        routes.append(
            {
                "vehicle_id": vehicle_id,
                "route_time_sec": int(route_time),
                "route_load": int(route_load),
                "stops": stops,
            }
        )

    return {
        "status": "solved",
        "routes": routes,
        "metrics": {
            "total_time_sec": int(total_time),
            "total_load": int(total_load),
            "served_orders": int(served_orders),
            "num_routes": int(len(routes)),
            "solve_mode": solve_mode,
        },
    }
