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
    if n <= 1:
        return matrix

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

    # Vectorized pair generation
    i_idx, j_idx = np.where(~np.eye(n, dtype=bool))
    
    lat1 = frame["lat"].iloc[i_idx].reset_index(drop=True)
    lon1 = frame["lon"].iloc[i_idx].reset_index(drop=True)
    lat2 = frame["lat"].iloc[j_idx].reset_index(drop=True)
    lon2 = frame["lon"].iloc[j_idx].reset_index(drop=True)
    
    dist = haversine(lat1, lon1, lat2, lon2)
    
    if baseline_type == "constant":
        etas = dist / 7.0
    elif baseline_type == "median":
        etas = dist / 6.0
    elif baseline_type == "ml":
        cyc = encode_time_cyclical(pd.Series([hour] * len(dist)))
        feats = pd.DataFrame({
            "distance_m": dist,
            "haversine_m": dist,
            "pickup_hour": float(hour),
            "pickup_weekday": float(weekday),
            "is_weekend": float(1 if weekday >= 5 else 0),
            "hour_sin": cyc["hour_sin"].values,
            "hour_cos": cyc["hour_cos"].values,
        })
        if feature_names:
            feats = feats.reindex(columns=feature_names, fill_value=0.0)
        etas = model.predict(feats)
    else:
        raise ValueError("baseline_type must be one of: constant, median, ml")

    etas = np.maximum(1.0, etas)
    matrix[i_idx, j_idx] = np.round(etas).astype(np.int64)
    return matrix


def solve_vrptw(
    time_matrix: np.ndarray,
    orders_df: pd.DataFrame,
    num_vehicles: int = 3,
) -> dict[str, Any]:
    """solve VRPTW and return parsed routes with metrics.

    uses AddDisjunction to allow dropping orders that can't fit,
    returning partial solutions instead of no_solution.
    """
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

    # relax time windows: add buffer to due times so the solver has room
    window_buffer_min = 120
    due_adj = np.maximum(due, ready + 1) + window_buffer_min
    due_adj[0] = max(due_adj[0], int(due_adj.max()) + 60)

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
    capacity = max(10, int(np.ceil(demand.sum() / max(1, num_vehicles) * 2.0)))
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx,
        0,
        [capacity] * num_vehicles,
        True,
        "Capacity",
    )

    horizon = int(max(due_adj.max(), 24 * 60) * 60)
    slack = 4 * 3600  # 4 hours slack for waiting
    routing.AddDimension(
        transit_callback_idx,
        slack,
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
        time_dimension.CumulVar(start_idx).SetRange(0, int(due_adj[0] * 60))
        time_dimension.CumulVar(end_idx).SetRange(0, horizon)

    # allow dropping any order node (not depot) with a high penalty
    # this ensures we always get a solution — the solver drops orders it can't fit
    drop_penalty = int(time_matrix.max() * 10)
    for node in range(1, len(frame)):
        routing.AddDisjunction([manager.NodeToIndex(node)], drop_penalty)

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.seconds = 15

    solution = routing.SolveWithParameters(search)

    if solution is None:
        return {"status": "no_solution", "routes": [], "metrics": {}}

    routes: list[dict[str, Any]] = []
    total_time = 0
    total_load = 0
    served_orders = 0
    dropped_orders: list[int] = []

    # collect served node ids to find dropped ones
    served_nodes: set[int] = set()

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
                served_nodes.add(node)
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

    # identify dropped orders
    for node in range(1, len(frame)):
        if node not in served_nodes:
            dropped_orders.append(int(frame.iloc[node].get("order_id", node)))

    total_orders = len(frame) - 1  # exclude depot
    return {
        "status": "solved",
        "routes": routes,
        "metrics": {
            "total_time_sec": int(total_time),
            "total_load": int(total_load),
            "served_orders": int(served_orders),
            "total_orders": int(total_orders),
            "dropped_orders": len(dropped_orders),
            "dropped_order_ids": dropped_orders,
            "num_routes": int(len(routes)),
        },
    }


def solve_vrptw_compare(
    orders_df: pd.DataFrame,
    model_path: str | Path,
    num_vehicles: int = 3,
) -> dict[str, Any]:
    """run solver with all three baseline types and return comparative metrics"""
    results: dict[str, Any] = {}
    for baseline_type in ("ml", "constant", "median"):
        tm = build_time_matrix(
            orders_df=orders_df,
            model_path=model_path,
            baseline_type=baseline_type,
        )
        sol = solve_vrptw(time_matrix=tm, orders_df=orders_df, num_vehicles=num_vehicles)
        results[baseline_type] = {
            "status": sol["status"],
            "metrics": sol.get("metrics", {}),
        }
    return results

