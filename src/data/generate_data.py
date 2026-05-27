"""synthetic Manhattan trips and orders datasets"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    """config for synthetic data generation"""

    trips_rows: int
    orders_rows: int
    seed: int
    raw_dir: Path


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_manhattan_graph() -> nx.MultiDiGraph:
    """Load drivable Manhattan street graph via OSMnx."""
    manhattan = ox.geocode_to_gdf("Manhattan, New York, USA")
    polygon = manhattan.geometry.iloc[0]
    graph = ox.graph_from_polygon(polygon, network_type="drive", simplify=True)
    graph = ox.distance.add_edge_lengths(graph)
    LOGGER.info("Loaded Manhattan graph: nodes=%s, edges=%s", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _sample_nodes(graph: nx.MultiDiGraph, n: int, rng: np.random.Generator) -> np.ndarray:
    node_ids = np.array(list(graph.nodes))
    return rng.choice(node_ids, size=n, replace=True)


def _node_xy(graph: nx.MultiDiGraph, node_id: int) -> tuple[float, float]:
    data = graph.nodes[node_id]
    return float(data["y"]), float(data["x"])  # lat, lon


def _time_multiplier(hour: int) -> float:
    """synthetic traffic multiplier by time of day."""
    if hour in {7, 8, 9, 17, 18, 19}:
        return 1.35
    if hour in {0, 1, 2, 3, 4, 5}:
        return 0.82
    return 1.0


def _generate_trips(
    graph: nx.MultiDiGraph,
    cfg: GenerationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pickup_nodes = _sample_nodes(graph, cfg.trips_rows, rng)
    dropoff_nodes = _sample_nodes(graph, cfg.trips_rows, rng)

    start = pd.Timestamp("2025-01-01T00:00:00Z")
    pickup_offsets_min = rng.integers(0, 60 * 24 * 30, size=cfg.trips_rows)
    pickup_ts = start + pd.to_timedelta(pickup_offsets_min, unit="m")

    records: list[dict[str, float | int | str]] = []
    for idx, (u, v, ts) in enumerate(zip(pickup_nodes, dropoff_nodes, pickup_ts, strict=True)):
        if u == v:
            continue
        try:
            distance_m = nx.shortest_path_length(graph, source=int(u), target=int(v), weight="length")
        except nx.NetworkXNoPath:
            continue

        hour = int(ts.hour)
        base_speed_m_s = 7.5  # ~27 km/h
        noisy_speed = base_speed_m_s / _time_multiplier(hour)
        stochastic_noise = float(rng.normal(loc=1.0, scale=0.08))
        speed = max(2.5, noisy_speed * stochastic_noise)
        travel_time_sec = float(distance_m / speed)

        pickup_lat, pickup_lon = _node_xy(graph, int(u))
        dropoff_lat, dropoff_lon = _node_xy(graph, int(v))
        records.append(
            {
                "trip_id": idx,
                "pickup_node": int(u),
                "dropoff_node": int(v),
                "pickup_lat": pickup_lat,
                "pickup_lon": pickup_lon,
                "dropoff_lat": dropoff_lat,
                "dropoff_lon": dropoff_lon,
                "pickup_ts": ts.isoformat(),
                "distance_m": float(distance_m),
                "travel_time_sec": max(60.0, travel_time_sec),
            }
        )

    trips_df = pd.DataFrame.from_records(records)
    return trips_df


def _generate_orders(
    graph: nx.MultiDiGraph,
    cfg: GenerationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    nodes = _sample_nodes(graph, cfg.orders_rows, rng)
    now = pd.Timestamp("2025-01-31T08:00:00Z")

    records: list[dict[str, float | int | str]] = []
    for i, node_id in enumerate(nodes):
        lat, lon = _node_xy(graph, int(node_id))
        created_ts = now + pd.to_timedelta(int(rng.integers(0, 240)), unit="m")
        ready_min = int(rng.integers(0, 120))
        due_min = ready_min + int(rng.integers(30, 180))
        records.append(
            {
                "order_id": i + 1,
                "node_id": int(node_id),
                "lat": lat,
                "lon": lon,
                "created_ts": created_ts.isoformat(),
                "demand": int(rng.integers(1, 5)),
                "service_time_min": int(rng.integers(5, 15)),
                "ready_time_min": ready_min,
                "due_time_min": due_min,
            }
        )
    return pd.DataFrame.from_records(records)


def _validate_dataframe(df: pd.DataFrame, name: str, min_rows: int) -> None:
    if len(df) < min_rows:
        raise ValueError(f"{name} has too few rows: {len(df)} < {min_rows}")
    if df.isna().any().any():
        raise ValueError(f"{name} contains NaN values.")
    LOGGER.info("Validated %s shape=%s columns=%s", name, df.shape, list(df.columns))
    LOGGER.info("%s dtypes=%s", name, df.dtypes.astype(str).to_dict())


def parse_args() -> GenerationConfig:
    """Parse CLI args for dataset generation."""
    parser = argparse.ArgumentParser(description="generate synthetic route optimization datasets")
    parser.add_argument("--trips-rows", type=int, default=1200)
    parser.add_argument("--orders-rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    return GenerationConfig(
        trips_rows=args.trips_rows,
        orders_rows=args.orders_rows,
        seed=args.seed,
        raw_dir=args.raw_dir,
    )


def main() -> None:
    """entry point for synthetic trips and orders generation"""
    _configure_logging()
    cfg = parse_args()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    graph = _load_manhattan_graph()
    trips_df = _generate_trips(graph=graph, cfg=cfg, rng=rng)
    orders_df = _generate_orders(graph=graph, cfg=cfg, rng=rng)

    _validate_dataframe(trips_df, name="trips_df", min_rows=500)
    _validate_dataframe(orders_df, name="orders_df", min_rows=50)

    trips_path = cfg.raw_dir / "trips.csv"
    orders_path = cfg.raw_dir / "orders.csv"
    trips_df.to_csv(trips_path, index=False)
    orders_df.to_csv(orders_path, index=False)
    LOGGER.info("Saved trips to %s", trips_path.as_posix())
    LOGGER.info("Saved orders to %s", orders_path.as_posix())
    LOGGER.info("Generation completed successfully")


if __name__ == "__main__":
    main()
