"""NYC TLC real + synthetic Manhattan trips and orders datasets"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

LOGGER = logging.getLogger(__name__)

TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
TLC_CACHE_NAME = "tlc_2023_01.parquet"

# Manhattan bounding box
MH_LAT_MIN, MH_LAT_MAX = 40.70, 40.88
MH_LON_MIN, MH_LON_MAX = -74.02, -73.90

# Trip filters
MIN_TRAVEL_SEC, MAX_TRAVEL_SEC = 60, 7200
MIN_DISTANCE_M, MAX_DISTANCE_M = 100, 50_000

MILES_TO_METERS = 1609.34
REAL_SAMPLE_SIZE = 50_000
REAL_RANDOM_STATE = 42


@dataclass(frozen=True)
class GenerationConfig:
    """config for data generation"""

    source: str  # "real" | "synthetic"
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
    LOGGER.info("loaded manhattan graph: nodes=%s edges=%s", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _nearest_nodes_batch(
    graph: nx.MultiDiGraph,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Return nearest OSMnx node ids for arrays of lat/lon."""
    return np.array(ox.nearest_nodes(graph, X=lons, Y=lats))


# Real data: NYC TLC download + transform

def _download_tlc(dest: Path) -> None:
    """download TLC parquet with tqdm progress bar. Skips if cached"""
    if dest.exists():
        LOGGER.info("cached TLC file found at %s — skipping download", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("downloading NYC TLC data from %s …", TLC_URL)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    req = urllib.request.Request(TLC_URL, headers={"User-Agent": "routeOptimizer/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk_size = 1 << 20  # 1 MiB

        if tqdm is not None and total > 0:
            bar = tqdm(total=total, unit="B", unit_scale=True, desc="TLC download")
        else:
            bar = None

        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))

        if bar is not None:
            bar.close()

    LOGGER.info("saved TLC parquet to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def _generate_real_trips(
    graph: nx.MultiDiGraph,
    cfg: GenerationConfig,
) -> pd.DataFrame:
    """load NYC TLC parquet, filter, map to trips.csv schema"""
    cache_path = cfg.raw_dir / TLC_CACHE_NAME
    _download_tlc(cache_path)

    LOGGER.info("reading parquet …")
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(cache_path)
    available_cols = parquet_file.schema.names

    expected_gps = {"pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"}
    has_gps = expected_gps.issubset(available_cols)

    if has_gps:
        read_cols = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "trip_distance",
        ]
        df = pd.read_parquet(cache_path, columns=read_cols)
    else:
        LOGGER.info("lat/lon columns not found — trying newer TLC schema with LocationID")
        read_cols = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
        ]
        df = pd.read_parquet(cache_path, columns=read_cols)
        _apply_zone_centroids(df)

    # rename to standard columns
    rename_map = {
        "pickup_latitude": "pickup_lat",
        "pickup_longitude": "pickup_lon",
        "dropoff_latitude": "dropoff_lat",
        "dropoff_longitude": "dropoff_lon",
    }
    df = df.rename(columns=rename_map)

    # compute derived columns
    df["distance_m"] = df["trip_distance"] * MILES_TO_METERS
    df["travel_time_sec"] = (
        pd.to_datetime(df["tpep_dropoff_datetime"]) - pd.to_datetime(df["tpep_pickup_datetime"])
    ).dt.total_seconds()

    # filters
    before = len(df)

    # travel time bounds
    df = df[(df["travel_time_sec"] >= MIN_TRAVEL_SEC) & (df["travel_time_sec"] <= MAX_TRAVEL_SEC)]
    # distance bounds
    df = df[(df["distance_m"] >= MIN_DISTANCE_M) & (df["distance_m"] <= MAX_DISTANCE_M)]
    # manhattan bbox — pickup AND dropoff inside
    df = df[
        (df["pickup_lat"] >= MH_LAT_MIN) & (df["pickup_lat"] <= MH_LAT_MAX)
        & (df["pickup_lon"] >= MH_LON_MIN) & (df["pickup_lon"] <= MH_LON_MAX)
        & (df["dropoff_lat"] >= MH_LAT_MIN) & (df["dropoff_lat"] <= MH_LAT_MAX)
        & (df["dropoff_lon"] >= MH_LON_MIN) & (df["dropoff_lon"] <= MH_LON_MAX)
    ]
    # drop NaN coordinates
    df = df.dropna(subset=["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"])

    LOGGER.info("filtered: %s → %s rows", before, len(df))

    if len(df) < REAL_SAMPLE_SIZE:
        LOGGER.warning("only %s rows after filter (requested %s) — using all", len(df), REAL_SAMPLE_SIZE)
    else:
        df = df.sample(n=REAL_SAMPLE_SIZE, random_state=REAL_RANDOM_STATE)

    df = df.reset_index(drop=True)

    # nearest osmnx nodes
    LOGGER.info("snapping %s coordinates to OSMnx nodes", len(df))
    pickup_nodes = _nearest_nodes_batch(graph, df["pickup_lat"].values, df["pickup_lon"].values)
    dropoff_nodes = _nearest_nodes_batch(graph, df["dropoff_lat"].values, df["dropoff_lon"].values)

    # build output DataFrame - same schema as synthetic trips.csv
    trips = pd.DataFrame({
        "trip_id": df.index.astype(str),
        "pickup_node": pickup_nodes.astype(int),
        "dropoff_node": dropoff_nodes.astype(int),
        "pickup_lat": df["pickup_lat"].values,
        "pickup_lon": df["pickup_lon"].values,
        "dropoff_lat": df["dropoff_lat"].values,
        "dropoff_lon": df["dropoff_lon"].values,
        "pickup_ts": pd.to_datetime(df["tpep_pickup_datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "distance_m": df["distance_m"].round(2).values,
        "travel_time_sec": df["travel_time_sec"].round(2).values,
    })

    return trips



_ZONE_CENTROID_URL = (
    "https://data.cityofnewyork.us/api/views/755u-8jsi/rows.csv?accessType=DOWNLOAD"
)


def _apply_zone_centroids(df: pd.DataFrame) -> None:
    """add pickup/dropoff lat/lon from taxi zone centroids in-place"""
    import os
    import urllib.request
    from pathlib import Path

    zip_path = Path("data/raw/taxi_zones.zip")
    if not zip_path.exists():
        LOGGER.info("downloading taxi zones zip locally...")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "routeOptimizer/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as fh:
            fh.write(resp.read())

    extract_dir = Path("data/raw/taxi_zones")
    if not extract_dir.exists():
        LOGGER.info("extracting taxi zones zip...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    try:
        import geopandas as gpd
        shapefile_path = next(extract_dir.glob("**/*.shp"))
        zones = gpd.read_file(shapefile_path.resolve().as_posix())
        zones = zones.to_crs(epsg=4326)
    except Exception as exc:
        LOGGER.error("failed to read taxi zones shapefile: %s", exc)
        raise exc

    if "the_geom" in zones.columns or "geometry" in zones.columns:
        if hasattr(zones, "geometry"):
            centroids = zones.geometry.centroid
            zones["centroid_lat"] = centroids.y
            zones["centroid_lon"] = centroids.x
        else:
            raise ValueError("unable to compute centroids from taxi zone data")

    zone_lookup = zones.set_index("LocationID")[["centroid_lat", "centroid_lon"]]

    for prefix, col in [("pickup", "PULocationID"), ("dropoff", "DOLocationID")]:
        mapped = df[col].map(zone_lookup["centroid_lat"])
        df[f"{prefix}_latitude"] = mapped
        mapped_lon = df[col].map(zone_lookup["centroid_lon"])
        df[f"{prefix}_longitude"] = mapped_lon


# synthetic data (original - kept intact)

def _sample_nodes(graph: nx.MultiDiGraph, n: int, rng: np.random.Generator) -> np.ndarray:
    node_ids = np.array(list(graph.nodes))
    return rng.choice(node_ids, size=n, replace=True)


def _node_xy(graph: nx.MultiDiGraph, node_id: int) -> tuple[float, float]:
    data = graph.nodes[node_id]
    return float(data["y"]), float(data["x"])  # lat, lon


def _time_multiplier(hour: int) -> float:
    """synthetic traffic multiplier by time of day"""
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
    LOGGER.info("validated %s shape=%s columns=%s", name, df.shape, list(df.columns))
    LOGGER.info("%s dtypes=%s", name, df.dtypes.astype(str).to_dict())


def parse_args() -> GenerationConfig:
    """parse CLI args for dataset generation"""
    parser = argparse.ArgumentParser(description="generate route optimization datasets")
    parser.add_argument("--source", choices=["real", "synthetic"], default="real",
                        help="data source: 'real' (NYC TLC) or 'synthetic' (default: real)")
    parser.add_argument("--trips-rows", type=int, default=1200,
                        help="number of synthetic trips (ignored when --source=real)")
    parser.add_argument("--orders-rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    return GenerationConfig(
        source=args.source,
        trips_rows=args.trips_rows,
        orders_rows=args.orders_rows,
        seed=args.seed,
        raw_dir=args.raw_dir,
    )


def main() -> None:
    """entry point for trips and orders generation"""
    _configure_logging()
    cfg = parse_args()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    graph = _load_manhattan_graph()

    if cfg.source == "real":
        LOGGER.info("source=real — loading NYC TLC 2023-01 data")
        trips_df = _generate_real_trips(graph=graph, cfg=cfg)
    else:
        LOGGER.info("source=synthetic — generating synthetic trips")
        trips_df = _generate_trips(graph=graph, cfg=cfg, rng=rng)

    orders_df = _generate_orders(graph=graph, cfg=cfg, rng=rng)

    min_trips = 500 if cfg.source == "synthetic" else 1000
    _validate_dataframe(trips_df, name="trips_df", min_rows=min_trips)
    _validate_dataframe(orders_df, name="orders_df", min_rows=50)

    trips_path = cfg.raw_dir / "trips.csv"
    orders_path = cfg.raw_dir / "orders.csv"
    trips_df.to_csv(trips_path, index=False)
    orders_df.to_csv(orders_path, index=False)
    LOGGER.info("saved trips to %s (%s rows)", trips_path.as_posix(), len(trips_df))
    LOGGER.info("saved orders to %s (%s rows)", orders_path.as_posix(), len(orders_df))
    LOGGER.info("generation completed successfully (source=%s)", cfg.source)


if __name__ == "__main__":
    main()
