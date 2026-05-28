"""streamlit UI for route optimization demo"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import folium
import osmnx as ox
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html


def call_optimize_api(
    api_url: str,
    orders: list[dict[str, Any]],
    baseline_type: str,
    num_vehicles: int,
) -> dict[str, Any]:
    """call FastAPI optimize endpoint via JSON payload"""
    payload = {
        "orders": orders,
        "baseline_type": baseline_type,
        "num_vehicles": num_vehicles,
    }
    req = urllib.request.Request(
        url=f"{api_url.rstrip('/')}/optimize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8")
        raise RuntimeError(f"API error: {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API connection error: {exc}") from exc


@st.cache_resource(show_spinner=False)
def _load_drive_graph(north: float, south: float, east: float, west: float):
    """Load and cache a drivable OSM graph for map rendering."""
    return ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type="drive")


def _build_road_polyline(
    graph,
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[list[float]]:
    """Build road-following polyline between two coordinates."""
    start_node = ox.nearest_nodes(graph, X=start[1], Y=start[0])
    end_node = ox.nearest_nodes(graph, X=end[1], Y=end[0])
    node_path = ox.shortest_path(graph, start_node, end_node, weight="length")
    if not node_path:
        return [list(start), list(end)]
    return [[float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])] for n in node_path]


def render_map(orders_df: pd.DataFrame, routes: list[dict[str, Any]], use_road_paths: bool) -> None:
    """Render folium route map."""
    center = [orders_df["lat"].mean(), orders_df["lon"].mean()]
    route_map = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    order_lookup = {int(row["order_id"]): row for _, row in orders_df.iterrows() if "order_id" in row}
    if 0 not in order_lookup and len(orders_df) > 0:
        first = orders_df.iloc[0]
        order_lookup[0] = first

    for _, row in orders_df.iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color="#666666",
            fill=True,
            fill_opacity=0.6,
            tooltip=f"order {int(row.get('order_id', -1))}",
        ).add_to(route_map)

    graph = None
    if use_road_paths and len(orders_df) > 1:
        margin = 0.02
        north = float(orders_df["lat"].max() + margin)
        south = float(orders_df["lat"].min() - margin)
        east = float(orders_df["lon"].max() + margin)
        west = float(orders_df["lon"].min() - margin)
        try:
            graph = _load_drive_graph(north=north, south=south, east=east, west=west)
        except Exception as exc:
            st.warning(f"Road graph unavailable, using straight lines: {exc}")

    palette = ["red", "blue", "green", "purple", "orange", "black"]
    for route in routes:
        color = palette[route["vehicle_id"] % len(palette)]
        points: list[list[float]] = []
        for stop in route["stops"]:
            oid = int(stop["order_id"])
            if oid in order_lookup:
                row = order_lookup[oid]
                points.append([float(row["lat"]), float(row["lon"])])
                folium.CircleMarker(
                    location=[float(row["lat"]), float(row["lon"])],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.9,
                    tooltip=f"order {oid}, arrival={stop['arrival_sec']}s",
                ).add_to(route_map)
        if len(points) > 1:
            if graph is None:
                folium.PolyLine(points, color=color, weight=3, opacity=0.9).add_to(route_map)
            else:
                for idx in range(len(points) - 1):
                    a = (points[idx][0], points[idx][1])
                    b = (points[idx + 1][0], points[idx + 1][1])
                    try:
                        road_segment = _build_road_polyline(graph, start=a, end=b)
                    except Exception:
                        road_segment = [list(a), list(b)]
                    folium.PolyLine(road_segment, color=color, weight=3, opacity=0.9).add_to(route_map)

    html(route_map._repr_html_(), height=520)


def main() -> None:
    """streamlit entrypoint"""
    st.set_page_config(page_title="routeOptimizer", layout="wide")
    st.title("routeOptimizer Demo")

    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("API URL", value="http://127.0.0.1:8000")
        baseline_type = st.selectbox("Baseline type", ["ml", "constant", "median"], index=0)
        num_vehicles = st.slider("Number of vehicles", min_value=1, max_value=10, value=3)
        use_road_paths = st.checkbox("Road-following paths (OSM)", value=True)

    uploaded = st.file_uploader("Upload orders CSV", type=["csv"])
    if uploaded is None:
        st.info("upload a CSV file with columns: order_id, lat, lon, ready_time_min, due_time_min, demand")
        return

    orders_df = pd.read_csv(uploaded)
    st.subheader("input orders")
    st.dataframe(orders_df.head(50), use_container_width=True)

    if st.button("optimize routes", type="primary"):
        with st.spinner("optimizing routes..."):
            try:
                result = call_optimize_api(
                    api_url=api_url,
                    orders=orders_df.to_dict(orient="records"),
                    baseline_type=baseline_type,
                    num_vehicles=num_vehicles,
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.subheader("Route Map")
                render_map(orders_df=orders_df, routes=[], use_road_paths=use_road_paths)
                return

        st.success(f"Status: {result['status']}")
        metrics = result.get("metrics", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("total time (sec)", metrics.get("total_time_sec", 0))
        col2.metric("served orders", metrics.get("served_orders", 0))
        col3.metric("routes", metrics.get("num_routes", 0))

        routes = result.get("routes", [])
        st.subheader("route table")
        rows: list[dict[str, Any]] = []
        for route in routes:
            rows.append(
                {
                    "vehicle_id": route["vehicle_id"],
                    "route_time_sec": route["route_time_sec"],
                    "route_load": route["route_load"],
                    "stops_count": len(route["stops"]),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("metrics chart")
        if rows:
            chart_df = pd.DataFrame(rows)[["vehicle_id", "route_time_sec", "route_load"]].set_index("vehicle_id")
            st.bar_chart(chart_df)

        st.subheader("Route Map")
        render_map(orders_df=orders_df, routes=routes, use_road_paths=use_road_paths)


if __name__ == "__main__":
    main()
