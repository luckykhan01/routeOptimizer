"""streamlit UI for route optimization demo"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Any

import folium
import osmnx as ox
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html


def _format_duration(seconds: int) -> str:
    """format seconds into HH:MM:SS duration string"""
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _haversine_km(p1: list[float], p2: list[float]) -> float:
    """return the great-circle distance in km between two [lat, lon] points"""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _collect_arrival_by_order(routes: list[dict[str, Any]]) -> dict[int, int]:
    """collect predicted arrival time (sec) for each order_id"""
    arrival: dict[int, int] = {}
    for route in routes:
        for stop in route.get("stops", []):
            order_id = int(stop.get("order_id", -1))
            if order_id <= 0:
                continue
            arrival_sec = int(stop.get("arrival_sec", 0))
            if order_id not in arrival or arrival_sec < arrival[order_id]:
                arrival[order_id] = arrival_sec
    return arrival


def validate_orders_df(orders_df: pd.DataFrame) -> tuple[bool, str]:
    """validate uploaded orders schema and data quality"""
    required_columns = {
        "order_id",
        "lat",
        "lon",
        "ready_time_min",
        "due_time_min",
        "demand",
        "service_time_min",
    }
    missing = sorted(required_columns.difference(set(orders_df.columns)))
    if missing:
        return False, f"missing required columns: {missing}"
    if len(orders_df) == 0:
        return False, "orders file is empty"
    if orders_df[list(required_columns)].isna().any().any():
        return False, "orders contains NaN values in required columns"

    try:
        for col in ["lat", "lon", "ready_time_min", "due_time_min", "demand", "service_time_min"]:
            orders_df[col] = pd.to_numeric(orders_df[col], errors="raise")
    except ValueError:
        return False, "numeric columns contain invalid values"

    if ((orders_df["lat"] < -90) | (orders_df["lat"] > 90)).any():
        return False, "latitude values are out of range [-90, 90]"
    if ((orders_df["lon"] < -180) | (orders_df["lon"] > 180)).any():
        return False, "longitude values are out of range [-180, 180]"
    if (orders_df["due_time_min"] < orders_df["ready_time_min"]).any():
        return False, "due_time_min must be >= ready_time_min for all rows."
    if (orders_df["demand"] < 0).any() or (orders_df["service_time_min"] < 0).any():
        return False, "demand and service_time_min must be non-negative"
    return True, "ok"


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


def call_compare_api(
    api_url: str,
    orders: list[dict[str, Any]],
    num_vehicles: int,
) -> dict[str, Any]:
    """call FastAPI compare endpoint"""
    payload = {
        "orders": orders,
        "num_vehicles": num_vehicles,
    }
    req = urllib.request.Request(
        url=f"{api_url.rstrip('/')}/compare",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
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
    """render folium route map"""
    center = [orders_df["lat"].mean(), orders_df["lon"].mean()]
    depot_location = center  # geocenter of all orders serves as depot
    route_map = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
    arrival_by_order = _collect_arrival_by_order(routes)

    order_lookup = {int(row["order_id"]): row for _, row in orders_df.iterrows() if "order_id" in row}
    if 0 not in order_lookup and len(orders_df) > 0:
        first = orders_df.iloc[0]
        order_lookup[0] = first

    # collect order_ids that are assigned to a route (will be drawn as numbered stops)
    routed_order_ids: set[int] = set()
    for route in routes:
        for stop in route.get("stops", []):
            routed_order_ids.add(int(stop["order_id"]))

    # draw unrouted orders as plain grey CircleMarkers
    for _, row in orders_df.iterrows():
        oid = int(row.get("order_id", -1))
        if oid in routed_order_ids:
            continue
        eta_label = "ETA: pending"
        if oid in arrival_by_order:
            eta_label = f"ETA: {_format_duration(arrival_by_order[oid])}"
        address_text = str(row["address"]) if "address" in orders_df.columns else f"order {oid}"
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color="#666666",
            fill=True,
            fill_opacity=0.6,
            tooltip=f"{address_text} | {eta_label}",
            popup=f"{address_text}<br>{eta_label}",
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
            st.warning(f"road graph unavailable, using straight lines: {exc}")

    # hex palette for routes
    palette = ["#e63946", "#2a9d8f", "#e9c46a", "#a8dadc", "#f4a261", "#264653"]
    # for legend
    legend_rows: list[dict[str, Any]] = []

    for route in routes:
        vehicle_id: int = route["vehicle_id"]
        color = palette[vehicle_id % len(palette)]
        stops = route["stops"]
        points: list[list[float]] = []

        for stop_idx, stop in enumerate(stops):
            oid = int(stop["order_id"])
            if oid not in order_lookup:
                continue
            row = order_lookup[oid]
            lat, lon = float(row["lat"]), float(row["lon"])
            points.append([lat, lon])
            stop_num = stop_idx + 1
            eta_str = _format_duration(int(stop["arrival_sec"]))
            # Build ETA as HH:MM
            eta_hhmm = ":".join(eta_str.split(":")[:2])

            # Numbered DivIcon marker
            icon_html = (
                f'<div style="'
                f'width:22px;height:22px;'
                f'background:#ffffff;'
                f'border:2.5px solid {color};'
                f'border-radius:50%;'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-weight:700;font-size:11px;color:#1a1a1a;'
                f'box-shadow:0 1px 3px rgba(0,0,0,.4);'
                f'line-height:1;'
                f'">{stop_num}</div>'
            )
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(22, 22),
                    icon_anchor=(11, 11),
                ),
                tooltip=f"остановка {stop_num} — order_id: {oid}, ETA: {eta_hhmm}",
                popup=f"order {oid}<br>ETA: {eta_str}",
            ).add_to(route_map)

        # dashed line: depot - first stop
        if points:
            folium.PolyLine(
                [depot_location, points[0]],
                color="#888888",
                weight=1.5,
                opacity=0.7,
                dash_array="6 4",
            ).add_to(route_map)

        # route polylines between stops
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

        # collect legend data
        route_time_sec: int = int(route.get("route_time_sec", 0))
        route_load: float = float(route.get("route_load", 0))
        # rough distance: sum haversine between consecutive points (km)
        dist_km = sum(_haversine_km(points[i], points[i + 1]) for i in range(len(points) - 1)) if len(points) > 1 else 0.0
        hours, mins = divmod(route_time_sec // 60, 60)
        legend_rows.append({
            "vehicle_id": vehicle_id,
            "color": color,
            "stops": len(points),
            "time_str": f"{hours}:{mins:02d}",
            "dist_km": round(dist_km, 1),
            "time_sec": route_time_sec,
        })

    # Depot DivIcon marker
    depot_icon_html = (
        '<div style="'
        'font-size:18px;color:#1a1a1a;'
        'text-shadow:0 0 3px #fff,0 0 3px #fff;'
        'line-height:1;'
        '">&#9632;</div>'
    )
    folium.Marker(
        location=depot_location,
        icon=folium.DivIcon(
            html=depot_icon_html,
            icon_size=(20, 20),
            icon_anchor=(10, 10),
        ),
        popup="Депо",
        tooltip="Депо",
    ).add_to(route_map)

    # legend
    if legend_rows:
        total_stops = sum(r["stops"] for r in legend_rows)
        total_sec = sum(r["time_sec"] for r in legend_rows)
        total_km = sum(r["dist_km"] for r in legend_rows)
        t_h, t_m = divmod(total_sec // 60, 60)
        total_time_str = f"{t_h}:{t_m:02d}"

        rows_html = ""
        for r in legend_rows:
            rows_html += (
                f'<tr>'
                f'<td><span style="color:{r["color"]};font-size:16px;">●</span></td>'
                f'<td>Курьер&nbsp;{r["vehicle_id"]}</td>'
                f'<td style="text-align:right;">{r["stops"]}</td>'
                f'<td style="text-align:right;">{r["time_str"]}</td>'
                f'<td style="text-align:right;">{r["dist_km"]}&nbsp;км</td>'
                f'</tr>'
            )
        rows_html += (
            f'<tr style="border-top:1px solid #444;font-weight:700;">'
            f'<td></td>'
            f'<td>Все</td>'
            f'<td style="text-align:right;">{total_stops}</td>'
            f'<td style="text-align:right;">{total_time_str}</td>'
            f'<td style="text-align:right;">{round(total_km, 1)}&nbsp;км</td>'
            f'</tr>'
        )

        legend_html = (
            f'<div style="'
            f'position:fixed;bottom:16px;left:16px;z-index:9999;'
            f'background:#1e1e1e;color:#ffffff;'
            f'font-family:monospace;font-size:12px;'
            f'padding:10px 14px;border-radius:6px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,.6);'
            f'line-height:1.6;'
            f'">'
            f'<table style="border-collapse:collapse;border-spacing:0 2px;">'
            f'<thead><tr style="color:#aaa;font-size:11px;">'
            f'<th></th><th style="text-align:left;">Курьер</th>'
            f'<th style="text-align:right;padding-left:10px;">Ост.</th>'
            f'<th style="text-align:right;padding-left:10px;">Время</th>'
            f'<th style="text-align:right;padding-left:10px;">Расст.</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>'
        )
        route_map.get_root().html.add_child(folium.Element(legend_html))

    html(route_map._repr_html_(), height=520)


def main() -> None:
    """streamlit entrypoint"""
    st.set_page_config(page_title="routeOptimizer", layout="wide")
    st.title("routeOptimizer Demo")

    with st.sidebar:
        st.header("Settings")
        default_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
        api_url = st.text_input("API URL", value=default_url)
        baseline_type = st.selectbox("Baseline type", ["ml", "constant", "median"], index=0)
        num_vehicles = st.slider("Number of vehicles", min_value=1, max_value=10, value=3)
        use_road_paths = st.checkbox("Road-following paths (OSM)", value=True)

    uploaded = st.file_uploader("Upload orders CSV", type=["csv"])
    if uploaded is not None:
        orders_df = pd.read_csv(uploaded)
    else:
        local_path = "data/raw/orders.csv"
        if os.path.exists(local_path):
            st.info(f"using local orders from '{local_path}'. upload another csv to override.")
            orders_df = pd.read_csv(local_path)
        else:
            st.info("upload a CSV file with columns: order_id, lat, lon, ready_time_min, due_time_min, demand")
            return
    is_valid, message = validate_orders_df(orders_df)
    st.subheader("input orders")
    st.dataframe(orders_df.head(50), use_container_width=True)
    if not is_valid:
        st.error(f"validation failed: {message}")
        st.subheader("route map")
        render_map(orders_df=orders_df, routes=[], use_road_paths=use_road_paths)
        return

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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("total time (sec)", metrics.get("total_time_sec", 0))
        col2.metric("served orders", f"{metrics.get('served_orders', 0)} / {metrics.get('total_orders', 0)}")
        col3.metric("routes", metrics.get("num_routes", 0))
        col4.metric("dropped orders", metrics.get("dropped_orders", 0))
        
        dropped = metrics.get("dropped_order_ids", [])
        if dropped:
            st.warning(f"Could not fit {len(dropped)} orders into time windows: {dropped}")

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

        arrival_by_order = _collect_arrival_by_order(routes)
        if arrival_by_order:
            st.subheader("ETA for specific address")
            options_df = orders_df[orders_df["order_id"].isin(arrival_by_order.keys())].copy()
            if "address" in options_df.columns:
                options_df["label"] = (
                    options_df["address"].astype(str) + " (order " + options_df["order_id"].astype(int).astype(str) + ")"
                )
            else:
                options_df["label"] = "order " + options_df["order_id"].astype(int).astype(str)
            labels = options_df["label"].tolist()
            selected_label = st.selectbox("Choose destination", labels)
            selected_row = options_df[options_df["label"] == selected_label].iloc[0]
            selected_order_id = int(selected_row["order_id"])
            eta_sec = arrival_by_order[selected_order_id]
            st.info(
                f"Predicted arrival for {selected_label}: {_format_duration(eta_sec)} "
                f"({eta_sec // 60} min {eta_sec % 60} sec from route start)."
            )

    st.markdown("---")
    if st.button("compare baselines"):
        with st.spinner("running comparison (this takes longer)..."):
            try:
                comp = call_compare_api(
                    api_url=api_url,
                    orders=orders_df.to_dict(orient="records"),
                    num_vehicles=num_vehicles,
                )
            except RuntimeError as exc:
                st.error(str(exc))
                return
        
        st.subheader("baseline comparison")
        c_rows: list[dict[str, Any]] = []
        for btype, cres in comp.items():
            cm = cres.get("metrics", {})
            c_rows.append({
                "baseline": btype,
                "status": cres["status"],
                "total_time_sec": cm.get("total_time_sec", 0),
                "served": cm.get("served_orders", 0),
                "dropped": cm.get("dropped_orders", 0),
            })
        
        cdf = pd.DataFrame(c_rows)
        st.dataframe(cdf, use_container_width=True)
        if len(cdf) > 0 and "total_time_sec" in cdf.columns:
            st.bar_chart(cdf.set_index("baseline")["total_time_sec"])


if __name__ == "__main__":
    main()
