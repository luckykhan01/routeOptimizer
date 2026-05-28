from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.solver import build_time_matrix, solve_vrptw


class OptimizePayload(BaseModel):
    """Payload for JSON optimization request."""

    orders: list[dict[str, Any]] = Field(default_factory=list)
    baseline_type: str = "ml"
    num_vehicles: int = 3


def _load_config() -> dict[str, Any]:
    cfg_path = PROJECT_ROOT / "configs" / "train.yaml"
    with cfg_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _orders_from_upload(upload: UploadFile) -> pd.DataFrame:
    raw = upload.file.read()
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(raw))
    if suffix == ".json":
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, dict) and "orders" in parsed:
            parsed = parsed["orders"]
        return pd.DataFrame(parsed)
    raise HTTPException(status_code=400, detail="Only .csv and .json are supported.")


app = FastAPI(title="routeOptimizer API", version="0.1.0")
APP_STATE: dict[str, Any] = {}


@app.on_event("startup")
def startup() -> None:
    """load runtime configuration and model path"""
    cfg = _load_config()
    model_path = PROJECT_ROOT / cfg["paths"]["model_path"]
    APP_STATE["cfg"] = cfg
    APP_STATE["model_path"] = model_path


@app.get("/health")
def health() -> dict[str, str]:
    """health check endpoint"""
    return {"status": "ok"}


@app.post("/optimize")
async def optimize(
    request: Request,
    baseline_type: str = Form(default="ml"),
    num_vehicles: int = Form(default=3),
    orders_file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """optimize delivery routes from JSON payload or uploaded CSV/JSON file"""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        payload = OptimizePayload(**body)
        orders_df = pd.DataFrame(payload.orders)
        baseline_type = payload.baseline_type
        num_vehicles = payload.num_vehicles
    elif orders_file is not None:
        orders_df = _orders_from_upload(orders_file)
    else:
        raise HTTPException(status_code=400, detail="provide orders in JSON body or upload orders_file")

    model_path: Path = APP_STATE["model_path"]
    if baseline_type == "ml" and not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"model not found at {model_path.as_posix()}. train model first",
        )

    try:
        time_matrix = build_time_matrix(
            orders_df=orders_df,
            model_path=model_path,
            baseline_type=baseline_type,
        )
        result = solve_vrptw(time_matrix=time_matrix, orders_df=orders_df, num_vehicles=num_vehicles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": result["status"],
        "baseline_type": baseline_type,
        "num_vehicles": num_vehicles,
        "routes": result.get("routes", []),
        "metrics": result.get("metrics", {}),
    }
