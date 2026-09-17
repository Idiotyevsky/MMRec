"""Minimal FastAPI wrapper around :class:`Recommender`.

    python -m src.serving.app --run-dir results/runs/<run_id> --port 8000

    curl -s localhost:8000/recommend -H 'content-type: application/json' \
         -d '{"history": [12, 45, 91, 102], "top_k": 5}'

Serving is deliberately last in the project's priority order; it reuses exactly
the same code path as the offline retrieval demo, so it cannot disagree with the
evaluated model.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from ..serving.recommender import Recommender

_REC: Recommender | None = None
_STATS: dict[str, Any] = {"requests": 0, "total_ms": 0.0}


def get_recommender() -> Recommender:
    if _REC is None:
        raise RuntimeError("recommender not loaded; call load(run_dir) first")
    return _REC


def load(run_dir: str, device: str = "cpu") -> Recommender:
    global _REC
    _REC = Recommender.from_run(run_dir, device=device)
    _REC.warm_start()
    return _REC


def create_app(run_dir: str, device: str = "cpu"):
    from fastapi import FastAPI
    from pydantic import BaseModel

    class RecommendRequest(BaseModel):
        history: list[int]
        top_k: int = 20

    class RecommendResponse(BaseModel):
        items: list[dict]
        latency_ms: float

    app = FastAPI(title="MMRec", version="0.1.0")

    @app.on_event("startup")
    def _startup() -> None:  # pragma: no cover - requires fastapi
        load(run_dir, device)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": get_recommender().describe()}

    @app.get("/stats")
    def stats() -> dict:
        s = dict(_STATS)
        s["avg_latency_ms"] = round(s["total_ms"] / max(s["requests"], 1), 3)
        return s

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        t0 = time.perf_counter()
        items = get_recommender().recommend(req.history, top_k=req.top_k)
        dt = (time.perf_counter() - t0) * 1000
        _STATS["requests"] += 1
        _STATS["total_ms"] += dt
        return RecommendResponse(items=items, latency_ms=round(dt, 3))

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.run_dir, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
