"""ShortRec serving API.

    python -m src.serving.app                      # http://127.0.0.1:8000
    python -m src.serving.app --device cuda --port 8001

Two-stage recommendation over HTTP:

    GET /system                      dataset / models / recall channels
    GET /models                      rankers + offline metrics (from results/tables)
    GET /users/{u}                   history with cold / bucket metadata
    GET /users/{u}/recall            candidate generation trace
    GET /users/{u}/recommend         the served top-K
    GET /users/{u}/inspect           full trace for the Recommendation Inspector
    GET /cold/summary                cold-start experiment results
    GET /cold/items                  sample cold items with content availability
    GET /cold/items/{item_id}        one cold item
    POST /recommend                  legacy history-based endpoint

Interactive docs: http://127.0.0.1:8000/docs
"""

# NOTE: no `from __future__ import annotations` here on purpose.  FastAPI resolves
# endpoint parameter annotations at runtime, and with postponed evaluation the
# locally-imported Pydantic models would become unresolvable strings, silently
# turning a JSON body into a required *query* parameter.
import argparse
import os
import time
from typing import Any

from .service import ShortRecService


def configure_threads(n: int | None = None) -> int:
    """Cap intra-op threads before the model is loaded.

    PyTorch defaults to one thread per core.  Inside FastAPI a sync endpoint runs
    in a threadpool, so every request would spawn ~20 OpenMP threads on a machine
    that is already busy — measured at ~190 ms per request versus ~9 ms with a
    single thread.  Serving many small requests wants few threads, not many.
    """
    import torch

    n = int(n if n is not None else os.environ.get("SHORTREC_TORCH_THREADS", "1"))
    n = max(1, n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass  # already initialised; harmless
    return n

_SERVICE: ShortRecService | None = None
_STATS: dict[str, Any] = {"requests": 0, "total_ms": 0.0}


def get_service() -> ShortRecService:
    """Module-level accessor, used by scripts and the CLI entry point."""
    if _SERVICE is None:
        raise RuntimeError("service not loaded")
    return _SERVICE


def load(**kwargs) -> ShortRecService:
    global _SERVICE
    _SERVICE = ShortRecService(**kwargs)
    return _SERVICE


def create_app(service: ShortRecService | None = None, **service_kwargs):
    configure_threads()
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware

    from .schemas import (
        ColdItemOut,
        ColdSummaryResponse,
        FeedResponse,
        LegacyRecommendRequest,
        LegacyRecommendResponse,
        MediaManifestResponse,
        RecallResponse,
        RecommendResponse,
        SystemResponse,
        UserResponse,
    )

    app = FastAPI(
        title="ShortRec",
        version="0.2.0",
        description="Multimodal two-stage recommendation for short-video feeds",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local demo only; never deploy this as-is
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # The service is held per app instance, not in a module global: two apps in
    # one process (tests, or a base + cold deployment) must not share it.
    _holder: dict[str, ShortRecService | None] = {"service": service}

    def svc() -> ShortRecService:
        if _holder["service"] is None:
            _holder["service"] = load(**service_kwargs)
        return _holder["service"]

    @app.on_event("startup")
    def _startup() -> None:  # pragma: no cover - requires fastapi
        svc()

    def _timed(fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        _STATS["requests"] += 1
        _STATS["total_ms"] += (time.perf_counter() - t0) * 1000
        return out

    # ------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "ShortRec"}

    @app.get("/stats")
    def stats() -> dict:
        s = dict(_STATS)
        s["avg_latency_ms"] = round(s["total_ms"] / max(s["requests"], 1), 2)
        return s

    @app.get("/system", response_model=SystemResponse)
    def system() -> dict:
        return _timed(svc().system)

    @app.get("/models")
    def models() -> dict:
        return {"models": svc().model_infos()}

    @app.get("/evaluation")
    def evaluation() -> dict:
        """Offline recall / pipeline / latency artifacts for the System page."""
        return svc().evaluation()

    # ------------------------------------------------------------------
    # demo media: real MicroLens videos resolved by verified item id mapping
    # ------------------------------------------------------------------
    @app.get("/media/manifest", response_model=MediaManifestResponse)
    def media_manifest() -> dict:
        """What media is prepared, and the evidence that the id mapping is correct."""
        return svc().media_manifest()

    @app.get("/feed", response_model=FeedResponse)
    def feed(user_id: int | None = Query(None)) -> dict:
        """Playable feed in recommendation order (only items with verified media)."""
        return svc().feed(user_id)

    @app.get("/items/{item_id}/media")
    def item_media(item_id: int) -> dict:
        return svc().item_media(item_id)

    @app.get("/media/video/{item_id}")
    def media_video(item_id: int):
        """Serve one prepared mp4.

        ``item_id`` is an integer path parameter and the file path comes from the
        manifest lookup only, so a crafted id cannot escape the media directory.
        """
        from fastapi.responses import FileResponse

        path = svc().media_video_path(int(item_id))
        if path is None:
            raise HTTPException(status_code=404, detail=f"no prepared media for item {item_id}")
        return FileResponse(path, media_type="video/mp4")

    # ------------------------------------------------------------------
    @app.get("/users/{user_id}", response_model=UserResponse)
    def user(user_id: int) -> dict:
        try:
            return _timed(svc().user, user_id)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/users/{user_id}/recall", response_model=RecallResponse)
    def recall(
        user_id: int,
        top_k: int = Query(200, ge=1, le=2000, description="per-channel budget"),
        sources: str | None = Query(None, description="comma separated channel names"),
    ) -> dict:
        src = [s.strip() for s in sources.split(",")] if sources else None
        try:
            return _timed(svc().recall, user_id, top_k, src)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/users/{user_id}/recommend", response_model=RecommendResponse)
    def recommend(
        user_id: int,
        ranker: str | None = Query(None, description="sasrec | mm_concat | mm_gated"),
        recall_k: int = Query(200, ge=1, le=2000, description="per-channel budget"),
        top_k: int = Query(20, ge=1, le=100),
        exploration: bool = Query(False, description="reserve exposure slots for zero-train items"),
        exploration_quota: int = Query(2, ge=0, le=20),
        cold_exploration: bool | None = Query(None, deprecated=True,
                                              description="alias of `exploration`"),
        cold_quota: int | None = Query(None, ge=0, le=20, deprecated=True),
        max_candidates: int | None = Query(None, ge=1, le=20000),
    ) -> dict:
        service_ = svc()
        if cold_exploration is not None:
            exploration = cold_exploration
        if cold_quota is not None:
            exploration_quota = cold_quota
        if ranker is not None and ranker not in service_.registry.names:
            raise HTTPException(status_code=400,
                                detail=f"unknown ranker {ranker!r}; available {service_.registry.names}")
        try:
            return _timed(service_.recommend, user_id, recall_k, top_k, ranker,
                          exploration, exploration_quota, max_candidates)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/users/{user_id}/inspect")
    def inspect(
        user_id: int,
        ranker: str | None = Query(None),
        compare: str | None = Query("sasrec", description="baseline ranker to diff against"),
        recall_k: int = Query(200, ge=1, le=2000),
        top_n: int = Query(30, ge=1, le=200),
        exploration: bool = Query(False),
        cold_exploration: bool | None = Query(None, deprecated=True),
    ) -> dict:
        if cold_exploration is not None:
            exploration = cold_exploration
        service_ = svc()
        for name in (ranker, compare):
            if name is not None and name not in service_.registry.names:
                raise HTTPException(status_code=400, detail=f"unknown ranker {name!r}")
        try:
            return _timed(service_.inspect, user_id, recall_k, top_n, ranker,
                          compare, exploration)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    @app.get("/cold/summary", response_model=ColdSummaryResponse)
    def cold_summary() -> dict:
        return svc().cold_summary()

    @app.get("/cold/items", response_model=list[ColdItemOut])
    def cold_items(n: int = Query(20, ge=1, le=100), seed: int = 0) -> list[dict]:
        return svc().cold_items(n=n, seed=seed)

    @app.get("/cold/items/{item_id}", response_model=ColdItemOut)
    def cold_item(item_id: int) -> dict:
        out = svc().cold_item(item_id)
        if out is None:
            raise HTTPException(status_code=404, detail=f"item {item_id} is not a cold item")
        return out

    # ------------------------------------------------------------------
    @app.post("/recommend", response_model=LegacyRecommendResponse)
    def legacy_recommend(req: LegacyRecommendRequest) -> dict:
        """History-based endpoint kept for backwards compatibility."""
        service_ = svc()
        internal = [service_.to_internal(i) for i in req.history]
        internal = [i for i in internal if i > 0]
        t0 = time.perf_counter()
        model = service_.registry.get(service_.default_ranker)
        if not internal:
            return {"items": [], "latency_ms": 0.0}
        merged = service_.merger.recall_and_merge(
            user_id=0, history=internal, per_source_k=max(req.top_k * 10, 200),
            total_k=1000,
        )
        ids = merged.item_ids()
        scores = model.score_candidates(internal, ids) if ids else []
        order = sorted(range(len(ids)), key=lambda j: -scores[j])[: req.top_k]
        items = [{"item_id": service_.to_raw(ids[j]), "score": float(scores[j])} for j in order]
        return {"items": items, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the ShortRec API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--cold-processed-dir", default="data/processed/cold10")
    ap.add_argument("--default-ranker", default="mm_concat")
    ap.add_argument("--history-mode", default="test", choices=["test", "full"])
    args = ap.parse_args()

    import uvicorn

    configure_threads()
    app = create_app(
        processed_dir=args.processed_dir,
        cold_processed_dir=args.cold_processed_dir,
        device=args.device,
        history_mode=args.history_mode,
        default_ranker=args.default_ranker,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
