"""API schemas.

Two id spaces meet here and nowhere else:

* the model / recall layer works with **internal** ids (``1..num_items``);
* the API and the frontend speak **raw** MicroLens ids.

Every response therefore carries raw ids, and the conversion is done once in
:mod:`src.serving.app`.  Keeping it in one place is what makes it possible to
state "the ids in the UI are the ids in the dataset".
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecallSourceStat(BaseModel):
    name: str
    score: float
    rank: int
    norm_score: float | None = None


class RecommendationItem(BaseModel):
    item_id: int = Field(..., description="raw MicroLens item id")
    ranking_score: float
    final_rank: int
    sources: list[str]
    recall_rank: dict[str, int] = Field(default_factory=dict)
    is_cold: bool
    popularity_bucket: str
    train_interactions: int
    exploration: bool = False
    compare_score: float | None = None
    score_delta: float | None = None


class RecallStats(BaseModel):
    per_source: dict[str, int] = Field(default_factory=dict)
    before_dedup: int = 0
    after_dedup: int = 0
    duplicates_removed: int = 0
    candidates_ranked: int = 0
    cold_in_pool: int = 0
    source_coverage_in_pool: dict[str, int] = Field(default_factory=dict)


class Latency(BaseModel):
    recall: float | None = None
    rank: float | None = None
    rerank: float | None = None
    total: float


class RecommendResponse(BaseModel):
    user_id: int
    history: list[int] = Field(..., description="raw item ids, oldest first")
    history_mode: str
    ranker: str
    recall_k: int
    final_k: int
    recall: RecallStats
    rerank: dict
    latency_ms: Latency
    target_item: int | None = None
    target_hit: bool | None = None
    recommendations: list[RecommendationItem]


class HistoryItem(BaseModel):
    item_id: int
    popularity_bucket: str
    is_cold: bool
    train_interactions: int


class UserResponse(BaseModel):
    user_id: int
    history_length: int
    history: list[HistoryItem]
    recent: list[HistoryItem]
    target_item: int | None = None


class RecallCandidateOut(BaseModel):
    item_id: int
    sources: list[RecallSourceStat]
    merge_score: float
    popularity_bucket: str
    is_cold: bool
    train_interactions: int


class RecallResponse(BaseModel):
    user_id: int
    recall_k: int
    stats: RecallStats
    candidates: list[RecallCandidateOut]


class ModelInfo(BaseModel):
    name: str
    description: str
    run_dir: str | None = None
    offline: dict = Field(default_factory=dict)


class SystemResponse(BaseModel):
    dataset: str
    users: int
    items: int
    interactions: int
    default_ranker: str
    rankers: list[ModelInfo]
    recall_sources: list[str]
    recall_readiness: dict[str, list] = Field(default_factory=dict)
    item_metadata: dict
    history_mode: str


class ColdItemOut(BaseModel):
    item_id: int
    train_interactions: int
    is_cold: bool
    content_available: dict[str, bool] = Field(default_factory=dict)
    content_similar: list[dict] = Field(default_factory=list)


class ColdSummaryResponse(BaseModel):
    dataset: str
    num_cold_items: int
    num_users_with_cold_target: int
    experiments: list[dict]
    note: str


class LegacyRecommendRequest(BaseModel):
    history: list[int]
    top_k: int = 20


class LegacyRecommendResponse(BaseModel):
    items: list[dict]
    latency_ms: float
