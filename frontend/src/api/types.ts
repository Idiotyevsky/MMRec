// Response types mirroring src/serving/schemas.py.
// Item ids are raw MicroLens ids everywhere in the UI.

export type Bucket = "head" | "middle" | "tail" | "cold" | "unknown";

export interface SourceTrace {
  name: string;
  score: number;
  rank: number;
  norm_score?: number;
}

export interface HistoryItem {
  item_id: number;
  popularity_bucket: Bucket;
  is_cold: boolean;
  train_interactions: number;
}

export interface RecommendationItem {
  item_id: number;
  ranking_score: number;
  // present on the inspect endpoint (the recall merge score), absent on /recommend
  merge_score?: number;
  final_rank: number;
  sources: string[];
  recall_rank: Record<string, number>;
  is_cold: boolean;
  popularity_bucket: Bucket;
  train_interactions: number;
  exploration: boolean;
  compare_score?: number | null;
  score_delta?: number | null;
}

export interface RecallStats {
  per_source: Record<string, number>;
  before_dedup: number;
  after_dedup: number;
  duplicates_removed: number;
  candidates_ranked: number;
  cold_in_pool: number;
  source_coverage_in_pool: Record<string, number>;
}

export interface RecommendResponse {
  user_id: number;
  history: number[];
  history_mode: string;
  ranker: string;
  recall_k: number;
  final_k: number;
  recall: RecallStats;
  rerank: Record<string, unknown>;
  latency_ms: { recall?: number; rank?: number; rerank?: number; total: number };
  target_item: number | null;
  target_hit: boolean | null;
  recommendations: RecommendationItem[];
}

export interface UserResponse {
  user_id: number;
  history_length: number;
  history: HistoryItem[];
  recent: HistoryItem[];
  target_item: number | null;
}

export interface RecallResponse {
  user_id: number;
  recall_k: number;
  stats: RecallStats;
  candidates: {
    item_id: number;
    sources: SourceTrace[];
    merge_score: number;
    popularity_bucket: Bucket;
    is_cold: boolean;
    train_interactions: number;
  }[];
}

export interface ModelInfo {
  name: string;
  description: string;
  run_dir: string | null;
  tag?: string;
  offline: Record<string, number>;
}

export interface SystemResponse {
  dataset: string;
  users: number;
  items: number;
  interactions: number;
  default_ranker: string;
  rankers: ModelInfo[];
  recall_sources: string[];
  recall_readiness: Record<string, [boolean, string]>;
  item_metadata: {
    num_items: number;
    bucket_rule: string;
    bucket_counts: Record<string, number>;
    has_cold_split: boolean;
    cold_items: number;
    zero_train_frequency_items: number;
  };
  history_mode: string;
}

export interface InspectResponse {
  user_id: number;
  history: number[];
  history_items: HistoryItem[];
  history_length: number;
  history_mode: string;
  target_item: number | null;
  ranker: string;
  compare_ranker: string | null;
  recall_summary: RecallStats;
  recall_candidates: {
    item_id: number;
    sources: SourceTrace[];
    merge_score: number;
  }[];
  top_candidates: RecommendationItem[];
  baseline_top: {
    item_id: number;
    baseline_rank: number;
    baseline_score: number;
    multimodal_score: number;
    delta: number;
    popularity_bucket: Bucket;
    is_cold: boolean;
    train_interactions: number;
  }[];
  moved_up_by_multimodal: (RecommendationItem & { position_delta: number })[];
  moved_down_by_multimodal: (RecommendationItem & { position_delta: number })[];
  final_top_k: RecommendationItem[];
  rerank: Record<string, unknown>;
  latency_ms: { total: number };
}

export interface ColdItem {
  item_id: number;
  train_interactions: number;
  is_cold: boolean;
  popularity_bucket: Bucket;
  content_available: Record<string, boolean>;
  content_similar: {
    item_id: number;
    similarity: number;
    popularity_bucket: Bucket;
    train_interactions: number;
  }[];
}

export interface ColdExperiment {
  model: string;
  kind: string;
  modalities: string;
  // the API keys contain '@', so they must be quoted
  "cold_recall@10": number | null;
  "cold_recall@20": number | null;
  "cold_only_recall@10": number | null;
  "cold_only_recall@20": number | null;
}

export interface ColdSummary {
  dataset: string;
  num_cold_items: number;
  num_users_with_cold_target: number;
  experiments: ColdExperiment[];
  note: string;
}
