// Response types mirroring src/serving/schemas.py.
// Item ids are raw MicroLens ids everywhere in the UI.

export type Bucket = "head" | "middle" | "tail" | "simulated_cold" | "cold" | "unknown";

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
  is_simulated_cold?: boolean;
  is_zero_train_signal?: boolean;
  exploration_candidate?: boolean;
  train_interactions: number;
}

export interface RecommendationItem {
  item_id: number;
  ranking_score: number;
  final_rank: number;
  merge_score?: number;
  sources: string[];
  source_trace?: SourceTrace[];
  recall_rank: Record<string, number>;
  is_cold: boolean;
  is_simulated_cold?: boolean;
  is_zero_train_signal?: boolean;
  exploration_candidate?: boolean;
  exploration: boolean;
  popularity_bucket: Bucket;
  train_interactions: number;
  compare_score?: number | null;
  score_delta?: number | null;
  /** position in the baseline ranking of the same pool (inspect only) */
  baseline_position?: number | null;
  rank_delta?: number | null;
  position_delta?: number;
}

export interface RecallChannelTrace {
  name: string;
  recalled: number;
  in_pool: number;
  unique_contribution: number;
  latency_ms: number | null;
  target_hit: boolean | null;
}

export interface RecallStats {
  per_source: Record<string, number>;
  unique_contribution: Record<string, number>;
  before_dedup: number;
  after_dedup: number;
  duplicates_removed: number;
  candidates_ranked: number;
  exploration_in_pool: number;
  source_coverage_in_pool: Record<string, number>;
  per_source_latency_ms: Record<string, number>;
}

export interface RecommendResponse {
  user_id: number;
  history: number[];
  history_mode: string;
  ranker: string;
  recall_k: number;
  final_k: number;
  recall: RecallStats;
  recall_channels: RecallChannelTrace[];
  rerank: Record<string, unknown>;
  exploration: Record<string, unknown>;
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
    is_zero_train_signal?: boolean;
    exploration_candidate?: boolean;
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

export interface ExplorationSummary {
  serving_zero_train_items: number;
  serving_exploration_candidates: number;
  benchmark_simulated_cold_items: number;
  note: string;
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
    exploration_candidates: number;
  };
  exploration?: ExplorationSummary;
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
  recall_channels: RecallChannelTrace[];
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
  is_simulated_cold?: boolean;
  is_zero_train_signal?: boolean;
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
  "cold_recall@10": number | null;
  "cold_recall@20": number | null;
  "cold_only_recall@10": number | null;
  "cold_only_recall@20": number | null;
}

export interface ColdSummary {
  dataset: string;
  num_cold_items: number;
  num_users_with_cold_target: number;
  serving_zero_train_items: number;
  experiments: ColdExperiment[];
  note: string;
}

export interface EvaluationResponse {
  recall_by_budget: {
    ks: number[];
    channels: { channel: string; values: (number | null)[] }[];
    num_users: number | null;
  };
  pipeline_tradeoff: Record<string, number | string | null>[];
  latency: Record<string, number | string | null>[];
  source_contribution: Record<string, number | string | null>[];
  notes: Record<string, string>;
}
