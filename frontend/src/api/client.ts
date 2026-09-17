import type {
  ColdItem,
  ColdSummary,
  InspectResponse,
  ModelInfo,
  RecallResponse,
  RecommendResponse,
  SystemResponse,
  UserResponse,
} from "./types";

const BASE = "/api";

async function get<T>(path: string, params?: Record<string, unknown>): Promise<T> {
  const url = new URL(BASE + path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url.toString());
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body.slice(0, 300)}`);
  }
  return (await res.json()) as T;
}

export const api = {
  system: () => get<SystemResponse>("/system"),
  models: () => get<{ models: ModelInfo[] }>("/models"),
  user: (id: number) => get<UserResponse>(`/users/${id}`),
  recall: (id: number, topK = 200, sources?: string) =>
    get<RecallResponse>(`/users/${id}/recall`, { top_k: topK, sources }),
  recommend: (
    id: number,
    opts: {
      ranker?: string;
      recall_k?: number;
      top_k?: number;
      cold_exploration?: boolean;
      cold_quota?: number;
    } = {},
  ) => get<RecommendResponse>(`/users/${id}/recommend`, opts),
  inspect: (
    id: number,
    opts: { ranker?: string; compare?: string; recall_k?: number; top_n?: number } = {},
  ) => get<InspectResponse>(`/users/${id}/inspect`, opts),
  coldSummary: () => get<ColdSummary>("/cold/summary"),
  coldItems: (n = 12, seed = 0) => get<ColdItem[]>("/cold/items", { n, seed }),
  coldItem: (id: number) => get<ColdItem>(`/cold/items/${id}`),
};
