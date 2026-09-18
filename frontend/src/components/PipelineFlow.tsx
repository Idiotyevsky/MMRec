import type { RecommendResponse } from "../api/types";
import { Badge, BucketBadge, num } from "./common";

/** Visual identity of each recall channel — one colour per source, used in
 *  every chart and card so the reader can follow a channel across the page. */
export const SOURCE_COLORS: Record<string, string> = {
  popular: "#7d99b5",
  itemcf: "#2f5d8a",
  semantic: "#4f7f6a",
};

export const SOURCE_LABEL: Record<string, string> = {
  popular: "Popular",
  itemcf: "ItemCF",
  semantic: "Semantic",
};

export const SOURCE_WHAT: Record<string, string> = {
  popular: "Global training-frequency prior. Always available, even with no history.",
  itemcf: "Items that co-occur with what this user watched (cosine ItemCF).",
  semantic: "Content-similar items from text + cover-image features. Works with zero interactions.",
};

export interface StageState {
  /** index of the stage currently running; stages before it are done */
  active: number;
  total: number;
}

function cls(idx: number, state: StageState): string {
  if (state.active > idx) return "step done";
  if (state.active === idx) return "step active";
  return "step";
}

/** Values appear only once their stage has completed, so each stage of the
 *  reveal has a visible payoff instead of everything appearing at once. */
function val(state: StageState, idx: number, value: React.ReactNode): React.ReactNode {
  return state.active > idx ? value : "—";
}

function chCls(idx: number, state: StageState): string {
  if (state.active > idx) return "channel done";
  if (state.active === idx) return "channel active";
  return "channel";
}

/** Staged reveal of one real request. No numbers are invented: every value is
 *  read from the API response, the animation only controls *when* it appears. */
export function PipelineFlow({
  rec,
  state,
}: {
  rec: RecommendResponse | null;
  state: StageState;
}) {
  const channels = rec?.recall_channels ?? [];
  const stats = rec?.recall;

  return (
    <div className="flow">
      {/* STEP 1 — history */}
      <div className={cls(0, state)}>
        <div className="step-no">STEP 1</div>
        <div className="step-title">Load user history</div>
        <div className="step-val">{val(state, 0, rec ? rec.history.length : 0)}</div>
        <div className="step-sub">recent interactions</div>
      </div>

      <div className="flow-connector">↓</div>

      {/* STEPS 2-4 — recall channels */}
      <div className="channels">
        {channels.map((c, i) => (
          <div key={c.name} className={chCls(i + 1, state)}>
            <h3>
              <span className="swatch" style={{ background: SOURCE_COLORS[c.name] ?? "#888" }} />
              {SOURCE_LABEL[c.name] ?? c.name} Recall
            </h3>
            <div className="what">{SOURCE_WHAT[c.name] ?? ""}</div>
            <div className="metric">
              <span className="muted">recalled</span>
              <span className="v">{val(state, i + 1, num(c.recalled))}</span>
            </div>
            <div className="metric">
              <span className="muted">only source for</span>
              <span className="v">{val(state, i + 1, num(c.unique_contribution))}</span>
            </div>
            <div className="metric">
              <span className="muted">latency</span>
              <span className="v">
                {val(state, i + 1, c.latency_ms === null ? "—" : `${c.latency_ms.toFixed(1)} ms`)}
              </span>
            </div>
          </div>
        ))}
        {!channels.length && (
          <div className="channel done">
            <h3 className="muted">Recall channels</h3>
            <div className="what">Run a recommendation to see the per-channel trace.</div>
          </div>
        )}
      </div>

      <div className="flow-connector">↓</div>

      {/* STEP 5 — merge */}
      <div className={cls(4, state)}>
        <div className="row between">
          <div>
            <div className="step-no">STEP 5</div>
            <div className="step-title">Candidate merge · dedup · reciprocal rank fusion</div>
          </div>
          {stats && (
            <div className="row" style={{ gap: 22 }}>
              <div style={{ textAlign: "right" }}>
                <div className="small faint">before dedup</div>
                <strong>{val(state, 4, num(stats.before_dedup))}</strong>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="small faint">duplicates merged</div>
                <strong>{val(state, 4, num(stats.duplicates_removed))}</strong>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="small faint">after merge</div>
                <strong>{val(state, 4, num(stats.after_dedup))}</strong>
              </div>
            </div>
          )}
        </div>
        <div className="step-sub" style={{ marginTop: 8 }}>
          RRF combines the channels by <em>rank</em>, because their raw scores are not
          comparable (counts vs similarity sums vs cosine). Duplicate items are merged, and
          every contributing channel is kept on the candidate.
        </div>
      </div>

      <div className="flow-connector">↓</div>

      {/* STEP 6 — ranking */}
      <div className={`${cls(5, state)} rank`}>
        <div className="step-no">STEP 6</div>
        <div className="step-title">
          {rec?.ranker ?? "MM-SASRec"} — personalized ranking
        </div>
        <div className="step-val">{val(state, 5, stats ? num(stats.candidates_ranked) : 0)}</div>
        <div className="step-sub">
          candidates scored with the user sequence + item ID + text + image features
        </div>
      </div>

      <div className="flow-connector">↓</div>

      {/* STEP 7 — rerank */}
      <div className={cls(6, state)}>
        <div className="step-no">STEP 7</div>
        <div className="step-title">Serving rerank</div>
        <div className="step-sub" style={{ marginTop: 6 }}>
          seen filter · dedup ·{" "}
          {rec && (rec.rerank as { config?: { exploration?: boolean } })?.config?.exploration
            ? "zero-train exploration on"
            : "zero-train exploration off"}
          {stats ? val(state, 6, ` · ${num(stats.exploration_in_pool)} zero-train candidates in the pool`) : ""}
        </div>
      </div>

      <div className="flow-connector">↓</div>

      {/* STEP 8 — feed */}
      <div className={cls(7, state)}>
        <div className="step-no">STEP 8</div>
        <div className="step-title">Top-K feed</div>
        <div className="step-val">{val(state, 7, rec ? rec.recommendations.length : 0)}</div>
        <div className="step-sub">
          {rec && rec.target_item !== null
            ? `held-out target #${rec.target_item} ${rec.target_hit ? "retrieved" : "not retrieved"}`
            : "served items"}
        </div>
      </div>
    </div>
  );
}

/** One served item, with the recall trace and (optionally) the rank movement
 *  against the ID-only baseline. */
export function RecommendationCard({
  item,
  showMovement,
}: {
  item: import("../api/types").RecommendationItem;
  showMovement: boolean;
}) {
  const delta = item.position_delta ?? null;
  return (
    <div className={`rcard${item.exploration ? " explored" : ""}`}>
      <div className="top">
        <span className="rank">#{item.final_rank}</span>
        <span className="row" style={{ gap: 4 }}>
          <BucketBadge bucket={item.popularity_bucket} />
          {item.is_zero_train_signal ? (
            <Badge kind="cold">zero-train</Badge>
          ) : (
            <Badge kind="plain">warm</Badge>
          )}
        </span>
      </div>

      <div className="iid">Item #{item.item_id}</div>

      <div className="rows">
        <div className="r">
          <span className="k">Score</span>
          <span className="v">{item.ranking_score.toFixed(3)}</span>
        </div>
        <div className="r">
          <span className="k">Train interactions</span>
          <span className="v">{num(item.train_interactions)}</span>
        </div>
      </div>

      <div>
        <div className="small faint" style={{ marginBottom: 4 }}>Recalled by</div>
        <div className="sources">
          {item.sources.map((s) => (
            <Badge key={s} kind="src">
              {SOURCE_LABEL[s] ?? s}
              {item.recall_rank?.[s] !== undefined ? ` #${item.recall_rank[s]}` : ""}
            </Badge>
          ))}
        </div>
      </div>

      {item.exploration && (
        <div className="r">
          <span className="k">Note</span>
          <span className="v"><Badge kind="cold">reserved exploration slot</Badge></span>
        </div>
      )}

      {showMovement && delta !== null && (
        <div className="move">
          <span className="muted">SASRec → MM</span>
          <span className={delta > 0 ? "up" : delta < 0 ? "down" : "faint"}>
            {item.baseline_position ?? "—"} → {item.final_rank}
            {delta !== 0 ? `  ${delta > 0 ? "↑" : "↓"}${Math.abs(delta)}` : ""}
          </span>
        </div>
      )}
    </div>
  );
}
