import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RecommendResponse, SystemResponse, UserResponse } from "../api/types";
import { Badge, BucketBadge, ColdBadge, ErrorBox, Panel, Stat, fmt, num } from "../components/common";
import { UserPicker } from "../components/UserPicker";

export default function Feed() {
  const [system, setSystem] = useState<SystemResponse | null>(null);
  const [userId, setUserId] = useState(7);
  const [ranker, setRanker] = useState<string>("");
  const [topK, setTopK] = useState(10);
  const [coldExploration, setColdExploration] = useState(false);

  const [user, setUser] = useState<UserResponse | null>(null);
  const [rec, setRec] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api.system().then((s) => {
      setSystem(s);
      setRanker((r) => r || s.default_ranker);
    }).catch(setError);
  }, []);

  useEffect(() => {
    setError(null);
    api.user(userId).then(setUser).catch(setError);
  }, [userId]);

  useEffect(() => {
    if (!ranker) return;
    setLoading(true);
    setError(null);
    api
      .recommend(userId, { ranker, top_k: topK, cold_exploration: coldExploration, recall_k: 200 })
      .then(setRec)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [userId, ranker, topK, coldExploration]);

  const modelInfo = system?.rankers.find((m) => m.name === ranker);

  return (
    <>
      <div className="page-head">
        <h1>Feed</h1>
        <p>
          Multi-channel recall → candidate merge → multimodal sequential ranking → rerank.
          Pick a user, switch the ranker, and compare the served top-K.
        </p>
      </div>

      <ErrorBox error={error} />

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        {/* ---------------- left: user + controls ---------------- */}
        <div>
          <Panel title="User">
            <UserPicker userId={userId} onChange={setUserId} maxUsers={system?.users ?? 100000} />
            {user && (
              <>
                <div className="row" style={{ marginTop: 12, gap: 16 }}>
                  <span className="small muted">history length</span>
                  <strong>{user.history_length}</strong>
                </div>
                <div className="panel-title" style={{ marginTop: 14 }}>History</div>
                <div className="hist">
                  {user.history.map((h, i) => (
                    <div
                      key={`${h.item_id}-${i}`}
                      className={`h${h.item_id === user.target_item ? " target" : ""}`}
                      title={`train interactions: ${h.train_interactions}`}
                    >
                      #{h.item_id} <BucketBadge bucket={h.popularity_bucket} />
                    </div>
                  ))}
                </div>
                {user.target_item !== null && (
                  <div className="panel-note">
                    Held-out next item, <strong>not</strong> part of the history above and never fed
                    to the model: <code>#{user.target_item}</code>. The Feed page reports whether it
                    made it into the served top-K.
                  </div>
                )}
              </>
            )}
          </Panel>

          <Panel title="Ranker">
            <div className="seg" style={{ width: "100%" }}>
              {system?.rankers.map((m) => (
                <button
                  key={m.name}
                  className={ranker === m.name ? "on" : ""}
                  onClick={() => setRanker(m.name)}
                  style={{ flex: 1 }}
                  title={m.description}
                >
                  {m.name === "sasrec" ? "SASRec (ID)" : m.name === "mm_concat" ? "MM Concat" : "MM Gated"}
                </button>
              ))}
            </div>
            {modelInfo && (
              <div className="mt">
                <div className="small muted">{modelInfo.description}</div>
                <div className="row" style={{ marginTop: 8, gap: 18 }}>
                  <div>
                    <div className="k faint small">Recall@20 (offline)</div>
                    <strong>{fmt(modelInfo.offline["Recall@20"])}</strong>
                  </div>
                  <div>
                    <div className="k faint small">NDCG@20</div>
                    <strong>{fmt(modelInfo.offline["NDCG@20"])}</strong>
                  </div>
                  <div>
                    <div className="k faint small">seeds</div>
                    <strong>{modelInfo.offline["n_seeds"] ?? "—"}</strong>
                  </div>
                </div>
                <div className="panel-note">
                  Offline metrics are read from <code>results/tables/overall.csv</code> and come from
                  full-catalogue ranking, not from this pipeline.
                </div>
              </div>
            )}
          </Panel>

          <Panel title="Serving options">
            <div className="row between">
              <span className="small muted">Top-K</span>
              <div className="seg">
                {[10, 20].map((k) => (
                  <button key={k} className={topK === k ? "on" : ""} onClick={() => setTopK(k)}>
                    {k}
                  </button>
                ))}
              </div>
            </div>
            <div className="row between mt">
              <span className="small muted">Cold exploration</span>
              <div className="seg">
                <button className={!coldExploration ? "on" : ""} onClick={() => setColdExploration(false)}>
                  off
                </button>
                <button className={coldExploration ? "on" : ""} onClick={() => setColdExploration(true)}>
                  on
                </button>
              </div>
            </div>
            <div className="panel-note">
              Cold exploration reserves slots for cold items. It is an exposure policy, not a model
              improvement — offline metrics are measured with it off.
            </div>
          </Panel>
        </div>

        {/* ---------------- right: feed ---------------- */}
        <div>
          {rec && (
            <div className="grid cols-4" style={{ marginBottom: 16 }}>
              <Stat label="Candidates" value={num(rec.recall.after_dedup)} sub={`from ${num(rec.recall.before_dedup)} recalled`} />
              <Stat label="Recall latency" value={`${rec.latency_ms.recall ?? "—"} ms`} />
              <Stat label="Rank latency" value={`${rec.latency_ms.rank ?? "—"} ms`} />
              <Stat
                label="Target in top-K"
                value={rec.target_hit === null ? "—" : rec.target_hit ? "hit" : "miss"}
                sub={rec.target_item !== null ? `target #${rec.target_item}` : undefined}
              />
            </div>
          )}

          <Panel
            title={`Recommended feed — top ${topK}`}
            right={
              rec ? (
                <span className="small faint">
                  ranker <code>{rec.ranker}</code> · total {rec.latency_ms.total} ms
                </span>
              ) : undefined
            }
          >
            {loading && <div className="loading">loading…</div>}
            {!loading && rec && (
              <div className="feed">
                {rec.recommendations.map((r) => (
                  <div className="card" key={r.item_id}>
                    <div className="row between">
                      <span className="rank">#{r.final_rank}</span>
                      <span className="row" style={{ gap: 4 }}>
                        <BucketBadge bucket={r.popularity_bucket} />
                        <ColdBadge cold={r.is_cold} />
                      </span>
                    </div>
                    <div className="id">Item #{r.item_id}</div>
                    <div className="kv">
                      <span className="k">Score</span>
                      <span className="v">{fmt(r.ranking_score, 3)}</span>
                    </div>
                    <div className="kv">
                      <span className="k">Recall</span>
                      <span className="v row" style={{ gap: 4 }}>
                        {r.sources.map((s) => (
                          <Badge key={s} kind="src">{s}</Badge>
                        ))}
                      </span>
                    </div>
                    <div className="kv">
                      <span className="k">Train interactions</span>
                      <span className="v">{num(r.train_interactions)}</span>
                    </div>
                    {r.exploration && (
                      <div className="kv">
                        <span className="k">Note</span>
                        <span className="v"><Badge kind="cold">exploration slot</Badge></span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
            {!loading && !rec && <div className="loading">no recommendations yet</div>}
          </Panel>

          {rec && (
            <Panel title="Recall sources" note="How many candidates each channel contributed after dedup.">
              <div className="row" style={{ gap: 24 }}>
                {Object.entries(rec.recall.source_coverage_in_pool ?? {}).map(([k, v]) => (
                  <div key={k}>
                    <div className="small faint">{k}</div>
                    <strong>{num(v)}</strong>
                  </div>
                ))}
                <div>
                  <div className="small faint">duplicates removed</div>
                  <strong>{num(rec.recall.duplicates_removed)}</strong>
                </div>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </>
  );
}
