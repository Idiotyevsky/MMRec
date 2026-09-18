import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { InspectResponse, SystemResponse, UserResponse } from "../api/types";
import { Badge, BucketBadge, ErrorBox, Panel, Stat, fmt, num, pct } from "../components/common";
import { PipelineFlow, RecommendationCard, SOURCE_LABEL } from "../components/PipelineFlow";
import { UserPicker } from "../components/UserPicker";

/** Total staged-reveal time in ms (the spec asks for 3-5 s). */
const REVEAL_MS = 4200;
const STAGES = 8;

export default function LiveDemo() {
  const [system, setSystem] = useState<SystemResponse | null>(null);
  const [userId, setUserId] = useState(7);
  const [ranker, setRanker] = useState("mm_concat");
  const [compare, setCompare] = useState(true);
  const [topK, setTopK] = useState(10);
  const [exploration, setExploration] = useState(false);

  const [user, setUser] = useState<UserResponse | null>(null);
  const [trace, setTrace] = useState<InspectResponse | null>(null);
  const [stage, setStage] = useState(STAGES); // STAGES = idle/finished
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    api.system().then((s) => {
      setSystem(s);
      setRanker((r) => r || s.default_ranker);
    }).catch(setError);
  }, []);

  useEffect(() => {
    api.user(userId).then(setUser).catch(setError);
  }, [userId]);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    setTrace(null);
    setStage(0);
    if (timer.current) window.clearInterval(timer.current);

    let res: InspectResponse;
    try {
      res = await api.inspect(userId, {
        ranker,
        compare: compare ? "sasrec" : "",
        recall_k: 200,
        top_n: topK,
      });
    } catch (e) {
      setError(e);
      setRunning(false);
      setStage(STAGES);
      return;
    }

    // The reveal starts *after* the response lands.  Animating while waiting
    // would reveal empty placeholders; every number shown here is real, the
    // staging only controls when it becomes visible.
    setTrace(res);
    setRunning(false);
    let i = 0;
    const perStage = REVEAL_MS / STAGES;
    timer.current = window.setInterval(() => {
      i += 1;
      setStage(i);
      if (i >= STAGES && timer.current) window.clearInterval(timer.current);
    }, perStage);
  }, [userId, ranker, compare, topK]);

  useEffect(() => () => {
    if (timer.current) window.clearInterval(timer.current);
  }, []);

  // present the inspect payload in the shape the flow component expects
  const flowRec = useMemo(() => {
    if (!trace) return null;
    return {
      user_id: trace.user_id,
      history: trace.history,
      history_mode: trace.history_mode,
      ranker: trace.ranker,
      recall_k: 0,
      final_k: trace.final_top_k.length,
      recall: trace.recall_summary,
      recall_channels: trace.recall_channels,
      rerank: trace.rerank,
      exploration: {},
      latency_ms: trace.latency_ms,
      target_item: trace.target_item,
      target_hit:
        trace.target_item !== null &&
        trace.final_top_k.some((r) => r.item_id === trace.target_item),
      recommendations: trace.final_top_k,
    };
  }, [trace]);

  const served = useMemo(() => {
    if (!trace) return [];
    const byId = new Map(trace.top_candidates.map((c) => [c.item_id, c]));
    return trace.final_top_k.slice(0, topK).map((r) => {
      const detailed = byId.get(r.item_id);
      return { ...r, ...(detailed ?? {}) } as typeof r;
    });
  }, [trace, topK]);

  const modelInfo = system?.rankers.find((m) => m.name === ranker);
  const baseInfo = system?.rankers.find((m) => m.name === "sasrec");
  const showMovement = compare && !!trace?.compare_ranker;

  const gained =
    modelInfo?.offline["Recall@20"] && baseInfo?.offline["Recall@20"]
      ? modelInfo.offline["Recall@20"] / baseInfo.offline["Recall@20"] - 1
      : null;

  return (
    <>
      <div className="hero">
        <h1>Live Recommendation</h1>
        <p>
          Watch how ShortRec turns a user's recent history into a ranked short-video feed:
          three recall channels propose candidates, they are merged, and a multimodal
          sequential ranker reorders them using behaviour, text and cover-image features.
        </p>
      </div>

      <ErrorBox error={error} />

      <Panel title="Request">
        <div className="row" style={{ gap: 18, alignItems: "flex-end" }}>
          <label className="field">
            User
            <UserPicker userId={userId} onChange={setUserId} maxUsers={system?.users ?? 100000} />
          </label>
          <label className="field">
            Ranker
            <select value={ranker} onChange={(e) => setRanker(e.target.value)}>
              {system?.rankers.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name === "sasrec" ? "SASRec (ID only)" : m.name === "mm_concat" ? "MM-SASRec Concat" : "MM-SASRec Gated"}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Top-K
            <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              {[10, 20].map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </label>
          <label className="field">
            Compare with SASRec
            <div className="seg">
              <button className={compare ? "on" : ""} onClick={() => setCompare(true)}>on</button>
              <button className={!compare ? "on" : ""} onClick={() => setCompare(false)}>off</button>
            </div>
          </label>
          <label className="field">
            Zero-train exploration
            <div className="seg">
              <button className={exploration ? "on" : ""} onClick={() => setExploration(true)}>on</button>
              <button className={!exploration ? "on" : ""} onClick={() => setExploration(false)}>off</button>
            </div>
          </label>
          <span className="spacer" />
          <button className="run" onClick={run} disabled={running}>
            {running ? "Running…" : "Run Recommendation"}
          </button>
        </div>
        <div className="run-hint">
          The staged reveal below is driven by the numbers this request actually returned.
        </div>
      </Panel>

      {user && (
        <Panel title="User history" note={
          user.target_item !== null
            ? `Held-out next item #${user.target_item} — for offline inspection only, never fed to the model.`
            : undefined
        }>
          <div className="hist">
            {user.history.map((h, i) => (
              <div key={`${h.item_id}-${i}`} className="h" title={`train interactions: ${h.train_interactions}`}>
                #{h.item_id} <BucketBadge bucket={h.popularity_bucket} />
                {h.is_zero_train_signal ? <Badge kind="cold">zero-train</Badge> : null}
              </div>
            ))}
          </div>
          <div className="panel-note">
            MicroLens-100K ships no titles or captions, so items are identified by their raw id.
          </div>
        </Panel>
      )}

      <Panel
        title="Recommendation pipeline"
        right={trace ? (
          <span className="small faint">
            total {trace.latency_ms.total} ms · {trace.ranker}
          </span>
        ) : undefined}
      >
        <PipelineFlow rec={flowRec} state={{ active: stage, total: STAGES }} />
      </Panel>

      {trace && stage >= STAGES && (
        <>
          <div className="grid cols-4" style={{ marginTop: 16 }}>
            <Stat label="Candidates ranked" value={num(trace.recall_summary.candidates_ranked)} />
            <Stat label="Zero-train in pool" value={num(trace.recall_summary.exploration_in_pool)}
                  sub={trace.recall_summary.exploration_in_pool > 0 ? "no collaborative signal" : "none recalled"} />
            <Stat label="Held-out target"
                  value={trace.target_item === null ? "—" : trace.target_item}
                  sub={served.some((r) => r.item_id === trace.target_item) ? "retrieved in top-K" : "not retrieved"} />
            <Stat label="Recall → rank" value={`${trace.latency_ms.total} ms`} sub="CPU serving" />
          </div>

          <Panel
            title={`Served top-${served.length}`}
            note={showMovement
              ? "SASRec → MM shows where the ID-only baseline placed the same item inside the same candidate pool. Rank movement is the honest comparison; raw scores from different models are not on a common scale."
              : undefined}
          >
            <div className="feed2">
              {served.map((r) => (
                <RecommendationCard key={r.item_id} item={r} showMovement={showMovement} />
              ))}
            </div>
          </Panel>
        </>
      )}

      {modelInfo && baseInfo && (
        <Panel title="Why multimodal?">
          <div className="vs">
            <div className="vs-card">
              <h3>ID-only SASRec</h3>
              <div className="cap">Collaborative signal only</div>
              <div className="big">{pct(baseInfo.offline["Recall@20"])}</div>
              <div className="cap">full-catalogue Recall@20 · {baseInfo.offline["n_seeds"] ?? "—"} seeds</div>
            </div>
            <div className="vs-mid">VS</div>
            <div className="vs-card good">
              <h3>MM-SASRec Concat</h3>
              <div className="cap">ID + text + cover image</div>
              <div className="big">{pct(modelInfo.offline["Recall@20"])}</div>
              <div className="cap">
                full-catalogue Recall@20
                {gained !== null && (
                  <> · <strong>+{(gained * 100).toFixed(1)}%</strong></>
                )}
              </div>
            </div>
          </div>
          <div className="panel-note">
            Offline metrics are read from <code>results/tables/overall.csv</code> and come from
            strict full-catalogue ranking — a different protocol from the pipeline above.
          </div>
        </Panel>
      )}

      {trace && trace.recall_channels.length > 0 && (
        <Panel title="This request, per channel" note="Numbers from the request you just ran.">
          <table>
            <thead>
              <tr>
                <th>Channel</th>
                <th>Recalled</th>
                <th>Only source for</th>
                <th>Latency</th>
                <th>Target hit</th>
              </tr>
            </thead>
            <tbody>
              {trace.recall_channels.map((c) => (
                <tr key={c.name}>
                  <td><Badge kind="src">{SOURCE_LABEL[c.name] ?? c.name}</Badge></td>
                  <td>{num(c.recalled)}</td>
                  <td>{num(c.unique_contribution)}</td>
                  <td>{c.latency_ms === null ? "—" : `${c.latency_ms.toFixed(1)} ms`}</td>
                  <td>{c.target_hit ? <Badge kind="cold">yes</Badge> : <span className="faint">no</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}

      {!trace && !running && (
        <Panel>
          <div className="muted">
            Press <strong>Run Recommendation</strong> to send one request through the pipeline.
            {system && (
              <span className="faint"> ({fmt(system.items, 0)} items, {system.recall_sources.join(" / ")} recall)</span>
            )}
          </div>
        </Panel>
      )}
    </>
  );
}
