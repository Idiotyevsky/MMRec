import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { SystemResponse } from "../api/types";
import { Badge, Bars, ErrorBox, Panel, Stat, fmt, num, pct } from "../components/common";
import { ArchitectureDiagram } from "../components/ArchitectureDiagram";
import { Legend, LineChart, type Series } from "../components/Charts";
import { SOURCE_COLORS, SOURCE_LABEL } from "../components/PipelineFlow";

interface Evaluation {
  recall_by_budget: {
    ks: number[];
    channels: { channel: string; values: (number | null)[] }[];
    num_users: number | null;
  };
  pipeline_tradeoff: Record<string, number | string | null>[];
  latency: Record<string, number | string | null>[];
  notes: Record<string, string>;
}

export default function System() {
  const [system, setSystem] = useState<SystemResponse | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api.system().then(setSystem).catch(setError);
    api.evaluation().then(setEvaluation).catch(() => undefined); // charts are optional
  }, []);

  if (error) return <ErrorBox error={error} />;
  if (!system) return <div className="loading">loading system info…</div>;

  const buckets = system.item_metadata.bucket_counts ?? {};
  const bucketData = ["head", "middle", "tail", "simulated_cold"]
    .filter((b) => (buckets[b] ?? 0) > 0)
    .map((b) => ({ label: b, value: buckets[b], display: num(buckets[b]) }));

  const recall = evaluation?.recall_by_budget;
  const series: Series[] = (recall?.channels ?? []).map((c) => ({
    name: SOURCE_LABEL[c.channel] ?? c.channel,
    color: SOURCE_COLORS[c.channel] ?? "#888",
    values: c.values,
  }));

  const tradeoff = (evaluation?.pipeline_tradeoff ?? [])
    .filter((r) => r.candidate_k !== undefined)
    .map((r) => ({
      k: Number(r.candidate_k),
      cand: Number(r.candidate_recall),
      rec: Number(r["pipeline_Recall@20"]),
      retention: Number(r.recall_retention),
    }));
  const fullRecall = evaluation?.pipeline_tradeoff?.[0]
    ? Number((evaluation.pipeline_tradeoff[0] as Record<string, unknown>)["full_Recall@20"])
    : null;

  const latencyRows = (evaluation?.latency ?? []).map((r) => ({
    k: r.scope === "full_catalogue" ? "full" : String(r.candidate_k),
    recall: Number(r.recall_ms_median),
    encode: Number(r.encode_ms_median),
    score: Number(r.score_ms_median),
    e2e: Number(r.end_to_end_ms_median),
  }));

  return (
    <>
      <div className="page-head">
        <h1>System</h1>
        <p>
          Is this an actual system or just a model demo? Below: the offline build, the online
          serving path, the recall/ranking evaluation, and the API that exposes all of it.
        </p>
      </div>

      <div className="grid cols-4">
        <Stat label="Dataset" value={system.dataset} sub={`history mode: ${system.history_mode}`} />
        <Stat label="Users" value={num(system.users)} />
        <Stat label="Items" value={num(system.items)} />
        <Stat label="Interactions" value={num(system.interactions)} />
      </div>

      <Panel title="Architecture" note="Offline batch build on the left, online request path below. Same diagram as the README.">
        <ArchitectureDiagram />
      </Panel>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div>
          <Panel title="Recall channels">
            <table>
              <thead>
                <tr><th>Channel</th><th>Status</th><th>Detail</th></tr>
              </thead>
              <tbody>
                {system.recall_sources.map((s) => {
                  const [ok, why] = system.recall_readiness[s] ?? [false, "not loaded"];
                  return (
                    <tr key={s}>
                      <td>
                        <Badge kind="src">{SOURCE_LABEL[s] ?? s}</Badge>
                      </td>
                      <td>{ok ? <Badge kind="cold">ready</Badge> : <Badge kind="tail">unavailable</Badge>}</td>
                      <td className="small muted">{why}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="panel-note">
              Content retrieval uses <code>faiss.IndexFlatIP</code> — exact inner-product search,
              not an approximate nearest-neighbour index.
            </div>
          </Panel>

          <Panel title="Item catalogue" note={`Popularity rule: ${system.item_metadata.bucket_rule} (training interactions only)`}>
            <Bars data={bucketData} />
            <div className="row mt" style={{ gap: 20 }}>
              <div>
                <div className="small faint">zero train signal</div>
                <strong>{num(system.item_metadata.zero_train_frequency_items)}</strong>
              </div>
              <div>
                <div className="small faint">simulated cold split</div>
                <strong>{system.item_metadata.has_cold_split ? "present" : "not in base split"}</strong>
              </div>
            </div>
            {system.exploration && (
              <div className="panel-note">{system.exploration.note}</div>
            )}
          </Panel>

          <Panel
            title="Rankers and offline metrics"
            note="Full-catalogue ranking on the held-out test split, mean over seeds. Model-quality numbers — not comparable to the two-stage pipeline."
          >
            <table>
              <thead>
                <tr><th>Ranker</th><th>Recall@20</th><th>NDCG@20</th><th>Coverage@20</th><th>seeds</th></tr>
              </thead>
              <tbody>
                {system.rankers.map((m) => (
                  <tr key={m.name}>
                    <td>
                      <div>{m.name}</div>
                      <div className="small faint">{m.description}</div>
                    </td>
                    <td><strong>{fmt(m.offline["Recall@20"])}</strong></td>
                    <td>{fmt(m.offline["NDCG@20"])}</td>
                    <td>{pct(m.offline["Coverage@20"])}</td>
                    <td>{m.offline["n_seeds"] ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="panel-note">
              Default ranker <code>{system.default_ranker}</code>, read from{" "}
              <code>results/tables/overall.csv</code>.
            </div>
          </Panel>
        </div>

        <div>
          {recall && recall.ks.length > 0 && (
            <Panel
              title="Recall quality vs candidate budget"
              note={`Test target, history masked, ${num(recall.num_users)} users. Read from results/tables/recall_eval.csv.`}
            >
              <LineChart
                xLabels={recall.ks.map((k) => `K=${k}`)}
                series={series}
                yLabel="Recall@K"
                xLabel="candidate budget"
              />
              <Legend series={series.map((s) => ({ name: s.name, color: s.color }))} />
              <div className="note mt">
                ItemCF is strongest at small budgets. Semantic recall starts lower but catches up
                as the budget grows, and the merged pool beats every single channel at every
                budget — the argument for multi-channel recall rather than tuning one channel
                harder.
              </div>
            </Panel>
          )}

          {tradeoff.length > 0 && (
            <Panel
              title="Two-stage trade-off"
              note="Same checkpoint and same user sample for the pipeline and the full-catalogue baseline, so the retention column is apples-to-apples."
            >
              <table>
                <thead>
                  <tr>
                    <th>Candidate budget</th>
                    <th>Candidate recall</th>
                    <th>Pipeline R@20</th>
                    <th>Retention</th>
                  </tr>
                </thead>
                <tbody>
                  {tradeoff.map((r) => (
                    <tr key={r.k}>
                      <td>{r.k}</td>
                      <td>{fmt(r.cand)}</td>
                      <td><strong>{fmt(r.rec)}</strong></td>
                      <td>{pct(r.retention, 1)}</td>
                    </tr>
                  ))}
                  {fullRecall !== null && (
                    <tr>
                      <td><em>full catalogue</em></td>
                      <td>—</td>
                      <td><strong>{fmt(fullRecall)}</strong></td>
                      <td>{pct(1, 0)}</td>
                    </tr>
                  )}
                </tbody>
              </table>
              <div className="panel-note">
                {evaluation?.notes?.pipeline}
              </div>
            </Panel>
          )}

          {latencyRows.length > 0 && (
            <Panel
              title="Serving latency by stage"
              note="CPU demo benchmark, median, on this machine. For relative comparison between stages, not a production SLA."
            >
              <table>
                <thead>
                  <tr>
                    <th>Candidates</th>
                    <th>Recall</th>
                    <th>Encode</th>
                    <th>Score</th>
                    <th>End to end</th>
                  </tr>
                </thead>
                <tbody>
                  {latencyRows.map((r) => (
                    <tr key={r.k}>
                      <td>{r.k}</td>
                      <td>{fmt(r.recall, 1)} ms</td>
                      <td>{fmt(r.encode, 2)} ms</td>
                      <td>{fmt(r.score, 2)} ms</td>
                      <td>{fmt(r.e2e, 1)} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="note mt">
                The ranking stage is dominated by encoding the user sequence, not by the
                candidate matmul: scoring 100 candidates and scoring all 19 738 cost almost the
                same. The two-stage split buys recall quality and headroom for a larger
                catalogue, not latency, at this scale.
              </div>
            </Panel>
          )}

          <Panel title="API">
            <table>
              <thead><tr><th>Endpoint</th><th>Returns</th></tr></thead>
              <tbody>
                <tr><td className="mono">GET /system</td><td className="small">dataset, rankers, recall readiness</td></tr>
                <tr><td className="mono">GET /models</td><td className="small">rankers + offline metrics</td></tr>
                <tr><td className="mono">GET /evaluation</td><td className="small">recall / pipeline / latency artifacts</td></tr>
                <tr><td className="mono">GET /users/&#123;u&#125;</td><td className="small">history with cold / bucket metadata</td></tr>
                <tr><td className="mono">GET /users/&#123;u&#125;/recall</td><td className="small">candidates + per-source trace</td></tr>
                <tr><td className="mono">GET /users/&#123;u&#125;/recommend</td><td className="small">served top-K</td></tr>
                <tr><td className="mono">GET /users/&#123;u&#125;/inspect</td><td className="small">full request trace</td></tr>
                <tr><td className="mono">GET /cold/summary</td><td className="small">cold-start experiment results</td></tr>
              </tbody>
            </table>
            <div className="panel-note">
              Single-process, CPU-only, unauthenticated, no caching: this is a local demo, not a
              deployment.
            </div>
          </Panel>
        </div>
      </div>

      <details className="dev">
        <summary>Developer details</summary>
        <pre>{`offline
  raw interactions -> preprocess -> train SASRec / MM-SASRec
                                 -> ItemCF neighbour index
                                 -> content embeddings + Faiss index

online
  request -> popular | itemcf | semantic recall
          -> merge / dedup / RRF (source trace kept)
          -> MM-SASRec ranking over the pool
          -> rerank (seen filter, dedup, zero-train exploration quota)
          -> top-K feed

evaluation
  full-catalogue ranking   results/tables/overall.csv
  recall@K by channel      results/tables/recall_eval.csv
  two-stage retention      results/tables/pipeline_tradeoff.csv
  latency by stage         results/tables/latency_benchmark.csv`}</pre>
      </details>
    </>
  );
}
