import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { SystemResponse } from "../api/types";
import { Badge, Bars, ErrorBox, Panel, Stat, fmt, num, pct } from "../components/common";

export default function System() {
  const [system, setSystem] = useState<SystemResponse | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api.system().then(setSystem).catch(setError);
  }, []);

  if (error) return <ErrorBox error={error} />;
  if (!system) return <div className="loading">loading system info…</div>;

  const buckets = system.item_metadata.bucket_counts ?? {};
  const bucketData = ["head", "middle", "tail", "cold"]
    .filter((b) => (buckets[b] ?? 0) > 0)
    .map((b) => ({ label: b, value: buckets[b], display: num(buckets[b]) }));

  return (
    <>
      <div className="page-head">
        <h1>System</h1>
        <p>
          Dataset, recall channels and rankers currently loaded by the serving process. All offline
          metrics are read from <code>results/tables/</code>; nothing here is hard-coded in the UI.
        </p>
      </div>

      <div className="grid cols-4">
        <Stat label="Dataset" value={system.dataset} sub={`history mode: ${system.history_mode}`} />
        <Stat label="Users" value={num(system.users)} />
        <Stat label="Items" value={num(system.items)} />
        <Stat label="Interactions" value={num(system.interactions)} />
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div>
          <Panel title="Recall channels">
            <table>
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Status</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {system.recall_sources.map((s) => {
                  const [ok, why] = system.recall_readiness[s] ?? [false, "not loaded"];
                  return (
                    <tr key={s}>
                      <td><Badge kind="src">{s}</Badge></td>
                      <td>{ok ? <Badge kind="cold">ready</Badge> : <Badge kind="tail">unavailable</Badge>}</td>
                      <td className="small muted">{why}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="panel-note">
              Content retrieval uses <code>faiss.IndexFlatIP</code> — exact inner-product search, not
              an approximate nearest-neighbour index.
            </div>
          </Panel>

          <Panel title="Item catalogue" note={`Popularity rule: ${system.item_metadata.bucket_rule} (training interactions only)`}>
            <Bars data={bucketData} />
            <div className="row mt" style={{ gap: 20 }}>
              <div>
                <div className="small faint">zero train frequency</div>
                <strong>{num(system.item_metadata.zero_train_frequency_items)}</strong>
              </div>
              <div>
                <div className="small faint">cold split present</div>
                <strong>{system.item_metadata.has_cold_split ? "yes" : "no (base dataset)"}</strong>
              </div>
            </div>
          </Panel>
        </div>

        <div>
          <Panel
            title="Rankers and offline metrics"
            note="Full-catalogue ranking on the held-out test split. These are model-quality numbers and are not comparable to the two-stage pipeline."
          >
            <table>
              <thead>
                <tr>
                  <th>Ranker</th>
                  <th>Recall@20</th>
                  <th>NDCG@20</th>
                  <th>Coverage@20</th>
                  <th>seeds</th>
                </tr>
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
              Default ranker: <code>{system.default_ranker}</code>. Read from{" "}
              <code>results/tables/overall.csv</code>.
            </div>
          </Panel>

          <Panel title="Architecture">
            <pre className="mono small" style={{ margin: 0, overflowX: "auto", lineHeight: 1.6 }}>
{`offline
  raw interactions ─► preprocess ─► train ─► SASRec / MM-SASRec
                                    │
                                    ├─► ItemCF neighbour index
                                    └─► content embeddings + Faiss index

online (this process)
  user request
      │
      ▼
  popular ─┐
  itemcf  ─┼─► merge / dedup ─► ranker ─► rerank ─► top-K feed
  semantic─┘   (source trace)   (multimodal) (cold quota)`}
            </pre>
          </Panel>
        </div>
      </div>
    </>
  );
}
