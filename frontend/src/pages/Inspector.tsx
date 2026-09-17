import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { InspectResponse, SystemResponse } from "../api/types";
import { Badge, Bars, BucketBadge, ColdBadge, ErrorBox, Panel, Stat, fmt, num } from "../components/common";
import { UserPicker } from "../components/UserPicker";

type SortKey = "merge_score" | "ranking_score" | "compare_score" | "score_delta" | "item_id";

export default function Inspector() {
  const [system, setSystem] = useState<SystemResponse | null>(null);
  const [userId, setUserId] = useState(7);
  const [ranker, setRanker] = useState("mm_concat");
  const [compare, setCompare] = useState("sasrec");
  const [data, setData] = useState<InspectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [sortKey, setSortKey] = useState<SortKey>("ranking_score");
  const [sortDesc, setSortDesc] = useState(true);

  useEffect(() => {
    api.system().then(setSystem).catch(setError);
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .inspect(userId, { ranker, compare, recall_k: 200, top_n: 40 })
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [userId, ranker, compare]);

  const rows = useMemo(() => {
    if (!data) return [];
    const r = [...data.top_candidates];
    r.sort((a, b) => {
      const av = (a[sortKey as keyof typeof a] ?? -1e9) as number;
      const bv = (b[sortKey as keyof typeof b] ?? -1e9) as number;
      return sortDesc ? bv - av : av - bv;
    });
    return r;
  }, [data, sortKey, sortDesc]);

  const toggleSort = (k: SortKey) => {
    if (k === sortKey) setSortDesc(!sortDesc);
    else {
      setSortKey(k);
      setSortDesc(true);
    }
  };

  const head = (label: string, k: SortKey) => (
    <th onClick={() => toggleSort(k)}>
      {label} {sortKey === k ? (sortDesc ? "▾" : "▴") : ""}
    </th>
  );

  return (
    <>
      <div className="page-head">
        <h1>Recommendation Inspector</h1>
        <p>
          The full request trace for one user: which channels recalled each candidate, how the pool
          was merged, what the ranker scored it, and what the multimodal model changed relative to
          the ID-only baseline.
        </p>
      </div>

      <ErrorBox error={error} />

      <Panel title="Request">
        <div className="row" style={{ gap: 18 }}>
          <UserPicker userId={userId} onChange={setUserId} maxUsers={system?.users ?? 100000} />
          <span className="spacer" />
          <label className="field">
            Ranker
            <select value={ranker} onChange={(e) => setRanker(e.target.value)}>
              {system?.rankers.map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </label>
          <label className="field">
            Compare against
            <select value={compare} onChange={(e) => setCompare(e.target.value)}>
              {system?.rankers.filter((m) => m.name !== ranker).map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </label>
        </div>
      </Panel>

      {loading && <div className="loading">loading trace…</div>}

      {data && !loading && (
        <>
          {/* ---------- pipeline ---------- */}
          <Panel title="Pipeline" note={`history mode: ${data.history_mode} · latency ${data.latency_ms.total} ms`}>
            <div className="pipe">
              <div className="stage">
                <div className="t">History</div>
                <div className="n">{data.history_length}</div>
                <div className="small faint">items</div>
              </div>
              <div className="arrow">→</div>
              <div className="stage">
                <div className="t">Recalled</div>
                <div className="n">{num(data.recall_summary.before_dedup)}</div>
                <div className="small faint">
                  {Object.entries(data.recall_summary.per_source ?? {})
                    .map(([k, v]) => `${k} ${v}`)
                    .join(" · ")}
                </div>
              </div>
              <div className="arrow">→</div>
              <div className="stage">
                <div className="t">After dedup</div>
                <div className="n">{num(data.recall_summary.after_dedup)}</div>
                <div className="small faint">{num(data.recall_summary.duplicates_removed)} merged</div>
              </div>
              <div className="arrow">→</div>
              <div className="stage">
                <div className="t">Ranked</div>
                <div className="n">{num(data.recall_summary.candidates_ranked)}</div>
                <div className="small faint">{ranker}</div>
              </div>
              <div className="arrow">→</div>
              <div className="stage">
                <div className="t">Served</div>
                <div className="n">{data.final_top_k.length}</div>
                <div className="small faint">
                  target {data.target_item !== null ? `#${data.target_item}` : "—"}
                </div>
              </div>
            </div>
          </Panel>

          <div className="grid cols-2" style={{ marginTop: 16 }}>
            <div>
              <Panel title="Recall channels">
                <Bars
                  data={Object.entries(data.recall_summary.per_source ?? {}).map(([k, v]) => ({
                    label: k,
                    value: v,
                    display: num(v),
                  }))}
                />
                <div className="mt">
                  <Bars
                    alt
                    data={[
                      { label: "before dedup", value: data.recall_summary.before_dedup, display: num(data.recall_summary.before_dedup) },
                      { label: "after dedup", value: data.recall_summary.after_dedup, display: num(data.recall_summary.after_dedup) },
                    ]}
                  />
                </div>
                <div className="panel-note">
                  Candidates per channel, then the merged pool. Duplicate items are merged, not
                  dropped: every contributing channel is kept on the candidate.
                </div>
              </Panel>

              <Panel title="History">
                <div className="hist">
                  {data.history_items?.map((h, i) => (
                    <div key={`${h.item_id}-${i}`} className="h" title={`train interactions: ${h.train_interactions}`}>
                      #{h.item_id} <BucketBadge bucket={h.popularity_bucket} />
                    </div>
                  ))}
                </div>
              </Panel>
            </div>

            <div>
              <Panel
                title={`What ${data.compare_ranker ?? "the baseline"} vs ${data.ranker} changed`}
                note="Same candidate pool, same user. Only the ranker differs, so any movement is attributable to the multimodal item representation."
              >
                <div className="grid cols-2" style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <div>
                    <div className="panel-title">Moved up by multimodal</div>
                    {data.moved_up_by_multimodal.length === 0 && <div className="muted small">none</div>}
                    {data.moved_up_by_multimodal.map((e) => (
                      <div className="kv" key={e.item_id}>
                        <span className="k">#{e.item_id}</span>
                        <span className="v">
                          <Badge kind="cold">+{e.position_delta}</Badge>{" "}
                          <span className="faint">Δ{fmt(e.score_delta, 3)}</span>
                        </span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <div className="panel-title">Moved down</div>
                    {data.moved_down_by_multimodal.length === 0 && <div className="muted small">none</div>}
                    {data.moved_down_by_multimodal.map((e) => (
                      <div className="kv" key={e.item_id}>
                        <span className="k">#{e.item_id}</span>
                        <span className="v">
                          <Badge kind="tail">−{e.position_delta}</Badge>{" "}
                          <span className="faint">Δ{fmt(e.score_delta, 3)}</span>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="panel-note">
                  Position deltas compare the ranking of the two models inside the same candidate
                  pool. They explain <em>what</em> the content features changed, which is the point
                  of this page.
                </div>
              </Panel>

              <Panel title={`Baseline top-${data.baseline_top.length} (${data.compare_ranker})`}>
                <table>
                  <thead>
                    <tr>
                      <th>Item</th>
                      <th>Base rank</th>
                      <th>Base score</th>
                      <th>MM score</th>
                      <th>Δ</th>
                      <th>Bucket</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.baseline_top.map((b) => (
                      <tr key={b.item_id}>
                        <td className="mono">#{b.item_id}</td>
                        <td>{b.baseline_rank}</td>
                        <td>{fmt(b.baseline_score, 3)}</td>
                        <td>{fmt(b.multimodal_score, 3)}</td>
                        <td style={{ color: b.delta >= 0 ? "var(--ok)" : "var(--warn)" }}>
                          {b.delta >= 0 ? "+" : ""}
                          {fmt(b.delta, 3)}
                        </td>
                        <td><BucketBadge bucket={b.popularity_bucket} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Panel>
            </div>
          </div>

          {/* ---------- candidate table ---------- */}
          <Panel
            title="Candidate table"
            note={`${rows.length} candidates ranked. Click a column header to sort. "Sources" shows every channel that recalled the item, with its rank inside that channel.`}
          >
            <div style={{ maxHeight: 520, overflow: "auto" }}>
              <table>
                <thead>
                  <tr>
                    {head("Item", "item_id")}
                    <th>Sources</th>
                    <th>Recall ranks</th>
                    {head("Merge", "merge_score")}
                    {head("Rank score", "ranking_score")}
                    {head("Baseline", "compare_score")}
                    {head("Δ", "score_delta")}
                    <th>Bucket</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((c) => (
                    <tr key={c.item_id}>
                      <td className="mono">#{c.item_id}</td>
                      <td>
                        {c.sources.map((s) => (
                          <Badge key={s} kind="src">{s}</Badge>
                        ))}
                      </td>
                      <td className="small muted">
                        {Object.entries(c.recall_rank ?? {})
                          .map(([k, v]) => `${k}:${v}`)
                          .join(" ")}
                      </td>
                      <td>{fmt(c.merge_score, 4)}</td>
                      <td>{fmt(c.ranking_score, 3)}</td>
                      <td className="muted">{fmt(c.compare_score ?? null, 3)}</td>
                      <td style={{ color: (c.score_delta ?? 0) >= 0 ? "var(--ok)" : "var(--warn)" }}>
                        {c.score_delta === null || c.score_delta === undefined
                          ? "—"
                          : `${c.score_delta >= 0 ? "+" : ""}${fmt(c.score_delta, 3)}`}
                      </td>
                      <td><BucketBadge bucket={c.popularity_bucket} /></td>
                      <td><ColdBadge cold={c.is_cold} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title={`Served top-${data.final_top_k.length}`}>
            <div className="grid cols-4">
              {data.final_top_k.map((r) => (
                <Stat
                  key={r.item_id}
                  label={`#${r.final_rank} · item ${r.item_id}`}
                  value={fmt(r.ranking_score, 3)}
                  sub={
                    <>
                      {r.sources.join(" + ")} · {r.popularity_bucket}
                      {r.is_cold ? " · cold" : ""}
                    </>
                  }
                />
              ))}
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
