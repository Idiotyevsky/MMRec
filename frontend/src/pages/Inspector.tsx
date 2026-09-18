import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { InspectResponse, SystemResponse } from "../api/types";
import { Badge, Bars, BucketBadge, ErrorBox, Panel, fmt, num } from "../components/common";
import { SOURCE_COLORS, SOURCE_LABEL } from "../components/PipelineFlow";
import { UserPicker } from "../components/UserPicker";

type SortKey =
  | "merge_score" | "ranking_score" | "compare_score" | "score_delta"
  | "item_id" | "baseline_position" | "rank_delta";

/** One node of the traced path for a single item. */
function TraceNode({
  label, value, note, state,
}: {
  label: string; value: string; note?: string; state?: "hit" | "miss";
}) {
  return (
    <div className={`trace-node${state ? ` ${state}` : ""}`}>
      <div className="lbl">{label}</div>
      <div className="val">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export default function Inspector() {
  const [system, setSystem] = useState<SystemResponse | null>(null);
  const [userId, setUserId] = useState(7);
  const [ranker, setRanker] = useState("mm_concat");
  const [compare, setCompare] = useState("sasrec");
  const [data, setData] = useState<InspectResponse | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
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
      .inspect(userId, { ranker, compare, recall_k: 200, top_n: 60 })
      .then((res) => {
        setData(res);
        setSelected(res.final_top_k[0]?.item_id ?? res.top_candidates[0]?.item_id ?? null);
      })
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
    else { setSortKey(k); setSortDesc(true); }
  };
  const head = (label: string, k: SortKey) => (
    <th onClick={() => toggleSort(k)}>
      {label} {sortKey === k ? (sortDesc ? "▾" : "▴") : ""}
    </th>
  );

  const picked = useMemo(() => {
    if (!data || selected === null) return null;
    const entry = data.top_candidates.find((e) => e.item_id === selected);
    if (!entry) return null;
    // Use the candidate's own source trace, not `recall_candidates` (which is
    // only a preview of the first N pool entries and would show "—" for any
    // item outside it even though the channel did recall it).
    const trace = entry.source_trace ?? [];
    const rankOf = (name: string): number | null => {
      const t = trace.find((s) => s.name === name);
      if (t) return t.rank;
      const r = entry.recall_rank?.[name];
      return r === undefined ? null : r;
    };
    const finalRank = data.final_top_k.findIndex((e) => e.item_id === selected) + 1;
    return { entry, rankOf, finalRank: finalRank > 0 ? finalRank : null };
  }, [data, selected]);

  return (
    <>
      <div className="page-head">
        <h1>Recommendation Inspector</h1>
        <p>
          Why is this item recommended? Pick any candidate and follow it from the recall
          channels through the merge and both rankers to its final position.
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
              {system?.rankers.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
          </label>
          <label className="field">
            Compare against
            <select value={compare} onChange={(e) => setCompare(e.target.value)}>
              <option value="">none</option>
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
          {/* ---------- trace-first view ---------- */}
          <Panel
            title="Recommendation trace"
            right={
              <span className="small faint">
                {data.recall_summary.candidates_ranked} candidates ranked
              </span>
            }
          >
            <div className="row" style={{ gap: 6, marginBottom: 14 }}>
              <span className="small muted">Pick a served item:</span>
              {(() => {
                // the served items, plus whatever is selected (which may be a
                // candidate that moved but did not make the final top-K)
                const ids = data.final_top_k.slice(0, 12).map((e) => e.item_id);
                if (selected !== null && !ids.includes(selected)) ids.unshift(selected);
                return ids.map((id) => (
                  <button
                    key={id}
                    className={selected === id ? "primary" : ""}
                    onClick={() => setSelected(id)}
                  >
                    #{id}
                  </button>
                ));
              })()}
            </div>

            {picked && (
              <>
                <div className="trace">
                  {(["popular", "itemcf", "semantic"] as const).map((name) => {
                    const r = picked.rankOf(name);
                    return (
                      <div key={name} style={{ display: "contents" }}>
                        <TraceNode
                          label={name.toUpperCase()}
                          value={r === null ? "—" : `#${r}`}
                          state={r === null ? "miss" : "hit"}
                        />
                        <div className="trace-arrow">→</div>
                      </div>
                    );
                  })}
                  <TraceNode label="MERGE" value={fmt(picked.entry.merge_score, 4)} note="RRF score" />
                  <div className="trace-arrow">→</div>
                  <TraceNode
                    label={`${(data.compare_ranker ?? "baseline").toUpperCase()}`}
                    value={picked.entry.baseline_position ? `#${picked.entry.baseline_position}` : "—"}
                    note={picked.entry.compare_score !== null && picked.entry.compare_score !== undefined
                      ? `score ${fmt(picked.entry.compare_score, 3)}` : undefined}
                  />
                  <div className="trace-arrow">→</div>
                  <TraceNode
                    label={`${data.ranker.toUpperCase()}`}
                    value={`#${rows.findIndex((r) => r.item_id === picked.entry.item_id) + 1}`}
                    note={`score ${fmt(picked.entry.ranking_score, 3)}`}
                    state="hit"
                  />
                  <div className="trace-arrow">→</div>
                  <TraceNode
                    label="FINAL"
                    value={picked.finalRank ? `#${picked.finalRank}` : "—"}
                    note={picked.finalRank ? "served" : "in pool, not served"}
                    state={picked.finalRank ? "hit" : "miss"}
                  />
                </div>

                <div className="grid cols-4" style={{ marginTop: 16 }}>
                  <div className="stat">
                    <div className="k">Rank movement</div>
                    <div className="v" style={{
                      color: (picked.entry.rank_delta ?? 0) > 0 ? "var(--ok)"
                        : (picked.entry.rank_delta ?? 0) < 0 ? "var(--warn)" : undefined,
                    }}>
                      {picked.entry.rank_delta === null || picked.entry.rank_delta === undefined
                        ? "—"
                        : `${picked.entry.baseline_position} → ${rows.findIndex((r) => r.item_id === picked.entry.item_id) + 1}`}
                    </div>
                    <div className="s">
                      {picked.entry.rank_delta === null || picked.entry.rank_delta === undefined
                        ? "no baseline selected"
                        : picked.entry.rank_delta > 0
                          ? `↑ ${picked.entry.rank_delta} positions with multimodal`
                          : picked.entry.rank_delta < 0
                            ? `↓ ${-picked.entry.rank_delta} positions with multimodal`
                            : "unchanged"}
                    </div>
                  </div>
                  <div className="stat">
                    <div className="k">Recall sources</div>
                    <div className="v" style={{ fontSize: 15 }}>
                      {picked.entry.sources.map((s) => (
                        <Badge key={s} kind="src">{SOURCE_LABEL[s] ?? s}</Badge>
                      ))}
                    </div>
                    <div className="s">
                      {picked.entry.sources.length > 1
                        ? "found by several channels"
                        : "found by a single channel"}
                    </div>
                  </div>
                  <div className="stat">
                    <div className="k">Popularity</div>
                    <div className="v" style={{ fontSize: 17 }}>
                      <BucketBadge bucket={picked.entry.popularity_bucket} />
                    </div>
                    <div className="s">{num(picked.entry.train_interactions)} training interactions</div>
                  </div>
                  <div className="stat">
                    <div className="k">Status</div>
                    <div className="v" style={{ fontSize: 17 }}>
                      {picked.entry.is_zero_train_signal
                        ? <Badge kind="cold">zero-train</Badge>
                        : <Badge kind="plain">warm</Badge>}
                    </div>
                    <div className="s">
                      {picked.entry.is_zero_train_signal
                        ? "no collaborative signal available"
                        : "collaborative signal available"}
                    </div>
                  </div>
                </div>

                <div className="note mt">
                  Raw scores from two different models are not on a common scale, so the
                  primary comparison here is <strong>rank movement inside the same candidate
                  pool</strong>. Score deltas are shown for completeness only.
                </div>
              </>
            )}
          </Panel>

          {/* ---------- pipeline stages ---------- */}
          <div className="grid cols-2" style={{ marginTop: 16 }}>
            <div>
              <Panel title="Recall channels">
                <Bars
                  data={data.recall_channels.map((c) => ({
                    label: SOURCE_LABEL[c.name] ?? c.name,
                    value: c.recalled,
                    display: num(c.recalled),
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
                  Duplicates are merged, not dropped: every channel that found an item stays
                  on its source trace.
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
                note="Same candidate pool, same user, same checkpoint family. Only the item representation differs, so any movement is attributable to it."
              >
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                  <div>
                    <div className="panel-title">Moved up by multimodal</div>
                    {data.moved_up_by_multimodal.length === 0 && <div className="muted small">none</div>}
                    {data.moved_up_by_multimodal.map((e) => (
                      <div className="kv" key={e.item_id}
                           onClick={() => setSelected(e.item_id)}
                           style={{ cursor: "pointer" }}
                           title="show this item's trace">
                        <span className="k mono">#{e.item_id}</span>
                        <span className="v">
                          <Badge kind="cold">↑{e.position_delta}</Badge>{" "}
                          <span className="faint">{e.sources.map((s) => SOURCE_LABEL[s] ?? s).join("+")}</span>
                        </span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <div className="panel-title">Moved down</div>
                    {data.moved_down_by_multimodal.length === 0 && <div className="muted small">none</div>}
                    {data.moved_down_by_multimodal.map((e) => (
                      <div className="kv" key={e.item_id}
                           onClick={() => setSelected(e.item_id)}
                           style={{ cursor: "pointer" }}
                           title="show this item's trace">
                        <span className="k mono">#{e.item_id}</span>
                        <span className="v">
                          <Badge kind="tail">↓{e.position_delta}</Badge>{" "}
                          <span className="faint">{e.sources.map((s) => SOURCE_LABEL[s] ?? s).join("+")}</span>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </Panel>

              <Panel title={`Baseline top-${data.baseline_top.length} (${data.compare_ranker})`}>
                <table>
                  <thead>
                    <tr>
                      <th>Item</th><th>Base rank</th><th>Base score</th>
                      <th>MM score</th><th>Bucket</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.baseline_top.slice(0, 12).map((b) => (
                      <tr key={b.item_id}>
                        <td className="mono">#{b.item_id}</td>
                        <td>{b.baseline_rank}</td>
                        <td>{fmt(b.baseline_score, 3)}</td>
                        <td>{fmt(b.multimodal_score, 3)}</td>
                        <td><BucketBadge bucket={b.popularity_bucket} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Panel>
            </div>
          </div>

          {/* ---------- full candidate table ---------- */}
          <Panel
            title="Candidate table"
            note={`${rows.length} candidates ranked. Click a header to sort. "Sources" shows every channel that recalled the item, with its rank inside that channel.`}
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
                    {head("Base pos", "baseline_position")}
                    {head("Δ rank", "rank_delta")}
                    <th>Bucket</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((c) => (
                    <tr key={c.item_id}
                        onClick={() => setSelected(c.item_id)}
                        style={{ cursor: "pointer" }}>
                      <td className="mono">#{c.item_id}</td>
                      <td>
                        {c.sources.map((s) => (
                          <span key={s} className="badge src"
                                style={{ borderLeft: `3px solid ${SOURCE_COLORS[s] ?? "#999"}` }}>
                            {SOURCE_LABEL[s] ?? s}
                          </span>
                        ))}
                      </td>
                      <td className="small muted">
                        {Object.entries(c.recall_rank ?? {}).map(([k, v]) => `${k}:${v}`).join(" ")}
                      </td>
                      <td>{fmt(c.merge_score ?? null, 4)}</td>
                      <td>{fmt(c.ranking_score, 3)}</td>
                      <td className="muted">{fmt(c.compare_score ?? null, 3)}</td>
                      <td className="muted">{c.baseline_position ?? "—"}</td>
                      <td style={{
                        color: (c.rank_delta ?? 0) > 0 ? "var(--ok)"
                          : (c.rank_delta ?? 0) < 0 ? "var(--warn)" : undefined,
                      }}>
                        {c.rank_delta === null || c.rank_delta === undefined
                          ? "—"
                          : `${c.rank_delta > 0 ? "+" : ""}${c.rank_delta}`}
                      </td>
                      <td><BucketBadge bucket={c.popularity_bucket} /></td>
                      <td>{c.is_zero_train_signal ? <Badge kind="cold">zero-train</Badge> : <Badge kind="plain">warm</Badge>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
