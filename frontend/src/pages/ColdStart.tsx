import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ColdItem, ColdSummary } from "../api/types";
import { Badge, Bars, BucketBadge, ErrorBox, Panel, Stat, fmt, num, pct } from "../components/common";

export default function ColdStart() {
  const [summary, setSummary] = useState<ColdSummary | null>(null);
  const [items, setItems] = useState<ColdItem[]>([]);
  const [selected, setSelected] = useState<ColdItem | null>(null);
  const [seed, setSeed] = useState(0);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api.coldSummary().then(setSummary).catch(setError);
  }, []);

  useEffect(() => {
    api
      .coldItems(12, seed)
      .then((list) => {
        setItems(list);
        setSelected(list[0] ?? null);
      })
      .catch(setError);
  }, [seed]);

  const byTag = (tag: string) => summary?.experiments.find((e) => e.model === tag);
  const idOnly = byTag("cold_sasrec");
  const contentOnly = byTag("cold_content_only");
  const gated = byTag("cold_mm_gated");
  const rand = byTag("cold_random");

  // property names contain '@', so bracket access is required
  const coldOnly = (e?: ColdSummary["experiments"][number]) => e?.["cold_only_recall@20"] ?? null;
  const coldFull = (e?: ColdSummary["experiments"][number]) => e?.["cold_recall@20"] ?? null;

  const chart = [
    { label: "random", value: coldOnly(rand) ?? 0, display: pct(coldOnly(rand)) },
    { label: "ID-only", value: coldOnly(idOnly) ?? 0, display: pct(coldOnly(idOnly)) },
    { label: "content-only", value: coldOnly(contentOnly) ?? 0, display: pct(coldOnly(contentOnly)) },
    { label: "MM gated", value: coldOnly(gated) ?? 0, display: pct(coldOnly(gated)) },
  ];

  return (
    <>
      <div className="page-head">
        <h1>Cold Start Explorer</h1>
        <p>
          New videos have little or no interaction history. ShortRec uses content features to give
          them an initial recommendation signal. This page shows what that signal can and cannot do.
        </p>
      </div>

      <ErrorBox error={error} />

      <div className="grid cols-4" style={{ marginTop: 16 }}>
        <Stat label="Cold items" value={num(summary?.num_cold_items)} sub="10% of the catalogue" />
        <Stat label="Cold val+test targets" value={num(summary?.num_users_with_cold_target)} sub="cold interactions to predict" />
        <Stat
          label="ID-only ColdOnly R@20"
          value={pct(coldOnly(idOnly))}
          sub="full ranking: also 0.00%"
        />
        <Stat
          label="Content-only ColdOnly R@20"
          value={pct(coldOnly(contentOnly))}
          sub="ranking among cold items only"
        />
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div>
          <Panel
            title="Cold item browser"
            right={<button onClick={() => setSeed((s) => s + 1)}>Sample again</button>}
            note="Cold items have every training interaction removed by the simulated protocol. Their content features are untouched."
          >
            <div className="row" style={{ gap: 6 }}>
              {items.map((it) => (
                <button
                  key={it.item_id}
                  className={selected?.item_id === it.item_id ? "primary" : ""}
                  onClick={() => setSelected(it)}
                >
                  #{it.item_id}
                </button>
              ))}
            </div>
          </Panel>

          {selected && (
            <Panel title={`Item #${selected.item_id}`}>
              <div className="kv">
                <span className="k">Training interactions</span>
                <span className="v"><strong>{selected.train_interactions}</strong></span>
              </div>
              <div className="kv">
                <span className="k">ID signal</span>
                <span className="v"><Badge kind="tail">unavailable</Badge></span>
              </div>
              <div className="kv">
                <span className="k">Status</span>
                <span className="v"><Badge kind="cold">cold</Badge></span>
              </div>
              <div className="panel-title" style={{ marginTop: 14 }}>Content available</div>
              <div className="row" style={{ gap: 8 }}>
                {Object.entries(selected.content_available).map(([m, ok]) => (
                  <Badge key={m} kind={ok ? "cold" : "plain"}>
                    {m} {ok ? "✓" : "✗"}
                  </Badge>
                ))}
              </div>
              <div className="panel-note">
                MicroLens-100K ships no titles or captions. Items are identified by their raw id and
                by their nearest neighbours in the raw text+image feature space.
              </div>
            </Panel>
          )}

          {selected && selected.content_similar.length > 0 && (
            <Panel
              title="Content-similar items"
              note="Nearest neighbours in the raw text+image feature space. These are feature-space neighbours, not catalogue metadata."
            >
              <table>
                <thead>
                  <tr>
                    <th>Item</th>
                    <th>Similarity</th>
                    <th>Bucket</th>
                    <th>Train interactions</th>
                  </tr>
                </thead>
                <tbody>
                  {selected.content_similar.map((s) => (
                    <tr key={s.item_id}>
                      <td className="mono">#{s.item_id}</td>
                      <td>{fmt(s.similarity, 3)}</td>
                      <td><BucketBadge bucket={s.popularity_bucket} /></td>
                      <td>{num(s.train_interactions)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          )}
        </div>

        <div>
          <Panel
            title="Cold-only ranking (content vs collaborative)"
            note="Ranking restricted to the cold catalogue, for users whose next item is cold. This measures how well a model can tell cold items apart."
          >
            <Bars data={chart} />
          </Panel>

          <Panel title="The honest part: full-catalogue ranking">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>ColdOnly R@20</th>
                  <th>Full-catalogue cold R@20</th>
                </tr>
              </thead>
              <tbody>
                {[rand, idOnly, contentOnly, gated].filter(Boolean).map((e) => (
                  <tr key={e!.model}>
                    <td>{e!.model}</td>
                    <td>{pct(coldOnly(e))}</td>
                    <td>{pct(coldFull(e))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="note mt">
              Content features can tell cold items apart, but they are still out-scored by warm items
              in the full catalogue. The gap between the two columns is the cold/warm score
              calibration problem — it is a real limitation, not a modelling detail.
            </div>
            <div className="panel-note">{summary?.note}</div>
          </Panel>

          <Panel title="Why this happens">
            <ul className="small" style={{ margin: 0, paddingLeft: 18, color: "var(--text-muted)" }}>
              <li>Cold items have zero training interactions, so their ID embedding carries no signal.</li>
              <li>
                The protocol zeroes the ID component for cold items in <em>every</em> model, so only
                content can help.
              </li>
              <li>Content recall ranks cold items against each other well, but not against popular items.</li>
              <li>
                The serving layer's cold exploration quota is the mitigation: it reserves exposure
                slots instead of pretending the ranking score is comparable.
              </li>
            </ul>
          </Panel>
        </div>
      </div>
    </>
  );
}
