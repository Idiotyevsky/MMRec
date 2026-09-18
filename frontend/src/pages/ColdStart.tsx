import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ColdItem, ColdSummary } from "../api/types";
import { Badge, BucketBadge, ErrorBox, Panel, Stat, fmt, num, pct } from "../components/common";

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
      .then((list) => { setItems(list); setSelected(list[0] ?? null); })
      .catch(setError);
  }, [seed]);

  const byTag = (tag: string) => summary?.experiments.find((e) => e.model === tag);
  const idOnly = byTag("cold_sasrec");
  const contentOnly = byTag("cold_content_only");
  const gated = byTag("cold_mm_gated");
  const rand = byTag("cold_random");

  const coldOnly = (e?: ColdSummary["experiments"][number]) => e?.["cold_only_recall@20"] ?? null;
  const coldFull = (e?: ColdSummary["experiments"][number]) => e?.["cold_recall@20"] ?? null;

  // the honest headline: content separates cold items well, but not against warm ones
  const best = Math.max(coldOnly(contentOnly) ?? 0, coldOnly(gated) ?? 0, 1e-9);
  const worst = coldFull(contentOnly) ?? 0;

  return (
    <>
      <div className="page-head">
        <h1>Cold Start Explorer</h1>
        <p>
          New videos have little or no interaction history. ShortRec uses content features to
          give them an initial recommendation signal. This page shows what that signal can and
          cannot do.
        </p>
      </div>

      <ErrorBox error={error} />

      <div className="grid cols-4" style={{ marginTop: 16 }}>
        <Stat label="Simulated cold items" value={num(summary?.num_cold_items)}
              sub="controlled benchmark, 10% of the catalogue" />
        <Stat label="Cold eval targets" value={num(summary?.num_users_with_cold_target)}
              sub="cold interactions to predict" />
        <Stat label="Serving zero-train items" value={num(summary?.serving_zero_train_items)}
              sub="base split, no training signal" />
        <Stat label="ID-only ColdOnly R@20" value={pct(coldOnly(idOnly))}
              sub="cannot rank cold items at all" />
      </div>

      {/* ---------- the headline comparison ---------- */}
      <Panel title="Same item, same zero interactions, two different signals">
        <div className="vs">
          <div className="vs-card bad">
            <h3>ID-only SASRec</h3>
            <div className="kv"><span className="k">Training interactions</span><span className="v"><strong>0</strong></span></div>
            <div className="kv"><span className="k">Collaborative signal</span><span className="v"><Badge kind="tail">none</Badge></span></div>
            <div className="big" style={{ marginTop: 12 }}>{pct(coldOnly(idOnly))}</div>
            <div className="cap">ColdOnly Recall@20 — the model has nothing to rank with</div>
          </div>
          <div className="vs-mid">VS</div>
          <div className="vs-card good">
            <h3>Text + image content</h3>
            <div className="kv"><span className="k">Training interactions</span><span className="v"><strong>0</strong></span></div>
            <div className="kv"><span className="k">Content signal</span><span className="v"><Badge kind="cold">available</Badge></span></div>
            <div className="big" style={{ marginTop: 12 }}>{pct(coldOnly(contentOnly))}</div>
            <div className="cap">ColdOnly Recall@20 — content alone separates cold items</div>
          </div>
        </div>
        <div className="panel-note">
          ColdOnly ranks items <em>within the cold catalogue</em>, for users whose next item is
          cold. Numbers are read from <code>results/tables/cold_start.csv</code>.
        </div>
      </Panel>

      {/* ---------- the honest limitation ---------- */}
      <Panel title="But: content does not fix cold/warm calibration">
        <div className="bars">
          <div className="bar-row">
            <span className="muted">Cold vs cold</span>
            <div className="bar-track"><div className="bar-fill" style={{ width: "100%" }} /></div>
            <span className="bar-val">{pct(best)}</span>
          </div>
          <div className="bar-row">
            <span className="muted">Cold vs full catalogue</span>
            <div className="bar-track">
              <div className="bar-fill alt" style={{ width: `${Math.max((worst / best) * 100, 1)}%` }} />
            </div>
            <span className="bar-val">{pct(worst)}</span>
          </div>
        </div>
        <div className="note mt">
          Content features can tell cold items apart from each other. They still lose to warm
          items in a shared ranking, because the two score distributions are not calibrated
          against each other. The gap between the two bars <em>is</em> the cold-start problem —
          it is a real limitation, not a modelling detail.
        </div>
        <table style={{ marginTop: 14 }}>
          <thead>
            <tr>
              <th>Model</th>
              <th>ColdOnly Recall@20</th>
              <th>Full-catalogue cold Recall@20</th>
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
        <div className="panel-note">
          The serving layer's zero-train exploration quota is the mitigation: it reserves
          exposure slots instead of pretending the scores are comparable.
        </div>
      </Panel>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div>
          <Panel
            title="Cold item browser"
            right={<button onClick={() => setSeed((s) => s + 1)}>Sample again</button>}
            note="Simulated cold items had every training interaction removed by a fixed seed. Their content features are untouched."
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
                <span className="v"><Badge kind="cold">simulated cold</Badge></span>
              </div>
              <div className="panel-title" style={{ marginTop: 14 }}>Content available</div>
              <div className="row" style={{ gap: 8 }}>
                {Object.entries(selected.content_available).map(([m, ok]) => (
                  <Badge key={m} kind={ok ? "cold" : "plain"}>{m} {ok ? "✓" : "✗"}</Badge>
                ))}
              </div>
            </Panel>
          )}
        </div>

        <div>
          {selected && selected.content_similar.length > 0 && (
            <Panel
              title="Content-similar items"
              note="Nearest neighbours in the raw text+image feature space. These are feature-space neighbours, not catalogue metadata — MicroLens-100K ships no titles."
            >
              <table>
                <thead>
                  <tr><th>Item</th><th>Similarity</th><th>Bucket</th><th>Train interactions</th></tr>
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

          <Panel title="Two different notions of cold">
            <div className="kv" style={{ alignItems: "flex-start" }}>
              <span className="k">Simulated cold<br /><span className="faint small">benchmark</span></span>
              <span className="v small" style={{ maxWidth: 380, textAlign: "left" }}>
                A fixed-seed subset of items with <em>all</em> training interactions removed
                (<code>data/processed/cold10</code>). This is the controlled experiment reported
                above and in the README.
              </span>
            </div>
            <div className="kv" style={{ alignItems: "flex-start", marginTop: 10 }}>
              <span className="k">Zero-train signal<br /><span className="faint small">serving</span></span>
              <span className="v small" style={{ maxWidth: 380, textAlign: "left" }}>
                An item that simply has zero observed training interactions in the base split
                ({num(summary?.serving_zero_train_items)} items). This is what a genuinely new
                video looks like to the model, and it drives the exploration quota.
              </span>
            </div>
            <div className="panel-note">
              They are different things: one is a scientific control, the other is a serving
              condition. The API and the UI keep them separate.
            </div>
          </Panel>

          <Panel title="Why this happens">
            <ul className="small" style={{ margin: 0, paddingLeft: 18, color: "var(--text-muted)" }}>
              <li>Cold items have zero training interactions, so their ID embedding carries no signal.</li>
              <li>The protocol zeroes the ID component for cold items in <em>every</em> model, so only content can help.</li>
              <li>Content recall ranks cold items against each other well, but not against popular items.</li>
              <li>The exploration quota reserves exposure instead of pretending the ranking score is comparable.</li>
            </ul>
          </Panel>
        </div>
      </div>
    </>
  );
}
