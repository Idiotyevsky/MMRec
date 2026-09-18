import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { InspectResponse, MediaItem, MediaManifest } from "../api/types";
import { Badge, BucketBadge, ErrorBox, Panel, fmt, num } from "../components/common";
import { SOURCE_COLORS, SOURCE_LABEL } from "../components/PipelineFlow";

/** Real MicroLens videos rendered as a playable feed.
 *
 *  The overlay is the same recommendation trace the Inspector shows — nothing
 *  is recomputed or mocked here.  Titles, when present, are the official
 *  MicroLens catalogue titles resolved through the verified id mapping. */
export default function WatchFeed() {
  const [params] = useSearchParams();
  const userId = Number(params.get("user") ?? 68317);
  const ranker = params.get("ranker") ?? "mm_concat";

  const [manifest, setManifest] = useState<MediaManifest | null>(null);
  const [feed, setFeed] = useState<MediaItem[]>([]);
  const [trace, setTrace] = useState<InspectResponse | null>(null);
  const [index, setIndex] = useState(0);
  const [showWhy, setShowWhy] = useState(false);
  const [muted, setMuted] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    api.mediaManifest().then(setManifest).catch(setError);
    api.feed(userId).then((f) => setFeed(f.items)).catch(setError);
    api
      .inspect(userId, { ranker, compare: "sasrec", recall_k: 200, top_n: 60 })
      .then(setTrace)
      .catch(() => undefined); // the trace is optional; the feed still plays
  }, [userId, ranker]);

  const current = feed[index] ?? null;

  // the served item's own trace, straight from /inspect
  const detail = useMemo(() => {
    if (!trace || !current) return null;
    const cand = trace.top_candidates.find((c) => c.item_id === current.item_id);
    const served = trace.final_top_k.find((c) => c.item_id === current.item_id);
    const recall = cand?.source_trace ?? [];
    return {
      cand: cand ?? served ?? null,
      recall,
      recallRank: (name: string) => {
        const hit = recall.find((s) => s.name === name);
        if (hit) return hit.rank;
        const r = (cand ?? served)?.recall_rank?.[name];
        return r === undefined ? null : r;
      },
    };
  }, [trace, current]);

  const go = useCallback(
    (delta: number) => {
      setShowWhy(false);
      setIndex((i) => {
        const n = feed.length;
        if (n === 0) return 0;
        return (i + delta + n) % n;
      });
    },
    [feed.length],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown" || e.key === "ArrowRight" || e.key === "j") go(1);
      else if (e.key === "ArrowUp" || e.key === "ArrowLeft" || e.key === "k") go(-1);
      else if (e.key === "Escape") setShowWhy(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  // restart playback when the item changes
  useEffect(() => {
    const v = videoRef.current;
    if (v) {
      v.load();
      v.play().catch(() => undefined);
    }
  }, [current?.media_url]);

  const served = detail?.cand;
  const delta = served?.rank_delta ?? current?.rank_delta ?? null;

  return (
    <>
      <div className="page-head">
        <h1>Short-video Feed</h1>
        <p>
          The ranked recommendation list rendered as a playable feed.
          {manifest?.prepared
            ? " Videos are resolved from the official MicroLens media source by raw item id."
            : " Demo media not prepared."}
        </p>
      </div>

      <ErrorBox error={error} />

      {!manifest?.prepared && (
        <Panel title="Demo media not prepared">
          <div className="muted">
            Run <code>python scripts/prepare_media_demo.py</code> to download the videos for one
            user's recommendations, then reload this page.
          </div>
          {manifest?.hint && <div className="panel-note">{manifest.hint}</div>}
        </Panel>
      )}

      {manifest?.prepared && feed.length === 0 && (
        <Panel title="No playable items">
          <div className="muted">
            Media is prepared but none of this user's recommendations has a verified video.
            Try a different <code>user</code> in the URL.
          </div>
        </Panel>
      )}

      {current && manifest && (
        <div className="watch">
          <div className="stage">
            {current.available ? (
              <video
                ref={videoRef}
                className="player"
                src={current.media_url ?? undefined}
                autoPlay
                muted={muted}
                loop
                playsInline
                controls={false}
                onClick={() => setMuted((m) => !m)}
              />
            ) : (
              <div className="player unavailable">
                <div>
                  <strong>Media unavailable</strong>
                  <div className="small">Item #{current.item_id}</div>
                  <div className="small faint">Recommendation metadata remains available.</div>
                </div>
              </div>
            )}

            <div className="stage-bar">
              <button onClick={() => go(-1)} disabled={feed.length < 2}>↑ Previous</button>
              <span className="small faint">
                {index + 1} / {feed.length} playable
                {manifest.num_items && manifest.num_items !== feed.length
                  ? ` (of ${manifest.num_items} in the top-K)`
                  : ""}
              </span>
              <button onClick={() => go(1)} disabled={feed.length < 2}>Next ↓</button>
              <button onClick={() => setMuted((m) => !m)} title="toggle sound">
                {muted ? "unmute" : "mute"}
              </button>
            </div>
          </div>

          <aside className="why">
            <div className="card-head">
              <span className="rank">#{current.rank ?? index + 1}</span>
              <span className="row" style={{ gap: 4 }}>
                <BucketBadge bucket={(served?.popularity_bucket ?? current.popularity_bucket ?? "unknown") as never} />
                {current.is_zero_train_signal ? <Badge kind="cold">zero-train</Badge> : <Badge kind="plain">warm</Badge>}
              </span>
            </div>

            <div className="iid">Item #{current.item_id}</div>
            {current.title && <div className="title">{current.title}</div>}
            {!current.title && (
              <div className="small faint">
                No catalogue title available for this item.
              </div>
            )}

            <div className="rows">
              <div className="r">
                <span className="k">Recalled by</span>
                <span className="v">
                  {(served?.sources ?? current.sources).map((s) => (
                    <Badge key={s} kind="src">
                      {SOURCE_LABEL[s] ?? s}
                      {detail?.recallRank(s) != null ? ` #${detail.recallRank(s)}` : ""}
                    </Badge>
                  ))}
                </span>
              </div>
              <div className="r">
                <span className="k">Training interactions</span>
                <span className="v">{num(served?.train_interactions ?? current.train_interactions)}</span>
              </div>
              {delta !== null && delta !== undefined && (
                <div className="r">
                  <span className="k">ID-only → multimodal</span>
                  <span className="v">
                    <span className="muted">{served?.baseline_position ?? current.baseline_rank}</span>
                    {" → "}
                    <strong>{current.rank ?? index + 1}</strong>{" "}
                    <span style={{ color: delta > 0 ? "var(--ok)" : delta < 0 ? "var(--warn)" : undefined }}>
                      {delta > 0 ? `↑${delta}` : delta < 0 ? `↓${-delta}` : ""}
                    </span>
                  </span>
                </div>
              )}
            </div>

            <button className="why-btn" onClick={() => setShowWhy((v) => !v)}>
              {showWhy ? "Hide trace" : "Why this video?"}
            </button>

            {showWhy && detail && (
              <div className="trace-panel">
                <div className="trace">
                  {(["popular", "itemcf", "semantic"] as const).map((name) => {
                    const r = detail.recallRank(name);
                    return (
                      <div key={name} style={{ display: "contents" }}>
                        <div className={`trace-node${r === null ? " miss" : " hit"}`}>
                          <div className="lbl" style={{ color: SOURCE_COLORS[name] }}>
                            {SOURCE_LABEL[name].toUpperCase()}
                          </div>
                          <div className="val">{r === null ? "—" : `#${r}`}</div>
                        </div>
                        <div className="trace-arrow">→</div>
                      </div>
                    );
                  })}
                  <div className="trace-node">
                    <div className="lbl">MERGE</div>
                    <div className="val">{fmt(detail.cand?.merge_score ?? null, 4)}</div>
                    <div className="note">RRF</div>
                  </div>
                </div>

                <div className="rows" style={{ marginTop: 12 }}>
                  <div className="r">
                    <span className="k">SASRec (ID only)</span>
                    <span className="v">#{served?.baseline_position ?? "—"}</span>
                  </div>
                  <div className="r">
                    <span className="k">MM-SASRec Concat</span>
                    <span className="v">#{current.rank ?? index + 1}</span>
                  </div>
                  {delta !== null && delta !== undefined && delta !== 0 && (
                    <div className="r">
                      <span className="k">Movement</span>
                      <span className="v" style={{ color: delta > 0 ? "var(--ok)" : "var(--warn)" }}>
                        {delta > 0 ? `↑ ${delta} positions` : `↓ ${-delta} positions`}
                      </span>
                    </div>
                  )}
                  {current.official_video_id !== null && current.official_video_id !== undefined && (
                    <div className="r">
                      <span className="k">MicroLens video id</span>
                      <span className="v mono">{current.official_video_id}</span>
                    </div>
                  )}
                  {current.views !== null && current.views !== undefined && (
                    <div className="r">
                      <span className="k">Catalogue views</span>
                      <span className="v">{num(current.views)}</span>
                    </div>
                  )}
                </div>

                <div className="note mt">
                  Rank movement is the primary comparison; raw scores from two different models
                  are not on a common scale. All values come from the live{" "}
                  <code>/users/{userId}/inspect</code> response.
                </div>
              </div>
            )}

            <div className="modalities">
              <div className="panel-title" style={{ marginBottom: 6 }}>Ranking model</div>
              <div className="small muted">MM-SASRec Concat</div>
              <div className="row" style={{ gap: 6, marginTop: 6 }}>
                <Badge kind="cold">ID ✓</Badge>
                <Badge kind="cold">Text ✓</Badge>
                <Badge kind="cold">Image ✓</Badge>
                <Badge kind="plain">Video feature ✗</Badge>
              </div>
              <div className="small faint" style={{ marginTop: 6 }}>
                The feed plays the raw video for the recommended item. The headline model ranks
                with ID + text + image features, not the video stream.
              </div>
            </div>

            {manifest.verified_mapping && (
              <div className="small faint" style={{ marginTop: 10 }}>
                Media resolved by a verified id mapping —{" "}
                <a href="/system">see System</a>.
              </div>
            )}
          </aside>
        </div>
      )}
    </>
  );
}
