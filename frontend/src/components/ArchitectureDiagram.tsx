/** Offline/online architecture as SVG. Same story as the README diagram. */

const BOX = { fill: "#ffffff", stroke: "#e2e5ea" } as const;
const ACCENT = { fill: "#eaf1f8", stroke: "#2f5d8a" } as const;

function Node({
  x, y, w, h, label, sub, accent,
}: {
  x: number; y: number; w: number; h: number;
  label: string; sub?: string; accent?: boolean;
}) {
  const s = accent ? ACCENT : BOX;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={7} fill={s.fill} stroke={s.stroke} strokeWidth={1} />
      <text x={x + w / 2} y={y + (sub ? h / 2 - 3 : h / 2 + 4)} textAnchor="middle"
            fontSize={12.5} fontWeight={600} fill="#1c2024">{label}</text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 13} textAnchor="middle" fontSize={10.5} fill="#667085">
          {sub}
        </text>
      )}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#cdd3db" strokeWidth={1.4}
               markerEnd="url(#arrowhead)" />;
}

export function ArchitectureDiagram() {
  return (
    <svg className="diagram" viewBox="0 0 760 470" role="img">
      <defs>
        <marker id="arrowhead" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <polygon points="0 0, 7 3.5, 0 7" fill="#cdd3db" />
        </marker>
      </defs>

      {/* ---- offline ---- */}
      <text x={16} y={22} fontSize={11} fontWeight={700} fill="#98a2b3" letterSpacing="0.08em">
        OFFLINE · BATCH
      </text>

      <Node x={16} y={36} w={140} h={44} label="Interactions" sub="MicroLens-100K" />
      <Arrow x1={156} y1={58} x2={196} y2={58} />
      <Node x={198} y={36} w={150} h={44} label="Preprocess" sub="chronological split" />
      <Arrow x1={348} y1={58} x2={388} y2={58} />

      <Node x={390} y={16} w={170} h={40} label="Train rankers" sub="SASRec / MM-SASRec" accent />
      <Node x={390} y={64} w={170} h={40} label="Build ItemCF index" sub="cosine co-occurrence" />
      <Node x={390} y={112} w={170} h={40} label="Build content index" sub="text + image, exact IP" />

      <Node x={590} y={64} w={150} h={40} label="artifacts/" sub="indices + checkpoints" />

      <line x1={560} y1={36} x2={585} y2={36} stroke="#cdd3db" strokeWidth={1.4} />
      <line x1={560} y1={84} x2={585} y2={84} stroke="#cdd3db" strokeWidth={1.4} />
      <line x1={560} y1={132} x2={585} y2={132} stroke="#cdd3db" strokeWidth={1.4} />

      <line x1={16} y1={196} x2={744} y2={196} stroke="#e2e5ea" strokeWidth={1} />

      {/* ---- online ---- */}
      <text x={16} y={220} fontSize={11} fontWeight={700} fill="#98a2b3" letterSpacing="0.08em">
        ONLINE · SERVING
      </text>

      <Node x={16} y={236} w={150} h={44} label="User request" sub="history + user id" />
      <Arrow x1={91} y1={280} x2={91} y2={312} />

      <Node x={16} y={314} w={150} h={36} label="Popular recall" />
      <Node x={16} y={356} w={150} h={36} label="ItemCF recall" />
      <Node x={16} y={398} w={150} h={36} label="Semantic recall" />

      <line x1={166} y1={332} x2={232} y2={352} stroke="#cdd3db" strokeWidth={1.4} />
      <line x1={166} y1={374} x2={232} y2={374} stroke="#cdd3db" strokeWidth={1.4} />
      <line x1={166} y1={416} x2={232} y2={396} stroke="#cdd3db" strokeWidth={1.4} />

      <Node x={234} y={352} w={152} h={44} label="Candidate merge" sub="dedup + RRF" accent />
      <Arrow x1={386} y1={374} x2={420} y2={374} />
      <Node x={422} y={352} w={152} h={44} label="MM-SASRec" sub="multimodal ranking" accent />
      <Arrow x1={574} y1={374} x2={608} y2={374} />
      <Node x={610} y={352} w={130} h={44} label="Rerank" sub="exploration quota" />
      <Arrow x1={675} y1={396} x2={675} y2={428} />
      <Node x={610} y={430} w={130} h={30} label="Top-K feed" />
    </svg>
  );
}
