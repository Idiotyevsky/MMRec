import type { Bucket } from "../api/types";

export function Badge({ kind, children }: { kind?: string; children: React.ReactNode }) {
  return <span className={`badge ${kind ?? "plain"}`}>{children}</span>;
}

export function BucketBadge({ bucket }: { bucket: Bucket }) {
  const label = bucket === "unknown" ? "n/a" : bucket;
  return <Badge kind={bucket}>{label}</Badge>;
}

export function ColdBadge({ cold }: { cold: boolean }) {
  return cold ? <Badge kind="cold">cold</Badge> : <Badge kind="plain">warm</Badge>;
}

export function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
}) {
  return (
    <div className="stat">
      <div className="k">{label}</div>
      <div className="v">{value}</div>
      {sub !== undefined && <div className="s">{sub}</div>}
    </div>
  );
}

export function Panel({
  title,
  children,
  note,
  right,
}: {
  title?: string;
  children: React.ReactNode;
  note?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <section className="panel">
      {(title || right) && (
        <div className="row between">
          {title && <h2 className="panel-title" style={{ margin: 0 }}>{title}</h2>}
          {right}
        </div>
      )}
      <div style={{ marginTop: title ? 12 : 0 }}>{children}</div>
      {note && <div className="panel-note">{note}</div>}
    </section>
  );
}

export function Bars({
  data,
  alt,
}: {
  data: { label: string; value: number; display?: string }[];
  alt?: boolean;
}) {
  const max = Math.max(...data.map((d) => d.value), 1e-9);
  return (
    <div className="bars">
      {data.map((d) => (
        <div className="bar-row" key={d.label}>
          <span className="muted">{d.label}</span>
          <div className="bar-track">
            <div className={`bar-fill${alt ? " alt" : ""}`} style={{ width: `${(d.value / max) * 100}%` }} />
          </div>
          <span className="bar-val">{d.display ?? d.value}</span>
        </div>
      ))}
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="error">
      <strong>Request failed.</strong> {msg}
      <div className="small" style={{ marginTop: 4 }}>
        Is the backend running? <code>python -m src.serving.app</code>
      </div>
    </div>
  );
}

export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function pct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function num(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString();
}
