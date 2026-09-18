/** Minimal SVG charts. Deliberately hand-rolled: the project should not pull in
 *  a charting library for three plots, and hand-rolled SVG keeps the styling
 *  consistent with the rest of the console. */

export interface Series {
  name: string;
  color: string;
  values: (number | null)[];
}

export function LineChart({
  xLabels,
  series,
  yLabel,
  yMax,
  width = 640,
  height = 260,
  xLabel,
}: {
  xLabels: (string | number)[];
  series: Series[];
  yLabel?: string;
  yMax?: number;
  width?: number;
  height?: number;
  xLabel?: string;
}) {
  const padL = 58, padR = 18, padT = 16, padB = 42;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null));
  const top = yMax ?? Math.max(...all, 1e-9) * 1.12;
  const n = xLabels.length;

  const px = (i: number) => padL + (n === 1 ? w / 2 : (i / (n - 1)) * w);
  const py = (v: number) => padT + h - (v / top) * h;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * top);

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img">
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={padL} x2={padL + w} y1={py(t)} y2={py(t)} stroke="#eceff2" strokeWidth={1} />
          <text x={padL - 8} y={py(t) + 4} textAnchor="end" fontSize={10.5} fill="#98a2b3">
            {top < 0.05 ? t.toFixed(3) : t.toFixed(2)}
          </text>
        </g>
      ))}

      {xLabels.map((lbl, i) => (
        <text key={i} x={px(i)} y={height - padB + 18} textAnchor="middle" fontSize={11} fill="#667085">
          {lbl}
        </text>
      ))}

      {series.map((s) => {
        const pts = s.values
          .map((v, i) => (v === null ? null : `${px(i)},${py(v)}`))
          .filter((p): p is string => p !== null);
        return (
          <g key={s.name}>
            <polyline points={pts.join(" ")} fill="none" stroke={s.color} strokeWidth={2}
                      strokeLinejoin="round" strokeLinecap="round" />
            {s.values.map((v, i) =>
              v === null ? null : (
                <circle key={i} cx={px(i)} cy={py(v)} r={3.4} fill="#fff" stroke={s.color} strokeWidth={2} />
              ),
            )}
          </g>
        );
      })}

      {yLabel && (
        <text x={14} y={padT + h / 2} fontSize={11} fill="#667085"
              transform={`rotate(-90 14 ${padT + h / 2})`} textAnchor="middle">
          {yLabel}
        </text>
      )}
      {xLabel && (
        <text x={padL + w / 2} y={height - 6} fontSize={11} fill="#667085" textAnchor="middle">
          {xLabel}
        </text>
      )}
    </svg>
  );
}

export function Legend({ series }: { series: { name: string; color: string }[] }) {
  return (
    <div className="legend">
      {series.map((s) => (
        <span className="item" key={s.name}>
          <span className="dot" style={{ background: s.color }} />
          {s.name}
        </span>
      ))}
    </div>
  );
}
