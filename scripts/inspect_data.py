#!/usr/bin/env python
"""Ground-truth data inspection.

Never trust a filename.  This script prints the *actual* schema of the raw
MicroLens-100K files so that ``docs/data_schema.md`` can be written from
evidence rather than assumption.

Usage:
    python scripts/inspect_data.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INTER_CANDIDATES = ["microlens_100k.inter", "micro_lens.inter", "interactions.csv"]
MODALITIES = ["text", "image", "video"]


def read_inter(path: Path, nrows: int = 8) -> tuple[list[list[str]], int]:
    """Read the first ``nrows`` lines and the total line count."""
    rows: list[list[str]] = []
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            if len(rows) < nrows:
                rows.append(line.rstrip("\n").split("\t"))
    return rows, n


def describe_npy(path: Path, sample: int = 20000) -> dict:
    arr = np.load(path, mmap_mode="r")
    info: dict = {
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size_mb": round(path.stat().st_size / 1024**2, 2),
    }
    # sample rows for statistics without loading everything
    k = min(sample, arr.shape[0])
    idx = np.linspace(0, arr.shape[0] - 1, k).astype(np.int64)
    sub = np.asarray(arr[idx], dtype=np.float64)
    info.update(
        {
            "sampled_rows": int(k),
            "mean": float(sub.mean()),
            "std": float(sub.std()),
            "min": float(sub.min()),
            "max": float(sub.max()),
            "nan_count": int(np.isnan(sub).sum()),
            "inf_count": int(np.isinf(sub).sum()),
            "row_norm_mean": float(np.linalg.norm(sub, axis=1).mean()),
            "row_norm_std": float(np.linalg.norm(sub, axis=1).std()),
            "zero_rows": int((np.linalg.norm(sub, axis=1) == 0).sum()),
        }
    )
    if np.isfinite(sub).all():
        info["std_per_dim_mean"] = float(sub.std(axis=0).mean())
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="artifacts/raw_data_report.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    report: dict = {"data_dir": str(data_dir), "files": {}}

    print(f"== files in {data_dir} ==")
    for p in sorted(data_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name:32s} {p.stat().st_size/1024**2:10.2f} MB")
            report["files"][p.name] = {"size_mb": round(p.stat().st_size / 1024**2, 3)}

    # ---- interactions -------------------------------------------------
    inter_path = None
    for c in INTER_CANDIDATES:
        if (data_dir / c).exists():
            inter_path = data_dir / c
            break
    if inter_path is None:
        cands = list(data_dir.glob("*.inter")) + list(data_dir.glob("*.csv"))
        if cands:
            inter_path = cands[0]

    if inter_path is not None:
        rows, n_lines = read_inter(inter_path)
        print(f"\n== {inter_path.name} ==")
        print(f"  lines: {n_lines}")
        print(f"  delimiter: {'TAB' if len(rows[0]) > 1 else 'COMMA/OTHER'}")
        print(f"  columns per row: {[len(r) for r in rows]}")
        print("  first rows:")
        for r in rows[:5]:
            print(f"    {r}")

        # full column scan (cheap enough for ~20MB)
        cols: list[list[int]] = []
        n_rows = 0
        with open(inter_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if n_rows == 0:
                    cols = [[] for _ in parts]
                if len(parts) != len(cols):
                    n_rows += 1
                    continue
                for j, tok in enumerate(parts):
                    try:
                        cols[j].append(int(tok))
                    except ValueError:
                        pass
                n_rows += 1

        col_stats = []
        for j, c in enumerate(cols):
            a = np.asarray(c, dtype=np.int64)
            if a.size == 0:
                col_stats.append({"col": j, "empty": True})
                continue
            col_stats.append(
                {
                    "col": j,
                    "n": int(a.size),
                    "min": int(a.min()),
                    "max": int(a.max()),
                    "n_unique": int(np.unique(a).size),
                    "mean": float(a.mean()),
                }
            )
        print(f"  rows: {n_rows}")
        print("  column stats:")
        for s in col_stats:
            print(f"    {s}")

        report["interactions"] = {
            "file": inter_path.name,
            "n_rows": n_rows,
            "delimiter": "tab" if len(rows[0]) > 1 else "other",
            "n_cols": len(rows[0]),
            "columns": col_stats,
            "sample_rows": rows[:5],
        }

    # ---- modality features --------------------------------------------
    report["features"] = {}
    for m in MODALITIES:
        p = data_dir / f"{m}_feat.npy"
        if not p.exists():
            print(f"\n== {m}: MISSING ==")
            continue
        info = describe_npy(p)
        report["features"][m] = info
        print(f"\n== {m}_feat.npy ==")
        for k, v in info.items():
            print(f"  {k}: {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
