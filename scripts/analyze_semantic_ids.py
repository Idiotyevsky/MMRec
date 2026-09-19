#!/usr/bin/env python
"""Canonicalise Semantic IDs and measure their quality.

    python scripts/analyze_semantic_ids.py
    python scripts/analyze_semantic_ids.py --semantic-ids artifacts/semantic_ids_content.npz \
        --rqvae artifacts/rqvae_content.pt --out-tag content

Produces

    artifacts/semantic_ids.npz              canonical id table (three id spaces)
    artifacts/semantic_ids_meta.json        provenance: hashes, config, collision stats
    results/tables/semantic_id_quality.csv  per-level utilisation / perplexity / collision
    results/tables/semantic_id_collisions.csv
    results/tables/semantic_id_prefix_similarity.csv

Four id spaces are kept explicitly distinct (a previous round of this project lost
time to an internal/raw off-by-one, so nothing here is called a bare ``item_id``):

    internal_item_id   1..N, the model's index space (0 is PAD)
    raw_item_id        the MicroLens id the API and the UI speak
    semantic_id        the tuple of codes
    feature_row        row of the content embedding the codes were fit on
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.models.semantic_id import SemanticIDMapper  # noqa: E402
from src.recall.semantic import build_content_embeddings  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.semantic_ids")
TABLES = ROOT / "results" / "tables"


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _entropy_perplexity(counts: np.ndarray) -> tuple[float, float]:
    p = counts[counts > 0].astype(np.float64)
    p = p / p.sum()
    ent = float(-(p * np.log(p)).sum())
    return ent, float(np.exp(ent))


def quality_rows(mapper: SemanticIDMapper) -> list[dict]:
    """One row per level plus an overall row."""
    codes = mapper.codes  # (N, K)
    n, k = codes.shape
    rows = []
    utils, perps = [], []
    for level in range(k):
        counts = np.bincount(codes[:, level], minlength=mapper.codebook_size)
        util = float((counts > 0).mean())
        ent, perp = _entropy_perplexity(counts)
        utils.append(util)
        perps.append(perp)
        rows.append({
            "scope": f"level_{level + 1}",
            "num_items": int(n),
            "num_levels": int(k),
            "codebook_size": int(mapper.codebook_size),
            "utilization": util,
            "dead_codes": int((counts == 0).sum()),
            "entropy": ent,
            "perplexity": perp,
            "max_code_count": int(counts.max()),
            "min_code_count": int(counts.min()),
        })

    sid_counts = Counter(mapper.item_to_sid.values())
    n_unique = len(sid_counts)
    stats = mapper.statistics()
    rows.append({
        "scope": "overall",
        "num_items": int(n),
        "num_levels": int(k),
        "codebook_size": int(mapper.codebook_size),
        "utilization": float(np.mean(utils)),
        "dead_codes": int(sum(r["dead_codes"] for r in rows)),
        "entropy": float(np.mean([r["entropy"] for r in rows])),
        "perplexity": float(np.mean(perps)),
        "unique_semantic_ids": n_unique,
        "unique_ratio": n_unique / max(n, 1),
        "collision_count": int(n - n_unique),
        "collision_rate": stats["collision_rate"],
        "num_colliding_sids": stats["num_colliding_sids"],
        "max_items_per_sid": stats["max_items_per_sid"],
        "num_items_in_collisions": stats["num_items_in_collisions"],
    })
    return rows


def collision_rows(mapper: SemanticIDMapper, raw_item_ids: np.ndarray) -> list[dict]:
    rows = []
    for sid, items in sorted(mapper.sid_to_items.items()):
        if len(items) < 2:
            continue
        rows.append({
            "semantic_id": "-".join(str(c) for c in sid),
            "num_items": len(items),
            "internal_item_ids": " ".join(str(i) for i in sorted(items)),
            "raw_item_ids": " ".join(str(int(raw_item_ids[i - 1])) for i in sorted(items)),
        })
    rows.sort(key=lambda r: (-r["num_items"], r["semantic_id"]))
    return rows


def prefix_similarity_rows(
    mapper: SemanticIDMapper,
    embeddings: np.ndarray,
    n_pairs: int = 200_000,
    seed: int = 0,
) -> list[dict]:
    """Do items sharing a longer SID prefix have more similar content?

    For each prefix length k we sample item pairs that share the first k codes
    (from ``k = 1`` up to the full SID) and compare their content cosine
    similarity against a random-pair baseline.  Reported as measured: if the
    trend is not monotonic, that is what the table says.
    """
    rng = np.random.default_rng(seed)
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.shape[0] != mapper.num_items + 1:
        raise ValueError(
            f"expected {mapper.num_items + 1} embedding rows (row 0 is PAD), "
            f"got {emb.shape[0]}"
        )
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms

    def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (unit[a] * unit[b]).sum(axis=1)

    def sample_pairs(groups: dict, budget: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Pairs sharing a prefix, sampled over **items** rather than groups.

        Sampling groups proportionally to their size would make short prefixes
        dominate and would over-represent the few large groups; sampling a
        random item and then a random partner within its group keeps every item
        equally likely, which is the comparison we actually want.
        """
        members = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items() if len(v) >= 2}
        eligible = np.concatenate(list(members.values())) if members else np.array([], dtype=np.int64)
        if eligible.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), 0
        keys = list(members.keys())
        # map each eligible item back to its group
        owner = {}
        for k in keys:
            for it in members[k].tolist():
                owner[it] = k
        picks = rng.choice(eligible, size=min(budget, eligible.size * 4), replace=True)
        a, b = [], []
        for it in picks.tolist():
            grp = members[owner[it]]
            others = grp[grp != it]
            if others.size == 0:
                continue
            a.append(it)
            b.append(int(rng.choice(others)))
        return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), int(eligible.size)

    rows = []
    for k in range(1, mapper.num_levels + 1):
        groups: dict[tuple, list[int]] = defaultdict(list)
        for internal in range(1, mapper.num_items + 1):
            groups[mapper.item_to_sid[internal][:k]].append(internal - 1)
        a, b, n_eligible = sample_pairs(groups, n_pairs)
        sims = cos(a, b) if a.size else np.array([])
        rows.append({
            "prefix_length": k,
            "scope": "same_prefix",
            "num_groups": len(groups),
            "num_items_with_partner": n_eligible,
            "num_pairs": int(a.size),
            "mean_cosine": float(sims.mean()) if sims.size else float("nan"),
            "median_cosine": float(np.median(sims)) if sims.size else float("nan"),
        })

    a = rng.integers(0, mapper.num_items, size=min(n_pairs, mapper.num_items * 20))
    b = rng.integers(0, mapper.num_items, size=a.shape[0])
    sims = cos(a, b)
    baseline = float(sims.mean())
    rows.append({
        "prefix_length": 0,
        "scope": "random_pairs",
        "num_groups": 1,
        "num_items_with_partner": int(mapper.num_items),
        "num_pairs": int(a.size),
        "mean_cosine": baseline,
        "median_cosine": float(np.median(sims)),
    })
    # express each prefix level as a lift over the random-pair baseline, which
    # is what the reader actually needs: content embeddings share a common
    # direction, so the absolute cosine is not informative on its own
    for r in rows:
        v = r.get("mean_cosine")
        r["mean_cosine_lift"] = float("nan") if v is None or np.isnan(v) else v - baseline
    return rows


def neighbour_agreement_rows(
    mapper: SemanticIDMapper,
    embeddings: np.ndarray,
    n_items: int = 3000,
    k_nn: int = 10,
    seed: int = 0,
) -> list[dict]:
    """Do an item's *content* nearest neighbours share its codes?

    Mean cosine over sampled pairs is a blunt instrument because the content
    space is concentrated.  This is the sharper question for retrieval: take an
    item's true top-k content neighbours and ask what share of them share the
    first ``k`` codes, against the rate expected from the code distribution
    (``1 / 256`` per level).  If the SID does not preserve content
    neighbourhoods, generative retrieval cannot recover them either.
    """
    rng = np.random.default_rng(seed)
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.shape[0] != mapper.num_items + 1:
        raise ValueError(
            f"expected {mapper.num_items + 1} embedding rows (row 0 is PAD), "
            f"got {emb.shape[0]}"
        )
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms

    codes = mapper.codes
    n = mapper.num_items
    # work in internal ids (1..N); codes row i belongs to internal id i+1
    sample = rng.choice(np.arange(1, n + 1), size=min(n_items, n), replace=False)
    rows = []
    for k in range(1, mapper.num_levels + 1):
        hits = 0
        total = 0
        for i in sample.tolist():
            sims = unit @ unit[i]
            sims[i] = -np.inf
            sims[0] = -np.inf  # never treat PAD as a neighbour
            nbrs = np.argpartition(-sims, k_nn)[:k_nn]
            same = (codes[nbrs - 1, :k] == codes[i - 1, :k]).all(axis=1)
            hits += int(same.sum())
            total += int(same.size)
        observed = hits / max(total, 1)
        chance = 1.0 / (mapper.codebook_size ** k)
        rows.append({
            "prefix_length": k,
            "num_items_sampled": int(sample.size),
            "neighbours_per_item": int(k_nn),
            "num_pairs": int(total),
            "share_sharing_prefix": observed,
            "chance_rate": chance,
            "lift_vs_chance": observed / chance if chance else float("nan"),
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    LOG.info(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semantic-ids", default="artifacts/semantic_ids_content.npz")
    ap.add_argument("--rqvae", default="artifacts/rqvae_content.pt")
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--feature-dir", default="data/raw")
    ap.add_argument("--modalities", nargs="*", default=["text", "image"])
    ap.add_argument("--out-tag", default="content")
    ap.add_argument("--pairs", type=int, default=200_000)
    args = ap.parse_args()

    data = ProcessedData.load(ROOT / args.processed_dir)
    npz = np.load(ROOT / args.semantic_ids)
    codes = npz["codes"].astype(np.int64)
    codebook_size = int(npz["codebook_size"])
    if codes.shape[0] != data.num_items:
        raise SystemExit(
            f"semantic IDs cover {codes.shape[0]} items but the dataset has {data.num_items}"
        )
    mapper = SemanticIDMapper(codes, num_items=data.num_items, codebook_size=codebook_size)
    LOG.info(f"loaded {codes.shape[0]} semantic IDs ({mapper.num_levels} levels x {codebook_size})")

    # ---- content embeddings, for the prefix semantic check ----------------
    emb = build_content_embeddings(
        feature_dir=ROOT / args.feature_dir,
        row_for_item_dir=ROOT / args.processed_dir,
        num_items=data.num_items,
        modalities=tuple(args.modalities),
    )

    quality = quality_rows(mapper)
    collisions = collision_rows(mapper, data.raw_item_ids)
    prefix = prefix_similarity_rows(mapper, emb, n_pairs=args.pairs)
    neighbour = neighbour_agreement_rows(mapper, emb)

    write_csv(quality, TABLES / "semantic_id_quality.csv")
    write_csv(neighbour, TABLES / "semantic_id_neighbour_agreement.csv")
    write_csv(collisions, TABLES / "semantic_id_collisions.csv")
    write_csv(prefix, TABLES / "semantic_id_prefix_similarity.csv")

    # ---- canonical artifact ----------------------------------------------
    internal = np.arange(1, data.num_items + 1, dtype=np.int64)
    out_npz = ROOT / "artifacts" / "semantic_ids.npz"
    np.savez_compressed(
        out_npz,
        internal_item_id=internal,
        raw_item_id=data.raw_item_ids.astype(np.int64),
        codes=codes.astype(np.int16),
    )
    sid_hashes = hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest()[:16]
    meta = {
        "source_artifact": args.semantic_ids,
        "rqvae_checkpoint": args.rqvae,
        "rqvae_checkpoint_sha16": _sha16(ROOT / args.rqvae) if (ROOT / args.rqvae).exists() else None,
        "processed_dir": args.processed_dir,
        "dataset_hash": hashlib.sha256(
            np.ascontiguousarray(data.train_freq).tobytes()
        ).hexdigest()[:16],
        "semantic_id_hash": sid_hashes,
        "modalities": list(args.modalities),
        "num_items": int(data.num_items),
        "num_levels": int(mapper.num_levels),
        "codebook_size": int(codebook_size),
        "collision_rate": float(mapper.statistics()["collision_rate"]),
        "unique_semantic_ids": int(len(mapper.sid_to_items)),
        "id_spaces": {
            "internal_item_id": "1..N model index space, 0 is PAD",
            "raw_item_id": "MicroLens id used by the API and the UI",
            "codes": "semantic id, row i belongs to internal_item_id i+1",
        },
    }
    save_json(meta, ROOT / "artifacts" / "semantic_ids_meta.json")
    LOG.info(f"wrote {out_npz.relative_to(ROOT)} and semantic_ids_meta.json")

    print("\n=== Semantic ID quality ===")
    for r in quality:
        if r["scope"] == "overall":
            print(f"  overall: unique {r['unique_semantic_ids']}/{r['num_items']} "
                  f"({r['unique_ratio']:.4f}) | collision {r['collision_rate']:.4f} | "
                  f"mean util {r['utilization']:.3f} | mean perplexity {r['perplexity']:.1f}")
        else:
            print(f"  {r['scope']:8s}: util {r['utilization']:.3f} | dead {r['dead_codes']:3d} | "
                  f"perplexity {r['perplexity']:6.1f}")
    print("\n=== Prefix semantic check (mean cosine, lift over random pairs) ===")
    for r in prefix:
        print(f"  {r['scope']:13s} k={r['prefix_length']} n={r['num_pairs']:7d} "
              f"mean {r['mean_cosine']:+.4f} lift {r['mean_cosine_lift']:+.4f}")

    print("\n=== Do content neighbours share codes? ===")
    for r in neighbour:
        print(f"  k={r['prefix_length']} neighbours sharing prefix "
              f"{r['share_sharing_prefix']:.4f} vs chance {r['chance_rate']:.6f} "
              f"({r['lift_vs_chance']:.1f}x)")


if __name__ == "__main__":
    main()
