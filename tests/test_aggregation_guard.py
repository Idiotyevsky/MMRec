"""The aggregation gate: no run enters a table without agreeing provenance.

These tests exist because of a real incident: SASRec seed 42 and seed 3407 were
once aggregated as replicates although their parameter counts differed by 128
(one config had allocated an optional regulariser), and seed aggregates were
printed as if they were the same experiment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from analysis import aggregate_results as agg


def _git(*args: str):
    return subprocess.run(["git", *args], cwd=agg.ROOT, capture_output=True, text=True)


def _rev(sha: str) -> str:
    """Resolve a commit, skipping the test when this checkout does not have it."""
    out = _git("rev-parse", "--verify", f"{sha}^{{commit}}")
    if out.returncode != 0:
        pytest.skip(f"commit {sha} is not in this checkout")
    return out.stdout.strip()


def _config(seed: int = 42, **training) -> dict:
    t = {"seed": seed, "batch_size": 512, "learning_rate": 0.001}
    t.update(training)
    return {
        "model": {"name": "sasrec", "hidden_size": 128, "num_layers": 2,
                  "item_dropout_prob": 0.0, "id_dropout_prob": 0.0,
                  "modality_dropout_prob": 0.0},
        "data": {"processed_dir": "data/processed/base"},
        "training": t,
        "evaluation": {"ks": [5, 10, 20]},
    }


def _make_run(root: Path, name: str, *, seed: int = 42, params: int = 100,
              provenance: bool = True, git_sha: str = "a" * 40,
              git_dirty: bool = False,
              dataset_hash: str = "b" * 32, recall20: float = 0.08,
              best_epoch: int = 12, config: dict | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = config if config is not None else _config(seed=seed)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    metrics = {
        "run_id": name,
        "model": cfg["model"]["name"],
        "seed": seed,
        "train": {"num_parameters": params, "best_epoch": best_epoch},
        "val": {"NDCG@10": recall20 / 2},
        "test": {"Recall@20": recall20, "NDCG@20": recall20 / 2, "Recall@10": recall20 / 2},
    }
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (d / "train_summary.json").write_text(json.dumps({"num_parameters": params}), encoding="utf-8")
    if provenance:
        (d / "run_manifest.json").write_text(json.dumps({
            "run_id": name, "git_sha": git_sha, "git_dirty": git_dirty,
            "dataset_hash": dataset_hash, "config_hash": "c" * 16,
            "num_parameters": params, "seed": seed,
        }), encoding="utf-8")
    return d


def _signature_of(runs, name):
    return next(agg._signature(r) for r in runs if r["run_id"].startswith(name))


# --------------------------------------------------------------------------- #
# group key
# --------------------------------------------------------------------------- #

def test_group_key_strips_the_seed_suffix_of_the_same_seed():
    assert agg._group_key({"tag": "sasrec_s2026", "seed": 2026}) == "sasrec"
    assert agg._group_key({"tag": "mm_gated_s3407", "seed": 3407}) == "mm_gated"


def test_group_key_keeps_a_tag_that_only_looks_like_a_seed_suffix():
    # `_s2026` is a seed suffix only when the run really was seed 2026
    assert agg._group_key({"tag": "sasrec_s2026", "seed": 42}) == "sasrec_s2026"
    assert agg._group_key({"tag": "cold_mm_gated", "seed": 42}) == "cold_mm_gated"


# --------------------------------------------------------------------------- #
# config hash
# --------------------------------------------------------------------------- #

def test_config_core_hash_ignores_only_the_seed():
    base = _config(seed=42)
    assert agg._config_core_hash(base) == agg._config_core_hash(_config(seed=3407))

    other = _config(seed=42, batch_size=256)
    assert agg._config_core_hash(base) != agg._config_core_hash(other)


def test_config_core_hash_is_order_insensitive():
    a = _config(seed=42)
    b = {k: a[k] for k in reversed(list(a))}
    assert agg._config_core_hash(a) == agg._config_core_hash(b)


# --------------------------------------------------------------------------- #
# the gate itself
# --------------------------------------------------------------------------- #

def test_seeds_of_one_experiment_are_grouped_and_kept(tmp_path):
    runs = []
    for tag, seed in (("sasrec", 42), ("sasrec_s2026", 2026), ("sasrec_s3407", 3407)):
        _make_run(tmp_path, f"{tag}_20260917-000000_aaaaaa", seed=seed)
    loaded = agg.load_runs(tmp_path)
    assert len(loaded) == 3
    kept = agg._enforce_compatibility(loaded, allow_mixed=False)
    assert len(kept) == 3


def test_the_128_parameter_mismatch_is_excluded_not_averaged(tmp_path, capsys):
    """The incident that motivated the gate, reproduced on disk."""
    _make_run(tmp_path, "sasrec_20260917-000000_aaaaaa", seed=42, params=2929792)
    _make_run(tmp_path, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, params=2929792)
    _make_run(tmp_path, "sasrec_s3407_20260917-000000_cccccc", seed=3407, params=2929920)

    loaded = agg.load_runs(tmp_path)
    kept = agg._enforce_compatibility(loaded, allow_mixed=False)
    ids = sorted(r["run_id"] for r in kept)
    assert len(kept) == 2, "the odd parameter count must not be pooled with the others"
    assert not any("s3407" in i for i in ids)

    out = capsys.readouterr().out
    assert "mixes 2 incompatible provenances" in out
    assert "excluded sasrec_s3407_20260917-000000_cccccc" in out


def test_allow_mixed_keeps_everything_with_a_warning(tmp_path, capsys):
    _make_run(tmp_path, "sasrec_20260917-000000_aaaaaa", seed=42, params=100)
    _make_run(tmp_path, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, params=228)
    loaded = agg.load_runs(tmp_path)
    kept = agg._enforce_compatibility(loaded, allow_mixed=True)
    assert len(kept) == 2
    assert "--allow-mixed" in capsys.readouterr().out


def test_different_datasets_are_not_pooled(tmp_path):
    _make_run(tmp_path, "sasrec_20260917-000000_aaaaaa", seed=42, dataset_hash="b" * 32)
    _make_run(tmp_path, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, dataset_hash="e" * 32)
    kept = agg._enforce_compatibility(agg.load_runs(tmp_path), allow_mixed=False)
    assert len(kept) == 1


def test_different_code_versions_are_not_pooled(tmp_path):
    _make_run(tmp_path, "sasrec_20260917-000000_aaaaaa", seed=42, git_sha="a" * 40)
    _make_run(tmp_path, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, git_sha="f" * 40)
    kept = agg._enforce_compatibility(agg.load_runs(tmp_path), allow_mixed=False)
    assert len(kept) == 1


@pytest.mark.parametrize("git_dirty", [True, None], ids=["dirty", "unknown"])
def test_a_run_whose_tree_is_not_the_commit_is_never_pooled(tmp_path, git_dirty):
    """A dirty run's code is not the commit's code, so no SHA claim covers it.

    The same holds when the check could not run at all (``None``): an
    unanswerable question is not a clean answer.
    """
    _make_run(tmp_path, "sasrec_20260917-000000_aaaaaa", seed=42, git_sha="a" * 40)
    _make_run(tmp_path, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026,
              git_sha="a" * 40, git_dirty=git_dirty)
    loaded = agg.load_runs(tmp_path)
    assert _signature_of(loaded, "sasrec_") != _signature_of(loaded, "sasrec_s2026")
    assert len(agg._enforce_compatibility(loaded, allow_mixed=False)) == 1


# --------------------------------------------------------------------------- #
# declared code equivalence (analysis/code_equivalence.json)
# --------------------------------------------------------------------------- #

def _declared_classes() -> list[dict]:
    blob = json.loads((agg.ROOT / "analysis" / "code_equivalence.json").read_text(encoding="utf-8"))
    return blob.get("classes") or []


def test_the_equivalence_file_is_self_consistent_and_every_commit_exists():
    """Each class must be checkable: a reference, a reason, and real commits."""
    classes = _declared_classes()
    assert classes, "the file must declare at least one class, or it is dead weight"
    for cls in classes:
        assert cls.get("id"), "a class without an id cannot be reported"
        assert cls.get("reason", "").strip(), f"{cls['id']}: a class needs its reason"
        assert cls["reference"] in cls["commits"], (
            f"{cls['id']}: the reference commit must be one of the class's commits"
        )
        for sha in cls["commits"]:
            assert _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, (
                f"{cls['id']}: commit {sha} does not exist in this repository"
            )


def test_a_class_diff_touches_no_file_outside_its_declared_list():
    """The machine-checkable half of "this diff cannot change a number".

    The reason text is a human judgement; the file list is not.  Whatever the
    class claims its commits touch, the real diff against the reference must
    stay inside it -- so a commit that quietly reaches some other file can
    never be excused by the class's prose.
    """
    paths = agg.RESULT_DETERMINING_PATHS
    for cls in _declared_classes():
        touched: set[str] = set()
        for sha in cls["commits"]:
            out = _git("diff", "--name-only", cls["reference"], _rev(sha), "--", *paths)
            assert out.returncode == 0, f"{cls['id']}: git diff failed for {sha}"
            touched.update(out.stdout.split())
        allowed = set(cls.get("touches") or [])
        assert touched <= allowed, (
            f"{cls['id']}: commits differ from the reference in {sorted(touched - allowed)}, "
            f"which the class does not declare"
        )
        assert touched, f"{cls['id']}: no listed commit differs from the reference at all"


def test_a_declared_equivalence_class_pools_its_commits_end_to_end(tmp_path, monkeypatch, capsys):
    """The 28 runs behind the tables span nine commits; they must still pool."""
    cls = _declared_classes()[0]
    older, newer = _rev(cls["reference"]), _rev(cls["commits"][-1])
    runs, tables = tmp_path / "runs", tmp_path / "tables"
    _make_run(runs, "sasrec_20260917-000000_aaaaaa", seed=42, git_sha=older)
    _make_run(runs, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, git_sha=newer)

    _run_main(monkeypatch, runs, tables)

    index = (tables / "runs_index.csv").read_text(encoding="utf-8")
    assert "sasrec_20260917-000000_aaaaaa" in index
    assert "sasrec_s2026_20260917-000000_bbbbbb" in index
    # the class is named when it is used, with the commits it actually pooled
    out = capsys.readouterr().out
    assert f"code equivalence class {cls['id']!r}" in out
    assert "mixes" not in out


def test_a_dirty_run_cannot_join_a_declared_class(tmp_path):
    """The class is a claim about *commits*; it cannot cover an edited tree."""
    cls = _declared_classes()[0]
    older, newer = _rev(cls["reference"]), _rev(cls["commits"][-1])
    _make_run(tmp_path, "sasrec_20260917-000000_aaaaaa", seed=42, git_sha=older)
    _make_run(tmp_path, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026,
              git_sha=newer, git_dirty=True)
    kept = agg._enforce_compatibility(agg.load_runs(tmp_path), allow_mixed=False)
    assert len(kept) == 1


def test_an_undeclared_commit_difference_is_still_refused(tmp_path, capsys):
    """Declaring one class must not soften the gate for anything else."""
    older, newer = _rev("9c82a19"), _rev("503aa40")
    assert agg.declared_class(older) is None and agg.declared_class(newer) is None
    _make_run(tmp_path, "sasrec_20260917-000000_aaaaaa", seed=42, git_sha=older)
    _make_run(tmp_path, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, git_sha=newer)

    kept = agg._enforce_compatibility(agg.load_runs(tmp_path), allow_mixed=False)
    assert len(kept) == 1
    assert "mixes 2 incompatible provenances" in capsys.readouterr().out


def test_equal_determining_trees_always_compare_equal():
    """The declaration *widens* comparability; it never narrows it.

    Equal result-determining trees are proof of equal code, so they must always
    end up in the same class -- including for a commit the file does not list
    whose tree matches a member's (the closure that keeps the class index from
    being stricter than the raw tree hash it replaced).
    """
    hist = _git("log", "--format=%h").stdout.split()
    if not hist:
        pytest.skip("not a git checkout")
    by_commit = agg.equivalence_classes()["by_commit"]
    seen: dict[str, str] = {}
    closed_by_tree = 0
    for sha in hist:
        tree = agg.code_identity(sha)
        if tree is None:
            continue
        cls = agg.code_class(sha)
        if tree in seen:
            assert seen[tree] == cls, (
                f"{sha} has the same result-determining tree as another commit "
                f"but a different code identity ({cls} vs {seen[tree]})"
            )
        seen[tree] = cls
        if cls.startswith("class:") and sha not in by_commit and sha[:7] not in by_commit:
            closed_by_tree += 1
    assert closed_by_tree, "no commit reached a class through its tree hash; the file is dead weight"


def test_a_manifest_less_run_is_never_pooled_with_a_documented_one(tmp_path):
    _make_run(tmp_path, "sasrec_20260917-000000_aaaaaa", seed=42, provenance=True)
    _make_run(tmp_path, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, provenance=False)
    loaded = agg.load_runs(tmp_path)
    assert len(loaded) == 2
    assert _signature_of(loaded, "sasrec_") != _signature_of(loaded, "sasrec_s2026")
    kept = agg._enforce_compatibility(loaded, allow_mixed=False)
    assert [r["provenance"] for r in kept] == [True]


def test_duplicate_seed_keeps_the_longer_trained_run(tmp_path):
    _make_run(tmp_path, "sasrec_20260917-000000_aaaaaa", seed=42, best_epoch=8)
    _make_run(tmp_path, "sasrec_20260917-010000_bbbbbb", seed=42, best_epoch=31)
    kept, dropped = agg._drop_duplicate_seeds(agg.load_runs(tmp_path))
    assert len(kept) == 1
    assert kept[0]["run_id"].endswith("bbbbbb")
    assert dropped == ["sasrec_20260917-000000_aaaaaa"]


# --------------------------------------------------------------------------- #
# end to end: what actually lands in the CSV
# --------------------------------------------------------------------------- #

def _run_main(monkeypatch, runs_dir: Path, tables_dir: Path, *extra: str) -> str:
    monkeypatch.setattr(sys, "argv", [
        "aggregate_results.py", "--runs-dir", str(runs_dir),
        "--tables-dir", str(tables_dir), *extra,
    ])
    agg.main()
    return tables_dir


def test_main_writes_only_documented_runs(tmp_path, monkeypatch):
    runs, tables = tmp_path / "runs", tmp_path / "tables"
    _make_run(runs, "sasrec_20260917-000000_aaaaaa", seed=42)
    _make_run(runs, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, provenance=False)

    _run_main(monkeypatch, runs, tables)

    index = (tables / "runs_index.csv").read_text(encoding="utf-8")
    assert "sasrec_20260917-000000_aaaaaa" in index
    assert "sasrec_s2026_20260917-000000_bbbbbb" not in index
    # the provenance columns are what a reviewer checks first
    header = index.splitlines()[0]
    for col in ("git_sha", "dataset_hash", "config_hash", "params", "seed"):
        assert col in header


def test_main_include_legacy_still_applies_the_provenance_gate(tmp_path, monkeypatch, capsys):
    """Reading a legacy run and pooling it with a documented one are two gates."""
    runs, tables = tmp_path / "runs", tmp_path / "tables"
    _make_run(runs, "sasrec_20260917-000000_aaaaaa", seed=42)
    _make_run(runs, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, provenance=False)

    _run_main(monkeypatch, runs, tables, "--include-legacy")
    out = capsys.readouterr().out
    assert "of unknown provenance" in out
    assert "mixes 2 incompatible provenances" in out
    assert "sasrec_s2026_20260917-000000_bbbbbb" not in (
        tables / "runs_index.csv"
    ).read_text(encoding="utf-8")


def test_main_include_legacy_allow_mixed_writes_everything(tmp_path, monkeypatch, capsys):
    runs, tables = tmp_path / "runs", tmp_path / "tables"
    _make_run(runs, "sasrec_20260917-000000_aaaaaa", seed=42)
    _make_run(runs, "sasrec_s2026_20260917-000000_bbbbbb", seed=2026, provenance=False)

    _run_main(monkeypatch, runs, tables, "--include-legacy", "--allow-mixed")
    out = capsys.readouterr().out
    assert "NOT valid for a result table" in out
    index = (tables / "runs_index.csv").read_text(encoding="utf-8")
    assert "sasrec_20260917-000000_aaaaaa" in index
    assert "sasrec_s2026_20260917-000000_bbbbbb" in index


def test_main_on_a_directory_without_finished_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    _run_main(monkeypatch, runs, tmp_path / "tables")
    for name in ("overall", "ablation", "cold_start", "long_tail", "runs_index"):
        assert (tmp_path / "tables" / f"{name}.csv").read_text(encoding="utf-8") == ""


def test_summary_reports_std_only_for_real_replicates(tmp_path, capsys):
    runs = []
    for tag, seed, recall in (("sasrec", 42, 0.08), ("sasrec_s2026", 2026, 0.10)):
        _make_run(tmp_path, f"{tag}_20260917-000000_aaaaaa", seed=seed, recall20=recall)
    runs = agg.load_runs(tmp_path)
    agg.print_group_summary(runs)
    out = capsys.readouterr().out
    assert "n=2 seeds=[42, 2026]" in out
    assert "0.0900±0.0100" in out
