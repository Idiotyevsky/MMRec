"""The gate table groups runs by variant; the key must not merge unlike runs."""

from __future__ import annotations

import textwrap

import yaml

from analysis import analyze_gates as ag


def _run_dir(tmp_path, name, *, model="mm_sasrec", modalities=("id", "text", "image"),
             id_dropout=0.0, item_dropout=0.0, processed_dir="data/processed/base"):
    d = tmp_path / name
    d.mkdir()
    cfg = {
        "model": {
            "name": model,
            "modalities": {k: (k in modalities) for k in ("id", "text", "image", "video")},
            "id_dropout_prob": id_dropout,
            "item_dropout_prob": item_dropout,
        },
        "data": {"processed_dir": processed_dir},
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return d


def test_default_and_ablation_variants_are_distinct_keys(tmp_path):
    """Regression: the key used to be the model name plus dropout only, so an
    ablation run (different modality set) or a cold-split run would be averaged
    into the default gated rows."""
    keys = {
        ag._variant(_run_dir(tmp_path, "a")),
        ag._variant(_run_dir(tmp_path, "b", modalities=("id", "text"))),
        ag._variant(_run_dir(tmp_path, "c", modalities=("id", "video"))),
        ag._variant(_run_dir(tmp_path, "d", processed_dir="data/processed/cold10")),
        ag._variant(_run_dir(tmp_path, "e", id_dropout=0.2)),
    }
    assert len(keys) == 5


def test_variant_names_are_readable(tmp_path):
    assert ag._variant(_run_dir(tmp_path, "a")) == "mm_sasrec(id+text+image)"
    assert ag._variant(_run_dir(tmp_path, "b", id_dropout=0.2)) == \
        "mm_sasrec_iddrop(id+text+image)"
    assert ag._variant(_run_dir(tmp_path, "c", modalities=("text",))) == "mm_sasrec(text)"
    assert ag._variant(_run_dir(tmp_path, "d", processed_dir="data/processed/cold10")) == \
        "mm_sasrec(id+text+image)@cold10"


def test_id_only_models_keep_their_historical_names(tmp_path):
    """``sasrec`` / ``sasrec_reg`` are the labels the committed table already uses."""
    assert ag._variant(_run_dir(tmp_path, "a", model="sasrec", modalities=("id",))) == "sasrec"
    assert ag._variant(
        _run_dir(tmp_path, "b", model="sasrec", modalities=("id",), item_dropout=0.2)
    ) == "sasrec_reg"


def test_unreadable_config_falls_back_to_the_run_name(tmp_path):
    d = tmp_path / "some_run"
    d.mkdir()
    assert ag._variant(d) == "some_run"


def _write_run(tmp_path, run_id, *, git_dirty=False):
    run = tmp_path / run_id
    run.mkdir()
    (run / "run_manifest.json").write_text(
        textwrap.dedent(f"""\
            {{"run_id": "{run_id}", "git_sha": "{'0' * 40}", "git_dirty": {str(git_dirty).lower()}}}
        """),
        encoding="utf-8",
    )
    return run


def test_manifest_less_runs_are_listed_not_used(tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "RUNS", tmp_path)
    _write_run(tmp_path, "documented_20260101-000000_aaaaaa")
    undocumented = tmp_path / "legacy_20260101-000000_bbbbbb"
    undocumented.mkdir()
    (undocumented / "gate_weights.npz").write_bytes(b"")
    (tmp_path / "documented_20260101-000000_aaaaaa" / "gate_weights.npz").write_bytes(b"")

    files, skipped = ag._gate_files()
    assert [p.parent.name for p in files] == ["documented_20260101-000000_aaaaaa"]
    assert skipped == ["legacy_20260101-000000_bbbbbb"]
