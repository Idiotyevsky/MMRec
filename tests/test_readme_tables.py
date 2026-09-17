"""The README tables must be generated, never typed.

These tests pin the two properties that make the generated findings trustworthy:
a claim is computed from ``results/tables/*.csv`` (change the CSV, change the
claim), and anything that has not been measured renders as ``TBD`` rather than
as a remembered number.
"""

import csv

import pytest

from analysis import make_readme_tables as mrt

OVERALL_FIELDS = [
    "tag", "dataset", "model", "fusion", "modalities", "id_dropout", "item_dropout",
    "seed", "Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "NDCG@20",
    "MRR@20", "Coverage@20", "num_users", "params", "best_epoch", "train_time_s", "run_id",
]


def _write(tables_dir, name, rows, fields):
    with open(tables_dir / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _overall_row(tag, model, seed, recall20, ndcg20):
    return {
        "tag": tag, "dataset": "base", "model": model, "fusion": "-", "modalities": "id",
        "id_dropout": "0.0", "item_dropout": "0.0", "seed": seed,
        "Recall@20": recall20, "NDCG@20": ndcg20, "num_users": "1000",
        "params": "0", "run_id": f"{tag}_20260101-000000_abc123",
    }


@pytest.fixture()
def tables(tmp_path, monkeypatch):
    # both are re-pointed: efficiency_table also reads ROOT/results/ for the
    # committed dataset stats and retrieval benchmarks, which must not leak
    # real values into a fixture-driven assertion
    monkeypatch.setattr(mrt, "TABLES", tmp_path)
    monkeypatch.setattr(mrt, "ROOT", tmp_path)
    return tmp_path


def test_findings_are_empty_until_runs_exist(tables):
    text = mrt.findings()
    assert "TBD" in text
    assert "Recall@20" not in text  # no remembered number leaks in


def test_ordering_claim_comes_from_the_csv(tables):
    _write(tables, "overall.csv", [
        _overall_row("popular", "popular", 42, "0.0040", "0.0016"),
        _overall_row("bpr", "bpr", 42, "0.0300", "0.0120"),
        _overall_row("sasrec", "sasrec", 42, "0.1200", "0.0550"),
    ], OVERALL_FIELDS)
    text = mrt.findings()
    assert "**Ordering holds**" in text
    assert "0.0040" in text and "0.0300" in text and "0.1200" in text
    assert "1 000" in text  # the denominator is named


def test_a_violated_ordering_is_reported_as_violated(tables):
    _write(tables, "overall.csv", [
        _overall_row("popular", "popular", 42, "0.0040", "0.0016"),
        _overall_row("bpr", "bpr", 42, "0.0020", "0.0010"),
        _overall_row("sasrec", "sasrec", 42, "0.1200", "0.0550"),
    ], OVERALL_FIELDS)
    text = mrt.findings()
    assert "**Ordering VIOLATED**" in text
    assert "diagnosed in the implementation" in text


def test_multiseed_runs_render_mean_and_std(tables):
    _write(tables, "overall.csv", [
        _overall_row("sasrec", "sasrec", 42, "0.1200", "0.0550"),
        _overall_row("sasrec_s2026", "sasrec", 2026, "0.1300", "0.0600"),
    ], OVERALL_FIELDS)
    text = mrt.overall_table()
    assert "0.1250 ± 0.0071" in text


LONGTAIL_FIELDS = [
    "tag", "model", "dataset", "fusion", "modalities", "seed", "id_dropout",
    "item_dropout", "bucket_rule", "num_users", "params",
    "head_Recall@20", "middle_Recall@20", "tail_Recall@20",
    "head_NDCG@20", "middle_NDCG@20", "tail_NDCG@20",
]


def test_bucket_metrics_are_averaged_across_seeds(tables):
    """Regression: a hardcoded metric list left ``tail_Recall@20`` at seed 42.

    The table said "3 runs" while every bucket value came from the first one.
    """
    def row(seed, head, tail):
        return {
            "tag": "mm", "model": "mm_sasrec", "dataset": "base", "fusion": "gated",
            "modalities": "id+text+image", "seed": seed, "id_dropout": "0.2",
            "item_dropout": "0.0", "bucket_rule": "frequency_quantile",
            "num_users": "100000", "params": "3179395",
            "head_Recall@20": head, "middle_Recall@20": head, "tail_Recall@20": tail,
            "head_NDCG@20": head, "middle_NDCG@20": head, "tail_NDCG@20": tail,
        }
    _write(tables, "long_tail.csv",
           [row("42", "0.2000", "0.0800"), row("2026", "0.2100", "0.0900")],
           LONGTAIL_FIELDS)
    text = mrt.long_tail_table()
    assert "0.2050" in text and "0.0850" in text  # the two-seed means
    assert "0.2000" not in text                   # not the first seed's value


def test_identity_fields_are_not_averaged(tables):
    """``params`` and ``num_users`` identify a run; they must not render as ±."""
    _write(tables, "overall.csv", [
        _overall_row("sasrec", "sasrec", 42, "0.1200", "0.0550"),
        _overall_row("sasrec_s2026", "sasrec", 2026, "0.1300", "0.0600"),
    ], OVERALL_FIELDS)
    merged = mrt.group_seeds(mrt.read("overall.csv"))
    assert len(merged) == 1
    assert merged[0]["params"] == "0"
    assert merged[0]["num_users"] == "1000"
    assert merged[0]["Recall@20"] == "0.1250 ± 0.0071"


def _mm_row(tag, modalities, id_dropout, recall20, ndcg20="0.0591", n_seeds=1):
    row = _overall_row(tag, "mm_sasrec", 42, recall20, ndcg20)
    row.update({"fusion": "gated", "modalities": modalities, "id_dropout": id_dropout,
                "params": "3179395"})
    return row


def test_findings_quote_the_headline_mm_row_not_an_ablation_row(tables):
    """Regression: ``_pick`` matched model/fusion/dropout only, so an ablation
    row (tags sort before ``mm_gated*``) would be quoted as MM-SASRec."""
    _write(tables, "overall.csv", [
        _overall_row("popular", "popular", 42, "0.0036", "0.0014"),
        _overall_row("bpr", "bpr", 42, "0.0334", "0.0131"),
        _overall_row("sasrec", "sasrec", 42, "0.1224", "0.0557"),
        _overall_row("sasrec_itemdrop", "sasrec", 42, "0.1283", "0.0579"),
        # sorted first by tag: an ablation variant with a suspicious value that
        # matches every headline criterion except the modality set
        _mm_row("ab_id_text", "id+text", "0.2", "0.9999"),
        _mm_row("mm_gated_iddrop", "id+text+image", "0.2", "0.1314"),
    ], OVERALL_FIELDS)
    text = mrt.findings()
    assert "0.1314" in text
    assert "0.9999" not in text


def test_long_tail_findings_quote_the_headline_mm_row(tables):
    def lt_row(tag, modalities, id_dropout, recall20):
        return {
            "tag": tag, "model": "mm_sasrec", "dataset": "base", "fusion": "gated",
            "modalities": modalities, "seed": "42", "id_dropout": id_dropout,
            "item_dropout": "0.0", "bucket_rule": "frequency_quantile",
            "num_users": "100000", "params": "3179395",
            "head_Recall@20": recall20, "middle_Recall@20": recall20,
            "tail_Recall@20": recall20, "head_NDCG@20": recall20,
            "middle_NDCG@20": recall20, "tail_NDCG@20": recall20,
            "head_users": "100", "middle_users": "100", "tail_users": "100",
        }
    # findings() reads both files; the bucket picks come from the long-tail one
    _write(tables, "overall.csv", [
        _overall_row("popular", "popular", 42, "0.0036", "0.0014"),
        _overall_row("bpr", "bpr", 42, "0.0334", "0.0131"),
        _overall_row("sasrec", "sasrec", 42, "0.1224", "0.0557"),
    ], OVERALL_FIELDS)
    id_row = lt_row("sasrec", "id", "0.0", "0.1500")
    id_row["model"] = "sasrec"
    reg_row = lt_row("sasrec_itemdrop", "id", "0.0", "0.1600")
    reg_row["model"] = "sasrec"
    reg_row["item_dropout"] = "0.2"
    _write(tables, "long_tail.csv", [
        id_row, reg_row,
        lt_row("ab_id_text", "id+text", "0.2", "0.9999"),
        lt_row("mm_gated_iddrop", "id+text+image", "0.2", "0.2000"),
    ], LONGTAIL_FIELDS)
    text = mrt.findings()
    assert "0.9999" not in text
    assert "0.2000" in text


ABLATION_FIELDS = [
    "model", "dataset", "modalities", "ID", "Text", "Image", "Video", "Fusion",
    "item_dropout", "id_dropout", "modality_dropout", "Recall@20", "NDCG@20",
    "Recall@10", "NDCG@10", "params", "seed", "tag", "run_id",
]


def _ablation_row(tag, dataset, model, recall20):
    return {
        "tag": tag, "dataset": dataset, "model": model, "modalities": "id",
        "ID": "1", "Text": "0", "Image": "0", "Video": "0", "Fusion": "-",
        "id_dropout": "0.0", "item_dropout": "0.0", "modality_dropout": "0.0",
        "Recall@20": recall20, "NDCG@20": "0.0550", "Recall@10": "0.0850",
        "NDCG@10": "0.0450", "params": "2929792", "seed": "42",
        "run_id": f"{tag}_20260101-000000_abc123",
    }


def test_cold_split_runs_stay_out_of_the_headline_tables(tables):
    """The cold10 split is a different protocol; its rows must not appear as
    a second, unexplained "ID-only" row in the overall or ablation table."""
    cold = _overall_row("cold_sasrec", "sasrec", 42, "0.9999", "0.9999")
    cold["dataset"] = "cold10"
    _write(tables, "overall.csv", [
        _overall_row("sasrec", "sasrec", 42, "0.1200", "0.0550"),
        cold,
    ], OVERALL_FIELDS)
    _write(tables, "ablation.csv", [
        _ablation_row("sasrec", "base", "sasrec", "0.1200"),
        _ablation_row("cold_sasrec", "cold10", "sasrec", "0.9999"),
    ], ABLATION_FIELDS)
    buckets = {"bucket_rule": "frequency_quantile", "num_users": "100"}
    for name in ("head", "middle", "tail"):
        buckets[f"{name}_Recall@20"] = "0.9999"
        buckets[f"{name}_NDCG@20"] = "0.1"
    lt = dict(_ablation_row("cold_sasrec", "cold10", "sasrec", "0.9999"), **buckets)
    base_lt = dict(lt, tag="sasrec", dataset="base",
                   **{f"{n}_Recall@20": "0.1000" for n in ("head", "middle", "tail")})
    _write(tables, "long_tail.csv", [base_lt, lt], LONGTAIL_FIELDS)

    assert "0.9999" not in mrt.overall_table()
    assert "0.9999" not in mrt.ablation_table()
    assert "0.9999" not in mrt.long_tail_table()
    assert "0.1000" in mrt.long_tail_table()


def test_gates_table_is_empty_without_a_documented_export(tables):
    _write(tables, "gate_by_bucket.csv", [], ["model", "run", "bucket", "modality", "mean_gate"])
    assert "TBD" in mrt.gates_table()


def test_gates_table_averages_documented_runs(tables):
    rows = [
        {"model": "mm_sasrec", "run": "a", "bucket": "head", "modality": "id", "mean_gate": "0.9"},
        {"model": "mm_sasrec", "run": "b", "bucket": "head", "modality": "id", "mean_gate": "0.8"},
        {"model": "mm_sasrec_iddrop", "run": "c", "bucket": "head", "modality": "id", "mean_gate": "0.7"},
    ]
    _write(tables, "gate_by_bucket.csv", rows, ["model", "run", "bucket", "modality", "mean_gate"])
    text = mrt.gates_table()
    assert "| mm_sasrec | id | 0.850 |" in text
    assert "| mm_sasrec_iddrop | id | 0.700 |" in text  # variants stay separate rows
    assert "3 documented run(s)" in text


RUNS_INDEX_FIELDS = [
    "run_id", "tag", "model", "dataset", "fusion", "modalities", "seed", "id_dropout",
    "item_dropout", "params", "git_sha", "git_dirty", "dataset_hash", "config_hash",
    "best_epoch", "epochs", "train_time_s", "provenance",
]


def test_efficiency_table_names_its_runs(tables):
    fields = RUNS_INDEX_FIELDS
    _write(tables, "runs_index.csv", [{
        "run_id": "sasrec_20260101-000000_aaa", "tag": "sasrec", "model": "sasrec",
        "fusion": "-", "modalities": "id", "seed": "42", "id_dropout": "0.0",
        "item_dropout": "0.0", "params": "100", "best_epoch": "3", "epochs": "10",
        "train_time_s": "50.0", "git_sha": "x", "git_dirty": "False",
        "dataset_hash": "d", "config_hash": "c", "provenance": "True",
    }, {
        "run_id": "mm_gated_20260101-000000_bbb", "tag": "mm_gated", "model": "mm_sasrec",
        "fusion": "gated", "modalities": "id+text+image", "seed": "42", "id_dropout": "0.2",
        "item_dropout": "0.0", "params": "300", "best_epoch": "4", "epochs": "8",
        "train_time_s": "80.0", "git_sha": "x", "git_dirty": "False",
        "dataset_hash": "d", "config_hash": "c", "provenance": "True",
    }], fields)
    text = mrt.efficiency_table()
    assert "100" in text and "300" in text
    assert "+200.0 %" in text
    assert "5.0 / 10.0 s" in text  # 50 s over 10 epochs, 80 s over 8 epochs
    assert "runs behind these numbers" in text


def test_efficiency_table_ignores_ablation_and_cold_runs(tables):
    """Regression: the pickers matched model/fusion/dropout, so a cold or
    ablation row sorted first would supply the headline params and timings."""
    def row(tag, model, dataset, modalities, id_dropout, item_dropout, params,
            epochs, train_time_s):
        return {
            "run_id": f"{tag}_20260101-000000_aaa", "tag": tag, "model": model,
            "dataset": dataset, "fusion": "gated" if model == "mm_sasrec" else "-",
            "modalities": modalities, "seed": "42", "id_dropout": id_dropout,
            "item_dropout": item_dropout, "params": params, "best_epoch": "1",
            "epochs": epochs, "train_time_s": train_time_s, "git_sha": "x",
            "git_dirty": "False", "dataset_hash": "d", "config_hash": "c",
            "provenance": "True",
        }
    _write(tables, "runs_index.csv", [
        # these sort first and previously matched the headline criteria
        row("cold_sasrec", "sasrec", "cold10", "id", "0.0", "0.0", "999999", "999", "9990.0"),
        row("ab_id_text", "mm_sasrec", "base", "id+text", "0.2", "0.0", "999999", "999",
            "9990.0"),
        row("sasrec", "sasrec", "base", "id", "0.0", "0.0", "100", "10", "50.0"),
        row("mm_gated_iddrop", "mm_sasrec", "base", "id+text+image", "0.2", "0.0",
            "300", "8", "80.0"),
    ], RUNS_INDEX_FIELDS)
    text = mrt.efficiency_table()
    assert "999999" not in text and "999" not in text
    assert "300" in text and "100" in text
