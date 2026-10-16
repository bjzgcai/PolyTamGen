"""Scientific reporting rules, exercised without models or archived datasets."""

import csv
import gc
import importlib
import json
import subprocess
import sys
import weakref
from pathlib import Path

import pandas as pd
import pytest


PAPER_DIR = Path(__file__).resolve().parents[1] / "scripts/paper"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(PAPER_DIR))
    tables = importlib.import_module("diagnostics.generate_consistency_tables_from_per_molecule")
    verifier = importlib.import_module("verify_paper_alignment")
    return tables, verifier


def molecule(**overrides):
    row = dict(method="TamGen", ref_smiles="CCC", p1="A", p2="B",
               dock_avg=-8.5, dock_worst=-6.0, qed=0.5, sa_norm=0.7,
               validity=1.0, diversity=0.6)
    row.update(overrides)
    return row


def test_strict_cutoffs_finite_denominator_and_context_weighting(modules):
    tables, _ = modules
    frame = pd.DataFrame([
        molecule(dock_avg=-8.18),
        molecule(dock_avg=-8.19),
        molecule(dock_avg=-7.5),
        molecule(dock_avg=float("nan")),
        molecule(dock_avg=float("inf")),
        molecule(dock_avg=float("-inf")),
        molecule(p1="C", dock_avg=-9.0, qed=0.25),
        molecule(p1="C", dock_avg=-9.0, sa_norm=0.59),
    ])
    columns = ["ref_smiles", "p1", "p2"]
    primary = tables.pair_level_stats(frame, "TamGen", columns, True)
    sensitivity = tables.pair_level_stats(frame, "TamGen", columns, True, -7.0)
    assert primary["sr_strict"].tolist() == pytest.approx([1 / 3, 0])
    assert primary["sr_strict"].mean() == pytest.approx(1 / 6)
    assert sensitivity["sr_strict"].tolist() == pytest.approx([1, 0])


def test_verifier_uses_average_docking_primary_cutoff(tmp_path, modules):
    _, verifier = modules
    path = tmp_path / "per_molecule.json"
    path.write_text(json.dumps([molecule()]))
    checks = []
    verifier.check_main_scored(checks, path, "T2", {
        "TamGen": [(-8.5, 0), (1, 0), (1, 0), (0.6, 0), (0.5, 0), (0.7, 0)],
    })
    assert checks and all(check["status"] == "ok" for check in checks)


def test_missing_scored_method_is_reported(tmp_path, modules):
    _, verifier = modules
    path = tmp_path / "per_molecule.json"
    path.write_text(json.dumps([molecule(method="different")]))
    checks = []
    verifier.check_main_scored(checks, path, "T2", {"TamGen": [(0, 0)] * 6})
    assert checks and all(check["status"] == "fail" for check in checks)


def test_serial_tables_release_input_frame(tmp_path, modules):
    tables, _ = modules
    frame = pd.DataFrame([molecule()])
    reference = weakref.ref(frame)
    tables.table1_t2(frame, tmp_path / "t2.tex", num_workers=1)
    del frame
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("threshold, expected", [(None, "0.500"), (-7.0, "1.000")])
def test_table_cli_labels_and_applies_cutoff(tmp_path, threshold, expected):
    rows = [molecule(dock_avg=-8.5), molecule(dock_avg=-7.5)]
    t2 = tmp_path / "t2.json"
    t3 = tmp_path / "t3.json"
    t2.write_text(json.dumps(rows))
    t3.write_text(json.dumps([dict(row, p3="C") for row in rows]))
    output = tmp_path / "tables"
    command = [sys.executable, str(PAPER_DIR / "diagnostics/generate_consistency_tables_from_per_molecule.py"),
               "--t2_path", str(t2), "--t3_path", str(t3),
               "--output_dir", str(output), "--num_workers", "1"]
    if threshold is not None:
        command += ["--strict-threshold", str(threshold)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    for name in ["table1_t2_main.tex", "table3_t3_stress.tex"]:
        text = (output / name).read_text()
        assert f"SR\\_strict@{threshold if threshold is not None else -8.18}" in text
        row = next(line for line in text.splitlines() if line.startswith("TamGen &"))
        assert row.split(" & ")[2].startswith(expected)


def test_missing_revised_conditioning_inputs_fail(tmp_path, monkeypatch, modules):
    _, verifier = modules
    monkeypatch.setattr(verifier, "RESULTS", tmp_path)
    checks = []
    verifier.check_target_conditioning(checks)
    assert checks and all(check["status"] == "fail" for check in checks)


def test_conditioning_csv_equal_context_means_and_delta_validation(tmp_path, monkeypatch, modules):
    _, verifier = modules
    monkeypatch.setattr(verifier, "RESULTS", tmp_path)
    folder = tmp_path / "supplement/cross_target"
    folder.mkdir(parents=True)
    summary = {
        "coverage": {"paired_contexts": 2, "paired_rank_molecule_pairs": 4},
        "scores_kcal_per_mol": {
            "ab_conditioned_on_ab_context_equal_mean": -9.0,
            "c_conditioned_on_ab_context_equal_mean": -7.0,
            "conditioning_advantage_c_minus_ab_context_equal_mean": 2.0,
            "conditioning_advantage_context_bootstrap_95ci": [0.613, 0.790],
        },
        "bootstrap": {"unit": "target context", "replicates": 10000, "seed": 20260823},
    }
    (folder / "target_conditioning_summary.json").write_text(json.dumps(summary))
    rows = [dict(target_id="A", source_rank=1, ab_conditioned_on_ab=-10,
                 c_conditioned_on_ab=-9, conditioning_advantage=1)]
    rows += [dict(target_id="B", source_rank=rank, ab_conditioned_on_ab=-8,
                  c_conditioned_on_ab=-5, conditioning_advantage=3) for rank in range(1, 4)]
    rows[-1]["conditioning_advantage"] = 4
    with (folder / "rank_matched_conditioning_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checks = []
    verifier.check_target_conditioning(checks)
    csv_checks = [check for check in checks if check["name"].startswith("target_conditioning.csv.")]
    assert len(csv_checks) == 3 and all(check["status"] == "ok" for check in csv_checks)
    delta_check = next(check for check in checks if check["name"] == "target_conditioning.delta_mismatches")
    assert delta_check["status"] == "fail" and delta_check["actual"] == 1
