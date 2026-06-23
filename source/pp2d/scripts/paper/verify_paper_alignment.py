#!/usr/bin/env python3
"""Verify that result artifacts match the reported paper numbers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MAIN_TABLE_EXPECTED = {
    "T2": {
        "TamGen": [(-6.536, 0.997), (0.306, 0.281), (0.998, 0.003), (0.863, 0.008), (0.495, 0.045), (0.717, 0.013)],
        "PocketXMol": [(-6.581, 1.503), (0.356, 0.302), (0.999, 0.004), (0.863, 0.047), (0.533, 0.081), (0.787, 0.063)],
        "\\ourM{}-Single": [(-6.989, 0.877), (0.380, 0.239), (0.936, 0.030), (0.878, 0.007), (0.407, 0.044), (0.746, 0.021)],
        "\\ourM{}-SFT": [(-7.347, 1.017), (0.475, 0.290), (0.959, 0.046), (0.864, 0.025), (0.408, 0.084), (0.753, 0.048)],
        "\\ourM{}-DPO": [(-7.642, 1.064), (0.648, 0.338), (0.968, 0.110), (0.837, 0.041), (0.565, 0.069), (0.805, 0.032)],
    },
    "T3": {
        "TamGen": [(-6.737, 0.734), (0.348, 0.245), (0.998, 0.003), (0.866, 0.004), (0.488, 0.028), (0.715, 0.008)],
        "PocketXMol": [(-6.788, 1.114), (0.413, 0.277), (1.000, 0.001), (0.858, 0.053), (0.531, 0.068), (0.778, 0.054)],
        "\\ourM{}-SFT": [(-7.540, 0.793), (0.546, 0.240), (0.948, 0.053), (0.865, 0.020), (0.395, 0.075), (0.742, 0.049)],
        "\\ourM{}-DPO": [(-7.918, 0.819), (0.767, 0.264), (0.966, 0.079), (0.841, 0.019), (0.567, 0.068), (0.806, 0.029)],
    },
}

MAIN_METHOD_MAP = {
    "\\ourM{}-Single": "PolyTamGen-Single",
    "\\ourM{}-SFT": "PolyTamGen-SFT",
    "\\ourM{}-DPO": "PolyTamGen-DPO",
}


def public_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        pass
    try:
        return str(path.relative_to(ROOT.parents[1]))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name if path.is_absolute() else path.as_posix()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def numeric_tokens(line: str) -> List[float]:
    out: List[float] = []
    for token in re.findall(r"-?\d[\d,]*(?:\.\d+)?", line):
        out.append(float(token.replace(",", "")))
    return out


def latex_row(path: Path, needle: str) -> List[float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if needle in line and "&" in line:
            return numeric_tokens(line.split("&", 1)[1])
    raise KeyError(f"Row containing {needle!r} not found in {path}")


def first_number(text: str) -> float:
    values = numeric_tokens(text)
    if not values:
        raise ValueError(f"No numeric value in {text!r}")
    return values[0]


def fmt(value: Any, decimals: Optional[int]) -> str:
    if decimals is None:
        return str(value)
    return f"{float(value):.{decimals}f}"


def same(actual: Any, expected: Any, decimals: Optional[int]) -> bool:
    if decimals is None:
        return actual == expected
    return fmt(actual, decimals) == fmt(expected, decimals)


def add(
    checks: List[Dict[str, Any]],
    name: str,
    actual: Any,
    expected: Any,
    source: Path,
    *,
    decimals: Optional[int] = 3,
    status: Optional[str] = None,
    note: str = "",
) -> None:
    ok = same(actual, expected, decimals)
    checks.append(
        {
            "name": name,
            "status": status or ("ok" if ok else "fail"),
            "actual": fmt(actual, decimals) if decimals is not None else actual,
            "expected": fmt(expected, decimals) if decimals is not None else expected,
            "source": public_path(source),
            "note": note,
        }
    )


def load_per_molecule_dataframe(path: Path):
    import pandas as pd

    try:
        return pd.read_json(path, orient="records")
    except Exception:
        return pd.DataFrame(load_json(path))


def pocket_cols_from_df(df: Any) -> List[str]:
    cols = [col for col in df.columns if isinstance(col, str) and col.startswith("p") and col[1:].isdigit()]
    return sorted(cols, key=lambda col: int(col[1:]))


def mean_std_from_series(series: Any) -> Tuple[float, float]:
    import numpy as np
    import pandas as pd

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def check_main_scored(
    checks: List[Dict[str, Any]],
    path: Path,
    table_name: str,
    expected: Dict[str, Sequence[Tuple[float, float]]],
) -> None:
    import numpy as np

    metrics = ["AvgDock", "SR_strict", "Validity", "Diversity", "QED", "SA"]
    agg_cols = ["avgdock", "sr_strict", "validity", "diversity", "qed", "sa"]
    df = load_per_molecule_dataframe(path)
    pair_cols = ["ref_smiles"] + pocket_cols_from_df(df)
    for label, pairs in expected.items():
        method = MAIN_METHOD_MAP.get(label, label)
        sub = df[df["method"] == method].copy()
        for col in ["validity", "diversity", "qed", "sa_norm"]:
            if col not in sub.columns:
                sub[col] = np.nan
        dock_avg = sub["dock_avg"].astype(float)
        strict = (dock_avg < -7.0) & (sub["qed"].astype(float) > 0.25) & (sub["sa_norm"].astype(float) > 0.59)
        strict = strict.where(np.isfinite(dock_avg), np.nan)
        sub["strict"] = strict.astype(float)
        agg = sub.groupby(pair_cols).agg(
            avgdock=("dock_avg", "mean"),
            sr_strict=("strict", "mean"),
            validity=("validity", "mean"),
            diversity=("diversity", "mean"),
            qed=("qed", "mean"),
            sa=("sa_norm", "mean"),
        )
        for metric, col, (exp_mean, exp_std) in zip(metrics, agg_cols, pairs):
            actual_mean, actual_std = mean_std_from_series(agg[col])
            add(checks, f"{table_name}.{label}.{metric}.mean", actual_mean, exp_mean, path)
            add(checks, f"{table_name}.{label}.{metric}.std", actual_std, exp_std, path)


def check_weakest(checks: List[Dict[str, Any]]) -> None:
    path = RESULTS / "tables/supplement/table_t2_weakest_target.tex"
    if not path.exists():
        return
    expected = {
        "TamGen": (-6.017, 0.176),
        "PocketXMol": (-5.836, 0.211),
        "\\ourM{}-Single": (-6.421, 0.206),
        "\\ourM{}-SFT": (-6.753, 0.287),
        "\\ourM{}-DPO": (-7.048, 0.454),
    }
    for needle, (dock, sr) in expected.items():
        values = latex_row(path, needle)
        add(checks, f"weakest.{needle}.worst_dock", values[0], dock, path)
        add(checks, f"weakest.{needle}.SR_strict_worst", values[1], sr, path)


def check_winrate(checks: List[Dict[str, Any]]) -> None:
    path = RESULTS / "tables/supplement/table2_winrate.tex"
    if not path.exists():
        return
    expected = {
        "\\ourM{}-Single": (0.967, 0.987),
        "TamGen": (0.993, 0.993),
        "PocketXMol": (0.861, 0.894),
    }
    for needle, (full, avg) in expected.items():
        values = latex_row(path, needle)
        add(checks, f"winrate.{needle}.FullWinRate", values[0], full, path)
        add(checks, f"winrate.{needle}.AvgBetterRate", values[1], avg, path)


def check_scale(checks: List[Dict[str, Any]]) -> None:
    path = RESULTS / "tables/supplement/table4_scale_ablation.tex"
    if not path.exists():
        return
    expected = {
        "Data Records": (21040, 4871455, None),
        "Unique Ligands": (2047, 1649070, None),
        "AvgDock": (-5.610, 1.280, -7.347, 1.017, 3),
        "SR}_{\\mathrm{strict}": (0.148, 0.210, 0.475, 0.290, 3),
    }
    values = latex_row(path, "Data Records")
    add(checks, "scale.DataRecords.Struct", int(values[0]), 21040, path, decimals=None)
    add(checks, "scale.DataRecords.SFT", int(values[1]), 4871455, path, decimals=None)
    values = latex_row(path, "Unique Ligands")
    add(checks, "scale.UniqueLigands.Struct", int(values[0]), 2047, path, decimals=None)
    add(checks, "scale.UniqueLigands.SFT", int(values[1]), 1649070, path, decimals=None)
    values = latex_row(path, "AvgDock")
    for name, actual, expected_value in [
        ("scale.AvgDock.Struct.mean", values[0], -5.610),
        ("scale.AvgDock.Struct.std", values[1], 1.280),
        ("scale.AvgDock.SFT.mean", values[2], -7.347),
        ("scale.AvgDock.SFT.std", values[3], 1.017),
    ]:
        add(checks, name, actual, expected_value, path)
    values = latex_row(path, "SR}_{\\mathrm{strict}")
    for name, actual, expected_value in [
        ("scale.SR.Struct.mean", values[0], 0.148),
        ("scale.SR.Struct.std", values[1], 0.210),
        ("scale.SR.SFT.mean", values[2], 0.475),
        ("scale.SR.SFT.std", values[3], 0.290),
    ]:
        add(checks, name, actual, expected_value, path)


def parse_formatted_row(row: Dict[str, str], metric: str) -> float:
    return first_number(row[metric])


def finite_float(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return score if math.isfinite(score) else float("nan")


def mean_worst_target(path: Path) -> float:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    worst_values: List[float] = []
    for row in rows:
        scores: List[float] = []
        for key, value in row.items():
            if not key.startswith("score_"):
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isnan(score):
                scores.append(score)
        if scores:
            worst_values.append(max(scores))
    if not worst_values:
        raise ValueError(f"No target scores in {path}")
    return sum(worst_values) / len(worst_values)


def check_setwise_redock_results(checks: List[Dict[str, Any]]) -> None:
    gen_path = RESULTS / "analysis_inputs/module_ablation/set_interaction_ablation_generations.json"
    csv_path = RESULTS / "supplement/molecule_tables/set_interaction_ablation.csv"
    master_path = RESULTS / "supplement/molecule_tables/paper_molecule_table.csv"
    manifest_path = RESULTS / "analysis_inputs/module_ablation/manifest.json"
    counts_path = RESULTS / "supplement/molecule_tables/table_counts.json"

    generation = load_json(gen_path)
    expected_rows: Dict[Tuple[str, int], Tuple[str, float, float, float, float, float]] = {}
    bad_targets = 0
    finite_both = 0
    finite_avg = 0
    equal_both = 0
    for target_id, entry in generation.items():
        pockets = entry.get("active_pockets") or []
        scores_by_pocket = entry.get("dock_scores_by_pocket") or {}
        smiles_list = entry.get("generated_smiles") or []
        if (
            len(pockets) != 2
            or not isinstance(scores_by_pocket, dict)
            or not isinstance(smiles_list, list)
            or any(len(scores_by_pocket.get(pocket, [])) != len(smiles_list) for pocket in pockets)
        ):
            bad_targets += 1
            continue
        for rank, smiles in enumerate(smiles_list, start=1):
            dock_a = finite_float(scores_by_pocket[pockets[0]][rank - 1])
            dock_b = finite_float(scores_by_pocket[pockets[1]][rank - 1])
            values = [score for score in [dock_a, dock_b] if math.isfinite(score)]
            avg = sum(values) / len(values) if values else float("nan")
            worst = max(values) if values else float("nan")
            gap = abs(dock_a - dock_b) if math.isfinite(dock_a) and math.isfinite(dock_b) else float("nan")
            finite_avg += int(math.isfinite(avg))
            if math.isfinite(dock_a) and math.isfinite(dock_b):
                finite_both += 1
                equal_both += int(dock_a == dock_b)
            expected_rows[(target_id, rank)] = (smiles, dock_a, dock_b, avg, worst, gap)

    add(checks, "setwise_redock.generation.target_count", len(generation), 151, gen_path, decimals=None)
    add(checks, "setwise_redock.generation.row_count", len(expected_rows), 1510, gen_path, decimals=None)
    add(checks, "setwise_redock.generation.bad_target_shapes", bad_targets, 0, gen_path, decimals=None)
    add(checks, "setwise_redock.generation.finite_avg_pairs", finite_avg, 1308, gen_path, decimals=None)
    add(checks, "setwise_redock.generation.finite_both_pairs", finite_both, 1289, gen_path, decimals=None)
    add(checks, "setwise_redock.generation.equal_finite_both_pairs", equal_both, 0, gen_path, decimals=None)

    def check_csv(path: Path, label: str) -> None:
        rows = 0
        finite_both_rows = 0
        finite_avg_rows = 0
        equal_rows = 0
        mismatches = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("task") != "set_interaction_ablation":
                    continue
                rows += 1
                rank = int(row["rank"])
                expected = expected_rows.get((row["target_id"], rank))
                dock_a = finite_float(row.get("dock_a"))
                dock_b = finite_float(row.get("dock_b"))
                dock_avg = finite_float(row.get("dock_avg"))
                dock_worst = finite_float(row.get("dock_worst"))
                dock_gap = finite_float(row.get("dock_gap_ab"))
                if math.isfinite(dock_avg):
                    finite_avg_rows += 1
                if math.isfinite(dock_a) and math.isfinite(dock_b):
                    finite_both_rows += 1
                    equal_rows += int(dock_a == dock_b)
                if expected is None:
                    mismatches += 1
                    continue
                exp_smiles, exp_a, exp_b, exp_avg, exp_worst, exp_gap = expected
                for actual, expected_value in [
                    (dock_a, exp_a),
                    (dock_b, exp_b),
                    (dock_avg, exp_avg),
                    (dock_worst, exp_worst),
                    (dock_gap, exp_gap),
                ]:
                    if math.isfinite(actual) or math.isfinite(expected_value):
                        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-9):
                            mismatches += 1
                            break
                if row.get("smiles") != exp_smiles:
                    mismatches += 1

        add(checks, f"setwise_redock.{label}.row_count", rows, 1510, path, decimals=None)
        add(checks, f"setwise_redock.{label}.finite_avg_pairs", finite_avg_rows, 1308, path, decimals=None)
        add(checks, f"setwise_redock.{label}.finite_both_pairs", finite_both_rows, 1289, path, decimals=None)
        add(checks, f"setwise_redock.{label}.equal_finite_both_pairs", equal_rows, 0, path, decimals=None)
        add(checks, f"setwise_redock.{label}.generation_mismatches", mismatches, 0, path, decimals=None)

    check_csv(csv_path, "molecule_table")
    check_csv(master_path, "master_table")

    manifest = load_json(manifest_path)
    add(checks, "setwise_redock.manifest.generated_smiles_count", manifest["counts"]["generated_smiles_count"], 1510, manifest_path, decimals=None)
    add(checks, "setwise_redock.manifest.finite_docking_pairs", manifest["counts"]["finite_docking_pairs"], 1308, manifest_path, decimals=None)
    add(checks, "setwise_redock.manifest.finite_all_active_pairs", manifest["counts"]["finite_all_active_pairs"], 1289, manifest_path, decimals=None)
    table_counts = load_json(counts_path)["task_counts"]
    add(checks, "setwise_redock.table_counts.set_interaction_ablation", table_counts["set_interaction_ablation"], 1510, counts_path, decimals=None)


def check_json_artifacts(checks: List[Dict[str, Any]]) -> None:
    flex_path = RESULTS / "supplement/flexribbon/t2_rdkit_aligned_summary.json"
    flex = load_json(flex_path)
    for label, source_key, avg, worst in [
        ("Reference", "reference", -0.5409, -0.7726),
        ("PocketXMol", "PocketXMol", -0.4458, -0.6377),
        ("PolyTamGen-DPO", "PolyTamGen-DPO", 0.0594, -0.0559),
    ]:
        add(checks, f"flexribbon.{label}.Avg", flex[source_key]["AvgAffinity"], avg, flex_path, decimals=4)
        add(checks, f"flexribbon.{label}.Worst", flex[source_key]["WorstAffinity"], worst, flex_path, decimals=4)

    flex_dist_path = RESULTS / "supplement/flexribbon/t2_rdkit_aligned_distribution_summary.json"
    flex_dist = load_json(flex_dist_path)
    for label, avg, worst, count in [
        ("Reference ligand", -0.5409, -0.7726, 151),
        ("PocketXMol", -0.4458, -0.6377, 41794),
        ("PolyTamGen-DPO", 0.0594, -0.0559, 14621),
    ]:
        add(checks, f"flexribbon_distribution.{label}.Avg", flex_dist[label]["AvgAffinity"], avg, flex_dist_path, decimals=4)
        add(checks, f"flexribbon_distribution.{label}.Worst", flex_dist[label]["WorstAffinity"], worst, flex_dist_path, decimals=4)
        add(checks, f"flexribbon_distribution.{label}.n_instances", flex_dist[label]["n_instances"], count, flex_dist_path, decimals=None)

    cross_path = RESULTS / "supplement/cross_target/summary.json"
    cross = load_json(cross_path)
    top5 = cross["topk_summaries"]["5"]
    add(checks, "cross_target.paper_task_count", cross["paper_task_count"], 151, cross_path, decimals=None)
    add(
        checks,
        "cross_target.scored_task_count",
        top5["num_pairs"],
        150,
        cross_path,
        decimals=None,
        note="150 is the scored subset behind the paper's -8.038/-7.232 means.",
    )
    add(checks, "cross_target.AB_to_AB", top5["ab_to_ab_mean"], -8.038, cross_path)
    add(checks, "cross_target.AB_to_C", top5["ab_to_c_mean"], -7.232, cross_path)

    strain_path = RESULTS / "supplement/strain/summary.json"
    strain = load_json(strain_path)["topk_summaries"]["5"]
    add(checks, "strain.num_pairs", strain["num_pairs"], 151, strain_path, decimals=None)
    add(checks, "strain.mean_pose_energy_gap", strain["mean_pose_energy_gap"], 2.377, strain_path)
    add(checks, "strain.mean_worst_strain", strain["mean_worst_strain"], 113.143, strain_path)
    add(checks, "strain.pose_gap_le_5_rate", strain["pose_gap_le_5_rate"], 0.918, strain_path)
    add(checks, "strain.worst_strain_le_10_rate", strain["worst_strain_le_10_rate"], 0.194, strain_path)

    setwise_path = RESULTS / "supplement/module_ablation/set_interaction_ablation_table_row.json"
    setwise = load_json(setwise_path)
    for metric, expected in [
        ("AvgDock", -7.327),
        ("SR_strict", 0.387),
        ("Validity", 0.954),
        ("Diversity", 0.867),
        ("QED", 0.394),
        ("SA", 0.763),
    ]:
        add(checks, f"module_ablation.SetWiseOff.{metric}", parse_formatted_row(setwise, metric), expected, setwise_path)
    setwise_summary_path = RESULTS / "supplement/module_ablation/set_interaction_ablation_summary.json"
    setwise_summary = load_json(setwise_summary_path)
    setwise_weakest = setwise_summary["weakest_target_metrics"]
    add(checks, "module_ablation.SetWiseOff.worst_dock", setwise_weakest["worst_dock"]["mean"], -6.732, setwise_summary_path)
    add(
        checks,
        "module_ablation.SetWiseOff.SR_strict_worst@-7.0",
        setwise_weakest["SR_strict_worst@-7.0"]["mean"],
        0.252,
        setwise_summary_path,
    )
    add(
        checks,
        "module_ablation.SetWiseOff.finite_all_active_pairs",
        setwise_weakest["finite_all_active_pairs"],
        1289,
        setwise_summary_path,
        decimals=None,
    )

    linear_path = RESULTS / "supplement/module_ablation/linear_adapter_ablation_summary.json"
    linear_summary = load_json(linear_path)
    linear = linear_summary.get("paper_table_row")
    if linear is None:
        linear = {
            key: parse_formatted_row(linear_summary["table_metrics"]["formatted_row"], key)
            for key in ["AvgDock", "SR_strict", "Validity", "Diversity", "QED", "SA"]
        }
    for metric, expected in [
        ("AvgDock", -6.794),
        ("SR_strict", 0.348),
        ("Validity", 0.959),
        ("Diversity", 0.865),
        ("QED", 0.469),
        ("SA", 0.772),
    ]:
        add(checks, f"module_ablation.LinearAdaptor.{metric}", linear[metric], expected, linear_path)

    dpo_path = RESULTS / "supplement/dpo_reward_sensitivity/summary.json"
    dpo = load_json(dpo_path)["paper_table_rows"]
    for label, expected in [
        ("1QED", {"AvgDock": -7.653, "Validity": 0.980, "Diversity": 0.843, "QED": 0.538, "SA": 0.808}),
        ("10QED", {"AvgDock": -7.526, "Validity": 0.974, "Diversity": 0.831, "QED": 0.660, "SA": 0.822}),
    ]:
        for metric, value in expected.items():
            add(checks, f"dpo_reward.{label}.{metric}", dpo[label][metric], value, dpo_path)

    ref_path = RESULTS / "supplement/reference_redocking/reference_docking_summary.json"
    ref = load_json(ref_path)
    add(checks, "reference_redocking.T2.mean", ref["T2"]["mean_active_dock_over_groups"], -5.582, ref_path)
    add(checks, "reference_redocking.T3.mean", ref["T3"]["mean_active_dock_over_groups"], -5.958, ref_path)
    t2_csv = RESULTS / "supplement/reference_redocking/t2_reference_docking_per_group.csv"
    t3_csv = RESULTS / "supplement/reference_redocking/t3_reference_docking_per_group.csv"
    add(checks, "reference_redocking.T2.hardest_target", mean_worst_target(t2_csv), -5.161, t2_csv)
    add(checks, "reference_redocking.T3.hardest_target", mean_worst_target(t3_csv), -5.366, t3_csv)

def check_raw_counts(checks: List[Dict[str, Any]]) -> None:
    for label, rel, expected in [
        ("T2", "raw_generation/T2/PolyTamGen-DPO.json", 151),
        ("T3", "raw_generation/T3/PolyTamGen-DPO.json", 59),
    ]:
        path = RESULTS / rel
        data = load_json(path)
        add(checks, f"raw_generation.{label}.group_count", len(data), expected, path, decimals=None)


def check_figures(checks: List[Dict[str, Any]]) -> None:
    for rel in [
        "figures/flexribbon_affinity_avg_density.pdf",
        "figures/joint_dist_panels.pdf",
    ]:
        path = RESULTS / rel
        add(checks, f"figure_artifact.{rel}", path.exists(), True, path, decimals=None)
    for rel in [
        "supplement/flexribbon/t2_rdkit_aligned_per_instance.csv",
    ]:
        path = RESULTS / rel
        add(checks, f"result_artifact.{rel}", path.exists(), True, path, decimals=None)


def write_report(checks: List[Dict[str, Any]], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    failures = [c for c in checks if c["status"] == "fail"]
    warnings = [c for c in checks if c["status"] == "warn"]
    payload = {
        "status": "fail" if failures else "ok",
        "num_checks": len(checks),
        "num_failures": len(failures),
        "num_warnings": len(warnings),
        "checks": checks,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Paper Alignment Report",
        "",
        f"Status: {payload['status']}",
        f"Checks: {len(checks)}",
        f"Failures: {len(failures)}",
        f"Warnings: {len(warnings)}",
        "",
        "| status | check | actual | expected | source |",
        "|---|---|---:|---:|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c['status']} | {c['name']} | {c['actual']} | {c['expected']} | {c['source']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=RESULTS / "tables/recomputed/paper_alignment_report.json")
    parser.add_argument("--out-md", type=Path, default=RESULTS / "tables/recomputed/paper_alignment_report.md")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    checks: List[Dict[str, Any]] = []

    check_main_scored(checks, RESULTS / "scored/T2/per_molecule.json", "T2", MAIN_TABLE_EXPECTED["T2"])
    check_main_scored(checks, RESULTS / "scored/T3/per_molecule.json", "T3", MAIN_TABLE_EXPECTED["T3"])
    check_weakest(checks)
    check_winrate(checks)
    check_scale(checks)
    check_json_artifacts(checks)
    check_setwise_redock_results(checks)
    check_raw_counts(checks)
    check_figures(checks)

    failures = [c for c in checks if c["status"] == "fail"]
    warnings = [c for c in checks if c["status"] == "warn"]
    if not args.no_write:
        write_report(checks, args.out_json, args.out_md)
    if not args.quiet:
        print(f"paper alignment: {len(checks)} checks, {len(failures)} failures, {len(warnings)} warnings")
        if failures:
            for item in failures[:20]:
                print(f"FAIL {item['name']}: actual={item['actual']} expected={item['expected']}")
        if warnings:
            for item in warnings:
                print(f"WARN {item['name']}: actual={item['actual']} expected={item['expected']} {item['note']}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
