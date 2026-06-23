#!/usr/bin/env python3
"""Plot aligned per-molecule FlexRibbon affinity distributions for T2."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


METHOD_ORDER = ["Reference ligand", "PocketXMol", "PolyTamGen-DPO"]
PALETTE = {
    "Reference ligand": "#8a8a8a",
    "PocketXMol": "#d98c2b",
    "PolyTamGen-DPO": "#2b6cb0",
}

RDLogger.DisableLog("rdApp.*")
UNCHARGER = rdMolStandardize.Uncharger()


def read_rows(path: Path, delimiter: str = "\t") -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


@lru_cache(maxsize=None)
def standardize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = UNCHARGER.uncharge(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_pred_map(path: Path, smiles_key: str) -> Dict[Tuple[str, str], float]:
    pred_map: Dict[Tuple[str, str], float] = {}
    for row in read_rows(path):
        pred = row.get("pred_affinity", "")
        if not pred:
            continue
        std = standardize_smiles(row[smiles_key])
        if std is None:
            continue
        pred_map[(std, row["protein_sequence"])] = float(pred)
    return pred_map


def summarize_generated(
    input_path: Path,
    pred_map: Dict[Tuple[str, str], float],
    method_label: str,
) -> List[Dict[str, object]]:
    rows = read_rows(input_path)
    per_instance: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for row in rows:
        std = standardize_smiles(row["generated_smiles"])
        if std is None:
            continue
        per_instance[(row["entry_id"], row["generated_index"], std)].append(row["protein_sequence"])

    records: List[Dict[str, object]] = []
    for (entry_id, generated_index, std), proteins in per_instance.items():
        scores: List[float] = []
        for protein in proteins:
            val = pred_map.get((std, protein))
            if val is None:
                scores = []
                break
            scores.append(val)
        if not scores:
            continue
        records.append(
            {
                "method": method_label,
                "entry_id": entry_id,
                "generated_index": generated_index,
                "standardized_smiles": std,
                "n_targets": len(scores),
                "avg_affinity": mean(scores),
                "worst_affinity": min(scores),
            }
        )
    return records


def summarize_reference(
    input_path: Path,
    pred_map: Dict[Tuple[str, str], float],
) -> List[Dict[str, object]]:
    rows = read_rows(input_path)
    per_entry: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in rows:
        std = standardize_smiles(row["reference_smiles"])
        if std is None:
            continue
        per_entry[(row["entry_id"], std)].append(row["protein_sequence"])

    records: List[Dict[str, object]] = []
    for (entry_id, std), proteins in per_entry.items():
        scores: List[float] = []
        for protein in proteins:
            val = pred_map.get((std, protein))
            if val is None:
                scores = []
                break
            scores.append(val)
        if not scores:
            continue
        records.append(
            {
                "method": "Reference ligand",
                "entry_id": entry_id,
                "generated_index": "reference",
                "standardized_smiles": std,
                "n_targets": len(scores),
                "avg_affinity": mean(scores),
                "worst_affinity": min(scores),
            }
        )
    return records


def build_distribution(args: argparse.Namespace) -> pd.DataFrame:
    ours_map = load_pred_map(args.ours_merged, "generated_smiles")
    baseline_map = load_pred_map(args.baseline_merged, "generated_smiles")
    reference_map = load_pred_map(args.reference_merged, "reference_smiles")

    records: List[Dict[str, object]] = []
    records.extend(summarize_reference(args.ours_input, reference_map))
    records.extend(summarize_generated(args.pocketxmol_input, baseline_map, "PocketXMol"))
    records.extend(summarize_generated(args.ours_input, ours_map, "PolyTamGen-DPO"))
    return pd.DataFrame.from_records(records)


def write_outputs(df: pd.DataFrame, per_instance_output: Path, summary_output: Path) -> None:
    per_instance_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(per_instance_output, index=False)
    summary: Dict[str, Dict[str, float | int]] = {}
    for method, sub in df.groupby("method"):
        summary[method] = {
            "AvgAffinity": round(float(sub["avg_affinity"].mean()), 4),
            "WorstAffinity": round(float(sub["worst_affinity"].mean()), 4),
            "MedianAvgAffinity": round(float(sub["avg_affinity"].median()), 4),
            "MedianWorstAffinity": round(float(sub["worst_affinity"].median()), 4),
            "n_instances": int(len(sub)),
        }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_distribution(df: pd.DataFrame, output: Path, png_output: Path | None) -> None:
    plot_df = df.melt(
        id_vars=["method", "entry_id", "generated_index"],
        value_vars=["avg_affinity", "worst_affinity"],
        var_name="metric",
        value_name="affinity",
    )
    metric_titles = {
        "avg_affinity": "Average over intended targets",
        "worst_affinity": "Weakest intended target",
    }

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    y_min = float(plot_df["affinity"].quantile(0.005))
    y_max = float(plot_df["affinity"].quantile(0.995))
    pad = 0.08 * (y_max - y_min)
    x_limits = (y_min - pad, y_max + pad)

    for ax, metric in zip(axes, ["avg_affinity", "worst_affinity"]):
        sub = plot_df[plot_df["metric"] == metric]
        for method in METHOD_ORDER:
            vals = sub[sub["method"] == method]["affinity"].to_numpy()
            vals = np.sort(vals)
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(
                vals,
                y,
                where="post",
                color=PALETTE[method],
                linewidth=1.8,
                label=method,
                alpha=0.95,
            )
            median = float(np.median(vals))
            ax.plot(
                [median],
                [0.5],
                marker="o",
                markersize=3.5,
                color=PALETTE[method],
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

        ax.axvline(0, color="#505050", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(0.5, color="#cbd5e1", linewidth=0.7, linestyle=":", alpha=0.9)
        ax.set_title(metric_titles[metric], fontweight="bold", pad=8)
        ax.set_xlabel("FlexRibbon predicted affinity")
        ax.set_xlim(*x_limits)
        ax.set_ylim(0, 1)
        ax.grid(axis="both", color="#e5e7eb", linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Cumulative fraction")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.5,
    )
    fig.suptitle("Aligned T2 FlexRibbon affinity distributions", y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.9, w_pad=1.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if png_output is not None:
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ours-input",
        type=Path,
        default=Path("results/analysis_inputs/flexribbon_affinity/polytamgen_dpo_affinity_input.tsv"),
    )
    parser.add_argument(
        "--pocketxmol-input",
        type=Path,
        default=Path("results/analysis_inputs/flexribbon_affinity/pocketxmol_affinity_input.tsv"),
    )
    parser.add_argument(
        "--ours-merged",
        type=Path,
        default=Path("results/supplement/flexribbon/ours_generated_merged.tsv"),
    )
    parser.add_argument(
        "--baseline-merged",
        type=Path,
        default=Path("results/supplement/flexribbon/baseline_generated_merged.tsv"),
    )
    parser.add_argument(
        "--reference-merged",
        type=Path,
        default=Path("results/supplement/flexribbon/reference_merged.tsv"),
    )
    parser.add_argument(
        "--per-instance-output",
        type=Path,
        default=Path("results/supplement/flexribbon/t2_rdkit_aligned_per_instance.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/supplement/flexribbon/t2_rdkit_aligned_distribution_summary.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_rebuild/figures/flexribbon_affinity.pdf"))
    parser.add_argument("--png-output", type=Path, default=None)
    args = parser.parse_args()

    df = build_distribution(args)
    if df.empty:
        raise SystemExit("No aligned FlexRibbon instances were recovered.")
    write_outputs(df, args.per_instance_output, args.summary_output)
    plot_distribution(df, args.output, args.png_output)
    print(args.summary_output.read_text(encoding="utf-8").rstrip())


if __name__ == "__main__":
    main()
