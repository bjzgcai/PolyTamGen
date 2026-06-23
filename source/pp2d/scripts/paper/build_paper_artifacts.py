#!/usr/bin/env python3
"""Build paper tables and figures from the result files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run(cmd: List[str], dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate available paper artifacts and verify paper alignment.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--include-consistency-tables", action="store_true")
    parser.add_argument(
        "--consistency-source",
        choices=["paper", "raw-rebuild"],
        default="paper",
        help=(
            "per_molecule source for optional consistency tables. "
            "'paper' uses results/scored files that reproduce the main paper tables; "
            "'raw-rebuild' uses the compact diagnostic rebuild under outputs/raw_generation_rebuild."
        ),
    )
    args = parser.parse_args()

    py = args.python

    paper_t2_per_molecule = Path("results/scored/T2/per_molecule.json")
    paper_t3_per_molecule = Path("results/scored/T3/per_molecule.json")
    raw_rebuild_t2_per_molecule = Path("outputs/raw_generation_rebuild/T2/per_molecule.json")
    raw_rebuild_t3_per_molecule = Path("outputs/raw_generation_rebuild/T3/per_molecule.json")
    if args.consistency_source == "paper":
        t2_per_molecule = paper_t2_per_molecule
        t3_per_molecule = paper_t3_per_molecule
    else:
        t2_per_molecule = raw_rebuild_t2_per_molecule
        t3_per_molecule = raw_rebuild_t3_per_molecule

    if args.include_consistency_tables and t2_per_molecule.exists() and t3_per_molecule.exists():
        run(
            [
                py,
                "scripts/paper/diagnostics/generate_consistency_tables_from_per_molecule.py",
                "--t2_path",
                str(t2_per_molecule),
                "--t3_path",
                str(t3_per_molecule),
                "--output_dir",
                "outputs/paper_rebuild/consistency_tables",
                "--num_workers",
                str(args.num_workers),
            ],
            args.dry_run,
        )
    elif args.include_consistency_tables:
        print(
            "skip molecule-level consistency table rebuild: missing "
            f"{args.consistency_source} per-molecule result JSON"
        )
    else:
        print("skip molecule-level consistency tables: pass --include-consistency-tables to generate them")

    per_target = Path("results/scored/T2/per_target.csv")
    if args.include_diagnostics and per_target.exists():
        run(
            [
                py,
                "scripts/paper/diagnostics/generate_tables.py",
                "--per-target",
                str(per_target),
                "--out-dir",
                "outputs/paper_rebuild/tables/diagnostics",
                "--n-boot",
                "2000",
            ],
            args.dry_run,
        )
        run(
            [
                py,
                "scripts/paper/diagnostics/assemble_table_bundle.py",
                "--in-dir",
                "outputs/paper_rebuild/tables/diagnostics",
                "--out-dir",
                "outputs/paper_rebuild/tables/diagnostic_bundle",
            ],
            args.dry_run,
        )
    else:
        print("skip table diagnostics: pass --include-diagnostics to generate them")

    if paper_t2_per_molecule.exists():
        run(
            [
                py,
                "scripts/paper/plot_joint_distribution.py",
                "--t2_path",
                str(paper_t2_per_molecule),
                "--output",
                "outputs/paper_rebuild/figures/joint_dist_panels.pdf",
            ],
            args.dry_run,
        )
    else:
        print(f"skip joint distribution plot: missing {paper_t2_per_molecule}")

    flex_inputs = [
        Path("results/analysis_inputs/flexribbon_affinity/polytamgen_dpo_affinity_input.tsv"),
        Path("results/analysis_inputs/flexribbon_affinity/pocketxmol_affinity_input.tsv"),
        Path("results/supplement/flexribbon/ours_generated_merged.tsv"),
        Path("results/supplement/flexribbon/baseline_generated_merged.tsv"),
        Path("results/supplement/flexribbon/reference_merged.tsv"),
        Path("results/supplement/flexribbon/t2_rdkit_aligned_per_instance.csv"),
    ]
    missing_flex_inputs = [path for path in flex_inputs if not path.exists()]
    if not missing_flex_inputs:
        if args.include_diagnostics:
            run(
                [
                    py,
                    "scripts/paper/plot_flexribbon_affinity.py",
                    "--output",
                    "outputs/paper_rebuild/figures/flexribbon_affinity.pdf",
                ],
                args.dry_run,
            )
        run(
            [
                py,
                "scripts/paper/plot_flexribbon_split_views.py",
                "--avg-ecdf-output",
                "outputs/paper_rebuild/figures/flexribbon_affinity_avg_ecdf.pdf",
                "--avg-density-output",
                "outputs/paper_rebuild/figures/flexribbon_affinity_avg_density.pdf",
                "--weakest-ecdf-output",
                "outputs/paper_rebuild/figures/flexribbon_affinity_weakest_ecdf.pdf",
            ],
            args.dry_run,
        )
    else:
        print(
            "skip FlexRibbon affinity plot: missing "
            + ", ".join(str(path) for path in missing_flex_inputs)
        )

    run([py, "scripts/paper/verify_paper_alignment.py"], args.dry_run)


if __name__ == "__main__":
    main()
