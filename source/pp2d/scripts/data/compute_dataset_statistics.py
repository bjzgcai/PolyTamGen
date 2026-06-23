#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute paper-ready dataset statistics from files in this repo.

This script is intentionally lightweight and uses chunked I/O for large CSV/TSV.
It prints a JSON block plus a compact human-readable summary.

Targets:
  - data/sft_train_valid/Merged_Deduplicated.csv
  - data/sft_train_valid/ligand_profiles.pkl
  - optional 3D multi-target train TSV when available
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Set, Tuple


def _human_int(x: Optional[int]) -> str:
    if x is None:
        return "N/A"
    return f"{x:,}"


def _safe_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _iter_table_unique(
    path: str,
    sep: str,
    smiles_col: str,
    target_cols: Tuple[str, ...] = (),
    chunksize: int = 500_000,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (n_rows, n_unique_smiles, n_unique_targets).

    Works for very large files via pandas chunking.
    """

    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required for this script") from e

    if not _safe_exists(path):
        return None, None, None

    # Determine which columns exist to avoid pandas usecols errors.
    header = pd.read_csv(path, sep=sep, nrows=0)
    cols = set(str(c) for c in header.columns)

    if smiles_col not in cols:
        return None, None, None

    present_targets = tuple(c for c in target_cols if c in cols)
    usecols = [smiles_col, *present_targets]
    seen_smiles: Set[str] = set()
    seen_targets: Set[str] = set()
    n_rows = 0

    for chunk in pd.read_csv(path, sep=sep, usecols=usecols, chunksize=chunksize):
        n_rows += len(chunk)

        s = chunk.get(smiles_col)
        if s is not None:
            for v in s.dropna().astype(str).values.tolist():
                vv = v.strip()
                if vv and vv.lower() != "nan":
                    seen_smiles.add(vv)

        for col in present_targets:
            if col not in chunk.columns:
                continue
            for v in chunk[col].dropna().astype(str).values.tolist():
                vv = v.strip()
                if vv and vv.lower() != "nan":
                    seen_targets.add(vv)

    return n_rows, len(seen_smiles), len(seen_targets) if present_targets else None


def _analyze_ligand_profiles_pkl(path: str) -> Dict[str, Optional[int]]:
    if not _safe_exists(path):
        return {
            "n_profiles": None,
            "n_unique_ligands": None,
            "n_unique_targets": None,
            "n_total_active_links": None,
        }

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, list):
        raise ValueError(f"Unexpected PKL format: expected list, got {type(obj)}")

    smiles_set: Set[str] = set()
    target_set: Set[str] = set()
    total_links = 0

    for item in obj:
        if not isinstance(item, dict):
            continue
        smi = str(item.get("SMILES", "")).strip()
        if smi and smi.lower() != "nan":
            smiles_set.add(smi)

        act = item.get("Active_Protein_IDs", [])
        if isinstance(act, (list, tuple)):
            total_links += len(act)
            for t in act:
                tt = str(t).strip()
                if tt and tt.lower() != "nan":
                    target_set.add(tt)

    return {
        "n_profiles": len(obj),
        "n_unique_ligands": len(smiles_set),
        "n_unique_targets": len(target_set),
        "n_total_active_links": total_links,
    }


@dataclass
class Stats:
    bioactivity_rows: Optional[int]
    bioactivity_unique_ligands: Optional[int]
    bioactivity_unique_targets: Optional[int]

    profiles_n: Optional[int]
    profiles_unique_ligands: Optional[int]
    profiles_unique_targets: Optional[int]
    profiles_total_active_links: Optional[int]

    pdb_train_rows: Optional[int]
    pdb_train_unique_ligands: Optional[int]
    pdb_train_unique_targets: Optional[int]


def compute(
    bioactivity_csv: str,
    profiles_pkl: str,
    pdb_train_tsv: str,
) -> Stats:
    # Bioactivity table: best-effort column detection.
    # We try common SMILES column names.
    bio_rows = bio_ul = bio_ut = None
    if _safe_exists(bioactivity_csv):
        # Heuristic: try SMILES first, then smiles.
        for smiles_col in ("SMILES", "smiles"):
            try:
                bio_rows, bio_ul, bio_ut = _iter_table_unique(
                    bioactivity_csv,
                    sep=",",
                    smiles_col=smiles_col,
                    # Target column name is dataset-dependent; keep optional.
                    target_cols=(
                        "ID",  # Merged_Deduplicated.csv uses UniProt-like IDs
                        "target",
                        "Target",
                        "target_id",
                        "Target_ID",
                        "protein_id",
                        "Protein_ID",
                    ),
                )
                # If we succeeded in reading rows, stop.
                if bio_rows is not None:
                    break
            except ValueError:
                continue

        # If none of the target columns exist, _iter_table_unique returns 0 targets; treat as N/A.
        if bio_ut == 0:
            bio_ut = None

    prof = _analyze_ligand_profiles_pkl(profiles_pkl)

    # 3D PDB-based train.tsv: assume columns L/P1/P2.
    if pdb_train_tsv:
        pdb_rows, pdb_ul, pdb_ut = _iter_table_unique(
            pdb_train_tsv,
            sep="\t",
            smiles_col="L",
            target_cols=("P1", "P2"),
        )
    else:
        pdb_rows = pdb_ul = pdb_ut = None

    return Stats(
        bioactivity_rows=bio_rows,
        bioactivity_unique_ligands=bio_ul,
        bioactivity_unique_targets=bio_ut,
        profiles_n=prof["n_profiles"],
        profiles_unique_ligands=prof["n_unique_ligands"],
        profiles_unique_targets=prof["n_unique_targets"],
        profiles_total_active_links=prof["n_total_active_links"],
        pdb_train_rows=pdb_rows,
        pdb_train_unique_ligands=pdb_ul,
        pdb_train_unique_targets=pdb_ut,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute dataset statistics for paper tables")
    ap.add_argument(
        "--bioactivity_csv",
        default="data/sft_train_valid/Merged_Deduplicated.csv",
        help="Curated bioactivity table (CSV).",
    )
    ap.add_argument(
        "--profiles_pkl",
        default="data/sft_train_valid/ligand_profiles.pkl",
        help="Ligand profile dataset (PKL).",
    )
    ap.add_argument(
        "--pdb_train_tsv",
        default=None,
        help="Optional 3D multi-target train split TSV. Omit when only test splits are available.",
    )
    args = ap.parse_args()

    s = compute(args.bioactivity_csv, args.profiles_pkl, args.pdb_train_tsv)
    print(json.dumps(asdict(s), indent=2, sort_keys=True))

    print("\n[Paper table values]")
    print(f"Bioactivity table: rows={_human_int(s.bioactivity_rows)}  unique_ligands={_human_int(s.bioactivity_unique_ligands)}  unique_targets={_human_int(s.bioactivity_unique_targets)}")
    print(f"Ligand profiles:  profiles={_human_int(s.profiles_n)}  unique_ligands={_human_int(s.profiles_unique_ligands)}  unique_targets={_human_int(s.profiles_unique_targets)}  active_links={_human_int(s.profiles_total_active_links)}")
    print(f"3D train.tsv:     rows={_human_int(s.pdb_train_rows)}  unique_ligands={_human_int(s.pdb_train_unique_ligands)}  unique_targets={_human_int(s.pdb_train_unique_targets)}  (optional input)")


if __name__ == "__main__":
    main()
