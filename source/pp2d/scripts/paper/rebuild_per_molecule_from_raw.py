#!/usr/bin/env python3
"""Rebuild molecule-level scored JSONs from raw generations.

The raw generation JSONs already retain generated SMILES and per-pocket docking
scores. This script computes RDKit properties directly from the SMILES and
writes compact molecule-level diagnostic files.

The scored files under results/scored are the full benchmark molecule-level
sources used for paper table reproduction. This diagnostic rebuild writes under
outputs/ by default so it cannot overwrite those scored files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, RDConfig, rdFingerprintGenerator

import sys

sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer  # type: ignore  # noqa: E402


ResultEntry = Dict[str, Any]
PocketKey = Tuple[str, Tuple[str, ...]]

RDLogger.DisableLog("rdApp.*")
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


T2_MULTI = {
    "PolyTamGen-Struct": "raw_generation/T2/PolyTamGen-Struct.json",
    "PolyTamGen-SFT": "raw_generation/T2/PolyTamGen-SFT.json",
    "PolyTamGen-DPO": "raw_generation/T2/PolyTamGen-DPO.json",
}

T2_SINGLE = {
    "PolyTamGen-Single": "raw_generation/T2/PolyTamGen-Single.json",
    "TamGen": "raw_generation/T2/TamGen.json",
    "PocketXMol": "raw_generation/T2/PocketXMol.json",
}

T3_MULTI = {
    "PolyTamGen-SFT": "raw_generation/T3/PolyTamGen-SFT.json",
    "PolyTamGen-DPO": "raw_generation/T3/PolyTamGen-DPO.json",
}

T3_SINGLE = {
    "TamGen": "raw_generation/T3/TamGen.json",
    "PocketXMol": "raw_generation/T3/PocketXMol.json",
}


def _load_json(path: Path) -> Dict[str, ResultEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected target-id keyed JSON object: {path}")
    return data


def _is_valid_score(value: Any) -> bool:
    if not isinstance(value, (float, int, np.floating, np.integer)):
        return False
    val = float(value)
    return np.isfinite(val) and val < 0 and val > -200


def _score_or_nan(value: Any) -> float:
    return float(value) if _is_valid_score(value) else float("nan")


def _canonical_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _entry_pocket_ids(entry: Mapping[str, Any], n_pockets: int) -> Optional[Tuple[str, ...]]:
    dock_map = entry.get("dock_scores_by_pocket") or {}
    if not isinstance(dock_map, dict) or len(dock_map) < n_pockets:
        return None

    active = [pid for pid in (entry.get("active_pockets") or []) if isinstance(pid, str) and pid]
    if len(active) >= n_pockets and all(pid in dock_map for pid in active[:n_pockets]):
        return tuple(active[:n_pockets])

    return tuple(sorted(dock_map)[:n_pockets])


def _entry_key(entry: Mapping[str, Any], n_pockets: int) -> Optional[PocketKey]:
    ref = entry.get("reference_smiles")
    if not isinstance(ref, str) or not ref.strip():
        return None
    pocket_ids = _entry_pocket_ids(entry, n_pockets)
    if pocket_ids is None:
        return None
    return ref, tuple(pocket_ids)


def _aligned_entry(entry: Mapping[str, Any], pocket_ids: Sequence[str]) -> Optional[ResultEntry]:
    generated = entry.get("generated_smiles") or []
    dock_map = entry.get("dock_scores_by_pocket") or {}
    if not isinstance(generated, list) or not generated or not isinstance(dock_map, dict):
        return None

    lengths = [len(generated)]
    for pid in pocket_ids:
        scores = dock_map.get(pid)
        if not isinstance(scores, list):
            return None
        lengths.append(len(scores))
    n = min(lengths)
    if n <= 0:
        return None

    return {
        "reference_smiles": entry.get("reference_smiles", ""),
        "generated_smiles": generated[:n],
        "dock_scores_by_pocket": {pid: list(dock_map[pid])[:n] for pid in pocket_ids},
        "active_pockets": list(pocket_ids),
    }


def _build_union_map(data: Mapping[str, ResultEntry], n_pockets: int) -> Dict[PocketKey, ResultEntry]:
    union: Dict[PocketKey, ResultEntry] = {}
    for entry in data.values():
        parsed = _entry_key(entry, n_pockets)
        if parsed is None:
            continue
        ref, pocket_ids = parsed
        aligned = _aligned_entry(entry, pocket_ids)
        if aligned is None:
            continue

        out = union.get(parsed)
        if out is None:
            out = {
                "reference_smiles": ref,
                "generated_smiles": [],
                "dock_scores_by_pocket": {pid: [] for pid in pocket_ids},
                "active_pockets": list(pocket_ids),
            }
            union[parsed] = out
        out["generated_smiles"].extend(aligned["generated_smiles"])
        for pid in pocket_ids:
            out["dock_scores_by_pocket"][pid].extend(aligned["dock_scores_by_pocket"][pid])
    return union


def _build_source_map(data: Mapping[str, ResultEntry], n_pockets: int) -> Dict[Tuple[PocketKey, str], ResultEntry]:
    out: Dict[Tuple[PocketKey, str], ResultEntry] = {}
    for entry in data.values():
        parsed = _entry_key(entry, n_pockets)
        if parsed is None:
            continue
        pocket_ids = parsed[1]
        source_pocket = next(
            (pid for pid in (entry.get("active_pockets") or []) if isinstance(pid, str) and pid in pocket_ids),
            "",
        )
        if not source_pocket:
            continue
        aligned = _aligned_entry(entry, pocket_ids)
        if aligned is None:
            continue
        out[(parsed, source_pocket)] = aligned
    return out


def _iter_multi_entries(data: Mapping[str, ResultEntry], n_pockets: int) -> Iterable[Tuple[PocketKey, ResultEntry]]:
    for entry in data.values():
        parsed = _entry_key(entry, n_pockets)
        if parsed is None:
            continue
        aligned = _aligned_entry(entry, parsed[1])
        if aligned is None:
            continue
        yield parsed, aligned


def _mol_properties(smiles: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if smiles in cache:
        return cache[smiles]
    props: Dict[str, Any] = {
        "canonical_smiles": "",
        "validity": 0.0,
        "valid_mol": False,
        "sas": float("nan"),
        "sa_norm": float("nan"),
        "qed": float("nan"),
        "logp": float("nan"),
        "good_logp": False,
        "molwt": float("nan"),
        "num_atoms": float("nan"),
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
            sas = float(sascorer.calculateScore(mol))
            logp = float(Crippen.MolLogP(mol))
            props.update(
                {
                    "canonical_smiles": Chem.MolToSmiles(mol),
                    "validity": 1.0,
                    "valid_mol": True,
                    "sas": sas,
                    "sa_norm": 1.0 - (sas - 1.0) / 9.0,
                    "qed": float(QED.qed(mol)),
                    "logp": logp,
                    "good_logp": 0 <= logp <= 5,
                    "molwt": float(Descriptors.MolWt(mol)),
                    "num_atoms": float(mol.GetNumAtoms()),
                }
            )
    except Exception:
        pass
    cache[smiles] = props
    return props


@lru_cache(maxsize=None)
def _fingerprint_for_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MORGAN_GENERATOR.GetFingerprint(mol)
    except Exception:
        return None


def _group_diversity(smiles_list: Sequence[str]) -> float:
    fps = []
    for smiles in smiles_list:
        fp = _fingerprint_for_smiles(smiles)
        if fp is not None:
            fps.append(fp)
    if len(fps) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, fp in enumerate(fps[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :])
        total += float(sum(similarities))
        count += len(similarities)
    return 1.0 - total / count if count else 0.0


def _records_for_entry(
    *,
    method: str,
    target_id: str,
    ref_smiles: str,
    pocket_ids: Sequence[str],
    entry: Mapping[str, Any],
    prop_cache: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    generated = entry.get("generated_smiles") or []
    dock_map = entry.get("dock_scores_by_pocket") or {}
    if not generated or not isinstance(dock_map, dict):
        return []

    diversity = _group_diversity([s for s in generated if isinstance(s, str)])
    rows: List[Dict[str, Any]] = []
    for idx, smiles in enumerate(generated):
        if not isinstance(smiles, str):
            continue
        props = _mol_properties(smiles, prop_cache)
        scores = [_score_or_nan((dock_map.get(pid) or [float("nan")])[idx]) for pid in pocket_ids]
        finite_scores = [score for score in scores if math.isfinite(score)]
        dock_avg = float(np.mean(finite_scores)) if finite_scores else float("nan")
        dock_worst = float(np.max(finite_scores)) if finite_scores else float("nan")

        row: Dict[str, Any] = {
            "method": method,
            "target_id": target_id,
            "rank": idx + 1,
            "ref_smiles": ref_smiles,
            "smiles": smiles,
            "diversity": diversity,
            "dock_avg": dock_avg,
            "dock_worst": dock_worst,
        }
        for pos, pid in enumerate(pocket_ids, start=1):
            row[f"p{pos}"] = pid
            row[f"dock_p{pos}"] = scores[pos - 1]
        row.update(props)
        rows.append(row)
    return rows


def _target_id(ref: str, pocket_ids: Sequence[str]) -> str:
    safe_ref = re.sub(r"\s+", "", ref)[:32]
    return f"{safe_ref}__{'__'.join(pocket_ids)}"


def _build_split(
    *,
    results_root: Path,
    multi_specs: Mapping[str, str],
    single_specs: Mapping[str, str],
    n_pockets: int,
) -> pd.DataFrame:
    prop_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for method, rel_path in multi_specs.items():
        data = _load_json(results_root / rel_path)
        for (ref, pocket_ids), entry in _iter_multi_entries(data, n_pockets):
            tid = _target_id(ref, pocket_ids)
            rows.extend(
                _records_for_entry(
                    method=method,
                    target_id=tid,
                    ref_smiles=ref,
                    pocket_ids=pocket_ids,
                    entry=entry,
                    prop_cache=prop_cache,
                )
            )

    for method_prefix, rel_path in single_specs.items():
        data = _load_json(results_root / rel_path)
        union = _build_union_map(data, n_pockets)
        source_map = _build_source_map(data, n_pockets)

        for (ref, pocket_ids), entry in union.items():
            tid = _target_id(ref, pocket_ids)
            rows.extend(
                _records_for_entry(
                    method=method_prefix,
                    target_id=tid,
                    ref_smiles=ref,
                    pocket_ids=pocket_ids,
                    entry=entry,
                    prop_cache=prop_cache,
                )
            )

        for (ref, pocket_ids), _ in union.items():
            for pos, pid in enumerate(pocket_ids, start=1):
                entry = source_map.get(((ref, pocket_ids), pid))
                if entry is None:
                    continue
                rows.extend(
                    _records_for_entry(
                        method=f"{method_prefix}-p{pos}",
                        target_id=_target_id(ref, pocket_ids),
                        ref_smiles=ref,
                        pocket_ids=pocket_ids,
                        entry=entry,
                        prop_cache=prop_cache,
                    )
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_cols = [c for c in ["method", "target_id", "rank"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("method"):
        rows.append(
            {
                "method": method,
                "rows": len(sub),
                "target_count": sub["target_id"].nunique(),
                "validity": sub["validity"].mean(),
                "qed": sub["qed"].mean(),
                "sa_norm": sub["sa_norm"].mean(),
                "logp": sub["logp"].mean(),
                "molwt": sub["molwt"].mean(),
                "dock_avg": sub["dock_avg"].mean(),
                "dock_worst": sub["dock_worst"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("method")


def _write_json_records(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", indent=2)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--t2-output", type=Path, default=Path("outputs/raw_generation_rebuild/T2/per_molecule.json"))
    parser.add_argument("--t3-output", type=Path, default=Path("outputs/raw_generation_rebuild/T3/per_molecule.json"))
    args = parser.parse_args()

    t2 = _build_split(results_root=args.results_root, multi_specs=T2_MULTI, single_specs=T2_SINGLE, n_pockets=2)
    t3 = _build_split(results_root=args.results_root, multi_specs=T3_MULTI, single_specs=T3_SINGLE, n_pockets=3)

    _write_json_records(t2, args.t2_output)
    _write_json_records(t3, args.t3_output)

    _summary(t2).to_csv(args.t2_output.parent / "rdkit_property_summary.csv", index=False)
    _summary(t3).to_csv(args.t3_output.parent / "rdkit_property_summary.csv", index=False)

    manifest = {
        "schema_version": "per-molecule-rdkit-v1",
        "inputs": {
            "T2": {**T2_MULTI, **T2_SINGLE},
            "T3": {**T3_MULTI, **T3_SINGLE},
        },
        "outputs": {
            "T2": str(args.t2_output),
            "T3": str(args.t3_output),
            "T2_summary": str(args.t2_output.parent / "rdkit_property_summary.csv"),
            "T3_summary": str(args.t3_output.parent / "rdkit_property_summary.csv"),
        },
        "metric_definitions": {
            "validity": "RDKit sanitization success.",
            "qed": "rdkit.Chem.QED.qed.",
            "sas": "RDKit contrib SA_Score sascorer.calculateScore.",
            "sa_norm": "1 - (sas - 1) / 9.",
            "diversity": "1 - mean pairwise Morgan fingerprint Tanimoto within each target/method group.",
            "dock_avg": "Mean of finite per-pocket docking scores for the molecule.",
            "dock_worst": "Worst finite per-pocket docking score for the molecule.",
        },
    }
    (args.t2_output.parent.parent / "per_molecule_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"T2 molecule rows: {len(t2)} -> {args.t2_output}")
    print(f"T3 molecule rows: {len(t3)} -> {args.t3_output}")


if __name__ == "__main__":
    main()
