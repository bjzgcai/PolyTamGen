#!/usr/bin/env python3
"""
Prepare benchmark-ready results for all models listed in a config.

For single-target baselines, rebuild docking scores from cache.
For multi_target models, write a benchmark_ready copy of the input results.

Usage:
  python scripts/benchmark/prepare_benchmark_ready_from_config.py \
    --config scripts/benchmark/configs/t2_inputs.yaml \
    --pdb_dir /path/to/pdb_root \
    --docking_cache_dir /path/to/docking_cache \
    --output_root benchmark_ready_outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import yaml
except ImportError:
    raise RuntimeError("PyYAML is required. Please install it with `pip install PyYAML`.")

try:
    from rdkit import Chem
except Exception as e:
    raise RuntimeError("RDKit is required to canonicalize SMILES.") from e

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_pocket_file_path(pocket_id: str, pdb_root_dir: str) -> str:
    import os
    import re

    m = re.match(r"^([a-zA-Z0-9]{4})(_\w+)?$", pocket_id)
    if not m:
        raise ValueError(f"Invalid pocket_id: {pocket_id}")
    pdb_id = m.group(1)
    suffix = m.group(2) if m.group(2) else ""
    nested_dir = os.path.join(pdb_root_dir, pdb_id[:2].lower(), pdb_id.lower())
    path = os.path.join(nested_dir, f"{pdb_id.lower()}_pocket{suffix}.pdb")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return path


def load_results(path: Path) -> Any:
    if path.suffix == ".json":
        with path.open("r") as f:
            return json.load(f)
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            return pkl.load(f)
    raise ValueError(f"Unsupported results format: {path}")


def save_results(data: Any, out_json: Path, out_pkl: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(data, f, indent=2)
    with out_pkl.open("wb") as f:
        pkl.dump(data, f)


def cano_smiles(smiles: str) -> Optional[str]:
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def cache_file_for(
    receptor_path: str,
    ligand_smiles_cano: str,
    cache_dir: str,
    num_ligand_confs: int = 1,
    exhaustiveness: int = 32,
) -> str:
    unique_key = f"{receptor_path}|{ligand_smiles_cano}|{num_ligand_confs}|{exhaustiveness}"
    unique_hash = hashlib.md5(unique_key.encode()).hexdigest()
    receptor_name = Path(receptor_path).stem
    return str(Path(cache_dir) / f"{receptor_name}_{unique_hash}.pkl")


def is_valid_score(value: Any) -> bool:
    if not isinstance(value, (float, int, np.floating, np.integer)):
        return False
    val = float(value)
    return np.isfinite(val) and val < 0 and val > -200


def load_cached_score(receptor_path: str, ligand_smiles_raw: str, cache_dir: str) -> float:
    cano = cano_smiles(ligand_smiles_raw)
    if cano is None:
        return float("nan")
    path = cache_file_for(receptor_path, cano, cache_dir)
    p = Path(path)
    if not p.exists():
        return float("nan")
    try:
        with p.open("rb") as f:
            return float(pkl.load(f))
    except Exception:
        return float("nan")


def compute_missing_score(receptor_path: str, ligand_smiles_raw: str, cache_dir: str) -> float:
    try:
        from src.eval.dock_utils import dock

        val = dock(receptor_path=receptor_path, ligand_identifier=ligand_smiles_raw, cache_dir=cache_dir)
        return float(val) if is_valid_score(val) else float("nan")
    except Exception:
        return float("nan")


def load_or_compute_score(
    receptor_path: str,
    ligand_smiles_raw: str,
    cache_dir: str,
    compute_missing: bool,
    stats: Optional[Dict[str, int]] = None,
) -> float:
    val = load_cached_score(receptor_path, ligand_smiles_raw, cache_dir)
    if np.isfinite(val):
        if stats is not None:
            stats["cache_hits"] = stats.get("cache_hits", 0) + 1
        return val
    if not compute_missing:
        if stats is not None:
            stats["missing_without_compute"] = stats.get("missing_without_compute", 0) + 1
        return val
    val = compute_missing_score(receptor_path, ligand_smiles_raw, cache_dir)
    if np.isfinite(val):
        if stats is not None:
            stats["computed_missing"] = stats.get("computed_missing", 0) + 1
    elif stats is not None:
        stats["missing_after_compute"] = stats.get("missing_after_compute", 0) + 1
    return val


def complete_scores_for_pocket(
    receptor_path: str,
    smiles_list: List[str],
    existing_scores: Any,
    cache_dir: str,
    compute_missing: bool,
    stats: Dict[str, int],
) -> List[float]:
    if isinstance(existing_scores, list):
        existing = existing_scores
    elif existing_scores is None:
        existing = []
    else:
        try:
            existing = list(existing_scores)
        except TypeError:
            existing = []

    out: List[float] = []
    for idx, smi in enumerate(smiles_list):
        old = existing[idx] if idx < len(existing) else None
        if is_valid_score(old):
            stats["existing_scores"] = stats.get("existing_scores", 0) + 1
            out.append(float(old))
            continue
        out.append(load_or_compute_score(receptor_path, smi, cache_dir, compute_missing, stats))
    return out


def prepare_single_from_cache(
    supplemented_path: Path,
    pdb_dir: str,
    cache_dir: str,
    output_dir: Path,
    compute_missing: bool = False,
) -> None:
    logger.info("Loading supplemented results: %s", supplemented_path)
    data = load_results(supplemented_path)
    if not isinstance(data, dict):
        raise ValueError("supplemented results should be a dict")

    clean_data: Dict[str, Any] = {}
    updated = 0
    missing_pockets = 0
    missing_scores = 0
    score_stats: Dict[str, int] = {}

    for target_id, entry in data.items():
        gen = [s for s in (entry.get("generated_smiles") or []) if isinstance(s, str) and s.strip()]
        if not gen:
            continue

        active_pockets = entry.get("active_pockets", []) or []
        if not active_pockets:
            missing_pockets += 1
            continue

        p1_id = active_pockets[0]
        try:
            p1_path = get_pocket_file_path(p1_id, pdb_dir)
        except Exception:
            missing_pockets += 1
            continue

        existing_dock_map = entry.get("dock_scores_by_pocket")
        if not isinstance(existing_dock_map, dict):
            existing_dock_map = {}

        partner_pockets = entry.get("supplemented_partner_pockets") or entry.get("inactive_pockets") or []
        if not partner_pockets:
            partner_pockets = [pid for pid in existing_dock_map.keys() if pid != p1_id]
        partner_scores: Dict[str, List[float]] = {}

        for pid in partner_pockets:
            try:
                p_path = get_pocket_file_path(pid, pdb_dir)
            except Exception:
                continue
            scores = complete_scores_for_pocket(
                p_path,
                gen,
                existing_dock_map.get(pid),
                cache_dir,
                compute_missing,
                score_stats,
            )
            partner_scores[pid] = scores

        existing_p1_scores = entry.get("supplemented_dock_scores_p1") or existing_dock_map.get(p1_id)
        p1_scores = complete_scores_for_pocket(
            p1_path,
            gen,
            existing_p1_scores,
            cache_dir,
            compute_missing,
            score_stats,
        )

        p2_id = partner_pockets[0] if partner_pockets else None
        p2_scores = partner_scores.get(p2_id) if p2_id else None

        if p2_id is None or p2_scores is None:
            missing_scores += 1

        entry["generated_smiles"] = gen
        entry["supplemented_p1_pocket"] = p1_id
        entry["supplemented_dock_scores_p1"] = p1_scores

        if p2_id and p2_scores is not None:
            entry["supplemented_p2_pocket"] = p2_id
            entry["supplemented_dock_scores_p2"] = p2_scores

        dock_scores_by_pocket: Dict[str, List[float]] = {p1_id: p1_scores}
        for pid, scores in partner_scores.items():
            dock_scores_by_pocket[pid] = scores

        if partner_scores:
            entry["supplemented_dock_scores_partner_by_pocket"] = partner_scores
            entry["supplemented_partner_pockets"] = sorted(partner_scores.keys())

        entry["benchmark_prepared_from_cache"] = {
            "supplemented_results": str(supplemented_path),
            "docking_cache_dir": str(cache_dir),
            "compute_missing": compute_missing,
        }
        clean_entry = {
            "reference_smiles": entry.get("reference_smiles"),
            "generated_smiles": gen,
            "active_pockets": active_pockets,
            "inactive_pockets": entry.get("inactive_pockets") or [],
            "dock_scores_by_pocket": dock_scores_by_pocket,
        }
        clean_data[target_id] = clean_entry
        updated += 1

    logger.info(
        "Prepared %d entries (missing_pockets=%d, missing_scores=%d, score_stats=%s)",
        updated,
        missing_pockets,
        missing_scores,
        score_stats,
    )

    clean_json = output_dir / "clean_inference_results.json"
    clean_pkl = output_dir / "clean_inference_results.pkl"
    save_results(clean_data, clean_json, clean_pkl)

    bench_json = output_dir / "benchmark_ready_inference_results.json"
    bench_pkl = output_dir / "benchmark_ready_inference_results.pkl"
    save_results(data, bench_json, bench_pkl)
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(
            {
                "entries_prepared": updated,
                "missing_pockets": missing_pockets,
                "missing_scores": missing_scores,
                "score_stats": score_stats,
                "compute_missing": compute_missing,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _unique_pockets(pockets: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in pockets:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def prepare_multi_from_cache(
    input_path: Path,
    pdb_dir: str,
    cache_dir: str,
    output_dir: Path,
    compute_missing: bool = False,
) -> None:
    logger.info("Loading multi-target results: %s", input_path)
    data = load_results(input_path)
    if not isinstance(data, dict):
        raise ValueError("multi-target results should be a dict")

    clean_data: Dict[str, Any] = {}
    updated = 0
    missing_pockets = 0
    score_stats: Dict[str, int] = {}

    for target_id, entry in data.items():
        gen = [s for s in (entry.get("generated_smiles") or []) if isinstance(s, str) and s.strip()]
        if not gen:
            continue

        active_pockets = entry.get("active_pockets", []) or []
        inactive_pockets = entry.get("inactive_pockets", []) or []
        all_pockets = _unique_pockets(active_pockets + inactive_pockets)
        if not all_pockets:
            missing_pockets += 1
            continue

        dock_scores_by_pocket = entry.get("dock_scores_by_pocket")
        if not isinstance(dock_scores_by_pocket, dict):
            dock_scores_by_pocket = {}

        for pid in all_pockets:
            try:
                p_path = get_pocket_file_path(pid, pdb_dir)
            except Exception:
                continue
            dock_scores_by_pocket[pid] = [
                score
                for score in complete_scores_for_pocket(
                    p_path,
                    gen,
                    dock_scores_by_pocket.get(pid),
                    cache_dir,
                    compute_missing,
                    score_stats,
                )
            ]

        entry["generated_smiles"] = gen
        entry["dock_scores_by_pocket"] = dock_scores_by_pocket

        if active_pockets:
            p1_id = active_pockets[0]
            entry.setdefault("supplemented_p1_pocket", p1_id)
            if p1_id in dock_scores_by_pocket:
                entry.setdefault("supplemented_dock_scores_p1", dock_scores_by_pocket[p1_id])

        if len(active_pockets) > 1:
            p2_id = active_pockets[1]
            entry.setdefault("supplemented_p2_pocket", p2_id)
            if p2_id in dock_scores_by_pocket:
                entry.setdefault("supplemented_dock_scores_p2", dock_scores_by_pocket[p2_id])

        if active_pockets:
            partner_scores = {}
            for pid in active_pockets[1:]:
                if pid in dock_scores_by_pocket:
                    partner_scores[pid] = dock_scores_by_pocket[pid]
            if partner_scores:
                entry.setdefault("supplemented_dock_scores_partner_by_pocket", partner_scores)

        entry["benchmark_prepared_from_cache"] = {
            "input_results": str(input_path),
            "docking_cache_dir": str(cache_dir),
            "compute_missing": compute_missing,
        }

        clean_entry = {
            "reference_smiles": entry.get("reference_smiles"),
            "generated_smiles": gen,
            "active_pockets": active_pockets,
            "inactive_pockets": inactive_pockets,
            "dock_scores_by_pocket": dock_scores_by_pocket,
        }
        clean_data[target_id] = clean_entry
        updated += 1

    logger.info("Prepared %d entries (missing_pockets=%d, score_stats=%s)", updated, missing_pockets, score_stats)

    clean_json = output_dir / "clean_inference_results.json"
    clean_pkl = output_dir / "clean_inference_results.pkl"
    save_results(clean_data, clean_json, clean_pkl)

    bench_json = output_dir / "benchmark_ready_inference_results.json"
    bench_pkl = output_dir / "benchmark_ready_inference_results.pkl"
    save_results(data, bench_json, bench_pkl)
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(
            {
                "entries_prepared": updated,
                "missing_pockets": missing_pockets,
                "score_stats": score_stats,
                "compute_missing": compute_missing,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark-ready results for a config list.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--docking_cache_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--compute_missing",
        action="store_true",
        help="Run docking for scores that are absent from the cache.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r") as f:
        cfg_list = yaml.safe_load(f) if config_path.suffix in (".yaml", ".yml") else json.load(f)

    if not isinstance(cfg_list, list):
        raise ValueError("config should be a list")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for cfg in cfg_list:
        name = cfg.get("name")
        model_type = cfg.get("type")
        path = cfg.get("results_path") or cfg.get("path")
        if not name or not model_type or not path:
            logger.warning("Skipping invalid config entry: %s", cfg)
            continue

        input_path = Path(path)
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if model_type == "single_target":
            prepare_single_from_cache(
                input_path,
                args.pdb_dir,
                args.docking_cache_dir,
                out_dir,
                compute_missing=args.compute_missing,
            )
        elif model_type == "multi_target":
            prepare_multi_from_cache(
                input_path,
                args.pdb_dir,
                args.docking_cache_dir,
                out_dir,
                compute_missing=args.compute_missing,
            )
        else:
            logger.warning("Unknown model type '%s' for %s", model_type, name)


if __name__ == "__main__":
    main()
