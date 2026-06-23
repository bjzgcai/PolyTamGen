#!/usr/bin/env python3
"""
Benchmark multi-target docking v4.0 (Configurable)
- Loads models dynamically from a YAML/JSON config file.
- Supports arbitrary numbers of multi-target and single-target baselines.
- Strict categorization: "One Similar", "Neither Similar", "Both Similar".

Usage:
    python scripts/benchmark/benchmark_multitarget_docking.py --model_config eval_config.yaml ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import random
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import yaml
except ImportError:
    print("PyYAML is required. Please install it with `pip install PyYAML`.")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit import RDLogger
except Exception as e:
    raise RuntimeError("RDKit is required.") from e

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

import re

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.eval.scorer import MultiMolsScorer, MolsScorer
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import MultiMolsScorer.")
    MultiMolsScorer = None
    MolsScorer = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Helper Functions ---
def _pair_key(a: str, b: str) -> Tuple[str, str]:
    x, y = sorted((a, b))
    return (x, y)

def get_pocket_file_path(pocket_id: str, pdb_root_dir: str) -> str:
    m = re.match(r"^([a-zA-Z0-9]{4})(_\w+)?$", pocket_id)
    if not m: raise ValueError(f"Invalid pocket_id: {pocket_id}")
    pdb_id = m.group(1)
    suffix = m.group(2) if m.group(2) else ""
    nested_dir = os.path.join(pdb_root_dir, pdb_id[:2].lower(), pdb_id.lower())
    path = os.path.join(nested_dir, f"{pdb_id.lower()}_pocket{suffix}.pdb")
    if not os.path.exists(path): raise FileNotFoundError(f"Not found: {path}")
    return path

def _is_valid_score(x: object) -> bool:
    if x is None: return False
    if not isinstance(x, (float, int, np.floating, np.integer)): return False
    v = float(x)
    return np.isfinite(v) and v < 0 and v > -200

def _cano_smiles(smiles_raw: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    if smiles_raw in cache: return cache[smiles_raw]
    if not smiles_raw or not isinstance(smiles_raw, str) or not smiles_raw.strip():
        cache[smiles_raw] = None
        return None
    try:
        m = Chem.MolFromSmiles(smiles_raw.strip())
        if m is None:
            cache[smiles_raw] = None
            return None
        cano = Chem.MolToSmiles(m)
        cache[smiles_raw] = cano
        return cano
    except Exception:
        cache[smiles_raw] = None
        return None

def normalize_ref_smiles(smiles_raw: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    return _cano_smiles(smiles_raw, cache)

def _cache_file_for(receptor_path: str, ligand_smiles_cano: str, cache_dir: str) -> str:
    unique_key = f"{receptor_path}|{ligand_smiles_cano}|1|32"
    unique_hash = hashlib.md5(unique_key.encode()).hexdigest()
    return os.path.join(cache_dir, f"{Path(receptor_path).stem}_{unique_hash}.pkl")

def _lookup_precomputed_score(
    score_map: Optional[Dict[str, Dict[str, float]]],
    receptor_path: str,
    ligand_smiles_raw: str,
    smiles_cano_cache: Dict[str, Optional[str]],
) -> Optional[float]:
    if not score_map:
        return None
    cano = _cano_smiles(ligand_smiles_raw, smiles_cano_cache)
    if cano is None:
        return None
    per_receptor = score_map.get(receptor_path)
    if not per_receptor:
        return None
    val = per_receptor.get(cano)
    return float(val) if _is_valid_score(val) else None

def load_cached_dock_score(receptor_path: str, ligand_smiles_raw: str, cache_dir: str, smiles_cano_cache: Dict[str, Optional[str]]) -> float:
    cano = _cano_smiles(ligand_smiles_raw, smiles_cano_cache)
    if cano is None: return float("nan")
    path = _cache_file_for(receptor_path, cano, cache_dir)
    if not os.path.exists(path): return float("nan")
    try:
        with open(path, "rb") as f: return float(pickle.load(f))
    except: return float("nan")

def maybe_compute_missing(receptor_path: str, ligand_smiles_raw: str, cache_dir: str) -> float:
    try:
        from src.eval.dock_utils import dock
        val = dock(receptor_path=receptor_path, ligand_identifier=ligand_smiles_raw, cache_dir=cache_dir)
        return float(val) if _is_valid_score(val) else float("nan")
    except: return float("nan")

def extract_precomputed_scores(raw_data: Dict, pdb_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Build a mapping from receptor_path -> {canonical_smiles: score}
    using any precomputed docking scores in the inference results.
    """
    score_map: Dict[str, Dict[str, float]] = {}
    cano_cache: Dict[str, Optional[str]] = {}

    def add_scores(pocket_id: Optional[str], scores: Optional[List[object]], smiles_list: List[str]) -> None:
        if not pocket_id or not scores or not smiles_list:
            return
        try:
            pocket_path = get_pocket_file_path(pocket_id, pdb_dir)
        except Exception:
            return
        per_receptor = score_map.setdefault(pocket_path, {})
        for smi, sc in zip(smiles_list, scores):
            cano = _cano_smiles(smi, cano_cache)
            if cano is None or not _is_valid_score(sc):
                continue
            per_receptor[cano] = float(sc)

    def add_scores_by_pocket(scores_by_pocket: object, smiles_list: List[str]) -> None:
        if not isinstance(scores_by_pocket, dict):
            return
        for pocket_id, scores in scores_by_pocket.items():
            add_scores(pocket_id, scores, smiles_list)

    for ent in raw_data.values():
        gens = [s for s in (ent.get("generated_smiles") or []) if isinstance(s, str) and s.strip()]
        if not gens:
            continue
        add_scores(ent.get("supplemented_p2_pocket"), ent.get("supplemented_dock_scores_p2"), gens)
        add_scores(ent.get("supplemented_p1_pocket"), ent.get("supplemented_dock_scores_p1"), gens)

        add_scores_by_pocket(ent.get("supplemented_dock_scores_p1_by_pocket"), gens)
        add_scores_by_pocket(ent.get("supplemented_dock_scores_p2_by_pocket"), gens)

        add_scores_by_pocket(ent.get("supplemented_dock_scores_partner_by_pocket"), gens)

        add_scores_by_pocket(ent.get("dock_scores_by_pocket"), gens)

    return score_map

def load_inference_results(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
        
    if path.suffix == ".json":
        with path.open("r") as f: return cast(Dict[str, Dict], json.load(f))
    
    if path.suffix == ".pkl":
        with path.open("rb") as f: raw = pickle.load(f)
        if isinstance(raw, dict) and raw and all(isinstance(v, dict) for v in raw.values()):
            return cast(Dict[str, Dict], raw)
        if isinstance(raw, dict) and "target_ids" in raw:
            # Handle standard separate-list format
            target_ids = raw.get("target_ids") or []
            gen_smiles = raw.get("generated_smiles") or []
            ref_smiles = raw.get("reference_smiles") or []
            active_ids = raw.get("active_pocket_ids") or []
            inactive_ids = raw.get("inactive_pocket_ids") or []
            grouped: Dict[str, Dict] = {}
            for tid, gs, rs, act, inact in zip(target_ids, gen_smiles, ref_smiles, active_ids, inactive_ids):
                if not isinstance(tid, str): continue
                ent = grouped.get(tid)
                if ent is None:
                    ent = {"generated_smiles": [], "reference_smiles": rs, "active_pockets": act, "inactive_pockets": inact}
                    grouped[tid] = ent
                if isinstance(gs, str): ent["generated_smiles"].append(gs)
            return grouped
    raise ValueError(f"Unsupported format: {path}")

def load_m8_similarity(m8_path: Path) -> Dict[str, float]:
    sim_map = {}
    if not m8_path.exists(): return sim_map
    with open(m8_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            q = parts[0][5:] if parts[0].startswith("test_") else parts[0]
            try: sim_map[q] = max(sim_map.get(q, 0.0), float(parts[2]))
            except: continue
    return sim_map

@dataclass(frozen=True)
class PairItem:
    ref_smiles_raw: str
    ref_smiles_key: str
    p1: str
    p2: str
    key_pair: Tuple[str, Tuple[str, str]]
    key_p1: Tuple[str, str]
    key_p2: Tuple[str, str]

def read_pairs_from_tsv(tsv_path: Path, limit: Optional[int] = None) -> List[PairItem]:
    df = pd.read_csv(tsv_path, sep="\t")
    items = []
    cache = {}
    for _, row in df.iterrows():
        ref, p1, p2 = str(row["L"]).strip(), str(row["P1"]).strip(), str(row["P2"]).strip()
        if not ref or ref.lower() == "nan" or not p1 or not p2: continue
        k = normalize_ref_smiles(ref, cache)
        if k:
            items.append(PairItem(ref, k, p1, p2, (k, _pair_key(p1, p2)), (k, p1), (k, p2)))
        if limit and len(items) >= limit: break
    return items

def index_multi_results_for_pairs(multi_results: Dict) -> Dict:
    out = {}
    cache = {}
    for ent in multi_results.values():
        ref = (ent.get("reference_smiles") or "").strip()
        act = [p for p in (ent.get("active_pockets") or []) if isinstance(p, str) and p.strip()]
        gens = [s for s in (ent.get("generated_smiles") or []) if isinstance(s, str) and s.strip()]
        if not ref or len(act) < 2 or not gens: continue
        rk = normalize_ref_smiles(ref, cache)
        if not rk: continue
        for i in range(len(act)):
            for j in range(i+1, len(act)):
                k = (rk, _pair_key(act[i], act[j]))
                out[k] = list(set(out.get(k, []) + gens))
    return out

def index_single_results_for_pockets(single_results: Dict) -> Dict:
    out = {}
    cache = {}
    for ent in single_results.values():
        ref = (ent.get("reference_smiles") or "").strip()
        act = [p for p in (ent.get("active_pockets") or []) if isinstance(p, str) and p.strip()]
        gens = [s for s in (ent.get("generated_smiles") or []) if isinstance(s, str) and s.strip()]
        if not ref or len(act) != 1 or not gens: continue
        rk = normalize_ref_smiles(ref, cache)
        if not rk: continue
        k = (rk, act[0])
        out[k] = list(set(out.get(k, []) + gens))
    return out

# --- Worker Function for Phase 2 (Global Scope) ---
def _worker_score_group(args_tuple):
    key, hyp_list, ref_smiles = args_tuple
    try:
        scorer = MolsScorer(hyp_smiles_list=hyp_list, ref_smiles=ref_smiles, docking=False)
        return key, scorer.report(mean=False)
    except: return key, {}

# --- Docking Logic ---
def evaluate_target_docking(smiles, p1, p2, cache, cano_cache, compute, topks, thres, lock, properties=None, score_map: Optional[Dict[str, Dict[str, float]]] = None):
    s1 = np.full(len(smiles), np.nan)
    s2 = np.full(len(smiles), np.nan)
    
    for i, smi in enumerate(smiles):
        v1 = _lookup_precomputed_score(score_map, p1, smi, cano_cache)
        v2 = _lookup_precomputed_score(score_map, p2, smi, cano_cache)

        if v1 is None or v2 is None:
            if lock:
                with lock:
                    if v1 is None:
                        v1 = load_cached_dock_score(p1, smi, cache, cano_cache)
                    if v2 is None:
                        v2 = load_cached_dock_score(p2, smi, cache, cano_cache)
            else:
                if v1 is None:
                    v1 = load_cached_dock_score(p1, smi, cache, cano_cache)
                if v2 is None:
                    v2 = load_cached_dock_score(p2, smi, cache, cano_cache)
        
        if compute:
            if not _is_valid_score(v1): v1 = maybe_compute_missing(p1, smi, cache)
            if not _is_valid_score(v2): v2 = maybe_compute_missing(p2, smi, cache)
            
        s1[i] = v1 if _is_valid_score(v1) else np.nan
        s2[i] = v2 if _is_valid_score(v2) else np.nan
        
    avg = (s1 + s2) / 2.0
    worst = np.maximum(s1, s2)
    
    def mean(v): return float(np.mean(v[np.isfinite(v)])) if np.any(np.isfinite(v)) else np.nan
    def best(v): return float(np.min(v[np.isfinite(v)])) if np.any(np.isfinite(v)) else np.nan
    # Conditional success: among molecules with valid docking scores.
    def succ_cond(v, t):
        return float(np.mean(v[np.isfinite(v)] <= t)) if np.any(np.isfinite(v)) else np.nan

    # Unconditional success: invalid docking counts as failure.
    # Denominator = total generated molecules for this group.
    def succ_all(v, t):
        if len(v) == 0:
            return np.nan
        vv = np.asarray(v, dtype=float)
        return float(np.mean((np.isfinite(vv) & (vv <= t)).astype(float)))
        
    # Strict Success Rate: QED>0.25 AND SA_norm>0.59 AND Docking<=t
    def succ_strict(v, t, qeds, sas):
        if len(v) == 0: return np.nan
        vv = np.asarray(v, dtype=float)
        qq = np.asarray(qeds, dtype=float)
        ss = np.asarray(sas, dtype=float)
        
        # All conditions must be met.
        # Check if values are valid (finite) numbers first
        c_dock = (np.isfinite(vv) & (vv <= t))
        c_qed = (np.isfinite(qq) & (qq > 0.25))
        c_sa = (np.isfinite(ss) & (ss > 0.59))
        
        return float(np.mean((c_dock & c_qed & c_sa).astype(float)))

    # Pair-level hit: at least one molecule succeeds.
    def hit(v, t):
        vv = np.asarray(v, dtype=float)
        ok = vv[np.isfinite(vv)]
        return float(np.any(ok <= t)) if ok.size else np.nan
    
    out = {
        "n_total": float(len(smiles)),
        "n_valid": float(np.sum(np.isfinite(avg))),
        "mean_p1": mean(s1),
        "mean_p2": mean(s2),
        "mean_avg": mean(avg),
        "mean_worst": mean(worst),
        "best_avg": best(avg),
        "best_worst": best(worst)
    }
    for k in topks:
        out[f"top{k}_mean_avg"] = mean(np.sort(avg[np.isfinite(avg)])[:k])
        out[f"top{k}_mean_worst"] = mean(np.sort(worst[np.isfinite(worst)])[:k])
    for t in thres:
        # keep existing names as conditional success for backward compatibility
        out[f"success_avg@{t}"] = succ_cond(avg, t)
        out[f"success_worst@{t}"] = succ_cond(worst, t)
        # additional paper metrics
        out[f"success_all_avg@{t}"] = succ_all(avg, t)
        out[f"success_all_worst@{t}"] = succ_all(worst, t)
        
        if properties:
            q = properties.get('qed', [np.nan] * len(smiles))
            s = properties.get('sa_norm', [np.nan] * len(smiles))
            out[f"success_rate_strict_avg@{t}"] = succ_strict(avg, t, q, s)
            out[f"success_rate_strict_worst@{t}"] = succ_strict(worst, t, q, s)
            
        out[f"hit_avg@{t}"] = hit(avg, t)
        out[f"hit_worst@{t}"] = hit(worst, t)

        # hit@topk (optional but often useful): among the best-k (by docking), does any succeed?
        for k in topks:
            a = np.sort(avg[np.isfinite(avg)])[:k]
            w = np.sort(worst[np.isfinite(worst)])[:k]
            out[f"hit_top{k}_avg@{t}"] = float(np.any(a <= t)) if a.size else np.nan
            out[f"hit_top{k}_worst@{t}"] = float(np.any(w <= t)) if w.size else np.nan
    return out

# --- Main ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv_path", default="data/pdb_benchmark/test_all.tsv")
    p.add_argument("--similarity_m8", default=None)
    p.add_argument("--pdb_dir", required=True)
    p.add_argument("--docking_cache_dir", default=".cache")
    
    # REPLACED: Single/Multi results arguments with Model Config
    p.add_argument("--model_config", type=str, required=True, help="Path to YAML/JSON config file defining models.")
    
    p.add_argument("--output_dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--compute_missing", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--single_union_sample_size", type=int, default=100)
    p.add_argument("--single_union_repeats", type=int, default=5)
    
    # Categorization args
    p.add_argument("--category_cutoff", type=float, default=0.4, help="Threshold to categorize Similar vs Dissimilar")
    p.add_argument("--cutoffs", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    p.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--thresholds", type=float, nargs="+", default=[-6.0, -7.0, -8.0])
    
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Sim & Data
    sim_map = load_m8_similarity(Path(args.similarity_m8)) if args.similarity_m8 else {}
    pairs = read_pairs_from_tsv(Path(args.tsv_path), limit=args.limit)
    logger.info(f"Loaded {len(pairs)} pairs from TSV.")
    
    # --- Load Model Configurations ---
    logger.info(f"Reading model config from {args.model_config}")
    with open(args.model_config, 'r') as f:
        if args.model_config.endswith(('.yaml', '.yml')):
            model_configs = yaml.safe_load(f)
        else:
            model_configs = json.load(f)

    if not isinstance(model_configs, list):
        raise ValueError("Config file must contain a list of model definitions.")

    loaded_models = {} # Format: {Name: {type: str, data: dict}}

    for cfg in model_configs:
        name = cfg.get("name")
        m_type = cfg.get("type")
        path = Path(cfg.get("path"))
        
        if not name or not m_type or not path:
            logger.warning(f"Skipping invalid config entry: {cfg}")
            continue
            
        logger.info(f"Loading model '{name}' ({m_type}) from {path}...")
        try:
            raw_data = load_inference_results(path)
            score_map = extract_precomputed_scores(raw_data, args.pdb_dir)
            if m_type == "multi_target":
                indexed_data = index_multi_results_for_pairs(raw_data)
            elif m_type == "single_target":
                indexed_data = index_single_results_for_pockets(raw_data)
            else:
                logger.error(f"Unknown model type '{m_type}' for {name}. Skipping.")
                continue
            
            loaded_models[name] = {"type": m_type, "data": indexed_data, "score_map": score_map}
            logger.info(f" -> Loaded {len(indexed_data)} entries for {name}.")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    pocket_path_cache = {}
    def get_path(pid):
        if pid not in pocket_path_cache: pocket_path_cache[pid] = get_pocket_file_path(pid, args.pdb_dir)
        return pocket_path_cache[pid]

    # --- Phase 1: Collect Tasks ---
    tasks_map = {}
    batch_keys, batch_hyp, batch_ref = [], [], []
    rng = random.Random(42)

    for i, item in enumerate(tqdm(pairs, desc="Registering Tasks")):
        def reg(method_name, s_list):
            if not s_list: return
            k = f"{i}|{method_name}"
            tasks_map[k] = s_list
            batch_keys.append(k)
            batch_hyp.append(list(s_list))
            batch_ref.append(item.ref_smiles_raw)

        # 1. Ground Truth
        reg("gt", [item.ref_smiles_raw])
        
        # 2. Dynamic Models
        for model_name, info in loaded_models.items():
            m_type = info["type"]
            data = info["data"]
            
            if m_type == "multi_target":
                ms = data.get(item.key_pair)
                reg(model_name, ms)
            
            elif m_type == "single_target":
                sp1 = data.get(item.key_p1)
                sp2 = data.get(item.key_p2)
                
                # Single pocket results
                reg(f"{model_name}_p1", sp1)
                reg(f"{model_name}_p2", sp2)
                
                # Union
                if sp1 and sp2:
                    union = list(sp1) + list(sp2)
                    reg(f"{model_name}_union", union)
                    
                    # Sampled Union
                    if args.single_union_sample_size > 0:
                        n = min(args.single_union_sample_size, len(union))
                        for r in range(args.single_union_repeats):
                            # Naming: ModelName_union_sampled_N_xR|r
                            # Note: reg will prepend "i|"
                            sample_name = f"{model_name}_union_sampled_{n}_x{args.single_union_repeats}|{r}"
                            sampled_mols = union if n==len(union) else rng.sample(union, k=n)
                            # Directly register to avoid complex name parsing later if possible, 
                            # but keeping consistent with reg helper for Property calculation
                            reg(sample_name, sampled_mols)

    # --- Phase 2: Properties (Multiprocessing) ---
    prop_results = {}
    if MolsScorer and batch_keys:
        logger.info(f"Computing properties for {len(batch_keys)} groups...")
        tasks = list(zip(batch_keys, batch_hyp, batch_ref))
        with multiprocessing.Pool(args.num_workers) as pool:
            for k, res in tqdm(pool.imap_unordered(_worker_score_group, tasks, chunksize=5), total=len(tasks)):
                prop_results[k] = res

    # --- Phase 3: Docking (Threading) ---
    smiles_cano_cache = {}
    lock = threading.Lock()
    
    def process_pair(arg):
        i, item = arg
        p1_path, p2_path = get_path(item.p1), get_path(item.p2)
        s1, s2 = sim_map.get(item.p1, 0.0), sim_map.get(item.p2, 0.0)
        
        # Categorization Logic
        threshold = args.category_cutoff
        if s1 < threshold and s2 < threshold:
            cat = "Both_Dissimilar"
        elif s1 >= threshold and s2 >= threshold:
            cat = "Both_Similar"
        else:
            cat = "One_Similar"
            
        rows = []
        
        # GT Delta Base
        gt_metrics = evaluate_target_docking([item.ref_smiles_raw], p1_path, p2_path, args.docking_cache_dir, smiles_cano_cache, args.compute_missing, [1], [], lock)
        gt_avg = gt_metrics.get("best_avg", np.nan)
        gt_worst = gt_metrics.get("best_worst", np.nan)
        
        def add(method, task_key, score_map=None):
            s_list = tasks_map.get(task_key)
            if not s_list: return
            
            pm = prop_results.get(task_key, {})
            
            # Since _worker_score_group now returns full lists (mean=False),
            # we must compute averages for the CSV report manually.
            pm_means = {}
            for k, v in pm.items():
                if isinstance(v, list):
                    vals = [x for x in v if isinstance(x, (int, float)) and np.isfinite(x)]
                    pm_means[k] = np.mean(vals) if vals else np.nan
                else:
                    pm_means[k] = v

            dm = evaluate_target_docking(s_list, p1_path, p2_path, args.docking_cache_dir, smiles_cano_cache, args.compute_missing, args.topk, args.thresholds, lock, properties=pm, score_map=score_map)
            
            row = {
                "ref_smiles": item.ref_smiles_raw, "p1": item.p1, "p2": item.p2, 
                "sim_p1": s1, "sim_p2": s2, "max_sim": max(s1, s2),
                "sim_category": cat, "method": method,
                "delta_best_avg_vs_gt": dm.get("best_avg", np.nan) - gt_avg,
                "delta_best_worst_vs_gt": dm.get("best_worst", np.nan) - gt_worst
            }
            row.update(dm)
            row.update(pm_means)
            rows.append(row)

        # 1. GT
        add("gt", f"{i}|gt")
        
        # 2. Dynamic Models Evaluation
        for model_name, info in loaded_models.items():
            m_type = info["type"]
            score_map = info.get("score_map")
            
            if m_type == "multi_target":
                add(model_name, f"{i}|{model_name}", score_map=score_map)
            
            elif m_type == "single_target":
                add(f"{model_name}_p1", f"{i}|{model_name}_p1", score_map=score_map)
                add(f"{model_name}_p2", f"{i}|{model_name}_p2", score_map=score_map)
                add(f"{model_name}_union", f"{i}|{model_name}_union", score_map=score_map)
                
                # Handle Sampled Union Aggregation
                if args.single_union_sample_size > 0:
                    reps = args.single_union_repeats
                    # Use the task map to check if union exists and get its length
                    union_list = tasks_map.get(f"{i}|{model_name}_union") or []
                    n = min(args.single_union_sample_size, len(union_list))
                    
                    if n > 0:
                        subs = []
                        for r in range(reps):
                            # Reconstruct key used in Reg
                            sub_method_suffix = f"{model_name}_union_sampled_{n}_x{reps}|{r}"
                            k = f"{i}|{sub_method_suffix}"
                            s = tasks_map.get(k)
                            if s:
                                pm = prop_results.get(k, {})
                                dm = evaluate_target_docking(
                                    s,
                                    p1_path,
                                    p2_path,
                                    args.docking_cache_dir,
                                    smiles_cano_cache,
                                    args.compute_missing,
                                    args.topk,
                                    args.thresholds,
                                    lock,
                                    properties=pm,
                                    score_map=score_map,
                                )
                                subs.append({**dm, **pm})
                        
                        if subs:
                            avg_m = {k: np.nanmean([d.get(k, np.nan) for d in subs]) for k in subs[0].keys()}
                            row = {
                                "ref_smiles": item.ref_smiles_raw, "p1": item.p1, "p2": item.p2, 
                                "sim_p1": s1, "sim_p2": s2, "max_sim": max(s1, s2),
                                "sim_category": cat, 
                                "method": f"{model_name}_union_sampled_{n}_x{reps}",
                                "delta_best_avg_vs_gt": avg_m.get("best_avg", np.nan) - gt_avg,
                                "delta_best_worst_vs_gt": avg_m.get("best_worst", np.nan) - gt_worst
                            }
                            row.update(avg_m)
                            rows.append(row)

        return rows

    all_rows = []
    with ThreadPoolExecutor(args.num_workers) as ex:
        futs = [ex.submit(process_pair, (i, item)) for i, item in enumerate(pairs)]
        for f in tqdm(as_completed(futs), total=len(pairs), desc="Docking"):
            try: all_rows.extend(f.result())
            except Exception as e: logger.error(e)

    # --- Output ---
    if not all_rows:
        logger.error("No results generated! Check input data or error logs.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "per_target.csv", index=False)
    
    # 1. Standard Cumulative Cutoffs
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for c in args.cutoffs:
        sub = df[df["max_sim"] <= c]
        if not sub.empty:
            sub.groupby("method")[num_cols].mean().to_csv(out_dir / f"summary_sim_cutoff_{c}.csv")

    # 2. Strict Category Summaries
    for cat in df["sim_category"].unique():
        sub = df[df["sim_category"] == cat]
        if not sub.empty:
            sub.groupby("method")[num_cols].mean().to_csv(out_dir / f"summary_cat_{cat}.csv")
            logger.info(f"Saved summary for category '{cat}'")

if __name__ == "__main__":
    main()
    
