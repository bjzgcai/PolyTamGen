#!/usr/bin/env python3
"""
Optimized evaluation script for PP2D model (Listwise).
Evaluates molecular properties, docking scores (Listwise), and text generation quality.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# Optional RDKit for SMILES canonicalization
try:
    from rdkit import Chem  # type: ignore
    HAS_RDKIT = True
except Exception:
    Chem = None  # type: ignore
    HAS_RDKIT = False

# Add project root to path and import project modules
try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    from src.eval.scorer import MultiMolsScorer, TextScorer
    from src.features import get_pocket_file_path
except ImportError as e:
    print(f"Error: Could not import project modules. Details: {e}")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def sample_generated_smiles(
    smiles_list: List[str],
    max_samples: Optional[int],
    strategy: str = "first",
    seed_key: str = "",
) -> List[str]:
    """Optionally downsample generated SMILES per target before docking-heavy evaluation."""
    if max_samples is None or max_samples <= 0 or len(smiles_list) <= max_samples:
        return smiles_list
    if strategy == "hash":
        ordered = sorted(
            enumerate(smiles_list),
            key=lambda item: hashlib.md5(f"{seed_key}|{item[0]}|{item[1]}".encode()).hexdigest(),
        )
        keep = sorted(idx for idx, _ in ordered[:max_samples])
        return [smiles_list[idx] for idx in keep]
    return smiles_list[:max_samples]


def load_results(results_path: Path) -> Dict:
    """Load inference results from a .json or .pkl file."""
    logger.info(f"Loading inference results from: {results_path}")
    try:
        if results_path.suffix == '.json':
            with results_path.open('r') as f:
                return json.load(f)
        elif results_path.suffix == '.pkl':
            with results_path.open('rb') as f:
                return pkl.load(f)
        else:
            raise ValueError(f"Unsupported file format: {results_path.suffix}")
    except (IOError, json.JSONDecodeError, pkl.UnpicklingError) as e:
        logger.error(f"Failed to load or parse the results file at {results_path}: {e}")
        sys.exit(1)


def get_valid_smiles(smiles_list: List[str]) -> List[str]:
    """Filter a list of SMILES strings, returning only non-empty, valid ones."""
    return [s for s in smiles_list if s and isinstance(s, str) and s.strip()]


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit when available; otherwise return stripped string."""
    if not smiles or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None
    if HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(s)  # type: ignore
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)  # type: ignore
        except Exception:
            return None
    return s


def evaluate_molecular_properties(
    results: Dict, output_dir: Path, args: argparse.Namespace, pdb_dir: Optional[str]
) -> Dict:
    """Evaluate molecular properties, including Listwise docking."""
    logger.info("Starting molecular property evaluation...")
    
    # Organize data for scorer
    # results is Dict[target_id, entry]
    # entry: {generated_smiles, reference_smiles, active_pockets, inactive_pockets}
    
    hyp_smiles_list = []
    ref_smiles_list = []
    target_ids = []
    group_smiles_list = []
    
    active_pocket_paths_list = [] if args.enable_docking else None
    inactive_pocket_paths_list = [] if args.enable_docking else None
    
    for target_id, data in results.items():
        valid_generated = get_valid_smiles(data.get('generated_smiles', []))
        valid_generated = sample_generated_smiles(
            valid_generated,
            getattr(args, "max_generated_per_target", None),
            getattr(args, "generated_sample_strategy", "first"),
            seed_key=str(target_id),
        )
        if not valid_generated:
            continue
        
        # Reference SMILES
        ref = data.get('reference_smiles', "")
        
        # Docking Setup
        if args.enable_docking and pdb_dir:
            active_ids = data.get('active_pockets', [])
            inactive_ids = data.get('inactive_pockets', [])
            
            # Convert IDs to Paths
            curr_active_paths = []
            for pid in active_ids:
                try:
                    curr_active_paths.append(get_pocket_file_path(pid, pdb_dir))
                except (ValueError, FileNotFoundError) as e:
                    logger.warning(f"Failed to get pocket path for {pid}: {e}")
            
            curr_inactive_paths = []
            for pid in inactive_ids:
                try:
                    curr_inactive_paths.append(get_pocket_file_path(pid, pdb_dir))
                except (ValueError, FileNotFoundError) as e:
                    logger.warning(f"Failed to get pocket path for {pid}: {e}")
            
            active_pocket_paths_list.append(curr_active_paths)
            inactive_pocket_paths_list.append(curr_inactive_paths)
        
        hyp_smiles_list.append(valid_generated)
        ref_smiles_list.append(ref)
        target_ids.append(target_id)
        group_smiles_list.append(valid_generated)

    if not hyp_smiles_list:
        logger.warning("No valid molecules found for property evaluation. Skipping.")
        return {}

    try:
        scorer = MultiMolsScorer(
            hyp_smiles_list=hyp_smiles_list,
            ref_smiles_list=ref_smiles_list,
            docking=args.enable_docking,
            # Pass the new Listwise arguments
            active_pocket_paths_list=active_pocket_paths_list,
            inactive_pocket_paths_list=inactive_pocket_paths_list,
            
            docking_cache_dir=args.docking_cache_dir,
            output_complex_base_dir=str(output_dir / 'docking_complexes') if args.enable_docking else None,
            num_workers=args.num_workers
        )
        summary_results = scorer.report(mean=True)

        # Save results
        with (output_dir / 'molecular_evaluation_summary.json').open('w') as f:
            json.dump(summary_results, f, indent=2)
        logger.info(f"Molecular evaluation summary saved to {output_dir / 'molecular_evaluation_summary.json'}")

        if getattr(args, "save_docking_scores", False):
            per_group = scorer.report(mean=False)
            docking_scores_path = getattr(args, "docking_scores_path", None)
            out_path = Path(docking_scores_path) if docking_scores_path else (output_dir / "molecular_evaluation_per_group.pkl")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                pkl.dump(
                    {
                        "target_ids": target_ids,
                        "ref_smiles": ref_smiles_list,
                        "generated_smiles": group_smiles_list,
                        "group_reports": per_group,
                    },
                    f,
                )
            logger.info(f"Per-group docking scores saved to {out_path}")
        
        return summary_results
    except Exception as e:
        logger.error(f"Molecular evaluation failed: {e}", exc_info=True)
        return {}


def calculate_basic_statistics(results: Dict) -> Dict:
    """Calculate basic statistics like validity and uniqueness."""
    logger.info("Calculating basic statistics...")
    
    all_generated_smiles = []
    for data in results.values():
        all_generated_smiles.extend(data.get('generated_smiles', []))
        
    valid_smiles = get_valid_smiles(all_generated_smiles)
    
    total_generated = len(all_generated_smiles)
    total_valid = len(valid_smiles)
    total_unique = len(set(valid_smiles))
    
    stats = {
        'total_targets': len(results),
        'total_generated_strings': total_generated,
        'total_valid_molecules': total_valid,
        'total_unique_molecules': total_unique,
        'validity': total_valid / total_generated if total_generated > 0 else 0.0,
        'uniqueness': total_unique / total_valid if total_valid > 0 else 0.0,
    }
    return stats


def compute_hit_rate(results: Dict) -> Dict[str, Any]:
    """Compute hit rate: fraction of targets where any generated SMILES equals the reference SMILES."""
    logger.info("Computing hit rate with canonicalized SMILES equality...")
    
    total_with_ref = 0
    num_hits = 0
    per_target_hits: Dict[str, bool] = {}
    
    for target_id, data in results.items():
        ref_raw = data.get('reference_smiles')
        ref = canonicalize_smiles(ref_raw) if ref_raw else None
        
        if not ref:
            continue
            
        total_with_ref += 1
        generated_raw = get_valid_smiles(data.get('generated_smiles', []))
        generated_cano = [canonicalize_smiles(s) for s in generated_raw]
        
        hit = any((g is not None and g == ref) for g in generated_cano)
        per_target_hits[target_id] = hit
        if hit:
            num_hits += 1
            
    hit_rate = (num_hits / total_with_ref) if total_with_ref > 0 else 0.0
    return {
        'hit_rate': hit_rate,
        'num_targets_with_ref': total_with_ref,
        'num_hits': num_hits,
        'per_target_hits': per_target_hits,
    }


def generate_final_report(all_results: Dict, output_dir: Path):
    """Combine all evaluation metrics into a single report and print a summary."""
    logger.info("Generating final evaluation report...")
    
    report_path = output_dir / 'evaluation_report.json'
    with report_path.open('w') as f:
        json.dump(all_results, f, indent=4)

    # --- Print Summary ---
    summary = all_results.get('summary', {})
    stats = all_results.get('basic_statistics', {})
    hit_summary = all_results.get('hit_rate_summary', {})
    
    report_str = f"""
==================================================
EVALUATION SUMMARY
==================================================
Targets Evaluated:  {stats.get('total_targets', 'N/A')}
Validity:           {summary.get('validity', 0.0):.3f}
Uniqueness:         {summary.get('uniqueness', 0.0):.3f}
Hit Rate:           {summary.get('hit_rate', 0.0):.3f} ({hit_summary.get('num_hits', 0)}/{hit_summary.get('num_targets_with_ref', 0)})

--- Molecular Properties ---
QED:                {summary.get('qed', 'N/A')}
SA Score:           {summary.get('sas', 'N/A')}
Diversity:          {summary.get('diversity', 'N/A')}
Similarity to Ref:  {summary.get('similarity', 'N/A')}

--- Listwise Docking Scores (Average per Molecule) ---
Active Affinity:    {summary.get('dock_scores_active_avg', 'N/A')}
Inactive Affinity:  {summary.get('dock_scores_inactive_avg', 'N/A')}
==================================================
Full report saved to: {report_path}
"""
    print("\n".join(line.strip() for line in report_str.strip().split('\n')))


def run_evaluation(args: argparse.Namespace):
    """Main evaluation pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Results (NatureLM JSON format expected)
    raw_results = load_results(Path(args.results))
    
    pdb_dir = None
    if args.enable_docking:
        if not args.pdb_dir:
            logger.error("--enable_docking requires --pdb_dir to be set.")
            sys.exit(1)
        pdb_dir = args.pdb_dir
        if not os.path.isdir(pdb_dir):
            logger.error(f"Invalid --pdb_dir: {pdb_dir}. Must be a directory containing pocket PDB files.")
            sys.exit(1)
        logger.info(f"Using PDB directory for pocket files: {pdb_dir}")

    # Run all evaluation components
    basic_stats = calculate_basic_statistics(raw_results)
    mol_results = evaluate_molecular_properties(raw_results, output_dir, args, pdb_dir)
    hit_results = compute_hit_rate(raw_results)
    
    # Compile the final report
    final_report = {
        'basic_statistics': basic_stats,
        'molecular_properties_summary': mol_results,
        'hit_rate_summary': hit_results,
        'summary': {
            'validity': basic_stats.get('validity'),
            'uniqueness': basic_stats.get('uniqueness'),
            'hit_rate': hit_results.get('hit_rate'),
            **mol_results, # Unpack docking scores etc.
        }
    }

    generate_final_report(final_report, output_dir)
    logger.info("Evaluation pipeline completed successfully! ✨")


def main():
    """Parse arguments and launch the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate molecules generated by the PP2D model (Listwise).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    core_group = parser.add_argument_group('Core Arguments')
    core_group.add_argument("--results", type=str, required=True, help="Path to inference results file (.json or .pkl).")
    core_group.add_argument("--output_dir", type=str, required=True, help="Directory to save all evaluation results.")
    core_group.add_argument(
        "--max_generated_per_target",
        type=int,
        default=None,
        help="Optional cap on generated molecules used per target for evaluation.",
    )
    core_group.add_argument(
        "--generated_sample_strategy",
        choices=["first", "hash"],
        default="first",
        help="How to downsample generated molecules when max_generated_per_target is set.",
    )

    # Docking
    dock_group = parser.add_argument_group('Docking Arguments')
    dock_group.add_argument("--enable_docking", action="store_true", help="Enable molecular docking evaluation.")
    dock_group.add_argument("--pdb_dir", type=str, default=None, help="Root directory containing the pocket PDB files.")
    dock_group.add_argument("--docking_cache_dir", type=str, default=".cache", help="Directory to cache docking results.")
    dock_group.add_argument("--save_docking_scores", action="store_true", help="Save per-group docking scores to a pickle file.")
    dock_group.add_argument("--docking_scores_path", type=str, default=None, help="Path for per-group docking scores pickle.")
    
    # System
    sys_group = parser.add_argument_group('System Arguments')
    sys_group.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for scoring.")
    
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
