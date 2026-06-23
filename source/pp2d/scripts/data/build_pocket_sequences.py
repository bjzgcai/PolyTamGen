#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build pocket chain sequences for test split pockets.

Workflow:
1. Read test TSV files under data/pdb_benchmark and collect all pocket IDs (columns P1, P2).
2. For each pocket_id, locate the pocket PDB in the PP2D nested PDB directory layout.
3. Parse the pocket PDB, find the chain with the largest number of residues (dominant chain_id).
4. Using the same pdb_dir layout, locate the corresponding full structure file (prefer .cif, fallback to .pdb).
5. Extract the amino-acid sequence for the dominant chain from the full structure.
6. Save:
   - sequences as a pickle dict: {pocket_id: sequence}
   - chain IDs as a JSON dict: {pocket_id: chain_id}

This script assumes Biopython is available (Bio.PDB).
"""

import os
import sys
import argparse
import json
import pickle
import re
from typing import Dict, Set, Tuple

import multiprocessing
import pandas as pd
import requests
from tqdm import tqdm

try:
    from Bio.PDB import MMCIFParser, PDBParser, PPBuilder
except ImportError as e:
    raise ImportError(
        "Biopython is required for this script. Please install it with 'pip install biopython'."
    ) from e

# Ensure we can import from the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def _default_pdb_path(pocket_id: str, pdb_dir: str) -> str:
    match = re.match(r"^([a-zA-Z0-9]{4})(_\w+)?$", pocket_id)
    if not match:
        raise ValueError(f"Invalid pocket_id format: {pocket_id}")
    pdb_id = match.group(1).lower()
    suffix = match.group(2) or ""
    nested_dir = os.path.join(pdb_dir, pdb_id[:2], pdb_id)
    pocket_path = os.path.join(nested_dir, f"{pdb_id}_pocket{suffix}.pdb")
    if not os.path.exists(pocket_path):
        raise FileNotFoundError(pocket_path)
    return pocket_path


def _read_unique_pockets(tsv_files: list[str]) -> Set[str]:
    pockets: Set[str] = set()
    for path in tsv_files:
        df = pd.read_csv(path, sep="\t")
        for column in ("P1", "P2", "P3"):
            if column in df.columns:
                pockets.update(str(x).strip() for x in df[column].dropna() if str(x).strip())
    return pockets


def _full_structure_path_from_pocket_id(
    pocket_id: str,
    pdb_dir: str,
    download_cache_dir: str | None = None,
) -> str:
    """
    Infer the full structure path for a given pocket_id.

    The directory layout and pdb_id extraction follow _default_pdb_path:
        nested_dir = os.path.join(pdb_dir, pdb_id[:2].lower(), pdb_id.lower())
        pocket_pdb is   f"{pdb_id.lower()}_pocket{suffix}.pdb"
    Here we try:
        1) f"{pdb_id}.cif"
        2) f"{pdb_id}.pdb"
    """
    match = re.match(r"^([a-zA-Z0-9]{4})(_\w+)?$", pocket_id)
    if not match:
        raise ValueError(f"Invalid pocket_id format: {pocket_id}")
    pdb_id = match.group(1).lower()
    nested_dir_pdb = os.path.join(pdb_dir, pdb_id[:2].lower(), pdb_id.lower())

    cif_path_local = os.path.join(nested_dir_pdb, f"{pdb_id}.cif")
    pdb_path_local = os.path.join(nested_dir_pdb, f"{pdb_id}.pdb")

    # Prefer local mmCIF / PDB if already present under pdb_dir
    if os.path.exists(cif_path_local):
        return cif_path_local
    if os.path.exists(pdb_path_local):
        return pdb_path_local

    # If not present locally, optionally fetch mmCIF into a writable cache directory
    if download_cache_dir is None:
        raise FileNotFoundError(
            f"Neither CIF nor PDB full structure found for {pocket_id} "
            f"(tried: {cif_path_local}, {pdb_path_local})"
        )

    cache_nested_dir = os.path.join(
        download_cache_dir, pdb_id[:2].lower(), pdb_id.lower()
    )
    cif_cache_path = os.path.join(cache_nested_dir, f"{pdb_id}.cif")

    # Already cached
    if os.path.exists(cif_cache_path):
        return cif_cache_path

    os.makedirs(cache_nested_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        print(f"[INFO] Fetching mmCIF from RCSB for {pocket_id}: {url}")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            with open(cif_cache_path, "w") as f:
                f.write(resp.text)
            print(f"[INFO] Saved downloaded mmCIF to {cif_cache_path}")
            return cif_cache_path
        else:
            raise RuntimeError(
                f"Failed to download mmCIF for {pocket_id} from {url} "
                f"(status={resp.status_code})"
            )
    except Exception as e:
        raise FileNotFoundError(
            f"Neither CIF nor PDB full structure found locally for {pocket_id} "
            f"and failed to fetch from RCSB ({e}). "
            f"(tried: {cif_path_local}, {pdb_path_local}, {cif_cache_path})"
        )


def _read_test_pockets(tsv_dir: str) -> Set[str]:
    """
    Read TSV files under tsv_dir and return unique pocket IDs.

    Expected split files use P1/P2/P3 columns for pocket IDs.
    """
    test_files = [
        os.path.join(tsv_dir, "test_all.tsv"),
        os.path.join(tsv_dir, "test_novel_pair_both_pockets_known.tsv"),
        os.path.join(tsv_dir, "test_novel_pair_both_pockets_new.tsv"),
        os.path.join(tsv_dir, "test_novel_pair_one_pocket_known.tsv"),
    ]
    existing = [p for p in test_files if os.path.exists(p)]
    if existing:
        return _read_unique_pockets(existing)

    # Fallback: read all TSVs in the directory
    tsv_files = [
        os.path.join(tsv_dir, name)
        for name in os.listdir(tsv_dir)
        if name.lower().endswith(".tsv")
    ]
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found under {tsv_dir}")
    return _read_unique_pockets(tsv_files)


def _find_dominant_chain_id_from_pocket(pocket_pdb_path: str) -> str:
    """
    Parse the pocket PDB and return the chain_id with the largest number of residues.
    """
    parser = PDBParser(QUIET=True)
    structure_id = os.path.basename(pocket_pdb_path)
    structure = parser.get_structure(structure_id, pocket_pdb_path)

    chain_counts: Dict[str, int] = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            num_residues = sum(1 for _ in chain.get_residues())
            chain_counts[chain_id] = chain_counts.get(chain_id, 0) + num_residues

    if not chain_counts:
        raise ValueError(f"No residues found in pocket PDB: {pocket_pdb_path}")

    dominant_chain_id = max(chain_counts.items(), key=lambda kv: kv[1])[0]
    return dominant_chain_id


def _extract_chain_sequence(structure_path: str, pdb_id: str, chain_id: str) -> str:
    """
    Extract the amino-acid sequence of a chain from a full structure file.

    Supports both mmCIF (.cif) and PDB (.pdb) formats.
    """
    if structure_path.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure(pdb_id, structure_path)
    model = next(structure.get_models())

    if chain_id not in model:
        raise KeyError(
            f"Chain {chain_id} not found in structure {structure_path} (pdb_id={pdb_id})."
        )
    chain = model[chain_id]

    ppb = PPBuilder()
    sequences = []
    for pp in ppb.build_peptides(chain):
        sequences.append(str(pp.get_sequence()))

    if not sequences:
        raise ValueError(
            f"No peptide sequence could be built for chain {chain_id} in {structure_path}."
        )

    # Concatenate all polypeptide fragments into a single sequence
    return "".join(sequences)


def _process_one_pocket(
    pocket_id: str,
    pdb_dir: str,
    download_cache_dir: str | None,
) -> Tuple[str, str | None, str | None, str | None]:
    """
    Worker function to process a single pocket_id.
    Returns (pocket_id, sequence or None, chain_id or None, error_message or None).
    """
    try:
        pocket_pdb = _default_pdb_path(pocket_id, pdb_dir)
    except Exception as e:
        return pocket_id, None, None, f"failed to locate pocket PDB ({e})"

    try:
        chain_id = _find_dominant_chain_id_from_pocket(pocket_pdb)
    except Exception as e:
        return pocket_id, None, None, f"failed to find dominant chain ({e})"

    try:
        full_path = _full_structure_path_from_pocket_id(
            pocket_id, pdb_dir, download_cache_dir=download_cache_dir
        )
    except Exception as e:
        return pocket_id, None, None, f"failed to locate full structure ({e})"

    try:
        pdb_id = re.match(r"^([a-zA-Z0-9]{4})", pocket_id).group(1).lower()  # type: ignore[attr-defined]
        seq = _extract_chain_sequence(full_path, pdb_id, chain_id)
    except Exception as e:
        return pocket_id, None, None, (
            f"failed to extract sequence from {full_path} for chain {chain_id} ({e})"
        )

    return pocket_id, seq, chain_id, None


def _process_one_pocket_star(args: Tuple[str, str, str | None]) -> Tuple[str, str | None, str | None, str | None]:
    """Helper to unpack arguments for multiprocessing."""
    return _process_one_pocket(*args)


def _build_sequences_for_pockets(
    pockets: Set[str],
    pdb_dir: str,
    download_cache_dir: str | None,
    num_workers: int,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    For each pocket_id:
        1) locate pocket PDB
        2) determine dominant chain_id
        3) locate full structure
        4) extract chain sequence

    Returns:
        sequences: {pocket_id: sequence}
        chain_ids: {pocket_id: chain_id}
    """
    sequences: Dict[str, str] = {}
    chain_ids: Dict[str, str] = {}

    worker_args = [
        (pocket_id, pdb_dir, download_cache_dir) for pocket_id in sorted(pockets)
    ]

    if num_workers <= 1:
        # Fallback to sequential processing
        for pocket_id, pdb_dir_arg, cache_dir_arg in tqdm(
            worker_args, desc="Processing pockets", unit="pocket"
        ):
            pid, seq, chain_id, err = _process_one_pocket(
                pocket_id, pdb_dir_arg, cache_dir_arg
            )
            if err is not None:
                print(f"[WARN] Skip {pid}: {err}")
                continue
            sequences[pid] = seq  # type: ignore[assignment]
            chain_ids[pid] = chain_id  # type: ignore[assignment]
    else:
        print(f"Using {num_workers} worker processes for pocket processing.")
        with multiprocessing.Pool(processes=num_workers) as pool:
            iterator = pool.imap_unordered(_process_one_pocket_star, worker_args)
            for pid, seq, chain_id, err in tqdm(
                iterator,
                total=len(worker_args),
                desc="Processing pockets",
                unit="pocket",
            ):
                if err is not None:
                    print(f"[WARN] Skip {pid}: {err}")
                    continue
                sequences[pid] = seq  # type: ignore[assignment]
                chain_ids[pid] = chain_id  # type: ignore[assignment]

    return sequences, chain_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build pocket sequences and chain IDs for test split pockets.\n"
            "Reads TSVs under data/pdb_benchmark, infers pocket PDBs and full structures, "
            "and writes:\n"
            "  - sequences: pickle dict {pocket_id: sequence}\n"
            "  - chain_ids: JSON dict {pocket_id: chain_id}"
        )
    )
    parser.add_argument(
        "--tsv_dir",
        type=str,
        default="data/pdb_benchmark",
        help="Directory containing test TSV files (default: data/pdb_benchmark).",
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        default=os.environ.get("PP2D_PDB_DIR"),
        help="Root directory of pocket/full structures. Defaults to PP2D_PDB_DIR.",
    )
    parser.add_argument(
        "--download_cache_dir",
        type=str,
        default="data/pdb_cif_cache",
        help=(
            "Writable directory to cache downloaded mmCIF files "
            "(default: data/pdb_cif_cache)."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of worker processes for pocket processing (default: half of CPUs).",
    )
    parser.add_argument(
        "--output_sequences",
        type=str,
        default="data/pdb_benchmark/pocket_sequences.pkl",
        help="Output pickle file for sequences dict (default: data/pdb_benchmark/pocket_sequences.pkl).",
    )
    parser.add_argument(
        "--output_chain_ids",
        type=str,
        default="data/pdb_benchmark/pocket_chain_ids.json",
        help="Output JSON file for chain_id dict (default: data/pdb_benchmark/pocket_chain_ids.json).",
    )

    args = parser.parse_args()
    if not args.pdb_dir:
        parser.error("--pdb_dir is required unless PP2D_PDB_DIR is set")

    os.makedirs(os.path.dirname(args.output_sequences), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_chain_ids), exist_ok=True)
    if args.download_cache_dir:
        os.makedirs(args.download_cache_dir, exist_ok=True)

    print(f"Reading test TSVs from: {args.tsv_dir}")
    pockets = _read_test_pockets(args.tsv_dir)
    print(f"Collected {len(pockets)} unique pocket IDs from test split TSVs.")

    print(f"Using pdb_dir: {args.pdb_dir}")
    if args.download_cache_dir:
        print(f"Using download cache dir: {args.download_cache_dir}")
    sequences, chain_ids = _build_sequences_for_pockets(
        pockets,
        args.pdb_dir,
        download_cache_dir=args.download_cache_dir,
        num_workers=args.num_workers,
    )

    print(f"Writing sequences for {len(sequences)} pockets to: {args.output_sequences}")
    with open(args.output_sequences, "wb") as f:
        pickle.dump(sequences, f)

    print(f"Writing chain IDs for {len(chain_ids)} pockets to: {args.output_chain_ids}")
    with open(args.output_chain_ids, "w") as f:
        json.dump(chain_ids, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
