#!/usr/bin/env python3
"""Build the nested pocket-PDB root required by docking scripts.

The benchmark code resolves a pocket id such as ``6avs_1`` to
``<pdb_root>/6a/6avs/6avs_pocket_1.pdb``.  Previous runs kept these pocket PDBs
inside hashed receptor-cache directories, so this script copies only the
pockets referenced by the retained test splits and raw generation files into a
clean, deterministic tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


POCKET_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}(?:_[A-Za-z0-9]+)?$")
POCKET_LIST_FIELDS = {
    "active_pockets",
    "inactive_pockets",
    "active_pocket_ids",
    "inactive_pocket_ids",
    "pocket_ids",
    "pockets",
}
POCKET_SCALAR_FIELDS = {
    "pocket_id",
    "pocket",
    "supplemented_p1_pocket",
    "supplemented_p2_pocket",
}
POCKET_SCORE_MAP_FIELDS = {
    "dock_scores_by_pocket",
    "supplemented_dock_scores_p1_by_pocket",
    "supplemented_dock_scores_p2_by_pocket",
    "supplemented_dock_scores_partner_by_pocket",
}


def expected_filename(pocket_id: str, suffix: str = ".pdb") -> str:
    match = re.match(r"^([0-9][A-Za-z0-9]{3})(_\w+)?$", pocket_id)
    if not match:
        raise ValueError(f"Invalid pocket id: {pocket_id}")
    pdb_id = match.group(1).lower()
    pocket_suffix = match.group(2) or ""
    return f"{pdb_id}_pocket{pocket_suffix}{suffix}"


def expected_relpath(pocket_id: str, suffix: str = ".pdb") -> Path:
    pdb_id = pocket_id[:4].lower()
    return Path(pdb_id[:2]) / pdb_id / expected_filename(pocket_id, suffix=suffix)


def is_pocket_id(value: Any) -> bool:
    return isinstance(value, str) and bool(POCKET_ID_RE.match(value.strip()))


def collect_from_tsv(path: Path) -> Set[str]:
    out: Set[str] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            for key, value in row.items():
                if key and key.upper().startswith("P") and is_pocket_id(value):
                    out.add(value.strip().lower())
    return out


def walk_json_pocket_ids(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in POCKET_LIST_FIELDS and isinstance(value, list):
                for item in value:
                    if is_pocket_id(item):
                        yield item.strip().lower()
            elif key in POCKET_SCALAR_FIELDS and is_pocket_id(value):
                yield value.strip().lower()
            elif key in POCKET_SCORE_MAP_FIELDS and isinstance(value, dict):
                for item in value:
                    if is_pocket_id(item):
                        yield item.strip().lower()
            yield from walk_json_pocket_ids(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_json_pocket_ids(value)
    elif is_pocket_id(obj):
        yield obj.strip().lower()


def collect_from_json(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(walk_json_pocket_ids(data))


def collect_needed_pockets(split_paths: Sequence[Path], raw_paths: Sequence[Path]) -> Set[str]:
    out: Set[str] = set()
    for path in split_paths:
        out.update(collect_from_tsv(path))
    for path in raw_paths:
        out.update(collect_from_json(path))
    return out


def source_label(root: Path) -> str:
    if root.name == ".cache" and root.parent.name:
        return f"{root.parent.name}/.cache"
    return root.name or str(root)


def index_source_files(source_roots: Sequence[Path], suffix: str) -> Dict[str, List[Tuple[Path, Path]]]:
    indexed: Dict[str, List[Tuple[Path, Path]]] = {}
    for root in source_roots:
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            name = path.name.lower()
            if "_pocket" not in name:
                continue
            indexed.setdefault(name, []).append((root, path))
    for values in indexed.values():
        values.sort(key=lambda item: str(item[1]))
    return indexed


def first_source(
    pocket_id: str,
    *,
    indexed: Dict[str, List[Tuple[Path, Path]]],
    suffix: str,
) -> Optional[Tuple[Path, Path]]:
    candidates = indexed.get(expected_filename(pocket_id, suffix=suffix).lower())
    return candidates[0] if candidates else None


def copy_one(
    pocket_id: str,
    *,
    indexed: Dict[str, List[Tuple[Path, Path]]],
    output_root: Path,
    suffix: str,
) -> Dict[str, str]:
    relpath = expected_relpath(pocket_id, suffix=suffix)
    dst = output_root / relpath
    src = first_source(pocket_id, indexed=indexed, suffix=suffix)
    row = {
        "pocket_id": pocket_id,
        "kind": suffix.lstrip("."),
        "expected_relpath": str(relpath),
        "status": "missing",
        "source_root": "",
        "source_relpath": "",
    }
    if src is None:
        return row

    src_root, src_path = src
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    row["status"] = "copied"
    row["source_root"] = source_label(src_root)
    try:
        row["source_relpath"] = str(src_path.relative_to(src_root))
    except ValueError:
        row["source_relpath"] = src_path.name
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        type=Path,
        default=[],
        help="TSV split file with P1/P2/P3 pocket columns. Can be repeated.",
    )
    parser.add_argument(
        "--raw",
        action="append",
        type=Path,
        default=[],
        help="Raw generation JSON whose pocket ids should be included. Can be repeated.",
    )
    parser.add_argument(
        "--source-cache",
        action="append",
        type=Path,
        default=[],
        help="Source receptor-cache root containing *_pocket*.pdb files. Can be repeated.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/pdb_benchmark/pocket_pdbs"),
        help="Nested output root consumed by benchmark_multitarget_docking.py.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/pdb_benchmark/pocket_pdbs_manifest.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_source = os.environ.get("PP2D_SOURCE_DOCKING_CACHE")
    source_roots = list(args.source_cache)
    if env_source:
        source_roots.extend(Path(p) for p in env_source.split(os.pathsep) if p)
    source_roots = [p for p in source_roots if str(p)]
    if not source_roots:
        raise SystemExit("Provide --source-cache or PP2D_SOURCE_DOCKING_CACHE.")

    split_paths = args.split or [
        Path("data/pdb_benchmark/test_all.tsv"),
        Path("data/pdb_benchmark/test_T3.tsv"),
    ]
    raw_paths = args.raw or sorted(Path("results/raw_generation").glob("*/*.json"))
    pockets = sorted(collect_needed_pockets(split_paths, raw_paths))
    if not pockets:
        raise SystemExit("No pocket ids found in requested split/raw inputs.")

    pdb_index = index_source_files(source_roots, ".pdb")
    rows = [
        copy_one(pocket_id, indexed=pdb_index, output_root=args.output_root, suffix=".pdb")
        for pocket_id in pockets
    ]

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pocket_id", "kind", "expected_relpath", "status", "source_root", "source_relpath"],
        )
        writer.writeheader()
        writer.writerows(rows)

    copied = sum(row["status"] == "copied" and row["kind"] == "pdb" for row in rows)
    missing = sum(row["status"] != "copied" and row["kind"] == "pdb" for row in rows)
    print(f"Needed pockets: {len(pockets)}")
    print(f"Copied PDBs: {copied}")
    print(f"Missing PDBs: {missing}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
