#!/usr/bin/env python3
"""Reproduce paper benchmark tables from provided raw generation and scored files.

This is the T2/T3 benchmark reproduction entry point. It checks that
raw_generation provenance aligns with the scored molecule rows, rebuilds
consistency tables from scored per_molecule files, and runs the paper alignment
verifier.

Run from source/pp2d:

    python scripts/paper/reproduce_benchmark_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

TargetKey = Tuple[str, Tuple[str, ...]]
RawIndex = Dict[TargetKey, List[str]]

SMILES_CACHE: Dict[str, Optional[str]] = {}


def cano_smiles(smiles: Any) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    if smiles in SMILES_CACHE:
        return SMILES_CACHE[smiles]
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        out = Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        out = None
    SMILES_CACHE[smiles] = out
    return out


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path, base: Path) -> str:
    """Return a stable report path for display."""
    base_abs = base.absolute()
    path_abs = path if path.is_absolute() else base_abs / path
    try:
        return path_abs.relative_to(base_abs).as_posix()
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        pass
    if path.is_absolute():
        return f"<outside-cwd>/{path.name}"
    return path.as_posix()


def unique_append(out: Dict[TargetKey, List[str]], key: TargetKey, values: Iterable[str]) -> None:
    bucket = out.setdefault(key, [])
    seen = set(bucket)
    for value in values:
        if isinstance(value, str) and value.strip() and value not in seen:
            bucket.append(value)
            seen.add(value)


def build_single_pocket_index(raw_path: Path) -> RawIndex:
    data = load_json(raw_path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in {raw_path}")
    out: RawIndex = {}
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        ref = cano_smiles(entry.get("reference_smiles"))
        active = [p for p in (entry.get("active_pockets") or []) if isinstance(p, str) and p.strip()]
        generated = [s for s in (entry.get("generated_smiles") or []) if isinstance(s, str) and s.strip()]
        if not ref or len(active) != 1 or not generated:
            continue
        unique_append(out, (ref, (active[0],)), generated)
    return out


def build_multi_target_index(raw_path: Path, n_pockets: int) -> RawIndex:
    data = load_json(raw_path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in {raw_path}")
    out: RawIndex = {}
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        ref = cano_smiles(entry.get("reference_smiles"))
        active = [p for p in (entry.get("active_pockets") or []) if isinstance(p, str) and p.strip()]
        generated = [s for s in (entry.get("generated_smiles") or []) if isinstance(s, str) and s.strip()]
        if not ref or len(active) < n_pockets or not generated:
            continue
        for pocket_group in combinations(active, n_pockets):
            unique_append(out, (ref, tuple(sorted(pocket_group))), generated)
    return out


def scored_target_columns(split: str) -> List[str]:
    return ["p1", "p2"] + (["p3"] if split == "T3" else [])


def load_scored_groups(scored_paths: Mapping[str, Path], manifest_rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, Any]]:
    wanted_by_split: Dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        wanted_by_split[row["split"]].add(row["primary_scored_method"])

    out: Dict[str, Dict[str, Any]] = {}
    for split, path in scored_paths.items():
        df = pd.read_json(path)
        pocket_cols = scored_target_columns(split)
        groups: Dict[Tuple[str, TargetKey], Counter[str]] = defaultdict(Counter)
        method_counts: Dict[str, Tuple[int, int]] = {}
        for method, sub in df[df["method"].isin(wanted_by_split[split])].groupby("method"):
            method_counts[method] = (len(sub), sub[["ref_smiles"] + pocket_cols].drop_duplicates().shape[0])
            for row in sub[["ref_smiles"] + pocket_cols + ["smiles"]].itertuples(index=False, name=None):
                ref = cano_smiles(row[0])
                pockets = tuple(str(x).strip() for x in row[1 : 1 + len(pocket_cols)] if str(x).strip())
                smiles = row[-1]
                if ref and len(pockets) == len(pocket_cols) and isinstance(smiles, str) and smiles.strip():
                    groups[(method, (ref, pockets))][smiles] += 1
        out[split] = {"groups": groups, "method_counts": method_counts}
    return out


def raw_counter_for_target(raw_index: RawIndex, granularity: str, target: TargetKey) -> Counter[str]:
    ref, pockets = target
    if granularity.startswith("single_pocket"):
        counter: Counter[str] = Counter()
        for pocket in pockets:
            counter.update(raw_index.get((ref, (pocket,)), []))
        return counter
    return Counter(raw_index.get((ref, tuple(sorted(pockets))), []))


def check_manifest_and_raw(
    results_root: Path,
    rows: Sequence[Mapping[str, str]],
    scored_data: Mapping[str, Dict[str, Any]],
    report_base: Path,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    raw_cache: Dict[Tuple[str, str], RawIndex] = {}

    for row in rows:
        split = row["split"]
        label = f"{split}/{row['paper_label']}"
        scored_path = results_root.parent / row["scored_path"]
        raw_path = results_root.parent / row["raw_generation_path"] if row.get("raw_generation_path") else None
        method = row["primary_scored_method"]

        method_counts = scored_data[split]["method_counts"]
        actual_rows, actual_targets = method_counts.get(method, (None, None))
        manifest_rows = int(row["scored_molecule_rows"]) if row.get("scored_molecule_rows") else None
        manifest_targets = int(row["target_count"]) if row.get("target_count") else None
        count_ok = (actual_rows == manifest_rows) and (actual_targets == manifest_targets) and scored_path.exists()
        checks.append(
            {
                "name": f"manifest_counts.{label}",
                "status": "ok" if count_ok else "fail",
                "details": {
                    "method": method,
                    "manifest_rows": manifest_rows,
                    "actual_rows": actual_rows,
                    "manifest_targets": manifest_targets,
                    "actual_targets": actual_targets,
                    "scored_path_exists": scored_path.exists(),
                },
            }
        )

        if not raw_path:
            checks.append(
                {
                    "name": f"raw_alignment.{label}",
                    "status": "skip",
                    "details": "No provided raw_generation archive for this scored-only row.",
                }
            )
            continue
        if not raw_path.exists():
            checks.append(
                {
                    "name": f"raw_alignment.{label}",
                    "status": "fail",
                    "details": f"Missing {display_path(raw_path, report_base)}",
                }
            )
            continue

        raw_key = (str(raw_path), row["raw_granularity"])
        if raw_key not in raw_cache:
            if row["raw_granularity"].startswith("single_pocket"):
                raw_cache[raw_key] = build_single_pocket_index(raw_path)
            elif row["raw_granularity"] == "multi_target_pair":
                raw_cache[raw_key] = build_multi_target_index(raw_path, 2)
            elif row["raw_granularity"] == "multi_target_triple":
                raw_cache[raw_key] = build_multi_target_index(raw_path, 3)
            else:
                checks.append(
                    {
                        "name": f"raw_alignment.{label}",
                        "status": "skip",
                        "details": f"Unsupported raw granularity {row['raw_granularity']}",
                    }
                )
                continue

        scored_groups = scored_data[split]["groups"]
        scored_total = 0
        raw_total = 0
        overlap_total = 0
        target_count = 0
        mismatches: List[Dict[str, Any]] = []
        for (group_method, target), scored_counter in scored_groups.items():
            if group_method != method:
                continue
            target_count += 1
            raw_counter = raw_counter_for_target(raw_cache[raw_key], row["raw_granularity"], target)
            scored_rows = sum(scored_counter.values())
            raw_rows = sum(raw_counter.values())
            overlap = sum((scored_counter & raw_counter).values())
            scored_total += scored_rows
            raw_total += raw_rows
            overlap_total += overlap
            if scored_counter != raw_counter and len(mismatches) < 5:
                mismatches.append(
                    {
                        "target": [target[0], *target[1]],
                        "scored_rows": scored_rows,
                        "raw_rows": raw_rows,
                        "overlap": overlap,
                    }
                )

        ok = scored_total == raw_total == overlap_total and not mismatches
        checks.append(
            {
                "name": f"raw_alignment.{label}",
                "status": "ok" if ok else "fail",
                "details": {
                    "raw_path": display_path(raw_path, report_base),
                    "method": method,
                    "target_groups": target_count,
                    "scored_rows": scored_total,
                    "raw_rows": raw_total,
                    "exact_smiles_overlap": overlap_total,
                    "mismatch_sample": mismatches,
                },
            }
        )
    return checks


def run_command(cmd: Sequence[str], cwd: Path) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout, proc.stderr


def command_path(path: Path, cwd: Path) -> str:
    try:
        return path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return str(path)


def sanitize_text(text: str, cwd: Path, extra_paths: Sequence[Path] = ()) -> str:
    out = text
    replacements = [(cwd, ".")]
    for path in extra_paths:
        replacements.append((path, display_path(path, cwd)))
    for path, label in replacements:
        try:
            out = out.replace(str(path.resolve()), label)
        except OSError:
            out = out.replace(str(path), label)
    return out


def tail(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def write_reports(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reproduction_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# Paper Benchmark Reproduction Report", "", f"Status: `{report['status']}`", ""]
    lines.append("## Checks")
    lines.append("")
    lines.append("| Status | Check | Details |")
    lines.append("|---|---|---|")
    for check in report["checks"]:
        details = check.get("details", "")
        if not isinstance(details, str):
            details = json.dumps(details, sort_keys=True)
        if len(details) > 240:
            details = details[:237] + "..."
        details = details.replace("|", r"\|")
        lines.append(f"| {check['status']} | `{check['name']}` | {details} |")
    lines.append("")
    (output_dir / "reproduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--manifest", type=Path, default=Path("results/benchmark/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_reproduction"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-raw-alignment", action="store_true")
    parser.add_argument("--skip-verifier", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    results_root = args.results_root if args.results_root.is_absolute() else cwd / args.results_root
    manifest_path = args.manifest if args.manifest.is_absolute() else cwd / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else cwd / args.output_dir
    consistency_dir = output_dir / "consistency_tables"

    rows = read_manifest(manifest_path)
    scored_paths = {
        "T2": results_root / "scored/T2/per_molecule.json",
        "T3": results_root / "scored/T3/per_molecule.json",
    }

    checks: List[Dict[str, Any]] = []
    for split, path in scored_paths.items():
        checks.append(
            {
                "name": f"input.scored.{split}",
                "status": "ok" if path.exists() else "fail",
                "details": display_path(path, cwd),
            }
        )
    checks.append(
        {
            "name": "input.manifest",
            "status": "ok" if manifest_path.exists() else "fail",
            "details": display_path(manifest_path, cwd),
        }
    )

    scored_data = load_scored_groups(scored_paths, rows)
    if not args.skip_raw_alignment:
        checks.extend(check_manifest_and_raw(results_root, rows, scored_data, cwd))

    cmd = [
        sys.executable,
        "scripts/paper/diagnostics/generate_consistency_tables_from_per_molecule.py",
        "--t2_path",
        command_path(scored_paths["T2"], cwd),
        "--t3_path",
        command_path(scored_paths["T3"], cwd),
        "--output_dir",
        command_path(consistency_dir, cwd),
        "--num_workers",
        str(args.num_workers),
    ]
    code, stdout, stderr = run_command(cmd, cwd)
    checks.append(
        {
            "name": "consistency_tables.generate_from_scored",
            "status": "ok" if code == 0 else "fail",
            "details": {
                "returncode": code,
                "stdout_tail": tail(sanitize_text(stdout, cwd, [output_dir, consistency_dir])),
                "stderr_tail": tail(sanitize_text(stderr, cwd, [output_dir, consistency_dir])),
            },
        }
    )
    if code == 0:
        for name, actual_rel in [
            ("consistency_tables.output.T2", "table1_t2_main.tex"),
            ("consistency_tables.output.T3", "table3_t3_stress.tex"),
        ]:
            path = consistency_dir / actual_rel
            checks.append(
                {
                    "name": name,
                    "status": "ok" if path.exists() else "fail",
                    "details": display_path(path, cwd),
                }
            )

    if not args.skip_verifier:
        code, stdout, stderr = run_command([
            sys.executable, "scripts/paper/verify_paper_alignment.py",
            "--results-root", str(results_root),
            "--out-json", str(output_dir / "paper_alignment_report.json"),
            "--out-md", str(output_dir / "paper_alignment_report.md"),
        ], cwd)
        checks.append(
            {
                "name": "verify_paper_alignment",
                "status": "ok" if code == 0 else "fail",
                "details": {
                    "returncode": code,
                    "stdout": sanitize_text(stdout.strip(), cwd, [output_dir, consistency_dir]),
                    "stderr_tail": tail(sanitize_text(stderr, cwd, [output_dir, consistency_dir])),
                },
            }
        )

    failures = [check for check in checks if check["status"] == "fail"]
    report = {
        "status": "fail" if failures else "ok",
        "num_checks": len(checks),
        "num_failures": len(failures),
        "results_root": display_path(results_root, cwd),
        "manifest": display_path(manifest_path, cwd),
        "output_dir": display_path(output_dir, cwd),
        "checks": checks,
    }
    write_reports(report, output_dir)
    print(json.dumps({k: report[k] for k in ["status", "num_checks", "num_failures", "output_dir"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
