#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def build_pocket_set(tsv_path: Path) -> set[str]:
    df = pd.read_csv(tsv_path, sep="\t")
    pockets = set()
    for col in ("P1", "P2"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {tsv_path}")
        for v in df[col].astype(str).tolist():
            v = v.strip()
            if v and v.lower() != "nan":
                pockets.add(v)
    return pockets


def infer_active_pockets_from_key(key: str, pocket_set: set[str]) -> tuple[str, str] | None:
    # Prefer longest-match for the left pocket, to handle ids like "4p4w_2".
    for left in sorted(pocket_set, key=len, reverse=True):
        prefix = left + "_"
        if not key.startswith(prefix):
            continue
        right = key[len(prefix) :]
        if right in pocket_set:
            return left, right
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert inference results into the multi-target combined format: "
            "{id: {generated_smiles, reference_smiles, active_pockets, inactive_pockets}}"
        )
    )
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--pairs_tsv",
        default=Path("data/pdb_benchmark/test_all.tsv"),
        type=Path,
        help="TSV providing the universe of pocket ids (columns P1, P2)",
    )
    ap.add_argument(
        "--allow_unmatched",
        action="store_true",
        help="If set, keep unmatched entries with empty active_pockets instead of failing.",
    )
    args = ap.parse_args()

    pocket_set = build_pocket_set(args.pairs_tsv)

    with args.input.open("r") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Expected top-level dict in input json")

    converted: dict[str, dict] = {}
    unmatched: list[str] = []

    for key, ent in raw.items():
        if not isinstance(key, str) or not isinstance(ent, dict):
            continue

        gs = ent.get("generated_smiles") or []
        if isinstance(gs, str):
            gs = [gs]
        if not isinstance(gs, list):
            gs = []
        gs = [s for s in gs if isinstance(s, str) and s.strip()]

        rs = (ent.get("reference_smiles") or "").strip()

        pair = infer_active_pockets_from_key(key, pocket_set)
        if pair is None:
            unmatched.append(key)
            act = []
        else:
            act = [pair[0], pair[1]]

        converted[key] = {
            "generated_smiles": gs,
            "reference_smiles": rs,
            "active_pockets": act,
            "inactive_pockets": [],
        }

    if unmatched and not args.allow_unmatched:
        sample = "\n".join(unmatched[:20])
        raise SystemExit(
            f"Failed to infer active_pockets for {len(unmatched)} keys. Sample:\n{sample}\n"
            "Re-run with --allow_unmatched to keep them (they will be skipped by benchmark)."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(converted, f, indent=2)

    print(
        f"Wrote {args.output} ({len(converted)} entries). "
        f"Unmatched: {len(unmatched)}."
    )


if __name__ == "__main__":
    main()
