#!/usr/bin/env python3
"""Build DPO preference pairs from scored generated molecules.

Input format is the benchmark-ready JSON used in this project:
group_id -> {
  generated_smiles: [...],
  active_pockets: [...],
  inactive_pockets: [...],
  dock_scores_by_pocket: {pocket_id: [score per generated SMILES]}
}

For each group the script computes
reward = -mean(active-target Vina scores) + qed_weight * QED(smiles),
then selects the highest-reward molecule as the DPO winner and the
lowest-reward molecule as the loser.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_qed():
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
        from rdkit import RDLogger
    except Exception as exc:  # pragma: no cover - depends on optional RDKit install
        raise RuntimeError("RDKit is required to compute QED for DPO preferences.") from exc
    RDLogger.DisableLog("rdApp.*")
    return Chem, QED


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean_active_dock(record: Dict[str, Any], idx: int) -> Optional[float]:
    scores_by_pocket = record.get("dock_scores_by_pocket") or {}
    active_pockets = record.get("active_pockets") or list(scores_by_pocket)
    scores = []
    for pocket_id in active_pockets:
        pocket_scores = scores_by_pocket.get(pocket_id)
        if not isinstance(pocket_scores, list) or idx >= len(pocket_scores):
            continue
        score = _finite_float(pocket_scores[idx])
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _iter_pairs(input_path: Path, qed_weight: float) -> Iterable[Dict[str, Any]]:
    Chem, QED = _load_qed()
    payload = json.loads(input_path.read_text())
    for group_id, record in payload.items():
        generated = record.get("generated_smiles") or []
        candidates = []
        for idx, smiles in enumerate(generated):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            avg_dock = _mean_active_dock(record, idx)
            if avg_dock is None:
                continue
            qed = float(QED.qed(mol))
            reward = -float(avg_dock) + float(qed_weight) * qed
            candidates.append(
                {
                    "smiles": str(smiles),
                    "reward": reward,
                    "avg_active_dock": float(avg_dock),
                    "qed": qed,
                }
            )
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda item: item["reward"])
        loser = candidates[0]
        winner = candidates[-1]
        yield {
            "group_id": group_id,
            "Active_Protein_IDs": record.get("active_pockets", []),
            "Inactive_Protein_IDs": record.get("inactive_pockets", []),
            "chosen_smiles": winner["smiles"],
            "rejected_smiles": loser["smiles"],
            "chosen_reward": winner["reward"],
            "rejected_reward": loser["reward"],
            "chosen_avg_active_dock": winner["avg_active_dock"],
            "rejected_avg_active_dock": loser["avg_active_dock"],
            "chosen_qed": winner["qed"],
            "rejected_qed": loser["qed"],
            "qed_weight": float(qed_weight),
            "reward_formula": "reward = -mean_active_vina + qed_weight * QED",
        }


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PP2D DPO preference pairs from scored candidates.")
    parser.add_argument("--input", required=True, help="Benchmark-ready generated/scored JSON.")
    parser.add_argument("--output", required=True, help="Output JSONL path, or train JSONL when --valid-output is set.")
    parser.add_argument("--valid-output", default=None, help="Optional validation JSONL path.")
    parser.add_argument("--valid-fraction", type=float, default=0.03, help="Validation fraction when --valid-output is set.")
    parser.add_argument("--qed-weight", type=float, default=5.0, help="Lambda in reward = -dock + lambda * QED.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = list(_iter_pairs(Path(args.input), args.qed_weight))
    if not rows:
        raise RuntimeError(f"No DPO preference pairs could be built from {args.input}")

    random.Random(args.seed).shuffle(rows)
    if args.valid_output:
        valid_count = max(1, int(round(len(rows) * args.valid_fraction)))
        valid_rows = rows[:valid_count]
        train_rows = rows[valid_count:]
        _write_jsonl(train_rows, Path(args.output))
        _write_jsonl(valid_rows, Path(args.valid_output))
        print(f"Wrote {len(train_rows)} train pairs to {args.output}")
        print(f"Wrote {len(valid_rows)} valid pairs to {args.valid_output}")
    else:
        _write_jsonl(rows, Path(args.output))
        print(f"Wrote {len(rows)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
