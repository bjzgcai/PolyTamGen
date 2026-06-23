# Data Scripts

Files:

- `build_dpo_preferences.py`: build DPO winner/loser JSONL pairs from scored result JSON.
- `build_pocket_sequences.py`: extract pocket-chain sequences for benchmark split pockets.
- `compute_dataset_statistics.py`: compute compact dataset statistics for paper summaries.
- `materialize_docking_pocket_root.py`: copy the exact nested pocket-PDB root required by benchmark docking reruns from a prepared receptor cache.

Canonical DPO preferences live in `data/dpo_preferences/`.

Docking PDB inputs live in `data/pdb_benchmark/pocket_pdbs/`. Rebuild them
only when moving to a new storage layout:

```bash
python scripts/data/materialize_docking_pocket_root.py \
  --source-cache /path/to/source/pp2d/.cache
```
