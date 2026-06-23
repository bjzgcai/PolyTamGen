# Benchmark Scripts

Files:

- `benchmark_multitarget_docking.py`: main benchmark docking/scoring runner for T2/T3-style multi-target outputs.
- `prepare_benchmark_ready_from_config.py`: prepare benchmark-ready result files from a config containing model result paths.

Typical scoring command:

```bash
export PP2D_PDB_DIR=data/pdb_benchmark/pocket_pdbs
test -d "$PP2D_PDB_DIR"

python scripts/benchmark/benchmark_multitarget_docking.py \
  --model_config scripts/benchmark/configs/t2_inputs.yaml \
  --tsv_path data/pdb_benchmark/test_all.tsv \
  --pdb_dir "$PP2D_PDB_DIR" \
  --docking_cache_dir .cache/benchmark_docking \
  --output_dir outputs/benchmark/T2/recomputed
```

`PP2D_PDB_DIR` must point to the nested pocket-PDB root with paths like
`8a/8a0z/8a0z_pocket_1.pdb`. The repository includes the 288 paper split pocket
PDBs under `data/pdb_benchmark/pocket_pdbs/`, with provenance in
`data/pdb_benchmark/pocket_pdbs_manifest.csv`.

Outputs from a rerun should be written under `outputs/benchmark/*/recomputed`
until manually checked.
