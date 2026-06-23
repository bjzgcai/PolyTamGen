# Pocket PDB Root

Nested pocket-PDB files for the 288 unique pockets used by the T2/T3
paper splits and raw generation archives.

Use this directory as the docking PDB root:

```bash
export PP2D_PDB_DIR=data/pdb_benchmark/pocket_pdbs
```

Each pocket id maps to the path expected by
`scripts/benchmark/benchmark_multitarget_docking.py`, for example
`8a0z_1 -> 8a/8a0z/8a0z_pocket_1.pdb`.

`../pocket_pdbs_manifest.csv` records every copied pocket and its source
path provenance without absolute local paths.
