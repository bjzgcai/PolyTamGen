# SFT Train/Validation Data

This is the SFT data entrypoint used by `configs/polytamgen_sft.yaml`.

Files:

- `ligand_profiles.pkl`: ligand/profile records for SFT.
- `Merged_Deduplicated.csv`: source molecule table used by dataset statistics.
- `protein_features.lmdb`: UniProt-keyed protein feature LMDB for SFT inputs.
- `protein_sequences.pkl`: protein sequence map used by feature-preparation helpers.
- `splits/train_valid_split.pkl`: train/validation split file.

Split convention:

```text
train: 1,632,581 rows
valid: 16,489 rows
test: 0 rows
```

The old scaffold split had a small `test` subset. For the paper workflow, that
subset is not the reported test set; the reported T2/T3 tests are PDB-selected
multi-target pockets under `../pdb_multitarget/` and scored with docking inputs
under `../pdb_benchmark/`. To avoid two competing test definitions, the old
SFT `test` rows are folded into validation here.
