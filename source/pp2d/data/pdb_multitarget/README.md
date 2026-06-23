# PDB Multi-Target Paper Test Data

This is the paper test/inference data selected from PDB pockets.

Files:

- `protein_features.lmdb`: PDB-pocket keyed protein feature LMDB.
- `T2/ligand_profiles.pkl`: T2 target-pair inference profiles.
- `T2/split.pkl`: T2 inference split; use `--split_key test`.
- `T3/ligand_profiles.pkl`: T3 target-triplet inference profiles.
- `T3/split.pkl`: T3 inference split; use `--split_key test`.

It is intentionally separate from `../sft_train_valid/`: SFT uses UniProt-style
features and no paper test split, while the reported T2/T3 tests use PDB-pocket
IDs so that docking scores can be computed.
