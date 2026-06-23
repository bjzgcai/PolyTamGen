# Data

This directory contains curated data entrypoints used by training, inference, and benchmark scoring.

Files:

- `Components-smiles-stereo-oe.smi`: component SMILES dictionary used by chemistry utilities.
- `prompt_templates.json`: prompt templates for non-listwise generation.
- `prompt_templates_listwise.json`: prompt templates for listwise/multi-target generation.

Directories. Some large entries are supplied by the external data archive
described in `../../../ASSETS.md`:

- `sft_train_valid/`: SFT train/validation profiles, UniProt protein features, and the curated split with the original scaffold-test rows folded into validation.
- `dpo_preferences/`: compact DPO winner/loser preference JSONL files used as training input.
- `pdb_multitarget/`: PDB-selected paper T2/T3 inference/test profiles and PDB-pocket feature LMDB.
- `pdb_benchmark/`: T2/T3 benchmark TSVs, exported pocket sequences, and nested docking pocket PDBs.

Important split convention:

- `sft_train_valid/` is for SFT training and validation only. It does not define
  the reported paper test set.
- The reported paper T2/T3 test data are PDB-selected so that docking can be
  computed. Those profiles live in `pdb_multitarget/`, and the matching docking
  TSV/PDB inputs live in `pdb_benchmark/`.

Use `scripts/data/build_dpo_preferences.py` to rebuild DPO preferences from scored result files under `results/raw_generation/`.
