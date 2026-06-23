# DPO Preferences

Clean DPO preference files:

- `preferences_5qed_train.jsonl`: training preference pairs.
- `preferences_5qed_valid.jsonl`: validation preference pairs.

Each JSONL row is a winner/loser pair for one target context. Preferences are rebuilt with:

```bash
python scripts/data/build_dpo_preferences.py \
  --input results/raw_generation/T2/PolyTamGen-SFT.json \
  --output data/dpo_preferences/preferences_5qed_train.jsonl \
  --valid-output data/dpo_preferences/preferences_5qed_valid.jsonl \
  --valid-fraction 0.2 \
  --qed-weight 5.0
```

Reward convention:

```text
reward = -mean_active_vina + 5.0 * QED
```

These files use PDB pocket IDs such as `3nlh_1`; train them with
`data/pdb_multitarget/protein_features.lmdb`, not the SFT
UniProt LMDB under `data/sft_train_valid/`.
