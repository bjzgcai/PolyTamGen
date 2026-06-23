# Configs

Canonical training configs:

- `polytamgen_sft.yaml`: SFT training config and SFT inference defaults.
- `polytamgen_dpo.yaml`: DPO training config; initializes from an SFT checkpoint and uses the curated DPO preference JSONL files with the PDB-pocket LMDB.

Non-training configs live with their script families:

- `scripts/benchmark/configs/t2_inputs.yaml`: T2 benchmark raw-generation inputs.
- `scripts/benchmark/configs/t3_inputs.yaml`: T3 benchmark raw-generation inputs.

Both canonical configs use `models/base-llm/NLM-1b` as the LLM path. The base language model is provided as `base-llm_NLM-1b.tar` and is required for checkpoint inference. The configs set `distributed.world_size: null`, so the launcher auto-detects the available GPUs. SFT uses `data/sft_train_valid/protein_features.lmdb`; DPO uses `data/pdb_multitarget/protein_features.lmdb` because its preference rows are keyed by PDB pocket IDs.
