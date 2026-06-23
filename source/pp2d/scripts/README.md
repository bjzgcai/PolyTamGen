# Scripts

Primary entrypoints:

- `train_pp2d.py`: train SFT models from YAML configs; use `--dry_run` for a one-batch startup check.
- `train_dpo.py`: train the DPO model from `configs/polytamgen_dpo.yaml`.
- `inference.py`: generate molecules from the DPO checkpoint or from an SFT checkpoint.
- `evaluate.py`: evaluate generated molecules and docking/property outputs.

Subdirectories:

- `data/`: build DPO preferences, compute dataset stats, export pocket sequences.
- `benchmark/`: prepare benchmark-ready files and run multi-target docking evaluation; benchmark input configs are under `benchmark/configs/`.
- `paper/`: verify paper alignment and rebuild paper artifacts.
- `paper/diagnostics/`: optional consistency tables for paper-result checks.
- `convert/`: format conversion helpers.

The primary training, inference, evaluation, and reproduction commands are listed here.
