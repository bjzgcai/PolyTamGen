# PolyTamGen

PolyTamGen is a multi-target molecule generation framework for designing
compounds against two or more protein targets. This repository contains the
source code, lightweight metadata, and external-asset instructions needed to
reproduce the paper tables and supplement analyses.

Developed by researchers affiliated with Zhongguancun Academy, University of
Science and Technology of China, and Global Health Drug Discovery Institute.

The repository covers three main workflows:

- Training PolyTamGen with supervised fine-tuning and preference optimization.
- Running T2 and T3 multi-target molecule generation benchmarks.
- Recomputing the reported benchmark values and supplement figures from the
  downloaded result files.

## Repository Contents

```text
source/pp2d/        Installable PolyTamGen package, configs, scripts, tests
results/            Download note for the external full-results archive
models/             Model archive notes; runtime model paths live under source/pp2d/models
data/               Data archive notes; runtime data paths live under source/pp2d/data
ASSETS.md           External archive layout and manifest notes
REPRODUCIBILITY.md  Step-by-step paper-result reproduction guide
```

## Setup

Create a Python environment with the dependencies in
`source/pp2d/requirements.txt`, then install the package in editable mode:




```bash
cd source/pp2d
conda create -y -n pp2d python=3.12
conda activate pp2d
PP2D_SKIP_CREATE=1 bash setup_env.sh
```

Install the PyTorch, PyTorch Geometric, and CUDA packages that match the target
machine before running GPU workloads. Set `PP2D_INSTALL_TORCH_STACK=1` only when
the torch stack listed in `requirements.txt` is appropriate for that machine.

## Model And Data Assets

Large model, data, and result payloads are available as Zenodo files
and are not tracked by git. The public Zenodo record is
https://zenodo.org/records/20594128 (DOI: `10.5281/zenodo.20594128`).
Extract or place assets under `source/pp2d/`, because
command examples are run from that directory and use relative paths such as
`models/`, `data/`, and `results/`.

Model assets should be placed at:

```text
source/pp2d/models/polytamgen-dpo/polytamgen-dpo.pth
source/pp2d/models/base-llm/NLM-1b
```

Download `polytamgen-dpo.pth` and place it under
`source/pp2d/models/polytamgen-dpo/`. Extract `base-llm_NLM-1b.tar` from
`source/pp2d/models/` so it creates `source/pp2d/models/base-llm/NLM-1b/`.

The result archive is `polytamgen_full_results.tar.gz`; after
extraction it should provide `source/pp2d/results/`, including scored
T2/T3 metrics, raw generations, supplement outputs, figures, and analysis
inputs. See `ASSETS.md` and `ASSET_MANIFEST.csv` for the expected layout.

Processed datasets are stored under `source/pp2d/data/`:

```text
sft_train_valid/       SFT training and validation profiles
pdb_multitarget/       T2/T3 benchmark inference profiles and pocket features
pdb_benchmark/         Benchmark TSVs, pocket sequences, and docking pockets
dpo_preferences/       Preference pairs used for DPO training
```

## Paper Results

Paper-result files are provided by `polytamgen_full_results.tar.gz`.
Extract it before running the main paper-alignment checks. Start with
`REPRODUCIBILITY.md` for the validated reproduction workflow.

## Inference

PolyTamGen-DPO quick inference check on the T2 benchmark input:

```bash
cd source/pp2d
python scripts/inference.py \
  --config configs/polytamgen_dpo.yaml \
  --checkpoint models/polytamgen-dpo/polytamgen-dpo.pth \
  --test_data_path data/pdb_multitarget/T2/ligand_profiles.pkl \
  --split_path data/pdb_multitarget/T2/split.pkl \
  --lmdb_path data/pdb_multitarget/protein_features.lmdb \
  --split_key test \
  --output_dir outputs/inference/polytamgen-dpo-quick-check \
  --num_return_sequences 1 \
  --max_new_tokens 1 \
  --limit 2 \
  --batch_size 1 \
  --disable_sampling
```

Remove `--limit` and increase `--num_return_sequences` for full benchmark-scale
generation. Additional inference examples are documented in
`source/pp2d/README.md`.

## Training

SFT training:

```bash
cd source/pp2d
python scripts/train_pp2d.py --config configs/polytamgen_sft.yaml
```

DPO training:

```bash
cd source/pp2d
python scripts/train_dpo.py --config configs/polytamgen_dpo.yaml
```

For a quick configuration check, add `--dry_run` to either command. To force a
small CPU-friendly run, add:

```text
--distributed false --batch_size 1 --num_workers 0 --train_subset true --train_subset_size 1 --valid_subset true --valid_subset_size 1
```

The DPO preferences are built from scored T2 results with:

```bash
cd source/pp2d
python scripts/data/build_dpo_preferences.py \
  --input results/raw_generation/T2/PolyTamGen-SFT.json \
  --output data/dpo_preferences/preferences_5qed_train.jsonl \
  --valid-output data/dpo_preferences/preferences_5qed_valid.jsonl \
  --valid-fraction 0.2 \
  --qed-weight 5.0
```

The preference reward is:

```text
reward = -mean_active_vina + 5.0 * QED
```

Full DPO retraining starts from an SFT checkpoint produced by the SFT stage.

## Notes

- Numeric table checks are aligned to the precision reported in the paper.
- A full docking rerun requires the docking software stack and sufficient GPU or
  CPU compute; the provided scored files are the reference inputs for paper
  table reproduction.
- The numeric reproduction scripts regenerate the table values and available
  analysis figures from the released result files.
- Model assets provide the DPO checkpoint and base LLM. Data and result
  archives are distributed separately and should preserve the logical paths
  documented in `ASSETS.md`.
