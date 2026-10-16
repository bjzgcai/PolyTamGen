# PP2D / PolyTamGen Source

This is the installable PolyTamGen package. `PP2D` is the
package name used by the code; it implements PolyTamGen, a
sequence-based multi-target molecule generator for polypharmacological design.

PolyTamGen conditions a NatureLM SMILES generator on a set of protein targets.
The paper model uses frozen ESM3 protein features, a Q-Former-style protein
adapter, an explicit set-wise interaction module across target tokens, and
soft-prompt token injection into the autoregressive molecule generator. Training
has two stages: supervised fine-tuning (SFT) on ligand-centric bioactivity
profiles, followed by direct preference optimization (DPO) using docking and QED
preference pairs. Unless an ablation is being reproduced, use the DPO checkpoint
as the default PolyTamGen model.

## Paper Connection

The paper studies sequence-based polypharmacological target-aware molecule
generation. This tree contains the source, configs, data entrypoints,
and paper-alignment scripts for that study.

Data and tasks:

- `D_bio`: ChEMBL/BindingDB-derived ligand-centric bioactivity profiles with
  about 1.65 million unique ligands and 4.87 million ligand-target records.
- `D_PDB`: PDB-derived multi-target structural records with 2,047 unique ligands
  and about 21k ligand-target records, used for DPO preferences and paper
  docking benchmarks.
- T2 benchmark: 151 PDB-selected target pairs.
- T3 benchmark: 59 PDB-selected target triples. T2 and T3 are different
  benchmark populations, so their headline numbers should not be read as a
  strict monotonic difficulty comparison.

Main metrics:

- `AvgDock`: average Vina docking score over all intended targets; lower is
  better.
- `SR_strict`: strict success rate requiring `QED > 0.25`,
  `SA_norm > 0.59`, and `AvgDock < -8.18`; records with non-finite AvgDock
  are excluded from its denominator.
- Validity, diversity, QED, and synthetic accessibility (SA).

PolyTamGen-DPO values under the revised main protocol:

| Split | AvgDock | SR_strict | Validity | Diversity | QED | SA |
|---|---:|---:|---:|---:|---:|---:|
| T2 | -7.642 | 0.331 | 0.968 | 0.837 | 0.565 | 0.805 |
| T3 | -7.918 | 0.405 | 0.966 | 0.841 | 0.567 | 0.806 |

The earlier `-7.0` cutoff is retained for sensitivity analysis. Use the
`2026.10.16` results bundle; see
[release scope](../../REPRODUCIBILITY.md#release-scope) for the revised inputs
and two known T3 standard-deviation differences.

The paper also reports weakest-target diagnostics, FlexRibbon affinity
prediction, cross-target stress testing, strain analysis, module ablations, DPO
reward sensitivity, and a structural case study. The current `results/`
bundle contains the revised derived analyses. Its rank-matched conditioning
outputs replace the earlier cross-target top-k summaries.

## Install

Create and activate a Conda environment named `pp2d`. Install PyTorch and
PyTorch Geometric builds that match your CPU or CUDA environment first.
Then run these commands from this directory:

```bash
python -m pip install --upgrade pip setuptools wheel
PP2D_SKIP_CREATE=1 bash setup_env.sh
python -m pip install -e . --no-deps --no-build-isolation
```

Dependencies are recorded in `requirements.txt`. The editable install is run
with `--no-deps` because torch and torch_geometric must match the machine CUDA
stack. `setup_env.sh` installs the non-torch dependency set after a compatible
torch stack is present.

Set `PP2D_INSTALL_TORCH_STACK=1` only when the torch stack in
`requirements.txt` already matches the target machine.


## Required Assets

Large assets are downloaded separately and placed under this directory.
After extraction or preparation, these logical paths should exist:

```text
models/polytamgen-dpo/polytamgen-dpo.pth
models/base-llm/NLM-1b
results/scored/T2/per_molecule.json
results/scored/T3/per_molecule.json
data/pdb_multitarget/T2/ligand_profiles.pkl
data/pdb_multitarget/protein_features.lmdb
```

Zenodo provides `polytamgen-dpo.pth` and `base-llm_NLM-1b.tar`.
Place them so `models/polytamgen-dpo/polytamgen-dpo.pth` and
`models/base-llm/NLM-1b` exist. See `../../ASSETS.md` and
`../../ASSET_MANIFEST.csv` for the archive layout.

## Directory Map

Top-level files:

- `pyproject.toml`: editable package metadata.
- `requirements.txt`: Python dependencies recorded for the project.
- `setup_env.sh`: helper for building a compatible environment.

Directories:

- `configs/`: canonical PolyTamGen SFT and DPO training configs.
- `data/`: SFT train/valid profiles, PDB-selected T2/T3 test profiles,
  benchmark docking inputs, and DPO preference JSONL.
- `models/`: downloaded or prepared checkpoint and base-LLM paths expected by configs.
- `results/`: downloaded full-results archive contents used by paper checks.
- `scripts/`: command-line entrypoints for training, inference, scoring, paper
  checks, and data preparation.
- `src/`: importable PP2D implementation.
- `tests/`: lightweight, unit, and smoke tests.

Important split convention:

- `data/sft_train_valid/` is for SFT training and validation only.
- The T2/T3 paper tests are PDB-selected so docking can be computed.
  Their inference profiles live in `data/pdb_multitarget/`, and the matching
  docking TSV/PDB inputs live in `data/pdb_benchmark/`.

## Training

SFT uses `data/sft_train_valid/ligand_profiles.pkl` and the UniProt-style
feature LMDB:

```bash
python scripts/train_pp2d.py --config configs/polytamgen_sft.yaml
```

DPO training uses the PDB-pocket keyed preference JSONL files and starts from
an SFT checkpoint produced by the SFT stage:

```bash
python scripts/train_dpo.py --config configs/polytamgen_dpo.yaml
```

Both configs auto-detect available GPUs when `distributed.world_size: null`.
For a quick startup check, use `--dry_run`; for tiny local checks, also override
batch size, workers, and subset options on the CLI.

The DPO reward used by the curated preference files is:

```text
reward = -mean_active_vina + 5.0 * QED
```

## Inference

DPO inference is the primary model workflow. It uses the DPO checkpoint plus
the base language model at `models/base-llm/NLM-1b`.

Optional SFT quick check on the validation split. This requires an SFT
checkpoint:

```bash
python scripts/inference.py \
  --config configs/polytamgen_sft.yaml \
  --checkpoint models/polytamgen-sft/polytamgen-sft.pth \
  --test_data_path data/sft_train_valid/ligand_profiles.pkl \
  --split_path data/sft_train_valid/splits/train_valid_split.pkl \
  --lmdb_path data/sft_train_valid/protein_features.lmdb \
  --split_key valid \
  --output_dir outputs/inference/polytamgen-sft-quick-check \
  --num_return_sequences 1 \
  --max_new_tokens 1 \
  --limit 2 \
  --batch_size 1 \
  --disable_sampling
```

Generate a small DPO sample on two T2 benchmark inputs:

```bash
python scripts/inference.py \
  --config configs/polytamgen_dpo.yaml \
  --checkpoint models/polytamgen-dpo/polytamgen-dpo.pth \
  --test_data_path data/pdb_multitarget/T2/ligand_profiles.pkl \
  --split_path data/pdb_multitarget/T2/split.pkl \
  --lmdb_path data/pdb_multitarget/protein_features.lmdb \
  --split_key test \
  --output_dir outputs/inference/polytamgen-dpo-T2-quick-check \
  --num_return_sequences 1 \
  --max_new_tokens 256 \
  --limit 2 \
  --batch_size 1
```

### Preparing New Multi-Target Inputs

Inference does not extract protein features on the fly. For a new target set,
first extract and save protein features, then point inference to the prepared
profile, split, and LMDB files:

```text
--test_data_path  ligand profile pickle
--split_path      split-index pickle
--lmdb_path       protein feature LMDB
```

The protein IDs in the profile must exactly match the LMDB keys. A minimal
two-target profile looks like this:

```python
[
    {
        "SMILES": "reference_or_seed_smiles",
        "Active_Protein_IDs": ["targetA_chainX", "targetB_chainY"],
        "Inactive_Protein_IDs": [],
    }
]
```

Use a split file such as:

```python
{"train": [], "valid": [], "val": [], "test": [0]}
```

Feature preparation checklist:

1. Pick stable IDs for every chain or pocket, for example
   `9w2f_chainB` and `9w2f_chainC`.
2. Extract ESM-C/ESM3-C token embeddings with the same code path used for the
   existing feature stores. Keep the per-token tensor, including ESM special
   tokens, so the shape is normally `sequence_length + 2` by `1152` for the
   ESM-C 600M features used by the current checkpoints.
3. Store each tensor in LMDB under its exact protein ID using
   `pickle.dumps(tensor.half().cpu())`.
4. Sanity-check a known protein by re-extracting it and comparing with the old
   LMDB value. A matching setup should give `max_abs=0.0` after loading both
   tensors in the same dtype.

When downloading HuggingFace models, point caches to a writable location if the
default user cache is not suitable:

```bash
export HF_HOME=/path/to/writable/huggingface-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

Then run normal multi-target inference:

```bash
conda run -n pp2d python scripts/inference.py \
  --config configs/polytamgen_dpo.yaml \
  --checkpoint models/polytamgen-dpo/polytamgen-dpo.pth \
  --test_data_path work/new_target/ligand_profiles.pkl \
  --split_path work/new_target/split.pkl \
  --lmdb_path work/new_target/protein_features.lmdb \
  --split_key test \
  --output_dir outputs/inference/new-target \
  --num_return_sequences 100 \
  --max_new_tokens 256 \
  --batch_size 1 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 100
```

For benchmark-scale generation, omit `--limit`, increase
`--num_return_sequences` to the intended sampling scale, keep sampling enabled
and `--max_new_tokens 256`, and use the T2 or T3 paths under
`data/pdb_multitarget/`. Choose a separate output directory for the full run.
Multi-GPU inference is available with
`--devices 0,1,2,3`; the script shards the selected split and merges
`combined_inference_results.json`.

## Evaluation And Paper Checks

Basic generated-molecule evaluation:

```bash
python scripts/evaluate.py \
  --results outputs/inference/polytamgen-dpo-T2/combined_inference_results.json \
  --output_dir outputs/eval/polytamgen-dpo-T2
```

Docking-enabled evaluation requires the nested paper pocket-PDB root:

```bash
python scripts/evaluate.py \
  --results outputs/inference/polytamgen-dpo-T2/combined_inference_results.json \
  --output_dir outputs/eval/polytamgen-dpo-T2-docking \
  --enable_docking \
  --pdb_dir data/pdb_benchmark/pocket_pdbs \
  --docking_cache_dir outputs/eval/docking_cache \
  --save_docking_scores
```

To rerun benchmark scoring from the raw-generation files:

```bash
python scripts/benchmark/benchmark_multitarget_docking.py \
  --model_config scripts/benchmark/configs/t2_inputs.yaml \
  --tsv_path data/pdb_benchmark/test_all.tsv \
  --pdb_dir data/pdb_benchmark/pocket_pdbs \
  --docking_cache_dir outputs/benchmark/docking_cache \
  --output_dir outputs/benchmark/T2/recomputed
```

For T3, use `scripts/benchmark/configs/t3_inputs.yaml` and
`data/pdb_benchmark/test_T3.tsv`.

The verifier recomputes the revised main-table metrics from the scored files
and checks the revised derived artifacts. It retains the manuscript's
reference values, including the two known standard-deviation mismatches:

```bash
python scripts/paper/verify_paper_alignment.py
```

The paper artifact builder writes regenerated figures and optional consistency
tables under `outputs/paper_rebuild/`, then runs the alignment verifier:

```bash
python scripts/paper/build_paper_artifacts.py --include-consistency-tables
```

The canonical scored files are:

```text
results/scored/T2/per_molecule.json
results/scored/T3/per_molecule.json
```

They contain molecule-level scores for the full archived benchmarks. The
table builder and verifier use the revised main cutoff; passing the old
stored strict-success flag through unchanged is not sufficient. See
[release scope](../../REPRODUCIBILITY.md#release-scope) before comparing
outputs with manuscript tables. The raw-generation rebuild path
writes only under
`outputs/raw_generation_rebuild/` and is an auxiliary consistency check rather
than the scored source for paper table reproduction.

## Tests

Run the lightweight test suite from this directory:

```bash
python -m pytest -q
```

The optional real-model checks require the DPO checkpoint and base LLM through
the `models/` paths. SFT checks additionally require an SFT checkpoint.

## Scope Notes

PolyTamGen is an early-stage computational hypothesis-generation framework. The
paper results are docking- and predictor-based benchmark evidence, not
experimental validation of polypharmacology. The current model is trained with
active-only supervision and is not an explicit anti-target or selectivity-aware
generator. The cross-target stress test therefore belongs to evaluation and
diagnosis, not to the core training objective.
