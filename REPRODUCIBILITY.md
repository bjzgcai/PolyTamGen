# Reproducibility Guide

This guide describes the commands used to reproduce the reported PolyTamGen
paper numbers from the source tree and the Zenodo assets. Commands assume
the repository root unless a `cd source/pp2d` command is shown.

## 1. Install The Package

Create a Python environment with a PyTorch and PyTorch Geometric build that
matches the local CUDA stack. Then install PolyTamGen in editable mode:

```bash
cd source/pp2d
PY=${PY:-python}
python -m pip install -e . --no-deps --no-build-isolation
```

The helper script can install the non-torch dependency set after a compatible
torch stack is already present:

```bash
cd source/pp2d
PP2D_SKIP_CREATE=1 bash setup_env.sh
```

Set `PP2D_INSTALL_TORCH_STACK=1` only when the torch stack pinned in
`requirements.txt` is appropriate for the target machine.

## 2. Download And Extract Assets

The Zenodo record is https://zenodo.org/records/20594128
(DOI: `10.5281/zenodo.20594128`). Download the four asset files listed in
`ASSETS.md`:

```text
polytamgen_data.tar.gz
base-llm_NLM-1b.tar
polytamgen-dpo.pth
polytamgen_full_results.tar.gz
```

Place them under `source/pp2d/` so these paths exist:

```text
source/pp2d/data/
source/pp2d/results/
source/pp2d/models/base-llm/NLM-1b
source/pp2d/models/polytamgen-dpo/polytamgen-dpo.pth
```

The DPO checkpoint and base LLM are required for real checkpoint inference. The
result archive alone is sufficient for checking the paper tables against the
downloaded scored results.

## 3. Run Lightweight Checks

From `source/pp2d/`, run the unit tests that do not require the full model
checkpoint:

```bash
cd source/pp2d
python -m pytest -q
```

Some tests are marked as real-model checks and require the DPO checkpoint plus
the base LLM at the documented `models/` paths.

## 4. Reproduce Paper Numbers

After extracting `polytamgen_full_results.tar.gz`, run the paper-alignment
checker:

```bash
cd source/pp2d
python scripts/paper/verify_paper_alignment.py
```

The expected result is zero alignment failures. The canonical scored sources are:

```text
results/scored/T2/per_molecule.json
results/scored/T3/per_molecule.json
```

These files reproduce the T2/T3 main paper tables exactly. They also support the
supplemental checks described in `source/pp2d/README.md`.

## 5. Rebuild Available Analysis Artifacts

To regenerate available figures and run the same paper-alignment verifier:

```bash
cd source/pp2d
python scripts/paper/build_paper_artifacts.py
```

Optional table consistency outputs can be generated with:

```bash
cd source/pp2d
python scripts/paper/build_paper_artifacts.py --include-consistency-tables
```

These outputs are written under `outputs/` and do not modify the downloaded result
files.

## 6. Run A Quick DPO Inference Check

With `polytamgen-dpo.pth`, `base-llm_NLM-1b.tar`, and `polytamgen_data.tar.gz`
placed under `source/pp2d/`, run:

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

Remove `--limit` and increase `--num_return_sequences` for benchmark-scale
generation.

## 7. Optional Full Reruns

Full training and full docking-score recomputation are larger workloads. The
DPO checkpoint is ready for inference; full DPO retraining starts from an SFT
checkpoint produced by the SFT stage. A complete docking rerun requires the
docking software stack and the PDB pocket root at
`data/pdb_benchmark/pocket_pdbs`.
