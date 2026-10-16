# Reproducibility guide

This guide describes installation, inference, and recomputation from the
PolyTamGen scored records. Each command block starts from the repository root
unless stated otherwise.

## Release scope

Use [Zenodo version 2026.10.16](https://doi.org/10.5281/zenodo.23417382)
with this code release. The original
[2026.06.11 record](https://zenodo.org/records/20594128) remains available.
Raw scores and generations are unchanged; the new results bundle updates the
derived tables, target-conditioning analysis, and verification reports.

The current table builder and verifier use `AvgDock < -8.18`, `QED > 0.25`,
and `SA_norm > 0.59` for the main strict-success metric. Non-finite AvgDock
values are excluded from its denominator. Metrics are computed within each
target context and then averaged equally across contexts. The former `-7.0`
cutoff is available as an explicitly labelled sensitivity analysis.

In addition to the unchanged scored records, the revised bundle includes
the revised weakest-target table and these rank-matched files required by
the full verifier:

```text
results/supplement/cross_target/target_conditioning_summary.json
results/supplement/cross_target/rank_matched_conditioning_pairs.csv
```

These files contain 1,300 rank-matched comparisons across 151 contexts. The
verifier checks row counts, rank uniqueness, paired score differences, and
context-equal means against the revised summary. It reads the existing
bootstrap interval and its seed/replicate metadata; it does not rerun the
bootstrap or regenerate docking scores. The old top-k summary is not a
replacement for these inputs.

## Known table differences

Recomputation and the existing pooled-protocol audit agree on two T3
strict-success standard deviations that differ from the revised manuscript:

| Method | Revised table | Recomputed standard deviation | Rounded to three decimals |
| --- | ---: | ---: | ---: |
| PocketXMol | 0.165 | 0.165540153 | 0.166 |
| PolyTamGen-SFT | 0.202 | 0.201470454 | 0.201 |

The verifier retains the manuscript reference values and reports these two
mismatches. They must be reconciled before claiming exact reproduction of all
main-table values. The nine T2/T3 main strict-success means match the revised
tables. See `analysis_inputs/revision_20260823/pooled_protocol/` in the revised
bundle for the earlier independent audit.

## 1. Install the package

Create and activate a Conda environment named `pp2d` with Python 3.12. Install
PyTorch and PyTorch Geometric builds that match your CPU or CUDA environment
before running the helper. With that environment active, run:

```bash
cd source/pp2d
python -m pip install --upgrade pip setuptools wheel
PP2D_SKIP_CREATE=1 bash setup_env.sh
python -m pip install -e . --no-deps --no-build-isolation
```

Set `PP2D_INSTALL_TORCH_STACK=1` when invoking the helper only if the torch
packages in `requirements.txt` suit your environment.

## 2. Download And Extract Assets

The Zenodo record is https://zenodo.org/records/23417382
(DOI: `10.5281/zenodo.23417382`). Download the four asset files listed in
`ASSETS.md`:

```text
polytamgen_data.tar.gz
base-llm_NLM-1b.tar
polytamgen-dpo.pth
polytamgen_full_results.tar.xz
```

Place them under `source/pp2d/` so these paths exist:

```text
source/pp2d/data/
source/pp2d/results/
source/pp2d/models/base-llm/NLM-1b
source/pp2d/models/polytamgen-dpo/polytamgen-dpo.pth
```

The DPO checkpoint and base LLM are required for checkpoint inference. The
result archive alone supplies the scored inputs for checking the archived
tables; see [release scope](#release-scope) for the revision boundary.

## 3. Run Lightweight Checks

From `source/pp2d/`, run the unit tests that do not require the full model
checkpoint:

```bash
cd source/pp2d
python -m pytest -q
```

Some tests are marked as real-model checks and require the DPO checkpoint plus
the base LLM at the documented `models/` paths.

## 4. Recompute main tables

The scored files from either results bundle can be used to recompute the
main tables under the revised cutoff:

```bash
cd source/pp2d
python scripts/paper/diagnostics/generate_consistency_tables_from_per_molecule.py --num_workers 1
```

The source files are:

```text
results/scored/T2/per_molecule.json
results/scored/T3/per_molecule.json
```

The output tables under `outputs/paper_rebuild/consistency_tables/` include
the cutoff in their headers. To reproduce the earlier endpoint as a separate
sensitivity output, run:

```bash
cd source/pp2d
python scripts/paper/diagnostics/generate_consistency_tables_from_per_molecule.py \
  --strict-threshold -7.0 \
  --output_dir outputs/paper_rebuild/sensitivity_m7p0 \
  --num_workers 1
```

## 5. Verify revised artifacts

After placing the revised results bundle under `source/pp2d/results/`, run:

```bash
cd source/pp2d
python scripts/paper/verify_paper_alignment.py
```

The reports are written under `outputs/paper_reproduction/`, outside the
downloaded inputs. Use `--results-root` to select an external results tree,
and `--out-json`/`--out-md` to select report paths. Missing revised
target-conditioning inputs are reported as failures. With the prepared
revised bundle, the two known standard-deviation differences above also
remain failures.

## 6. Rebuild available analysis artifacts

To regenerate available figures and run the same verifier:

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

## 7. Generate a small DPO sample

With `polytamgen-dpo.pth`, `base-llm_NLM-1b.tar`, and `polytamgen_data.tar.gz`
placed under `source/pp2d/`, generate a small sample on two T2 inputs:

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
  --max_new_tokens 256 \
  --limit 2 \
  --batch_size 1
```

Remove `--limit` and increase `--num_return_sequences` for benchmark-scale
generation. Keep sampling enabled and `--max_new_tokens 256`; choose a
separate output directory for the full run.

## 8. Optional full reruns

Full training and full docking-score recomputation are larger workloads. The
DPO checkpoint is ready for inference; full DPO retraining starts from an SFT
checkpoint produced by the SFT stage. A complete docking rerun requires the
docking software stack and the PDB pocket root at
`data/pdb_benchmark/pocket_pdbs`.
