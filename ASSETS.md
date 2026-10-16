# External Assets

This git repository tracks source code, lightweight metadata, and download
instructions. Large model, data, and result payloads are distributed as separate
archives and are not tracked by git.

## Zenodo Record

The Zenodo record is DOI `10.5281/zenodo.23417382`, version `2026.10.16`.
It updates the derived results while keeping the data archive, base LLM,
and DPO checkpoint unchanged. The original `2026.06.11` record remains at
DOI `10.5281/zenodo.20594128`. See
[release scope](REPRODUCIBILITY.md#release-scope) for the verifier's input
requirements and known table differences.

## Asset Files

- `polytamgen_data.tar.gz`: training, benchmark, and feature-store data
  assets.
- `base-llm_NLM-1b.tar`: base NatureLM/NLM directory required by the
  PolyTamGen configs for checkpoint inference.
- `polytamgen-dpo.pth`: final PolyTamGen-DPO checkpoint used for the main
  benchmark generation.
- `polytamgen_full_results.tar.gz`: revised benchmark result bundle, including
  molecule-level scored results, raw-generation outputs, supplement materials,
  figures, revised main tables, rank-matched conditioning outputs, and the
  full verification report with two known standard-deviation mismatches.

## Where To Extract

Commands in `source/pp2d/` use paths relative to that directory. After cloning
this repository, extract or place the asset files so these logical paths exist:

```text
source/pp2d/data/                         # from polytamgen_data.tar.gz or prepared locally
source/pp2d/results/                      # from polytamgen_full_results.tar.gz
source/pp2d/models/polytamgen-dpo/        # contains polytamgen-dpo.pth
source/pp2d/models/base-llm/NLM-1b        # from base-llm_NLM-1b.tar
```

Place `polytamgen-dpo.pth` at
`source/pp2d/models/polytamgen-dpo/polytamgen-dpo.pth`. Extract
`base-llm_NLM-1b.tar` from `source/pp2d/models/` so it creates
`source/pp2d/models/base-llm/NLM-1b/`.

The expected asset paths are listed in `ASSET_MANIFEST.csv`. Without the result
and data archives, you can read and install the source but commands that load
benchmark results, LMDB feature stores, or training profiles will fail. Without
the base LLM and DPO checkpoint, archived-result checks from downloaded full
results still work, but checkpoint inference or training will not run.
