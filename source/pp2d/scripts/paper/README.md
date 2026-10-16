# Paper Scripts

These scripts expect the full-results archive to be extracted from `source/pp2d/`
so `results/` exists next to `scripts/`.

Files:

- `reproduce_benchmark_results.py`: benchmark reproduction entry point. It validates `results/benchmark/manifest.csv`, checks raw-generation SMILES alignment against scored rows, rebuilds T2/T3 consistency tables from scored `per_molecule.json`, and runs the alignment verifier.
- `verify_paper_alignment.py`: recomputes main-table metrics at `AvgDock < -8.18` and checks the revised weakest-target and rank-matched target-conditioning outputs against manuscript reference values. It accepts `--results-root`, writes reports under `outputs/paper_reproduction/` by default, and never writes into the input results tree.
- `build_paper_artifacts.py`: rebuilds available figures under `outputs/paper_rebuild/`, optionally rebuilds molecule-level consistency tables, and then runs the alignment verifier.
- `rebuild_per_molecule_from_raw.py`: recomputes compact auxiliary molecule rows from raw generation JSONs and available per-pocket docking scores under `outputs/raw_generation_rebuild/`.
- `plot_joint_distribution.py`: rebuilds the joint distribution figure from per-molecule scored data.
- `plot_flexribbon_affinity.py`: rebuilds the main-text FlexRibbon affinity distribution figure from FlexRibbon predictions and RDKit-aligned analysis inputs.

The scored files used for paper table reproduction are
`results/scored/T2/per_molecule.json` and
`results/scored/T3/per_molecule.json`. The builder and verifier share the same
context-level metric calculation. The main cutoff is `-8.18`; `-7.0` is an
explicit sensitivity setting. Use the `2026.10.16` results bundle, which includes
the revised target-conditioning outputs. See
[release scope](../../../../REPRODUCIBILITY.md#release-scope) for input details.

The raw generation files can also be rebuilt into compact auxiliary molecule
rows with RDKit properties plus available per-pocket docking scores:

```bash
python scripts/paper/rebuild_per_molecule_from_raw.py
```

The recommended benchmark reproduction command is:

```bash
python scripts/paper/reproduce_benchmark_results.py
```

It writes `outputs/paper_reproduction/reproduction_report.{json,md}`.
The verifier reads the same `--results-root` as the reproduction command and
writes its JSON/Markdown reports into that command's output directory. When
using an external results tree, also point `--manifest` to that tree's
`benchmark/manifest.csv`.

The lower-level T2/T3 main-table consistency check uses the scored files:

```bash
python scripts/paper/build_paper_artifacts.py --include-consistency-tables
```

Use `--include-consistency-tables --consistency-source raw-rebuild` only when comparing
the compact raw-generation rebuild after running `rebuild_per_molecule_from_raw.py`.

Full docking-score recomputation is a separate benchmark rerun using
`scripts/benchmark/benchmark_multitarget_docking.py` and
`data/pdb_benchmark/pocket_pdbs/`.
