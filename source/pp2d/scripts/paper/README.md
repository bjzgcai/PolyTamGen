# Paper Scripts

These scripts expect the full-results archive to be extracted from `source/pp2d/`
so `results/` exists next to `scripts/`.

Files:

- `reproduce_benchmark_results.py`: benchmark reproduction entry point. It validates `results/benchmark/manifest.csv`, checks raw-generation SMILES alignment against scored rows, rebuilds T2/T3 consistency tables from scored `per_molecule.json`, and runs the alignment verifier.
- `verify_paper_alignment.py`: verifies stored artifacts against the paper numbers. This is the main alignment check.
- `build_paper_artifacts.py`: rebuilds available figures under `outputs/paper_rebuild/`, optionally rebuilds molecule-level consistency tables, and then runs the alignment verifier.
- `rebuild_per_molecule_from_raw.py`: recomputes compact auxiliary molecule rows from raw generation JSONs and available per-pocket docking scores under `outputs/raw_generation_rebuild/`.
- `plot_joint_distribution.py`: rebuilds the joint distribution figure from per-molecule scored data.
- `plot_flexribbon_affinity.py`: rebuilds the main-text FlexRibbon affinity distribution figure from FlexRibbon predictions and RDKit-aligned analysis inputs.

The scored files used for paper table reproduction are
`results/scored/T2/per_molecule.json` and
`results/scored/T3/per_molecule.json`. They reproduce the T2/T3 main paper
tables exactly.

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

The lower-level T2/T3 main-table consistency check uses the scored files:

```bash
python scripts/paper/build_paper_artifacts.py --include-consistency-tables
```

Use `--include-consistency-tables --consistency-source raw-rebuild` only when comparing
the compact raw-generation rebuild after running `rebuild_per_molecule_from_raw.py`.

Full docking-score recomputation is a separate benchmark rerun using
`scripts/benchmark/benchmark_multitarget_docking.py` and
`data/pdb_benchmark/pocket_pdbs/`.
