# Paper Consistency Tables

Optional consistency utilities. These scripts write regenerated checks under
`outputs/` and do not replace the scored benchmark sources.

Files:

- `generate_tables.py`: generates CSV consistency summaries from `results/scored/T2/per_target.csv`.
- `assemble_table_bundle.py`: copies a selected subset of generated consistency CSVs into a bundled output directory.
- `generate_consistency_tables_from_per_molecule.py`: rebuilds T2/T3 tables from a supplied `per_molecule.json` pair. It defaults to the revised `AvgDock < -8.18` strict-success cutoff and labels the cutoff in the generated headers. Use `--strict-threshold -7.0` with a separate output directory for sensitivity analysis. It loads one split at a time.

The verifier imports this builder's context-level metric function so both
use the same finite-score denominator and equal weighting across contexts.

Run through:

```bash
python scripts/paper/build_paper_artifacts.py --include-diagnostics
python scripts/paper/build_paper_artifacts.py --include-consistency-tables
```

Do not use generated consistency CSVs as replacements for scored benchmark
sources. The compact raw-generation rebuild can be compared
with `--include-consistency-tables --consistency-source raw-rebuild`, but those compact
tables are auxiliary and are not replacements for the scored benchmark sources.
