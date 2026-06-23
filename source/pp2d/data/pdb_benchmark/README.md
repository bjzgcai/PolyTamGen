# PDB Benchmark Splits And Docking Inputs

Benchmark split files:

- `test_all.tsv`: T2 benchmark target-pair split used by the main benchmark scoring command.
- `test_T3.tsv`: T3 benchmark target-triplet split.
- `test_T2_protein_sequences.tsv`: T2 pocket sequence table.
- `test_T2_protein_sequences.fasta`: T2 pocket sequence FASTA.
- `test_T2_protein_sequence_failures.tsv`: pockets where sequence extraction failed.
- `test_T3_protein_sequences.tsv`: T3 pocket sequence table.
- `test_T3_protein_sequences.fasta`: T3 pocket sequence FASTA.
- `test_T3_protein_sequence_failures.tsv`: pockets where sequence extraction failed.
- `pocket_pdbs/`: nested pocket-PDB root used by docking reruns; set `PP2D_PDB_DIR` to this directory.
- `pocket_pdbs_manifest.csv`: pocket id, expected relative path, copy status, and source path provenance for the 288 paper split pockets.

These files define the paper T2/T3 benchmark targets and the receptor inputs
needed to recompute docking scores. They are not model checkpoints or generated
molecules.

The docking script expects paths like:

```text
pocket_pdbs/8a/8a0z/8a0z_pocket_1.pdb
```

To rebuild `pocket_pdbs/` from a prepared receptor cache:

```bash
python scripts/data/materialize_docking_pocket_root.py \
  --source-cache /path/to/source/pp2d/.cache
```
