# Result Assets

This directory is intentionally empty in git. Download
the four `polytamgen_full_results.tar.xz.part01` through `.part04` volumes,
then assemble and extract them from `source/pp2d/`:

```bash
cd source/pp2d
cat /path/to/polytamgen_full_results.tar.xz.part0[1-4] > polytamgen_full_results.tar.xz
tar -xf polytamgen_full_results.tar.xz
```

After extraction, this directory should contain `scored/`, `raw_generation/`,
`benchmark/`, `analysis_inputs/`, `supplement/`, and `figures/`.
