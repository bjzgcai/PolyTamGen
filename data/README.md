# Data Assets

Training and benchmark data assets are distributed outside git. The
canonical runtime paths are under `source/pp2d/data/`.

Download and extract the data archive so commands run from `source/pp2d/`
can access paths such as:

```text
data/sft_train_valid/ligand_profiles.pkl
data/pdb_multitarget/T2/ligand_profiles.pkl
data/pdb_multitarget/protein_features.lmdb
```

See `../ASSETS.md` and `../ASSET_MANIFEST.csv` for the expected archive layout.
