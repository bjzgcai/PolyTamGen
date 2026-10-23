# PP2D / PolyTamGen Source

This is the installable PolyTamGen package. `PP2D` is the
package name used by the code; it implements PolyTamGen, a
sequence-based multi-target molecule generator for polypharmacological design.

PolyTamGen conditions a NatureLM SMILES generator on a set of protein targets.
The released model uses frozen ESM-C 600M protein features, a Q-Former-style protein
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
[release scope](../../REPRODUCIBILITY.md#release-scope) for the revised inputs.

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

See [Protein feature extraction](#protein-feature-extraction) for the exact
encoder output, LMDB format, and a complete preparation example.

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

## Protein feature extraction

The published LMDBs are ready to use. Prepare new features only when adding
targets or checking an existing entry. Run the examples from `source/pp2d/`.

### Choose the sequence

Give each target a stable protein ID and its exact amino-acid sequence.
For a structure input, select the model and chain before extracting the
sequence. A sequence read from ATOM coordinates can omit unresolved residues
present in SEQRES or a supplied full-chain sequence.

A pocket PDB gives the residues included in that pocket. It cannot supply
missing full-chain residues. Decide whether to encode the full chain or the
pocket, and keep that choice when reusing the feature. For a reference check,
use the original sequence for that LMDB key. The released benchmark sequence
tables are `data/pdb_benchmark/test_T2_protein_sequences.tsv` and
`test_T3_protein_sequences.tsv`.

`scripts/data/build_pocket_sequences.py` selects the dominant chain in a
pocket, finds it in the full structure, and concatenates peptide fragments
built from its coordinates. It writes sequences and chain IDs. It does not
encode ESM features or fill unresolved residues from SEQRES. Check its output
against the sequence you intend to encode. Docking coordinates are separate
inputs and are not passed to the sequence encoder below.

### Extract the token features

Use ESM-C 600M with the December 2024 weights
(`esmc_600m_2024_12_v0.pth`), 36 transformer blocks, 18 attention heads, and
hidden width 1152. With `esm==3.2.1`, the forward outputs include:

```text
output.hidden_states        [36, batch, L + 2, 1152]
output.hidden_states[35, 0]  [L + 2, 1152]  <- save this for one sequence
output.embeddings           [batch, L + 2, 1152]  (after final LayerNorm)
```

Save the last block's hidden state before final LayerNorm. Keep BOS and EOS,
and do not average layers or pool tokens. The precomputed-input path consumes
`[L + 2, 1152]` features. A residue-only view, `states[1:-1]`, has `L` rows
and is a different input. Store each target without batch padding; the
collator treats BOS/EOS as valid tokens and adds right padding with a false
mask where needed.

`src/esm_feature_extractor.py` is a generic Forge helper. It defaults to
`esmc-300m-2024-12` and `pool=True`, so its defaults do not produce this
format. The ESM3 label in the paper figure refers to the encoding stage;
the released 1152-dimensional inputs use ESM-C 600M.

### Save the LMDB, profile, and split

The base requirements do not include the ESM SDK. This example uses
`esm==3.2.1` and its model builder, which downloads or reuses the ESM-C weights
through the Hugging Face cache. Obtain access to those weights if the upstream
repository requires it. The cache variables shown above can select a writable
location.

```bash
python -m pip install 'esm==3.2.1'
```

Create `work/new_target/sequences.json` as a JSON object mapping your protein
IDs to actual amino-acid sequences, without FASTA headers, whitespace, or gaps.
The example makes one inference profile with all those targets active.
Choose the CUDA device for your machine.

```python
import json
import pickle
from pathlib import Path

import lmdb
import torch
from esm.pretrained import ESMC_600M_202412
from esm.sdk.api import ESMProtein

work = Path("work/new_target")
sequences = json.loads((work / "sequences.json").read_text())
assert isinstance(sequences, dict) and sequences
assert all(isinstance(pid, str) and pid for pid in sequences)
assert all(isinstance(seq, str) and seq and seq.isalpha() and seq.isupper()
           for seq in sequences.values())

lmdb_path = work / "protein_features.lmdb"
profile_path = work / "ligand_profiles.pkl"
split_path = work / "split.pkl"
for path in (lmdb_path, profile_path, split_path):
    if path.exists():
        raise FileExistsError(path)

device = torch.device("cuda:0")
model = ESMC_600M_202412(device=device, use_flash_attn=False)
model = model.to(torch.bfloat16).eval().requires_grad_(False)
estimated_bytes = sum((len(seq) + 2) * 1152 * 2 for seq in sequences.values())
env = lmdb.open(str(lmdb_path), map_size=max(16 * 1024**2, 3 * estimated_bytes))
try:
    with torch.inference_mode(), env.begin(write=True) as txn:
        for protein_id, sequence in sequences.items():
            tokens = model.encode(ESMProtein(sequence=sequence)).sequence
            assert tokens.shape == (len(sequence) + 2,)
            assert int(tokens[0]) == model.tokenizer.bos_token_id
            assert int(tokens[-1]) == model.tokenizer.eos_token_id
            output = model(sequence_tokens=tokens.unsqueeze(0))
            assert output.hidden_states.shape == (36, 1, len(sequence) + 2, 1152)
            states = output.hidden_states[35, 0].to(dtype=torch.float16, device="cpu")
            assert torch.isfinite(states).all()
            txn.put(protein_id.encode("utf-8"), pickle.dumps(states), overwrite=False)
finally:
    env.close()

profile = [{"SMILES": "", "Active_Protein_IDs": list(sequences),
            "Inactive_Protein_IDs": []}]
with profile_path.open("xb") as handle:
    pickle.dump(profile, handle)
with split_path.open("xb") as handle:
    pickle.dump({"test": [0]}, handle)
```

Load the encoder once and run each sequence in its own forward. Features are
CPU float16 tensors serialized with pickle; `src/data.py` also accepts
`{"esm_hidden": states}`. Each key must match its protein ID in the profile.
Existing output files cause this example to stop before encoding.

Use the new-target inference command above after creating these three files.
The empty SMILES is for prompt-only inference. Training profiles require
observed ligand SMILES and training splits.

### Compare with a reference entry

Re-extract a known protein from its original sequence, using the same weights,
layer, special tokens, compute precision, attention backend, and batch size.
Compare the new CPU tensor `states` with that key in a trusted reference LMDB:

```python
reference = lmdb.open("/path/to/reference/protein_features.lmdb",
                      readonly=True, create=False, lock=False,
                      readahead=False, meminit=False)
try:
    with reference.begin() as txn:
        raw = txn.get(protein_id.encode("utf-8"))
    if raw is None:
        raise KeyError(protein_id)
    old = pickle.loads(raw)  # Only load a trusted local reference.
    if isinstance(old, dict):
        old = old["esm_hidden"]
    old = torch.as_tensor(old).cpu().to(torch.float16)
    assert old.shape == states.shape
    print("exact_equal:", torch.equal(old, states))
    print("max_abs:", (old.float() - states.float()).abs().max().item())
finally:
    reference.close()
```

Local checks matched a training-library entry (`A0A0B4J268`, 109 residues,
`[111, 1152]`) and a test-library entry (`6avs`, 236 residues, `[238, 1152]`)
element for element, with `max_abs=0.0`. The checked setup was A100 80GB,
CUDA 12.8, PyTorch 2.9.0+cu128, `esm==3.2.1`, CUDA bfloat16 computation,
Flash Attention disabled, and one sequence per forward. Batch size affects
the numerical result: a two-sequence forward did not match the stored test
entry exactly. Float16 storage does not imply float16 computation.

These are checks of two sampled rows. They do not establish equality for the
whole archive or recover its complete historical production workflow.
Other hardware or software may change the numbers; matching shapes or a high
cosine similarity alone does not establish exact equality.

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
and checks the revised derived artifacts:

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
