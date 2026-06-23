# Models

This directory documents the model files used by PolyTamGen commands. Runtime
commands are run from `source/pp2d/`, so place model assets under
`source/pp2d/models/`. Large payloads are not tracked by git; see
`../ASSETS.md` for the asset layout.

## DPO Inference

Place these files before running DPO inference examples:

```text
source/pp2d/models/polytamgen-dpo/polytamgen-dpo.pth
source/pp2d/models/base-llm/NLM-1b
```

`polytamgen-dpo.pth` is the final PolyTamGen-DPO checkpoint used for the main
benchmark generation. `base-llm_NLM-1b.tar` contains the base NatureLM/NLM
generator expected by the DPO config.

## Training

The SFT checkpoint path is used by retraining workflows when the SFT stage is
rerun locally.
