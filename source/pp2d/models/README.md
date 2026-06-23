# Model Assets

This directory is intentionally empty in git. Place model assets here before
running checkpoint inference or training commands from `source/pp2d/`.

Required for DPO inference:

```text
models/polytamgen-dpo/polytamgen-dpo.pth
models/base-llm/NLM-1b
```

`polytamgen-dpo.pth` is the DPO checkpoint. `base-llm_NLM-1b.tar` contains the
base language model; extract it from `source/pp2d/models/` so it creates
`models/base-llm/NLM-1b/`.

SFT checkpoint paths are used by retraining workflows when the SFT stage is
rerun.
