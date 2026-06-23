# Source Modules

Core modules:

- `config.py`: YAML config dataclasses and command-line override helpers.
- `data.py`: datasets and collators for listwise PP2D.
- `dpo.py`: DPO trainer and preference-pair utilities.
- `trainer.py`: SFT training loop.
- `base_trainer.py`: base trainer utilities.
- `features.py`: feature handling helpers.
- `esm_feature_extractor.py`: ESM feature extraction utilities.
- `prompt_templates.py`: prompt-template loading and formatting.
- `science_tokens.py`: science token handling.
- `tokenizer.py`: NatureLM tokenizer wrapper.

Subpackages:

- `models/`: PP2D model, adapter, pocket encoder, and set transformer.
- `eval/`: docking and scoring helpers.
- `utils/`: checkpoint, distributed, generation, loss, scheduler, and logging utilities.

