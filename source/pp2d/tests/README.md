# Tests

Test groups:

- `test_dpo_utils.py`: DPO pair parsing and utility behavior.
- `test_generation_metrics.py`: generated molecule metric helpers.
- `test_generation_composite_score.py`: composite generation scoring.
- `test_label_smoothing_allowed.py`: label smoothing behavior.
- `test_placeholder_alignment.py`: prompt/data placeholder alignment.
- `test_rope.py`: rotary position embedding behavior.
- `test_inference_prompt_only.py`: inference prompt path.
- `test_data_and_model_pipeline.py`: data/model integration smoke path.
- `test_adaptor.py`: adapter module behavior.
- `test_real_model_smoke.py`: optional real-model smoke test.
- `test_subset.py`: subset utility behavior.

Run all tests:

```bash
python -m pytest -q
```

