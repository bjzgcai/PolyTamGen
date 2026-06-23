import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.generation_utils import (
    finalize_generation_metrics,
    init_generation_metrics,
    summarize_generation_behavior,
    update_generation_metrics,
)


def test_generation_metrics_track_long_repeat_and_exclamation_degeneracy():
    metrics = init_generation_metrics()
    decoded_sequences = [
        "<mol>CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC</mol>",
        "<mol>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!</mol>",
        "<mol>NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)</mol>",
    ]
    cleaned_sequences = [
        "C" * 127,
        "!" * 128,
        "NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)",
    ]

    update_generation_metrics(metrics, decoded_sequences, cleaned_sequences)
    summary = finalize_generation_metrics(metrics)

    assert summary["total_sequences"] == 3
    assert summary["mol_closure_count"] == 3
    assert summary["overlong_smiles_count"] == 2
    assert summary["exclamation_degeneracy_count"] == 1
    assert summary["repeat_like_count"] >= 1


def test_generation_metrics_still_count_valid_smiles():
    metrics = init_generation_metrics()
    decoded_sequences = ["<mol>CCO</mol>"]
    cleaned_sequences = ["CCO"]

    update_generation_metrics(metrics, decoded_sequences, cleaned_sequences)
    summary = finalize_generation_metrics(metrics)

    assert summary["valid_smiles_count"] == 1


def test_generation_behavior_summary_tracks_no_eos_rate():
    metrics = init_generation_metrics()
    decoded_sequences = ["<mol>CCO</mol>", "<mol>CCCCCCCC"]
    cleaned_sequences = ["CCO", "CCCCCCCC"]

    update_generation_metrics(metrics, decoded_sequences, cleaned_sequences)
    behavior = summarize_generation_behavior(
        metrics=metrics,
        cleaned_sequences=cleaned_sequences,
        generated_ids=torch.tensor([[11, 12, 99], [21, 22, 23]], dtype=torch.long),
        eos_token_id=99,
        max_new_tokens=16,
    )

    assert behavior["sample_count"] == 2
    assert behavior["no_eos_rate"] == 0.5
    assert behavior["avg_eos_position"] == 2.0
