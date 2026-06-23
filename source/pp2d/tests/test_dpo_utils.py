import math

import torch

from src.dpo import _normalize_pair, sequence_logps


def test_sequence_logps_ignores_prompt_and_padding():
    logits = torch.full((1, 4, 5), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0
    labels = torch.tensor([[-100, 1, 2, -100]])

    logps, counts = sequence_logps(logits, labels)

    assert counts.tolist() == [2]
    assert logps.item() > -1e-3


def test_sequence_logps_can_length_normalize():
    logits = torch.zeros((1, 3, 4))
    labels = torch.tensor([[-100, 1, 2]])

    summed, counts = sequence_logps(logits, labels, average_log_prob=False)
    averaged, _ = sequence_logps(logits, labels, average_log_prob=True)

    assert counts.tolist() == [2]
    assert math.isclose(averaged.item(), summed.item() / 2, rel_tol=1e-6)


def test_normalize_pair_accepts_project_column_aliases():
    pair = _normalize_pair(
        {
            "group_id": "g1",
            "winner_smiles": "CCO",
            "loser_smiles": "N#N",
            "P1": "1abc",
            "P2": "2abc",
            "inactive_pockets": "3abc,4abc",
        }
    )

    assert pair["pair_id"] == "g1"
    assert pair["chosen_smiles"] == "CCO"
    assert pair["rejected_smiles"] == "N#N"
    assert pair["Active_Protein_IDs"] == ["1abc", "2abc"]
    assert pair["Inactive_Protein_IDs"] == ["3abc", "4abc"]

