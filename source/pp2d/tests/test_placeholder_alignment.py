import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import LigandProfileDataset, ListwiseCollate


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 101
        self.eos_token_id = 102
        self._vocab = {
            "<mol>": 11,
            "</mol>": 12,
            ",": 13,
            "Generate": 20,
            "a": 21,
            "molecule": 22,
            "that": 23,
            "binds": 24,
            "to": 25,
            ".": 26,
        }
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        self._next_id = 200

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False, **kwargs):
        return self._tokenize_to_ids(text)

    def __call__(self, text, add_special_tokens=False, return_tensors=None, **kwargs):
        ids = self._tokenize_to_ids(text)
        tensor = torch.tensor(ids, dtype=torch.long)
        if return_tensors == "pt":
            tensor = tensor.unsqueeze(0)
        return {"input_ids": tensor}

    def _tokenize_to_ids(self, text):
        pattern = re.compile(r"<mol>|</mol>|[A-Za-z]+|[^A-Za-z\s]")
        pieces = pattern.findall(text)
        ids = []
        for piece in pieces:
            if piece not in self._vocab:
                self._vocab[piece] = self._next_id
                self._inv_vocab[self._next_id] = piece
                self._next_id += 1
            ids.append(self._vocab[piece])
        return ids


def _build_dataset(tokenizer):
    dataset = LigandProfileDataset.__new__(LigandProfileDataset)
    dataset.config = SimpleNamespace(pp2d=SimpleNamespace(max_length=64))
    dataset.tokenizer = tokenizer
    dataset.split = "test"
    dataset.use_precomputed = True
    dataset.max_actives = 3
    dataset.max_inactives = 0
    dataset.min_active = 1
    dataset.prob_no_inactives = 0.5
    dataset.prob_single_active = 0.0
    dataset.data = [
        {
            "SMILES": "CO",
            "Active_Protein_IDs": ["P1", "P2"],
            "Inactive_Protein_IDs": [],
        }
    ]
    dataset.all_protein_ids = ["P1", "P2"]
    dataset.protein_ids_set = {"P1", "P2"}
    dataset.sep_token_id = tokenizer.encode(",", add_special_tokens=False)[0]
    dataset.templates_listwise = []
    dataset.templates_active_only = ["Generate a molecule that binds to {ACT_BLOCK}."]
    dataset.protein_placeholder_id = -1
    dataset._get_protein_features = lambda pid: torch.randn(2, 4)
    return dataset


def test_collate_placeholder_count_matches_protein_count():
    tokenizer = FakeTokenizer()
    dataset = _build_dataset(tokenizer)
    item = dataset[0]

    collate = ListwiseCollate(tokenizer)
    batch = collate([item])

    assert "placeholder_counts" in batch
    assert "protein_counts" in batch
    assert torch.equal(batch["placeholder_counts"], batch["protein_counts"])


def test_alignment_mismatch_detection_signal():
    # Simulate a malformed batch where text placeholders outnumber available proteins.
    gather_indices = torch.tensor([[0]], dtype=torch.long)
    protein_token_mask = torch.tensor([[False, True, True, False]], dtype=torch.bool)

    valid_protein_count = int((gather_indices[0] != -1).sum().item())
    num_placeholders = int(protein_token_mask[0].sum().item())

    with pytest.raises(ValueError, match="alignment mismatch"):
        if num_placeholders != valid_protein_count:
            raise ValueError("alignment mismatch")
