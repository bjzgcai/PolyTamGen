import os
import re
import sys
from types import SimpleNamespace

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.data import LigandProfileDataset, ListwiseCollate, has_complete_mol_span, infer_mol_token_ids
from src.tokenizer import NatureLM1BTokenizer
from src.utils.generation_utils import (
    build_generation_allowed_token_ids,
    build_prefix_allowed_tokens_fn,
    build_generation_suppressed_token_ids,
    build_prompt_only_batch,
    extract_smiles_from_decoded_sequence,
)


_TOKENIZER = None


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

    def decode(self, ids):
        parts = []
        for idx in ids:
            idx = int(idx)
            if idx in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            parts.append(self._inv_vocab.get(idx, f"<unk:{idx}>"))
        return " ".join(parts)

    @property
    def eos_token(self):
        return "</s>"

    @property
    def pad_token(self):
        return "<pad>"

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


def _make_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    cfg = Config.from_yaml("configs/polytamgen_sft.yaml")
    model_path = cfg.pp2d.llm_model_name_or_path
    try:
        tokenizer = NatureLM1BTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"Using real tokenizer from: {model_path}")
        _TOKENIZER = tokenizer
        return _TOKENIZER
    except Exception as exc:
        print(f"Falling back to FakeTokenizer because real tokenizer load failed: {exc}")
        _TOKENIZER = FakeTokenizer()
        return _TOKENIZER


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
            "Active_Protein_IDs": ["P1"],
            "Inactive_Protein_IDs": [],
        }
    ]
    dataset.all_protein_ids = ["P1"]
    dataset.protein_ids_set = {"P1"}
    dataset.sep_token_id = tokenizer.encode(",", add_special_tokens=False)[0]
    dataset.templates_listwise = []
    dataset.templates_active_only = ["Generate a molecule that binds to {ACT_BLOCK}."]
    dataset.protein_placeholder_id = -1
    dataset._get_protein_features = lambda pid: torch.randn(2, 4)
    return dataset


def test_dataset_item_contains_reference_smiles_after_mol_token():
    tokenizer = _make_tokenizer()
    dataset = _build_dataset(tokenizer)

    item = dataset[0]
    mol_encoding = tokenizer.encode("<mol>CO</mol>", add_special_tokens=False)
    mol_id = mol_encoding[0]
    smiles_token_ids = mol_encoding[1:-1]
    end_mol_id = mol_encoding[-1]

    input_ids = item["input_ids"].tolist()
    mol_pos = input_ids.index(mol_id)

    assert input_ids[mol_pos + 1 : mol_pos + 1 + len(smiles_token_ids)] == smiles_token_ids
    assert input_ids[mol_pos + 1 + len(smiles_token_ids)] == end_mol_id


def test_collate_masks_prompt_but_keeps_reference_smiles_in_input_ids():
    tokenizer = _make_tokenizer()
    dataset = _build_dataset(tokenizer)
    item = dataset[0]
    collate = ListwiseCollate(tokenizer)

    batch = collate([item])
    mol_encoding = tokenizer.encode("<mol>CO</mol>", add_special_tokens=False)
    mol_id = mol_encoding[0]
    smiles_token_ids = mol_encoding[1:-1]

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    mol_pos = (input_ids == mol_id).nonzero(as_tuple=False)[0].item()

    assert torch.all(labels[:mol_pos] == -100)
    for token_id in smiles_token_ids:
        assert (input_ids == token_id).any().item()
        assert (labels == token_id).any().item()


def test_prompt_only_batch_removes_reference_smiles_tokens():
    tokenizer = _make_tokenizer()
    dataset = _build_dataset(tokenizer)
    item = dataset[0]
    collate = ListwiseCollate(tokenizer)

    batch = collate([item])
    prompt_only_batch = build_prompt_only_batch(batch, tokenizer)

    mol_encoding = tokenizer.encode("<mol>CO</mol>", add_special_tokens=False)
    mol_id = mol_encoding[0]
    smiles_token_ids = mol_encoding[1:-1]
    end_mol_id = mol_encoding[-1]

    prompt_ids = prompt_only_batch["input_ids"][0]

    assert (prompt_ids == mol_id).any().item()
    for token_id in smiles_token_ids:
        assert not (prompt_ids == token_id).any().item()
    assert not (prompt_ids == end_mol_id).any().item()


def test_extract_smiles_truncates_forbidden_domain_tokens():
    seq = "<mol><m>O<m>C<m>[C@H]<a>G<a>H</mol>"
    cleaned = extract_smiles_from_decoded_sequence(seq)
    assert cleaned == "OC[C@H]"


def test_truncated_samples_without_end_mol_are_rejected():
    tokenizer = _make_tokenizer()
    dataset = _build_dataset(tokenizer)
    dataset.config = SimpleNamespace(pp2d=SimpleNamespace(max_length=6))

    item = dataset[0]
    assert item is None


def test_complete_mol_span_helper_detects_missing_end_token():
    tokenizer = _make_tokenizer()
    mol_id, end_mol_id = infer_mol_token_ids(tokenizer)

    complete = torch.tensor([tokenizer.bos_token_id, mol_id, 201, end_mol_id, tokenizer.eos_token_id])
    truncated = torch.tensor([tokenizer.bos_token_id, mol_id, 201, tokenizer.eos_token_id])

    assert has_complete_mol_span(complete, mol_id, end_mol_id)
    assert not has_complete_mol_span(truncated, mol_id, end_mol_id)


def test_generation_allowed_tokens_exclude_non_molecule_tokens():
    tokenizer = _make_tokenizer()
    bang_id = tokenizer.encode("!", add_special_tokens=False)[0]

    allowed = build_generation_allowed_token_ids(tokenizer)
    suppressed = build_generation_suppressed_token_ids(tokenizer)
    prefix_fn = build_prefix_allowed_tokens_fn(allowed)

    assert bang_id not in allowed
    assert bang_id in suppressed
    assert tokenizer.eos_token_id in allowed
    assert prefix_fn(0, torch.tensor([tokenizer.bos_token_id, 11], dtype=torch.long)) == allowed


if __name__ == "__main__":
    tests = [
        test_dataset_item_contains_reference_smiles_after_mol_token,
        test_collate_masks_prompt_but_keeps_reference_smiles_in_input_ids,
        test_prompt_only_batch_removes_reference_smiles_tokens,
        test_extract_smiles_truncates_forbidden_domain_tokens,
        test_truncated_samples_without_end_mol_are_rejected,
        test_complete_mol_span_helper_detects_missing_end_token,
        test_generation_allowed_tokens_exclude_non_molecule_tokens,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
