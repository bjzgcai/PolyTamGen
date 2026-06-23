import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import LigandProfileDataset, ListwiseCollate, efficient_pad_sequences
from src.models.pp2d_model import PP2D


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
    dataset.all_protein_ids = ["P1", "P2"]
    dataset.protein_ids_set = {"P1", "P2"}
    dataset.sep_token_id = tokenizer.encode(",", add_special_tokens=False)[0]
    dataset.templates_listwise = []
    dataset.templates_active_only = ["Generate a molecule that binds to {ACT_BLOCK}."]
    dataset.protein_placeholder_id = -1
    dataset._get_protein_features = lambda pid: torch.full((2, 4), 1.0 if pid == "P1" else 2.0)
    return dataset


def _make_items(tokenizer):
    dataset = _build_dataset(tokenizer)
    dataset.data = [
        {
            "SMILES": "CO",
            "Active_Protein_IDs": ["P1"],
            "Inactive_Protein_IDs": [],
        },
        {
            "SMILES": "CCN",
            "Active_Protein_IDs": ["P1", "P2"],
            "Inactive_Protein_IDs": [],
        },
    ]
    return dataset[0], dataset[1]


class DummyAdapter(nn.Module):
    def __init__(self, output_tokens):
        super().__init__()
        self.output_tokens = output_tokens

    def forward(self, flat_feats, flat_mask):
        return self.output_tokens.to(flat_feats.device)


class DummyEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(vocab_size, hidden_size))

    def forward(self, input_ids):
        return self.weight[input_ids]


class RecordingLLM(nn.Module):
    def __init__(self, vocab_size=512, hidden_size=4):
        super().__init__()
        self.emb = DummyEmbedding(vocab_size, hidden_size)
        self.config = SimpleNamespace(pad_token_id=0, eos_token_id=102)
        self.dtype = torch.float32
        self.calls = []

    def get_input_embeddings(self):
        return self.emb

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        inputs_embeds = kwargs["inputs_embeds"]
        batch, seq_len, _ = inputs_embeds.shape
        logits = torch.zeros(batch, seq_len, self.emb.weight.shape[0], dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        return SimpleNamespace(loss=torch.tensor(3.25, dtype=inputs_embeds.dtype, device=inputs_embeds.device), logits=logits)


def _build_minimal_model(prot_tokens, strict_alignment_check=True):
    model = PP2D.__new__(PP2D)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        pp2d=SimpleNamespace(protein_adapter_type="qformer"),
        training=SimpleNamespace(strict_protein_alignment_check=strict_alignment_check),
        loss=SimpleNamespace(smiles_loss_type="cross_entropy"),
    )
    model.adapter = DummyAdapter(prot_tokens)
    model.set_transformer = None
    model.llm = RecordingLLM(hidden_size=prot_tokens.shape[-1])
    model._warned_missing_allowed_target = False
    model.register_buffer("label_smoothing_allowed_mask", torch.zeros(0, dtype=torch.bool), persistent=False)
    return model


def test_efficient_pad_sequences_left_padding_keeps_tail_alignment():
    seqs = [
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([3, 4, 5, 6], dtype=torch.long),
    ]

    padded = efficient_pad_sequences(seqs, padding_value=0, padding_side="left")

    assert padded.tolist() == [
        [0, 0, 1, 2],
        [3, 4, 5, 6],
    ]


def test_listwise_collate_left_padding_attention_and_label_masking():
    tokenizer = FakeTokenizer()
    item_short, item_long = _make_items(tokenizer)
    collate = ListwiseCollate(tokenizer)

    batch = collate([item_short, item_long])

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape

    # The shorter example should be left padded.
    assert batch["input_ids"][0, 0].item() == tokenizer.pad_token_id
    assert batch["attention_mask"][0, 0].item() == 0
    assert batch["labels"][0, 0].item() == -100

    # Prompt tokens stay masked, target tokens after <mol> remain supervised.
    mol_id = tokenizer.get_vocab()["<mol>"]
    for row in range(2):
        input_ids = batch["input_ids"][row]
        labels = batch["labels"][row]
        mol_pos = (input_ids == mol_id).nonzero(as_tuple=False)[0].item()
        assert torch.all(labels[:mol_pos] == -100)
        assert (labels[mol_pos:] != -100).any().item()

    # Placeholder count must match available proteins per sample.
    assert torch.equal(batch["placeholder_counts"], batch["protein_counts"])
    assert batch["protein_counts"].tolist() == [1, 2]


def test_prepare_llm_inputs_replaces_placeholders_with_feature_tokens_and_masks_labels():
    tokenizer = FakeTokenizer()
    item_short, item_long = _make_items(tokenizer)
    batch = ListwiseCollate(tokenizer)([item_short, item_long])

    prot_tokens = torch.tensor(
        [
            [[10.0, 10.0, 10.0, 10.0], [11.0, 11.0, 11.0, 11.0]],
            [[20.0, 20.0, 20.0, 20.0], [21.0, 21.0, 21.0, 21.0]],
            [[30.0, 30.0, 30.0, 30.0], [31.0, 31.0, 31.0, 31.0]],
        ],
        dtype=torch.float32,
    )
    model = _build_minimal_model(prot_tokens)

    prepared = model._prepare_llm_inputs(batch)
    final_embeds = prepared["final_embeds"]

    # Sample 0 had one placeholder, so sequence grows by S-1 = 1 token.
    assert final_embeds.shape[0] == 2
    assert batch["labels"].shape[:2] == final_embeds.shape[:2]

    # Injected protein spans must be masked out in labels.
    sample0_labels = batch["labels"][0]
    sample1_labels = batch["labels"][1]
    assert (sample0_labels == -100).sum().item() >= 2
    assert (sample1_labels == -100).sum().item() >= 4

    # Confirm the exact injected token vectors appear in order.
    sample0 = final_embeds[0]
    assert ((sample0 == prot_tokens[0, 0]).all(dim=-1)).any().item()
    assert ((sample0 == prot_tokens[0, 1]).all(dim=-1)).any().item()

    sample1 = final_embeds[1]
    for expected in [prot_tokens[1, 0], prot_tokens[1, 1], prot_tokens[2, 0], prot_tokens[2, 1]]:
        assert ((sample1 == expected).all(dim=-1)).any().item()


def test_prepare_llm_inputs_linear_adapter_trims_to_original_feature_lengths():
    tokenizer = FakeTokenizer()
    item_short, item_long = _make_items(tokenizer)
    batch = ListwiseCollate(tokenizer)([item_short, item_long])

    # Simulate per-protein original sequence lengths [1, 2, 1].
    batch["flat_protein_mask"] = torch.tensor(
        [
            [1, 0],
            [1, 1],
            [1, 0],
        ],
        dtype=torch.bool,
    )
    prot_tokens = torch.tensor(
        [
            [[10.0, 10.0, 10.0, 10.0], [11.0, 11.0, 11.0, 11.0]],
            [[20.0, 20.0, 20.0, 20.0], [21.0, 21.0, 21.0, 21.0]],
            [[30.0, 30.0, 30.0, 30.0], [31.0, 31.0, 31.0, 31.0]],
        ],
        dtype=torch.float32,
    )

    model = _build_minimal_model(prot_tokens)
    model.config.pp2d.protein_adapter_type = "linear"

    prepared = model._prepare_llm_inputs(batch)
    final_embeds = prepared["final_embeds"]

    # Sample 0 had one protein with valid len 1, so only the first token should survive.
    sample0 = final_embeds[0]
    assert ((sample0 == prot_tokens[0, 0]).all(dim=-1)).any().item()
    assert not ((sample0 == prot_tokens[0, 1]).all(dim=-1)).any().item()

    # Sample 1 uses proteins 1 and 2 with valid lens 2 and 1.
    sample1 = final_embeds[1]
    assert ((sample1 == prot_tokens[1, 0]).all(dim=-1)).any().item()
    assert ((sample1 == prot_tokens[1, 1]).all(dim=-1)).any().item()
    assert ((sample1 == prot_tokens[2, 0]).all(dim=-1)).any().item()
    assert not ((sample1 == prot_tokens[2, 1]).all(dim=-1)).any().item()

    # New attention mask must match the compacted sequence lengths.
    assert batch["attention_mask"].shape[:2] == final_embeds.shape[:2]
    real_tokens = batch["attention_mask"].sum(dim=1).tolist()
    assert all(x > 0 for x in real_tokens)


def test_prepare_llm_inputs_raises_on_placeholder_feature_mismatch():
    tokenizer = FakeTokenizer()
    item_short, _ = _make_items(tokenizer)
    batch = ListwiseCollate(tokenizer)([item_short])

    # Force mismatch: one text placeholder but zero valid gathered proteins.
    batch["gather_indices"] = torch.full_like(batch["gather_indices"], -1)

    model = _build_minimal_model(torch.zeros(1, 2, 4), strict_alignment_check=True)

    with pytest.raises(ValueError, match="alignment mismatch"):
        model._prepare_llm_inputs(batch)


def test_forward_cross_entropy_passes_masked_labels_into_llm():
    tokenizer = FakeTokenizer()
    item_short, _ = _make_items(tokenizer)
    batch = ListwiseCollate(tokenizer)([item_short])

    model = _build_minimal_model(
        torch.tensor([[[10.0, 10.0, 10.0, 10.0], [11.0, 11.0, 11.0, 11.0]]], dtype=torch.float32)
    )
    output = model.forward(batch)

    assert torch.isclose(output["loss"], torch.tensor(3.25))
    assert len(model.llm.calls) == 1

    llm_call = model.llm.calls[0]
    assert "labels" in llm_call
    assert llm_call["labels"].shape == batch["labels"].shape
    assert torch.equal(llm_call["labels"], batch["labels"])

    # The loss path should still preserve prompt masking after feature injection.
    assert (llm_call["labels"] == -100).any().item()
    assert (llm_call["labels"] != -100).any().item()
