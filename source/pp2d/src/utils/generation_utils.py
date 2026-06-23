import re
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from .allowed_words import allowed_words

try:
    from rdkit import Chem
except Exception:
    Chem = None


FORBIDDEN_DOMAIN_PREFIXES = ("<a>", "<d>", "<r>", "<i>")
FORBIDDEN_DOMAIN_TAGS = (
    "<protein>", "</protein>",
    "<dna>", "</dna>",
    "<rna>", "</rna>",
    "<material>", "</material>",
    "<antibody>", "</antibody>",
)


def get_mol_token_id(tokenizer: Any) -> int:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else tokenizer.vocab
    if "<mol>" in vocab:
        return vocab["<mol>"]
    ids = tokenizer.encode("<mol></mol>", add_special_tokens=False)
    if not ids:
        raise ValueError("Failed to infer <mol> token ID.")
    return ids[0]


def left_pad_tensors(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    flipped = [t.flip(0) for t in tensors]
    padded = pad_sequence(flipped, batch_first=True, padding_value=pad_value)
    return padded.flip(1)


def build_prompt_only_batch(batch: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    mol_token_id = get_mol_token_id(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_slices: List[torch.Tensor] = []
    label_slices: List[torch.Tensor] = []
    protein_mask_slices: List[torch.Tensor] = []

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    protein_token_mask = batch["protein_token_mask"]

    for row_ids, row_labels, row_protein_mask in zip(input_ids, labels, protein_token_mask):
        mol_pos = (row_ids == mol_token_id).nonzero(as_tuple=False)
        cut_idx = row_ids.shape[0] if mol_pos.numel() == 0 else mol_pos[0].item() + 1

        input_slices.append(row_ids[:cut_idx])
        label_slices.append(row_labels[:cut_idx])
        protein_mask_slices.append(row_protein_mask[:cut_idx])

    prompt_input_ids = left_pad_tensors(input_slices, pad_token_id)
    prompt_labels = left_pad_tensors(label_slices, -100)
    prompt_protein_mask = left_pad_tensors([x.long() for x in protein_mask_slices], 0).bool()

    prompt_batch = dict(batch)
    prompt_batch["input_ids"] = prompt_input_ids
    prompt_batch["labels"] = prompt_labels
    prompt_batch["protein_token_mask"] = prompt_protein_mask
    prompt_batch["attention_mask"] = (prompt_input_ids != pad_token_id).long()
    return prompt_batch


def build_bad_words_ids(tokenizer: Any) -> List[List[int]]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else tokenizer.vocab
    bad_token_ids = []
    eos_token = getattr(tokenizer, "eos_token", None)

    for token, token_id in vocab.items():
        if token == eos_token:
            continue
        if token.startswith(FORBIDDEN_DOMAIN_PREFIXES) or token in FORBIDDEN_DOMAIN_TAGS:
            bad_token_ids.append([int(token_id)])

    return bad_token_ids


def build_allowed_token_ids(tokenizer: Any) -> List[int]:
    """
    Build token IDs for molecule-allowed vocabulary based on `allowed_words.py`.
    Tokens missing from tokenizer vocab are skipped.
    """
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else tokenizer.vocab
    token_ids: List[int] = []
    for token in allowed_words:
        token_id = vocab.get(token)
        if token_id is not None:
            token_ids.append(int(token_id))
    return sorted(set(token_ids))


def build_generation_allowed_token_ids(tokenizer: Any) -> List[int]:
    token_ids = set(build_allowed_token_ids(tokenizer))
    for token_id in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ):
        if token_id is not None:
            token_ids.add(int(token_id))
    return sorted(token_ids)


def build_generation_suppressed_token_ids(tokenizer: Any) -> List[int]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else tokenizer.vocab
    allowed_token_ids = set(build_generation_allowed_token_ids(tokenizer))
    suppressed_token_ids: List[int] = []
    for _, token_id in vocab.items():
        token_id = int(token_id)
        if token_id not in allowed_token_ids:
            suppressed_token_ids.append(token_id)
    return sorted(set(suppressed_token_ids))


def build_prefix_allowed_tokens_fn(allowed_token_ids: List[int]) -> Callable[[int, torch.Tensor], List[int]]:
    allowed = list(sorted(set(int(token_id) for token_id in allowed_token_ids)))

    def _prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        return allowed

    return _prefix_allowed_tokens_fn


def extract_smiles_from_decoded_sequence(seq: str) -> str:
    cleaned = seq.rsplit("<mol>", 1)[-1] if "<mol>" in seq else seq

    stop_positions = []
    if "</mol>" in cleaned:
        stop_positions.append(cleaned.index("</mol>"))

    for marker in FORBIDDEN_DOMAIN_PREFIXES + FORBIDDEN_DOMAIN_TAGS:
        pos = cleaned.find(marker)
        if pos != -1:
            stop_positions.append(pos)

    if stop_positions:
        cleaned = cleaned[:min(stop_positions)]

    cleaned = cleaned.replace("<m>", "").replace("<mol>", "").replace("</mol>", "")
    cleaned = cleaned.strip().replace(" ", "")
    return cleaned


def contains_forbidden_domain_token(seq: str) -> bool:
    return any(marker in seq for marker in FORBIDDEN_DOMAIN_PREFIXES + FORBIDDEN_DOMAIN_TAGS)


def has_mol_closure(seq: str) -> bool:
    if "<mol>" in seq and "</mol>" in seq:
        return seq.rfind("</mol>") > seq.rfind("<mol>")
    return "</mol>" in seq


def is_valid_smiles(smiles: str) -> bool:
    if not smiles or not smiles.strip() or Chem is None:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def is_overlong_smiles(smiles: str, max_length: int = 120) -> bool:
    return len(smiles) >= int(max_length)


def has_excessive_repetition(smiles: str) -> bool:
    if not smiles:
        return False
    if re.search(r"(.)\1{5,}", smiles):
        return True
    token_pattern = re.compile(r"\[[^\]]+\]|Br|Cl|@@?|\\|/|\(|\)|\.|=|#|-|\+|[A-Z][a-z]?|[0-9]|[bcnohps]")
    tokens = token_pattern.findall(smiles)
    if not tokens:
        return False
    longest_run = 1
    current_run = 1
    for prev_token, token in zip(tokens, tokens[1:]):
        if token == prev_token:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    return longest_run >= 4


def has_exclamation_degeneracy(smiles: str) -> bool:
    return bool(smiles) and set(smiles) == {"!"}


def init_generation_metrics() -> Dict[str, int]:
    return {
        "total_sequences": 0,
        "valid_smiles_count": 0,
        "forbidden_domain_token_count": 0,
        "mol_closure_count": 0,
        "overlong_smiles_count": 0,
        "repeat_like_count": 0,
        "exclamation_degeneracy_count": 0,
    }


def update_generation_metrics(metrics: Dict[str, int], decoded_sequences: List[str], cleaned_sequences: List[str]) -> None:
    for raw_seq, cleaned in zip(decoded_sequences, cleaned_sequences):
        metrics["total_sequences"] += 1
        if contains_forbidden_domain_token(raw_seq):
            metrics["forbidden_domain_token_count"] += 1
        if has_mol_closure(raw_seq):
            metrics["mol_closure_count"] += 1
        if is_valid_smiles(cleaned):
            metrics["valid_smiles_count"] += 1
        if is_overlong_smiles(cleaned):
            metrics["overlong_smiles_count"] += 1
        if has_excessive_repetition(cleaned):
            metrics["repeat_like_count"] += 1
        if has_exclamation_degeneracy(cleaned):
            metrics["exclamation_degeneracy_count"] += 1


def finalize_generation_metrics(metrics: Dict[str, int]) -> Dict[str, Any]:
    total = max(metrics.get("total_sequences", 0), 1)
    return {
        **metrics,
        "valid_smiles_rate": metrics.get("valid_smiles_count", 0) / total,
        "forbidden_domain_token_rate": metrics.get("forbidden_domain_token_count", 0) / total,
        "mol_closure_rate": metrics.get("mol_closure_count", 0) / total,
        "overlong_smiles_rate": metrics.get("overlong_smiles_count", 0) / total,
        "repeat_like_rate": metrics.get("repeat_like_count", 0) / total,
        "exclamation_degeneracy_rate": metrics.get("exclamation_degeneracy_count", 0) / total,
        "rdkit_available": Chem is not None,
    }


def summarize_generation_behavior(
    metrics: Dict[str, int],
    cleaned_sequences: List[str],
    generated_ids: Any,
    eos_token_id: Optional[int],
    max_new_tokens: int,
) -> Dict[str, Any]:
    summary = finalize_generation_metrics(metrics)

    generated_rows: List[List[int]] = []
    if isinstance(generated_ids, torch.Tensor):
        generated_rows = generated_ids.detach().cpu().tolist()
    elif generated_ids is not None:
        generated_rows = [list(row) for row in generated_ids]

    unique_rate = (len(set(cleaned_sequences)) / max(len(cleaned_sequences), 1)) if cleaned_sequences else 0.0
    avg_len = float(sum(len(x) for x in cleaned_sequences) / max(len(cleaned_sequences), 1))

    eos_positions: List[int] = []
    hit_eos_count = 0
    generated_lens: List[int] = []
    for row in generated_rows:
        generated_lens.append(len(row))
        if eos_token_id is not None and eos_token_id in row:
            hit_eos_count += 1
            eos_positions.append(row.index(eos_token_id))

    avg_eos_position = float(sum(eos_positions) / len(eos_positions)) if eos_positions else -1.0
    no_eos_rate = 1.0 - (hit_eos_count / max(len(generated_rows), 1))

    near_max_threshold = max(0, int(max_new_tokens) - 5)
    near_max_flags = [gl >= near_max_threshold for gl in generated_lens]
    repeat_flags = [has_excessive_repetition(x) for x in cleaned_sequences]
    repeat_near_max_count = sum(1 for rep, near in zip(repeat_flags, near_max_flags) if rep and near)
    repeat_count = sum(1 for x in repeat_flags if x)
    repeat_near_max_ratio = repeat_near_max_count / max(repeat_count, 1)

    return {
        "metrics": summary,
        "unique_rate": unique_rate,
        "avg_smiles_length": avg_len,
        "avg_eos_position": avg_eos_position,
        "no_eos_rate": no_eos_rate,
        "repeat_near_max_ratio": repeat_near_max_ratio,
        "repeat_count": repeat_count,
        "sample_count": len(cleaned_sequences),
    }


def compute_generation_composite_score(
    forbidden_domain_token_rate: float,
    valid_smiles_rate: float,
    val_loss: float,
    mol_closure_rate: float = 0.0,
    overlong_smiles_rate: float = 0.0,
    repeat_like_rate: float = 0.0,
    exclamation_degeneracy_rate: float = 0.0,
    no_eos_rate: float = 0.0,
    zero_tol: float = 1e-12,
) -> float:
    """
    Composite score for generation-based checkpoint ranking.

    Ranking rule:
    1) forbidden_domain_token_rate == 0 first
    2) then valid_smiles_rate (higher is better)
    3) reward mol closure, penalize overlong / repetitive / degenerate outputs
    4) penalize generations that fail to emit EOS
    4) then val_loss (lower is better) as tie-break
    """
    forbidden = float(forbidden_domain_token_rate)
    valid = float(valid_smiles_rate)
    loss = float(val_loss)
    closure = float(mol_closure_rate)
    overlong = float(overlong_smiles_rate)
    repeat_like = float(repeat_like_rate)
    exclamation = float(exclamation_degeneracy_rate)
    no_eos = float(no_eos_rate)

    is_zero_forbidden = 1.0 if abs(forbidden) <= zero_tol else 0.0

    return (
        is_zero_forbidden * 1_000_000.0
        + valid * 1_000.0
        + closure * 100.0
        - overlong * 100.0
        - repeat_like * 100.0
        - exclamation * 1_000.0
        - no_eos * 200.0
        - loss
    )
