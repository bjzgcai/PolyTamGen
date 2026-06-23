import csv
import json
import logging
import math
import os
import pickle
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .config import Config
from .data import (
    BaseProteinDataset,
    ListwiseCollate,
    efficient_pad_sequences,
    has_complete_mol_span,
    infer_mol_token_ids,
)
from .models.pp2d_model import PP2D
from .tokenizer import NatureLM1BTokenizer
from .trainer import PP2DTrainer
from .utils.checkpoint_manager import CheckpointManager
from .utils.distributed_utils import ddp_barrier
from .utils.generation_utils import build_allowed_token_ids

logger = logging.getLogger(__name__)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    for sep in ["|", ";", ","]:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return text.split()


def _first_present(record: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _read_records(path: str) -> List[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            rows = []
            for key, value in obj.items():
                if isinstance(value, dict):
                    value = dict(value)
                    value.setdefault("group_id", key)
                    rows.append(value)
            return rows
        raise ValueError(f"Unsupported JSON preference payload in {path}: {type(obj)!r}")
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return list(obj.values())
        raise ValueError(f"Unsupported pickle preference payload in {path}: {type(obj)!r}")
    if suffix in {".tsv", ".csv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(path, "r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported preference file extension: {path}")


def _normalize_pair(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    chosen = _first_present(
        record,
        [
            "chosen_smiles",
            "winner_smiles",
            "positive_smiles",
            "preferred_smiles",
            "y_w",
            "chosen",
        ],
    )
    rejected = _first_present(
        record,
        [
            "rejected_smiles",
            "loser_smiles",
            "negative_smiles",
            "dispreferred_smiles",
            "y_l",
            "rejected",
        ],
    )
    if not chosen or not rejected:
        return None

    actives = _as_list(
        _first_present(
            record,
            ["Active_Protein_IDs", "active_ids", "active_pockets", "targets", "target_ids"],
        )
    )
    if not actives:
        actives = [str(record.get(k, "")).strip() for k in ("P1", "P2", "P3") if str(record.get(k, "")).strip()]

    inactives = _as_list(
        _first_present(
            record,
            ["Inactive_Protein_IDs", "inactive_ids", "inactive_pockets", "anti_targets", "off_targets"],
        )
    )

    return {
        "pair_id": str(_first_present(record, ["pair_id", "group_id", "id"]) or ""),
        "chosen_smiles": str(chosen),
        "rejected_smiles": str(rejected),
        "Active_Protein_IDs": actives,
        "Inactive_Protein_IDs": inactives,
        "chosen_reward": _first_present(record, ["chosen_reward", "winner_reward"]),
        "rejected_reward": _first_present(record, ["rejected_reward", "loser_reward"]),
    }


class PreferencePairDataset(BaseProteinDataset):
    """Preference-pair dataset used by PP2D DPO training."""

    def __init__(self, config: Config, tokenizer, split: str = "train"):
        super().__init__(config, tokenizer, split)
        self.max_actives = config.training.max_active
        self.max_inactives = config.training.max_inactive
        self.min_active = config.training.min_active
        self.prob_no_inactives = getattr(config.training, "prob_no_inactives", 0.5)
        self.prob_single_active = getattr(config.training, "prob_single_active", 0.5)

        self.data = self._load_pairs()
        self._init_lmdb_and_keys()
        self._filter_data()
        self._apply_subset()

        self.sep_token_id = None
        for token in [",", ", "]:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                self.sep_token_id = ids[0]
                break
        if self.sep_token_id is None:
            self.sep_token_id = self.tokenizer.encode(",", add_special_tokens=False)[0]

        self.templates_listwise = config.pp2d.prompt_templates
        self.templates_active_only = config.pp2d.prompt_templates_active_only
        self.protein_placeholder_id = -1

    def _resolve_preference_path(self) -> str:
        cfg = self.config.data
        if self.split == "train" and cfg.preference_train_path:
            return cfg.preference_train_path
        if self.split == "valid" and cfg.preference_valid_path:
            return cfg.preference_valid_path
        if self.split == "test" and cfg.preference_test_path:
            return cfg.preference_test_path
        if cfg.preference_path:
            return cfg.preference_path
        raise ValueError(
            "DPO training needs data.preference_path or split-specific "
            "data.preference_train_path/data.preference_valid_path."
        )

    def _load_pairs(self) -> List[Dict[str, Any]]:
        path = self._resolve_preference_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preference file not found: {path}")
        rows = _read_records(path)
        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if "split" in row and not self.config.data.preference_train_path:
                row_split = "valid" if str(row["split"]).lower() == "val" else str(row["split"]).lower()
                if row_split != self.split:
                    continue
            pair = _normalize_pair(row)
            if pair is not None:
                normalized.append(pair)
        if not normalized:
            raise ValueError(
                f"No preference pairs found in {path}. Expected columns such as "
                "chosen_smiles/rejected_smiles plus active protein IDs."
            )
        logger.info("Loaded %d DPO preference pairs from %s", len(normalized), path)
        return normalized

    def _filter_data(self) -> None:
        if not self.data:
            return
        if not hasattr(self, "protein_ids_set"):
            self.protein_ids_set = set(self.all_protein_ids)
        valid_data = []
        sample_requested_ids = []
        for item in self.data:
            sample_requested_ids.extend(item.get("Active_Protein_IDs", [])[:2])
            valid_actives = [pid for pid in item.get("Active_Protein_IDs", []) if pid in self.protein_ids_set]
            if len(valid_actives) >= self.min_active:
                item = dict(item)
                item["Active_Protein_IDs"] = valid_actives
                item["Inactive_Protein_IDs"] = [
                    pid for pid in item.get("Inactive_Protein_IDs", []) if pid in self.protein_ids_set
                ]
                valid_data.append(item)
        logger.info("Filtered DPO pairs: %d -> %d", len(self.data), len(valid_data))
        self.data = valid_data
        if not self.data:
            sample_ids = sorted(set(sample_requested_ids))[:8]
            raise ValueError(
                "All DPO preference pairs were filtered out because their active protein IDs "
                f"were absent from LMDB path {self.lmdb_path!r}. "
                "Check that data.lmdb_path matches the ID namespace used by the preference file. "
                f"Sample requested IDs: {sample_ids}; LMDB key count: {len(self.protein_ids_set)}."
            )

    def __len__(self) -> int:
        return len(self.data)

    def _select_template(self, templates: List[str], idx: int) -> str:
        if not templates:
            return ""
        if self.split in ["test", "valid"]:
            return templates[idx % len(templates)]
        return random.choice(templates)

    def _sample_ids(self, item: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        all_actives = list(item.get("Active_Protein_IDs", []))
        all_inactives = list(item.get("Inactive_Protein_IDs", []))
        if len(all_actives) < self.min_active:
            return [], []

        if self.split in ["test", "valid"]:
            sampled_actives = all_actives[: min(len(all_actives), self.max_actives)]
            sampled_inactives = all_inactives[: min(len(all_inactives), self.max_inactives)]
            return sampled_actives, sampled_inactives

        if random.random() < self.prob_single_active:
            n_act_target = 1
        else:
            n_act_target = random.randint(self.min_active, self.max_actives)
        sampled_actives = random.sample(all_actives, min(len(all_actives), n_act_target))

        sampled_inactives: List[str] = []
        if random.random() >= self.prob_no_inactives:
            n_inact_target = random.randint(0, self.max_inactives)
            if all_inactives:
                sampled_inactives = random.sample(all_inactives, min(len(all_inactives), n_inact_target))
            else:
                active_set = set(all_actives)
                attempts = 0
                while len(sampled_inactives) < n_inact_target and attempts < n_inact_target * 2:
                    pid = random.choice(self.all_protein_ids)
                    if pid not in active_set and pid not in sampled_inactives:
                        sampled_inactives.append(pid)
                    attempts += 1
        return sampled_actives, sampled_inactives

    def _build_sample(
        self,
        smiles: str,
        sampled_actives_ids: List[str],
        sampled_inactives_ids: List[str],
        template: str,
    ) -> Optional[Dict[str, Any]]:
        active_feats = []
        kept_actives = []
        for pid in sampled_actives_ids:
            feat = self._get_protein_features(pid)
            if feat is not None:
                active_feats.append(feat)
                kept_actives.append(pid)

        inactive_feats = []
        kept_inactives = []
        for pid in sampled_inactives_ids:
            feat = self._get_protein_features(pid)
            if feat is not None:
                inactive_feats.append(feat)
                kept_inactives.append(pid)

        num_actives = len(active_feats)
        num_inactives = len(inactive_feats)
        if num_actives < self.min_active:
            return None

        def create_separated_placeholders(count: int) -> torch.Tensor:
            if count <= 0:
                return torch.tensor([], dtype=torch.long)
            if count == 1:
                return torch.full((1,), self.protein_placeholder_id, dtype=torch.long)
            total_len = 2 * count - 1
            out = torch.full((total_len,), self.sep_token_id, dtype=torch.long)
            out[0::2] = self.protein_placeholder_id
            return out

        parts = []
        if "{ACT_BLOCK}" in template:
            pre_act, post_act = template.split("{ACT_BLOCK}", 1)
            parts.append(self._tokenize_segment(pre_act))
            parts.append(create_separated_placeholders(num_actives))
            remaining = post_act
        else:
            remaining = template

        if "{INACT_BLOCK}" in remaining:
            pre_inact, post_inact = remaining.split("{INACT_BLOCK}", 1)
            parts.append(self._tokenize_segment(pre_inact))
            parts.append(create_separated_placeholders(num_inactives))
            parts.append(self._tokenize_segment(post_inact))
        else:
            parts.append(self._tokenize_segment(remaining))

        parts.append(self._tokenize_segment(f" <mol>{smiles}</mol>"))
        input_ids = torch.cat([part for part in parts if part.numel() > 0], dim=0)

        max_len = self.config.pp2d.max_length
        if self.tokenizer.bos_token_id is not None:
            input_ids = torch.cat([torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long), input_ids])
        if input_ids.shape[0] > max_len - 1:
            input_ids = input_ids[: max_len - 1]
        if self.tokenizer.eos_token_id is not None:
            input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)])

        mol_token_id, end_mol_token_id = infer_mol_token_ids(self.tokenizer)
        if not has_complete_mol_span(input_ids, mol_token_id, end_mol_token_id):
            return None

        return {
            "input_ids": input_ids,
            "active_feats": active_feats,
            "inactive_feats": inactive_feats,
            "num_actives": num_actives,
            "num_inactives": num_inactives,
            "smiles": smiles,
            "active_ids": kept_actives,
            "inactive_ids": kept_inactives,
        }

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item = self.data[idx]
        sampled_actives, sampled_inactives = self._sample_ids(item)
        if not sampled_actives:
            return None

        if sampled_inactives and self.templates_listwise:
            template = self._select_template(self.templates_listwise, idx)
        elif not sampled_inactives and self.templates_active_only:
            template = self._select_template(self.templates_active_only, idx)
        elif sampled_inactives:
            template = "Generate a molecule that binds to {ACT_BLOCK} and avoids {INACT_BLOCK}."
        else:
            template = "Generate a molecule that binds to {ACT_BLOCK}."

        chosen = self._build_sample(item["chosen_smiles"], sampled_actives, sampled_inactives, template)
        rejected = self._build_sample(item["rejected_smiles"], sampled_actives, sampled_inactives, template)
        if chosen is None or rejected is None:
            return None
        return {
            "chosen": chosen,
            "rejected": rejected,
            "pair_id": item.get("pair_id", str(idx)),
            "chosen_reward": item.get("chosen_reward"),
            "rejected_reward": item.get("rejected_reward"),
        }


class PreferencePairCollate:
    def __init__(self, tokenizer):
        self.inner = ListwiseCollate(tokenizer)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}
        chosen = self.inner([item["chosen"] for item in batch])
        rejected = self.inner([item["rejected"] for item in batch])
        if not chosen or not rejected:
            return {}
        return {
            "chosen": chosen,
            "rejected": rejected,
            "pair_ids": [item.get("pair_id", "") for item in batch],
            "chosen_rewards": [item.get("chosen_reward") for item in batch],
            "rejected_rewards": [item.get("rejected_reward") for item in batch],
        }


def sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~loss_mask, 0)
    per_token_logps = torch.gather(
        F.log_softmax(shift_logits, dim=-1),
        dim=2,
        index=safe_labels.unsqueeze(2),
    ).squeeze(2)
    token_counts = loss_mask.sum(dim=1).clamp(min=1)
    summed = (per_token_logps * loss_mask).sum(dim=1)
    if average_log_prob:
        return summed / token_counts, token_counts
    return summed, token_counts


def _logits_only_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(batch)
    if "attention_mask" in out and torch.is_tensor(out["attention_mask"]):
        out["attention_mask"] = out["attention_mask"].clone()
    if "labels" in out and torch.is_tensor(out["labels"]):
        out["labels"] = out["labels"].clone()
    out["compute_loss"] = False
    return out


class _DPOCheckpointView(torch.nn.Module):
    def __init__(self, policy_model: torch.nn.Module, reference_model: torch.nn.Module):
        super().__init__()
        self.policy_model = policy_model
        self.reference_model = reference_model

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        state = dict(self.policy_model.state_dict(*args, **kwargs))
        state.update({f"reference_model.{k}": v for k, v in self.reference_model.state_dict(*args, **kwargs).items()})
        return state


class DPOTrainer(PP2DTrainer):
    """PP2D trainer for direct preference optimization."""

    def _setup_model(self):
        self.recorder.info("Initializing DPO policy and frozen reference model...")
        model_path = self.config.pp2d.llm_model_name_or_path
        try:
            self.tokenizer = NatureLM1BTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.config.loss.pad_token_id = self.tokenizer.pad_token_id

        pretrained_path = getattr(self.config.pp2d, "pretrained_model_path", None)
        if not pretrained_path:
            raise ValueError("DPO requires pp2d.pretrained_model_path to initialize the policy from SFT.")

        self.model = PP2D.from_pretrained(pretrained_path, self.config).to(self.device)
        ref_path = self.config.dpo.reference_model_path or pretrained_path
        self.reference_model = PP2D.from_pretrained(ref_path, self.config).to(self.device)
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

        for model in [self.model, self.reference_model]:
            if hasattr(model, "set_label_smoothing_allowed_token_ids"):
                model.set_label_smoothing_allowed_token_ids(build_allowed_token_ids(self.tokenizer))

        if torch.distributed.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=getattr(self.config.distributed, "find_unused_parameters", True),
            )

    def _setup_data(self):
        self.recorder.info("Loading DPO preference datasets...")
        self.train_dataset = PreferencePairDataset(self.config, self.tokenizer, split="train")
        self.valid_dataset = PreferencePairDataset(self.config, self.tokenizer, split="valid")
        collate = PreferencePairCollate(self.tokenizer)

        num_workers = self.config.training.num_workers
        persistent_workers = bool(getattr(self.config.training, "persistent_workers", False)) and num_workers > 0
        pin_memory = bool(getattr(self.config.training, "pin_memory", True))
        dataloader_timeout_sec = int(getattr(self.config.training, "dataloader_timeout_sec", 0))

        if torch.distributed.is_initialized():
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=getattr(self.config.training, "seed", 42),
            )
            self.valid_sampler = DistributedSampler(
                self.valid_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            self.train_sampler = RandomSampler(self.train_dataset)
            self.valid_sampler = SequentialSampler(self.valid_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=self.train_sampler,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            timeout=dataloader_timeout_sec,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config.training.batch_size,
            sampler=self.valid_sampler,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            timeout=dataloader_timeout_sec,
        )
        self.recorder.info(
            f"DPO train samples: {len(self.train_dataset)} | valid samples: {len(self.valid_dataset)}"
        )
        self.recorder.info(f"DPO train batches: {len(self.train_loader)} | Workers: {num_workers}")

    def _dpo_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        chosen_batch = batch["chosen"]
        rejected_batch = batch["rejected"]
        average_log_prob = bool(getattr(self.config.dpo, "length_normalize_logps", False))

        policy_chosen = self.model(_logits_only_batch(chosen_batch))
        policy_rejected = self.model(_logits_only_batch(rejected_batch))
        policy_chosen_logps, _ = sequence_logps(
            policy_chosen["logits"],
            policy_chosen["labels"],
            average_log_prob=average_log_prob,
        )
        policy_rejected_logps, _ = sequence_logps(
            policy_rejected["logits"],
            policy_rejected["labels"],
            average_log_prob=average_log_prob,
        )

        if bool(getattr(self.config.dpo, "reference_free", False)):
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            with torch.no_grad():
                ref_chosen = self.reference_model(_logits_only_batch(chosen_batch))
                ref_rejected = self.reference_model(_logits_only_batch(rejected_batch))
                ref_chosen_logps, _ = sequence_logps(
                    ref_chosen["logits"],
                    ref_chosen["labels"],
                    average_log_prob=average_log_prob,
                )
                ref_rejected_logps, _ = sequence_logps(
                    ref_rejected["logits"],
                    ref_rejected["labels"],
                    average_log_prob=average_log_prob,
                )

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        beta = float(getattr(self.config.dpo, "beta", 0.1))
        losses = -F.logsigmoid(beta * logits)
        loss = losses.mean()

        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
        reward_margin = chosen_rewards - rejected_rewards
        reward_accuracy = reward_margin.gt(0).float().mean()

        return {
            "loss": loss,
            "dpo_loss": loss.detach(),
            "reward_accuracy": reward_accuracy.detach(),
            "reward_margin": reward_margin.mean().detach(),
            "chosen_logp": policy_chosen_logps.mean().detach(),
            "rejected_logp": policy_rejected_logps.mean().detach(),
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._dpo_step(batch)

    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._dpo_step(batch)

    def _resume_from_checkpoint(self):
        checkpoint = self.ckpt_manager.load_latest()
        if not checkpoint:
            self.recorder.info("No checkpoint found.")
            return

        state_dict = checkpoint["model_state_dict"]
        policy_state = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
            if not k.startswith("reference_model.") and not k.startswith("module.reference_model.")
        }
        ref_state = {}
        for key, value in state_dict.items():
            clean = key.replace("module.", "", 1)
            if clean.startswith("reference_model."):
                ref_state[clean[len("reference_model.") :]] = value

        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        missing, unexpected = raw_model.load_state_dict(policy_state, strict=False)
        if missing:
            self.recorder.warning(f"DPO resume missing policy keys: {missing[:10]}")
        if unexpected:
            self.recorder.warning(f"DPO resume unexpected policy keys: {unexpected[:10]}")
        if ref_state:
            self.reference_model.load_state_dict(ref_state, strict=False)

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif self.optimizer and getattr(self.config.training, "save_optimizer_last", False):
            opt_ckpt = self.ckpt_manager.load_latest_optimizer()
            if opt_ckpt and "optimizer_state_dict" in opt_ckpt:
                self.optimizer.load_state_dict(opt_ckpt["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("step", 0)
        saved_epoch = checkpoint.get("epoch", 0)
        saved_config = checkpoint.get("config", {}) or {}
        resume_state = saved_config.get("resume_state", {})
        self.resume_batch_idx = resume_state.get("batch_idx", 0)
        if self.resume_batch_idx > 0:
            self.start_epoch = saved_epoch
        else:
            self.start_epoch = saved_epoch + 1

    def save_checkpoint(
        self,
        metric_value: float,
        batch_idx: int = 0,
        manager: Optional[CheckpointManager] = None,
    ):
        if is_dataclass(self.config):
            config_dict = asdict(self.config)
        else:
            config_dict = vars(self.config) if hasattr(self.config, "__dict__") else {}

        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        raw_policy = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint_view = _DPOCheckpointView(raw_policy, self.reference_model)
        target_manager = manager if manager is not None else self.ckpt_manager
        target_manager.save(
            model=checkpoint_view,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            metric_value=metric_value,
            config={"static_config": config_dict, "resume_state": {"batch_idx": batch_idx, "rng_states": rng_states}},
        )
