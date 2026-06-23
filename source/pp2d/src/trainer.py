import torch
import math
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any
from transformers import PreTrainedTokenizer

from .base_trainer import BaseTrainer
from .models.pp2d_model import PP2D
from .data import LigandProfileDataset, ListwiseCollate
from .tokenizer import NatureLM1BTokenizer
from .utils.scheduler_utils import create_scheduler
from .utils.generation_utils import (
    build_allowed_token_ids,
    build_bad_words_ids,
    build_generation_allowed_token_ids,
    build_prefix_allowed_tokens_fn,
    build_generation_suppressed_token_ids,
    build_prompt_only_batch,
    extract_smiles_from_decoded_sequence,
    init_generation_metrics,
    summarize_generation_behavior,
    update_generation_metrics,
)

class PP2DTrainer(BaseTrainer):
    """
    Trainer for PP2D model.
    """

    def _setup_model(self):
        self.recorder.info("Initializing Model...")
        model_path = self.config.pp2d.llm_model_name_or_path
        try:
            self.tokenizer = NatureLM1BTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.config.loss.pad_token_id = self.tokenizer.pad_token_id

        pretrained_path = getattr(self.config.pp2d, 'pretrained_model_path', None)
        if pretrained_path:
            self.recorder.info(f"Using pretrained checkpoint: {pretrained_path}")
            self.model = PP2D.from_pretrained(pretrained_path, self.config).to(self.device)
        else:
            self.recorder.info("No pretrained checkpoint specified. Initializing from scratch.")
            self.model = PP2D(config=self.config).to(self.device)

        model_module = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(model_module, "set_label_smoothing_allowed_token_ids"):
            allowed_token_ids = build_allowed_token_ids(self.tokenizer)
            model_module.set_label_smoothing_allowed_token_ids(allowed_token_ids)
            self.recorder.info(
                f"Configured label-smoothing allowed tokens: {len(allowed_token_ids)} / {model_module.vocab_size}"
            )

        if torch.distributed.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=getattr(self.config.distributed, 'find_unused_parameters', True)
            )

    def _setup_data(self):
        """
        Initializes Datasets AND DataLoaders (Persistent).
        """
        self.recorder.info("Loading Datasets...")

        self.train_dataset = LigandProfileDataset(self.config, self.tokenizer, split='train')
        self.valid_dataset = LigandProfileDataset(self.config, self.tokenizer, split='valid')
        train_collate = ListwiseCollate(self.tokenizer)
        valid_collate = ListwiseCollate(self.tokenizer)
        
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
                seed=getattr(self.config.training, 'seed', 42)
            )
        else:
            self.train_sampler = RandomSampler(self.train_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=self.train_sampler,
            num_workers=num_workers,
            collate_fn=train_collate,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            timeout=dataloader_timeout_sec,
        )

        if torch.distributed.is_initialized():
            self.valid_sampler = DistributedSampler(
                self.valid_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )
        else:
            self.valid_sampler = SequentialSampler(self.valid_dataset)

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config.training.batch_size,
            sampler=self.valid_sampler,
            num_workers=num_workers, 
            collate_fn=valid_collate,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            timeout=dataloader_timeout_sec,
        )
        
        self.recorder.info(f"Train batches: {len(self.train_loader)} | Workers: {num_workers}")

    def _setup_optimizer(self):
        self.recorder.info("Initializing Optimizer...")
        model_module = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(model_module, "get_parameter_groups"):
            param_groups = model_module.get_parameter_groups()
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=getattr(self.config.training, 'weight_decay', 0.01)
            )
        else:
            trainable_params = [p for p in model_module.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=getattr(self.config.training, 'weight_decay', 0.01)
            )

        # Log trainable parameter groups
        total_params = 0
        for idx, group in enumerate(self.optimizer.param_groups):
            group_params = sum(p.numel() for p in group['params'])
            total_params += group_params
            group_name = group.get('name', f'group_{idx}')
            group_lr = group.get('lr', None)
            self.recorder.info(
                f"Trainable group [{group_name}] | params: {group_params:,} | lr: {group_lr}"
            )
        self.recorder.info(f"Total trainable params: {total_params:,}")
        
        update_freq = max(1, int(getattr(self.config.training, 'update_freq', 1)))
        # Scheduler should track optimizer updates, not micro-batches.
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / update_freq))
        
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config.training.lr_scheduler,
            total_epochs=self.config.training.epochs,
            base_lr=self.config.training.learning_rate,
            steps_per_epoch=steps_per_epoch
        )

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(batch)
        if isinstance(outputs, dict):
            loss = outputs.get('smiles_loss', outputs.get('loss'))
        else:
            loss = outputs[0]
        return {'loss': loss}

    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(batch)
        if isinstance(outputs, dict):
            loss = outputs.get('smiles_loss', outputs.get('loss'))
        else:
            loss = outputs[0]
        return {'loss': loss}

    def should_run_generation_validation(self) -> bool:
        enabled = bool(getattr(self.config.training, "generation_validation_enabled", False))
        if not enabled:
            return False

        if torch.distributed.is_initialized() and self.world_size > 1:
            return bool(getattr(self.config.training, "generation_validation_distributed", False))

        return True

    def run_generation_validation(self) -> Dict[str, float]:
        if self.valid_loader is None:
            return {}

        num_samples = max(0, int(getattr(self.config.training, "generation_validation_num_samples", 0)))
        if num_samples <= 0:
            return {}

        model_module = self.model.module if isinstance(self.model, DDP) else self.model
        bad_words_ids = build_bad_words_ids(self.tokenizer)
        generation_allowed_token_ids = build_generation_allowed_token_ids(self.tokenizer)
        prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(generation_allowed_token_ids)
        suppress_tokens = build_generation_suppressed_token_ids(self.tokenizer)
        metrics = init_generation_metrics()
        processed = 0
        all_cleaned_sequences = []
        all_generated_rows = []
        max_new_tokens = int(getattr(self.config.training, "generation_validation_max_new_tokens", 256))

        model_module.eval()
        for batch in self.valid_loader:
            if batch is None or (isinstance(batch, dict) and len(batch) == 0):
                continue

            batch_size = len(batch.get('active_ids', []))
            if batch_size == 0:
                continue

            prompt_batch = build_prompt_only_batch(batch, self.tokenizer)
            batch_on_device = self._move_to_device(prompt_batch)

            with torch.inference_mode():
                generated = model_module.generate(
                    batch=batch_on_device,
                    max_new_tokens=max_new_tokens,
                    num_beams=int(getattr(self.config.training, "generation_validation_num_beams", 1)),
                    do_sample=not bool(getattr(self.config.training, "generation_validation_disable_sampling", True)),
                    num_return_sequences=1,
                    temperature=float(getattr(self.config.training, "generation_validation_temperature", 1.0)),
                    top_p=float(getattr(self.config.training, "generation_validation_top_p", 0.95)),
                    top_k=int(getattr(self.config.training, "generation_validation_top_k", 100)),
                    tokenizer=self.tokenizer,
                    bad_words_ids=bad_words_ids,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    suppress_tokens=suppress_tokens,
                )

            decoded_sequences = generated.get("decoded_sequences", [])
            cleaned_sequences = [extract_smiles_from_decoded_sequence(seq) for seq in decoded_sequences]
            update_generation_metrics(metrics, decoded_sequences, cleaned_sequences)
            all_cleaned_sequences.extend(cleaned_sequences)
            all_generated_rows.extend(generated["generated_ids"].detach().cpu().tolist())

            processed += len(cleaned_sequences)
            if processed >= num_samples:
                break

        behavior = summarize_generation_behavior(
            metrics=metrics,
            cleaned_sequences=all_cleaned_sequences,
            generated_ids=all_generated_rows,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        summary = behavior["metrics"]
        self.recorder.info(
            "Generation validation | "
            f"samples={summary['total_sequences']} | "
            f"valid_smiles_rate={summary['valid_smiles_rate']:.4f} | "
            f"forbidden_domain_token_rate={summary['forbidden_domain_token_rate']:.4f} | "
            f"mol_closure_rate={summary['mol_closure_rate']:.4f} | "
            f"overlong_smiles_rate={summary['overlong_smiles_rate']:.4f} | "
            f"repeat_like_rate={summary['repeat_like_rate']:.4f} | "
            f"exclamation_degeneracy_rate={summary['exclamation_degeneracy_rate']:.4f} | "
            f"no_eos_rate={behavior['no_eos_rate']:.4f} | "
            f"repeat_near_max_ratio={behavior['repeat_near_max_ratio']:.4f}"
        )
        return {
            "gen_val_valid_smiles_rate": float(summary["valid_smiles_rate"]),
            "gen_val_forbidden_domain_token_rate": float(summary["forbidden_domain_token_rate"]),
            "gen_val_mol_closure_rate": float(summary["mol_closure_rate"]),
            "gen_val_overlong_smiles_rate": float(summary["overlong_smiles_rate"]),
            "gen_val_repeat_like_rate": float(summary["repeat_like_rate"]),
            "gen_val_exclamation_degeneracy_rate": float(summary["exclamation_degeneracy_rate"]),
            "gen_val_no_eos_rate": float(behavior["no_eos_rate"]),
            "gen_val_repeat_near_max_ratio": float(behavior["repeat_near_max_ratio"]),
            "gen_val_avg_smiles_length": float(behavior["avg_smiles_length"]),
            "gen_val_avg_eos_position": float(behavior["avg_eos_position"]),
            "gen_val_unique_rate": float(behavior["unique_rate"]),
        }
    
