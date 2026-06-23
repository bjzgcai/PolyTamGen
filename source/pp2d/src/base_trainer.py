import os
import sys
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Dict, Any, Optional, Union, Iterator
from tqdm import tqdm
from dataclasses import asdict, is_dataclass
from torch.utils.data import Sampler, DataLoader

from .utils.training_recorder import TrainingRecorder
from .utils.checkpoint_manager import CheckpointManager
from .utils.distributed_utils import (
    is_main_process, 
    get_rank, 
    get_world_size, 
    reduce_tensor,
    ddp_barrier,
)
from .utils.scheduler_utils import step_scheduler, get_current_lr
from .utils.misc import set_seed
from .utils.generation_utils import compute_generation_composite_score

class SkipSampler(Sampler):
    """
    Sampler designed for Fast Resume.
    It skips the first `skip_count` indices of the base sampler effectively.
    """
    def __init__(self, sampler: Sampler, skip_count: int):
        self.sampler = sampler
        self.skip_count = skip_count

    def __iter__(self) -> Iterator:
        iterator = iter(self.sampler)
        for _ in range(self.skip_count):
            try:
                next(iterator)
            except StopIteration:
                break
        return iterator

    def __len__(self):
        return max(0, len(self.sampler) - self.skip_count)


class BaseTrainer(ABC):
    """
    Abstract Base Trainer designed for robust Distributed Data Parallel (DDP) training.
    Features:
    - Fast Resume (skipping batches without reloading dataset)
    - Intra-epoch Validation & Saving (DDP-safe)
    - Gradient Accumulation
    - NaN Loss handling
    """

    def __init__(self, config: Any, recorder: TrainingRecorder):
        self.config = config
        self.recorder = recorder
        
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        # Mixed precision settings
        self.use_mixed_precision = getattr(config.training, "use_mixed_precision", False)
        self.mixed_precision_dtype = str(getattr(config.training, "mixed_precision_dtype", "fp16")).lower()
        self.amp_enabled = self.use_mixed_precision and self.device.type == "cuda"
        if self.mixed_precision_dtype in ("bf16", "bfloat16"):
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(
            scaler_device,
            enabled=self.amp_enabled and self.amp_dtype == torch.float16,
        )
        if self.is_main:
            self.recorder.info(
                f"AMP enabled: {self.amp_enabled} | dtype: {self.amp_dtype} | scaler: {self.scaler.is_enabled()}"
            )
            
        self.start_epoch = 0
        self.global_step = 0
        self.resume_batch_idx = 0 
        
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        
        # Persistent DataLoaders initialized in _setup_data
        self.train_loader: Optional[DataLoader] = None
        self.valid_loader: Optional[DataLoader] = None
        
        # Samplers needed for set_epoch in DDP
        self.train_sampler: Optional[Any] = None
        self.valid_sampler: Optional[Any] = None
        
        # Checkpoint Manager tracks the best models based on Validation Loss
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_top_k=getattr(config.training, 'keep_top_k', 3),
            metric_name='val_loss',
            lower_is_better=True,
            save_weights_only=getattr(config.training, 'save_weights_only', False),
            save_weights_dtype=getattr(config.training, 'save_weights_dtype', 'bf16'),
            save_optimizer_last=getattr(config.training, 'save_optimizer_last', False)
        )

        self.gen_ckpt_manager: Optional[CheckpointManager] = None
        if bool(getattr(config.training, 'generation_best_checkpoint_enabled', False)):
            gen_ckpt_dir = os.path.join(config.checkpoint_dir, 'best_generation')
            self.gen_ckpt_manager = CheckpointManager(
                checkpoint_dir=gen_ckpt_dir,
                keep_top_k=int(getattr(config.training, 'generation_best_keep_top_k', 1)),
                metric_name='gen_val_composite_score',
                lower_is_better=False,
                save_weights_only=getattr(config.training, 'save_weights_only', False),
                save_weights_dtype=getattr(config.training, 'save_weights_dtype', 'bf16'),
                save_optimizer_last=False,
            )

        seed = getattr(config.training, 'seed', 42) + self.rank
        set_seed(seed)
        self.recorder.info(f"Initialized Trainer on Rank {self.rank}")

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.amp.autocast(self.device.type, enabled=True, dtype=self.amp_dtype)

    def _sync_boolean_flag(self, flag: bool) -> tuple[bool, bool]:
        """
        Synchronize a boolean condition across ranks.

        Returns:
            Tuple[bool, bool]:
            - all_ranks_true
            - any_rank_true
        """
        if not dist.is_available() or not dist.is_initialized():
            return flag, flag

        flag_tensor = torch.tensor([1 if flag else 0], device=self.device, dtype=torch.int32)
        min_tensor = flag_tensor.clone()
        max_tensor = flag_tensor.clone()
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        return bool(min_tensor.item()), bool(max_tensor.item())

    def setup(self):
        """Setup components in correct order."""
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

        resume_enabled = bool(getattr(self.config.training, 'resume', False))
        if resume_enabled:
            self.recorder.info("Resume mode enabled: attempting to load latest checkpoint.")
            self._resume_from_checkpoint()
        else:
            self.recorder.info("Resume mode disabled: training starts from scratch.")
            
        ddp_barrier()

    @abstractmethod
    def _setup_model(self):
        """
        Initialize the neural network model.
        
        Responsibilities:
        1. Construct the model architecture.
        2. Move the model to the correct device (self.device).
        3. [CRITICAL] If using DDP, wrap the model with `DistributedDataParallel`.
        
        Must set:
            self.model (torch.nn.Module)
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_optimizer(self):
        """
        Initialize the optimizer and learning rate scheduler.
        
        Responsibilities:
        1. Create an optimizer (e.g., AdamW) for `self.model.parameters()`.
        2. Create a scheduler (e.g., CosineAnnealingLR) if needed.
        
        Must set:
            self.optimizer (torch.optim.Optimizer)
            self.scheduler (Optional[object])
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_data(self):
        """
        Initialize DataLoaders and Samplers for DDP training.
        
        Responsibilities:
        1. Create datasets for training and validation.
        2. Create a `DistributedSampler` for the training set (Required for DDP).
        3. Create DataLoaders using these samplers.
        
        Must set:
            self.train_loader (DataLoader)
            self.valid_loader (DataLoader)
            self.train_sampler  (Sampler) -> Critical for `train_sampler.set_epoch(epoch)`
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Define the forward pass and loss calculation for a single training step.
        
        Args:
            batch: A single batch of data from `self.train_loader`.
            
        Returns:
            Dict[str, torch.Tensor]: Must contain the key "loss" for backpropagation.
                                     Can contain other metrics for logging.
                                     
        Note:
            Do NOT call `loss.backward()` or `optimizer.step()` here. 
            The `BaseTrainer` loop handles gradient accumulation, scaling, and updates.
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Define the forward pass and metric calculation for a single validation step.
        
        Args:
            batch: A single batch of data from `self.valid_loader`.
            
        Returns:
            Dict[str, torch.Tensor]: Must contain "loss" for the CheckpointManager 
                                     to track the best model.
        """
        raise NotImplementedError

    def should_run_generation_validation(self) -> bool:
        return False

    def run_generation_validation(self) -> Dict[str, float]:
        return {}

    def _build_generation_composite_score(self, gen_metrics: Dict[str, float], val_loss: float) -> Optional[float]:
        forbidden_rate = gen_metrics.get('gen_val_forbidden_domain_token_rate', None)
        valid_rate = gen_metrics.get('gen_val_valid_smiles_rate', None)
        if forbidden_rate is None or valid_rate is None:
            return None
        return compute_generation_composite_score(
            forbidden_domain_token_rate=float(forbidden_rate),
            valid_smiles_rate=float(valid_rate),
            val_loss=float(val_loss),
            mol_closure_rate=float(gen_metrics.get('gen_val_mol_closure_rate', 0.0)),
            overlong_smiles_rate=float(gen_metrics.get('gen_val_overlong_smiles_rate', 0.0)),
            repeat_like_rate=float(gen_metrics.get('gen_val_repeat_like_rate', 0.0)),
            exclamation_degeneracy_rate=float(gen_metrics.get('gen_val_exclamation_degeneracy_rate', 0.0)),
            no_eos_rate=float(gen_metrics.get('gen_val_no_eos_rate', 0.0)),
        )

    def _run_generation_validation_and_maybe_save(
        self,
        val_loss: float,
        batch_idx: int = 0,
        log_prefix: str = "Epoch",
    ) -> Dict[str, float]:
        if not (self.is_main and self.should_run_generation_validation()):
            return {}

        gen_metrics = self.run_generation_validation()
        gen_score = self._build_generation_composite_score(gen_metrics, val_loss)
        if gen_score is not None:
            gen_metrics['gen_val_composite_score'] = float(gen_score)

            if self.gen_ckpt_manager is not None:
                self.recorder.info(
                    f"{log_prefix}: Saving generation-best checkpoint at step {self.global_step} "
                    f"with gen_val_composite_score={gen_score:.4f}"
                )
                self.save_checkpoint(
                    metric_value=float(gen_score),
                    batch_idx=batch_idx,
                    manager=self.gen_ckpt_manager,
                )
        return gen_metrics

    def train(self):
        """Main training loop."""
        total_epochs = self.config.training.epochs
        self.recorder.info(f"Starting Training: Epochs {self.start_epoch} -> {total_epochs}")
        
        for epoch in range(self.start_epoch, total_epochs):
            self.current_epoch = epoch
            
            # Set epoch for DistributedSampler to ensure shuffling works across epochs
            if self.train_sampler is not None and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            
            # Determine if we need to skip batches (Intra-epoch resume)
            current_skip_batches = self.resume_batch_idx if epoch == self.start_epoch else 0
            
            if current_skip_batches > 0:
                self.recorder.info(f"Resuming Epoch {epoch} from Batch Index {current_skip_batches}...")

            # --- RUN TRAINING ---
            train_metrics = self.run_epoch(epoch, is_train=True, skip_batches=current_skip_batches)
            
            # Reset resume index after a successful epoch run
            self.resume_batch_idx = 0
            
            # --- RUN END-OF-EPOCH VALIDATION ---
            # Even if we do intra-epoch validation, we usually validate at the end as well.
            val_metrics = self.run_epoch(epoch, is_train=False)
            val_loss = val_metrics.get('val_loss', float('inf'))
            gen_metrics = {}

            if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                step_scheduler(self.scheduler, val_loss)

            gen_metrics = self._run_generation_validation_and_maybe_save(
                val_loss=val_loss,
                batch_idx=0,
                log_prefix=f"Epoch {epoch+1}",
            )
            
            # Logging & Saving
            if self.is_main:
                lr = get_current_lr(self.optimizer)
                self.recorder.info(
                    f"Epoch {epoch+1}/{total_epochs} | "
                    f"Train Loss: {train_metrics.get('train_loss', 0):.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )
                
                metrics = {**train_metrics, **val_metrics, **gen_metrics, 'epoch': epoch + 1}
                self.recorder.log_metrics(metrics, step=self.global_step, prefix="Epoch")
                
                # Save standard end-of-epoch checkpoint
                self.recorder.info(f"Saving Checkpoint at Epoch {epoch+1}, Step {self.global_step} with Val Loss {val_loss:.4f}...")
                self.save_checkpoint(metric_value=val_loss, batch_idx=0)
            
            ddp_barrier()

    def run_epoch(self, epoch: int, is_train: bool = True, skip_batches: int = 0) -> Dict[str, float]:
        """
        Runs a single epoch (Train or Validation) with DDP optimizations.
        
        Key Features:
        - **Fast Resume**: Efficiently skips batches if resuming mid-epoch using SkipSampler.
        - **DDP-Safe Intra-Epoch Validation**: Ensures all ranks participate in validation to prevent deadlocks.
        - **Dynamic Logging**: Automatically extracts and logs all scalar metrics from `train_step`.
        - **Robustness**: Handles NaN losses and manages TQDM visuals cleanly.
        
        Args:
            epoch (int): Current epoch number.
            is_train (bool): Whether to run in training mode.
            skip_batches (int): Number of batches to skip (for resuming).
            
        Returns:
            Dict[str, float]: Dictionary containing averaged loss (e.g., {'train_loss': 0.123}).
        """
        # 1. Select the correct DataLoader
        if is_train:
            base_loader = self.train_loader
            if base_loader is None: raise RuntimeError("train_loader not initialized in _setup_data")
        else:
            base_loader = self.valid_loader
            if base_loader is None: raise RuntimeError("valid_loader not initialized in _setup_data")

        # 2. Set Model Mode (Critical for Dropout/BatchNorm)
        self.model.train() if is_train else self.model.eval()
        
        # 3. Initialize Accumulators for Epoch-level averaging
        total_loss_tensor = torch.zeros(1, device=self.device)
        total_batches_tensor = torch.zeros(1, device=self.device)
        
        # 4. Load Configuration Shortcuts
        update_freq = max(1, int(getattr(self.config.training, 'update_freq', 1)))
        log_interval = getattr(self.config.training, 'log_interval', 100)
        save_int = getattr(self.config.training, 'save_interval_steps', 0)
        
        # 5. Fast Resume Logic
        # If resuming mid-epoch, wrap the sampler to skip data efficiently.
        if is_train and skip_batches > 0:
            if not self.is_main or self.rank == 0:
                self.recorder.info(f"Creating SkipSampler for {skip_batches} batches...")
            
            skip_items = skip_batches * base_loader.batch_size
            skip_sampler = SkipSampler(base_loader.sampler, skip_items)
            
            # Create a temporary loader with the SkipSampler
            iterator_loader = DataLoader(
                base_loader.dataset,
                batch_size=base_loader.batch_size,
                sampler=skip_sampler,
                num_workers=base_loader.num_workers,
                collate_fn=base_loader.collate_fn,
                pin_memory=base_loader.pin_memory,
                persistent_workers=False # Close workers after this partial run
            )
        else:
            iterator_loader = base_loader

        # 6. Progress Bar Setup
        iterator = iterator_loader
        if self.is_main:
            desc = f"Epoch {epoch+1} [{'Train' if is_train else 'Val'}]"
            visual_total = len(base_loader) 
            if is_train and skip_batches > 0:
                visual_total = len(iterator_loader)
            
            # 'leave=False' ensures validation bars don't clutter the terminal after completion
            iterator = tqdm(iterator_loader, desc=desc, total=visual_total, leave=False)

        # 7. Main Loop
        accumulation_count = 0
        skipped_empty_batches = 0
        for i, batch in enumerate(iterator):
            current_batch_idx = skip_batches + i + 1

            local_batch_valid = not (batch is None or (isinstance(batch, dict) and len(batch) == 0))
            all_ranks_have_batch, any_rank_has_batch = self._sync_boolean_flag(local_batch_valid)
            if not all_ranks_have_batch:
                skipped_empty_batches += 1
                if self.is_main and skipped_empty_batches <= 5:
                    scope = "all ranks" if not any_rank_has_batch else "at least one rank"
                    self.recorder.warning(
                        f"Skipping batch {current_batch_idx} because batch data was missing on {scope} "
                        f"(is_train={is_train})."
                    )
                if is_train and self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                continue

            batch = self._move_to_device(batch)
            
            # Enable/Disable Gradients based on mode
            with torch.set_grad_enabled(is_train):
                if is_train:
                    # ============================
                    #      TRAINING LOGIC
                    # ============================
                    with self._autocast_context():
                        outputs = self.train_step(batch)
                        loss = outputs.get('loss')
                    
                    # A. NaN Loss Handling (Stability)
                    local_loss_is_finite = bool(torch.isfinite(loss).all().item())
                    all_ranks_finite, _ = self._sync_boolean_flag(local_loss_is_finite)
                    if not all_ranks_finite:
                        if self.is_main:
                            self.recorder.warning(
                                f"Skipping batch {current_batch_idx} because at least one rank produced "
                                "a NaN/Inf loss. Running synchronized dummy backward to keep DDP aligned."
                            )
                        dummy_loss = sum(p.sum() for p in self.model.parameters() if p.requires_grad) * 0.0
                        if self.scaler.is_enabled():
                            self.scaler.scale(dummy_loss).backward()
                        else:
                            dummy_loss.backward()
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                        
                    # B. Backward Pass & Gradient Accumulation
                    loss_scaled = loss / update_freq
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                    accumulation_count += 1
                    
                    # C. Optimizer Step (Once every update_freq batches)
                    if current_batch_idx % update_freq == 0:
                        # Gradient Clipping
                        if getattr(self.config.training, 'grad_clip', 0.0) > 0:
                            if self.scaler.is_enabled():
                                self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)

                        if self.scaler.is_enabled():
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()
                        accumulation_count = 0
                        self.global_step += 1
                        
                        if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
                            step_scheduler(self.scheduler, None) 

                        # D. Training Logging (Step-level)
                        if self.is_main and self.global_step % log_interval == 0:
                            # Log Learning Rate
                            self.recorder.log_metric("Train/LR", get_current_lr(self.optimizer), self.global_step)
                            self.recorder.log_metrics(outputs, self.global_step, prefix="Train")
                            
                        # E. Intra-Epoch Validation & Saving
                        # This block must run on ALL ranks to prevent deadlocks.
                        if save_int > 0 and self.global_step > 0 and self.global_step % save_int == 0:
                            
                            # Pause the current training progress bar for visual clarity
                            if self.is_main and isinstance(iterator, tqdm):
                                iterator.clear() 
                                self.recorder.info(f"Step {self.global_step}: Triggering Intra-epoch Validation...")

                            # Recursively call run_epoch for validation (is_train=False)
                            val_metrics = self.run_epoch(epoch, is_train=False)
                            val_loss = val_metrics.get('val_loss', float('inf'))

                            # Save Logic (Main Rank Only)
                            if self.is_main:
                                self.recorder.info(f"Step {self.global_step} Validation Loss: {val_loss:.4f}")
                                
                                # Log the validation loss to the Recorder explicitly
                                self.recorder.log_metric("Val/Loss", val_loss, self.global_step)
                                self.recorder.log_metrics(val_metrics, self.global_step, prefix="Val")

                                gen_metrics = self._run_generation_validation_and_maybe_save(
                                    val_loss=val_loss,
                                    batch_idx=current_batch_idx,
                                    log_prefix=f"Step {self.global_step}",
                                )
                                if gen_metrics:
                                    self.recorder.log_metrics(gen_metrics, self.global_step, prefix="GenVal")
                                
                                # Save checkpoint (metric_value allows CheckpointManager to track "Best")
                                self.recorder.info(f"Saving Checkpoint at Epoch {epoch+1}, Step {self.global_step} with Val Loss {val_loss:.4f}...")
                                self.save_checkpoint(metric_value=val_loss, batch_idx=current_batch_idx)
                            
                            # Re-sync ranks
                            ddp_barrier()
                            
                            # [IMPORTANT]: Switch back to TRAIN mode after validation
                            self.model.train()
                            
                            # Resume training progress bar
                            if self.is_main and isinstance(iterator, tqdm): iterator.display()

                else:
                    # ============================
                    #     VALIDATION LOGIC
                    # ============================
                    with self._autocast_context():
                        outputs = self.validation_step(batch)
                        loss = outputs.get('loss')
                    # Note: We do NOT log step-level validation metrics to the recorder here 
                    # to avoid flooding the logs. We rely on TQDM for real-time feedback.

            # 8. Visual Updates (TQDM) - Updates for both Train and Validation
            if self.is_main and isinstance(iterator, tqdm):
                prog_bar_dict = {}
                # Dynamically extract all scalars for display
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        prog_bar_dict[k] = f"{v.item():.4f}"
                    elif isinstance(v, (float, int)):
                        prog_bar_dict[k] = f"{v:.4f}"
                
                # Update the progress bar postfix
                if prog_bar_dict:
                    iterator.set_postfix(prog_bar_dict)

            # 9. Accumulate Totals (for Epoch Average)
            if loss is not None:
                total_loss_tensor += loss.detach()
                total_batches_tensor += 1

        # 9.5 Flush remainder gradients at epoch end (when #batches % update_freq != 0)
        if is_train and accumulation_count > 0:
            if getattr(self.config.training, 'grad_clip', 0.0) > 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
                step_scheduler(self.scheduler, None) 

        if skipped_empty_batches > 0 and self.is_main:
            self.recorder.warning(
                f"Epoch {epoch+1} {'Train' if is_train else 'Val'} skipped {skipped_empty_batches} empty batches."
            )

        # 10. End of Epoch Aggregation
        # Combine loss across all batches and all DDP ranks
        loss_tensor = torch.stack([total_loss_tensor, total_batches_tensor]).view(-1)
        reduced = reduce_tensor(loss_tensor, average=False)
        reduced_cpu = reduced.cpu()
        
        # Calculate average
        avg_loss = reduced_cpu[0].item() / max(reduced_cpu[1].item(), 1.0)
        
        prefix = "train" if is_train else "val"
        return {f"{prefix}_loss": avg_loss}

    def _resume_from_checkpoint(self):
        """Resumes model, optimizer, scheduler, and RNG states."""
        checkpoint = self.ckpt_manager.load_latest()
        if not checkpoint:
            self.recorder.info("No checkpoint found.")
            return

        self.recorder.info(f"Loading Checkpoint (Epoch {checkpoint['epoch']})...")
        
        state_dict = checkpoint['model_state_dict']
        is_ckpt_ddp = any(k.startswith('module.') for k in state_dict.keys())
        is_model_ddp = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        
        # Handle DDP vs non-DDP state dict mismatch
        if is_ckpt_ddp and not is_model_ddp:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not is_ckpt_ddp and is_model_ddp:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif self.optimizer and getattr(self.config.training, 'save_optimizer_last', False):
            opt_ckpt = self.ckpt_manager.load_latest_optimizer()
            if opt_ckpt and 'optimizer_state_dict' in opt_ckpt:
                self.optimizer.load_state_dict(opt_ckpt['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.global_step = checkpoint.get('step', 0)
        saved_epoch = checkpoint.get('epoch', 0)
        
        saved_config = checkpoint.get('config', {}) or {}
        resume_state = saved_config.get('resume_state', {})
        self.resume_batch_idx = resume_state.get('batch_idx', 0)
        
        rng_states = resume_state.get('rng_states', {})
        if rng_states:
            try:
                if 'python' in rng_states: random.setstate(rng_states['python'])
                if 'numpy' in rng_states: np.random.set_state(rng_states['numpy'])
                if 'torch_cpu' in rng_states: torch.set_rng_state(rng_states['torch_cpu'])
                if 'torch_cuda' in rng_states and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng_states['torch_cuda'])
                self.recorder.info("RNG states restored.")
            except Exception as e:
                self.recorder.warning(f"Failed to restore RNG states from checkpoint: {e}")

        if self.resume_batch_idx > 0:
            self.start_epoch = saved_epoch
            self.recorder.info(f"Resuming Intra-Epoch: {self.start_epoch}, Batch {self.resume_batch_idx}")
        else:
            self.start_epoch = saved_epoch + 1
            self.recorder.info(f"Resuming Start of Epoch: {self.start_epoch}")

    def save_checkpoint(self, metric_value: float, batch_idx: int = 0, manager: Optional[CheckpointManager] = None):
        """Prepares state and saves using CheckpointManager."""
        if is_dataclass(self.config): config_dict = asdict(self.config)
        else: config_dict = vars(self.config) if hasattr(self.config, '__dict__') else {}

        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }

        target_manager = manager if manager is not None else self.ckpt_manager

        target_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            metric_value=metric_value,
            config={'static_config': config_dict, 'resume_state': {'batch_idx': batch_idx, 'rng_states': rng_states}}
        )

    def _move_to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor): return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict): return {k: self._move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list): return [self._move_to_device(v) for v in batch]
        return batch
    
