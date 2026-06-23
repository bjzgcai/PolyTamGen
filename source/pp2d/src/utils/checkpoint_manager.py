import os
import torch
import json
import shutil
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

# Import helper to ensure only main process writes files
from .distributed_utils import is_main_process

logger = logging.getLogger(__name__)

@dataclass
class CheckpointMeta:
    """Metadata for a saved checkpoint."""
    epoch: int
    step: int
    metric_value: float
    path: str
    timestamp: str

class CheckpointManager:
    """
    Manages the saving and loading of model checkpoints with Top-K tracking.
    
    Features:
    - Atomic Saving: Writes to .tmp first, then renames, preventing corruption.
    - Top-K Tracking: Keeps only the best K checkpoints based on a metric.
    - Metadata Persistence: Saves a JSON file to track best checkpoints across restarts.
    - Resume Support: Easily load 'latest' or 'best' checkpoint.
    """
    
    def __init__(self, 
                 checkpoint_dir: str, 
                 keep_top_k: int = 3, 
                 metric_name: str = 'val_loss', 
                 lower_is_better: bool = True,
                 save_weights_only: bool = False,
                 save_weights_dtype: str = "bf16",
                 save_optimizer_last: bool = False):
        """
        Args:
            checkpoint_dir (str): Directory to store checkpoints.
            keep_top_k (int): Number of best checkpoints to keep.
            metric_name (str): The metric used to judge "best" (e.g. 'val_loss').
            lower_is_better (bool): True if lower metric is better (loss), False for accuracy.
        """
        self.ckpt_dir = checkpoint_dir
        self.keep_top_k = keep_top_k
        self.metric_name = metric_name
        self.lower_is_better = lower_is_better
        self.best_records: List[CheckpointMeta] = []
        self.save_weights_only = save_weights_only
        self.save_weights_dtype = str(save_weights_dtype).lower()
        self.save_optimizer_last = save_optimizer_last
        
        # Only Rank 0 should create directories and load writeable metadata
        if is_main_process():
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._load_meta()

    def _load_meta(self) -> None:
        """Loads the best_checkpoints.json metadata file if it exists."""
        meta_path = os.path.join(self.ckpt_dir, "best_checkpoints.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                    self.best_records = [CheckpointMeta(**r) for r in data]
                # Re-sort to be safe
                self._sort_records()
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}. Starting with empty records.")

    def _save_meta(self) -> None:
        """Persists the Top-K records to JSON."""
        if not is_main_process(): return
        
        meta_path = os.path.join(self.ckpt_dir, "best_checkpoints.json")
        try:
            with open(meta_path, 'w') as f:
                json.dump([asdict(r) for r in self.best_records], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")

    def _sort_records(self) -> None:
        """Sorts records based on the metric criteria."""
        self.best_records.sort(
            key=lambda x: x.metric_value, 
            reverse=not self.lower_is_better
        )

    def save(self, 
             model: torch.nn.Module, 
             optimizer: torch.optim.Optimizer, 
             scheduler: Optional[Any], 
             epoch: int, 
             step: int, 
             metric_value: float, 
             config: Any = None) -> None:
        """
        Saves the current state.
        
        1. Always updates 'checkpoint_latest.pth' (Atomic overwrite).
        2. Checks if current state qualifies for Top-K. If so, creates a named copy.
        
        Args:
            model: The model (will handle DDP unwrapping automatically).
            optimizer: The optimizer.
            scheduler: The learning rate scheduler (optional).
            epoch: Current epoch.
            step: Current global step.
            metric_value: The value of the monitored metric.
            config: Optional config to save with the checkpoint.
        """
        if not is_main_process(): 
            return

        # 1. Prepare State Dict
        # Unwrap DDP model if necessary
        raw_model = model.module if hasattr(model, 'module') else model
        
        if self.save_weights_only:
            def _convert_tensor(t: torch.Tensor) -> torch.Tensor:
                if not torch.is_tensor(t):
                    return t
                if t.is_floating_point():
                    dtype = torch.bfloat16 if self.save_weights_dtype in ("bf16", "bfloat16") else torch.float16
                    return t.detach().to(dtype=dtype, device="cpu")
                return t.detach().to(device="cpu")

            model_state = {k: _convert_tensor(v) for k, v in raw_model.state_dict().items()}
            state_dict = {
                'model_state_dict': model_state,
                'epoch': epoch,
                'step': step,
                'metric_value': metric_value,
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'weights_only': True,
                'weights_dtype': self.save_weights_dtype
            }
        else:
            state_dict = {
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'step': step,
                'metric_value': metric_value,
                'timestamp': datetime.now().isoformat(),
                'config': config
            }
        
        if scheduler is not None and not self.save_weights_only:
            state_dict['scheduler_state_dict'] = scheduler.state_dict()

        # 2. Atomic Save for Latest Checkpoint
        latest_name = "checkpoint_latest.pth"
        latest_path = os.path.join(self.ckpt_dir, latest_name)
        temp_path = latest_path + ".tmp"
        
        try:
            torch.save(state_dict, temp_path)
            # Atomic replacement: this ensures we never have a half-written file at latest_path
            os.replace(temp_path, latest_path) 
        except Exception as e:
            logger.error(f"Failed to save latest checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return

        # 2.5 Optional optimizer-only latest checkpoint
        if self.save_optimizer_last:
            opt_latest_name = "checkpoint_optimizer_latest.pth"
            opt_latest_path = os.path.join(self.ckpt_dir, opt_latest_name)
            opt_temp_path = opt_latest_path + ".tmp"
            try:
                opt_state = {
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'timestamp': datetime.now().isoformat()
                }
                torch.save(opt_state, opt_temp_path)
                os.replace(opt_temp_path, opt_latest_path)
            except Exception as e:
                logger.error(f"Failed to save optimizer latest checkpoint: {e}")
                if os.path.exists(opt_temp_path):
                    os.remove(opt_temp_path)

        # 3. Handle Top-K Logic
        if self._is_better_than_worst(metric_value):
            # Construct a unique filename for this specific checkpoint
            filename = f"checkpoint_ep{epoch}_step{step}_{self.metric_name}{metric_value:.4f}.pth"
            save_path = os.path.join(self.ckpt_dir, filename)
            
            # Copy from latest to avoid re-serializing (I/O optimization)
            try:
                shutil.copyfile(latest_path, save_path)
                
                # Update records
                new_record = CheckpointMeta(
                    epoch=epoch,
                    step=step,
                    metric_value=metric_value,
                    path=save_path,
                    timestamp=datetime.now().isoformat()
                )
                self.best_records.append(new_record)
                self._sort_records()
                
                # Remove excess checkpoints (pruning)
                while len(self.best_records) > self.keep_top_k:
                    to_remove = self.best_records.pop() # Removes the last (worst) one due to sort
                    if os.path.exists(to_remove.path):
                        try:
                            os.remove(to_remove.path)
                            logger.info(f"Pruned old checkpoint: {to_remove.path}")
                        except OSError as e:
                            logger.warning(f"Could not remove {to_remove.path}: {e}")
                
                self._save_meta()
                logger.info(f"Saved new best checkpoint: {filename} ({self.metric_name}={metric_value:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to copy/process best checkpoint: {e}")

    def _is_better_than_worst(self, val: float) -> bool:
        """Checks if the value qualifies to enter the Top-K list."""
        if len(self.best_records) < self.keep_top_k:
            return True
        
        # The list is sorted best -> worst. The last element is the worst.
        worst_val = self.best_records[-1].metric_value
        
        if self.lower_is_better:
            return val < worst_val
        else:
            return val > worst_val

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Loads the 'checkpoint_latest.pth' file."""
        path = os.path.join(self.ckpt_dir, "checkpoint_latest.pth")
        if not os.path.exists(path):
            return None
        
        logger.info(f"Loading latest checkpoint from {path}")
        try:
            # Map location to cpu to avoid GPU memory spikes during load
            return torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load latest checkpoint: {e}")
            return None

    def load_latest_optimizer(self) -> Optional[Dict[str, Any]]:
        """Loads the 'checkpoint_optimizer_latest.pth' file if present."""
        path = os.path.join(self.ckpt_dir, "checkpoint_optimizer_latest.pth")
        if not os.path.exists(path):
            return None

        logger.info(f"Loading latest optimizer checkpoint from {path}")
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load latest optimizer checkpoint: {e}")
            return None

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Loads the absolute best checkpoint available."""
        if not self.best_records:
            return None
        
        # Index 0 is the best due to sorting
        best_record = self.best_records[0]
        path = best_record.path
        
        if not os.path.exists(path):
            logger.warning(f"Best checkpoint record exists but file not found at {path}")
            return None
            
        logger.info(f"Loading best checkpoint from {path} ({self.metric_name}={best_record.metric_value})")
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load best checkpoint: {e}")
            return None
        