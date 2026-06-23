import torch
import logging
import math
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    _LRScheduler
)
from typing import Optional, Union, Any

logger = logging.getLogger(__name__)


class WarmupCosineLR(_LRScheduler):
    """Linear warmup followed by cosine decay, stepped per optimizer step."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1

        if self.warmup_steps > 0 and current_step <= self.warmup_steps:
            warmup_scale = current_step / self.warmup_steps
            return [base_lr * warmup_scale for base_lr in self.base_lrs]

        if self.total_steps <= self.warmup_steps:
            return list(self.base_lrs)

        progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + (base_lr - self.eta_min) * cosine_decay
            lrs.append(lr)
        return lrs

def create_scheduler(optimizer: torch.optim.Optimizer, 
                     lr_config: Any, 
                     total_epochs: int, 
                     base_lr: float, 
                     steps_per_epoch: int) -> Optional[Union[_LRScheduler, ReduceLROnPlateau]]:
    """
    Factory function to create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to wrap.
        lr_config: Configuration object containing scheduler params (type, warmup_steps, etc.).
        total_epochs: Total number of training epochs.
        base_lr: The initial learning rate.
        steps_per_epoch: Number of batches per epoch (used for step-based schedulers).
        
    Returns:
        The initialized scheduler, or None if use_scheduler is False.
        
    Raises:
        ValueError: If an unsupported scheduler type is provided.
    """
    if not getattr(lr_config, 'use_scheduler', True):
        logger.info("Learning rate scheduler is disabled in config.")
        return None

    # Ensure base_lr is numeric (YAML may load scientific notation as str)
    try:
        base_lr = float(base_lr)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid base_lr: {base_lr}")

    stype = lr_config.scheduler_type.lower()
    scheduler = None
    
    logger.info(f"Initializing scheduler: {stype}")

    if stype == "cosine":
        # Step-based cosine decay. If warmup_steps > 0, use linear warmup then cosine.
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = int(getattr(lr_config, 'warmup_steps', 0) or 0)
        eta_min = base_lr * getattr(lr_config, 'min_lr_ratio', 0.01)

        if warmup_steps > 0:
            scheduler = WarmupCosineLR(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                eta_min=eta_min,
            )
            logger.info(
                f"  total_steps: {total_steps}, warmup_steps: {warmup_steps}, eta_min: {eta_min}"
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
            logger.info(f"  T_max: {total_steps}, eta_min: {eta_min}")

    elif stype == "warmup_cosine":
        # Cosine with Warm Restarts
        t_0 = getattr(lr_config, 'warmup_steps', steps_per_epoch * 5) # Default 5 epochs warmup cycle
        eta_min = base_lr * getattr(lr_config, 'min_lr_ratio', 0.01)
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, eta_min=eta_min)
        logger.info(f"  T_0 (Restart Cycle): {t_0}, eta_min: {eta_min}")

    elif stype == "step":
        # Step Decay
        step_size = getattr(lr_config, 'step_size', 10) # Decay every N epochs
        gamma = getattr(lr_config, 'gamma', 0.1)
        
        # Note: StepLR usually works on epochs. If step_scheduler is called per batch,
        # ensure step_size is scaled or step_scheduler is called per epoch.
        # Here we assume standard StepLR usage (usually per epoch).
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"  Step Size: {step_size}, Gamma: {gamma}")

    elif stype == "exponential":
        gamma = getattr(lr_config, 'exponential_gamma', 0.95)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        logger.info(f"  Gamma: {gamma}")

    elif stype == "plateau":
        # Reduce on Plateau (Dynamic)
        factor = getattr(lr_config, 'factor', 0.1)
        patience = getattr(lr_config, 'patience', 10)
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=factor, 
            patience=patience, 
            verbose=True
        )
        logger.info(f"  Factor: {factor}, Patience: {patience}, Mode: min")

    else:
        raise ValueError(f"Unsupported scheduler type: {stype}")

    return scheduler

def step_scheduler(scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]], 
                   metric_value: Optional[float] = None) -> None:
    """
    Universal stepper for schedulers.
    
    Handles the difference between standard schedulers (step()) and 
    ReduceLROnPlateau (step(metrics)).
    
    Args:
        scheduler: The scheduler instance.
        metric_value: The validation metric (e.g., val loss). Required for ReduceLROnPlateau.
    """
    if scheduler is None:
        return

    if isinstance(scheduler, ReduceLROnPlateau):
        if metric_value is None:
            # Plateau scheduler needs a metric, but sometimes we might step blindly
            # (though incorrect usage). We log a warning if this happens.
            # However, typically this is called at validation end.
            pass 
        else:
            scheduler.step(metric_value)
    else:
        # Standard schedulers (Cosine, Step, etc.)
        scheduler.step()

def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Helper to retrieve the current learning rate from the optimizer.
    Assumes all param groups have the same LR (returns the first one).
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0
