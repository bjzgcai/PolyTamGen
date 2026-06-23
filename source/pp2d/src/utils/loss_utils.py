from typing import Optional

import torch


def masked_label_smoothing_nll(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    allowed_mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Label smoothing with vocabulary masking.

    All probability mass outside `allowed_mask` is treated as zero. The smoothing
    mass is distributed uniformly over allowed tokens except the gold token.
    """
    if log_probs.dim() != 2:
        raise ValueError(f"Expected 2D log_probs [N, V], got shape {tuple(log_probs.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"Expected 1D targets [N], got shape {tuple(targets.shape)}")
    if log_probs.size(0) != targets.size(0):
        raise ValueError("Batch size mismatch between log_probs and targets")
    if allowed_mask.dim() != 1 or allowed_mask.size(0) != log_probs.size(1):
        raise ValueError(
            "allowed_mask must be 1D with length equal to vocabulary size "
            f"(got {tuple(allowed_mask.shape)} vs vocab={log_probs.size(1)})"
        )

    epsilon = float(max(0.0, min(1.0, epsilon)))

    nll = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    if epsilon <= 0.0:
        return nll.mean()

    allowed_count = int(allowed_mask.long().sum().item())
    if allowed_count <= 0:
        return nll.mean()

    target_in_allowed = allowed_mask[targets]
    denom = allowed_count - target_in_allowed.long()

    allowed_neg_log_sum = (-log_probs[:, allowed_mask]).sum(dim=1)
    smooth_numer = allowed_neg_log_sum - torch.where(target_in_allowed, nll, torch.zeros_like(nll))

    smooth_loss = torch.where(
        denom > 0,
        smooth_numer / denom.to(log_probs.dtype),
        torch.zeros_like(nll),
    )

    total = (1.0 - epsilon) * nll + epsilon * smooth_loss
    return total.mean()
