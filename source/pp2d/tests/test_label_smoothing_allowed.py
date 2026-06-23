import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.loss_utils import masked_label_smoothing_nll


def test_masked_label_smoothing_uses_only_allowed_tokens():
    probs = torch.tensor([[0.10, 0.20, 0.30, 0.40]], dtype=torch.float32)
    log_probs = probs.log()
    targets = torch.tensor([2], dtype=torch.long)
    allowed_mask = torch.tensor([False, True, True, False], dtype=torch.bool)
    epsilon = 0.2

    loss = masked_label_smoothing_nll(
        log_probs=log_probs,
        targets=targets,
        allowed_mask=allowed_mask,
        epsilon=epsilon,
    )

    expected_nll = -torch.log(torch.tensor(0.30))
    expected_smooth = -torch.log(torch.tensor(0.20))
    expected = (1.0 - epsilon) * expected_nll + epsilon * expected_smooth

    assert torch.isclose(loss, expected, atol=1e-6)


def test_masked_label_smoothing_falls_back_to_nll_when_no_allowed_tokens():
    probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    log_probs = probs.log()
    targets = torch.tensor([1], dtype=torch.long)
    allowed_mask = torch.tensor([False, False, False, False], dtype=torch.bool)

    loss = masked_label_smoothing_nll(
        log_probs=log_probs,
        targets=targets,
        allowed_mask=allowed_mask,
        epsilon=0.3,
    )

    expected = -torch.log(torch.tensor(0.25))
    assert torch.isclose(loss, expected, atol=1e-6)
