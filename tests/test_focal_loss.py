"""Focal loss accepts both 2D last-step and 3D dense logits."""

import torch

from models.focal_loss import FocalLoss


def test_focal_loss_2d():
    crit = FocalLoss(alpha=torch.tensor([0.2, 0.8]), gamma=2.0)
    logits = torch.tensor([[2.0, -2.0], [-1.0, 1.0]])
    targets = torch.tensor([0, 1])
    loss = crit(logits, targets)
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_focal_loss_3d_matches_2d_when_T_is_1():
    crit = FocalLoss(alpha=torch.tensor([0.2, 0.8]), gamma=2.0)
    logits_2d = torch.tensor([[2.0, -2.0], [-1.0, 1.0]])
    targets_1d = torch.tensor([0, 1])
    loss_2d = crit(logits_2d, targets_1d).item()

    logits_3d = logits_2d.unsqueeze(-1)  # (B, C, 1)
    targets_2d = targets_1d.unsqueeze(-1)  # (B, 1)
    loss_3d = crit(logits_3d, targets_2d).item()

    assert abs(loss_2d - loss_3d) < 1e-6
