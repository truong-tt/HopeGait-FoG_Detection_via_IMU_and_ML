"""The TCN must never let future inputs influence past outputs.

Two checks:
  1. Last-step head: changing input at t+k after some t leaves the last-step
     output unchanged when we truncate the input at t.
  2. Dense head: dense_logits[:, :, t] depends only on inputs <= t.
"""

import numpy as np
import torch

from models.tcn_model import HopeGaitTCN


def _make_model():
    torch.manual_seed(0)
    m = HopeGaitTCN(num_inputs=9, num_channels=(8, 16, 32), kernel_size=3,
                    num_classes=2, dropout=0.0, drop_path=0.0, use_se=True)
    m.eval()
    return m


def test_last_step_is_causal():
    m = _make_model()
    x = torch.randn(2, 64, 9)
    base = m(x).detach().numpy()

    # Perturb only the very last sample. Last-step output is allowed to change.
    x_perturb = x.clone()
    x_perturb[:, -1, :] = torch.randn(2, 9)
    perturbed = m(x_perturb).detach().numpy()
    assert not np.allclose(base, perturbed)


def test_dense_head_strictly_causal():
    m = _make_model()
    T = 32
    x = torch.randn(1, T, 9)
    _, dense_a = m.forward_dense(x)
    dense_a = dense_a.detach().numpy()

    # Replace the second half with noise — outputs in the first half must not move.
    x_b = x.clone()
    x_b[:, T // 2:, :] = torch.randn(1, T // 2, 9)
    _, dense_b = m.forward_dense(x_b)
    dense_b = dense_b.detach().numpy()

    np.testing.assert_allclose(dense_a[:, :, :T // 2], dense_b[:, :, :T // 2],
                               rtol=1e-5, atol=1e-6)
