"""Polyak / exponential moving average of model weights.

Cheap regularizer + smoother validation curve. The shadow model is what we
evaluate on val and what we save to disk — the live model is the one
optimized by SGD.
"""

import copy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv.detach(), alpha=1.0 - d)
            else:
                # Buffers like num_batches_tracked / int counters: just copy.
                v.copy_(mv)

    def state_dict(self):
        return self.shadow.state_dict()
