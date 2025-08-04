import random

from .forward import ForwardKernel
from .policy import ReversePolicy
from .mask import FeasibilityMask


def train_epoch(loader, kernel: ForwardKernel, policy: ReversePolicy, mask: FeasibilityMask):
    """Example training loop (placeholder)."""
    for x0 in loader:
        t = random.randint(1, kernel.T)
        xt = kernel.sample_xt(x0, t)
        mask_ = mask.mask_edits(xt)
        logits = policy.logits(xt, t, mask_)
        loss = logits.sum() * 0  # Placeholder to keep graph
        loss.backward()
