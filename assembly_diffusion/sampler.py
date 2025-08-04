import random

from .mask import FeasibilityMask
from .policy import ReversePolicy
from .graph import MoleculeGraph

class Sampler:
    """Run reverse diffusion sampling using a policy and feasibility mask."""
    def __init__(self, policy: ReversePolicy, mask: FeasibilityMask):
        self.policy = policy
        self.masker = mask

    def sample(self, T: int, x_init: MoleculeGraph) -> MoleculeGraph:
        x = x_init.copy()
        for t in range(T, 0, -1):
            mask = self.masker.mask_edits(x)
            logits = self.policy.logits(x, t, mask)
            e = random.choice(list(mask.keys()))
            if e == 'STOP':
                break
            i, j, b = e
            x.bonds[i, j] = x.bonds[j, i] = b
        return x
