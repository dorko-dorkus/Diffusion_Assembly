import torch
from torch.distributions import Categorical

from .mask import FeasibilityMask
from .policy import ReversePolicy
from .graph import MoleculeGraph

class Sampler:
    """Run reverse diffusion sampling using a policy and feasibility mask."""
    def __init__(self, policy: ReversePolicy, mask: FeasibilityMask):
        self.policy = policy
        self.masker = mask

    def sample(self, T: int, x_init: MoleculeGraph, guidance=None) -> MoleculeGraph:
        """Generate a single sample by running ``T`` reverse steps.

        Parameters
        ----------
        T:
            Number of reverse diffusion steps.
        x_init:
            Starting molecular graph at time ``T``.
        guidance: callable, optional
            Function modifying the logits before sampling. It must accept
            ``(logits, x, t, mask)`` and return adjusted logits.
        """

        x = x_init.copy()
        for t in range(T, 0, -1):
            mask = self.masker.mask_edits(x)
            logits = self.policy.logits(x, t, mask)
            if guidance is not None:
                logits = guidance(logits, x, t, mask)
            probs = torch.softmax(logits, dim=0)
            dist = Categorical(probs)
            idx = dist.sample().item()
            action = self.policy._actions[idx]
            if action == "STOP":
                break
            i, j, b = action
            x.bonds[i, j] = x.bonds[j, i] = b
        return x

    def trajectory(self, T: int, x_init: MoleculeGraph, guidance=None):
        """Return the sequence of intermediate graphs for diagnostics.

        The returned list starts from ``x_init`` (``G_T``) and includes each
        intermediate prediction down to the final ``Ĝ_0``.
        """

        x = x_init.copy()
        traj = [x.copy()]
        for t in range(T, 0, -1):
            mask = self.masker.mask_edits(x)
            logits = self.policy.logits(x, t, mask)
            if guidance is not None:
                logits = guidance(logits, x, t, mask)
            probs = torch.softmax(logits, dim=0)
            dist = Categorical(probs)
            idx = dist.sample().item()
            action = self.policy._actions[idx]
            if action == "STOP":
                break
            i, j, b = action
            x.bonds[i, j] = x.bonds[j, i] = b
            traj.append(x.copy())
        return traj
