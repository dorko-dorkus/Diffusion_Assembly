import torch
from torch.distributions import Categorical

from .mask import FeasibilityMask
from .policy import ReversePolicy
from .graph import MoleculeGraph
from .guidance import reweight

class Sampler:
    """Run reverse diffusion sampling using a policy and feasibility mask."""
    def __init__(self, policy: ReversePolicy, mask: FeasibilityMask):
        self.policy = policy
        self.masker = mask

    def sample(
        self,
        T: int,
        x_init: MoleculeGraph,
        guidance=None,
        gamma: float = 0.0,
        clip_range: tuple[float, float] = (-float("inf"), float("inf")),
    ) -> MoleculeGraph:
        """Generate a single sample by running ``T`` reverse steps.

        Parameters
        ----------
        T:
            Number of reverse diffusion steps.
        x_init:
            Starting molecular graph at time ``T``.
        guidance: callable, optional
            Function producing guidance scores ``ΔS`` for candidate edits. It
            must accept ``(logits, x, t, mask)`` and return a tensor aligned
            with the non-STOP logits.
        gamma: float, optional
            Scale factor for the guidance scores.
        clip_range: tuple of float, optional
            Range ``(a, b)`` used to clip ``ΔS`` before scaling.
        """

        x = x_init.copy()
        for t in range(T, 0, -1):
            mask = self.masker.mask_edits(x)
            logits = self.policy.logits(x, t, mask)
            if guidance is not None:
                delta = guidance(logits, x, t, mask)
                logits = reweight(logits, x, delta, gamma, clip_range)
            probs = torch.softmax(logits, dim=0)
            dist = Categorical(probs)
            idx = dist.sample().item()
            action = self.policy._actions[idx]
            if action == "STOP":
                break
            i, j, b = action
            x.bonds[i, j] = x.bonds[j, i] = b
        return x

    def trajectory(
        self,
        T: int,
        x_init: MoleculeGraph,
        guidance=None,
        gamma: float = 0.0,
        clip_range: tuple[float, float] = (-float("inf"), float("inf")),
    ):
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
                delta = guidance(logits, x, t, mask)
                logits = reweight(logits, x, delta, gamma, clip_range)
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
