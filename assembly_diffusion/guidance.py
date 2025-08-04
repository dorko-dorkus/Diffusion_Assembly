import torch
from torch import Tensor
from typing import Tuple

from .assembly_index import approx_AI


def reweight(logits: Tensor, graph, delta_scores: Tensor, gamma: float, clip_range: Tuple[float, float]) -> Tensor:
    """Adjust policy logits using guidance scores.

    Parameters
    ----------
    logits:
        Original action logits including the STOP logit as the final entry.
    graph:
        Current :class:`MoleculeGraph` being edited. The argument is accepted
        for API compatibility but is not used directly.
    delta_scores:
        Guidance scores for each non-STOP action aligned with ``logits[:-1]``.
    gamma:
        Weighting factor applied to the clipped ``delta_scores``.
    clip_range:
        Tuple ``(a, b)`` specifying the clipping range for ``delta_scores``.

    Returns
    -------
    torch.Tensor
        Reweighted logits where ``gamma * clip(delta_scores, a, b)`` has been
        added to all non-STOP logits.
    """
    if gamma == 0:
        return logits

    a, b = clip_range
    if delta_scores.shape[0] != logits.shape[0] - 1:
        raise ValueError("delta_scores must align with non-STOP logits")

    adjusted = logits.clone()
    adjustment = torch.clamp(delta_scores, min=a, max=b) * gamma
    adjusted[:-1] = adjusted[:-1] + adjustment
    return adjusted


class AssemblyPrior:
    """Simple prior based on an approximate assembly index."""

    def __init__(self, coeff: float, target: int = 12) -> None:
        """Create an assembly prior.

        Parameters
        ----------
        coeff:
            Scaling coefficient applied to the deviation from ``target``.
        target:
            Desired assembly index value. Actions increasing the index incur a
            penalty on the logits.
        """

        self.coeff = coeff
        self.target = target

    def reweight(self, logits: Tensor, graph) -> Tensor:
        """Penalise edits based on the approximate assembly index."""

        ai = approx_AI(graph)
        penalty = self.coeff * (ai - self.target)
        logits[:-1] = logits[:-1] - penalty
        return logits
