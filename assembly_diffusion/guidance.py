import torch
from torch import Tensor
from typing import Tuple


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
