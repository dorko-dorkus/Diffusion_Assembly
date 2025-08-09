import torch
from torch import Tensor
from typing import Tuple

from .assembly_index import approx_AI, AssemblyIndex


@torch.no_grad()
def assembly_guidance_scores(batch_states, cand_actions, grammar, mode: str = "A_lower") -> torch.Tensor:
    """Compute guidance scores for candidate actions in each state.

    Parameters
    ----------
    batch_states:
        Iterable of state objects.
    cand_actions:
        List of per-state candidate action collections.
    grammar:
        Object providing ``apply_action``.
    mode:
        One of ``"A_exact"``, ``"A_lower"`` or ``"heuristic"`` controlling the
        scoring method.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[B, A_max]`` containing scores with ``-inf`` padding.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = len(batch_states)
    A_max = max((len(a) for a in cand_actions), default=0)
    scores = torch.full((B, A_max), float("-inf"), device=device)

    for i, (state, actions) in enumerate(zip(batch_states, cand_actions)):
        vals = []
        for a in actions:
            succ = grammar.apply_action(state, a)
            if mode == "A_exact":
                A = AssemblyIndex.A_star_exact_or_none(succ)
                if A is None:
                    A = AssemblyIndex.A_lower_bound(succ)
            elif mode == "A_lower":
                A = AssemblyIndex.A_lower_bound(succ)
            else:
                A = grammar.heuristic_A(succ)
            vals.append(float(A))
        if not vals:
            continue
        t = torch.tensor(vals, device=device)
        t = (t - t.mean()) / (t.std(unbiased=False) + 1e-8)
        scores[i, : len(vals)] = t
    return scores


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


def additive_guidance(logits: Tensor, A_hat: Tensor, lam: float) -> Tensor:
    """Apply additive guidance to action logits.

    Implements ``log p̂(a | s) = log p_θ(a | s) - λ · Â(s ⊕ a)``, where ``Â`` is
    the approximate assembly index of the successor state.

    Parameters
    ----------
    logits:
        Model logits ``log p_θ(a | s)`` for each action.
    A_hat:
        Tensor of ``Â(s ⊕ a)`` values aligned with ``logits``.
    lam:
        Guidance weight ``λ``.

    Returns
    -------
    torch.Tensor
        Logits after subtracting ``λ · Â(s ⊕ a)``.
    """

    if logits.shape != A_hat.shape:
        raise ValueError("logits and A_hat must share the same shape")
    return logits - lam * A_hat


def linear_lambda_schedule(step: int, total_steps: int, lambda_max: float) -> float:
    """Linearly ramp ``λ`` from ``0`` to ``lambda_max`` over diffusion steps."""

    if total_steps <= 1:
        return float(lambda_max)
    return float(lambda_max) * step / (total_steps - 1)


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


__all__ = [
    "assembly_guidance_scores",
    "reweight",
    "additive_guidance",
    "linear_lambda_schedule",
    "AssemblyPrior",
]
