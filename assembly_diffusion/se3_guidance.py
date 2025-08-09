import torch
from torch import Tensor
from typing import Optional

from .guidance import AssemblyPrior


class DummySE3Flow(torch.nn.Module):
    """Minimal SE(3)-equivariant-like flow operating on 3D coordinates.

    The flow simply maps distances from the origin to logits, serving as a
    placeholder for a real equivariant model."""

    def forward(self, coords: Tensor, t: int, mask: Optional[Tensor] = None) -> Tensor:
        # ``coords`` is expected to be ``(N, 3)``.  We use the negative radius as
        # a simple score, higher for atoms near the origin.
        logits = -torch.linalg.norm(coords, dim=-1)
        if mask is not None and mask.shape[0] != logits.shape[0]:
            logits = logits[: mask.shape[0]]
        return logits


class GeometryPrior(torch.nn.Module):
    """Very small geometric prior encouraging compact structures."""

    def forward(self, coords: Tensor) -> Tensor:
        # Penalise atoms further than unit distance from the origin.
        dist = torch.linalg.norm(coords, dim=-1)
        return -torch.clamp(dist - 1.0, min=0.0)


class SE3Guidance:
    """Plug-in guidance combining geometry priors and assembly heuristics."""

    def __init__(
        self,
        flow: torch.nn.Module,
        assembly_prior: Optional[AssemblyPrior] = None,
        geometry_prior: Optional[torch.nn.Module] = None,
    ) -> None:
        self.flow = flow
        self.assembly_prior = assembly_prior
        self.geometry_prior = geometry_prior

    def __call__(self, logits: Tensor, graph, t: int, mask: Tensor) -> Tensor:
        """Return guidance ``ΔS`` for non-STOP actions.

        The method computes logits from the ``flow`` based on the graph's 3D
        coordinates and contrasts them with the provided ``logits``.  Optionally
        ``assembly_prior`` and ``geometry_prior`` are applied and the resulting
        adjustment is returned for use with :func:`reweight`.
        """

        coords = getattr(graph, "coords", None)
        if coords is None:
            raise ValueError("SE3Guidance requires graphs with 3D coordinates")

        flow_logits = self.flow(coords, t, mask)
        flow_logits = flow_logits.to(logits.device)
        # Ensure alignment with non-STOP logits
        target_len = logits.shape[0] - 1
        if flow_logits.shape[0] < target_len:
            flow_logits = torch.nn.functional.pad(flow_logits, (0, target_len - flow_logits.shape[0]))
        elif flow_logits.shape[0] > target_len:
            flow_logits = flow_logits[:target_len]

        delta = flow_logits - logits[:-1]

        if self.geometry_prior is not None:
            delta = delta + self.geometry_prior(coords)

        if self.assembly_prior is not None:
            temp = self.assembly_prior.reweight(logits.clone(), graph)
            delta = delta + (temp[:-1] - logits[:-1])

        return delta


__all__ = ["DummySE3Flow", "GeometryPrior", "SE3Guidance"]
