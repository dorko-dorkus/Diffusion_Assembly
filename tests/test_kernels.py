import math
import torch
from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.forward import ForwardKernel
from assembly_diffusion import forward as forward_module
from assembly_diffusion.policy import ReversePolicy
from assembly_diffusion.backbone import GNNBackbone


def test_forward_alpha_bounds():
    kernel = ForwardKernel(beta0=0.1, T=10)
    assert kernel.alpha(0) == 1.0
    assert math.isclose(kernel.alpha(10), math.exp(-0.1))


def test_forward_sample_xt_removes_bond(monkeypatch):
    atoms = [6, 6]
    bonds = torch.tensor([[0, 1], [1, 0]], dtype=torch.int)
    g = MoleculeGraph(atoms, bonds)
    kernel = ForwardKernel(beta0=0.1, T=10)
    monkeypatch.setattr(forward_module.random, "random", lambda: 1.0)
    xt = kernel.sample_xt(g, t=10)
    assert xt.bonds[0, 1] == 0


def test_forward_sample_xt_keeps_bond(monkeypatch):
    atoms = [6, 6]
    bonds = torch.tensor([[0, 1], [1, 0]], dtype=torch.int)
    g = MoleculeGraph(atoms, bonds)
    kernel = ForwardKernel(beta0=0.1, T=10)
    monkeypatch.setattr(forward_module.random, "random", lambda: 0.0)
    xt = kernel.sample_xt(g, t=10)
    assert xt.bonds[0, 1] == 1


def test_reverse_policy_logits_shape():
    atoms = [6, 6]
    bonds = torch.zeros((2, 2), dtype=torch.int)
    g = MoleculeGraph(atoms, bonds)
    mask = {(0, 1, 0): 1, (0, 1, 1): 1, (0, 1, 2): 1, 'STOP': 1}
    policy = ReversePolicy(GNNBackbone())
    logits = policy.logits(g, t=0, mask=mask)
    assert logits.shape == (len(mask),)
