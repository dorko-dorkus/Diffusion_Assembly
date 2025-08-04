import pathlib
import random
import sys

import torch

# Ensure repository root on path for direct import
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembly_diffusion.forward import ForwardKernel
from assembly_diffusion.graph import MoleculeGraph


def test_step_removes_bond_when_prob_exceeds_alpha():
    random.seed(0)
    bonds = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    x = MoleculeGraph(["C", "C"], bonds)
    kernel = ForwardKernel(beta0=1.0, T=1)
    xt = kernel.step(x, t=1)
    assert int(xt.bonds[0, 1]) == 0


def test_step_no_change_at_t0():
    random.seed(0)
    bonds = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    x = MoleculeGraph(["C", "C"], bonds)
    kernel = ForwardKernel(beta0=1.0, T=10)
    xt = kernel.step(x, t=0)
    assert torch.equal(xt.bonds, x.bonds)
