import math
import random

from .graph import MoleculeGraph


class ForwardKernel:
    """Simple forward diffusion kernel that randomly removes bonds."""
    def __init__(self, beta0: float = 0.1, T: int = 10):
        self.beta0 = beta0
        self.T = T

    def alpha(self, t: int) -> float:
        return math.exp(-self.beta0 * t / self.T)

    def sample_xt(self, x0: MoleculeGraph, t: int) -> MoleculeGraph:
        x = x0.copy()
        for i in range(len(x.atoms)):
            for j in range(i + 1, len(x.atoms)):
                if x.bonds[i, j] > 0 and random.random() > self.alpha(t):
                    x.bonds[i, j] = x.bonds[j, i] = 0
        return x
