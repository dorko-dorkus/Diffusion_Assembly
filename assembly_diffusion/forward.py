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

    def step(self, x_prev: MoleculeGraph, t: int) -> MoleculeGraph:
        """Apply one forward step by masking existing bonds.

        Each present bond is independently kept with probability ``alpha(t)``
        and removed otherwise.  Bonds that are already absent remain absent.
        """

        x = x_prev.copy()
        a = self.alpha(t)
        for i in range(len(x.atoms)):
            for j in range(i + 1, len(x.atoms)):
                if x.bonds[i, j] > 0 and random.random() > a:
                    x.bonds[i, j] = x.bonds[j, i] = 0
        return x

    def teacher_edit(self, x0: MoleculeGraph, xt: MoleculeGraph):
        """Return a uniformly sampled bond missing in ``xt`` or ``'STOP'``.

        The returned tuple ``(i, j, b)`` corresponds to the bond present in
        ``x0`` but masked in ``xt``.  If no bonds are missing the string
        ``'STOP'`` is returned.
        """

        missing = []
        n = len(x0.atoms)
        for i in range(n):
            for j in range(i + 1, n):
                b = int(x0.bonds[i, j])
                if b > 0 and int(xt.bonds[i, j]) == 0:
                    missing.append((i, j, b))
        if not missing:
            return "STOP"
        return random.choice(missing)
