import math
import random

from .graph import MoleculeGraph, ALLOWED_ATOMS

class ForwardKernel:
    """Forward diffusion kernel removing bonds and inserting atoms."""

    def __init__(self, beta0: float = 0.1, T: int = 10):
        self.beta0 = beta0
        self.T = T

    def alpha(self, t: int) -> float:
        return math.exp(-self.beta0 * t / self.T)

    def delete_prob(self, t: int) -> float:
        """Probability of deleting an existing bond at timestep ``t``."""

        return 1.0 - self.alpha(t)

    def insert_prob(self, t: int) -> float:
        """Probability of inserting a random atom at timestep ``t``."""

        return (1.0 - self.alpha(t)) / 2.0

    def sample_xt(self, x0: MoleculeGraph, t: int) -> MoleculeGraph:
        x = x0.copy()
        # --- Bond deletions ---------------------------------------------
        for i in range(len(x.atoms)):
            for j in range(i + 1, len(x.atoms)):
                if x.bonds[i, j] > 0 and random.random() < self.delete_prob(t):
                    x.bonds[i, j] = x.bonds[j, i] = 0

        # --- Atom insertions -------------------------------------------
        if random.random() < self.insert_prob(t):
            sites = x.free_valence_sites()
            if sites:
                new_atom = random.choice(ALLOWED_ATOMS)
                attach = random.choice(sites)
                x.add_atom(new_atom, attach)
        return x

    def step(self, x_prev: MoleculeGraph, t: int) -> MoleculeGraph:
        """Apply one forward step including bond masking and atom insertion."""

        x = x_prev.copy()
        a = self.alpha(t)
        for i in range(len(x.atoms)):
            for j in range(i + 1, len(x.atoms)):
                if x.bonds[i, j] > 0 and random.random() > a:
                    x.bonds[i, j] = x.bonds[j, i] = 0

        if random.random() < self.insert_prob(t):
            sites = x.free_valence_sites()
            if sites:
                new_atom = random.choice(ALLOWED_ATOMS)
                attach = random.choice(sites)
                x.add_atom(new_atom, attach)
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
