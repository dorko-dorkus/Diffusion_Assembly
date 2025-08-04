from .graph import MoleculeGraph


class FeasibilityMask:
    """Compute feasibility masks over possible bond edits."""
    def mask_edits(self, x: MoleculeGraph):
        mask = {}
        for i in range(len(x.atoms)):
            for j in range(i + 1, len(x.atoms)):
                for b in [0, 1, 2]:
                    mask[(i, j, b)] = 1 if self.valence_ok(x, i, j, b) else 0
        mask['STOP'] = 1
        return mask

    def valence_ok(self, x: MoleculeGraph, i: int, j: int, b: int) -> bool:
        test = x.copy()
        test.bonds[i, j] = b
        test.bonds[j, i] = b
        return test.is_valid()
