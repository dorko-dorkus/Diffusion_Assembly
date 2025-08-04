from .graph import MoleculeGraph, Chem, VALENCE_CAP, ALLOWED_ATOMS


def valence_check(x: MoleculeGraph, i: int, j: int, b: int) -> bool:
    """Return ``True`` if editing bond ``(i, j)`` to ``b`` respects valence.

    The check first enforces simple valence caps based on atom types. When
    RDKit is available, a full molecule sanitization is performed for stricter
    validation.  When RDKit is not installed, only the fast valence cap check
    is used.
    """

    current = int(x.bonds[i, j])
    deg_i = int(x.bonds[i].sum().item()) - current + b
    deg_j = int(x.bonds[j].sum().item()) - current + b

    if deg_i > VALENCE_CAP.get(x.atoms[i], 4):
        return False
    if deg_j > VALENCE_CAP.get(x.atoms[j], 4):
        return False

    if Chem is not None:
        test = x.apply_edit(i, j, b)
        return test.is_valid()
    return True


class FeasibilityMask:
    """Compute feasibility masks over possible edit actions."""

    def mask_edits(self, x: MoleculeGraph):
        mask = {}
        # --- Bond edits -------------------------------------------------
        for i in range(len(x.atoms)):
            for j in range(i + 1, len(x.atoms)):
                for b in [0, 1, 2, 3]:
                    mask[(i, j, b)] = 1 if valence_check(x, i, j, b) else 0

        # --- Atom insertions -------------------------------------------
        for i in x.free_valence_sites():
            for atom in ALLOWED_ATOMS:
                mask[("ADD", i, atom)] = 1

        mask["STOP"] = 1
        return mask
