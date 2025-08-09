"""Feasibility masking governed by the molecular grammar.

The grammar is declared via :data:`GRAMMAR_SPEC` and enumerates the allowed
atoms, bond orders and valence caps for neutral molecules.  See
``docs/grammar.md`` for the full specification.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

from .graph import MoleculeGraph, Chem, VALENCE_CAP, ALLOWED_ATOMS


# ---------------------------------------------------------------------------
# Grammar declaration -------------------------------------------------------

#: Top level molecular grammar specification used throughout the project.
#: The specification is intentionally small: only neutral C, N, O and H atoms
#: are allowed and bond orders are limited to single, double and triple bonds.
GRAMMAR_SPEC = {
    "atoms": tuple(ALLOWED_ATOMS),
    "bond_orders": (0, 1, 2, 3),
    "max_valence": VALENCE_CAP,
    "neutral_charge": True,
    "assembly_step": (
        "Edit an existing bond or attach a new atom with a single bond",
    ),
}


EditTuple = Union[Tuple[int, int, int], Tuple[str, int, str], str]


def valence_check(x: MoleculeGraph, i: int, j: int, b: int) -> bool:
    """Return ``True`` if editing bond ``(i, j)`` to ``b`` respects valence."""

    current = int(x.bonds[i, j])
    deg_i = x.degree(i) - current + b
    deg_j = x.degree(j) - current + b

    max_val = GRAMMAR_SPEC["max_valence"]
    if deg_i > max_val.get(x.atoms[i], 4):
        return False
    if deg_j > max_val.get(x.atoms[j], 4):
        return False

    if Chem is not None:
        test = x.apply_edit(i, j, b)
        return test.is_valid()
    return True


def enforce_grammar_strict(edit: EditTuple, g: MoleculeGraph) -> None:
    """Raise ``AssertionError`` if ``edit`` violates :data:`GRAMMAR_SPEC`.

    The error message includes the offending ``edit`` tuple to ease debugging.
    """

    if edit == "STOP":
        return

    if isinstance(edit, tuple) and edit and edit[0] == "ADD":
        _, i, atom = edit
        if atom not in GRAMMAR_SPEC["atoms"]:
            raise AssertionError(f"GRAMMAR violation for edit {edit}: unknown atom")
        if not (0 <= i < len(g.atoms)):
            raise AssertionError(f"GRAMMAR violation for edit {edit}: bad index")
        if g.free_valence(i) <= 0:
            raise AssertionError(f"GRAMMAR violation for edit {edit}: no valence")
        return

    # Bond edit
    i, j, b = edit  # type: ignore[misc]
    n = len(g.atoms)
    if not (0 <= i < n and 0 <= j < n and i < j):
        raise AssertionError(f"GRAMMAR violation for edit {edit}: bad indices")
    if b not in GRAMMAR_SPEC["bond_orders"]:
        raise AssertionError(
            f"GRAMMAR violation for edit {edit}: illegal bond order",
        )


def build_feasibility_mask(g: MoleculeGraph) -> Dict[EditTuple, int]:
    """Return a feasibility mask of edits consistent with :data:`GRAMMAR_SPEC`.

    The returned dictionary has keys describing edit tuples and values ``1`` for
    legal edits and ``0`` for illegal ones.  Bond edits are enumerated for all
    pairs ``(i, j)`` with ``i < j`` and bond orders in the grammar.  Atom
    insertions are included for all sites with free valence.
    """

    # Validate the existing molecule against the grammar first
    for idx, atom in enumerate(g.atoms):
        if atom not in GRAMMAR_SPEC["atoms"]:
            raise AssertionError(
                f"GRAMMAR violation for edit ('ADD', {idx}, '{atom}')",
            )
        cap = GRAMMAR_SPEC["max_valence"][atom]
        if g.degree(idx) > cap:
            raise AssertionError(
                f"GRAMMAR violation for edit {(idx, None, None)}: degree exceeds cap",
            )
    n = len(g.atoms)
    for i in range(n):
        for j in range(i + 1, n):
            b = int(g.bonds[i, j])
            if b not in GRAMMAR_SPEC["bond_orders"]:
                raise AssertionError(
                    f"GRAMMAR violation for edit {(i, j, b)}: bond order not allowed",
                )

    mask: Dict[EditTuple, int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            for b in GRAMMAR_SPEC["bond_orders"]:
                edit = (i, j, b)
                enforce_grammar_strict(edit, g)
                mask[edit] = 1 if valence_check(g, i, j, b) else 0
    for i in g.free_valence_sites():
        for atom in GRAMMAR_SPEC["atoms"]:
            edit = ("ADD", i, atom)
            enforce_grammar_strict(edit, g)
            mask[edit] = 1
    mask["STOP"] = 1
    assert_mask_equals_grammar(g, mask)
    return mask


def assert_mask_equals_grammar(g: MoleculeGraph, mask: Dict[EditTuple, int]) -> None:
    """Assert that ``mask`` matches the grammar for molecule ``g`` exactly."""

    expected: Dict[EditTuple, int] = {}
    n = len(g.atoms)
    for i in range(n):
        for j in range(i + 1, n):
            for b in GRAMMAR_SPEC["bond_orders"]:
                expected[(i, j, b)] = 1 if valence_check(g, i, j, b) else 0
    for i in g.free_valence_sites():
        for atom in GRAMMAR_SPEC["atoms"]:
            expected[("ADD", i, atom)] = 1
    expected["STOP"] = 1

    mask_keys = set(mask.keys())
    expected_keys = set(expected.keys())
    if mask_keys != expected_keys:
        missing = expected_keys - mask_keys
        extra = mask_keys - expected_keys
        raise AssertionError(
            f"Grammar mismatch: missing={missing}, extra={extra}",
        )
    for edit, exp in expected.items():
        got = mask[edit]
        if got != exp:
            raise AssertionError(
                f"Grammar mismatch for edit {edit}: mask={got}, expected={exp}",
            )


# Backwards compatibility ----------------------------------------------------
class FeasibilityMask:
    """Compute feasibility masks over possible edit actions."""

    def mask_edits(self, x: MoleculeGraph):
        return build_feasibility_mask(x)

