from __future__ import annotations

"""Vocabularies over molecular bond edits.

This module defines a light-weight :class:`Edit` dataclass representing a bond
edit and exposes :class:`EditVocab` for enumerating and applying edits to
:meth:`~assembly_diffusion.graph.MoleculeGraph` instances.
"""

from dataclasses import dataclass
from typing import List

from .graph import MoleculeGraph


@dataclass(frozen=True)
class Edit:
    """Representation of a bond edit.

    An edit is either the special ``STOP`` action or a triple ``(i, j, b)``
    specifying the atoms ``i`` and ``j`` to connect and the bond order ``b``.
    The ``STOP`` action is represented by ``Edit()`` with all fields ``None``.
    """

    i: int | None = None
    j: int | None = None
    b: int | None = None

    @property
    def is_stop(self) -> bool:
        """Return ``True`` if this edit denotes the ``STOP`` action."""

        return self.i is None and self.j is None and self.b is None


class EditVocab:
    """Vocabulary over possible bond edits."""

    STOP = Edit()

    @staticmethod
    def enumerate_edits(x: MoleculeGraph) -> List[Edit]:
        """Enumerate all possible edits for ``x``.

        The edits are returned in a stable, deterministic order consisting of
        all triples ``(i, j, b)`` with ``i < j`` and ``b`` in ``{0, 1, 2}``,
        followed by the ``STOP`` action.
        """

        n = len(x.atoms)
        edits: List[Edit] = []
        for i in range(n):
            for j in range(i + 1, n):
                for b in [0, 1, 2]:
                    edits.append(Edit(i, j, b))
        edits.append(EditVocab.STOP)
        return edits

    @staticmethod
    def apply(x: MoleculeGraph, edit: Edit) -> MoleculeGraph:
        """Apply ``edit`` to ``x`` and return the resulting graph.

        The special ``STOP`` edit returns ``x`` unchanged.  Otherwise the
        operation delegates to :meth:`MoleculeGraph.apply_edit`.
        """

        if edit.is_stop:
            return x
        return x.apply_edit(edit.i, edit.j, edit.b)
