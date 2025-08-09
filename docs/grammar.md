# Molecular Grammar

The molecular diffusion model operates under a fixed grammar ``G`` describing
which molecules can be constructed and what edits are considered legal.

## Allowed atoms

Only neutral atoms of the following types are permitted:

| Atom | Maximum valence |
| ---- | --------------- |
| C    | 4 |
| N    | 3 |
| O    | 2 |
| H    | 1 |

## Bond orders

Bond orders are restricted to ``0`` (remove bond) and ``1``–``3`` for single,
double and triple bonds respectively.

## Charge state

All molecules are assumed to be neutral.  The grammar does not model explicit
charges and any edit resulting in a charged species is considered illegal.

## Assembly step

One *assembly step* performs exactly one of the following operations:

1. **Bond edit** – set the bond order ``b`` between two existing atoms ``i`` and
   ``j`` (with ``i < j``) to ``b`` in ``{0, 1, 2, 3}``.
2. **Atom insertion** – attach a new atom from the allowed set to an existing
   atom using a single bond.
3. **STOP** – terminate the assembly process.

An edit is *legal* if it respects the atom set, bond order limits and the
valence caps listed above.  Attempts to exceed valence or use unsupported bond
orders are masked out by the feasibility mask.

The helpers in ``assembly_diffusion.mask`` and ``assembly_diffusion.graph``
enforce this specification.

