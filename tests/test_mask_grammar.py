import torch
import pytest

from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.mask import build_feasibility_mask


@pytest.mark.parametrize(
    "atoms,bonds,edit,expected",
    [
        # Tree growth within valence limits is legal
        (
            ["C", "H", "H", "H"],
            [
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            ("ADD", 0, "H"),
            1,
        ),
        # Illegal bond orders are masked
        (
            ["O", "O"],
            [
                [0, 0],
                [0, 0],
            ],
            (0, 1, 3),
            0,
        ),
        # Over-valent edits are masked
        (
            ["C", "H", "H", "H", "H"],
            [
                [0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            (0, 1, 2),
            0,
        ),
    ],
)
def test_mask_grammar(atoms, bonds, edit, expected):
    g = MoleculeGraph(atoms, torch.tensor(bonds, dtype=torch.int64))
    mask = build_feasibility_mask(g)
    assert mask.get(edit, 0) == expected

