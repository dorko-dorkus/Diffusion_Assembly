from __future__ import annotations
import itertools
import math
import torch
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

Alphabet = Tuple[str, str, str]


device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


@dataclass
class StringGrammar:
    P: Alphabet = ("A", "B", "C")
    L_max: int = 12

    def sample_target_length(
        self,
        guided: bool = False,
        gamma: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> int:
        """Sample a target length using ``torch`` on the selected device.

        Baseline draws uniformly from ``{1..L_max}``. Guided sampling applies a
        softmax tilt toward larger lengths via ``exp(gamma * l)``.
        """

        if not guided or gamma == 0.0:
            return int(
                torch.randint(
                    1, self.L_max + 1, (1,), device=device, generator=generator
                ).item()
            )
        lengths = torch.arange(1, self.L_max + 1, device=device, dtype=torch.float)
        weights = torch.exp(gamma * lengths)
        idx = torch.multinomial(weights, num_samples=1, generator=generator).item()
        return int(lengths[idx].item())

    def sample_trajectory(
        self, length: int, generator: torch.Generator | None = None
    ) -> List[str]:
        """Path-uniform over symbol insertions of fixed length using ``torch``."""

        idx = torch.randint(
            0, len(self.P), (length,), device=device, generator=generator
        )
        return [self.P[i] for i in idx.tolist()]

    @staticmethod
    def to_object(traj: List[str]) -> str:
        return "".join(traj)

    @staticmethod
    def canonical_id(x: str) -> str:
        # Identity is fine for strings
        return x

    @staticmethod
    def A_star(x: str) -> int:
        return len(x)

    @staticmethod
    def D_min(x: str) -> int:
        # Under simple concatenation with atomic symbols, there is only 1 minimal pathway per string.
        return 1

    @classmethod
    def enumerate(cls, L_max: int) -> pd.DataFrame:
        """Enumerate all strings of length ``≤ L_max``.

        Returns a DataFrame compatible with :meth:`Sampler.sample_S` containing
        exact counts of minimal pathways.  Each string occurs once and has
        ``d_min = 1``.  The ``frequency`` column is normalised over the entire
        enumeration.
        """

        rows = []
        total = sum(len(cls.P) ** L for L in range(1, L_max + 1))
        freq = 1.0 / total
        for L in range(1, L_max + 1):
            for symbols in itertools.product(cls.P, repeat=L):
                s = "".join(symbols)
                A = cls.A_star(s)
                rows.append(
                    {
                        "id": cls.canonical_id(s),
                        "universe": "S",
                        "grammar": f"S_v0_L{L_max}",
                        "As_lower": A,
                        "As_upper": A,
                        "validity": 1,
                        "frequency": freq,
                        "d_min": cls.D_min(s),
                    }
                )

        return pd.DataFrame(rows)
