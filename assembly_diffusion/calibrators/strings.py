from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

Alphabet = Tuple[str, ...]

@dataclass
class StringGrammar:
    P: Alphabet = ("A", "B", "C")
    L_max: int = 12

    def sample_target_length(self, guided: bool = False, gamma: float = 0.0, rng: random.Random | None = None) -> int:
        """Sample a target length.
        Baseline: uniform over {1..L_max}. Guided: geometric tilt toward larger lengths via softmax weights w(l) = exp(gamma * l).
        """
        rng = rng or random
        if not guided or gamma == 0.0:
            return rng.randint(1, self.L_max)
        weights = [pow(2.718281828, gamma * l) for l in range(1, self.L_max + 1)]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for l, w in zip(range(1, self.L_max + 1), weights):
            acc += w
            if r <= acc:
                return l
        return self.L_max

    def sample_trajectory(self, length: int, rng: random.Random | None = None) -> List[str]:
        """Path-uniform over symbol insertions of fixed length."""
        rng = rng or random
        return [rng.choice(self.P) for _ in range(length)]

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
