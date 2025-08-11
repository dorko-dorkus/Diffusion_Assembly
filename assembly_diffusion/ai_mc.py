from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS
except ImportError:  # pragma: no cover - exercised when RDKit is absent
    Chem = None  # type: ignore[assignment]
    BRICS = None  # type: ignore[assignment]

from .graph import MoleculeGraph


def _require_rdkit() -> None:
    """Raise an informative error if RDKit is unavailable."""
    if Chem is None:  # pragma: no cover - exercised when RDKit is absent
        raise ImportError("RDKit is required for molecule operations")


def _mol_from_graph(g: MoleculeGraph) -> "Chem.Mol":
    _require_rdkit()
    return g.to_rdkit()


def _canon(sm: "Chem.Mol") -> str:
    _require_rdkit()
    # Canonical, include isotopes to preserve multiplicity cues if present
    return Chem.MolToSmiles(sm, isomericSmiles=True)


def _brics_cuts(mol: "Chem.Mol") -> List[Tuple[int, int]]:
    _require_rdkit()
    # return list of bond indices (begin,end) that are valid BRICS cuts
    bonds: List[Tuple[int, int]] = []
    for b in mol.GetBonds():
        if b.IsInRing():  # Cronin normally allows ring cuts via BRICS; be conservative by default
            continue
        bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    # If RDKit BRICS is available, prefer its decomposition points
    if BRICS is not None:
        try:
            brics: List[Tuple[int, int]] = []
            for (i, j), _ in BRICS.FindBRICSBonds(mol):
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is None or bond.IsInRing():
                    continue
                brics.append((i, j))
            if brics:
                bonds = brics
        except (ValueError, RuntimeError):  # pragma: no cover - defensive against RDKit issues
            pass
    return bonds


def _split_on_bond(
    mol: "Chem.Mol", i: int, j: int
) -> Optional[Tuple["Chem.Mol", "Chem.Mol"]]:
    _require_rdkit()
    em = Chem.EditableMol(Chem.Mol(mol))
    b = mol.GetBondBetweenAtoms(i, j)
    if b is None:
        return None
    em.RemoveBond(i, j)
    frag = em.GetMol()
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
    if len(frags) != 2:
        return None
    return frags[0], frags[1]


@dataclass
class AssemblyMC:
    # number of MC restarts
    samples: int = 256
    # beam width for partial states
    beam: int = 8
    # stop when we can’t improve after this many samples
    patience: int = 64
    # treat single atoms as primitives
    primitives: str = "atoms"  # or "brics"

    def _is_primitive(self, m: "Chem.Mol") -> bool:
        if self.primitives == "atoms":
            return m.GetNumAtoms() == 1
        # "brics" primitive: fragments with no valid BRICS cut remain primitive
        return len(_brics_cuts(m)) == 0

    def _state_cost(self, seen_unique: set[str]) -> int:
        # Heuristic MA: count of unique non-primitive fragments built once
        return len(seen_unique)

    def estimate(self, graph: MoleculeGraph) -> int:
        if Chem is None:
            # fall back to a trivial upper bound if RDKit missing
            # keep your previous behavior but mark as coarse
            g = graph
            return int(max(0, (g.bonds.sum().item() // 2)))

        root = _mol_from_graph(graph)
        best = float("inf")
        no_improve = 0

        # fragment cost cache across runs
        frag_cache: Dict[str, int] = {}

        for _ in range(self.samples):
            # Each hypothesis keeps a multiset of “to split” fragments and a set of unique built fragments
            Hyp = Tuple[List["Chem.Mol"], set[str]]
            beam: List[Hyp] = [([root], set())]

            while True:
                new_beam: List[Hyp] = []
                progressed = False
                for todo, uniq in beam:
                    # are we done? all primitives
                    if all(self._is_primitive(m) for m in todo):
                        new_beam.append((todo, uniq))
                        continue

                    # pick a non-primitive to split; prefer largest to expose duplicates
                    idx = max(
                        range(len(todo)),
                        key=lambda k: todo[k].GetNumAtoms() if not self._is_primitive(todo[k]) else -1,
                    )
                    mol = todo[idx]
                    cuts = _brics_cuts(mol) if self.primitives == "brics" else _brics_cuts(mol)
                    if not cuts:
                        # cannot split further; consider it as a unique built fragment
                        uniq2 = set(uniq)
                        uniq2.add(_canon(mol))
                        new_beam.append((todo, uniq2))
                        continue

                    # score each possible cut by duplicate-gain
                    scored: List[Tuple[int, Tuple["Chem.Mol", "Chem.Mol"]]] = []
                    for (i, j) in cuts:
                        pair = _split_on_bond(mol, i, j)
                        if pair is None:
                            continue
                        a, b = pair
                        ca, cb = _canon(a), _canon(b)
                        dup_gain = int(ca in uniq) + int(cb in uniq)
                        # optionally weight by global frequency from cache to bias toward reusable parts
                        dup_gain += int(ca in frag_cache) + int(cb in frag_cache)
                        scored.append((dup_gain, (a, b)))

                    if not scored:
                        uniq2 = set(uniq)
                        uniq2.add(_canon(mol))
                        new_beam.append((todo, uniq2))
                        continue

                    progressed = True
                    # keep top-N cuts, break ties randomly
                    random.shuffle(scored)
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top = scored[: max(1, self.beam // max(1, len(beam)))]

                    for _, (a, b) in top:
                        todo2 = list(todo)
                        # replace mol with its two parts
                        todo2.pop(idx)
                        todo2.extend([a, b])
                        uniq2 = set(uniq)
                        # any non-primitive fragment we “materialize” counts uniquely
                        for frag in (a, b):
                            if not self._is_primitive(frag):
                                uniq2.add(_canon(frag))
                        new_beam.append((todo2, uniq2))

                if not progressed:
                    # nothing else to split
                    beam = new_beam
                    break

                # prune beam
                beam = sorted(new_beam, key=lambda h: self._state_cost(h[1]))[: self.beam]

            # Finished a run: compute cost
            # Cost is number of unique non-primitive fragments encountered (proxy for MA)
            for _, uniq in beam:
                cost = self._state_cost(uniq)
                if cost < best:
                    best = cost
                    no_improve = 0
                else:
                    no_improve += 1

            if no_improve >= self.patience:
                break

        return int(best if best < float("inf") else 0)

