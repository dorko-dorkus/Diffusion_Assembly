"""Generate assembly index annotations for QM9 CHON subset."""
from __future__ import annotations

from typing import List

import pandas as pd

try:  # pragma: no cover - optional rdkit dependency
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.rdchem import MolSanitizeException
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None
    Descriptors = None
    rdMolDescriptors = None
    MurckoScaffold = None
    MolSanitizeException = RuntimeError


def _require_rdkit() -> None:
    """Raise an informative error if RDKit is unavailable."""
    if Chem is None or Descriptors is None or rdMolDescriptors is None or MurckoScaffold is None:
        raise ImportError(
            "RDKit is required for QM9 annotations. Install it via 'conda install -c conda-forge rdkit'."
        )

from .data import load_qm9_chon, DEFAULT_DATA_DIR
from .ai_surrogate import AISurrogate
from .ai_mc import AssemblyMC


def _descriptor_vec(mol: "Chem.Mol") -> List[float]:
    """Return a vector of six basic RDKit descriptors."""
    _require_rdkit()
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
    ]


def generate_qm9_chon_ai(
    output_path: str = "qm9_chon_ai.csv",
    max_heavy: int = 12,
    data_dir: str = DEFAULT_DATA_DIR,
    samples: int = 100,
) -> None:
    """Create ``output_path`` with assembly annotations for QM9 CHON subset."""

    try:
        dataset = load_qm9_chon(max_heavy=max_heavy, data_dir=data_dir)
    except (ImportError, RuntimeError, OSError):  # pragma: no cover - dependency missing
        dataset = []

    surrogate = AISurrogate()
    mc = AssemblyMC(samples=samples)

    rows = []
    for graph in dataset:
        try:
            _require_rdkit()
            mol = graph.to_rdkit()
            smiles = Chem.MolToSmiles(mol, canonical=True)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol)
            desc = _descriptor_vec(mol)
        except (ImportError, ValueError, RuntimeError, MolSanitizeException):
            smiles = "".join(graph.atoms)
            scaffold = "NA"
            desc = [0.0] * 6
        if hasattr(mc, "ai"):
            ai_exact = mc.ai(graph)
        else:
            ai_exact = mc.estimate(graph)
        ai_surr = surrogate.score(graph)
        ai_conflict = int(round(ai_surr) != ai_exact)
        rows.append([smiles, ai_exact, ai_surr, scaffold, *desc, ai_conflict])

    columns = [
        "smiles",
        "ai_exact",
        "ai_surrogate",
        "scaffold_id",
        "mol_wt",
        "logp",
        "hbd",
        "hba",
        "rot_bonds",
        "rings",
        "ai_conflict",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    generate_qm9_chon_ai()
