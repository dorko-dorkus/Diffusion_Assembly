import os
from pathlib import Path

import torch
import pytest

from assembly_diffusion.graph import MoleculeGraph
from assembly_diffusion.compute_ai import compute_ai
from assembly_diffusion.assembly_index import AssemblyIndex, approx_AI
import assembly_diffusion.extern.assemblymc as amc
from assembly_diffusion.extern.assemblymc import AssemblyMCError


def _simple_graph() -> MoleculeGraph:
    atoms = ["C"]
    bonds = torch.zeros((1, 1), dtype=torch.int64)
    return MoleculeGraph(atoms, bonds)


def _dummy_binary(path: Path) -> Path:
    script = path / "dummy_assemblymc.py"
    script.write_text(
        """#!/usr/bin/env python3
import json, sys
smiles = sys.argv[1]
print(f'A_star: {len(smiles)}')
print('d_min_est: 0')
with open('stats.json', 'w') as f:
    json.dump({}, f)
"""
    )
    script.chmod(0o755)
    return script


def test_surrogate_bounds():
    g = _simple_graph()
    lower, upper = compute_ai(g, method="surrogate")
    assert lower <= upper


def test_assemblymc_bounds(monkeypatch, tmp_path):
    g = _simple_graph()
    bin_path = _dummy_binary(tmp_path)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ASSEMBLYMC_BIN", str(bin_path))
    monkeypatch.setenv("ASSEMBLYMC_CACHE", str(cache_dir))
    lower, upper = compute_ai(g, method="assemblymc")
    assert lower == upper == 1


def test_assemblymc_missing_env(monkeypatch):
    g = _simple_graph()
    amc._CACHE.clear()
    monkeypatch.delenv("ASSEMBLYMC_BIN", raising=False)
    with pytest.raises(AssemblyMCError):
        compute_ai(g, method="assemblymc")
    lower, upper = compute_ai(g, method="assemblymc", allow_fallback=True)
    assert lower <= upper
    # Ensure fallback matches surrogate computation
    s_lower = AssemblyIndex.A_lower_bound(g)
    s_upper = max(s_lower, approx_AI(g))
    assert (lower, upper) == (s_lower, s_upper)
