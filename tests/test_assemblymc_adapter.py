import os
import time
from pathlib import Path

import pytest

if "ASSEMBLYMC_BIN" not in os.environ:
    pytest.skip("ASSEMBLYMC_BIN not set", allow_module_level=True)

from assembly_diffusion.extern.assemblymc import a_star_and_dmin, AssemblyMCTimeout


def _dummy_binary(path: Path) -> Path:
    script = path / "dummy_assemblymc.py"
    script.write_text(
        """#!/usr/bin/env python3
import json, os, sys, time
smiles = sys.argv[1]
trials = int(sys.argv[2])
seed = int(sys.argv[3])
time.sleep(float(os.environ.get('ASSEMBLYMC_SLEEP', '0')))
A_star = len(smiles)
d_min = len(smiles)//2
print(f'A_star: {A_star}')
print(f'd_min_est: {d_min}')
with open('stats.json', 'w') as f:
    json.dump({'smiles': smiles, 'trials': trials, 'seed': seed}, f)
"""
    )
    script.chmod(0o755)
    return script


def test_cache_speedup_and_output(monkeypatch, tmp_path):
    bin_path = _dummy_binary(tmp_path)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ASSEMBLYMC_BIN", str(bin_path))
    monkeypatch.setenv("ASSEMBLYMC_CACHE", str(cache_dir))
    monkeypatch.setenv("ASSEMBLYMC_SLEEP", "0.2")
    mols = [
        "c1ccccc1",
        "Cc1ccccc1",
        "c1[nH]c(=O)[nH]c(=O)n1",
    ]
    first = []
    for sm in mols:
        t0 = time.time()
        a, d, s = a_star_and_dmin(sm)
        first.append(time.time() - t0)
        assert isinstance(a, int)
    second = []
    for sm in mols:
        t0 = time.time()
        a_star_and_dmin(sm)
        second.append(time.time() - t0)
    assert sum(first) / sum(second) >= 5.0


def test_timeout(monkeypatch, tmp_path):
    bin_path = _dummy_binary(tmp_path)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("ASSEMBLYMC_BIN", str(bin_path))
    monkeypatch.setenv("ASSEMBLYMC_CACHE", str(cache_dir))
    monkeypatch.setenv("ASSEMBLYMC_SLEEP", "5")
    with pytest.raises(AssemblyMCTimeout):
        a_star_and_dmin("C", timeout_s=0.1)
