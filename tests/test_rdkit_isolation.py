import importlib
import builtins
import sys
import pytest


def test_missing_rdkit_allows_imports(monkeypatch):
    """Simulate missing RDKit and ensure modules degrade gracefully."""
    for key in list(sys.modules):
        if key.startswith("rdkit"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("rdkit"):
            raise ModuleNotFoundError("No module named 'rdkit'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        graph = importlib.reload(importlib.import_module("assembly_diffusion.graph"))
        assert graph.Chem is None

        run_logger = importlib.reload(importlib.import_module("assembly_diffusion.run_logger"))
        assert hasattr(run_logger, "init_run_logger")

        canonicalize = importlib.reload(importlib.import_module("assembly_diffusion.canonicalize"))
        with pytest.raises(ImportError, match="RDKit is required"):
            canonicalize.canonicalize_smiles("CC")
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(importlib.import_module("assembly_diffusion.graph"))
        importlib.reload(importlib.import_module("assembly_diffusion.run_logger"))
        importlib.reload(importlib.import_module("assembly_diffusion.canonicalize"))
