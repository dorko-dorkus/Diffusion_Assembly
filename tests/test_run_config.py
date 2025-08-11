import pytest

from assembly_diffusion.config import load_run_config
from assembly_diffusion.cli import build_ai


def _write_cfg(path, method):
    path.write_text(
        """seeds: [0]
N_samp: 1
guidance: 0.0
ai:
  method: {method}
  trials: 1
  timeout_s: 1
""".format(method=method)
    )


def test_ai_method_toggle(tmp_path):
    cfg_path = tmp_path / "run.yml"
    _write_cfg(cfg_path, "surrogate")
    cfg = load_run_config(cfg_path)
    ai = build_ai(cfg.ai.method)
    assert ai.__class__.__name__ == "AISurrogate"

    _write_cfg(cfg_path, "assemblymc")
    cfg = load_run_config(cfg_path)
    ai = build_ai(cfg.ai.method)
    assert ai.__class__.__name__ == "AssemblyMC"


def test_ai_method_exact_warns(tmp_path):
    cfg_path = tmp_path / "run.yml"
    _write_cfg(cfg_path, "exact")
    with pytest.warns(DeprecationWarning) as w:
        cfg = load_run_config(cfg_path)
    assert len(w) == 1
    ai = build_ai(cfg.ai.method)
    assert ai.__class__.__name__ == "AssemblyMC"
