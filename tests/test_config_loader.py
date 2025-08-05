from pathlib import Path

from assembly_diffusion.config import load_config


def test_load_config_ai_guided():
    path = Path(__file__).resolve().parent.parent / "configs" / "exp01.yaml"
    cfg = load_config(path, "ai_guided")
    assert cfg.dataset == "qm9_chon"
    assert cfg.hidden_dim == 128
    assert cfg.guid_mode == "exact_ai"
    assert cfg.guid_coeff == 0.5
