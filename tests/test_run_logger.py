import json

from assembly_diffusion import run_logger


def test_run_log_contains_header(tmp_path):
    log_file = tmp_path / "run.log"
    cfg = {"alpha": 1}
    logger = run_logger.init_run_logger(str(log_file), grammar="G", config=cfg, seed=42)
    logger.info("run start")
    lines = log_file.read_text().splitlines()[:5]
    header = None
    for line in lines:
        try:
            header = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    assert header is not None
    for key in ["seeds", "packages", "git_hash", "grammar", "config"]:
        assert key in header
    assert header["grammar"] == "G"
    assert header["config"] == cfg
