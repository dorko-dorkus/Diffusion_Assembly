import json
import subprocess

from assembly_diffusion import run_logger


def test_run_log_contains_header(tmp_path):
    log_file = tmp_path / "run.log"
    cfg = {"alpha": 1}

    # Ensure a clean logger even if another test initialised it
    run_logger.reset_run_logger()

    logger = run_logger.init_run_logger(
        str(log_file), grammar="G_MC", config=cfg, seed=42
    )
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
    for key in [
        "seeds",
        "packages",
        "git_hash",
        "grammar",
        "config",
        "command",
        "os_version",
    ]:
        assert key in header

    expected_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    assert header["git_hash"] == expected_hash
    assert header["grammar"] == "G_MC"
    assert header["config"] == cfg
    assert header["command"]
    assert header["os_version"]
