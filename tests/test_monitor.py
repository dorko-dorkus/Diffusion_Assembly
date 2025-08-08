import json
import time
from pathlib import Path

from assembly_diffusion.monitor import RunMonitor


def test_heartbeat_thread_runs(tmp_path):
    m = RunMonitor(tmp_path, use_tb=False, hb_interval=0.1)
    # Allow heartbeat thread to emit at least once
    time.sleep(0.25)
    m.close()
    hb_file = Path(tmp_path) / "heartbeat.json"
    assert hb_file.exists()
    with hb_file.open() as f:
        hb = json.load(f)
    assert "time" in hb and "step" in hb
