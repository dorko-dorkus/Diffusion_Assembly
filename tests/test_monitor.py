import json
import time
from pathlib import Path
from datetime import datetime, timedelta

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


def test_daily_rotation(tmp_path, monkeypatch):
    from assembly_diffusion import monitor as mon

    start = datetime(2020, 1, 1, 0, 0, 0)

    class FakeDateTime:
        current = start

        @classmethod
        def utcnow(cls):
            return cls.current

    monkeypatch.setattr(mon, "datetime", FakeDateTime)
    m = mon.RunMonitor(tmp_path, use_tb=False)
    m.scalar("a", 1.0, 0)
    time.sleep(0.1)
    FakeDateTime.current = start + timedelta(days=1)
    m.scalar("b", 2.0, 1)
    time.sleep(0.1)
    m.close()

    rotated = Path(tmp_path) / "events-20200101.jsonl"
    current = Path(tmp_path) / "events.jsonl"
    assert rotated.exists()
    assert current.exists()
    assert sum(1 for _ in rotated.open()) == 1
    assert sum(1 for _ in current.open()) == 1
