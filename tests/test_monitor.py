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


def test_sentinel_poll(tmp_path):
    m = RunMonitor(tmp_path, use_tb=False)
    (tmp_path / "dump").touch()
    (tmp_path / "checkpoint").touch()
    ckpt, dump = m.poll()
    assert ckpt and dump
    m.close()


def test_checkpoint_event_has_metadata(tmp_path):
    ckpt = tmp_path / "ckpt.bin"
    data = b"checkpoint-data"
    ckpt.write_bytes(data)
    m = RunMonitor(tmp_path, use_tb=False)
    m.set_checkpoint(str(ckpt))
    m.close()
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    evt = next(e for e in events if e["kind"] == "checkpoint")
    assert evt["path"] == str(ckpt)
    assert evt["size_bytes"] == len(data)
    import hashlib
    assert evt["checksum"] == hashlib.sha256(data).hexdigest()


def test_resume_event(tmp_path):
    m = RunMonitor(tmp_path, use_tb=False, hb_interval=0.1)
    src = "source.ckpt"
    m.resume_from(src)
    time.sleep(0.2)
    m.close()
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    evt = next(e for e in events if e["kind"] == "resume_ok")
    assert evt["path"] == src
    hb = json.loads((tmp_path / "heartbeat.json").read_text())
    assert hb["last_ckpt"] == src
