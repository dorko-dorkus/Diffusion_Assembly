import json
import os
import time
import queue
from pathlib import Path

import pytest

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


def test_drop_counter_and_retry(tmp_path):
    m = RunMonitor(tmp_path, use_tb=False)
    # stop background threads to control queue behavior
    m._stop.set()
    m._writer_thread.join()
    m._hb_thread.join()
    m._sampler_thread.join()

    class DummyQ:
        def __init__(self):
            self.items = []
            self.fail_first_n = 0

        def put_nowait(self, item):
            if self.fail_first_n > 0:
                self.fail_first_n -= 1
                raise queue.Full
            self.items.append(item)

        def put(self, item, timeout=None):
            if self.fail_first_n > 0:
                self.fail_first_n -= 1
                raise queue.Full
            self.items.append(item)

    q = DummyQ()
    m._q = q

    q.fail_first_n = 1
    m.scalar("a", 1.0, 0)
    assert m._dropped == 1
    assert q.items == []

    q.fail_first_n = 2
    m.set_checkpoint("ckpt")
    assert len(q.items) == 1 and q.items[0]["kind"] == "checkpoint"

    m._emit_dropped()
    assert len(q.items) == 2
    assert q.items[1]["kind"] == "dropped_events" and q.items[1]["count"] == 1

    m.close()


def test_eta_ema_smoothing(tmp_path, monkeypatch):
    import assembly_diffusion.monitor as mmod

    # Fix time during init so last_tick starts at 0.
    monkeypatch.setattr(mmod.time, "time", lambda: 0.0)
    m = mmod.RunMonitor(tmp_path, use_tb=False, hb_interval=999, eta_window=2)
    m._stop.set()
    m._writer_thread.join()
    m._hb_thread.join()
    m._sampler_thread.join()

    def tick_with_dt(step, dt):
        now = m._last_tick + dt
        monkeypatch.setattr(mmod.time, "time", lambda: now)
        m.tick(step=step, total=10)

    tick_with_dt(1, 1)
    tick_with_dt(2, 100)
    ema_after_outlier = m._dt_ema
    assert 1 < ema_after_outlier < 100

    tick_with_dt(3, 1)
    assert m._dt_ema < ema_after_outlier

    m.close()


def test_daily_rotation_symlink(tmp_path, monkeypatch):
    import datetime as dt
    import assembly_diffusion.monitor as mmod

    class FakeDatetime:
        current = dt.datetime(2024, 1, 1)

        @classmethod
        def utcnow(cls):
            return cls.current

        @classmethod
        def utcfromtimestamp(cls, ts):
            return dt.datetime.utcfromtimestamp(ts)

    monkeypatch.setattr(mmod, "datetime", FakeDatetime)

    m = mmod.RunMonitor(tmp_path, use_tb=False, hb_interval=999)
    m.scalar("a", 1.0, 0)
    time.sleep(0.2)

    FakeDatetime.current = dt.datetime(2024, 1, 2)
    m.scalar("b", 2.0, 1)
    time.sleep(0.5)
    m.close()

    first = tmp_path / "events-20240101.jsonl"
    second = tmp_path / "events-20240102.jsonl"
    link = tmp_path / "events.jsonl"
    assert first.exists()
    assert second.exists()
    assert link.is_symlink()
    assert os.path.samefile(link, second)
