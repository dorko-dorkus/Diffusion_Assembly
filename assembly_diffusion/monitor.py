import os
import json
import time
import threading
import queue
import signal
from datetime import datetime
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class RunMonitor:
    """Non-blocking run monitor.

    Writes JSONL events and periodically updates a heartbeat file to indicate
    liveness. Optionally records TensorBoard scalars and images if
    ``torch.utils.tensorboard`` is present.
    """

    def __init__(self, run_dir: str, use_tb: bool = True, hb_interval: float = 30.0):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.jsonl_path = os.path.join(run_dir, "events.jsonl")
        self.heartbeat_path = os.path.join(run_dir, "heartbeat.json")
        self._q = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        self._start_time = time.time()
        self._last_tick = time.time()
        self._step = 0
        self._total = None
        self._ckpt_path = None
        self._hb_interval = hb_interval
        self.tb = SummaryWriter(run_dir) if (use_tb and SummaryWriter) else None

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

        # Signal handlers are best-effort. They exist on Unix; on Windows they may not.
        try:
            signal.signal(signal.SIGUSR1, self._sig_dump)
            signal.signal(signal.SIGUSR2, self._sig_ckpt_req)
        except Exception:
            pass

    def close(self) -> None:
        self._stop.set()
        self._writer_thread.join(timeout=5)
        self._heartbeat_thread.join(timeout=5)
        if self.tb:
            try:
                self.tb.flush()
                self.tb.close()
            except Exception:
                pass

    # Public API
    def tick(self, step: int, total: Optional[int] = None) -> None:
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now
        self._step = step
        self._total = total
        eta = None
        if total is not None and total > 0:
            remaining = max(0, total - step)
            # Use a clamped dt as a crude EMA to avoid huge ETA swings
            avg_dt = max(1e-6, min(dt, 5.0))
            eta = remaining * avg_dt
        self._event("progress", step=step, total=total, eta_seconds=eta)

    def scalar(self, name: str, value: float, step: int) -> None:
        self._event("scalar", name=name, value=float(value), step=step)
        if self.tb:
            try:
                self.tb.add_scalar(name, value, step)
            except Exception:
                pass

    def resources(
        self,
        cpu: Optional[float] = None,
        ram_used_gb: Optional[float] = None,
        vram_used_gb: Optional[float] = None,
        gpu_util: Optional[float] = None,
    ) -> None:
        self._event(
            "resources",
            cpu=cpu,
            ram_used_gb=ram_used_gb,
            vram_used_gb=vram_used_gb,
            gpu_util=gpu_util,
        )

    def set_checkpoint(self, path: str) -> None:
        self._ckpt_path = path
        self._event("checkpoint", path=path, step=self._step)

    # Internals
    def _event(self, kind: str, **payload) -> None:
        evt = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "kind": kind,
        }
        evt.update({k: v for k, v in payload.items() if v is not None})
        try:
            self._q.put_nowait(evt)
        except queue.Full:
            # Drop telemetry rather than block the training loop
            pass

    def _writer_loop(self) -> None:
        with open(self.jsonl_path, "a", buffering=1) as f:
            while not self._stop.is_set() or not self._q.empty():
                try:
                    evt = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue
                try:
                    f.write(json.dumps(evt) + "\n")
                except Exception:
                    pass

    def _write_heartbeat(self) -> None:
        hb = {
            "time": datetime.utcnow().isoformat() + "Z",
            "step": self._step,
            "last_ckpt": self._ckpt_path,
        }
        try:
            with open(self.heartbeat_path, "w") as f:
                json.dump(hb, f)
        except Exception:
            pass

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            self._write_heartbeat()
            self._stop.wait(self._hb_interval)

    def _sig_dump(self, *_):
        self._event(
            "signal_dump",
            step=self._step,
            uptime_seconds=int(time.time() - self._start_time),
        )

    def _sig_ckpt_req(self, *_):
        self._event("signal_checkpoint_request", step=self._step)
