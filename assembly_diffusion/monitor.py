import os
import json
import time
import threading
import queue
import signal
import collections
import sys
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

    def __init__(
        self,
        run_dir: str,
        use_tb: bool = True,
        hb_interval: float = 30.0,
        eta_window: int = 100,
    ):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.jsonl_path = os.path.join(run_dir, "events.jsonl")
        self.heartbeat_path = os.path.join(run_dir, "heartbeat.json")
        self._error_logged = False
        self._q = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        self._start_time = time.time()
        self._last_tick = time.time()
        self._dt_window: collections.deque[float] = collections.deque(maxlen=eta_window)
        self._step = 0
        self._total = None
        self._ckpt_path = None
        self._hb_interval = hb_interval
        self.tb = None
        if use_tb and SummaryWriter:
            try:
                self.tb = SummaryWriter(run_dir)
            except Exception as e:
                self._log_error_once("TensorBoard init failed", e)

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

        # Sentinel files for cross-platform "poke" mechanism.
        self._dump_sentinel = os.path.join(run_dir, "dump")
        self._ckpt_sentinel = os.path.join(run_dir, "checkpoint")

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
        self._dt_window.append(max(1e-6, min(dt, 5.0)))
        self._step = step
        self._total = total
        eta = None
        if total is not None and total > 0:
            remaining = max(0, total - step)
            avg_dt = sum(self._dt_window) / len(self._dt_window)
            eta = remaining * avg_dt
        self._event("progress", step=step, total=total, eta_seconds=eta)

    def scalar(self, name: str, value: float, step: int) -> None:
        self._event("scalar", name=name, value=float(value), step=step)
        if self.tb:
            try:
                self.tb.add_scalar(name, value, step)
            except Exception as e:
                self._log_error_once("TensorBoard add_scalar failed", e)

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

    def poll(self) -> tuple[bool, bool]:
        """Check for sentinel files requesting actions.

        Returns a pair ``(checkpoint, dump)`` indicating whether a checkpoint
        or a status dump was requested.  Sentinel files are removed after the
        request is observed and appropriate events are emitted.
        """

        ckpt_req = os.path.exists(self._ckpt_sentinel)
        dump_req = os.path.exists(self._dump_sentinel)

        if ckpt_req:
            try:
                os.remove(self._ckpt_sentinel)
            except Exception:
                pass
            self._sig_ckpt_req()

        if dump_req:
            try:
                os.remove(self._dump_sentinel)
            except Exception:
                pass
            self._sig_dump()

        return ckpt_req, dump_req

    # Internals
    def _log_error_once(self, msg: str, exc: Exception) -> None:
        if self._error_logged:
            return
        err = f"{msg}: {exc}"
        err_path = os.path.join(self.run_dir, "events_error.log")
        try:
            with open(err_path, "a") as ef:
                ef.write(err + "\n")
        except Exception:
            pass
        try:
            print(err, file=sys.stderr)
        except Exception:
            pass
        self._error_logged = True

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
        try:
            f = open(self.jsonl_path, "a", buffering=1)
        except Exception as e:
            self._log_error_once("event writer failed to open", e)
            return
        with f:
            while not self._stop.is_set() or not self._q.empty():
                try:
                    evt = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue
                try:
                    f.write(json.dumps(evt) + "\n")
                except Exception as e:
                    self._log_error_once("event write failed", e)

    def _write_heartbeat(self) -> None:
        hb = {
            "time": datetime.utcnow().isoformat() + "Z",
            "step": self._step,
            "last_ckpt": self._ckpt_path,
        }
        try:
            with open(self.heartbeat_path, "w") as f:
                json.dump(hb, f)
        except Exception as e:
            self._log_error_once("heartbeat write failed", e)

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
