import os
import json
import time
import threading
import queue
import signal
import collections
import sys
import hashlib
from datetime import datetime
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# Optional resource monitoring dependencies.  These are best-effort and the
# monitor will operate without them if unavailable.
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - dependency not installed
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore

    try:
        pynvml.nvmlInit()
    except Exception:  # pragma: no cover - GPU may be absent
        pynvml = None  # type: ignore
except Exception:  # pragma: no cover - dependency not installed
    pynvml = None  # type: ignore


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
        self._dropped = 0
        self._drop_lock = threading.Lock()
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
        # Dedicated heartbeat thread for liveness reporting.
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._hb_thread.start()
        # Background sampler thread handles optional resource snapshots.  This
        # keeps monitoring out of the main training loop and avoids duplicated
        # boilerplate in callers.
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop, daemon=True
        )
        self._sampler_thread.start()

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
        self._hb_thread.join(timeout=5)
        self._sampler_thread.join(timeout=5)
        if self.tb:
            try:
                self.tb.flush()
                self.tb.close()
            except Exception:
                pass
        # Gracefully shut down NVML if it was initialized.
        if pynvml:
            try:  # pragma: no cover - GPU-specific
                pynvml.nvmlShutdown()
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

    def resume_from(self, path: str) -> None:
        self._ckpt_path = path
        self._event("resume_ok", path=path)

    def set_checkpoint(self, path: str) -> None:
        self._ckpt_path = path
        size = None
        checksum = None
        try:
            size = os.path.getsize(path)
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            checksum = h.hexdigest()
        except Exception as e:
            self._log_error_once("checkpoint metadata failed", e)
        self._event(
            "checkpoint",
            path=path,
            step=self._step,
            size_bytes=size,
            checksum=checksum,
        )

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

    def _roll_existing_jsonl(self, today: str) -> None:
        """Move leftover logs from previous days to a dated file."""
        if not os.path.exists(self.jsonl_path):
            return
        try:
            mtime = datetime.utcfromtimestamp(os.path.getmtime(self.jsonl_path))
            file_day = mtime.strftime("%Y%m%d")
            if file_day != today:
                rolled = os.path.join(self.run_dir, f"events-{file_day}.jsonl")
                os.replace(self.jsonl_path, rolled)
        except Exception as e:
            self._log_error_once("event log rotation failed", e)

    def _event(self, kind: str, **payload) -> None:
        evt = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "kind": kind,
        }
        evt.update({k: v for k, v in payload.items() if v is not None})
        if kind in {"checkpoint", "signal_checkpoint_request"}:
            for _ in range(3):
                try:
                    self._q.put(evt, timeout=0.05)
                    return
                except queue.Full:
                    continue
        else:
            try:
                self._q.put_nowait(evt)
                return
            except queue.Full:
                pass
        with self._drop_lock:
            self._dropped += 1

    def _writer_loop(self) -> None:
        current_day = datetime.utcnow().strftime("%Y%m%d")
        self._roll_existing_jsonl(current_day)
        try:
            f = open(self.jsonl_path, "a", buffering=1)
        except Exception as e:
            self._log_error_once("event writer failed to open", e)
            return
        while not self._stop.is_set() or not self._q.empty():
            today = datetime.utcnow().strftime("%Y%m%d")
            if today != current_day:
                try:
                    f.close()
                    rolled_path = os.path.join(self.run_dir, f"events-{current_day}.jsonl")
                    os.replace(self.jsonl_path, rolled_path)
                except Exception as e:
                    self._log_error_once("event log rotation failed", e)
                try:
                    f = open(self.jsonl_path, "a", buffering=1)
                    current_day = today
                except Exception as e:
                    self._log_error_once("event writer failed to open", e)
                    break
            try:
                evt = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                f.write(json.dumps(evt) + "\n")
            except Exception as e:
                self._log_error_once("event write failed", e)
        try:
            f.close()
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
        except Exception as e:
            self._log_error_once("heartbeat write failed", e)

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            self._write_heartbeat()
            self._emit_dropped()
            self._stop.wait(self._hb_interval)

    def _sample_resources(self) -> tuple[float | None, float | None, float | None, float | None]:
        """Best-effort CPU/GPU resource snapshot."""
        cpu = psutil.cpu_percent(interval=None) if psutil else None
        ram = (psutil.virtual_memory().used / 1e9) if psutil else None
        gpu_util = vram = None
        if pynvml:
            try:  # pragma: no cover - GPU-specific
                dev = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(dev)
                mem = pynvml.nvmlDeviceGetMemoryInfo(dev)
                gpu_util = float(util.gpu)
                vram = float(mem.used / 1e9)
            except Exception:
                pass
        return cpu, ram, vram, gpu_util

    def _sampler_loop(self) -> None:
        while not self._stop.is_set():
            cpu, ram, vram, gpu_util = self._sample_resources()
            if any(v is not None for v in (cpu, ram, vram, gpu_util)):
                self.resources(
                    cpu=cpu, ram_used_gb=ram, vram_used_gb=vram, gpu_util=gpu_util
                )
            self._stop.wait(self._hb_interval)

    def _sig_dump(self, *_):
        self._event(
            "signal_dump",
            step=self._step,
            uptime_seconds=int(time.time() - self._start_time),
        )

    def _sig_ckpt_req(self, *_):
        self._event("signal_checkpoint_request", step=self._step)

    def _emit_dropped(self) -> None:
        with self._drop_lock:
            dropped = self._dropped
            self._dropped = 0
        if dropped:
            self._event("dropped_events", count=dropped)
