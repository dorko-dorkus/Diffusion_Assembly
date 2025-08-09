import os
import json
import time
import threading
import queue
import signal
import sys
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Optional resource monitoring dependencies.  These are best-effort and the
# monitor will operate without them if unavailable.
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - dependency not installed
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:  # pragma: no cover - GPU may be absent
        pynvml = None  # type: ignore
except ImportError:  # pragma: no cover - dependency not installed
    pynvml = None  # type: ignore


def _git_hash() -> str:
    """Return the current git commit hash or ``"unknown"``."""
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".git").exists():
        return "unknown"
    try:
        return (
            subprocess.check_output([
                "git",
                "rev-parse",
                "HEAD",
            ], stderr=subprocess.DEVNULL, cwd=repo_root)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


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
        git_hash: Optional[str] = None,
        config: Optional[dict] = None,
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
        self._eta_window = max(1, eta_window)
        # Exponential moving average coefficient equivalent to the given window.
        self._eta_alpha = 2.0 / (self._eta_window + 1)
        self._dt_ema: float | None = None
        self._step = 0
        self._total = None
        self._ckpt_path = None
        self._eta_seconds: float | None = None
        self._hb_interval = hb_interval
        self._git_hash = git_hash or _git_hash()
        self._config = config or {}
        try:
            with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
                json.dump({"git_hash": self._git_hash, "config": self._config}, f)
        except OSError as e:
            self._log_error_once("metadata write failed", e)
        self.tb = None
        if use_tb and SummaryWriter:
            try:
                self.tb = SummaryWriter(run_dir)
            except (OSError, RuntimeError) as e:
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
        except (AttributeError, OSError, ValueError):
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
            except OSError:
                pass
        # Gracefully shut down NVML if it was initialized.
        if pynvml:
            try:  # pragma: no cover - GPU-specific
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    # Public API
    def tick(self, step: int, total: Optional[int] = None) -> None:
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now
        dt = max(1e-6, min(dt, 5.0))
        if self._dt_ema is None:
            self._dt_ema = dt
        else:
            # Cheap exponential moving average to smooth ETA without storing history.
            self._dt_ema += self._eta_alpha * (dt - self._dt_ema)
        self._step = step
        self._total = total
        eta = None
        if total is not None and total > 0 and self._dt_ema is not None:
            remaining = max(0, total - step)
            eta = remaining * self._dt_ema
        self._eta_seconds = eta
        self._event("progress", step=step, total=total, eta_seconds=eta)

    def scalar(self, name: str, value: float, step: int) -> None:
        self._event("scalar", name=name, value=float(value), step=step)
        if self.tb:
            try:
                self.tb.add_scalar(name, value, step)
            except (RuntimeError, OSError) as e:
                self._log_error_once("TensorBoard add_scalar failed", e)

    def sample_smiles(self, smiles: list[str], step: int) -> None:
        self._event("sample_smiles", smiles=smiles, step=step)
        fname = os.path.join(self.run_dir, f"smiles_step{step:08d}.smi")
        try:
            with open(fname, "w") as f:
                f.write(f"# git_hash: {self._git_hash}\n")
                f.write("# config: " + json.dumps(self._config) + "\n")
                for s in smiles:
                    f.write(s + "\n")
        except OSError as e:
            self._log_error_once("sample_smiles write failed", e)

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
            meta_path = path + ".meta.json"
            with open(meta_path, "w") as mf:
                json.dump({"git_hash": self._git_hash, "config": self._config}, mf)
        except OSError as e:
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
            except OSError:
                pass
            self._sig_ckpt_req()

        if dump_req:
            try:
                os.remove(self._dump_sentinel)
            except OSError:
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
        except OSError:
            pass
        logger.error(err)
        self._error_logged = True

    def _roll_existing_jsonl(self, today: str) -> None:
        """Handle leftover logs when starting a new run.

        If ``events.jsonl`` is a regular file from a previous run, move it to a
        dated filename so the contents are preserved. Any pre-existing symlink
        is simply removed so a fresh one can be created.
        """

        if not os.path.lexists(self.jsonl_path):
            return
        if os.path.islink(self.jsonl_path):
            try:
                os.unlink(self.jsonl_path)
            except OSError:
                pass
            return
        try:
            mtime = datetime.utcfromtimestamp(os.path.getmtime(self.jsonl_path))
            file_day = mtime.strftime("%Y%m%d")
            rolled = os.path.join(self.run_dir, f"events-{file_day}.jsonl")
            os.replace(self.jsonl_path, rolled)
        except OSError as e:
            self._log_error_once("event log rotation failed", e)

    def _open_log_file(self, day: str):
        """Open the daily log file and update ``events.jsonl`` symlink."""
        path = os.path.join(self.run_dir, f"events-{day}.jsonl")
        try:
            f = open(path, "a", buffering=1)
        except OSError as e:
            self._log_error_once("event writer failed to open", e)
            return None
        try:
            tmp_link = self.jsonl_path + ".tmp"
            try:
                os.remove(tmp_link)
            except FileNotFoundError:
                pass
            os.symlink(os.path.basename(path), tmp_link)
            os.replace(tmp_link, self.jsonl_path)
        except OSError:
            try:
                if os.path.islink(tmp_link):
                    os.unlink(tmp_link)
            except OSError:
                pass
            # Best effort; if symlinks are unsupported fall back to using the
            # plain filename.
            try:
                os.replace(path, self.jsonl_path)
                path = self.jsonl_path
                f.close()
                f = open(path, "a", buffering=1)
            except OSError as e:
                self._log_error_once("event writer failed to open", e)
                return None
        return f

    def _event(self, kind: str, **payload) -> None:
        evt = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "kind": kind,
            "git_hash": self._git_hash,
            "config": self._config,
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
        f = self._open_log_file(current_day)
        if f is None:
            return
        while not self._stop.is_set() or not self._q.empty():
            today = datetime.utcnow().strftime("%Y%m%d")
            if today != current_day:
                try:
                    f.close()
                except OSError:
                    pass
                current_day = today
                f = self._open_log_file(current_day)
                if f is None:
                    break
            try:
                evt = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                f.write(json.dumps(evt) + "\n")
            except OSError as e:
                self._log_error_once("event write failed", e)
        try:
            f.close()
        except OSError:
            pass

    def _write_heartbeat(self) -> None:
        hb = {
            "time": datetime.utcnow().isoformat() + "Z",
            "step": self._step,
            "last_ckpt": self._ckpt_path,
            "eta_seconds": self._eta_seconds,
            "git_hash": self._git_hash,
            "config": self._config,
        }
        try:
            with open(self.heartbeat_path, "w") as f:
                json.dump(hb, f)
        except OSError as e:
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
            except pynvml.NVMLError:
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

from dataclasses import dataclass
import csv


@dataclass
class CSVLogger:
    path: str
    newline: str = "\n"

    def open(self):
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["id","universe","grammar","As_lower","As_upper","validity","frequency","d_min"])
        return self

    def write_row(self, *row):
        self._writer.writerow(row)

    def close(self):
        if getattr(self, "_fh", None):
            self._fh.close()
