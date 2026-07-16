from __future__ import annotations

from dataclasses import dataclass, asdict
import gc
import os
import resource
import sys
import threading
import time
import tracemalloc
from typing import Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

_MIB = 1024.0 * 1024.0


def _proc_status_mib(field: str) -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(field + ":"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0  # Linux reports KiB
    except OSError:
        pass
    return float("nan")


def _ru_maxrss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux/BSD report KiB. macOS reports bytes.
    if sys.platform == "darwin":
        return value / _MIB
    return value / 1024.0


@dataclass
class MemoryStats:
    rss_start_mib: float
    rss_peak_mib: float
    rss_delta_peak_mib: float
    uss_start_mib: float
    uss_peak_mib: float
    uss_delta_peak_mib: float
    vmhwm_mib: float
    ru_maxrss_mib: float
    python_tracemalloc_peak_mib: float
    memory_samples: int
    memory_sample_interval_ms: float
    memory_measurement_method: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ProcessMemoryMonitor:
    """Sample process RSS/USS while a solve is active.

    Use one fresh process per benchmark case. RSS includes NumPy/native
    allocations and is the primary peak-memory metric. tracemalloc is recorded
    separately because it does not represent total process memory.
    """

    def __init__(self, interval_s: float = 0.01, trace_python: bool = True):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s = float(interval_s)
        self.trace_python = bool(trace_python)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = None
        self._rss_start = 0.0
        self._uss_start = float("nan")
        self._rss_peak = 0.0
        self._uss_peak = float("nan")
        self._samples = 0
        self.stats: Optional[MemoryStats] = None

    def _read(self) -> tuple[float, float]:
        if self._process is not None:
            try:
                rss = float(self._process.memory_info().rss) / _MIB
                try:
                    uss = float(self._process.memory_full_info().uss) / _MIB
                except Exception:
                    uss = float("nan")
                return rss, uss
            except Exception:
                pass
        rss = _proc_status_mib("VmRSS")
        return rss, float("nan")

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            rss, uss = self._read()
            if rss == rss:
                self._rss_peak = max(self._rss_peak, rss)
            if uss == uss:
                if self._uss_peak != self._uss_peak:
                    self._uss_peak = uss
                else:
                    self._uss_peak = max(self._uss_peak, uss)
            self._samples += 1

    def __enter__(self) -> "ProcessMemoryMonitor":
        gc.collect()
        if psutil is not None:
            self._process = psutil.Process(os.getpid())
        self._rss_start, self._uss_start = self._read()
        self._rss_peak = self._rss_start
        self._uss_peak = self._uss_start
        self._samples = 1
        if self.trace_python:
            tracemalloc.start()
        self._thread = threading.Thread(target=self._sample, name="rss-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5.0 * self.interval_s))
        rss, uss = self._read()
        if rss == rss:
            self._rss_peak = max(self._rss_peak, rss)
        if uss == uss:
            if self._uss_peak != self._uss_peak:
                self._uss_peak = uss
            else:
                self._uss_peak = max(self._uss_peak, uss)
        py_peak = float("nan")
        if self.trace_python and tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            py_peak = float(peak) / _MIB
            tracemalloc.stop()
        methods = ["sampled_rss"]
        if self._uss_start == self._uss_start:
            methods.append("sampled_uss")
        vmhwm = _proc_status_mib("VmHWM")
        if vmhwm == vmhwm:
            methods.append("linux_vmhwm")
        methods.append("resource_ru_maxrss")
        if py_peak == py_peak:
            methods.append("python_tracemalloc_separate")
        uss_delta = (self._uss_peak - self._uss_start
                     if self._uss_peak == self._uss_peak and self._uss_start == self._uss_start
                     else float("nan"))
        self.stats = MemoryStats(
            rss_start_mib=float(self._rss_start),
            rss_peak_mib=float(self._rss_peak),
            rss_delta_peak_mib=float(max(0.0, self._rss_peak - self._rss_start)),
            uss_start_mib=float(self._uss_start),
            uss_peak_mib=float(self._uss_peak),
            uss_delta_peak_mib=float(max(0.0, uss_delta)) if uss_delta == uss_delta else float("nan"),
            vmhwm_mib=float(vmhwm),
            ru_maxrss_mib=float(_ru_maxrss_mib()),
            python_tracemalloc_peak_mib=float(py_peak),
            memory_samples=int(self._samples),
            memory_sample_interval_ms=1000.0 * self.interval_s,
            memory_measurement_method="+".join(methods),
        )


def null_memory_stats() -> MemoryStats:
    return MemoryStats(*(float("nan"),) * 9, 0, 0.0, "disabled")
