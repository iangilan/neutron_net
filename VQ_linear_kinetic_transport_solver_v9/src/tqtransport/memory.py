from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Mapping, Sequence

import psutil

_MIB = 1024.0 * 1024.0


@dataclass
class MemoryProfile:
    command: list[str]
    return_code: int
    elapsed_sec: float
    rss_baseline_mib: float
    peak_rss_mib: float
    peak_rss_increment_mib: float
    peak_uss_mib: float | None
    final_rss_mib: float
    ru_maxrss_mib: float | None
    python_heap_peak_mib: float | None
    memory_sample_interval_ms: float
    memory_sample_count: int
    memory_measurement_backend: str

    def as_dict(self) -> dict:
        return asdict(self)


def _process_tree_memory(pid: int) -> tuple[int, int | None]:
    try:
        root = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, None

    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    rss = 0
    uss = 0
    have_uss = False
    seen: set[int] = set()
    for proc in processes:
        if proc.pid in seen:
            continue
        seen.add(proc.pid)
        try:
            rss += int(proc.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        try:
            uss += int(proc.memory_full_info().uss)
            have_uss = True
        except (AttributeError, psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return rss, uss if have_uss else None


def _ru_maxrss_mib(before: resource.struct_rusage, after: resource.struct_rusage) -> float | None:
    value = max(0.0, float(after.ru_maxrss) - float(before.ru_maxrss))
    if value == 0.0:
        value = float(after.ru_maxrss)
    if value <= 0.0:
        return None
    if sys.platform == "darwin":
        return value / _MIB
    return value / 1024.0


def run_profiled_command(
    command: Sequence[str],
    *,
    output_json: str | Path | None = None,
    interval_ms: float = 10.0,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
) -> MemoryProfile:
    """Run a command in a fresh subprocess while sampling process-tree memory.

    Peak RSS is the primary metric. USS is recorded when the platform exposes
    it. Linux/macOS `ru_maxrss` is retained as an independent high-water mark.
    """

    cmd = [str(part) for part in command]
    if not cmd:
        raise ValueError("command must not be empty")
    interval = max(0.001, float(interval_ms) / 1000.0)

    stdout_target = None
    log_handle = None
    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("w", encoding="utf-8")
        stdout_target = log_handle

    child_env = os.environ.copy()
    if env is not None:
        child_env.update({str(k): str(v) for k, v in env.items()})

    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=None if cwd is None else str(cwd),
        env=child_env,
        stdout=stdout_target,
        stderr=subprocess.STDOUT if stdout_target is not None else None,
        text=True,
    )

    baseline_rss = 0
    peak_rss = 0
    peak_uss: int | None = None
    final_rss = 0
    samples = 0

    try:
        while True:
            rss, uss = _process_tree_memory(proc.pid)
            if samples == 0:
                baseline_rss = rss
            peak_rss = max(peak_rss, rss)
            final_rss = rss
            if uss is not None:
                peak_uss = uss if peak_uss is None else max(peak_uss, uss)
            samples += 1
            if proc.poll() is not None:
                break
            time.sleep(interval)
        return_code = int(proc.wait())
    finally:
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()

    elapsed = time.perf_counter() - start
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    ru_mib = _ru_maxrss_mib(usage_before, usage_after)

    python_heap_peak_mib: float | None = None
    if output_json is not None:
        sidecar = Path(output_json).with_suffix(".heap.json")
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                python_heap_peak_mib = float(payload.get("python_heap_peak_mib"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                python_heap_peak_mib = None

    profile = MemoryProfile(
        command=cmd,
        return_code=return_code,
        elapsed_sec=float(elapsed),
        rss_baseline_mib=float(baseline_rss / _MIB),
        peak_rss_mib=float(peak_rss / _MIB),
        peak_rss_increment_mib=float(max(0, peak_rss - baseline_rss) / _MIB),
        peak_uss_mib=None if peak_uss is None else float(peak_uss / _MIB),
        final_rss_mib=float(final_rss / _MIB),
        ru_maxrss_mib=ru_mib,
        python_heap_peak_mib=python_heap_peak_mib,
        memory_sample_interval_ms=float(interval * 1000.0),
        memory_sample_count=int(samples),
        memory_measurement_backend="psutil-process-tree-rss-uss+resource-ru_maxrss",
    )

    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return profile
