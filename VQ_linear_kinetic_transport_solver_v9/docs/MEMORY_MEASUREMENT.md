# Memory measurement methodology

Version 8 used a single field named `peak_memory_mb`, which could be confused with process peak memory even when it was derived from Python allocation tracing. Version 9 records several distinct quantities.

## Primary metrics

- **Peak RSS**: sampled resident set size from `psutil.Process.memory_info().rss`. This includes NumPy arrays and other native allocations resident in memory.
- **Peak USS**: sampled unique set size from `memory_full_info().uss`, when supported. USS estimates memory private to the process.
- **VmHWM**: Linux process high-water mark from `/proc/self/status`.
- **ru_maxrss**: high-water mark from `resource.getrusage` with platform-dependent units converted to MiB.

## Secondary diagnostic

- **tracemalloc peak**: Python allocator tracing. This is useful for identifying Python-managed allocations, but it is not a substitute for RSS and is never labelled as total process memory.

## Fresh-process rule

Every benchmark configuration is executed through `scripts/run_case.py` in a fresh subprocess. This is necessary because `VmHWM` and `ru_maxrss` are process-lifetime high-water marks. Reusing one process across a matrix would contaminate later cases with earlier peaks.

## Recommended paper fields

Report `rss_peak_mib` and, where available, `uss_peak_mib`. Include `vmhwm_mib` or `ru_maxrss_mib` as a cross-check. Report `rss_delta_peak_mib` when the imported Python runtime and libraries form a large common baseline.

All units are binary mebibytes:

```text
1 MiB = 1024^2 bytes
```
