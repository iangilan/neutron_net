# Changes from v8 to v9

## Mixed-order 2D collision split

- Added independent `uncollided_angle_order` and `collided_angle_order` parameters.
- Added high-to-low moment coupling and low-to-high angular-history transfer.
- The transfer preserves scalar flux and current to roundoff.
- Added a dedicated line-source mixed-order study with `N_u=32` and `N_c in {8,16}`.
- Quantization remains confined to the collided angular field.

## Correct memory measurement

- Replaced the ambiguous `peak_memory_mb` interpretation with process-level RSS/USS sampling.
- Added Linux `VmHWM` and portable `resource.ru_maxrss` reporting.
- Retained `tracemalloc` only as a separately named Python-allocation metric.
- Every benchmark case is launched in a fresh subprocess.
- All binary units are reported as MiB.

## Quantizer implementation

- Replaced the dense Hadamard matrix multiplication with a chunked fast Walsh-Hadamard transform.
- Preserved deterministic random-sign rotation and exact protected-moment correction.

## Tests

- Moment-preserving angular transfer test.
- Equal-order transfer identity test.
- Mixed-order line-source smoke test.
- Process-memory monitor allocation test.
