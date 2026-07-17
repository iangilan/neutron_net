# v9 changes

## Mixed angular orders

- Added separate `N_u` and `N_c` angular orders for the 2D line-source collision split.
- Added a matched high/high uncompressed reference.
- Added uncompressed mixed-order baselines.
- Added quantized mixed-order cases at 3, 5, and 8 bits.
- Added explicit decomposition of mixed-order error, quantization increment, and total error.
- Added a cached high-order uncollided moment history so all collided-order and bit-width cases use identical uncollided data.
- Added a true fixed-point residual diagnostic for the final collided solve.

## Memory measurements

- Replaced ambiguous `peak_memory_mb` reporting with process-tree RSS and USS sampling in a fresh subprocess.
- Added Linux `ru_maxrss` as an independent high-water-mark diagnostic.
- Retained `tracemalloc` only as `python_heap_peak_mib`.
- Separated measured process memory from analytical raw and packed angular-field representation sizes.
- Standardized binary memory units as MiB.

## Reproduction outputs

- Added CSV summaries, LaTeX table generation, command manifest, and 300 dpi figures.
- Added tests for quadrature, moment preservation, mixed-order execution, and process memory monitoring.
