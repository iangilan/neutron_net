# VQ Linear Kinetic Transport Solver v9

This package implements the two revisions requested after the discussion with Ryan McClarren:

1. **Mixed angular resolution after collision splitting.** The 2D line-source study uses a high angular order `N_u` for the uncollided equation and a lower angular order `N_c` for the collided equation. Moment-preserving randomized quantization is applied only to the collided angular field.
2. **Correct process-level peak-memory measurement.** Every production case is launched in a fresh subprocess and sampled with `psutil`. The package records peak RSS, peak USS when available, peak-minus-baseline RSS, Linux `ru_maxrss`, and Python-heap peak separately.

The line-source problem is the primary mixed-order benchmark because its uncollided component is strongly directional, while scattering makes its collided component a more plausible low-order angular target. The high-order uncollided moments are computed from the exact scattering-free translation of the mollified initial line source and cached for reuse.

## Error separation

The driver reports three different errors and does not combine them:

- **Mixed-order error:** uncompressed `(N_u, N_c)` versus uncompressed `(N_u, N_u)`.
- **Quantization increment:** quantized `(N_u, N_c)` versus uncompressed `(N_u, N_c)`.
- **Total error:** quantized `(N_u, N_c)` versus uncompressed `(N_u, N_u)`.

This distinction is necessary because `N_u != N_c` introduces an angular-discretization approximation before quantization is applied.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Quick verification

```bash
make test
make smoke
```

The smoke target uses a small grid and writes all outputs under `results/smoke`.

## Full v9 line-source matrix

```bash
make full-v9
```

The default full matrix uses

```text
N_x = N_y = 201
N_u = 32
N_c in {4, 8, 16, 32}
bits in {none, 3, 5, 8}
```

The uncompressed `(N_u, N_c) = (32, 32)` case is the matched high-order reference. The uncompressed mixed-order cases isolate angular-order error. Quantized cases isolate the additional quantization perturbation.

The full study can be expensive. Parameters can be changed from the command line:

```bash
PYTHONPATH=src python scripts/reproduce_mixed_order_line_source.py \
  --preset custom \
  --out results/custom \
  --nx 101 \
  --uncollided-order 32 \
  --collided-orders 4 8 16 32 \
  --bits 3 5 8 \
  --n-steps 12 \
  --tol 1e-8 \
  --max-source-iters 200
```

## Memory measurements

Each numerical case is executed in a fresh child process. The following fields are stored in the case `summary.json` and aggregate CSV files:

- `rss_baseline_mib`
- `peak_rss_mib`
- `peak_rss_increment_mib`
- `peak_uss_mib`
- `final_rss_mib`
- `ru_maxrss_mib`
- `python_heap_peak_mib`
- `memory_sample_interval_ms`
- `memory_sample_count`

`peak_rss_mib` is the primary measured process-memory metric. `python_heap_peak_mib` is retained only as a diagnostic and must not be interpreted as total NumPy/process memory.

The analytical packed collided-field size is reported separately from measured RSS.

## Output files

A reproduction run stores:

```text
results/<preset>/
  cases/<case-name>/
    solution.npz
    summary.json
    memory.json
    run.log
  data/
    mixed_order_line_source_results.csv
    mixed_order_line_source_memory.csv
  figures/
    mixed_order_error_vs_nc.png
    quantization_increment_vs_nc.png
    peak_rss_vs_nc.png
    accuracy_memory_pareto.png
    representative_scalar_flux.png
    representative_error_map.png
  tables/
    mixed_order_summary.tex
  manifest.json
```

Figures are saved at 300 dpi. The field and error maps use the `jet` color map to remain consistent with the manuscript workflow requested by the author.

## Compatibility aliases

`make full-jcp` is an alias for `make full-v9`. The wrapper `scripts/reproduce_transport_benchmarks.py` forwards to the v9 mixed-order driver.

## Scope

The v9 mixed-order study currently focuses on the 2D line-source benchmark. The lattice problem is the recommended second study after the line-source results are established. The modified hohlraum should be tested last because its boundary inflow and strong material contrasts make error attribution less clean.
