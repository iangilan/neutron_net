from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
CMAP = "viridis"


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel_l2(a: np.ndarray, b: np.ndarray, cell_measure: float) -> float:
    return float(np.sqrt(cell_measure * np.sum((a - b) ** 2)) /
                 (np.sqrt(cell_measure * np.sum(b ** 2)) + 1e-300))


def load_case(path: Path):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    solution = np.load(path / "solution.npz")
    return summary, solution


def run_case(out: Path, *, nx: int, method: str, Nu: int, Nc: int,
             compression: str, bits: int, max_source_iters: int, tol: float,
             final_time: float | None, skip_existing: bool, tag: str) -> dict[str, object]:
    if skip_existing and (out / "summary.json").exists() and (out / "solution.npz").exists():
        return json.loads((out / "summary.json").read_text(encoding="utf-8"))
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_case.py"),
        "--dim", "2", "--problem", "line_source", "--nx", str(nx),
        "--method", method, "--angle-order", str(Nu),
        "--uncollided-order", str(Nu), "--collided-order", str(Nc),
        "--compression", compression, "--bits", str(bits),
        "--preserve", "phi_current", "--positivity-limiter", "repair",
        "--max-source-iters", str(max_source_iters), "--tol", str(tol),
        "--memory-interval-ms", "5", "--seed", "12345",
        "--tag", tag, "--out", str(out),
    ]
    if final_time is not None:
        cmd.extend(["--final-time", str(final_time)])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce the v9 mixed-order line-source study")
    p.add_argument("--preset", choices=["smoke", "paper"], default="paper")
    p.add_argument("--out", type=Path, default=Path("results/v9_mixed_order"))
    p.add_argument("--nx", type=int)
    p.add_argument("--uncollided-order", type=int)
    p.add_argument("--collided-orders", default="8,16")
    p.add_argument("--bits", default="5,8")
    p.add_argument("--max-source-iters", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    if args.preset == "smoke":
        nx = 11 if args.nx is None else args.nx
        Nu = 4 if args.uncollided_order is None else args.uncollided_order
        Ncs = [2]
        bits_list = [5]
        final_time = 0.05
    else:
        nx = 201 if args.nx is None else args.nx
        Nu = 32 if args.uncollided_order is None else args.uncollided_order
        Ncs = parse_int_list(args.collided_orders)
        bits_list = parse_int_list(args.bits)
        final_time = None
    if any(nc > Nu for nc in Ncs):
        raise ValueError("collided order must not exceed uncollided order")

    raw = ensure(args.out / "raw")
    data_dir = ensure(args.out / "data")
    fig_dir = ensure(args.out / "figures")
    summaries: list[dict[str, object]] = []

    ref_dir = raw / f"line_source_nx{nx}_monolithic_N{Nu}"
    summaries.append(run_case(ref_dir, nx=nx, method="monolithic", Nu=Nu, Nc=Nu,
                              compression="none", bits=0,
                              max_source_iters=args.max_source_iters, tol=args.tol,
                              final_time=final_time, skip_existing=args.skip_existing,
                              tag="monolithic_reference"))

    high_dir = raw / f"line_source_nx{nx}_split_Nu{Nu}_Nc{Nu}_none"
    summaries.append(run_case(high_dir, nx=nx, method="uc", Nu=Nu, Nc=Nu,
                              compression="none", bits=0,
                              max_source_iters=args.max_source_iters, tol=args.tol,
                              final_time=final_time, skip_existing=args.skip_existing,
                              tag="high_high_split"))

    for Nc in Ncs:
        base_dir = raw / f"line_source_nx{nx}_split_Nu{Nu}_Nc{Nc}_none"
        summaries.append(run_case(base_dir, nx=nx, method="uc", Nu=Nu, Nc=Nc,
                                  compression="none", bits=0,
                                  max_source_iters=args.max_source_iters, tol=args.tol,
                                  final_time=final_time, skip_existing=args.skip_existing,
                                  tag="mixed_uncompressed"))
        for b in bits_list:
            out = raw / f"line_source_nx{nx}_split_Nu{Nu}_Nc{Nc}_moment_b{b}"
            summaries.append(run_case(out, nx=nx, method="uc", Nu=Nu, Nc=Nc,
                                      compression="moment", bits=b,
                                      max_source_iters=args.max_source_iters, tol=args.tol,
                                      final_time=final_time, skip_existing=args.skip_existing,
                                      tag="mixed_compressed"))

    df = pd.DataFrame(summaries)
    df.to_csv(data_dir / "mixed_order_run_summaries.csv", index=False)

    ref_summary, ref_sol = load_case(ref_dir)
    high_summary, high_sol = load_case(high_dir)
    dx = float(high_sol["x"][1] - high_sol["x"][0]) if len(high_sol["x"]) > 1 else 1.0
    dy = float(high_sol["y"][1] - high_sol["y"][0]) if len(high_sol["y"]) > 1 else 1.0
    cell_measure = dx * dy
    rows = []
    for summary in summaries:
        path = Path(str(summary["out_dir"]))
        _, sol = load_case(path)
        jerr = math.sqrt(
            rel_l2(sol["jx"], high_sol["jx"], cell_measure) ** 2 +
            rel_l2(sol["jy"], high_sol["jy"], cell_measure) ** 2
        )
        rows.append({
            "tag": summary["tag"],
            "uncollided_order": int(summary["uncollided_angle_order"]),
            "collided_order": int(summary["collided_angle_order"]),
            "compression": summary["compression"],
            "bits": int(summary["bits"]),
            "rel_phi_vs_high_high": rel_l2(sol["phi"], high_sol["phi"], cell_measure),
            "rel_current_vs_high_high": jerr,
            "rel_phi_vs_monolithic": rel_l2(sol["phi"], ref_sol["phi"], cell_measure),
            "rss_peak_mib": float(summary.get("rss_peak_mib", np.nan)),
            "rss_delta_peak_mib": float(summary.get("rss_delta_peak_mib", np.nan)),
            "uss_peak_mib": float(summary.get("uss_peak_mib", np.nan)),
            "vmhwm_mib": float(summary.get("vmhwm_mib", np.nan)),
            "ru_maxrss_mib": float(summary.get("ru_maxrss_mib", np.nan)),
            "python_tracemalloc_peak_mib": float(summary.get("python_tracemalloc_peak_mib", np.nan)),
            "storage_ratio_collided": float(summary["storage_ratio_collided"]),
            "transfer_max_moment_error": float(summary.get("transfer_max_moment_error", 0.0)),
            "out_dir": str(path),
        })
    errors = pd.DataFrame(rows)
    errors.to_csv(data_dir / "mixed_order_errors.csv", index=False)

    # Representative field and error map for the smallest collided order at 5 bits.
    Nc0 = min(Ncs)
    mixed_base_dir = raw / f"line_source_nx{nx}_split_Nu{Nu}_Nc{Nc0}_none"
    mixed5_dir = raw / f"line_source_nx{nx}_split_Nu{Nu}_Nc{Nc0}_moment_b{bits_list[0]}"
    _, base_sol = load_case(mixed_base_dir)
    _, comp_sol = load_case(mixed5_dir)
    extent = [float(base_sol["x"][0]), float(base_sol["x"][-1]),
              float(base_sol["y"][0]), float(base_sol["y"][-1])]
    plt.figure(figsize=(5.4, 4.5))
    im = plt.imshow(base_sol["phi"].T, origin="lower", extent=extent, aspect="auto", cmap=CMAP)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"Line source: mixed split $N_u={Nu}$, $N_c={Nc0}$")
    plt.colorbar(im, label="scalar flux")
    plt.tight_layout(); plt.savefig(fig_dir / "mixed_order_scalar_flux.png", dpi=DPI); plt.close()

    errmap = np.abs(comp_sol["phi"] - base_sol["phi"]) / (np.max(np.abs(base_sol["phi"])) + 1e-300)
    plt.figure(figsize=(5.4, 4.5))
    im = plt.imshow(errmap.T, origin="lower", extent=extent, aspect="auto", cmap=CMAP)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"{bits_list[0]}-bit error: $N_u={Nu}$, $N_c={Nc0}$")
    plt.colorbar(im, label=r"$|\phi_b-\phi_0|/\max|\phi_0|$")
    plt.tight_layout(); plt.savefig(fig_dir / "mixed_order_5bit_error.png", dpi=DPI); plt.close()

    plot = errors[errors.tag.isin(["mixed_uncompressed", "mixed_compressed"])].copy()
    plt.figure(figsize=(6.4, 4.2))
    for (compression, bits), sub in plot.groupby(["compression", "bits"]):
        label = "uncompressed" if compression == "none" else f"{int(bits)}-bit"
        sub = sub.sort_values("collided_order")
        plt.semilogy(sub["collided_order"], sub["rel_phi_vs_high_high"], marker="o", label=label)
    plt.xlabel("collided angular order $N_c$")
    plt.ylabel("relative scalar-flux difference vs. $N_u=N_c$ split")
    plt.title(f"Mixed-order line source, $N_u={Nu}$")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "mixed_order_accuracy.png", dpi=DPI); plt.close()

    memplot = errors[errors.tag.isin(["high_high_split", "mixed_uncompressed", "mixed_compressed"])].copy()
    labels = []
    for _, r in memplot.iterrows():
        labels.append(f"Nc{int(r.collided_order)}-" + ("none" if r.compression == "none" else f"b{int(r.bits)}"))
    plt.figure(figsize=(8.0, 4.5))
    plt.bar(np.arange(len(memplot)), memplot["rss_peak_mib"])
    plt.xticks(np.arange(len(memplot)), labels, rotation=35, ha="right")
    plt.ylabel("fresh-process peak RSS (MiB)")
    plt.title("Process-level peak memory")
    plt.tight_layout(); plt.savefig(fig_dir / "mixed_order_peak_rss.png", dpi=DPI); plt.close()

    print(errors.to_string(index=False))
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
