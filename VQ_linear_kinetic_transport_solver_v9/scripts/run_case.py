from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from tqjcp.problems import make_problem_1d, make_problem_2d
from tqjcp.solver1d import solve_1d
from tqjcp.solver2d import solve_2d


def write_json(path: Path, obj: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run one transport case in a fresh process")
    p.add_argument("--dim", type=int, choices=[1, 2], default=2)
    p.add_argument("--problem", default="line_source")
    p.add_argument("--nx", type=int, default=51)
    p.add_argument("--ny", type=int)
    p.add_argument("--method", choices=["monolithic", "uc"], default="uc")
    p.add_argument("--angle-order", type=int, default=32)
    p.add_argument("--uncollided-order", type=int)
    p.add_argument("--collided-order", type=int)
    p.add_argument("--compression", choices=["none", "raw", "moment"], default="none")
    p.add_argument("--bits", type=int, default=5)
    p.add_argument("--preserve", choices=["none", "phi", "phi_current"], default="phi_current")
    p.add_argument("--positivity-limiter", choices=["none", "repair"], default="repair")
    p.add_argument("--max-source-iters", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--final-time", type=float)
    p.add_argument("--cfl", type=float)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--memory-interval-ms", type=float, default=10.0)
    p.add_argument("--save-angular", action="store_true")
    p.add_argument("--tag", default="case")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.dim == 1:
        problem = make_problem_1d(args.problem, args.nx, final_time=args.final_time)
        result = solve_1d(
            problem, method=args.method, angle_order=args.angle_order,
            compression=args.compression, bits=args.bits, preserve=args.preserve,
            max_source_iters=args.max_source_iters, tol=args.tol, seed=args.seed,
            final_time=args.final_time, cfl=args.cfl, save_angular=args.save_angular,
            positivity_limiter=args.positivity_limiter,
            memory_interval_ms=args.memory_interval_ms,
        )
        summary = result.summary()
        arrays = dict(x=result.x, mu=result.mu, w=result.w, phi=result.phi,
                      current=result.current, phi_u=result.phi_u, phi_c=result.phi_c)
        if args.save_angular:
            arrays["psi"] = result.psi
    else:
        problem = make_problem_2d(args.problem, args.nx, args.ny, final_time=args.final_time)
        result = solve_2d(
            problem, method=args.method, angle_order=args.angle_order,
            uncollided_angle_order=args.uncollided_order,
            collided_angle_order=args.collided_order,
            compression=args.compression, bits=args.bits, preserve=args.preserve,
            max_source_iters=args.max_source_iters, tol=args.tol, seed=args.seed,
            final_time=args.final_time, cfl=args.cfl, save_angular=args.save_angular,
            positivity_limiter=args.positivity_limiter,
            memory_interval_ms=args.memory_interval_ms,
        )
        summary = result.summary()
        arrays = dict(
            x=result.x, y=result.y,
            ox=result.ox, oy=result.oy, oz=result.oz, w=result.w,
            ox_c=result.ox_c, oy_c=result.oy_c, oz_c=result.oz_c, w_c=result.w_c,
            phi=result.phi, jx=result.jx, jy=result.jy,
            phi_u=result.phi_u, phi_c=result.phi_c,
        )
        if args.save_angular:
            arrays["psi"] = result.psi
            arrays["psi_c"] = result.psi_c

    summary["tag"] = args.tag
    summary["out_dir"] = str(args.out)
    summary["fresh_process_memory_measurement"] = True
    write_json(args.out / "summary.json", summary)
    np.savez_compressed(args.out / "solution.npz", **arrays)
    print(json.dumps(summary, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
