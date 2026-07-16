from __future__ import annotations

from dataclasses import dataclass, fields
import time
from typing import Optional
import numpy as np

from .memory import ProcessMemoryMonitor, null_memory_stats
from .problems import Problem2D
from .quadrature import Quadrature2D, upper_hemisphere_product_quadrature, basis_2d
from .quantization import WeightedMomentQuantizer, QuantizationInfo
from .transfer import angular_moments, transfer_angular_field


_ARRAY_FIELDS = {"x", "y", "ox", "oy", "oz", "w", "ox_c", "oy_c", "oz_c", "w_c",
                 "phi", "jx", "jy", "phi_u", "phi_c", "psi", "psi_c"}


@dataclass
class Result2D:
    problem: str
    method: str
    compression: str
    preserve: str
    bits: int
    nx: int
    ny: int
    angle_order: int
    uncollided_angle_order: int
    collided_angle_order: int
    n_angles: int
    n_uncollided_angles: int
    n_collided_angles: int
    n_steps: int
    dt: float
    iterations_total: int
    iterations_max: int
    final_delta: float
    storage_ratio_collided: float
    elapsed_sec: float
    complexity_updates: int
    min_scalar_flux: float
    max_scalar_flux: float
    min_angular_flux: float
    max_angular_flux: float
    mass: float
    transfer_max_moment_error: float
    positivity_limiter: str
    quantization: QuantizationInfo
    memory: dict[str, object]
    x: np.ndarray
    y: np.ndarray
    ox: np.ndarray
    oy: np.ndarray
    oz: np.ndarray
    w: np.ndarray
    ox_c: np.ndarray
    oy_c: np.ndarray
    oz_c: np.ndarray
    w_c: np.ndarray
    phi: np.ndarray
    jx: np.ndarray
    jy: np.ndarray
    phi_u: np.ndarray
    phi_c: np.ndarray
    psi: np.ndarray
    psi_c: np.ndarray

    def summary(self) -> dict[str, object]:
        d: dict[str, object] = {}
        for f in fields(self):
            if f.name in _ARRAY_FIELDS or f.name in {"quantization", "memory"}:
                continue
            d[f.name] = getattr(self, f.name)
        d.update({f"quantization_{k}": v for k, v in self.quantization.as_dict().items()})
        d.update(self.memory)
        return d


def moments_2d(psi: np.ndarray, q: Quadrature2D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.tensordot(psi, q.w, axes=([2], [0]))
    jx = np.tensordot(psi, q.w * q.ox, axes=([2], [0]))
    jy = np.tensordot(psi, q.w * q.oy, axes=([2], [0]))
    return phi, jx, jy


def boundary_arrays(problem: Problem2D, q: Quadrature2D, homogeneous: bool = False):
    left = np.zeros((problem.ny, q.n_angles), dtype=float)
    right = np.zeros_like(left)
    bottom = np.zeros((problem.nx, q.n_angles), dtype=float)
    top = np.zeros_like(bottom)
    if not homogeneous:
        left[:, q.ox > 0.0] = problem.left_inflow
        right[:, q.ox < 0.0] = problem.right_inflow
        bottom[:, q.oy > 0.0] = problem.bottom_inflow
        top[:, q.oy < 0.0] = problem.top_inflow
    return left, right, bottom, top


def sweep_2d(angular_source: np.ndarray | None, isotropic_source: np.ndarray | None,
             q: Quadrature2D, sigma_hat: np.ndarray, dx: float, dy: float,
             left: np.ndarray, right: np.ndarray, bottom: np.ndarray, top: np.ndarray,
             angular_scale: float = 1.0) -> np.ndarray:
    """First-order upwind sweep without materializing an isotropic 3D source."""
    nx, ny = sigma_hat.shape
    if angular_source is not None and angular_source.shape != (nx, ny, q.n_angles):
        raise ValueError("angular_source shape mismatch")
    if isotropic_source is not None and isotropic_source.shape != (nx, ny):
        raise ValueError("isotropic_source shape mismatch")
    out = np.zeros((nx, ny, q.n_angles), dtype=float)
    for m in range(q.n_angles):
        mx, my = float(q.ox[m]), float(q.oy[m])
        ax, ay = abs(mx) / dx, abs(my) / dy
        irange = range(nx) if mx > 0.0 else range(nx - 1, -1, -1)
        jrange = range(ny) if my > 0.0 else range(ny - 1, -1, -1)
        denom = sigma_hat + ax + ay
        for i in irange:
            for j in jrange:
                upx = (left[j, m] if i == 0 else out[i - 1, j, m]) if mx > 0.0 else (
                      right[j, m] if i == nx - 1 else out[i + 1, j, m])
                upy = (bottom[i, m] if j == 0 else out[i, j - 1, m]) if my > 0.0 else (
                      top[i, m] if j == ny - 1 else out[i, j + 1, m])
                src = 0.0
                if angular_source is not None:
                    src += angular_scale * float(angular_source[i, j, m])
                if isotropic_source is not None:
                    src += float(isotropic_source[i, j])
                out[i, j, m] = (src + ax * upx + ay * upy) / denom[i, j]
    return out


def _iterate_monolithic(problem: Problem2D, q: Quadrature2D, sigma_hat: np.ndarray,
                        history: np.ndarray, initial: np.ndarray, dt: float,
                        max_iters: int, tol: float):
    left, right, bottom, top = boundary_arrays(problem, q)
    psi = initial.copy()
    delta = np.inf
    for it in range(1, max_iters + 1):
        phi, _, _ = moments_2d(psi, q)
        isotropic = problem.q + problem.sigma_s * phi
        new = sweep_2d(history, isotropic, q, sigma_hat, problem.dx, problem.dy,
                       left, right, bottom, top, angular_scale=1.0 / dt)
        new_phi, _, _ = moments_2d(new, q)
        delta = float(np.linalg.norm(new_phi - phi) / (np.linalg.norm(new_phi) + 1e-300))
        psi = new
        if delta < tol:
            break
    return psi, it, delta


def solve_2d(problem: Problem2D, method: str = "uc", angle_order: int = 32,
             uncollided_angle_order: Optional[int] = None,
             collided_angle_order: Optional[int] = None,
             compression: str = "none", bits: int = 5,
             preserve: str = "phi_current", max_source_iters: int = 300,
             tol: float = 1e-8, final_time: Optional[float] = None,
             cfl: Optional[float] = None, seed: int = 12345,
             save_angular: bool = False, positivity_limiter: str = "none",
             memory_interval_ms: float = 10.0) -> Result2D:
    if method not in {"monolithic", "uc"}:
        raise ValueError("method must be monolithic or uc")
    if compression not in {"none", "raw", "moment"}:
        raise ValueError("compression must be none, raw, or moment")
    if positivity_limiter not in {"none", "repair"}:
        raise ValueError("positivity_limiter must be none or repair")
    Nu = int(angle_order if uncollided_angle_order is None else uncollided_angle_order)
    Nc = int(angle_order if collided_angle_order is None else collided_angle_order)
    if method == "monolithic":
        Nu = Nc = int(angle_order)
    q_hi = upper_hemisphere_product_quadrature(Nu)
    q_lo = upper_hemisphere_product_quadrature(Nc)
    tfinal = float(problem.final_time if final_time is None else final_time)
    cfl_value = float(problem.cfl if cfl is None else cfl)
    n_steps = max(1, int(np.ceil(tfinal / (cfl_value * min(problem.dx, problem.dy)))))
    dt = tfinal / n_steps
    sigma_hat = problem.sigma_t + 1.0 / dt

    quantizer = WeightedMomentQuantizer(q_lo.w, basis_2d(q_lo, preserve), seed=seed)
    qmode = "none" if compression == "none" else compression
    storage_ratio = 1.0 if compression == "none" else quantizer.storage_ratio(bits, qmode)
    qdiag = QuantizationInfo()
    transfer_error = 0.0
    total_iters = 0
    max_iters_seen = 0
    final_delta = np.inf
    complexity = 0

    monitor = ProcessMemoryMonitor(interval_s=float(memory_interval_ms) / 1000.0)
    with monitor:
        tic = time.perf_counter()
        psi_hi = np.repeat(problem.phi0[:, :, None], q_hi.n_angles, axis=2)
        phi_u_final = np.zeros_like(problem.phi0)
        phi_c_final = np.zeros_like(problem.phi0)
        psi_c_final = np.empty((0,), dtype=float)
        if method == "monolithic":
            for _ in range(n_steps):
                history = psi_hi
                psi_hi = np.empty((0,), dtype=float)
                new, its, final_delta = _iterate_monolithic(
                    problem, q_hi, sigma_hat, history, history, dt, max_source_iters, tol)
                del history
                psi_hi = new
                total_iters += its
                max_iters_seen = max(max_iters_seen, its)
                complexity += problem.nx * problem.ny * q_hi.n_angles * its
            phi, jx, jy = moments_2d(psi_hi, q_hi)
            phi_c_final = phi.copy()
        else:
            left_hi, right_hi, bottom_hi, top_hi = boundary_arrays(problem, q_hi)
            left_lo, right_lo, bottom_lo, top_lo = boundary_arrays(problem, q_lo, homogeneous=True)
            for _ in range(n_steps):
                history = psi_hi
                psi_hi = np.empty((0,), dtype=float)
                psi_u = sweep_2d(history, problem.q, q_hi, sigma_hat,
                                 problem.dx, problem.dy, left_hi, right_hi, bottom_hi, top_hi,
                                 angular_scale=1.0 / dt)
                del history
                phi_u, jx_u, jy_u = moments_2d(psi_u, q_hi)
                psi_c = np.zeros((problem.nx, problem.ny, q_lo.n_angles), dtype=float)
                delta = np.inf
                for its in range(1, max_source_iters + 1):
                    phi_c, _, _ = moments_2d(psi_c, q_lo)
                    new_c = sweep_2d(None, problem.sigma_s * (phi_u + phi_c), q_lo, sigma_hat,
                                     problem.dx, problem.dy, left_lo, right_lo, bottom_lo, top_lo)
                    if compression != "none":
                        target_moments = quantizer.moments(new_c)
                        new_c, info = quantizer.compress(new_c, bits=bits, mode=qmode, diagnostics=True)
                        if positivity_limiter == "repair":
                            new_c, me, neg, disp = quantizer.repair_positivity_and_moments(new_c, target_moments)
                            info.max_moment_error = max(info.max_moment_error, me)
                            info.max_post_repair_negative_part_norm = max(info.max_post_repair_negative_part_norm, neg)
                            info.max_repair_displacement = max(info.max_repair_displacement, disp)
                        qdiag.update(info)
                    new_phi_c, _, _ = moments_2d(new_c, q_lo)
                    delta = float(np.linalg.norm(new_phi_c - phi_c) /
                                  (np.linalg.norm(new_phi_c) + 1e-300))
                    psi_c = new_c
                    if delta < tol:
                        break
                final_delta = delta
                total_iters += its
                max_iters_seen = max(max_iters_seen, its)
                complexity += problem.nx * problem.ny * (q_hi.n_angles + its * q_lo.n_angles)
                psi_c_hi = transfer_angular_field(psi_c, q_lo, q_hi, preserve=preserve)
                source_moments = angular_moments(psi_c, q_lo, preserve)
                target_moments = angular_moments(psi_c_hi, q_hi, preserve)
                transfer_error = max(transfer_error, float(np.max(np.linalg.norm(source_moments - target_moments, axis=1))))
                psi_hi = psi_u
                psi_hi += psi_c_hi
                del psi_c_hi
                phi_c_final, jx_c, jy_c = moments_2d(psi_c, q_lo)
                phi_u_final = phi_u
                phi = phi_u + phi_c_final
                jx = jx_u + jx_c
                jy = jy_u + jy_c
                psi_c_final = psi_c
        elapsed = time.perf_counter() - tic
    mem = (monitor.stats or null_memory_stats()).as_dict()

    if method == "monolithic":
        phi, jx, jy = moments_2d(psi_hi, q_hi)
    mass = float(problem.dx * problem.dy * np.sum(phi))
    return Result2D(
        problem=problem.name, method=method, compression=compression, preserve=preserve,
        bits=int(bits), nx=problem.nx, ny=problem.ny, angle_order=Nc,
        uncollided_angle_order=Nu, collided_angle_order=Nc,
        n_angles=q_lo.n_angles, n_uncollided_angles=q_hi.n_angles,
        n_collided_angles=q_lo.n_angles, n_steps=n_steps, dt=float(dt),
        iterations_total=int(total_iters), iterations_max=int(max_iters_seen),
        final_delta=float(final_delta), storage_ratio_collided=float(storage_ratio),
        elapsed_sec=float(elapsed), complexity_updates=int(complexity),
        min_scalar_flux=float(np.min(phi)), max_scalar_flux=float(np.max(phi)),
        min_angular_flux=float(np.min(psi_hi)), max_angular_flux=float(np.max(psi_hi)),
        mass=mass, transfer_max_moment_error=float(transfer_error),
        positivity_limiter=positivity_limiter, quantization=qdiag, memory=mem,
        x=problem.x, y=problem.y, ox=q_hi.ox, oy=q_hi.oy, oz=q_hi.oz, w=q_hi.w,
        ox_c=q_lo.ox, oy_c=q_lo.oy, oz_c=q_lo.oz, w_c=q_lo.w,
        phi=phi, jx=jx, jy=jy, phi_u=phi_u_final, phi_c=phi_c_final,
        psi=psi_hi if save_angular else np.empty((0,), dtype=float),
        psi_c=psi_c_final if save_angular else np.empty((0,), dtype=float),
    )
