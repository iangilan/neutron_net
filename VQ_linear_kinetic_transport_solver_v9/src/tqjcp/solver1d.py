from __future__ import annotations

from dataclasses import dataclass, fields
import time
import numpy as np

from .memory import ProcessMemoryMonitor, null_memory_stats
from .problems import Problem1D
from .quadrature import gauss_legendre_1d, basis_1d
from .quantization import WeightedMomentQuantizer, QuantizationInfo

_ARRAY_FIELDS = {"x", "mu", "w", "phi", "current", "phi_u", "phi_c", "psi"}


@dataclass
class Result1D:
    problem: str
    method: str
    compression: str
    preserve: str
    bits: int
    nx: int
    angle_order: int
    n_angles: int
    n_steps: int
    dt: float
    iterations_total: int
    iterations_max: int
    final_delta: float
    storage_ratio_collided: float
    elapsed_sec: float
    min_scalar_flux: float
    max_scalar_flux: float
    min_angular_flux: float
    max_angular_flux: float
    mass: float
    quantization: QuantizationInfo
    memory: dict[str, object]
    x: np.ndarray
    mu: np.ndarray
    w: np.ndarray
    phi: np.ndarray
    current: np.ndarray
    phi_u: np.ndarray
    phi_c: np.ndarray
    psi: np.ndarray

    def summary(self) -> dict[str, object]:
        d: dict[str, object] = {}
        for f in fields(self):
            if f.name in _ARRAY_FIELDS or f.name in {"quantization", "memory"}:
                continue
            d[f.name] = getattr(self, f.name)
        d.update({f"quantization_{k}": v for k, v in self.quantization.as_dict().items()})
        d.update(self.memory)
        return d


def moments_1d(psi, mu, w):
    return psi @ w, psi @ (w * mu)


def sweep_1d(angular_source, isotropic_source, mu, sigma_hat, dx, left, right,
             angular_scale: float = 1.0):
    nx = sigma_hat.size
    out = np.zeros((nx, mu.size), dtype=float)
    for m, mm in enumerate(mu):
        if mm > 0.0:
            incoming = float(left[m]); a = float(mm) / dx
            for i in range(nx):
                src = (0.0 if angular_source is None else angular_scale * angular_source[i, m]) + (0.0 if isotropic_source is None else isotropic_source[i])
                out[i, m] = (src + a * incoming) / (sigma_hat[i] + a)
                incoming = out[i, m]
        else:
            incoming = float(right[m]); a = -float(mm) / dx
            for i in range(nx - 1, -1, -1):
                src = (0.0 if angular_source is None else angular_scale * angular_source[i, m]) + (0.0 if isotropic_source is None else isotropic_source[i])
                out[i, m] = (src + a * incoming) / (sigma_hat[i] + a)
                incoming = out[i, m]
    return out


def solve_1d(problem: Problem1D, method: str = "uc", angle_order: int = 63,
             compression: str = "none", bits: int = 5, preserve: str = "phi_current",
             max_source_iters: int = 300, tol: float = 1e-8, seed: int = 12345,
             final_time: float | None = None, cfl: float | None = None,
             save_angular: bool = True, positivity_limiter: str = "none",
             memory_interval_ms: float = 10.0) -> Result1D:
    if method not in {"monolithic", "uc"}:
        raise ValueError("method must be monolithic or uc")
    q = gauss_legendre_1d(angle_order)
    tf = problem.final_time if final_time is None else float(final_time)
    cfl_value = problem.cfl if cfl is None else float(cfl)
    n_steps = max(1, int(np.ceil(tf / (cfl_value * problem.dx))))
    dt = tf / n_steps
    sigma_hat = problem.sigma_t + 1.0 / dt
    left = np.where(q.mu > 0.0, problem.left_inflow, 0.0)
    right = np.where(q.mu < 0.0, problem.right_inflow, 0.0)
    zero = np.zeros_like(left)
    quantizer = WeightedMomentQuantizer(q.w, basis_1d(q, preserve), seed=seed)
    qmode = "none" if compression == "none" else compression
    ratio = 1.0 if compression == "none" else quantizer.storage_ratio(bits, qmode)
    qdiag = QuantizationInfo()
    monitor = ProcessMemoryMonitor(interval_s=memory_interval_ms / 1000.0)
    total_iters = 0; max_iters_seen = 0; delta = np.inf
    with monitor:
        tic = time.perf_counter()
        psi = np.repeat(problem.phi0[:, None], q.n_angles, axis=1)
        phi_u_final = np.zeros(problem.nx)
        phi_c_final = np.zeros(problem.nx)
        for _ in range(n_steps):
            history = psi
            if method == "monolithic":
                psi = history.copy()
                for its in range(1, max_source_iters + 1):
                    phi, _ = moments_1d(psi, q.mu, q.w)
                    new = sweep_1d(history, problem.q + problem.sigma_s * phi,
                                   q.mu, sigma_hat, problem.dx, left, right, angular_scale=1.0 / dt)
                    new_phi, _ = moments_1d(new, q.mu, q.w)
                    delta = float(np.linalg.norm(new_phi - phi) / (np.linalg.norm(new_phi) + 1e-300))
                    psi = new
                    if delta < tol: break
            else:
                psi_u = sweep_1d(history, problem.q, q.mu, sigma_hat, problem.dx, left, right,
                                 angular_scale=1.0 / dt)
                phi_u, _ = moments_1d(psi_u, q.mu, q.w)
                psi_c = np.zeros_like(psi_u)
                for its in range(1, max_source_iters + 1):
                    phi_c, _ = moments_1d(psi_c, q.mu, q.w)
                    new_c = sweep_1d(None, problem.sigma_s * (phi_u + phi_c),
                                     q.mu, sigma_hat, problem.dx, zero, zero)
                    if compression != "none":
                        target = quantizer.moments(new_c)
                        new_c, info = quantizer.compress(new_c, bits, qmode, diagnostics=True)
                        if positivity_limiter == "repair":
                            new_c, me, neg, disp = quantizer.repair_positivity_and_moments(new_c, target)
                            info.max_moment_error = max(info.max_moment_error, me)
                            info.max_post_repair_negative_part_norm = max(info.max_post_repair_negative_part_norm, neg)
                            info.max_repair_displacement = max(info.max_repair_displacement, disp)
                        qdiag.update(info)
                    new_phi, _ = moments_1d(new_c, q.mu, q.w)
                    delta = float(np.linalg.norm(new_phi - phi_c) / (np.linalg.norm(new_phi) + 1e-300))
                    psi_c = new_c
                    if delta < tol: break
                psi = psi_u + psi_c
                phi_u_final = phi_u
                phi_c_final, _ = moments_1d(psi_c, q.mu, q.w)
            total_iters += its; max_iters_seen = max(max_iters_seen, its)
        elapsed = time.perf_counter() - tic
    phi, current = moments_1d(psi, q.mu, q.w)
    mem = (monitor.stats or null_memory_stats()).as_dict()
    return Result1D(problem.name, method, compression, preserve, bits, problem.nx,
                    angle_order, q.n_angles, n_steps, dt, total_iters, max_iters_seen,
                    float(delta), ratio, elapsed, float(phi.min()), float(phi.max()),
                    float(psi.min()), float(psi.max()), float(problem.dx * phi.sum()),
                    qdiag, mem, problem.x, q.mu, q.w, phi, current, phi_u_final,
                    phi_c_final, psi if save_angular else np.empty((0,)))
