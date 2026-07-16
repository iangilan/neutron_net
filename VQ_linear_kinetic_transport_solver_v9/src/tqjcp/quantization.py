from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def fwht(X: np.ndarray) -> np.ndarray:
    """Normalized Walsh-Hadamard transform along the last axis."""
    Y = np.asarray(X, dtype=float).copy()
    n = Y.shape[-1]
    if n & (n - 1):
        raise ValueError("FWHT dimension must be a power of two")
    h = 1
    while h < n:
        view = Y.reshape(Y.shape[:-1] + (n // (2 * h), 2, h))
        a = view[..., 0, :].copy()
        b = view[..., 1, :].copy()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        h *= 2
    Y /= np.sqrt(float(n))
    return Y


@dataclass
class QuantizationInfo:
    calls: int = 0
    max_weighted_error: float = 0.0
    max_bound: float = 0.0
    mean_scale: float = 0.0
    max_empirical_eta: float = 0.0
    mean_empirical_eta: float = 0.0
    max_moment_error: float = 0.0
    max_negative_part_norm: float = 0.0
    max_post_repair_negative_part_norm: float = 0.0
    max_repair_displacement: float = 0.0

    def update(self, other: "QuantizationInfo") -> None:
        n0, n1 = self.calls, other.calls
        total = n0 + n1
        if total:
            self.mean_scale = (n0 * self.mean_scale + n1 * other.mean_scale) / total
            self.mean_empirical_eta = (n0 * self.mean_empirical_eta + n1 * other.mean_empirical_eta) / total
        self.calls = total
        for key in ["max_weighted_error", "max_bound", "max_empirical_eta",
                    "max_moment_error", "max_negative_part_norm",
                    "max_post_repair_negative_part_norm", "max_repair_displacement"]:
            setattr(self, key, max(float(getattr(self, key)), float(getattr(other, key))))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class WeightedMomentQuantizer:
    def __init__(self, weights: np.ndarray, basis: np.ndarray | None,
                 seed: int = 12345, max_chunk_bytes: int = 128 * 1024 * 1024):
        self.w = np.asarray(weights, dtype=float).reshape(-1)
        if np.any(self.w <= 0.0):
            raise ValueError("quadrature weights must be positive")
        self.sqrtw = np.sqrt(self.w)
        self.d = self.w.size
        self.D = next_power_of_two(self.d)
        self.max_chunk_bytes = int(max_chunk_bytes)
        rng = np.random.default_rng(int(seed))
        self.signs = rng.choice(np.array([-1.0, 1.0]), size=self.D)
        if basis is None or np.asarray(basis).size == 0:
            self.B = None
            self.WB = None
            self.A = None
            self.Ginv = None
        else:
            B = np.asarray(basis, dtype=float)
            if B.ndim == 1:
                B = B[:, None]
            if B.shape[0] != self.d:
                raise ValueError("basis must have one row per angle")
            G = B.T @ (self.w[:, None] * B)
            self.B = B
            self.WB = self.w[:, None] * B
            self.A = self.sqrtw[:, None] * B
            self.Ginv = np.linalg.inv(G)

    @property
    def n_preserved(self) -> int:
        return 0 if self.B is None else self.B.shape[1]

    def moments(self, V: np.ndarray) -> np.ndarray:
        rows = np.asarray(V, dtype=float).reshape(-1, self.d)
        if self.B is None:
            return np.zeros((rows.shape[0], 0))
        return rows @ self.WB

    def project(self, V: np.ndarray) -> np.ndarray:
        shape = np.asarray(V).shape
        rows = np.asarray(V, dtype=float).reshape(-1, self.d)
        if self.B is None:
            return np.zeros_like(rows).reshape(shape)
        coeff = (rows @ self.WB) @ self.Ginv.T
        return (coeff @ self.B.T).reshape(shape)

    def _chunk_rows(self, n_rows: int) -> int:
        bytes_per_row = max(1, 6 * self.D * 8)
        return max(1, min(n_rows, self.max_chunk_bytes // bytes_per_row))

    def _quantize(self, Y: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if bits < 2:
            raise ValueError("bits must be at least 2")
        padded = np.zeros((Y.shape[0], self.D), dtype=float)
        padded[:, :self.d] = Y
        U = fwht(padded * self.signs)
        qmax = float(2 ** (bits - 1) - 1)
        maxabs = np.max(np.abs(U), axis=1)
        scale = np.where(maxabs > 0.0, maxabs / qmax, 1.0)
        codes = np.clip(np.rint(U / scale[:, None]), -qmax, qmax)
        Uhat = codes * scale[:, None]
        Yhat = fwht(Uhat) * self.signs
        bound = 0.5 * np.sqrt(float(self.D)) * scale
        return Yhat[:, :self.d], scale, bound

    def compress(self, V: np.ndarray, bits: int, mode: str = "moment",
                 diagnostics: bool = False):
        shape = np.asarray(V).shape
        if shape[-1] != self.d:
            raise ValueError("last dimension must equal the angular dimension")
        rows = np.asarray(V, dtype=float).reshape(-1, self.d)
        if mode == "none":
            out = rows.copy().reshape(shape)
            return (out, QuantizationInfo()) if diagnostics else out
        out = np.empty_like(rows)
        errors, bounds, scales, etas = [], [], [], []
        moment_error = 0.0
        negative = 0.0
        chunk = self._chunk_rows(rows.shape[0])
        for start in range(0, rows.shape[0], chunk):
            stop = min(rows.shape[0], start + chunk)
            src = rows[start:stop]
            if mode == "raw" or self.B is None:
                target_moments = None
                P = np.zeros_like(src)
                Y = src * self.sqrtw
            elif mode == "moment":
                target_moments = self.moments(src)
                P = self.project(src).reshape(-1, self.d)
                Y = (src - P) * self.sqrtw
            else:
                raise ValueError(f"unknown mode {mode!r}")
            Yhat, scale, bound = self._quantize(Y, bits)
            if mode == "moment" and self.B is not None:
                leakage = (Yhat @ self.A) @ self.Ginv.T
                Yhat -= leakage @ self.A.T
            decoded = P + Yhat / self.sqrtw
            out[start:stop] = decoded
            err = np.linalg.norm(Yhat - Y, axis=1)
            residual_norm = np.linalg.norm(Y, axis=1)
            eta = err / (residual_norm + 1e-300)
            errors.append(err); bounds.append(bound); scales.append(scale); etas.append(eta)
            if target_moments is not None:
                moment_error = max(moment_error, float(np.max(np.linalg.norm(self.moments(decoded) - target_moments, axis=1))))
            negative = max(negative, float(np.max(np.linalg.norm(np.minimum(decoded, 0.0) * self.sqrtw, axis=1))))
        error = np.concatenate(errors) if errors else np.zeros(0)
        bound = np.concatenate(bounds) if bounds else np.zeros(0)
        scale = np.concatenate(scales) if scales else np.zeros(0)
        eta = np.concatenate(etas) if etas else np.zeros(0)
        info = QuantizationInfo(
            calls=1,
            max_weighted_error=float(np.max(error)) if error.size else 0.0,
            max_bound=float(np.max(bound)) if bound.size else 0.0,
            mean_scale=float(np.mean(scale)) if scale.size else 0.0,
            max_empirical_eta=float(np.max(eta)) if eta.size else 0.0,
            mean_empirical_eta=float(np.mean(eta)) if eta.size else 0.0,
            max_moment_error=float(moment_error),
            max_negative_part_norm=float(negative),
        )
        decoded = out.reshape(shape)
        return (decoded, info) if diagnostics else decoded

    def repair_positivity_and_moments(self, V: np.ndarray, target_moments: np.ndarray,
                                      max_iter: int = 8) -> tuple[np.ndarray, float, float, float]:
        shape = np.asarray(V).shape
        U0 = np.asarray(V, dtype=float).reshape(-1, self.d)
        U = U0.copy()
        if self.B is None:
            U = np.maximum(U, 0.0)
            return U.reshape(shape), 0.0, 0.0, float(np.max(np.linalg.norm((U - U0) * self.sqrtw, axis=1)))
        target = np.asarray(target_moments, dtype=float).reshape(U.shape[0], -1)
        for _ in range(int(max_iter)):
            U = np.maximum(U, 0.0)
            correction = ((target - self.moments(U)) @ self.Ginv.T) @ self.B.T
            U += correction
        correction = ((target - self.moments(U)) @ self.Ginv.T) @ self.B.T
        U += correction
        moment_error = float(np.max(np.linalg.norm(self.moments(U) - target, axis=1)))
        negative = float(np.max(np.linalg.norm(np.minimum(U, 0.0) * self.sqrtw, axis=1)))
        displacement = float(np.max(np.linalg.norm((U - U0) * self.sqrtw, axis=1)))
        return U.reshape(shape), moment_error, negative, displacement

    def storage_ratio(self, bits: int, mode: str = "moment", baseline_bits: int = 64) -> float:
        if mode == "none":
            return 1.0
        metadata = 1 if mode == "raw" else 1 + self.n_preserved
        return float((baseline_bits * self.d) / (bits * self.D + baseline_bits * metadata))
