from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np

from .quadrature import ProductQuadrature, moment_basis


@dataclass
class QuantizationStats:
    bits: int
    n_angles: int
    padded_dimension: int
    protected_moments: int
    cells: int
    raw_bytes: int
    packed_bytes_estimate: int
    storage_ratio_estimate: float
    max_weighted_error_pre_repair: float
    max_weighted_error_post_repair: float
    max_relative_residual_distortion: float
    max_moment_residual: float
    max_negative_part_norm: float
    min_angular_value: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    return 1 << (int(n) - 1).bit_length()


def _fwht_normalized(values: np.ndarray) -> np.ndarray:
    """Normalized Walsh--Hadamard transform along axis zero.

    The returned transform is orthogonal and self-inverse. The first axis
    length must be a power of two.
    """

    out = np.array(values, dtype=np.float64, copy=True, order="C")
    n = int(out.shape[0])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard dimension must be a power of two")
    h = 1
    while h < n:
        for start in range(0, n, 2 * h):
            left = out[start : start + h].copy()
            right = out[start + h : start + 2 * h].copy()
            out[start : start + h] = left + right
            out[start + h : start + 2 * h] = left - right
        h *= 2
    out /= np.sqrt(float(n))
    return out


class MomentPreservingQuantizer:
    """Random-sign/Hadamard scalar quantizer with exact moment correction."""

    def __init__(
        self,
        quad: ProductQuadrature,
        bits: int,
        preserve: str = "phi_current",
        seed: int = 12345,
        chunk_cells: int = 2048,
    ) -> None:
        if int(bits) < 2:
            raise ValueError("bits must be at least two")
        self.quad = quad
        self.bits = int(bits)
        self.preserve = preserve
        self.seed = int(seed)
        self.chunk_cells = max(1, int(chunk_cells))
        self.m = quad.n_angles
        self.d = _next_power_of_two(self.m)
        self.qmax = (1 << (self.bits - 1)) - 1
        self.basis = moment_basis(quad, preserve)
        self.r = int(self.basis.shape[1])
        self.weights = quad.weights.astype(np.float64, copy=True)
        self.sqrt_w = np.sqrt(self.weights)
        self.inv_sqrt_w = 1.0 / self.sqrt_w
        self.gram = self.basis.T @ (self.weights[:, None] * self.basis)
        self.gram_inv = np.linalg.inv(self.gram)
        self.bhat = self.sqrt_w[:, None] * self.basis
        rng = np.random.default_rng(self.seed)
        self.signs = rng.choice(np.array([-1.0, 1.0]), size=self.d)

    def protected_moments(self, field: np.ndarray) -> np.ndarray:
        field = np.asarray(field, dtype=np.float64)
        if field.shape[0] != self.m:
            raise ValueError("field angular dimension does not match quantizer")
        flat = field.reshape(self.m, -1)
        return self.basis.T @ (self.weights[:, None] * flat)

    def _repair(self, values: np.ndarray, target: np.ndarray, iterations: int) -> np.ndarray:
        repaired = np.array(values, copy=True)
        for _ in range(max(0, int(iterations))):
            np.maximum(repaired, 0.0, out=repaired)
            defect = target - self.basis.T @ (self.weights[:, None] * repaired)
            repaired += self.basis @ (self.gram_inv @ defect)
        if iterations > 0:
            defect = target - self.basis.T @ (self.weights[:, None] * repaired)
            repaired += self.basis @ (self.gram_inv @ defect)
        return repaired

    def apply(self, field: np.ndarray, repair_iterations: int = 0) -> tuple[np.ndarray, QuantizationStats]:
        """Quantize a field whose angular dimension is the first axis."""

        field = np.asarray(field, dtype=np.float64)
        if field.ndim < 2:
            raise ValueError("field must have an angular axis and at least one cell axis")
        if field.shape[0] != self.m:
            raise ValueError("field angular dimension does not match quantizer")

        original_shape = field.shape
        flat = field.reshape(self.m, -1)
        cells = int(flat.shape[1])
        reconstructed = np.empty_like(flat)

        max_pre = 0.0
        max_post = 0.0
        max_relative = 0.0
        max_moment = 0.0
        max_negative = 0.0
        min_angular = np.inf

        for start in range(0, cells, self.chunk_cells):
            stop = min(cells, start + self.chunk_cells)
            u = flat[:, start:stop]
            target = self.basis.T @ (self.weights[:, None] * u)
            protected = self.basis @ (self.gram_inv @ target)
            residual = u - protected
            z = self.sqrt_w[:, None] * residual

            padded = np.zeros((self.d, stop - start), dtype=np.float64)
            padded[: self.m] = z
            padded *= self.signs[:, None]
            y = _fwht_normalized(padded)

            scale = np.max(np.abs(y), axis=0) / float(self.qmax)
            zero = scale == 0.0
            safe_scale = scale.copy()
            safe_scale[zero] = 1.0
            codes = np.rint(y / safe_scale[None, :])
            np.clip(codes, -self.qmax, self.qmax, out=codes)
            yhat = codes * safe_scale[None, :]
            if np.any(zero):
                yhat[:, zero] = 0.0

            decoded = _fwht_normalized(yhat)
            decoded *= self.signs[:, None]
            zbar = decoded[: self.m]

            leakage = self.bhat.T @ zbar
            zcorr = zbar - self.bhat @ (self.gram_inv @ leakage)
            pre = protected + self.inv_sqrt_w[:, None] * zcorr

            pre_error = pre - u
            pre_norm = np.sqrt(np.sum(self.weights[:, None] * pre_error * pre_error, axis=0))
            residual_norm = np.sqrt(np.sum(self.weights[:, None] * residual * residual, axis=0))
            relative = pre_norm / np.maximum(residual_norm, 1.0e-300)
            relative[residual_norm <= 1.0e-300] = 0.0

            post = self._repair(pre, target, repair_iterations)
            post_error = post - u
            post_norm = np.sqrt(np.sum(self.weights[:, None] * post_error * post_error, axis=0))
            moment_residual = self.basis.T @ (self.weights[:, None] * post) - target
            negative = np.minimum(post, 0.0)
            negative_norm = np.sqrt(np.sum(self.weights[:, None] * negative * negative, axis=0))

            reconstructed[:, start:stop] = post
            max_pre = max(max_pre, float(np.max(pre_norm, initial=0.0)))
            max_post = max(max_post, float(np.max(post_norm, initial=0.0)))
            max_relative = max(max_relative, float(np.max(relative, initial=0.0)))
            max_moment = max(max_moment, float(np.max(np.abs(moment_residual), initial=0.0)))
            max_negative = max(max_negative, float(np.max(negative_norm, initial=0.0)))
            min_angular = min(min_angular, float(np.min(post, initial=np.inf)))

        raw_bytes = int(field.size * np.dtype(np.float64).itemsize)
        bytes_per_cell = int(ceil(self.bits * self.d / 8.0) + 8 * (self.r + 1))
        packed_bytes = int(bytes_per_cell * cells)
        ratio = float(raw_bytes / packed_bytes) if packed_bytes > 0 else 1.0

        stats = QuantizationStats(
            bits=self.bits,
            n_angles=self.m,
            padded_dimension=self.d,
            protected_moments=self.r,
            cells=cells,
            raw_bytes=raw_bytes,
            packed_bytes_estimate=packed_bytes,
            storage_ratio_estimate=ratio,
            max_weighted_error_pre_repair=max_pre,
            max_weighted_error_post_repair=max_post,
            max_relative_residual_distortion=max_relative,
            max_moment_residual=max_moment,
            max_negative_part_norm=max_negative,
            min_angular_value=min_angular,
        )
        return reconstructed.reshape(original_shape), stats
