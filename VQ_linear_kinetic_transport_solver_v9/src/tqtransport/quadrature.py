from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProductQuadrature:
    """Upper-hemisphere product quadrature used by the reduced 2D solver.

    The quadrature weights are normalized to sum to one. Only the x and y
    components enter the Cartesian sweep. The z component is retained for
    completeness and for consistency with the reduced-sphere construction.
    """

    order: int
    omega: np.ndarray
    weights: np.ndarray

    @property
    def n_angles(self) -> int:
        return int(self.omega.shape[0])

    def validate(self) -> None:
        if self.order < 2:
            raise ValueError("quadrature order must be at least two")
        if self.omega.ndim != 2 or self.omega.shape[1] != 3:
            raise ValueError("omega must have shape (M, 3)")
        if self.weights.shape != (self.n_angles,):
            raise ValueError("weights must have shape (M,)")
        if np.any(self.weights <= 0.0):
            raise ValueError("quadrature weights must be positive")
        if not np.isclose(float(np.sum(self.weights)), 1.0, atol=5e-14):
            raise ValueError("quadrature weights must sum to one")


def product_quadrature(order: int) -> ProductQuadrature:
    """Construct the N-by-N upper-hemisphere product quadrature.

    A Gauss--Legendre rule is mapped from [-1, 1] to zeta in [0, 1].
    Azimuthal midpoints are alpha_j = 2*pi*(j+1/2)/N. The resulting number
    of angular directions is M=N^2.
    """

    order = int(order)
    if order < 2:
        raise ValueError("order must be at least two")

    nodes, weights = np.polynomial.legendre.leggauss(order)
    zeta = 0.5 * (nodes + 1.0)
    wz = 0.5 * weights
    alpha = 2.0 * np.pi * (np.arange(order, dtype=np.float64) + 0.5) / order

    omega = np.empty((order * order, 3), dtype=np.float64)
    w = np.empty(order * order, dtype=np.float64)
    k = 0
    for i in range(order):
        radial = float(np.sqrt(max(0.0, 1.0 - zeta[i] * zeta[i])))
        for j in range(order):
            omega[k, 0] = radial * np.cos(alpha[j])
            omega[k, 1] = radial * np.sin(alpha[j])
            omega[k, 2] = zeta[i]
            w[k] = wz[i] / order
            k += 1

    w /= np.sum(w)
    quad = ProductQuadrature(order=order, omega=omega, weights=w)
    quad.validate()
    return quad


def moment_basis(quad: ProductQuadrature, preserve: str = "phi_current") -> np.ndarray:
    """Return the protected angular basis evaluated on a quadrature."""

    key = preserve.strip().lower()
    one = np.ones(quad.n_angles, dtype=np.float64)
    if key in {"phi", "scalar", "density"}:
        return one[:, None]
    if key in {"phi_current", "scalar_current", "density_current"}:
        return np.column_stack((one, quad.omega[:, 0], quad.omega[:, 1]))
    raise ValueError(f"unsupported protected-moment set: {preserve!r}")


def moments(field: np.ndarray, quad: ProductQuadrature, preserve: str = "phi_current") -> np.ndarray:
    """Compute protected moments for a field with angular axis first."""

    field = np.asarray(field, dtype=np.float64)
    if field.shape[0] != quad.n_angles:
        raise ValueError("field angular dimension does not match quadrature")
    b = moment_basis(quad, preserve)
    flat = field.reshape(quad.n_angles, -1)
    return (b.T @ (quad.weights[:, None] * flat)).reshape((b.shape[1],) + field.shape[1:])
