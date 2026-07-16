from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Quadrature1D:
    mu: np.ndarray
    w: np.ndarray
    order: int

    @property
    def n_angles(self) -> int:
        return int(self.w.size)


@dataclass(frozen=True)
class Quadrature2D:
    ox: np.ndarray
    oy: np.ndarray
    oz: np.ndarray
    w: np.ndarray
    order: int

    @property
    def n_angles(self) -> int:
        return int(self.w.size)

    @property
    def directions(self) -> np.ndarray:
        return np.column_stack((self.ox, self.oy, self.oz))


def gauss_legendre_1d(order: int) -> Quadrature1D:
    if order < 2:
        raise ValueError("1D angular order must be at least 2")
    mu, w = np.polynomial.legendre.leggauss(int(order))
    w = w / np.sum(w)
    return Quadrature1D(mu.astype(float), w.astype(float), int(order))


def upper_hemisphere_product_quadrature(order: int) -> Quadrature2D:
    """Return the N^2 upper-hemisphere product rule used by the 2D solver.

    Weights are normalized to one, so an isotropic angular field with value c
    has scalar flux c. Only ox and oy enter the Cartesian streaming operator.
    """
    order = int(order)
    if order < 2:
        raise ValueError("2D angular order must be at least 2")
    z_nodes, z_weights = np.polynomial.legendre.leggauss(order)
    z = 0.5 * (z_nodes + 1.0)
    wz = 0.5 * z_weights
    alpha = (np.arange(order, dtype=float) + 0.5) * (2.0 * np.pi / order)

    ox = np.empty(order * order, dtype=float)
    oy = np.empty_like(ox)
    oz = np.empty_like(ox)
    w = np.empty_like(ox)
    k = 0
    for iz in range(order):
        radius = math.sqrt(max(0.0, 1.0 - float(z[iz]) ** 2))
        for ia in range(order):
            ox[k] = radius * math.cos(float(alpha[ia]))
            oy[k] = radius * math.sin(float(alpha[ia]))
            oz[k] = float(z[iz])
            w[k] = float(wz[iz]) / order
            k += 1
    w /= np.sum(w)
    return Quadrature2D(ox, oy, oz, w, order)


def basis_1d(q: Quadrature1D, preserve: str) -> np.ndarray:
    if preserve in ("", "none"):
        return np.zeros((q.n_angles, 0), dtype=float)
    cols = [np.ones(q.n_angles, dtype=float)]
    if preserve == "phi_current":
        cols.append(q.mu.copy())
    elif preserve != "phi":
        raise ValueError(f"unknown preserve mode {preserve!r}")
    return np.column_stack(cols)


def basis_2d(q: Quadrature2D, preserve: str) -> np.ndarray:
    if preserve in ("", "none"):
        return np.zeros((q.n_angles, 0), dtype=float)
    cols = [np.ones(q.n_angles, dtype=float)]
    if preserve == "phi_current":
        cols.extend([q.ox.copy(), q.oy.copy()])
    elif preserve != "phi":
        raise ValueError(f"unknown preserve mode {preserve!r}")
    return np.column_stack(cols)
