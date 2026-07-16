from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class Problem1D:
    name: str
    nx: int
    x_min: float
    x_max: float
    final_time: float
    cfl: float
    sigma_a: np.ndarray
    sigma_s: np.ndarray
    q: np.ndarray
    phi0: np.ndarray
    left_inflow: float
    right_inflow: float
    metadata: Dict[str, object]

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def x(self) -> np.ndarray:
        return self.x_min + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def sigma_t(self) -> np.ndarray:
        return self.sigma_a + self.sigma_s


@dataclass
class Problem2D:
    name: str
    nx: int
    ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    final_time: float
    cfl: float
    sigma_a: np.ndarray
    sigma_s: np.ndarray
    q: np.ndarray
    phi0: np.ndarray
    left_inflow: float
    right_inflow: float
    bottom_inflow: float
    top_inflow: float
    metadata: Dict[str, object]

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def x(self) -> np.ndarray:
        return self.x_min + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y(self) -> np.ndarray:
        return self.y_min + (np.arange(self.ny) + 0.5) * self.dy

    @property
    def sigma_t(self) -> np.ndarray:
        return self.sigma_a + self.sigma_s


def make_problem_1d(name: str, nx: int, final_time: float | None = None) -> Problem1D:
    canonical = name.lower().replace("-", "_")
    nx = int(nx)
    if canonical == "gaussian":
        x_min, x_max = -1.0, 1.0
        x = x_min + (np.arange(nx) + 0.5) * ((x_max - x_min) / nx)
        zeta = 0.05
        phi0 = np.exp(-(x * x) / (2.0 * zeta * zeta)) / np.sqrt(2.0 * np.pi * zeta * zeta)
        sigma_s = np.full(nx, 0.5)
        sigma_a = np.zeros(nx)
        q = np.zeros(nx)
        tf = 0.5 if final_time is None else float(final_time)
        meta = {"source": "Gaussian benchmark", "zeta": zeta}
    elif canonical in {"vanishing", "vanishing_cross_section"}:
        canonical = "vanishing_cross_section"
        x_min, x_max = -1.0, 1.0
        x = x_min + (np.arange(nx) + 0.5) * ((x_max - x_min) / nx)
        phi0 = ((x > -0.2) & (x < 0.2)).astype(float)
        sigma_s = 100.0 * x**4
        sigma_a = np.zeros(nx)
        q = np.zeros(nx)
        tf = 0.5 if final_time is None else float(final_time)
        meta = {"source": "Vanishing-cross-section benchmark"}
    elif canonical == "reed":
        x_min, x_max = 0.0, 8.0
        x = x_min + (np.arange(nx) + 0.5) * ((x_max - x_min) / nx)
        sigma_s = np.zeros(nx)
        sigma_t = np.zeros(nx)
        q = np.zeros(nx)
        masks = [
            (x >= 0.0) & (x < 2.0), (x >= 2.0) & (x < 3.0),
            (x >= 3.0) & (x < 5.0), (x >= 5.0) & (x < 6.0),
            (x >= 6.0) & (x <= 8.0),
        ]
        q[masks[0]], sigma_s[masks[0]], sigma_t[masks[0]] = 50.0, 0.0, 50.0
        q[masks[1]], sigma_s[masks[1]], sigma_t[masks[1]] = 0.0, 0.0, 5.0
        q[masks[2]], sigma_s[masks[2]], sigma_t[masks[2]] = 0.0, 0.0, 0.0
        q[masks[3]], sigma_s[masks[3]], sigma_t[masks[3]] = 1.0, 0.9, 1.0
        q[masks[4]], sigma_s[masks[4]], sigma_t[masks[4]] = 0.0, 0.9, 1.0
        sigma_a = np.maximum(sigma_t - sigma_s, 0.0)
        phi0 = np.zeros(nx)
        tf = 5.0 if final_time is None else float(final_time)
        meta = {"source": "Reed benchmark", "left_boundary_note": "vacuum in this compact v9 solver"}
    else:
        raise ValueError(f"unknown 1D problem {name!r}")
    return Problem1D(canonical, nx, x_min, x_max, tf, 0.5,
                     sigma_a.astype(float), sigma_s.astype(float), q.astype(float),
                     phi0.astype(float), 0.0, 0.0, meta)


def _mesh_2d(nx: int, ny: int, bounds: tuple[float, float, float, float]):
    x0, x1, y0, y1 = bounds
    x = x0 + (np.arange(nx) + 0.5) * ((x1 - x0) / nx)
    y = y0 + (np.arange(ny) + 0.5) * ((y1 - y0) / ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y


def make_problem_2d(name: str, nx: int, ny: int | None = None,
                    final_time: float | None = None) -> Problem2D:
    canonical = name.lower().replace("-", "_")
    nx = int(nx)
    ny = nx if ny is None else int(ny)
    if canonical == "line_source":
        bounds = (-1.5, 1.5, -1.5, 1.5)
        _, _, X, Y = _mesh_2d(nx, ny, bounds)
        zeta = 0.03
        phi0 = np.exp(-(X * X + Y * Y) / (2.0 * zeta)) / (2.0 * np.pi * zeta)
        sigma_a = np.zeros((nx, ny))
        sigma_s = np.ones((nx, ny))
        q = np.zeros((nx, ny))
        tf, cfl = 1.0, 0.5
        bcs = (0.0, 0.0, 0.0, 0.0)
        meta = {
            "source": "Krotz-Hauck-McClarren line source",
            "mixed_order_recommended": True,
            "reason": "high-order ballistic uncollided field and smoother collided field",
        }
    elif canonical == "lattice":
        bounds = (0.0, 7.0, 0.0, 7.0)
        _, _, X, Y = _mesh_2d(nx, ny, bounds)
        sigma_a = np.zeros((nx, ny))
        sigma_s = np.ones((nx, ny))
        q = np.zeros((nx, ny))
        I = np.clip(np.floor(X).astype(int), 0, 6)
        J = np.clip(np.floor(Y).astype(int), 0, 6)
        source = (I == 3) & (J == 3)
        absorber_cells = {(1,5),(5,5),(2,4),(4,4),(1,3),(5,3),(2,2),(4,2),(1,1),(3,1),(5,1)}
        absorbers = np.zeros((nx, ny), dtype=bool)
        for ii, jj in absorber_cells:
            absorbers |= (I == ii) & (J == jj)
        sigma_a[absorbers] = 10.0
        sigma_s[absorbers] = 0.0
        q[source] = 1.0
        phi0 = np.zeros((nx, ny))
        tf, cfl = 3.2, 25.6
        bcs = (0.0, 0.0, 0.0, 0.0)
        meta = {"source": "Krotz-Hauck-McClarren lattice"}
    elif canonical in {"hohlraum", "modified_hohlraum"}:
        canonical = "hohlraum"
        bounds = (0.0, 1.3, 0.0, 1.3)
        _, _, X, Y = _mesh_2d(nx, ny, bounds)
        sigma_a = np.zeros((nx, ny))
        sigma_s = np.full((nx, ny), 0.1)
        q = np.zeros((nx, ny))
        h = 0.05
        black = (X > 1.3 - h) | (Y < h) | (Y > 1.3 - h)
        sigma_s[black] = 100.0
        red = (X < h) & (Y >= 0.20) & (Y <= 1.00)
        sigma_a[red], sigma_s[red] = 5.0, 95.0
        green = (X >= 0.40) & (X <= 0.90) & (Y >= 0.25) & (Y <= 1.05)
        blue = (X >= 0.45) & (X <= 0.85) & (Y >= 0.30) & (Y <= 1.00)
        sigma_a[green], sigma_s[green] = 10.0, 90.0
        sigma_a[blue], sigma_s[blue] = 50.0, 50.0
        phi0 = np.zeros((nx, ny))
        tf, cfl = 2.6, 52.0
        bcs = (1.0, 0.0, 0.0, 0.0)
        meta = {"source": "modified Krotz-Hauck-McClarren hohlraum", "left_inflow": "uniform"}
    else:
        raise ValueError(f"unknown 2D problem {name!r}")
    if final_time is not None:
        tf = float(final_time)
    return Problem2D(canonical, nx, ny, *bounds, float(tf), float(cfl),
                     sigma_a.astype(float), sigma_s.astype(float), q.astype(float),
                     phi0.astype(float), *bcs, meta)
