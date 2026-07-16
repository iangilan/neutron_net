from __future__ import annotations

import numpy as np
from .quadrature import Quadrature2D, basis_2d


def angular_moments(V: np.ndarray, q: Quadrature2D, preserve: str = "phi_current") -> np.ndarray:
    B = basis_2d(q, preserve)
    M = np.asarray(V, dtype=float).reshape(-1, q.n_angles)
    return M @ (q.w[:, None] * B)


def _nearest_direction_indices(source: Quadrature2D, target: Quadrature2D) -> np.ndarray:
    dots = target.directions @ source.directions.T
    return np.argmax(dots, axis=1)


def transfer_angular_field(V: np.ndarray, source: Quadrature2D, target: Quadrature2D,
                           preserve: str = "phi_current", chunk_rows: int = 4096) -> np.ndarray:
    """Transfer angular data between product quadratures and preserve moments.

    A nearest-direction reconstruction provides a high-order shape estimate.
    A weighted affine correction on the target quadrature then enforces exact
    equality of the selected source and target moments.
    """
    arr = np.asarray(V, dtype=float)
    if arr.shape[-1] != source.n_angles:
        raise ValueError("last dimension does not match the source quadrature")
    if source.order == target.order and source.n_angles == target.n_angles:
        return arr.copy()
    shape = arr.shape[:-1] + (target.n_angles,)
    src_rows = arr.reshape(-1, source.n_angles)
    out = np.empty((src_rows.shape[0], target.n_angles), dtype=float)

    Bsrc = basis_2d(source, preserve)
    Bdst = basis_2d(target, preserve)
    WBsrc = source.w[:, None] * Bsrc
    WBdst = target.w[:, None] * Bdst
    Gdst = Bdst.T @ WBdst
    Ginv = np.linalg.inv(Gdst)
    nearest = _nearest_direction_indices(source, target)

    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, src_rows.shape[0], chunk_rows):
        stop = min(src_rows.shape[0], start + chunk_rows)
        src = src_rows[start:stop]
        initial = src[:, nearest]
        target_moments = src @ WBsrc
        initial_moments = initial @ WBdst
        coeff = (target_moments - initial_moments) @ Ginv.T
        out[start:stop] = initial + coeff @ Bdst.T
    return out.reshape(shape)


def transfer_moment_error(V: np.ndarray, source: Quadrature2D, target: Quadrature2D,
                          preserve: str = "phi_current") -> float:
    transferred = transfer_angular_field(V, source, target, preserve=preserve)
    a = angular_moments(V, source, preserve)
    b = angular_moments(transferred, target, preserve)
    return float(np.max(np.linalg.norm(a - b, axis=1))) if a.size else 0.0
