"""Rectangular (level, slope) grid with clamped bilinear interpolation."""

from __future__ import annotations

import numpy as np


class FactorGrid:
    def __init__(self, center: np.ndarray, sd: np.ndarray, n_std: float, n_level: int, n_slope: int):
        self.level = np.linspace(center[0] - n_std * sd[0], center[0] + n_std * sd[0], n_level)
        self.slope = np.linspace(center[1] - n_std * sd[1], center[1] + n_std * sd[1], n_slope)
        L, S = np.meshgrid(self.level, self.slope, indexing="ij")
        self.points = np.column_stack([L.ravel(), S.ravel()])  # index = i_level * n_slope + i_slope
        self.shape = (n_level, n_slope)
        self.n = len(self.points)

    def _axis(self, axis: np.ndarray, x: np.ndarray):
        x = np.clip(x, axis[0], axis[-1])
        i = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, len(axis) - 2)
        u = (x - axis[i]) / (axis[i + 1] - axis[i])
        return i, u

    def weights(self, f: np.ndarray):
        """Bilinear (indices (N,4), weights (N,4)) of points f (N,2); clamped at the edges."""
        f = np.asarray(f, dtype=float).reshape(-1, 2)
        i, u = self._axis(self.level, f[:, 0])
        j, v = self._axis(self.slope, f[:, 1])
        ns = self.shape[1]
        idx = np.stack([i * ns + j, i * ns + j + 1, (i + 1) * ns + j, (i + 1) * ns + j + 1], 1)
        w = np.stack([(1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v], 1)
        return idx, w

    def interpolate(self, values: np.ndarray, f: np.ndarray) -> np.ndarray:
        """values (..., n) on the grid -> (N, ...) at points f."""
        idx, w = self.weights(f)
        gathered = values[..., idx]  # (..., N, 4)
        out = (gathered * w).sum(-1)
        return np.moveaxis(out, -1, 0)

    def inner_mask(self, center, sd, n_std: float) -> np.ndarray:
        z = np.abs((self.points - center) / sd)
        return (z <= n_std + 1e-9).all(axis=1)
