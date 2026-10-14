"""Bond maths: the first-order return model the problem uses, and the exact pricer that checks it.

The decision problem uses, per bucket and per month,

    r_i  ~=  y_i * dt  -  D_i * dy_i            (carry minus duration effect)

Convexity (+ 1/2 C_i dy_i^2) is deliberately left out of v1. Gate B measures how wrong that is.
Buckets are constant-maturity zero-coupon exposures, so modified duration (continuous
compounding) equals maturity.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


# ------------------------------------------------------------------ zeros
def zero_price(y, m):
    """Continuously-compounded zero-coupon price."""
    return np.exp(-np.asarray(y) * np.asarray(m))


def zero_modified_duration(m):
    return np.asarray(m, dtype=float)


def zero_convexity(m):
    return np.asarray(m, dtype=float) ** 2


def approx_return(y0, dy, duration, dt):
    """First-order one-period return: carry minus duration effect."""
    return np.asarray(y0) * dt - np.asarray(duration) * np.asarray(dy)


def exact_zero_return(y0, dy, m, dt):
    """Exact holding return of a zero bought at maturity m and yield y0, sold one period later
    (maturity m - dt) after its own yield moved by dy."""
    y0, dy, m = np.asarray(y0), np.asarray(dy), np.asarray(m)
    return zero_price(y0 + dy, m - dt) / zero_price(y0, m) - 1.0


def approximation_break_point(m, y0, dt, tol, scan_max, n=20001) -> float:
    """Smallest |dy| at which |exact - approx| exceeds `tol` (either direction), or inf."""
    grid = np.linspace(0.0, scan_max, n)
    err = np.maximum(
        np.abs(exact_zero_return(y0, grid, m, dt) - approx_return(y0, grid, m, dt)),
        np.abs(exact_zero_return(y0, -grid, m, dt) - approx_return(y0, -grid, m, dt)))
    over = np.nonzero(err > tol)[0]
    return float(grid[over[0]]) if len(over) else float("inf")


# --------------------------------------------------------- coupon bonds
def coupon_bond_price(ytm: float, coupon: float, maturity: float, freq: int = 2) -> float:
    """Price per 1 face of a bullet bond, yield compounded at `freq` (integer periods)."""
    n = int(round(maturity * freq))
    k = np.arange(1, n + 1)
    disc = (1 + ytm / freq) ** (-k)
    return float((coupon / freq) * disc.sum() + disc[-1])


def coupon_bond_ytm(price: float, coupon: float, maturity: float, freq: int = 2) -> float:
    return brentq(lambda y: coupon_bond_price(y, coupon, maturity, freq) - price, -0.5, 2.0,
                  xtol=1e-14)


def coupon_bond_modified_duration(ytm: float, coupon: float, maturity: float,
                                  freq: int = 2) -> float:
    n = int(round(maturity * freq))
    k = np.arange(1, n + 1)
    cf = np.full(n, coupon / freq)
    cf[-1] += 1.0
    disc = (1 + ytm / freq) ** (-k)
    price = (cf * disc).sum()
    macaulay = (k / freq * cf * disc).sum() / price
    return float(macaulay / (1 + ytm / freq))
