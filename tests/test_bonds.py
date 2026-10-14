import numpy as np
import pytest

from duralloc import checks
from duralloc.bonds import (approx_return, approximation_break_point, coupon_bond_modified_duration,
                            coupon_bond_price, coupon_bond_ytm, exact_zero_return, zero_price)


def test_gate_b_small_moves_within_tolerance(problem, rng):
    ok, det = checks.gate_b(problem, rng, n_paths=200)
    assert ok, det
    for b in det["buckets"].values():
        assert b["break_point_dy"] > problem.cfg.gates.b.small_move  # breaks only beyond "small"


def test_error_grows_quadratically_with_move():
    # convexity: doubling the move roughly quadruples the error for a long zero
    e1 = exact_zero_return(0.05, 0.002, 20, 1 / 12) - approx_return(0.05, 0.002, 20, 1 / 12)
    e2 = exact_zero_return(0.05, 0.004, 20, 1 / 12) - approx_return(0.05, 0.004, 20, 1 / 12)
    assert 3.0 < e2 / e1 < 5.0


def test_approximation_understates_returns_symmetrically():
    for dy in (-0.01, 0.01):
        assert exact_zero_return(0.05, dy, 20, 1 / 12) > approx_return(0.05, dy, 20, 1 / 12)


def test_break_point_ordered_by_duration():
    bps = [approximation_break_point(m, 0.05, 1 / 12, 5e-4, 0.05) for m in (2, 7, 20)]
    assert bps[0] > bps[1] > bps[2]


def test_zero_duration_matches_numerical_derivative():
    y, m, h = 0.05, 7.0, 1e-6
    num = -(zero_price(y + h, m) - zero_price(y - h, m)) / (2 * h) / zero_price(y, m)
    assert num == pytest.approx(m, rel=1e-8)


def test_coupon_bond_ytm_roundtrip_and_par():
    assert coupon_bond_price(0.04, 0.04, 10) == pytest.approx(1.0)
    p = coupon_bond_price(0.037, 0.05, 12)
    assert coupon_bond_ytm(p, 0.05, 12) == pytest.approx(0.037, abs=1e-12)


def test_coupon_bond_modified_duration_matches_numerical():
    y, c, T, h = 0.045, 0.03, 15, 1e-6
    num = -(coupon_bond_price(y + h, c, T) - coupon_bond_price(y - h, c, T)) / (2 * h) / coupon_bond_price(y, c, T)
    assert coupon_bond_modified_duration(y, c, T) == pytest.approx(num, rel=1e-6)
