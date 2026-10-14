import numpy as np

from duralloc.policy import FunctionPolicy
from duralloc.policy_map import compare_tables, policy_map, state_table


def test_extraction_on_known_policy(problem):
    p = problem
    hi = int(np.argmax(np.where(p.band_ok, p.actions[:, p.long_index], -1)))
    lo = int(np.argmin(np.where(p.band_ok, p.actions[:, p.long_index], 9)))
    mu = p.model.mu[0]
    known = FunctionPolicy(p, lambda t, w, f: np.where(f[:, 0] > mu, hi, lo), "known")
    levels = np.linspace(mu - 0.02, mu + 0.02, 9)
    slopes = np.linspace(0.0, 0.04, 5)
    m = policy_map(known, p, levels, slopes, p.static_idx, 0)
    expected = np.where(levels[:, None] > mu, p.actions[hi, p.long_index], p.actions[lo, p.long_index])
    np.testing.assert_allclose(m["long_weight"], np.broadcast_to(expected, (9, 5)))
    assert m["long_weight"].shape == (9, 5)
    assert not m["hold"].any() or p.static_idx in (hi, lo)


def test_compare_tables_identical_and_shifted(problem):
    p = problem
    pts = np.array([[0.05, 0.02], [0.06, 0.01]])
    hold = FunctionPolicy(p, lambda t, w, f: w, "hold")
    a = state_table(hold, p, pts, 0)
    same = compare_tables(p, a, a, 0.1)
    assert same["exact_match"] == 1.0 and same["hold_jaccard"] == 1.0 and same["hold_fraction_a"] == 1.0
    target = p.static_idx
    other = state_table(FunctionPolicy(p, lambda t, w, f: np.full(len(w), target), "s"), p, pts, 0)
    diff = compare_tables(p, a, other, 0.1)
    assert diff["exact_match"] < 1.0
