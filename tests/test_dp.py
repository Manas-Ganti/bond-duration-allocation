import itertools

import numpy as np
import pytest

from duralloc.baselines import DPPolicy, MyopicPolicy, StaticPolicy
from duralloc.baselines.dp import solve_dp, transition_operator
from duralloc.env import rollout, simulate_market
from duralloc.problem import Problem
from duralloc.stats import mc_estimate


def _tiny(seed=0, nF=3, nA=3):
    rng = np.random.default_rng(seed)
    reward = rng.normal(0, 1, (nF, nA))
    acts = np.arange(nA, dtype=float)
    cost = 0.3 * np.abs(acts[:, None] - acts[None, :])
    feasible = np.abs(acts[:, None] - acts[None, :]) <= 1  # turnover limit: one step at a time
    return reward, cost, feasible


def test_dp_matches_brute_force_on_deterministic_curve():
    reward, cost, feasible = _tiny()
    nF, nA = reward.shape
    P = np.roll(np.eye(nF), 1, axis=1)  # f -> f+1 mod nF, deterministic
    T = 4
    sol = solve_dp(reward, P, cost, feasible, T)
    for w0, f0 in itertools.product(range(nA), range(nF)):
        best = -np.inf
        for seq in itertools.product(range(nA), repeat=T):
            w, f, v = w0, f0, 0.0
            for a in seq:
                if not feasible[w, a]:
                    v = -np.inf
                    break
                v += reward[f, a] - cost[w, a]
                w, f = a, (f + 1) % nF
            best = max(best, v)
        assert sol.value[0, w0, f0] == pytest.approx(best, abs=1e-9)


def test_dp_matches_nested_expectation_two_periods():
    reward, cost, feasible = _tiny(seed=3)
    nF, nA = reward.shape
    P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
    sol = solve_dp(reward, P, cost, feasible, 2)
    for w0, f0 in itertools.product(range(nA), range(nF)):
        def v1(w, f):
            return max(reward[f, a] - cost[w, a] for a in range(nA) if feasible[w, a])
        v0 = max(reward[f0, a] - cost[w0, a] + sum(P[f0, f1] * v1(a, f1) for f1 in range(nF))
                 for a in range(nA) if feasible[w0, a])
        assert sol.value[0, w0, f0] == pytest.approx(v0, abs=1e-9)


def test_hand_case_cost_exceeds_gain_so_hold():
    # two allocations, the other one earns 1 more per period for 2 periods; switching costs 3
    reward = np.array([[0.0, 1.0]])
    cost = np.array([[0.0, 3.0], [3.0, 0.0]])
    sol = solve_dp(reward, np.ones((1, 1)), cost, np.ones((2, 2), bool), 2)
    assert sol.policy[0, 0, 0] == 0 and sol.value[0, 0, 0] == pytest.approx(0.0, abs=1e-9)
    # with 4 periods the gain (4) beats the cost (3): switch immediately
    sol4 = solve_dp(reward, np.ones((1, 1)), cost, np.ones((2, 2), bool), 4)
    assert sol4.policy[0, 0, 0] == 1 and sol4.value[0, 0, 0] == pytest.approx(1.0)


def test_transition_operator_is_stochastic_and_mean_preserving(problem):
    dp = DPPolicy(problem)
    P, g = dp.P, dp.grid
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-12)
    m = problem.model
    inner = g.inner_mask(m.mu, m.stat_sd, 2.0)
    np.testing.assert_allclose((P @ g.points)[inner], m.cond_mean(g.points)[inner], atol=1e-7)


def test_dp_beats_baselines_and_predicts_its_own_value(cfg):
    p = Problem(cfg, horizon=6)
    dp = DPPolicy(p)
    market = simulate_market(p, 20000, np.random.default_rng(7))
    r = {pol.name: rollout(p, pol, market, p.static_idx)
         for pol in (dp, MyopicPolicy(p), StaticPolicy(p))}
    for other in ("myopic", "static"):
        assert mc_estimate(r["dp"].total - r[other].total).ci_low > -1e-5
    pred = dp.predicted_value(np.full(market.n_paths, p.static_idx), market.obs[0])
    est = mc_estimate(r["dp"].total - pred)
    assert abs(est.mean) < 4 * est.se + 2e-5  # interpolation error allowance: 0.2 bp/episode
