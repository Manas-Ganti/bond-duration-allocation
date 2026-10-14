import numpy as np
import pytest

from duralloc.agent.fitted_q import train_fqi
from duralloc.baselines import DPPolicy, ImmunizePolicy, MyopicPolicy, StaticPolicy
from duralloc.env import rollout, simulate_market
from duralloc.policy import FunctionPolicy
from duralloc.problem import Problem


def test_action_grid_respects_position_limit_and_simplex(problem):
    a = problem.actions
    assert (a >= -1e-12).all() and (a <= problem.cfg.portfolio.max_weight + 1e-9).all()
    np.testing.assert_allclose(a.sum(1), 1.0)


def test_every_state_has_a_feasible_action_including_hold(problem):
    for w in problem.states:
        assert problem.feasible[w, w]


def test_static_allocation_is_compliant(problem):
    assert problem.band_ok[problem.static_idx]


@pytest.fixture(scope="module")
def short_problem(cfg):
    return Problem(cfg, horizon=4)


@pytest.fixture(scope="module")
def all_policies(short_problem):
    p = short_problem
    return [StaticPolicy(p), ImmunizePolicy(p), MyopicPolicy(p), DPPolicy(p),
            train_fqi(p, seed=1, n_samples=30000)]


@pytest.mark.parametrize("variant", ["train", "curvature", "regime", "jump"])
def test_no_policy_ever_violates_limits(short_problem, all_policies, variant):
    market = simulate_market(short_problem, 5000, np.random.default_rng(3), variant)
    for pol in all_policies:
        r = rollout(short_problem, pol, market, short_problem.static_idx)
        assert r.violations == {"turnover": 0, "band": 0, "position": 0}, pol.name
        assert (r.penalty == 0).all()


def test_violating_policy_is_caught(short_problem):
    p = short_problem
    out_of_band = int(np.nonzero(~p.band_ok)[0][0])
    bad = FunctionPolicy(p, lambda t, w, f: np.full(len(w), out_of_band), "bad")
    market = simulate_market(p, 100, np.random.default_rng(0))
    r = rollout(p, bad, market, p.static_idx)
    assert r.violations["band"] > 0 and (r.penalty > 0).all()
