"""Gate computations. Pure functions returning (passed, details); the CLI records them and the
tests call them at reduced size, so a gate's logic is itself under test."""

from __future__ import annotations

import numpy as np

from duralloc.baselines import DPPolicy, MyopicPolicy, StaticPolicy
from duralloc.bonds import approx_return, approximation_break_point, exact_zero_return
from duralloc.curve import ar1_moments
from duralloc.env import rollout, simulate_market
from duralloc.policy import Policy
from duralloc.policy_map import compare_tables, state_table
from duralloc.problem import Problem
from duralloc.stats import mc_estimate


def gate_a(problem: Problem, rng: np.random.Generator, n_paths=None, n_steps=None) -> tuple[bool, dict]:
    g = problem.cfg.gates.a
    m = problem.model
    paths = m.simulate(n_paths or g.n_paths, n_steps or g.n_steps, rng)
    est = ar1_moments(m.observe(paths), m.dt)
    target = {"mean": m.mu.tolist(), "kappa": m.kappa.tolist(), "sigma": m.sigma.tolist(),
              "rho": m.rho}
    err = {
        "mean_abs": [abs(a - b) for a, b in zip(est["mean"], target["mean"])],
        "sigma_rel": [abs(a / b - 1) for a, b in zip(est["sigma"], target["sigma"])],
        "kappa_rel": [abs(a / b - 1) for a, b in zip(est["kappa"], target["kappa"])],
        "rho_abs": abs(est["rho"] - target["rho"]),
    }
    checks = {
        "mean": max(err["mean_abs"]) <= g.mean_abs_tol,
        "sigma": max(err["sigma_rel"]) <= g.sigma_rel_tol,
        "kappa": max(err["kappa_rel"]) <= g.kappa_rel_tol,
        "rho": err["rho_abs"] <= g.rho_abs_tol,
    }
    return all(checks.values()), {"target": target, "estimate": est, "error": err,
                                  "checks": checks}


def gate_b(problem: Problem, rng: np.random.Generator, n_paths: int = 2000) -> tuple[bool, dict]:
    g = problem.cfg.gates.b
    dt = problem.dt
    small = np.linspace(-g.small_move, g.small_move, 401)
    market = simulate_market(problem, n_paths, rng)
    dy_sim = np.diff(market.bucket_yields, axis=0)  # (T, N, 3)
    y_sim = market.bucket_yields[:-1]
    out, ok = {}, True
    for i, (name, m) in enumerate(zip(problem.cfg.buckets.names, problem.maturities)):
        err_small = np.abs(exact_zero_return(g.base_yield, small, m, dt)
                           - approx_return(g.base_yield, small, m, dt)).max()
        bp = approximation_break_point(m, g.base_yield, dt, g.tol, g.scan_max)
        moves = np.abs(dy_sim[..., i])
        # signed error on simulated moves: the bias the first-order model carries
        bias = exact_zero_return(y_sim[..., i], dy_sim[..., i], m, dt) - \
            approx_return(y_sim[..., i], dy_sim[..., i], m, dt)
        passed = bool(err_small <= g.tol)
        ok &= passed
        out[name] = {
            "maturity": float(m), "max_err_small_moves": float(err_small), "passed": passed,
            "break_point_dy": bp,
            "frac_sim_moves_beyond_break": float((moves > bp).mean()),
            "sim_monthly_move_sd": float(dy_sim[..., i].std()),
            "mean_bias_bp_per_year": float(bias.mean() * 1e4 / dt),
        }
    return ok, {"tolerance": g.tol, "small_move": g.small_move, "base_yield": g.base_yield,
                "buckets": out}


def eval_grid(problem: Problem, n_points: int, n_std: float) -> np.ndarray:
    m = problem.model
    lv = np.linspace(m.mu[0] - n_std * m.stat_sd[0], m.mu[0] + n_std * m.stat_sd[0], n_points)
    sl = np.linspace(m.mu[1] - n_std * m.stat_sd[1], m.mu[1] + n_std * m.stat_sd[1], n_points)
    L, S = np.meshgrid(lv, sl, indexing="ij")
    return np.column_stack([L.ravel(), S.ravel()])


def gate_c(problem_c: Problem, agent: Policy) -> tuple[bool, dict]:
    """problem_c must have zero cost and a one-period horizon."""
    assert problem_c.cost_bp == 0 and problem_c.horizon == 1
    g = problem_c.cfg.gates.c
    pts = eval_grid(problem_c, g.grid_points, g.n_std)
    myopic = MyopicPolicy(problem_c)
    a_agent = state_table(agent, problem_c, pts, 0)
    a_myo = state_table(myopic, problem_c, pts, 0)
    R = problem_c.expected_action_returns(pts)  # (M, nA)
    cols = np.arange(len(pts))[None, :]
    regret = R[cols, a_myo] - R[cols, a_agent]
    agreement = float((a_agent == a_myo).mean())
    mean_regret_bp = float(regret.mean() * 1e4)
    passed = agreement >= g.min_agreement and mean_regret_bp <= g.max_mean_regret_bp
    return passed, {"agreement": agreement, "mean_regret_bp": mean_regret_bp,
                    "max_regret_bp": float(regret.max() * 1e4), "n_states": int(regret.size),
                    "min_agreement": g.min_agreement, "max_mean_regret_bp": g.max_mean_regret_bp}


def gate_d(problem: Problem, dp: DPPolicy, agent: Policy, rng: np.random.Generator,
           n_paths: int | None = None, map_t: list[int] | None = None) -> tuple[bool, dict]:
    g = problem.cfg.gates.d
    ci = problem.cfg.evaluation.ci_level
    market = simulate_market(problem, n_paths or problem.cfg.evaluation.n_paths, rng)
    res = {p.name: rollout(problem, p, market, problem.static_idx)
           for p in (dp, agent, StaticPolicy(problem))}
    shortfall = mc_estimate(res["dp"].total - res[agent.name].total, ci)
    advantage = mc_estimate(res["dp"].total - res["static"].total, ci)
    value_ok = shortfall.ci_high <= g.max_shortfall_frac * advantage.mean

    grid = dp.grid
    inner = grid.inner_mask(problem.model.mu, problem.model.stat_sd, problem.cfg.policymap.inner_n_std)
    pts = grid.points[inner]
    maps = {}
    within = []
    regrets = []
    for t in (map_t if map_t is not None else problem.cfg.policymap.t):
        t = min(t, problem.horizon - 1)
        a_dp = state_table(dp, problem, pts, t)
        a_ag = state_table(agent, problem, pts, t)
        cmp = compare_tables(problem, a_dp, a_ag, g.map_step_tol)
        Q = dp.solution.q(t, problem.cost_matrix)[problem.states][:, inner]  # (nS, M, nA)
        r = np.take_along_axis(Q, a_dp[..., None], -1) - np.take_along_axis(Q, a_ag[..., None], -1)
        cmp["mean_regret_bp_grid"] = float(r.mean() * 1e4)
        maps[str(t)] = cmp
        within.append(cmp["long_within_step"])
        regrets.append(r.mean())
    map_agreement = float(np.mean(within))
    map_ok = map_agreement >= g.min_map_agreement
    k = 1e4 / (problem.horizon * problem.dt)  # episode total -> bp per year
    return bool(value_ok and map_ok), {
        "value_check_passed": bool(value_ok), "map_check_passed": bool(map_ok),
        "shortfall_bp_per_year": shortfall.scaled(k).to_dict(),
        "dp_advantage_over_static_bp_per_year": advantage.scaled(k).to_dict(),
        "shortfall_threshold_bp_per_year": g.max_shortfall_frac * advantage.mean * k,
        "map_agreement_within_step": map_agreement, "map_by_t": maps,
        "n_paths": market.n_paths, "cost_bp": problem.cost_bp,
    }
