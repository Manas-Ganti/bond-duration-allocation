"""Policy value over many simulated paths, with confidence intervals.

All policies run over the same paths, so comparisons are paired. Values are reported in basis
points per year (episode total annualised), and always decomposed.
"""

from __future__ import annotations

import numpy as np

from duralloc.env import MarketPaths, RolloutResult, rollout
from duralloc.policy import Policy
from duralloc.problem import Problem
from duralloc.stats import mc_estimate


def bp_per_year(problem: Problem, n_steps: int | None = None) -> float:
    return 1e4 / ((n_steps or problem.horizon) * problem.dt)


def summarise(problem: Problem, results: dict[str, RolloutResult], n_steps: int,
              reference: str = "static", ci: float = 0.95) -> dict:
    k = bp_per_year(problem, n_steps)
    out = {}
    for name, r in results.items():
        row = {
            "value": mc_estimate(r.total, ci).scaled(k).to_dict(),
            "decomposition_bp_per_year": {c: float(getattr(r, c).mean() * k)
                                          for c in ("carry", "duration", "cost", "penalty")},
            "turnover_per_year": float(r.turnover.mean() * 12 / n_steps),
            "violations": r.violations,
        }
        if reference in results and name != reference:
            row["advantage_over_" + reference] = mc_estimate(
                r.total - results[reference].total, ci).scaled(k).to_dict()
        out[name] = row
    if "dp" in results and "fqi" in results:
        out["fqi"]["gap_to_dp"] = mc_estimate(
            results["fqi"].total - results["dp"].total, ci).scaled(k).to_dict()
    return out


def evaluate_policies(problem: Problem, policies: list[Policy], market: MarketPaths,
                      ci: float | None = None) -> dict:
    ci = ci or problem.cfg.evaluation.ci_level
    results = {p.name: rollout(problem, p, market, problem.static_idx) for p in policies}
    return summarise(problem, results, market.n_steps, ci=ci)
