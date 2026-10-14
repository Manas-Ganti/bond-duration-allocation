"""Trained on two factors, evaluated where that is no longer true.

Each stress world is simulated from the same stationary start; policies keep observing only
the two-factor projection of the curve. Degradation is measured *relatively*: how much of each
policy's in-model advantage over the static allocation survives, and whether the learned
policy loses more of it than DP does.
"""

from __future__ import annotations

from duralloc.env import simulate_market
from duralloc.evaluate.simulated import evaluate_policies
from duralloc.gates import rng_for
from duralloc.policy import Policy
from duralloc.problem import Problem

WORLDS = ("train", "curvature", "regime", "jump")


def offmodel_evaluation(problem: Problem, policies: list[Policy], seed: int, n_paths: int) -> dict:
    out = {}
    for world in WORLDS:
        market = simulate_market(problem, n_paths, rng_for(seed, f"offmodel-{world}"), world)
        out[world] = evaluate_policies(problem, policies, market)
    base = {n: out["train"][n]["advantage_over_static"]["mean"]
            for n in out["train"] if n != "static"}
    retention = {}
    for world in WORLDS:
        retention[world] = {
            n: (out[world][n]["advantage_over_static"]["mean"] / base[n]) if base[n] else None
            for n in base}
    return {"worlds": out, "advantage_retention": retention, "n_paths": n_paths}
