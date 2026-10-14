"""Transaction-cost sweep: at what cost does the dynamic policy stop beating the static one?

At every cost level DP is re-solved and the agent re-trained, then both (and the classical
baselines) are evaluated against the static allocation on one common set of paths. The
break-even reported is the smallest swept cost at which the advantage over static is no longer
significantly positive (95% CI lower bound <= 0). DP can always replicate static, so its
advantage reaches zero exactly when it stops trading.
"""

from __future__ import annotations

import numpy as np

from duralloc.agent.fitted_q import train_fqi
from duralloc.baselines import DPPolicy, ImmunizePolicy, MyopicPolicy, StaticPolicy
from duralloc.checks import eval_grid
from duralloc.env import simulate_market
from duralloc.evaluate.simulated import evaluate_policies
from duralloc.gates import rng_for
from duralloc.policy_map import compare_tables, state_table
from duralloc.problem import Problem


def cost_sweep(cfg, seed: int, costs_bp: list[float], n_paths: int, verbose: bool = True) -> dict:
    base = Problem(cfg)
    market = simulate_market(base, n_paths, rng_for(seed, "cost-sweep"))
    pts = eval_grid(base, 21, cfg.policymap.inner_n_std)
    rows = []
    for c in costs_bp:
        p = Problem(cfg, cost_bp=c)
        dp, fqi = DPPolicy(p), train_fqi(p, seed)
        pols = [StaticPolicy(p), ImmunizePolicy(p), MyopicPolicy(p), dp, fqi]
        ev = evaluate_policies(p, pols, market)
        holds = compare_tables(p, state_table(dp, p, pts, 0), state_table(fqi, p, pts, 0),
                               cfg.gates.d.map_step_tol)
        rows.append({"cost_bp": c, "policies": ev, "hold_region_t0": holds})
        if verbose:
            adv = {n: ev[n]["advantage_over_static"]["mean"] for n in ("dp", "fqi", "myopic")}
            print(f"  cost {c:6.1f}bp  advantage bp/yr " +
                  "  ".join(f"{n}={v:7.2f}" for n, v in adv.items()) +
                  f"  hold dp={holds['hold_fraction_a']:.2f} fqi={holds['hold_fraction_b']:.2f}")
    breakeven = {}
    for n in ("dp", "fqi", "myopic", "immunize"):
        lo = [r["policies"][n]["advantage_over_static"]["ci_low"] for r in rows]
        hit = [r["cost_bp"] for r, l in zip(rows, lo) if l <= 0]
        breakeven[n] = hit[0] if hit else None
    return {"rows": rows, "breakeven_bp": breakeven, "n_paths": n_paths,
            "definition": "smallest swept cost with 95% CI lower bound of advantage over static <= 0"}


def max_cost_tested(rows) -> float:
    return float(np.max([r["cost_bp"] for r in rows]))
