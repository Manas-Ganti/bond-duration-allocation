"""`duralloc baseline | train | policymap | evaluate | figures`.

Run in that order. Each stage records gates and metrics as it goes; later stages refuse to run
on missing prerequisites rather than silently recomputing them.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import time

import numpy as np

from duralloc import checks
from duralloc.agent.fitted_q import train_fqi
from duralloc.baselines import DPPolicy, ImmunizePolicy, MyopicPolicy, StaticPolicy
from duralloc.config import REPO_ROOT, RESULTS_DIR, load_config
from duralloc.env import rollout, simulate_market
from duralloc.evaluate.costs import cost_sweep
from duralloc.evaluate.offmodel import offmodel_evaluation
from duralloc.evaluate.simulated import evaluate_policies
from duralloc.figures import all_figures
from duralloc.gates import (GateNotPassed, read_gates, read_metrics, record_gate, require_gates,
                            rng_for, update_metrics)
from duralloc.policy_map import compare_tables, policy_map, state_table
from duralloc.problem import Problem
from duralloc.stats import mc_estimate

POLICY_DIR = RESULTS_DIR / "policies"


def _sync_gates_md() -> None:
    """Rewrite the Status table in GATES.md from results/gates.json (criteria text untouched)."""
    path = REPO_ROOT / "GATES.md"
    g = read_gates()
    notes = {
        "A": lambda d: "max errors: mean {:.1f} bp, sigma {:.1%}, kappa {:.1%}, rho {:.3f}".format(
            max(d["error"]["mean_abs"]) * 1e4, max(d["error"]["sigma_rel"]),
            max(d["error"]["kappa_rel"]), d["error"]["rho_abs"]),
        "B": lambda d: "; ".join(
            f"{k}: err≤10bp move {v['max_err_small_moves'] * 1e4:.2f} bp, breaks at "
            f"{v['break_point_dy'] * 1e4:.0f} bp ({v['frac_sim_moves_beyond_break']:.0%} of sim moves)"
            for k, v in d["buckets"].items()),
        "C": lambda d: f"agreement {d['agreement']:.1%}, mean regret {d['mean_regret_bp']:.3f} bp",
        "D": lambda d: ("shortfall {:.2f} bp/yr (CI high {:.2f}) vs threshold {:.2f}; map within "
                        "one step {:.1%}").format(
            d["shortfall_bp_per_year"]["mean"], d["shortfall_bp_per_year"]["ci_high"],
            d["shortfall_threshold_bp_per_year"], d["map_agreement_within_step"]),
        "E": lambda d: d.get("note", ""),
    }
    rows = ["| Gate | Result | Date | Notes |", "|---|---|---|---|"]
    for k in "ABCDE":
        if k in g:
            rows.append(f"| {k} | {'PASS' if g[k]['passed'] else 'FAIL'} | {g[k]['date']} | "
                        f"{notes[k](g[k])} |")
        else:
            rows.append(f"| {k} | — | — | not yet run |")
    text = path.read_text()
    text = re.sub(r"## Status\n.*", "## Status\n\n" + "\n".join(rows) + "\n", text, flags=re.S)
    path.write_text(text)


def _save_policy(policy, name):
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    with open(POLICY_DIR / f"{name}.pkl", "wb") as fh:
        pickle.dump(policy, fh)


def _load_policy(name):
    path = POLICY_DIR / f"{name}.pkl"
    if not path.exists():
        sys.exit(f"missing {path}: run `duralloc train` first")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ------------------------------------------------------------------ stages
def cmd_baseline(cfg, args):
    seed = cfg.seed
    p = Problem(cfg)
    ok_a, det_a = checks.gate_a(p, rng_for(seed, "gate-a"))
    record_gate("A", ok_a, det_a, seed)
    ok_b, det_b = checks.gate_b(p, rng_for(seed, "gate-b"))
    record_gate("B", ok_b, det_b, seed)
    _sync_gates_md()

    t0 = time.time()
    dp = DPPolicy(p)
    t_dp = time.time() - t0
    c = cfg.dp.check
    dp_fine = DPPolicy(p, c.level_points, c.slope_points, c.quad_nodes)
    market = simulate_market(p, cfg.evaluation.n_paths, rng_for(seed, "eval-train"))
    pols = [StaticPolicy(p), ImmunizePolicy(p), MyopicPolicy(p), dp]
    ev = evaluate_policies(p, pols, market)
    k = 1e4 / (p.horizon * p.dt)
    r_dp = rollout(p, dp, market, p.static_idx)
    r_fine = rollout(p, dp_fine, market, p.static_idx)
    pred = dp.predicted_value(np.full(market.n_paths, p.static_idx), market.obs[0])
    ev["dp"]["grid_convergence"] = {
        "fine_minus_coarse_bp_per_year": mc_estimate(r_fine.total - r_dp.total).scaled(k).to_dict(),
        "coarse": cfg.dp.model_dump(exclude={"check"}), "fine": c.model_dump()}
    ev["dp"]["predicted_vs_mc_bp_per_year"] = mc_estimate(r_dp.total - pred).scaled(k).to_dict()
    ev["dp"]["solve_seconds"] = t_dp
    problem_info = {
        "n_actions": p.n_actions, "n_inband_allocations": int(len(p.states)),
        "static_allocation": p.actions[p.static_idx], "static_duration": p.action_duration[p.static_idx],
        "dp_states": int(p.n_actions * dp.grid.n), "horizon": p.horizon, "cost_bp": p.cost_bp,
        "bucket_durations": p.durations,
    }
    update_metrics("baselines", {"problem": problem_info, "in_model": ev,
                                 "n_paths": market.n_paths}, seed)
    for n, r in ev.items():
        print(f"  {n:9s} {r['value']['mean']:8.1f} bp/yr  "
              f"[{r['value']['ci_low']:.1f}, {r['value']['ci_high']:.1f}]")
    print(f"  DP coarse->fine change: {ev['dp']['grid_convergence']['fine_minus_coarse_bp_per_year']['mean']:.3f} bp/yr")
    if not (ok_a and ok_b):
        sys.exit("Gate A or B failed; stopping here.")


def cmd_train(cfg, args):
    seed = cfg.seed
    require_gates("A", "B", stage="training")
    pc = Problem(cfg, cost_bp=0.0, horizon=1)
    agent_c = train_fqi(pc, seed, verbose=args.verbose)
    ok_c, det_c = checks.gate_c(pc, agent_c)
    record_gate("C", ok_c, det_c, seed)
    _sync_gates_md()
    if not ok_c:
        sys.exit("Gate C failed; the agent is not reported. Stopping.")

    p = Problem(cfg)
    print(f"training fitted Q at {p.cost_bp:g} bp ...")
    fqi = train_fqi(p, seed, verbose=args.verbose)
    _save_policy(fqi, "fqi_main")
    print("training fitted Q at 0 bp (for the no-trade comparison) ...")
    p0 = Problem(cfg, cost_bp=0.0)
    _save_policy(train_fqi(p0, seed, verbose=args.verbose), "fqi_cost0")

    dp = DPPolicy(p)
    ok_d, det_d = checks.gate_d(p, dp, fqi, rng_for(seed, "eval-train"))
    record_gate("D", ok_d, det_d, seed)
    _sync_gates_md()
    update_metrics("train", {"fqi": fqi.info}, seed)
    print(f"  DP - FQI shortfall {det_d['shortfall_bp_per_year']['mean']:.2f} bp/yr "
          f"(threshold {det_d['shortfall_threshold_bp_per_year']:.2f}); "
          f"map agreement {det_d['map_agreement_within_step']:.1%}")


def cmd_policymap(cfg, args):
    seed = cfg.seed
    require_gates("A", "B", "C", stage="policy maps of the learned agent")
    p, p0 = Problem(cfg), Problem(cfg, cost_bp=0.0)
    dp, dp0 = DPPolicy(p), DPPolicy(p0)
    fqi, fqi0 = _load_policy("fqi_main"), _load_policy("fqi_cost0")
    m = p.model
    n = cfg.policymap.inner_n_std
    levels = np.linspace(m.mu[0] - n * m.stat_sd[0], m.mu[0] + n * m.stat_sd[0], 61)
    slopes = np.linspace(m.mu[1] - n * m.stat_sd[1], m.mu[1] + n * m.stat_sd[1], 61)
    ts = [min(t, p.horizon - 1) for t in cfg.policymap.t]
    maps = {name: {str(t): policy_map(pol, p, levels, slopes, p.static_idx, t) for t in ts}
            for name, pol in (("dp", dp), ("fqi", fqi))}
    inner = dp.grid.inner_mask(m.mu, m.stat_sd, n)
    pts = dp.grid.points[inner]
    agreement = {str(t): compare_tables(p, state_table(dp, p, pts, t), state_table(fqi, p, pts, t),
                                        cfg.gates.d.map_step_tol) for t in ts}
    hold = {"dp": {}, "fqi": {}}
    for c, (d_, f_, pp) in {"0": (dp0, fqi0, p0), f"{p.cost_bp:g}": (dp, fqi, p)}.items():
        hold["dp"][c] = policy_map(d_, pp, levels, slopes, p.static_idx, 0)
        hold["fqi"][c] = policy_map(f_, pp, levels, slopes, p.static_idx, 0)
    hold_stats = {c: compare_tables(pp, state_table(d_, pp, pts, 0), state_table(f_, pp, pts, 0),
                                    cfg.gates.d.map_step_tol)
                  for c, (d_, f_, pp) in {"0": (dp0, fqi0, p0), f"{p.cost_bp:g}": (dp, fqi, p)}.items()}
    update_metrics("policymap", {"maps": maps, "headline_t": ts[0], "agreement": agreement,
                                 "hold_regions": hold, "hold_stats_all_states": hold_stats,
                                 "cost_bp": p.cost_bp}, seed)
    for path in all_figures({"policymap": read_metrics()["policymap"]}):
        print(f"  wrote {path.relative_to(REPO_ROOT)}")
    for t, a in agreement.items():
        print(f"  t={t}: long weight within one step {a['long_within_step']:.1%}, exact {a['exact_match']:.1%}")
    for c, h in hold_stats.items():
        print(f"  cost {c} bp: hold fraction DP {h['hold_fraction_a']:.1%}  FQI {h['hold_fraction_b']:.1%}"
              f"  Jaccard {h['hold_jaccard']:.2f}")


def cmd_evaluate(cfg, args):
    seed = cfg.seed
    require_gates("A", "B", "C", stage="evaluation of the learned agent")
    p = Problem(cfg)
    fqi = _load_policy("fqi_main")
    pols = [StaticPolicy(p), ImmunizePolicy(p), MyopicPolicy(p), DPPolicy(p), fqi]
    parts = args.only or ["inmodel", "costs", "offmodel", "historical"]
    if "inmodel" in parts:
        print("in-model evaluation ...")
        market = simulate_market(p, cfg.evaluation.n_paths, rng_for(seed, "eval-inmodel"))
        ev = evaluate_policies(p, pols, market)
        update_metrics("in_model", {"policies": ev, "n_paths": market.n_paths,
                                    "cost_bp": p.cost_bp}, seed)
        for n, r in ev.items():
            adv = r.get("advantage_over_static", {}).get("mean", 0.0)
            print(f"  {n:9s} {r['value']['mean']:8.1f} bp/yr   vs static {adv:+7.2f}")
    if "costs" in parts:
        print("cost sweep (DP re-solved, agent re-trained at every level) ...")
        cs = cost_sweep(cfg, seed, cfg.costs.sweep_bp, cfg.evaluation.sweep_n_paths)
        update_metrics("cost_sweep", cs, seed)
        print(f"  break-even (bp): {cs['breakeven_bp']}")
    if "offmodel" in parts:
        print("off-model evaluation ...")
        om = offmodel_evaluation(p, pols, seed, cfg.evaluation.sweep_n_paths)
        update_metrics("offmodel", om, seed)
        for w, r in om["advantage_retention"].items():
            print(f"  {w:10s} retention " + "  ".join(f"{n}={v:.2f}" for n, v in r.items() if v is not None))
    if "historical" in parts:
        try:
            require_gates("A", "B", "C", "D", stage="historical evaluation (Gate E)")
        except GateNotPassed as e:
            record_gate("E", False, {"note": "historical data not loaded: A-D not all passed"}, seed)
            _sync_gates_md()
            print(f"  {e}")
        else:
            from duralloc.evaluate.historical import historical_evaluation
            print("historical evaluation (Gate E open) ...")
            h = historical_evaluation(p, pols, seed)
            update_metrics("historical", h, seed)
            record_gate("E", True, {"note": f"FRED first loaded after A-D passed; {h['period'][0]} to "
                                             f"{h['period'][1]}"}, seed)
            _sync_gates_md()
            for n, r in h["walk"].items():
                if "advantage_over_static" in r:
                    e = r["advantage_over_static"]
                    print(f"  {n:9s} walk advantage {e['mean']:+7.1f} bp/yr [{e['ci_low']:.1f}, {e['ci_high']:.1f}]")
    for path in all_figures(read_metrics()):
        print(f"  wrote {path.relative_to(REPO_ROOT)}")


def cmd_figures(cfg, args):
    for path in all_figures(read_metrics()):
        print(f"  wrote {path.relative_to(REPO_ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="duralloc")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline", help="Gates A, B; all classical policies incl. the DP optimum")
    t = sub.add_parser("train", help="Gate C, train fitted Q, Gate D")
    t.add_argument("--verbose", action="store_true")
    sub.add_parser("policymap", help="headline figure: DP vs learned over (level, slope)")
    e = sub.add_parser("evaluate", help="in-model, cost sweep, off-model, historical (Gate E)")
    e.add_argument("--only", nargs="+", choices=["inmodel", "costs", "offmodel", "historical"])
    sub.add_parser("figures", help="redraw every figure from results/metrics.json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"baseline": cmd_baseline, "train": cmd_train, "policymap": cmd_policymap,
     "evaluate": cmd_evaluate, "figures": cmd_figures}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
