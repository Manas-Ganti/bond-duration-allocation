"""Evaluation on actual Treasury yield-change paths (after Gate E).

The policies were trained on simulated curves only. Here they observe the two-factor projection
of the real curve each month and hold buckets priced off the real curve (bucket yields are
linearly interpolated in maturity between the available constant-maturity tenors).

Two views, both reported:
  * a walk through the whole history as consecutive 24-month episodes, giving a monthly
    advantage series whose mean gets a calendar-year cluster-bootstrap CI;
  * every overlapping 24-month window started from the static allocation, reported as a
    distribution (descriptive: overlapping windows are not independent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from duralloc.curve import calibrate
from duralloc.data.fred import load_monthly_yields
from duralloc.env import MarketPaths, rollout, trailing_vol
from duralloc.gates import rng_for
from duralloc.policy import Policy
from duralloc.problem import Problem
from duralloc.stats import cluster_bootstrap_mean


def interpolate_buckets(yields: np.ndarray, tenors: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    out = np.full((len(yields), len(maturities)), np.nan)
    for i, row in enumerate(yields):
        ok = ~np.isnan(row)
        if ok.sum() >= 2:
            out[i] = np.interp(maturities, tenors[ok], row[ok])
    return out


def historical_market(problem: Problem, monthly: pd.DataFrame, tenors: np.ndarray):
    Y = monthly.to_numpy()
    obs = problem.model.project(Y, tenors)
    buckets = interpolate_buckets(Y, tenors, problem.maturities)
    keep = ~(np.isnan(obs).any(1) | np.isnan(buckets).any(1))
    idx = monthly.index[keep]
    obs, buckets = obs[keep], buckets[keep]
    return idx, obs, buckets


def _market(problem, obs, buckets, label) -> MarketPaths:
    o = obs[:, None, :]
    return MarketPaths(buckets[:, None, :], o, trailing_vol(o[..., 0], problem), label)


def historical_evaluation(problem: Problem, policies: list[Policy], seed: int) -> dict:
    cfg = problem.cfg
    monthly, tenors = load_monthly_yields(cfg)
    idx, obs, buckets = historical_market(problem, monthly, tenors)
    T = problem.horizon
    k_year = 1e4 * 12  # monthly decimal -> bp per year
    m = problem.model

    # --- consecutive-episode walk
    walk = _market(problem, obs, buckets, "historical")
    res = {p.name: rollout(problem, p, walk, problem.static_idx, horizon_reset=T) for p in policies}
    months = idx[1:]
    years = np.asarray(months.year)
    walk_out = {}
    for name, r in res.items():
        row = {"mean_return_bp_per_year": float(r.rewards[:, 0].mean() * k_year),
               "decomposition_bp_per_year": {c: float(getattr(r, c)[0] / len(months) * k_year)
                                             for c in ("carry", "duration", "cost", "penalty")},
               "turnover_per_year": float(r.turnover[0] / len(months) * 12),
               "violations": r.violations}
        if name != "static":
            diff = r.rewards[:, 0] - res["static"].rewards[:, 0]
            est = cluster_bootstrap_mean(diff, years, rng_for(seed, f"hist-boot-{name}"),
                                         cfg.historical.n_boot, cfg.evaluation.ci_level)
            row["advantage_over_static"] = est.scaled(k_year).to_dict()
            row["cum_advantage_bp"] = (np.cumsum(diff) * 1e4).tolist()
            decade = (years // 10) * 10
            row["advantage_by_decade_bp_per_year"] = {
                str(d): float(diff[decade == d].mean() * k_year) for d in np.unique(decade)}
        walk_out[name] = row
    if "dp" in res and "fqi" in res:
        diff = res["fqi"].rewards[:, 0] - res["dp"].rewards[:, 0]
        walk_out["fqi"]["gap_to_dp"] = cluster_bootstrap_mean(
            diff, years, rng_for(seed, "hist-boot-gap"), cfg.historical.n_boot,
            cfg.evaluation.ci_level).scaled(k_year).to_dict()

    # --- overlapping windows, as a distribution
    starts = np.arange(0, len(obs) - T)
    win = np.stack([np.arange(s, s + T + 1) for s in starts], 1)  # (T+1, n_windows)
    wm = MarketPaths(buckets[win], obs[win], trailing_vol(obs[win][..., 0], problem), "hist-windows")
    wres = {p.name: rollout(problem, p, wm, problem.static_idx) for p in policies}
    k_win = 1e4 * 12 / T
    windows = {"start": [str(d.date()) for d in idx[starts]]}
    dist = {}
    for name, r in wres.items():
        windows[name] = (r.total * k_win).tolist()
        if name != "static":
            adv = (r.total - wres["static"].total) * k_win
            windows[name + "_advantage"] = adv.tolist()
            dist[name] = {"quantiles": dict(zip(["p05", "p25", "p50", "p75", "p95"],
                                                np.quantile(adv, [.05, .25, .5, .75, .95]).tolist())),
                          "frac_positive": float((adv > 0).mean()), "mean": float(adv.mean())}

    # --- how far off-model is history?
    z = np.abs((obs - m.mu) / m.stat_sd)
    calib = calibrate(m, monthly.loc[idx].to_numpy(), tenors)
    return {
        "period": [str(idx[0].date()), str(idx[-1].date())], "n_months": int(len(idx)),
        "tenors": tenors.tolist(),
        "walk": walk_out,
        "windows_advantage_distribution_bp_per_year": dist,
        "n_windows": int(len(starts)),
        "frac_months_outside_dp_grid": float((z > cfg.dp.n_std).any(1).mean()),
        "frac_months_outside_fqi_box": float((z > cfg.fqi.box_n_std).any(1).mean()),
        "calibration_vs_prior": {
            "estimated": calib,
            "prior": {"mu": m.mu.tolist(), "kappa": m.kappa.tolist(),
                      "sigma": m.sigma.tolist(), "rho": m.rho},
        },
        "series": {"month": [str(d.date()) for d in idx], "level": obs[:, 0].tolist(),
                   "slope": obs[:, 1].tolist()},
        "windows": windows,
    }
