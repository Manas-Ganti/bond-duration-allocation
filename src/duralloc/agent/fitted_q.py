"""Fitted Q-iteration for the finite-horizon allocation problem.

What the agent is given:
  * sampled transitions from the simulator: observed (level, slope), the realised one-month
    returns of the three buckets, and the next observed (level, slope);
  * its own cost schedule and constraints (the trading desk knows what it pays to trade).
What it is NOT given: the OU parameters, the yield-curve loadings, or the return formula. The
expected return and the continuation value are learned by regression from realised returns.

Two structural facts are used, both exact for this problem and both stated in the README:
  1. Q_t(w, f, a) = -cost(w, a) + G_t(f, a): the next state depends on the target allocation,
     not on where it came from, so only G has to be learned.
  2. The market does not react to the portfolio, and every bucket's return is observed whatever
     is held, so one sampled transition gives a regression target for *every* allocation
     ("all-action" updates). Targets for different actions share the same market noise, which
     is what makes their differences, and therefore the argmax, well estimated.

G_t is linear in normalised (partition-of-unity) Gaussian RBFs of the standardised curve state
plus linear terms, one coefficient vector per action, fitted by ridge regression, backwards
from t = T-1 with independent samples at each stage. States are drawn uniformly over a box of
+/- `box_n_std` stationary standard deviations (exploring starts).
"""

from __future__ import annotations

import time
import zlib

import numpy as np

from duralloc.env import sample_transitions
from duralloc.policy import Policy
from duralloc.problem import Problem


class RBFFeatures:
    def __init__(self, center, sd, box_n_std: float, n_level: int, n_slope: int, width: float):
        self.center, self.sd = np.asarray(center, float), np.asarray(sd, float)
        cl = np.linspace(-box_n_std, box_n_std, n_level)
        cs = np.linspace(-box_n_std, box_n_std, n_slope)
        self.centres = np.stack(np.meshgrid(cl, cs, indexing="ij"), -1).reshape(-1, 2)
        self.width = width * np.array([cl[1] - cl[0], cs[1] - cs[0]])
        self.n_features = len(self.centres) + 2

    def __call__(self, f: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(f)[:, :2] - self.center) / self.sd
        d2 = (((z[:, None, :] - self.centres[None]) / self.width) ** 2).sum(-1)
        g = np.exp(-0.5 * (d2 - d2.min(1, keepdims=True)))  # shift for stability; cancels below
        g /= g.sum(1, keepdims=True)
        return np.hstack([g, z])


def continuation(problem: Problem, g_next: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """V(a, f') = max_{a' feasible from a} [G(f', a') - cost(a, a')]  for every a. (N, nA)."""
    out = np.empty_like(g_next)
    cost = problem.cost_matrix[None]
    mask = problem.feasible[None]
    for s in range(0, len(g_next), chunk):
        q = g_next[s:s + chunk, None, :] - cost
        out[s:s + chunk] = np.where(mask, q, -np.inf).max(-1)
    return out


class FQIPolicy(Policy):
    name = "fqi"

    def __init__(self, problem: Problem, features: RBFFeatures, theta: np.ndarray, info: dict):
        super().__init__(problem)
        self.features, self.theta, self.info = features, theta, info

    def g(self, t, f):
        return self.features(f) @ self.theta[t]


def train_fqi(problem: Problem, seed: int, n_samples: int | None = None,
              verbose: bool = False) -> FQIPolicy:
    c = problem.cfg.fqi
    m = problem.model
    n = n_samples or c.n_samples
    feats = RBFFeatures(m.mu, m.stat_sd, c.box_n_std, c.rbf_level, c.rbf_slope, c.rbf_width)
    T, nA, K = problem.horizon, problem.n_actions, feats.n_features
    theta = np.zeros((T, K, nA))
    lo, hi = m.mu - c.box_n_std * m.stat_sd, m.mu + c.box_n_std * m.stat_sd
    r2 = []
    t0 = time.time()
    for t in range(T - 1, -1, -1):
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, zlib.crc32(b"fqi"), int(problem.cost_bp * 1000), T, t]))
        f = lo + (hi - lo) * rng.random((n, 2))
        obs0, y0, y1, obs1 = sample_transitions(problem, f, rng)
        bucket_r = y0 * problem.dt - (y1 - y0) * problem.durations
        Y = bucket_r @ problem.actions.T - problem.action_penalty
        if t < T - 1:
            Y += continuation(problem, feats(obs1) @ theta[t + 1])
        X = feats(obs0)
        A = X.T @ X + c.ridge * n * np.eye(K)
        theta[t] = np.linalg.solve(A, X.T @ Y)
        resid = Y - X @ theta[t]
        r2.append(float(1 - resid.var() / Y.var()))
        if verbose:
            print(f"  fqi stage t={t:2d}  R^2={r2[-1]:.4f}  ({time.time() - t0:.1f}s)")
    info = {"n_samples_per_stage": n, "n_features": K, "horizon": T,
            "cost_bp": problem.cost_bp, "r2_by_stage": r2[::-1],
            "train_seconds": time.time() - t0}
    return FQIPolicy(problem, feats, theta, info)
