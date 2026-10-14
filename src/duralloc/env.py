"""Gymnasium interface, plus the vectorised market/rollout machinery every evaluation uses.

The market does not react to the portfolio, so a set of curve paths can be generated once and
every policy run over the *same* paths (common random numbers): differences between policies
are then paired, and their confidence intervals are tight.

The Gym environment and the vectorised rollout share `Problem.decompose`; a test checks that
they produce identical rewards.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from duralloc.curve import CurvePaths
from duralloc.policy import Policy
from duralloc.problem import Problem


@dataclass
class MarketPaths:
    """bucket_yields (T+1, N, 3); obs (T+1, N, 2) observed (level, slope); vol (T+1, N)."""
    bucket_yields: np.ndarray
    obs: np.ndarray
    vol: np.ndarray
    label: str = "train"

    @property
    def n_steps(self) -> int:
        return self.bucket_yields.shape[0] - 1

    @property
    def n_paths(self) -> int:
        return self.bucket_yields.shape[1]


def trailing_vol(obs_level: np.ndarray, problem: Problem) -> np.ndarray:
    """Annualised EWMA volatility of observed level changes, (T+1, N)."""
    lam = problem.cfg.curve.vol_ewma
    v = np.empty_like(obs_level)
    v[0] = problem.model.step_cov[0, 0]
    d = np.diff(obs_level, axis=0, prepend=obs_level[:1])
    for t in range(1, len(v)):
        v[t] = lam * v[t - 1] + (1 - lam) * d[t] ** 2
    return np.sqrt(v / problem.dt)


def market_from_curve(problem: Problem, paths: CurvePaths) -> MarketPaths:
    obs = problem.model.observe(paths)
    return MarketPaths(problem.bucket_yields(paths.factors), obs,
                       trailing_vol(obs[..., 0], problem), paths.variant)


def simulate_market(problem: Problem, n_paths: int, rng: np.random.Generator,
                    variant: str = "train", n_steps: int | None = None) -> MarketPaths:
    paths = problem.model.simulate(n_paths, n_steps or problem.horizon, rng, variant)
    return market_from_curve(problem, paths)


def sample_transitions(problem: Problem, f: np.ndarray, rng: np.random.Generator):
    """One market step from arbitrary start states f (N, 2) under the training process.

    Returns what an agent observes: (obs, bucket returns decomposed as y0/y1, next obs).
    This is the black box the fitted-Q agent learns from; it never reads OU parameters.
    """
    m = market_from_curve(problem, problem.model.simulate(len(f), 1, rng, "train", f0=f))
    return m.obs[0], m.bucket_yields[0], m.bucket_yields[1], m.obs[1]


@dataclass
class RolloutResult:
    carry: np.ndarray
    duration: np.ndarray
    cost: np.ndarray
    penalty: np.ndarray
    total: np.ndarray
    turnover: np.ndarray
    violations: dict
    actions: np.ndarray  # (T, N)
    rewards: np.ndarray  # (T, N) per-step total


def rollout(problem: Problem, policy: Policy, market: MarketPaths, w0_idx: int | np.ndarray,
            t_offset: int = 0, horizon_reset: int | None = None) -> RolloutResult:
    """Run `policy` over `market`. `horizon_reset` restarts the policy clock every k steps
    (used for long historical walks made of consecutive episodes)."""
    T, N = market.n_steps, market.n_paths
    w = np.broadcast_to(np.asarray(w0_idx), (N,)).astype(int).copy()
    comps = {k: np.zeros(N) for k in ("carry", "duration", "cost", "penalty")}
    turnover = np.zeros(N)
    viol = {"turnover": 0, "band": 0, "position": 0}
    actions = np.zeros((T, N), dtype=np.int16)
    rewards = np.zeros((T, N))
    for t in range(T):
        clock = (t + t_offset) % horizon_reset if horizon_reset else t + t_offset
        a = policy.act(clock, w, market.obs[t])
        d = problem.decompose(w, a, market.bucket_yields[t], market.bucket_yields[t + 1])
        for k in comps:
            comps[k] += getattr(d, k)
        rewards[t] = d.total
        turnover += problem.turnover[w, a]
        for k, v in problem.violations(w, a).items():
            viol[k] += int(np.sum(v))
        actions[t] = a
        w = a
    total = comps["carry"] + comps["duration"] - comps["cost"] - comps["penalty"]
    return RolloutResult(**comps, total=total, turnover=turnover, violations=viol,
                         actions=actions, rewards=rewards)


class DurationAllocEnv(gym.Env):
    """Single-episode Gymnasium environment under the training process.

    Observation: [w_short, w_int, w_long, portfolio duration, level, slope, trailing vol,
    last turnover, t / T]. Action: index into `problem.actions`; `action_masks()` gives the
    feasible set. Infeasible actions are executed and reported in `info["violations"]` (and the
    duration penalty enters the reward); refusing them is the policy's job, checked by tests.
    """

    metadata = {"render_modes": []}

    def __init__(self, problem: Problem):
        self.problem = problem
        self.action_space = gym.spaces.Discrete(problem.n_actions)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float64)
        self._rng = np.random.default_rng()

    def _obs(self):
        p = self.problem
        w = p.actions[self.w]
        return np.array([*w, p.action_duration[self.w], *self.f_obs, self.vol,
                         self.last_turnover, self.t / p.horizon], dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        p = self.problem
        f = options.get("factors")
        self.f = np.asarray(f, float) if f is not None else p.model.sample_stationary(1, self._rng)[0]
        self.f_obs = self.f.copy()
        self.w = int(options.get("w_idx", p.static_idx))
        self.t = int(options.get("t", 0))
        self.var = p.model.step_cov[0, 0]
        self.vol = float(np.sqrt(self.var / p.dt))
        self.last_turnover = 0.0
        return self._obs(), {}

    def action_masks(self) -> np.ndarray:
        return self.problem.feasible[self.w].copy()

    def step(self, action):
        p = self.problem
        a = int(action)
        obs0, y0, y1, obs1 = sample_transitions(p, self.f[None], self._rng)
        d = p.decompose(np.array([self.w]), np.array([a]), y0, y1)
        viol = {k: bool(v[0]) for k, v in p.violations(np.array([self.w]), np.array([a])).items()}
        self.last_turnover = float(p.turnover[self.w, a])
        lam = p.cfg.curve.vol_ewma
        self.var = lam * self.var + (1 - lam) * float(obs1[0, 0] - self.f_obs[0]) ** 2
        self.vol = float(np.sqrt(self.var / p.dt))
        self.f, self.f_obs, self.w = obs1[0].copy(), obs1[0].copy(), a
        self.t += 1
        info = {"carry": float(d.carry[0]), "duration": float(d.duration[0]),
                "cost": float(d.cost[0]), "penalty": float(d.penalty[0]),
                "bucket_returns": (y0[0] * p.dt - (y1[0] - y0[0]) * p.durations),
                "violations": viol}
        return self._obs(), float(d.total[0]), self.t >= p.horizon, False, info
