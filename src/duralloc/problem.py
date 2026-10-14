"""The decision problem: action grid, constraints, costs and the reward decomposition.

Action space: target weights on a discretised simplex over (short, intermediate, long), already
restricted to the maximum position weight. The duration band and the turnover limit depend on
the current portfolio, so they form a feasibility mask `feasible[w_prev, a]` that every policy
must respect. Constraint compliance is checked by tests (tests/test_constraints.py), not by
assertions in the environment: the environment executes whatever it is given and reports
violations, and the duration penalty is part of the reward.

Simplification, named in the README: weights do not drift between rebalances, so the state's
"current weights" is the previous target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from duralloc.bonds import zero_modified_duration
from duralloc.config import Config
from duralloc.curve import CurveModel


def simplex_grid(step: float, n_assets: int = 3) -> np.ndarray:
    units = int(round(1.0 / step))
    pts = [(i, j, units - i - j) for i in range(units + 1) for j in range(units + 1 - i)]
    return np.array(pts, dtype=float) / units


@dataclass
class Decomposition:
    """Per-step reward components, each (N,)."""
    carry: np.ndarray
    duration: np.ndarray
    cost: np.ndarray
    penalty: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.carry + self.duration - self.cost - self.penalty


class Problem:
    def __init__(self, cfg: Config, cost_bp: float | None = None, horizon: int | None = None):
        p = cfg.portfolio
        self.cfg = cfg
        self.model = CurveModel(cfg)
        self.dt = cfg.time.dt
        self.horizon = cfg.time.horizon if horizon is None else horizon
        self.cost_bp = cfg.costs.train_bp if cost_bp is None else cost_bp
        self.cost_rate = self.cost_bp / 1e4
        self.maturities = np.array(cfg.buckets.maturities, dtype=float)
        self.durations = zero_modified_duration(self.maturities)
        self.band = tuple(p.duration_band)
        self.band_mode = p.band_mode
        self.band_penalty = p.band_penalty
        self.max_turnover = p.max_turnover
        self.step = p.weight_step

        grid = simplex_grid(p.weight_step)
        self.actions = grid[(grid <= p.max_weight + 1e-9).all(axis=1)]
        self.n_actions = len(self.actions)
        self.action_duration = self.actions @ self.durations
        self.band_ok = (self.action_duration >= self.band[0] - 1e-9) & \
                       (self.action_duration <= self.band[1] + 1e-9)
        self.band_excess = np.maximum(self.band[0] - self.action_duration, 0) + \
            np.maximum(self.action_duration - self.band[1], 0)
        self.action_penalty = self.band_penalty * self.band_excess  # per step
        self.turnover = 0.5 * np.abs(self.actions[:, None, :] - self.actions[None, :, :]).sum(-1)
        self.cost_matrix = self.cost_rate * 2.0 * self.turnover  # [w_prev, a]: rate * sum|dw|
        turnover_ok = self.turnover <= self.max_turnover + 1e-9
        self.feasible = turnover_ok & (self.band_ok[None, :] if self.band_mode == "hard" else True)
        self.states = np.nonzero(self.band_ok)[0]  # previous allocations a compliant policy can hold
        self.static_idx = self._static_index(p.static)
        self.long_index = int(np.argmax(self.maturities))

    # ------------------------------------------------------------- helpers
    def index_of(self, w) -> int:
        d = np.abs(self.actions - np.asarray(w, dtype=float)).sum(1)
        i = int(np.argmin(d))
        if d[i] > 1e-6:
            raise ValueError(f"{w} is not on the action grid")
        return i

    def _static_index(self, spec) -> int:
        if spec == "best_stationary":
            # do-nothing allocation with the highest stationary expected return: under the
            # stationary distribution E[dy] = 0, so this is the highest-carry compliant portfolio.
            carry = self.actions @ (self.model.yields(self.model.mu, self.maturities) * self.dt)
            carry = np.where(self.band_ok, carry, -np.inf)
            return int(np.argmax(carry))
        idx = self.index_of(spec)
        if not self.band_ok[idx]:
            raise ValueError("static allocation violates the duration band")
        return idx

    def bucket_yields(self, factors: np.ndarray) -> np.ndarray:
        return self.model.yields(factors, self.maturities)

    def expected_bucket_returns(self, f: np.ndarray) -> np.ndarray:
        """E[r_i | level, slope] under the training process, exact. (N, 3)."""
        f = np.asarray(f, dtype=float)[..., :2]
        y0 = self.bucket_yields(f)
        dy = self.bucket_yields(self.model.cond_mean(f)) - y0
        return y0 * self.dt - dy * self.durations

    def expected_action_returns(self, f: np.ndarray) -> np.ndarray:
        """(N, nA) expected one-period return of every action, before costs."""
        return self.expected_bucket_returns(f) @ self.actions.T

    def decompose(self, w_idx, a_idx, y0, y1) -> Decomposition:
        """Reward decomposition for moving w -> a, then holding a while bucket yields go y0 -> y1."""
        a = self.actions[a_idx]
        return Decomposition(
            carry=(a * y0).sum(-1) * self.dt,
            duration=-(a * (y1 - y0) * self.durations).sum(-1),
            cost=self.cost_matrix[w_idx, a_idx],
            penalty=self.action_penalty[a_idx],
        )

    def violations(self, w_idx, a_idx) -> dict:
        return {
            "turnover": self.turnover[w_idx, a_idx] > self.max_turnover + 1e-9,
            "band": ~self.band_ok[a_idx],
            "position": (self.actions[a_idx] > self.cfg.portfolio.max_weight + 1e-9).any(-1),
        }
