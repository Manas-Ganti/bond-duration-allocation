"""Backward-induction dynamic programming on the discretised state. The computable optimum.

State: (previous allocation w, level, slope, t). Level and slope live on a rectangular grid;
the one-step factor transition is a row-stochastic operator built from tensor Gauss-Hermite
quadrature of the exact OU shock, with bilinear redistribution of each node onto the grid. The
expected one-period return is the exact closed form. Off the grid (e.g. when evaluating on
simulated paths) the continuation value is bilinearly interpolated, so the DP policy is defined
on the whole continuous plane and is evaluated on exactly the same paths as every other policy.

Trailing volatility and last turnover are in the agent's observation but not in the DP state:
under the training process neither changes any future reward, so conditioning on them cannot
improve on this optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from duralloc.curve import CurveModel
from duralloc.grid import FactorGrid
from duralloc.policy import _HOLD_BONUS, Policy
from duralloc.problem import Problem


def transition_operator(model: CurveModel, grid: FactorGrid, n_nodes: int) -> np.ndarray:
    z, wq = np.polynomial.hermite_e.hermegauss(n_nodes)
    wq = wq / wq.sum()
    Z = np.stack(np.meshgrid(z, z, indexing="ij"), -1).reshape(-1, 2)
    W = np.outer(wq, wq).ravel()
    means = model.cond_mean(grid.points)
    nxt = means[:, None, :] + (Z @ model.step_chol.T)[None]
    idx, w = grid.weights(nxt.reshape(-1, 2))
    rows = np.repeat(np.arange(grid.n), len(W) * 4)
    vals = (np.tile(W, grid.n)[:, None] * w).ravel()
    P = np.zeros((grid.n, grid.n))
    np.add.at(P, (rows, idx.ravel()), vals)
    return P


@dataclass
class DPSolution:
    reward: np.ndarray  # (nF, nA) expected one-period reward before costs, on the grid
    cont: np.ndarray    # (T, nA, nF) E[V_{t+1}(a, f') | f]
    value: np.ndarray   # (T+1, nA, nF) V_t(w, f); value[T] = 0
    policy: np.ndarray  # (T, nA, nF) argmax action index

    def q(self, t: int, cost_matrix: np.ndarray) -> np.ndarray:
        """Q_t(w, f, a) on the grid: (nW, nF, nA), unmasked."""
        g = self.reward + self.cont[t].T
        return g[None] - cost_matrix[:, None, :]


def solve_dp(reward: np.ndarray, P: np.ndarray, cost_matrix: np.ndarray, feasible: np.ndarray,
             horizon: int) -> DPSolution:
    """Generic finite-horizon backward induction.

    reward[f, a]: expected immediate reward (before costs) of holding allocation a from factor
    state f; P[f, f']: factor transition; cost_matrix[w, a]; feasible[w, a].
    """
    nF, nA = reward.shape
    value = np.zeros((horizon + 1, nA, nF))
    cont = np.zeros((horizon, nA, nF))
    policy = np.zeros((horizon, nA, nF), dtype=np.int32)
    hold = np.eye(nA, dtype=bool)[:, None, :] * _HOLD_BONUS
    for t in range(horizon - 1, -1, -1):
        cont[t] = value[t + 1] @ P.T
        g = (reward + cont[t].T)[None]                       # (1, nF, nA)
        q = g - cost_matrix[:, None, :] + hold                # (nW, nF, nA)
        q = np.where(feasible[:, None, :], q, -np.inf)
        policy[t] = q.argmax(-1)
        value[t] = np.take_along_axis(q, policy[t][..., None], -1)[..., 0]
    return DPSolution(reward, cont, value, policy)


class DPPolicy(Policy):
    name = "dp"

    def __init__(self, problem: Problem, level_points: int | None = None,
                 slope_points: int | None = None, quad_nodes: int | None = None):
        super().__init__(problem)
        d = problem.cfg.dp
        m = problem.model
        self.grid = FactorGrid(m.mu, m.stat_sd, d.n_std, level_points or d.level_points,
                               slope_points or d.slope_points)
        self.P = transition_operator(m, self.grid, quad_nodes or d.quad_nodes)
        reward = problem.expected_action_returns(self.grid.points) - problem.action_penalty
        self.solution = solve_dp(reward, self.P, problem.cost_matrix, problem.feasible,
                                 problem.horizon)

    def g(self, t, f):
        f = np.atleast_2d(f)[:, :2]
        p = self.problem
        return p.expected_action_returns(f) - p.action_penalty + \
            self.grid.interpolate(self.solution.cont[t], f)

    def predicted_value(self, w_idx, f) -> np.ndarray:
        """V_0(w, f) by interpolation of the DP value function."""
        v = self.grid.interpolate(self.solution.value[0], np.atleast_2d(f))  # (N, nW)
        return v[np.arange(len(v)), np.asarray(w_idx)]
