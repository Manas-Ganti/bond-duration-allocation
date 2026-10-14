"""One policy interface for every allocator, learned or classical.

Every policy here scores actions with a function G_t(f, a) of the observed curve state and then
chooses

    a* = argmax_{a feasible from w}  G_t(f, a) - cost(w, a)

This is exact for DP and fitted Q (Q_t(w, f, a) = -cost(w, a) + G_t(f, a) because the next
state depends on the action, not on where it came from), and it expresses the baselines
naturally. Ties go to holding the current allocation, then to the lowest index.
"""

from __future__ import annotations

import numpy as np

from duralloc.problem import Problem

_HOLD_BONUS = 1e-12


class Policy:
    name = "policy"

    def __init__(self, problem: Problem):
        self.problem = problem

    def g(self, t: int, f: np.ndarray) -> np.ndarray:  # (N, nA)
        raise NotImplementedError

    def q(self, t: int, w_idx: np.ndarray, f: np.ndarray) -> np.ndarray:
        p = self.problem
        w_idx = np.asarray(w_idx)
        q = self.g(t, f) - p.cost_matrix[w_idx]
        q = np.where(p.feasible[w_idx], q, -np.inf)
        q[np.arange(len(w_idx)), w_idx] += _HOLD_BONUS
        return q

    def act(self, t: int, w_idx: np.ndarray, f: np.ndarray) -> np.ndarray:
        return np.argmax(self.q(t, w_idx, f), axis=1)


class FunctionPolicy(Policy):
    """Wrap a plain function (t, w_idx, f) -> a_idx; used in tests and for ad-hoc policies."""

    def __init__(self, problem: Problem, fn, name: str = "function"):
        super().__init__(problem)
        self.fn, self.name = fn, name

    def act(self, t, w_idx, f):
        return np.asarray(self.fn(t, np.asarray(w_idx), np.asarray(f)), dtype=int)
