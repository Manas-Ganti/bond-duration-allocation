"""Policy extraction: the allocation any policy chooses as a function of curve state.

This is the comparison surface between DP and the learned policy, and the source of the
headline figure (long-duration weight over the (level, slope) plane).
"""

from __future__ import annotations

import numpy as np

from duralloc.policy import Policy
from duralloc.problem import Problem


def extract_actions(policy: Policy, points: np.ndarray, w_idx: int | np.ndarray, t: int) -> np.ndarray:
    """Action index chosen at each (level, slope) point from previous allocation(s) w_idx."""
    points = np.atleast_2d(points)
    w = np.broadcast_to(np.asarray(w_idx), (len(points),)).astype(int)
    return np.asarray(policy.act(t, w, points), dtype=int)


def policy_map(policy: Policy, problem: Problem, levels: np.ndarray, slopes: np.ndarray,
               w_idx: int, t: int) -> dict:
    L, S = np.meshgrid(levels, slopes, indexing="ij")
    a = extract_actions(policy, np.column_stack([L.ravel(), S.ravel()]), w_idx, t)
    shape = L.shape
    return {
        "levels": np.asarray(levels), "slopes": np.asarray(slopes), "t": t,
        "w_prev": problem.actions[w_idx],
        "action": a.reshape(shape),
        "long_weight": problem.actions[a, problem.long_index].reshape(shape),
        "duration": problem.action_duration[a].reshape(shape),
        "hold": (a == w_idx).reshape(shape),
    }


def state_table(policy: Policy, problem: Problem, points: np.ndarray, t: int) -> np.ndarray:
    """Actions over every in-band previous allocation x points: (n_states, n_points)."""
    return np.stack([extract_actions(policy, points, w, t) for w in problem.states])


def compare_tables(problem: Problem, a: np.ndarray, b: np.ndarray, step_tol: float) -> dict:
    la, lb = problem.actions[a, problem.long_index], problem.actions[b, problem.long_index]
    hold_a = a == problem.states[:, None]
    hold_b = b == problem.states[:, None]
    union = (hold_a | hold_b).sum()
    return {
        "exact_match": float((a == b).mean()),
        "long_within_step": float((np.abs(la - lb) <= step_tol + 1e-9).mean()),
        "mean_abs_long_diff": float(np.abs(la - lb).mean()),
        "hold_fraction_a": float(hold_a.mean()),
        "hold_fraction_b": float(hold_b.mean()),
        "hold_jaccard": float((hold_a & hold_b).sum() / union) if union else 1.0,
    }
