"""Duration matching to a target: the classical fixed-income answer.

Among allocations within `tolerance` years of the target duration, hold the one with the highest
current carry (bullet versus barbell is the only choice left once duration is pinned). If no
such allocation is reachable within the turnover limit, move to the reachable one closest to
the target.
"""

from __future__ import annotations

import numpy as np

from duralloc.policy import Policy

_MISS_PENALTY = 1.0  # per year of duration miss; carry is ~1e-3, so matching always dominates


class ImmunizePolicy(Policy):
    name = "immunize"

    def g(self, t, f):
        p = self.problem
        imm = p.cfg.portfolio.immunize
        miss = np.maximum(np.abs(p.action_duration - imm.target_duration) - imm.tolerance, 0.0)
        carry = (p.bucket_yields(np.atleast_2d(f)[:, :2]) * p.dt) @ p.actions.T
        return carry - _MISS_PENALTY * miss - p.action_penalty
