"""Single-period optimiser: maximise expected return net of cost this month, ignore the future.

It has the true model's conditional mean, exactly as DP does, so the gap DP - myopic isolates
the value of being dynamic rather than the value of being optimised.
"""

from __future__ import annotations

from duralloc.policy import Policy


class MyopicPolicy(Policy):
    name = "myopic"

    def g(self, t, f):
        p = self.problem
        return p.expected_action_returns(f) - p.action_penalty
