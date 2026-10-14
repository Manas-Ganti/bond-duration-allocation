"""Fixed allocation. The do-nothing baseline, and harder to beat after costs than people expect.

By default the allocation is the compliant portfolio with the highest stationary expected
return (see Problem._static_index), i.e. the strongest do-nothing policy, which makes every
"advantage over static" in this project a conservative number.
"""

from __future__ import annotations

import numpy as np

from duralloc.policy import Policy


class StaticPolicy(Policy):
    name = "static"

    def g(self, t, f):
        p = self.problem
        d = np.abs(p.actions - p.actions[p.static_idx]).sum(1)  # move toward static if displaced
        return np.broadcast_to(-d, (len(np.atleast_2d(f)), p.n_actions)).copy()
