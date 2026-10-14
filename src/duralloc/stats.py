"""Interval estimates. A point estimate without an interval is not a result."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class Estimate:
    mean: float
    se: float
    ci_low: float
    ci_high: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)

    def scaled(self, k: float) -> "Estimate":
        return Estimate(self.mean * k, self.se * k, self.ci_low * k, self.ci_high * k, self.n)


def mc_estimate(values, ci_level: float = 0.95) -> Estimate:
    """Mean and normal CI of iid samples (paired differences are passed in as one array)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    z = stats.norm.ppf(0.5 + ci_level / 2)
    return Estimate(mean, se, mean - z * se, mean + z * se, n)


def cluster_bootstrap_mean(values, clusters, rng: np.random.Generator, n_boot: int,
                           ci_level: float = 0.95) -> Estimate:
    """Mean with a cluster bootstrap CI: whole clusters (calendar years) are resampled, because
    monthly yield changes within a year are correlated and an iid bootstrap understates
    uncertainty."""
    v = np.asarray(values, dtype=float)
    labels, inv = np.unique(np.asarray(clusters), return_inverse=True)
    k = len(labels)
    sums = np.bincount(inv, weights=v, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    draw = rng.integers(0, k, size=(n_boot, k))
    means = sums[draw].sum(1) / counts[draw].sum(1)
    a = (1 - ci_level) / 2
    return Estimate(float(v.mean()), float(means.std(ddof=1)), float(np.quantile(means, a)),
                    float(np.quantile(means, 1 - a)), k)
