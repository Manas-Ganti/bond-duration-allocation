"""Two-factor yield-curve generator.

Level L and slope S are Ornstein-Uhlenbeck processes, sampled with the exact discretisation,
and mapped to yields in the Nelson-Siegel spirit:

    y(tau) = L - S * h(tau) [+ C * (h(tau) - exp(-lambda tau))],   h(tau) = (1 - e^{-lambda tau}) / (lambda tau)

so L is the long-end asymptote and S is "long minus instantaneous short" (S > 0: upward sloping).
The curvature term C is a stress variant only. Policies never see the factors directly: they see
the least-squares projection of yields at the observation tenors onto the two training loadings,
which is exact under the training process and deliberately misspecified under the stress
variants and on real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from duralloc.config import Config

VARIANTS = ("train", "curvature", "regime", "jump")


def h_loading(tau, lam: float) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    x = lam * tau
    return np.where(x > 1e-12, (1.0 - np.exp(-x)) / np.where(x > 1e-12, x, 1.0), 1.0)


def curvature_loading(tau, lam: float) -> np.ndarray:
    return h_loading(tau, lam) - np.exp(-lam * np.asarray(tau, dtype=float))


@dataclass
class CurvePaths:
    """factors[t, path] = (level, slope, curvature); regime[t, path] in {0: low, 1: high}."""
    factors: np.ndarray
    regime: np.ndarray
    variant: str


class CurveModel:
    def __init__(self, cfg: Config):
        c = cfg.curve
        self.cfg = cfg
        self.dt = cfg.time.dt
        self.lam = c.ns_lambda
        self.mu = np.array([c.level.mu, c.slope.mu])
        self.kappa = np.array([c.level.kappa, c.slope.kappa])
        self.sigma = np.array([c.level.sigma, c.slope.sigma])
        self.rho = c.rho
        self.obs_tenors = np.array(c.obs_tenors, dtype=float)
        self.phi = np.exp(-self.kappa * self.dt)
        corr = np.array([[1.0, self.rho], [self.rho, 1.0]])
        ksum = self.kappa[:, None] + self.kappa[None, :]
        ss = np.outer(self.sigma, self.sigma) * corr
        self.step_cov = ss * (1.0 - np.exp(-ksum * self.dt)) / ksum
        self.stat_cov = ss / ksum
        self.step_chol = np.linalg.cholesky(self.step_cov)
        self.stat_chol = np.linalg.cholesky(self.stat_cov)
        self.stat_sd = np.sqrt(np.diag(self.stat_cov))
        self._proj = self._projection_matrix(self.obs_tenors)

    # ---------------------------------------------------------------- yields
    def loadings(self, tau) -> np.ndarray:
        """(len(tau), 2) loadings on (level, slope)."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        return np.stack([np.ones_like(tau), -h_loading(tau, self.lam)], axis=-1)

    def yields(self, factors: np.ndarray, tau) -> np.ndarray:
        """Yields at maturities `tau` for factors (..., 2 or 3) -> (..., len(tau))."""
        f = np.asarray(factors, dtype=float)
        y = f[..., :2] @ self.loadings(tau).T
        if f.shape[-1] > 2:
            y = y + f[..., 2:3] * curvature_loading(np.atleast_1d(tau), self.lam)
        return y

    def _projection_matrix(self, tenors) -> np.ndarray:
        X = self.loadings(tenors)
        return np.linalg.solve(X.T @ X, X.T)  # (2, m)

    def project(self, yields: np.ndarray, tenors=None) -> np.ndarray:
        """Least-squares (level, slope) from yields at `tenors` (default: observation tenors).

        NaNs are allowed (historical gaps): each row is fitted on its observed tenors only.
        """
        y = np.asarray(yields, dtype=float)
        if tenors is None:
            tenors = self.obs_tenors
        if not np.isnan(y).any():
            P = self._proj if np.array_equal(np.asarray(tenors, float), self.obs_tenors) \
                else self._projection_matrix(tenors)
            return y @ P.T
        out = np.full(y.shape[:-1] + (2,), np.nan)
        flat_y, flat_o = y.reshape(-1, y.shape[-1]), out.reshape(-1, 2)
        X = self.loadings(tenors)
        for i, row in enumerate(flat_y):
            m = ~np.isnan(row)
            if m.sum() >= 2:
                flat_o[i] = np.linalg.lstsq(X[m], row[m], rcond=None)[0]
        return out

    # -------------------------------------------------------------- dynamics
    def cond_mean(self, f: np.ndarray) -> np.ndarray:
        """E[f_{t+1} | f_t] for (level, slope), exact."""
        return self.mu + self.phi * (np.asarray(f)[..., :2] - self.mu)

    def sample_stationary(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.mu + rng.standard_normal((n, 2)) @ self.stat_chol.T

    def simulate(self, n_paths: int, n_steps: int, rng: np.random.Generator,
                 variant: str = "train", f0: np.ndarray | None = None) -> CurvePaths:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        s = self.cfg.stress
        F = np.zeros((n_steps + 1, n_paths, 3))
        F[0, :, :2] = self.sample_stationary(n_paths, rng) if f0 is None else f0[..., :2]
        regime = np.zeros((n_steps + 1, n_paths), dtype=np.int8)
        if variant == "curvature":
            cv = s.curvature
            phi_c = np.exp(-cv.kappa * self.dt)
            sd_c_step = cv.sigma * np.sqrt((1 - phi_c ** 2) / (2 * cv.kappa))
            F[0, :, 2] = cv.mu + cv.sigma / np.sqrt(2 * cv.kappa) * rng.standard_normal(n_paths)
        if variant == "regime":
            rg = s.regime
            p_high = rg.p_low_to_high / (rg.p_low_to_high + rg.p_high_to_low)
            regime[0] = rng.random(n_paths) < p_high
            scales = np.array([rg.low_scale, rg.high_scale])
        for t in range(n_steps):
            z = rng.standard_normal((n_paths, 2)) @ self.step_chol.T
            if variant == "regime":
                z = z * scales[regime[t]][:, None]
                u = rng.random(n_paths)
                stay_high = u >= rg.p_high_to_low
                go_high = u < rg.p_low_to_high
                regime[t + 1] = np.where(regime[t] == 1, stay_high, go_high)
            F[t + 1, :, :2] = self.cond_mean(F[t]) + z
            if variant == "curvature":
                F[t + 1, :, 2] = cv.mu + phi_c * (F[t, :, 2] - cv.mu) \
                    + sd_c_step * rng.standard_normal(n_paths)
            if variant == "jump":
                j = s.jump
                hit = rng.random(n_paths) < j.intensity * self.dt
                F[t + 1, :, 0] += hit * (j.mean + j.std * rng.standard_normal(n_paths))
        return CurvePaths(F, regime, variant)

    def observe(self, paths: CurvePaths) -> np.ndarray:
        """What a policy sees: (level, slope) projected from yields at the observation tenors."""
        return self.project(self.yields(paths.factors, self.obs_tenors))


# ------------------------------------------------------------------ estimation
def ar1_moments(factors: np.ndarray, dt: float) -> dict:
    """Pooled AR(1) on factor paths (T+1, N, k) -> continuous-time OU parameters.

    Used for Gate A on simulated paths and, after Gate E, for calibration to history
    (where `factors` is a single (T+1, 1, k) series).
    """
    x0 = factors[:-1].reshape(-1, factors.shape[-1])
    x1 = factors[1:].reshape(-1, factors.shape[-1])
    ok = ~(np.isnan(x0).any(1) | np.isnan(x1).any(1))
    x0, x1 = x0[ok], x1[ok]
    k = x0.shape[1]
    phi, c, resid = np.zeros(k), np.zeros(k), np.zeros_like(x0)
    for i in range(k):
        A = np.column_stack([np.ones(len(x0)), x0[:, i]])
        coef = np.linalg.lstsq(A, x1[:, i], rcond=None)[0]
        c[i], phi[i] = coef
        resid[:, i] = x1[:, i] - A @ coef
    kappa = -np.log(phi) / dt
    sd = resid.std(axis=0, ddof=2)
    sigma = sd * np.sqrt(2 * kappa / (1 - phi ** 2))
    out = {
        "mean": np.nanmean(factors.reshape(-1, k), axis=0).tolist(),
        "mu": (c / (1 - phi)).tolist(),
        "kappa": kappa.tolist(),
        "sigma": sigma.tolist(),
        "phi": phi.tolist(),
        "n_transitions": int(len(x0)),
    }
    if k >= 2:
        out["rho"] = float(np.corrcoef(resid[:, 0], resid[:, 1])[0, 1])
    return out


def calibrate(model: CurveModel, yields: np.ndarray, tenors) -> dict:
    """Fit the two-factor OU parameters to a monthly yield panel (T, m) at `tenors`."""
    f = model.project(yields, tenors)
    return ar1_moments(f[:, None, :], model.dt)
