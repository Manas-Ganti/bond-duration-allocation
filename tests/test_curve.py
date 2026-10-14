import numpy as np
import pytest

from duralloc import checks
from duralloc.curve import CurveModel, ar1_moments, calibrate, h_loading


def test_gate_a_moments_match_targets(problem, rng):
    # Gate A at reduced size (the full-size run is `duralloc baseline`)
    ok, det = checks.gate_a(problem, rng, n_paths=800, n_steps=400)
    assert ok, det["error"]


def test_projection_recovers_factors_exactly(problem, rng):
    m = problem.model
    f = m.sample_stationary(1000, rng)
    f_hat = m.project(m.yields(f, m.obs_tenors))
    np.testing.assert_allclose(f_hat, f, atol=1e-12)


def test_projection_handles_missing_tenors(problem, rng):
    m = problem.model
    f = m.sample_stationary(50, rng)
    y = m.yields(f, m.obs_tenors)
    y[::3, -1] = np.nan  # e.g. the 30y gap 2002-2006
    np.testing.assert_allclose(m.project(y), f, atol=1e-12)


def test_loadings_shape_and_limits(problem):
    lam = problem.model.lam
    assert h_loading(1e-9, lam) == pytest.approx(1.0)
    assert h_loading(1e6, lam) < 1e-5
    f = np.array([0.05, 0.02])
    y = problem.model.yields(f, [1e-6, 1e6])
    assert y[0] == pytest.approx(0.03, abs=1e-6) and y[1] == pytest.approx(0.05, abs=1e-6)


def test_conditional_mean_is_exact_ou(problem):
    m = problem.model
    f = m.mu + np.array([0.02, -0.01])
    np.testing.assert_allclose(m.cond_mean(f) - m.mu, np.exp(-m.kappa * m.dt) * (f - m.mu))


def test_calibration_recovers_known_parameters(problem, rng):
    m = problem.model
    paths = m.simulate(1, 20000, rng)
    est = calibrate(m, m.yields(paths.factors[:, 0], m.obs_tenors), m.obs_tenors)
    np.testing.assert_allclose(est["kappa"], m.kappa, rtol=0.25)
    np.testing.assert_allclose(est["sigma"], m.sigma, rtol=0.05)
    assert abs(est["rho"] - m.rho) < 0.05


@pytest.mark.parametrize("variant", ["curvature", "regime", "jump"])
def test_stress_variants_differ_from_training(problem, variant):
    m = problem.model
    base = m.simulate(4000, 24, np.random.default_rng(1))
    alt = m.simulate(4000, 24, np.random.default_rng(1), variant)
    assert alt.variant == variant
    obs_base = m.observe(base)
    obs_alt = m.observe(alt)
    assert not np.allclose(obs_base, obs_alt)
    if variant == "curvature":
        assert np.abs(alt.factors[..., 2]).max() > 0
    if variant == "regime":
        assert 0.1 < alt.regime.mean() < 0.4
    if variant == "jump":
        d_alt = np.diff(alt.factors[..., 0], axis=0)
        d_base = np.diff(base.factors[..., 0], axis=0)
        assert ((d_alt - d_alt.mean()) ** 4).mean() / d_alt.var() ** 2 > \
            ((d_base - d_base.mean()) ** 4).mean() / d_base.var() ** 2 + 1


def test_unknown_variant_rejected(problem, rng):
    with pytest.raises(ValueError):
        problem.model.simulate(10, 2, rng, "nonsense")
