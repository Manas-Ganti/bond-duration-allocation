import json

import numpy as np
import pytest

from duralloc import checks, config, gates
from duralloc.agent.fitted_q import train_fqi
from duralloc.baselines import DPPolicy
from duralloc.problem import Problem


def test_gate_c_trivial_case_reduced(cfg):
    pc = Problem(cfg, cost_bp=0.0, horizon=1)
    ok, det = checks.gate_c(pc, train_fqi(pc, seed=2, n_samples=600000))
    assert ok, det


def test_gate_d_agreement_reduced(cfg):
    p = Problem(cfg, horizon=4)
    dp = DPPolicy(p)
    ok, det = checks.gate_d(p, dp, train_fqi(p, seed=2),
                            np.random.default_rng(5), n_paths=150000, map_t=[0])
    assert ok, det


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_gate_e_refuses_historical_data_before_a_to_d(cfg, tmp_results, monkeypatch):
    from duralloc.data import fred

    def no_network(*a, **k):
        raise AssertionError("tried to download historical data before gates passed")
    monkeypatch.setattr(fred, "_download", no_network)
    with pytest.raises(gates.GateNotPassed):
        fred.load_monthly_yields(cfg)
    for g in "ABC":
        gates.record_gate(g, True, {}, 0)
    gates.record_gate("D", False, {}, 0)
    with pytest.raises(gates.GateNotPassed):
        fred.load_monthly_yields(cfg)
    gates.record_gate("D", True, {}, 0)
    with pytest.raises(AssertionError, match="tried to download"):
        fred.load_monthly_yields(cfg)  # now allowed through to the loader
    assert json.loads((tmp_results / "gates.json").read_text())["D"]["passed"]
