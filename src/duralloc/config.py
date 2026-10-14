"""Typed configuration. Everything numeric comes from configs/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
DATA_CACHE = REPO_ROOT / "data_cache"


class TimeCfg(BaseModel):
    dt: float
    horizon: int


class OUCfg(BaseModel):
    mu: float
    kappa: float
    sigma: float


class CurveCfg(BaseModel):
    ns_lambda: float
    level: OUCfg
    slope: OUCfg
    rho: float
    obs_tenors: list[float]
    vol_ewma: float


class RegimeCfg(BaseModel):
    low_scale: float
    high_scale: float
    p_low_to_high: float
    p_high_to_low: float


class JumpCfg(BaseModel):
    intensity: float
    mean: float
    std: float


class StressCfg(BaseModel):
    curvature: OUCfg
    regime: RegimeCfg
    jump: JumpCfg


class BucketsCfg(BaseModel):
    names: list[str]
    maturities: list[float]


class ImmunizeCfg(BaseModel):
    target_duration: float
    tolerance: float


class PortfolioCfg(BaseModel):
    weight_step: float
    max_weight: float
    max_turnover: float
    duration_band: tuple[float, float]
    band_mode: Literal["hard", "soft"] = "hard"
    band_penalty: float
    static: Literal["best_stationary"] | list[float] = "best_stationary"
    immunize: ImmunizeCfg


class CostsCfg(BaseModel):
    train_bp: float
    sweep_bp: list[float]


class DPGridCfg(BaseModel):
    level_points: int
    slope_points: int
    quad_nodes: int


class DPCfg(DPGridCfg):
    n_std: float
    check: DPGridCfg


class FQICfg(BaseModel):
    n_samples: int
    rbf_level: int
    rbf_slope: int
    rbf_width: float
    ridge: float
    box_n_std: float


class EvaluationCfg(BaseModel):
    n_paths: int
    ci_level: float
    sweep_n_paths: int


class PolicyMapCfg(BaseModel):
    t: list[int]
    inner_n_std: float


class HistoricalCfg(BaseModel):
    series: list[str]
    start: str
    n_boot: int


class GateACfg(BaseModel):
    n_paths: int
    n_steps: int
    mean_abs_tol: float
    sigma_rel_tol: float
    kappa_rel_tol: float
    rho_abs_tol: float


class GateBCfg(BaseModel):
    base_yield: float
    small_move: float
    tol: float
    scan_max: float


class GateCCfg(BaseModel):
    min_agreement: float
    max_mean_regret_bp: float
    grid_points: int
    n_std: float


class GateDCfg(BaseModel):
    max_shortfall_frac: float
    min_map_agreement: float
    map_step_tol: float


class GatesCfg(BaseModel):
    a: GateACfg
    b: GateBCfg
    c: GateCCfg
    d: GateDCfg


class Config(BaseModel):
    seed: int
    time: TimeCfg
    curve: CurveCfg
    stress: StressCfg
    buckets: BucketsCfg
    portfolio: PortfolioCfg
    costs: CostsCfg
    dp: DPCfg
    fqi: FQICfg
    evaluation: EvaluationCfg
    policymap: PolicyMapCfg
    historical: HistoricalCfg
    gates: GatesCfg


def load_config(path: str | Path | None = None) -> Config:
    with open(path or DEFAULT_CONFIG) as fh:
        return Config(**yaml.safe_load(fh))
