import numpy as np
import pytest

from duralloc.config import load_config
from duralloc.problem import Problem


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def problem(cfg):
    return Problem(cfg)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
