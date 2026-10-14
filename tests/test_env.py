import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from duralloc.env import DurationAllocEnv


def test_gymnasium_api(problem):
    check_env(DurationAllocEnv(problem), skip_render_check=True)


def test_step_reward_matches_decomposition(problem):
    env = DurationAllocEnv(problem)
    obs, _ = env.reset(seed=4)
    total = 0.0
    done = False
    while not done:
        a = int(np.nonzero(env.action_masks())[0][-1])
        obs, r, done, _, info = env.step(a)
        assert r == pytest.approx(info["carry"] + info["duration"] - info["cost"] - info["penalty"])
        assert not any(info["violations"].values())
        total += r
    assert env.t == problem.horizon
    assert obs.shape == (9,)


def test_bucket_returns_are_carry_minus_duration(problem):
    env = DurationAllocEnv(problem)
    env.reset(seed=1, options={"factors": problem.model.mu})
    _, r, _, _, info = env.step(problem.static_idx)
    w = problem.actions[problem.static_idx]
    assert r == pytest.approx(float(w @ info["bucket_returns"]))
