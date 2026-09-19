"""Tests for Gymnasium environment behavior and validation."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from static_maze_solver.environment import MazeEnv


def test_environment_passes_gymnasium_checker() -> None:
    env = MazeEnv(grid_size=(3, 3), goal=(2, 2), max_steps=10)
    check_env(env, skip_render_check=True)
    env.close()


def test_moves_collisions_goal_and_reward() -> None:
    env = MazeEnv(grid_size=(2, 3), goal=(0, 2), obstacles=[(1, 1)], max_steps=10)
    observation, info = env.reset(seed=7)
    assert np.array_equal(observation, [0, 0])
    assert info["steps"] == 0

    observation, reward, terminated, truncated, _ = env.step(0)
    assert np.array_equal(observation, [0, 0])
    assert (reward, terminated, truncated) == (-1.0, False, False)

    observation, *_ = env.step(3)
    assert np.array_equal(observation, [0, 1])
    observation, reward, terminated, truncated, info = env.step(3)
    assert np.array_equal(observation, [0, 2])
    assert (reward, terminated, truncated) == (100.0, True, False)
    assert info["is_success"] is True


def test_episode_is_truncated_at_step_limit() -> None:
    env = MazeEnv(grid_size=(3, 3), max_steps=2)
    env.reset()
    env.step(0)
    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated
    assert truncated


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grid_size": (1, 2)}, "at least 2"),
        ({"grid_size": (3, 3), "goal": (3, 2)}, "outside"),
        ({"grid_size": (3, 3), "obstacles": [(0, 0)]}, "cannot be obstacles"),
        (
            {"grid_size": (3, 3), "obstacles": [(0, 1), (1, 0)]},
            "reachable",
        ),
        ({"grid_size": (3, 3), "render_mode": "text"}, "Unsupported"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MazeEnv(**kwargs)


def test_invalid_action_is_rejected() -> None:
    env = MazeEnv(grid_size=(3, 3))
    env.reset()
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(9)
