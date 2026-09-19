"""Tests for Q-learning updates and model persistence."""

import json

import numpy as np
import pytest

from static_maze_solver.agent import QLearningAgent


def test_terminal_q_learning_update() -> None:
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0)
    value = agent.update([0, 0], 3, 100.0, [0, 1], terminated=True)
    assert value == 50.0
    assert agent.q_values([0, 0])[3] == 50.0


def test_non_terminal_update_bootstraps() -> None:
    agent = QLearningAgent(alpha=1.0, gamma=0.5, epsilon=0.0)
    agent.q_values([0, 1])[2] = 10.0
    value = agent.update([0, 0], 3, -1.0, [0, 1], terminated=False)
    assert value == 4.0


def test_epsilon_decay_has_lower_bound() -> None:
    agent = QLearningAgent(epsilon=0.2, epsilon_min=0.1, epsilon_decay=0.5)
    assert agent.decay_exploration() == 0.1
    assert agent.decay_exploration() == 0.1


def test_seed_makes_exploration_reproducible() -> None:
    first = QLearningAgent(epsilon=1.0, seed=8)
    second = QLearningAgent(epsilon=1.0, seed=8)
    assert [first.choose_action([0, 0]) for _ in range(10)] == [
        second.choose_action([0, 0]) for _ in range(10)
    ]


def test_save_and_load_round_trip(tmp_path) -> None:
    agent = QLearningAgent(alpha=0.2, seed=5)
    agent.q_values([1, 2])[:] = [1.0, 2.0, 3.0, 4.0]
    destination = agent.save(tmp_path / "models" / "agent.json")
    restored = QLearningAgent.load(destination)
    assert restored.alpha == 0.2
    assert restored.seed == 5
    assert np.array_equal(restored.q_values([1, 2]), [1.0, 2.0, 3.0, 4.0])


def test_bad_model_version_is_rejected(tmp_path) -> None:
    model = tmp_path / "bad.json"
    model.write_text(json.dumps({"model_version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        QLearningAgent.load(model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_count": 0},
        {"alpha": 0},
        {"gamma": 2},
        {"epsilon": 2},
        {"epsilon_decay": 0},
    ],
)
def test_invalid_hyperparameters_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        QLearningAgent(**kwargs)
