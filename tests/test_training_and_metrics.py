"""Integration tests for training, evaluation and experiment outputs."""

import csv
import json

from static_maze_solver.agent import QLearningAgent
from static_maze_solver.environment import MazeEnv
from static_maze_solver.metrics import moving_average, plot_convergence, summarize, write_metrics
from static_maze_solver.training import evaluate_agent, has_converged, train_agent


def test_training_learns_and_evaluation_does_not_mutate_epsilon() -> None:
    env = MazeEnv(grid_size=(2, 2), goal=(0, 1), max_steps=8)
    agent = QLearningAgent(
        alpha=0.5,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.0,
        epsilon_decay=0.9,
        seed=3,
    )
    trained = train_agent(
        env,
        agent,
        episodes=100,
        seed=3,
        early_stop_window=10,
        early_stop_success_rate=0.8,
    )
    epsilon = agent.epsilon
    evaluated = evaluate_agent(env, agent, episodes=5)
    assert len(trained.records) == 100
    assert all(record.success for record in evaluated.records)
    assert all(record.steps == 1 for record in evaluated.records)
    assert agent.epsilon == epsilon


def test_metrics_are_written_and_plotted(tmp_path) -> None:
    env = MazeEnv(grid_size=(2, 2), goal=(0, 1), max_steps=2)
    agent = QLearningAgent(epsilon=0.0)
    agent.q_values([0, 0])[3] = 1.0
    result = evaluate_agent(env, agent, episodes=2)

    csv_path, json_path = write_metrics(
        result.records,
        tmp_path,
        prefix="evaluation",
        metadata={"seed": 42},
    )
    with csv_path.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["success_rate"] == 1.0
    assert payload["metadata"]["seed"] == 42
    assert summarize(result.records)["path_efficiency_ratio"] == 1.0
    assert plot_convergence(result.records, tmp_path / "plot.png").exists()


def test_moving_average_and_convergence() -> None:
    assert moving_average([1.0, 3.0, 5.0], window=2) == [1.0, 2.0, 4.0]
    env = MazeEnv(grid_size=(2, 2), goal=(0, 1))
    agent = QLearningAgent(epsilon=0.0)
    agent.q_values([0, 0])[3] = 1.0
    records = evaluate_agent(env, agent, episodes=3).records
    assert has_converged(records, window=3, success_rate=1.0)
