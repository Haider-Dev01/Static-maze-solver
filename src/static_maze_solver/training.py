"""Training, evaluation and benchmark orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from static_maze_solver.agent import QLearningAgent
from static_maze_solver.baseline import State, astar_path
from static_maze_solver.environment import MazeEnv
from static_maze_solver.metrics import EpisodeMetrics

ProgressCallback = Callable[[EpisodeMetrics], None]


@dataclass(frozen=True)
class RunResult:
    """Records and final path produced by a multi-episode run."""

    records: tuple[EpisodeMetrics, ...]
    final_path: tuple[State, ...]
    stopped_early: bool = False


def _optimal_steps(env: MazeEnv) -> int | None:
    path = astar_path(env.grid_size, env.start, env.goal, env.obstacles)
    return len(path) - 1 if path else None


def _run_episode(
    env: MazeEnv,
    agent: QLearningAgent,
    episode: int,
    *,
    seed: int,
    train: bool,
    render: bool,
    optimal_steps: int | None,
) -> tuple[EpisodeMetrics, tuple[State, ...]]:
    observation, _ = env.reset(seed=seed)
    path = [env.start]
    total_reward = 0.0
    started = time.perf_counter()
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.choose_action(observation, explore=train)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        if train:
            agent.update(
                observation,
                action,
                reward,
                next_observation,
                terminated=terminated or truncated,
            )
        observation = next_observation
        total_reward += reward
        path.append(env.agent_position)
        if render:
            env.set_render_data(
                status={
                    "mode": "train" if train else "evaluation",
                    "episode": episode,
                    "step": env.step_count,
                    "reward": f"{total_reward:.1f}",
                    "epsilon": f"{agent.epsilon:.3f}",
                },
                learned_path=path,
                optimal_path=astar_path(env.grid_size, env.start, env.goal, env.obstacles) or (),
            )
            env.render()
            if env.close_requested:
                truncated = True

    record = EpisodeMetrics(
        episode=episode,
        reward=total_reward,
        steps=env.step_count,
        success=terminated,
        epsilon=agent.epsilon,
        duration_seconds=time.perf_counter() - started,
        optimal_steps=optimal_steps,
    )
    return record, tuple(path)


def train_agent(
    env: MazeEnv,
    agent: QLearningAgent,
    *,
    episodes: int = 500,
    seed: int = 42,
    render: bool = False,
    early_stop_window: int = 50,
    early_stop_success_rate: float = 0.98,
    callback: ProgressCallback | None = None,
) -> RunResult:
    """Train until the episode budget or convergence criterion is reached."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    records: list[EpisodeMetrics] = []
    final_path: tuple[State, ...] = ()
    optimal_steps = _optimal_steps(env)
    stopped_early = False

    for episode in range(1, episodes + 1):
        record, final_path = _run_episode(
            env,
            agent,
            episode,
            seed=seed + episode - 1,
            train=True,
            render=render,
            optimal_steps=optimal_steps,
        )
        records.append(record)
        agent.decay_exploration()
        if callback:
            callback(record)
        if env.close_requested:
            break
        recent = records[-early_stop_window:]
        if (
            episode >= 100
            and len(recent) == early_stop_window
            and agent.epsilon <= agent.epsilon_min
            and sum(item.success for item in recent) / early_stop_window >= early_stop_success_rate
            and optimal_steps is not None
            and sum(item.steps for item in recent) / early_stop_window <= optimal_steps * 1.5
        ):
            stopped_early = True
            break
    return RunResult(tuple(records), final_path, stopped_early)


def evaluate_agent(
    env: MazeEnv,
    agent: QLearningAgent,
    *,
    episodes: int = 20,
    seed: int = 10_000,
    render: bool = False,
    callback: ProgressCallback | None = None,
) -> RunResult:
    """Evaluate greedily without modifying Q-values or epsilon."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    records: list[EpisodeMetrics] = []
    final_path: tuple[State, ...] = ()
    optimal_steps = _optimal_steps(env)
    for episode in range(1, episodes + 1):
        record, final_path = _run_episode(
            env,
            agent,
            episode,
            seed=seed + episode - 1,
            train=False,
            render=render,
            optimal_steps=optimal_steps,
        )
        records.append(record)
        if callback:
            callback(record)
        if env.close_requested:
            break
    return RunResult(tuple(records), final_path)


def has_converged(
    records: Sequence[EpisodeMetrics],
    *,
    window: int = 50,
    success_rate: float = 0.98,
) -> bool:
    """Expose the convergence check for reports and tests."""
    recent = records[-window:]
    return (
        len(recent) == window and sum(record.success for record in recent) / window >= success_rate
    )
