"""Visual assets for portfolio demonstrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from static_maze_solver.agent import QLearningAgent
from static_maze_solver.baseline import State, astar_path
from static_maze_solver.environment import MazeEnv


def record_demo(
    env: MazeEnv,
    agent: QLearningAgent,
    destination: str | Path,
    *,
    seed: int = 42,
    fps: int = 6,
) -> tuple[Path, tuple[State, ...], bool]:
    """Record one greedy RGB-array episode as a GIF."""
    if env.render_mode != "rgb_array":
        raise ValueError("record_demo requires an environment with render_mode='rgb_array'")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    observation, _ = env.reset(seed=seed)
    learned_path = [env.start]
    optimal_path = astar_path(env.grid_size, env.start, env.goal, env.obstacles) or ()
    frames: list[Any] = []
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        env.set_render_data(
            status={
                "mode": "demo",
                "step": env.step_count,
                "reward": f"{total_reward:.1f}",
                "epsilon": "0.000",
            },
            learned_path=learned_path,
            optimal_path=optimal_path,
        )
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action = agent.choose_action(observation, explore=False)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        learned_path.append(env.agent_position)

    env.set_render_data(
        status={
            "mode": "success" if terminated else "truncated",
            "step": env.step_count,
            "reward": f"{total_reward:.1f}",
            "epsilon": "0.000",
        },
        learned_path=learned_path,
        optimal_path=optimal_path,
    )
    final_frame = env.render()
    if final_frame is not None:
        frames.extend([final_frame] * fps)
    imageio.mimsave(output, frames, duration=1000 / fps, loop=0)
    return output, tuple(learned_path), terminated


def save_snapshot(
    env: MazeEnv,
    destination: str | Path,
    *,
    learned_path: tuple[State, ...] = (),
) -> Path:
    """Save a PNG frame with the learned and optimal paths."""
    if env.render_mode != "rgb_array":
        raise ValueError("save_snapshot requires render_mode='rgb_array'")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    optimal = astar_path(env.grid_size, env.start, env.goal, env.obstacles) or ()
    env.set_render_data(
        status={"mode": "portfolio", "compare": "Q-learning / A*"},
        learned_path=learned_path,
        optimal_path=optimal,
    )
    frame = env.render()
    if frame is None:
        raise RuntimeError("The renderer did not produce an RGB frame")
    imageio.imwrite(output, frame)
    return output
