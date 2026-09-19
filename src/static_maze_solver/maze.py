"""Maze presets and reproducible random maze generation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from static_maze_solver.baseline import State, astar_path


@dataclass(frozen=True)
class MazeConfig:
    """Serializable definition of one maze."""

    grid_size: tuple[int, int]
    start: State
    goal: State
    obstacles: tuple[State, ...]
    name: str = "custom"


PRESETS: dict[str, MazeConfig] = {
    "classic": MazeConfig(
        grid_size=(10, 10),
        start=(0, 0),
        goal=(9, 9),
        obstacles=((2, 2), (3, 1), (1, 3), (5, 4), (6, 4), (7, 4)),
        name="classic",
    ),
    "small": MazeConfig(
        grid_size=(5, 5),
        start=(0, 0),
        goal=(4, 4),
        obstacles=((1, 1), (1, 2), (3, 2)),
        name="small",
    ),
}


def get_preset(name: str) -> MazeConfig:
    """Return a named immutable maze preset."""
    try:
        return PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown maze preset {name!r}. Available presets: {choices}") from exc


def generate_maze(
    rows: int = 10,
    cols: int = 10,
    obstacle_density: float = 0.2,
    seed: int = 42,
) -> MazeConfig:
    """Generate a reproducible solvable maze.

    Random obstacles are sampled until a path exists. A deterministic fallback
    keeps the top row and rightmost column open.
    """
    if rows < 2 or cols < 2:
        raise ValueError("Maze dimensions must both be at least 2")
    if not 0 <= obstacle_density < 0.8:
        raise ValueError("obstacle_density must be in [0, 0.8)")

    rng = random.Random(seed)
    start, goal = (0, 0), (rows - 1, cols - 1)
    candidates = [
        (row, col) for row in range(rows) for col in range(cols) if (row, col) not in {start, goal}
    ]
    obstacle_count = round(len(candidates) * obstacle_density)

    for _ in range(100):
        obstacles = tuple(sorted(rng.sample(candidates, obstacle_count)))
        if astar_path((rows, cols), start, goal, obstacles) is not None:
            return MazeConfig((rows, cols), start, goal, obstacles, f"generated-{seed}")

    guaranteed_path = {(0, col) for col in range(cols)} | {(row, cols - 1) for row in range(rows)}
    available = [cell for cell in candidates if cell not in guaranteed_path]
    count = min(obstacle_count, len(available))
    obstacles = tuple(sorted(rng.sample(available, count)))
    return MazeConfig((rows, cols), start, goal, obstacles, f"generated-{seed}")
