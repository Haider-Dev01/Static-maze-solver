"""Tests for generated mazes and the A* reference solver."""

import pytest

from static_maze_solver.baseline import astar_path
from static_maze_solver.maze import generate_maze, get_preset


def test_astar_returns_an_optimal_path() -> None:
    path = astar_path((3, 3), (0, 0), (2, 2), [(1, 1)])
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) - 1 == 4


def test_astar_returns_none_when_blocked() -> None:
    assert astar_path((2, 2), (0, 0), (1, 1), [(0, 1), (1, 0)]) is None


def test_generation_is_reproducible_and_solvable() -> None:
    first = generate_maze(8, 8, obstacle_density=0.3, seed=12)
    second = generate_maze(8, 8, obstacle_density=0.3, seed=12)
    assert first == second
    assert astar_path(first.grid_size, first.start, first.goal, first.obstacles)


def test_presets_and_generation_validation() -> None:
    assert get_preset("classic").grid_size == (10, 10)
    with pytest.raises(ValueError, match="Unknown"):
        get_preset("missing")
    with pytest.raises(ValueError, match="at least 2"):
        generate_maze(1, 4)
    with pytest.raises(ValueError, match="obstacle_density"):
        generate_maze(4, 4, obstacle_density=0.9)
