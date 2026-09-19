"""Classical shortest-path baselines used to evaluate the RL agent."""

from __future__ import annotations

import heapq
from collections.abc import Iterable

State = tuple[int, int]


def _neighbors(
    state: State,
    grid_size: tuple[int, int],
    obstacles: set[State],
) -> Iterable[State]:
    rows, cols = grid_size
    row, col = state
    for next_state in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        next_row, next_col = next_state
        if 0 <= next_row < rows and 0 <= next_col < cols and next_state not in obstacles:
            yield next_state


def astar_path(
    grid_size: tuple[int, int],
    start: State,
    goal: State,
    obstacles: Iterable[State] = (),
) -> list[State] | None:
    """Return an optimal path from start to goal, or ``None`` when unreachable."""
    blocked = set(obstacles)
    frontier: list[tuple[int, int, State]] = [(0, 0, start)]
    came_from: dict[State, State | None] = {start: None}
    cost_so_far: dict[State, int] = {start: 0}
    order = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            path: list[State] = []
            node: State | None = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            return list(reversed(path))

        for neighbor in _neighbors(current, grid_size, blocked):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heuristic = abs(goal[0] - neighbor[0]) + abs(goal[1] - neighbor[1])
                order += 1
                heapq.heappush(frontier, (new_cost + heuristic, order, neighbor))
                came_from[neighbor] = current

    return None
