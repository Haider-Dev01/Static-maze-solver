"""Gymnasium maze environment with optional Pygame rendering."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from static_maze_solver.baseline import State, astar_path

Observation = NDArray[np.int64]


class MazeEnv(gym.Env[Any, int]):
    """A validated grid-world environment for tabular reinforcement learning."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    ACTION_DELTAS: dict[int, State] = {
        0: (-1, 0),  # up
        1: (1, 0),  # down
        2: (0, -1),  # left
        3: (0, 1),  # right
    }

    def __init__(
        self,
        grid_size: tuple[int, int] = (10, 10),
        start: State = (0, 0),
        goal: State | None = None,
        obstacles: Iterable[State] = (),
        max_steps: int | None = None,
        render_mode: str | None = None,
        step_penalty: float = -1.0,
        goal_reward: float = 100.0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.start = start
        self.goal = goal if goal is not None else (grid_size[0] - 1, grid_size[1] - 1)
        self.obstacles = frozenset(obstacles)
        self.max_steps = max_steps or grid_size[0] * grid_size[1] * 4
        self.render_mode = render_mode
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self._validate_config()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(np.asarray(grid_size, dtype=np.int64))
        self.agent_position = self.start
        self.step_count = 0
        self.close_requested = False

        self._pygame: Any = None
        self._screen: Any = None
        self._clock: Any = None
        self._font: Any = None
        self._canvas_size = 600
        self._panel_width = 220
        self._status: dict[str, str] = {}
        self._learned_path: tuple[State, ...] = ()
        self._optimal_path: tuple[State, ...] = ()

    def _validate_config(self) -> None:
        rows, cols = self.grid_size
        if rows < 2 or cols < 2:
            raise ValueError("grid_size dimensions must both be at least 2")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        def valid(cell: State) -> bool:
            return 0 <= cell[0] < rows and 0 <= cell[1] < cols

        for label, cell in (("start", self.start), ("goal", self.goal)):
            if not valid(cell):
                raise ValueError(f"{label} position {cell} is outside the grid")
        invalid = [cell for cell in self.obstacles if not valid(cell)]
        if invalid:
            raise ValueError(f"Obstacle positions outside the grid: {invalid}")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("start and goal cannot be obstacles")
        if astar_path(self.grid_size, self.start, self.goal, self.obstacles) is None:
            raise ValueError("The goal must be reachable from the start")
        if self.render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

    def _observation(self) -> Observation:
        return np.asarray(self.agent_position, dtype=np.int64)

    def _info(self) -> dict[str, Any]:
        return {
            "position": self.agent_position,
            "steps": self.step_count,
            "is_success": self.agent_position == self.goal,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the agent and return the Gymnasium observation/info pair."""
        super().reset(seed=seed)
        del options
        self.agent_position = self.start
        self.step_count = 0
        self.close_requested = False
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one action using the Gymnasium terminated/truncated API."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected one of 0, 1, 2, 3")

        row, col = self.agent_position
        delta_row, delta_col = self.ACTION_DELTAS[int(action)]
        candidate = (row + delta_row, col + delta_col)
        rows, cols = self.grid_size
        if (
            0 <= candidate[0] < rows
            and 0 <= candidate[1] < cols
            and candidate not in self.obstacles
        ):
            self.agent_position = candidate

        self.step_count += 1
        terminated = self.agent_position == self.goal
        truncated = self.step_count >= self.max_steps and not terminated
        reward = self.goal_reward if terminated else self.step_penalty
        return self._observation(), reward, terminated, truncated, self._info()

    def set_render_data(
        self,
        *,
        status: dict[str, object] | None = None,
        learned_path: Iterable[State] = (),
        optimal_path: Iterable[State] = (),
    ) -> None:
        """Update optional paths and status text displayed by the renderer."""
        self._status = {key: str(value) for key, value in (status or {}).items()}
        self._learned_path = tuple(learned_path)
        self._optimal_path = tuple(optimal_path)

    def _initialize_renderer(self) -> None:
        if self._pygame is not None:
            return
        import pygame

        pygame.init()
        self._pygame = pygame
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 26)
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode(
                (self._canvas_size + self._panel_width, self._canvas_size)
            )
            pygame.display.set_caption("Static Maze Solver — Q-learning")

    def _draw(self) -> Any:
        self._initialize_renderer()
        pygame = self._pygame
        rows, cols = self.grid_size
        cell_width = self._canvas_size / cols
        cell_height = self._canvas_size / rows
        surface = pygame.Surface((self._canvas_size + self._panel_width, self._canvas_size))
        surface.fill((248, 249, 250))

        def rect(cell: State) -> Any:
            row, col = cell
            return pygame.Rect(
                round(col * cell_width),
                round(row * cell_height),
                round(cell_width),
                round(cell_height),
            )

        for cell in self._optimal_path:
            pygame.draw.rect(surface, (173, 216, 230), rect(cell))
        for cell in self._learned_path:
            pygame.draw.rect(surface, (255, 215, 120), rect(cell))
        for obstacle in self.obstacles:
            pygame.draw.rect(surface, (45, 52, 54), rect(obstacle))
        pygame.draw.rect(surface, (60, 179, 113), rect(self.goal))
        pygame.draw.rect(surface, (220, 53, 69), rect(self.agent_position))

        for row in range(rows + 1):
            y = round(row * cell_height)
            pygame.draw.line(surface, (180, 180, 180), (0, y), (self._canvas_size, y), 1)
        for col in range(cols + 1):
            x = round(col * cell_width)
            pygame.draw.line(surface, (180, 180, 180), (x, 0), (x, self._canvas_size), 1)

        panel_x = self._canvas_size + 18
        title = self._font.render("Q-learning", True, (30, 30, 30))
        surface.blit(title, (panel_x, 20))
        for index, (key, value) in enumerate(self._status.items()):
            line = self._font.render(f"{key}: {value}", True, (50, 50, 50))
            surface.blit(line, (panel_x, 65 + index * 30))

        legend = [
            ("Agent", (220, 53, 69)),
            ("Goal", (60, 179, 113)),
            ("Learned", (255, 215, 120)),
            ("A* optimal", (173, 216, 230)),
        ]
        y = self._canvas_size - 150
        for label, color in legend:
            pygame.draw.rect(surface, color, pygame.Rect(panel_x, y, 18, 18))
            surface.blit(self._font.render(label, True, (50, 50, 50)), (panel_x + 28, y - 2))
            y += 30
        return surface

    def render(self) -> NDArray[np.uint8] | None:
        """Render a human window or return an RGB array."""
        if self.render_mode is None:
            return None
        surface = self._draw()
        if self.render_mode == "rgb_array":
            frame = self._pygame.surfarray.array3d(surface)
            return np.asarray(np.transpose(frame, (1, 0, 2)), dtype=np.uint8)

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.close_requested = True
        self._screen.blit(surface, (0, 0))
        self._pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self) -> None:
        """Release Pygame resources."""
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = self._screen = self._clock = self._font = None
