"""Reproducible tabular Q-learning agent."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from static_maze_solver.baseline import State


class QLearningAgent:
    """A tabular epsilon-greedy Q-learning agent."""

    MODEL_VERSION = 1

    def __init__(
        self,
        action_count: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 <= epsilon <= 1 or not 0 <= epsilon_min <= 1:
            raise ValueError("epsilon and epsilon_min must both be in [0, 1]")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")

        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self._rng = random.Random(seed)
        self.q_table: dict[State, NDArray[np.float64]] = {}

    @staticmethod
    def state_key(state: ArrayLike) -> State:
        """Normalize Gymnasium array observations into hashable table keys."""
        values = np.asarray(state, dtype=np.int64).tolist()
        return int(values[0]), int(values[1])

    def q_values(self, state: ArrayLike) -> NDArray[np.float64]:
        """Return the state row, creating a zero-initialized row when needed."""
        key = self.state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_count, dtype=np.float64)
        return self.q_table[key]

    def choose_action(self, state: ArrayLike, *, explore: bool = True) -> int:
        """Choose an epsilon-greedy action with randomized tie-breaking."""
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.action_count)
        values = self.q_values(state)
        best_actions = np.flatnonzero(values == values.max()).tolist()
        return int(self._rng.choice(best_actions))

    def update(
        self,
        state: ArrayLike,
        action: int,
        reward: float,
        next_state: ArrayLike,
        *,
        terminated: bool,
    ) -> float:
        """Apply one terminal-aware Q-learning update and return the new value."""
        if not 0 <= action < self.action_count:
            raise ValueError(f"Invalid action {action}")
        current_values = self.q_values(state)
        future = 0.0 if terminated else float(self.q_values(next_state).max())
        target = reward + self.gamma * future
        current_values[action] += self.alpha * (target - current_values[action])
        return float(current_values[action])

    def decay_exploration(self) -> float:
        """Decay epsilon while respecting the configured lower bound."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, path: str | Path) -> Path:
        """Save hyperparameters and Q-values as portable, versioned JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_version": self.MODEL_VERSION,
            "algorithm": "tabular-q-learning",
            "hyperparameters": {
                "action_count": self.action_count,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "seed": self.seed,
            },
            "q_table": {
                f"{row},{col}": values.tolist()
                for (row, col), values in sorted(self.q_table.items())
            },
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> QLearningAgent:
        """Load and validate a model written by :meth:`save`."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("model_version") != cls.MODEL_VERSION:
            raise ValueError(f"Unsupported model version in {source}")
        agent = cls(**payload["hyperparameters"])
        for key, values in payload["q_table"].items():
            row, col = (int(value) for value in key.split(",", maxsplit=1))
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (agent.action_count,):
                raise ValueError(f"Invalid Q-table row for state {(row, col)}")
            agent.q_table[(row, col)] = array
        return agent
